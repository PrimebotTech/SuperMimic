"""Import joblib-pickled retargeted Q1 motions into this project's NPZ contract.

The source schema is the axis-angle motion dictionary used by several
retargeting pipelines: one pickle holds ``{motion_key: record}`` and each record
carries ``root_trans_offset [T, 3]``, ``pose_aa [T, B, 3]`` and a scalar ``fps``.
Records written from an explicit robot state additionally carry ``root_quat``,
``dof_pos`` and ``joint_names``, which is what this importer reads: those three
reconstruct MuJoCo ``qpos`` exactly, whereas recovering joint angles from
``pose_aa`` would require re-deriving every hinge axis and body order and would
silently permute joints if either disagreed with the XML. Records without them
are rejected rather than guessed.

Unpickling executes code from the file. Only import corpora you produced or
otherwise trust.
"""

import hashlib
import json
import multiprocessing
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
import tyro

from supermimic.resample import resample_qpos
from supermimic.robot import Q1Robot

REQUIRED_RECORD_FIELDS = ("root_trans_offset", "root_quat", "dof_pos", "fps")


def record_to_qpos(
  record: Any, joint_names: Sequence[str]
) -> tuple[npt.NDArray[np.float64], list[str], float]:
  """Rebuild 29-dimensional MuJoCo qpos from one retargeted motion record.

  Returns qpos in the record's own joint order together with that order, so the
  NPZ keeps the source ordering and the loader resolves it by name exactly as it
  does for every other motion file.
  """
  if not isinstance(record, dict):
    raise ValueError(f"record must be a dict, got {type(record).__name__}")
  missing = sorted(set(REQUIRED_RECORD_FIELDS).difference(record))
  if missing:
    raise ValueError(f"missing record fields {missing}; pose_aa alone is not accepted")
  if str(record.get("robot_name", "")) != "q1_v2":
    raise ValueError(f"expected robot_name=q1_v2, got {record.get('robot_name')!r}")
  metadata = record.get("metadata") or {}
  if not isinstance(metadata, dict):
    raise ValueError(f"metadata must be a dict, got {type(metadata).__name__}")
  # The writer records its own quaternion convention; never assume one.
  order = str(metadata.get("root_quat_order", ""))
  if order != "xyzw":
    raise ValueError(f"expected metadata root_quat_order=xyzw, got {order!r}")
  if "padded_from_frames" in metadata:
    raise ValueError(
      f"record was padded from {metadata['padded_from_frames']} frames; "
      "padding repeats the last pose and is not a tracked reference"
    )
  if str(metadata.get("dof_unit", "rad")) != "rad":
    raise ValueError(f"expected radians, got dof_unit={metadata.get('dof_unit')!r}")
  names = [str(name) for name in record["joint_names"]]
  if len(names) != 22 or set(names) != set(joint_names):
    raise ValueError("record joint names mismatch or duplicate against the Q1 XML")
  root_pos = np.asarray(record["root_trans_offset"], dtype=np.float64)
  root_quat = np.asarray(record["root_quat"], dtype=np.float64)
  dof_pos = np.asarray(record["dof_pos"], dtype=np.float64)
  shapes = ((root_pos, 3), (root_quat, 4), (dof_pos, 22))
  for array, width in shapes:
    if array.ndim != 2 or array.shape[1] != width:
      raise ValueError(f"expected shape [T, {width}], got {array.shape}")
  if len({len(array) for array, _ in shapes}) != 1:
    raise ValueError("root translation, root rotation and joint angles differ in T")
  fps = np.asarray(record["fps"]).reshape(-1)
  if fps.size != 1 or not np.isfinite(float(fps[0])) or float(fps[0]) <= 0:
    raise ValueError("fps must be a positive finite scalar")
  qpos = np.empty((len(root_pos), 29), dtype=np.float64)
  qpos[:, :3] = root_pos
  # XYZW -> MuJoCo WXYZ.
  qpos[:, 3:7] = root_quat[:, [3, 0, 1, 2]]
  qpos[:, 7:] = dof_pos
  if not np.isfinite(qpos).all():
    raise ValueError("nonfinite pose")
  norms = np.linalg.norm(qpos[:, 3:7], axis=-1)
  if np.any(np.abs(norms - 1) > 0.01):
    raise ValueError("root quaternion is not unit length")
  qpos[:, 3:7] /= norms[:, None]
  return qpos, names, float(fps[0])


@dataclass
class ImportCfg:
  robot_xml: Path
  output_dir: Path
  motion_dir: Path | None = None
  motion_paths: list[Path] = field(default_factory=list)
  motion_pattern: str = "*.pkl"
  max_motions: int = 8
  target_fps: float = 50.0
  min_frames: int = 20
  workers: int = 1


def import_one(path: Path, robot: Q1Robot, cfg: ImportCfg) -> dict[str, Any]:
  """Convert every clip of one source pickle, or report why the file was skipped.

  Each output is created exclusively, so two source files that disagree about a
  motion key raise instead of silently overwriting one another -- including when
  separate worker processes reach them at the same moment.
  """
  try:
    data = joblib.load(path)
    if not isinstance(data, dict) or not data:
      raise ValueError(f"expected a non-empty dict of motions, got {type(data)}")
    clips = []
    for key, record in data.items():
      qpos, names, source_fps = record_to_qpos(record, robot.joint_names)
      converted = (
        qpos.astype(np.float32)
        if np.isclose(source_fps, cfg.target_fps)
        else resample_qpos(qpos, source_fps, cfg.target_fps)
      )
      if len(converted) < cfg.min_frames:
        raise ValueError(
          f"{key}: {len(converted)} resampled frames is below min_frames"
        )
      clips.append((f"{key}.npz", key, names, qpos, converted, source_fps))
    # Write only once the whole file validated, so a partial pickle cannot
    # leave half its clips in the output directory.
    written = []
    for name, key, names, qpos, converted, source_fps in clips:
      with (cfg.output_dir / name).open("xb") as handle:
        np.savez_compressed(
          handle,
          qpos=converted,
          robot_name=np.array("q1_v2"),
          root_quat_order=np.array("wxyz"),
          fps=np.array(cfg.target_fps),
          robot_joint_names=np.array(names),
          motion_key=np.array(str(key)),
          source_file=np.array(str(path)),
          source_fps=np.array(source_fps),
          source_qpos_sha256=np.array(
            hashlib.sha256(qpos.astype(np.float32).tobytes()).hexdigest()
          ),
          import_method=np.array("joblib_root_quat_dof_pos_v1"),
          resample_method=np.array(
            "none"
            if np.isclose(source_fps, cfg.target_fps)
            else "linear_xyz_joints_shortest_path_slerp_v1"
          ),
        )
      written.append(
        {
          "output": name,
          "motion_key": str(key),
          "source_fps": source_fps,
          "source_frames": len(qpos),
          "output_frames": len(converted),
        }
      )
    return {"source": str(path), "clips": written, "ok": True}
  except (ValueError, KeyError, TypeError, OSError, EOFError) as error:
    return {
      "source": str(path),
      "ok": False,
      "error": f"{type(error).__name__}: {error}",
    }


_WORKER: dict[str, Any] = {}


def _worker_init(robot_xml: Path, cfg: ImportCfg) -> None:
  # Parsing the XML once per process, not once per file: the corpus has more
  # than a hundred thousand files and the model costs far more than a clip.
  _WORKER["robot"] = Q1Robot(robot_xml)
  _WORKER["cfg"] = cfg


def _worker_import(path: Path) -> dict[str, Any]:
  return import_one(path, _WORKER["robot"], _WORKER["cfg"])


def import_pkl(cfg: ImportCfg) -> Path:
  if cfg.max_motions < 0 or not np.isclose(cfg.target_fps, 50.0):
    raise ValueError("Use a nonnegative limit and the current 50 Hz Q1 contract")
  if cfg.min_frames < 20:
    raise ValueError("The training loader requires at least 20 frames per clip")
  if cfg.workers < 1:
    raise ValueError("Import with at least one worker process")
  if bool(cfg.motion_paths) == (cfg.motion_dir is not None):
    raise ValueError("Choose explicit motion paths or a motion directory")
  robot = Q1Robot(cfg.robot_xml)
  if cfg.motion_paths:
    paths = list(dict.fromkeys(path.resolve() for path in cfg.motion_paths))
    if cfg.max_motions and len(paths) > cfg.max_motions:
      raise ValueError("Explicit motion count exceeds max_motions")
  else:
    assert cfg.motion_dir is not None
    if not cfg.motion_dir.is_dir():
      raise ValueError(f"Motion directory does not exist: {cfg.motion_dir}")
    if Path(cfg.motion_pattern).is_absolute() or ".." in Path(cfg.motion_pattern).parts:
      raise ValueError("Motion pattern must stay within the motion directory")
    paths = sorted(cfg.motion_dir.glob(cfg.motion_pattern))
    if cfg.max_motions:
      paths = paths[: cfg.max_motions]
  if not paths:
    raise ValueError("No source motion files selected")
  cfg.output_dir.mkdir(parents=True, exist_ok=False)
  cfg.output_dir = cfg.output_dir.resolve()
  report: list[dict[str, Any]] = []
  if cfg.workers == 1:
    for path in paths:
      report.append(import_one(path, robot, cfg))
      print(json.dumps(report[-1]), flush=True)
  else:
    with ProcessPoolExecutor(
      cfg.workers,
      # Spawn, not fork: torch and warp are already loaded and multi-threaded by
      # the time a corpus import starts, and forking such a process deadlocks.
      mp_context=multiprocessing.get_context("spawn"),
      initializer=_worker_init,
      initargs=(cfg.robot_xml, cfg),
    ) as pool:
      # Ordered results keep the manifest reproducible regardless of which
      # worker finished first; a chunk amortizes dispatch over many small files.
      for row in pool.map(_worker_import, paths, chunksize=32):
        report.append(row)
        print(json.dumps(row), flush=True)
  manifest = cfg.output_dir / "import_manifest.json"
  imported = sum(bool(row["ok"]) for row in report)
  written = sum(len(row.get("clips", ())) for row in report)
  manifest.write_text(
    json.dumps(
      {
        "target_fps": cfg.target_fps,
        "xml_sha256": robot.sha256,
        "selected_files": len(paths),
        "imported_files": imported,
        "failed_files": len(paths) - imported,
        "written_clips": written,
        "files": report,
      },
      indent=2,
    )
    + "\n"
  )
  if not written:
    raise ValueError(f"No clips imported; inspect {manifest}")
  return manifest


def main() -> None:
  import_pkl(tyro.cli(ImportCfg))


if __name__ == "__main__":
  main()
