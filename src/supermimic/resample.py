"""Independent time resampling of named Q1 motions, preserving quaternion paths."""

import hashlib
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
import tyro

from supermimic.robot import Q1Robot


def resample_qpos(
  qpos: npt.NDArray, source_fps: float, target_fps: float = 50.0
) -> npt.NDArray[np.float32]:
  """Linear XYZ/joints and shortest-path quaternion SLERP on actual timestamps."""
  values = np.array(qpos, dtype=np.float64, copy=True)
  if values.ndim != 2 or values.shape[1] != 29 or len(values) < 2:
    raise ValueError("Expected at least two Q1 qpos frames with 29 columns")
  if not np.isfinite(values).all():
    raise ValueError("Nonfinite Q1 pose")
  if (
    not np.isfinite([source_fps, target_fps]).all() or min(source_fps, target_fps) <= 0
  ):
    raise ValueError("Frame rates must be finite and positive")
  norms = np.linalg.norm(values[:, 3:7], axis=-1)
  if np.any(np.abs(norms - 1) > 0.01):
    raise ValueError("Q1 root quaternions must be unit length")
  values[:, 3:7] /= norms[:, None]
  duration = (len(values) - 1) / source_fps
  count = int(np.floor(duration * target_fps + 1e-9)) + 1
  phase = np.arange(count, dtype=np.float64) / target_fps * source_fps
  phase = np.minimum(phase, len(values) - 1)
  left = np.minimum(phase.astype(np.int64), len(values) - 2)
  blend = (phase - left)[:, None]
  result = (1 - blend) * values[left] + blend * values[left + 1]
  qa, qb = values[left, 3:7], values[left + 1, 3:7]
  dot = (qa * qb).sum(-1, keepdims=True)
  qb = np.where(dot < 0, -qb, qb)
  dot = np.abs(dot).clip(0, 1)
  theta = np.arccos(dot)
  denominator = np.sin(theta)
  large = denominator > 1e-6
  wa = np.divide(
    np.sin((1 - blend) * theta), denominator, out=(1 - blend).copy(), where=large
  )
  wb = np.divide(np.sin(blend * theta), denominator, out=blend.copy(), where=large)
  rotation = wa * qa + wb * qb
  result[:, 3:7] = rotation / np.linalg.norm(rotation, axis=-1, keepdims=True)
  return result.astype(np.float32)


@dataclass
class ResampleCfg:
  robot_xml: Path
  output_dir: Path
  motion_dir: Path | None = None
  motion_paths: list[Path] = field(default_factory=list)
  motion_pattern: str = "*.npz"
  max_motions: int = 8
  target_fps: float = 50.0


def resample(cfg: ResampleCfg) -> Path:
  if cfg.max_motions < 0 or not np.isclose(cfg.target_fps, 50.0):
    raise ValueError("Use a nonnegative limit and the current 50 Hz Q1 contract")
  robot = Q1Robot(cfg.robot_xml)
  if bool(cfg.motion_paths) == (cfg.motion_dir is not None):
    raise ValueError("Choose explicit motion paths or a motion directory")
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
    raise ValueError("No Q1 motion files selected")
  if len({path.name for path in paths}) != len(paths):
    raise ValueError("Selected source filenames must be unique")
  cfg.output_dir.mkdir(parents=True, exist_ok=False)
  report = []
  for path in paths:
    try:
      with np.load(path, allow_pickle=False) as data:
        if str(data["robot_name"].item()) != "q1_v2":
          raise ValueError("Expected Q1 V2 metadata")
        if str(data["root_quat_order"].item()) != "wxyz":
          raise ValueError("Expected WXYZ root quaternions")
        names = data["robot_joint_names"].tolist()
        if len(names) != 22 or set(names) != set(robot.joint_names):
          raise ValueError("Q1 joint names differ from XML")
        original = data["qpos"]
        source_fps = float(data["fps"].item())
        converted = resample_qpos(original, source_fps, cfg.target_fps)
        if len(converted) < 20:
          raise ValueError("Resampled clip has fewer than 20 frames")
        metadata = {
          key: data[key]
          for key in data.files
          if data[key].ndim == 0 and key not in ("fps", "file", "allow_pickle")
        }
        dropped = [
          key
          for key in data.files
          if data[key].ndim > 0 and key not in ("qpos", "robot_joint_names")
        ]
        digest = hashlib.sha256(original.tobytes()).hexdigest()
      output = cfg.output_dir / path.name
      np.savez_compressed(
        output,
        allow_pickle=False,
        **(
          metadata
          | {
            "qpos": converted,
            "fps": np.array(cfg.target_fps),
            "robot_joint_names": np.array(names),
            "source_file": np.array(str(path.resolve())),
            "source_fps": np.array(source_fps),
            "source_qpos_sha256": np.array(digest),
            "resample_method": np.array("linear_xyz_joints_shortest_path_slerp_v1"),
          }
        ),
      )
      row = {
        "source": str(path.resolve()),
        "output": str(output.resolve()),
        "source_fps": source_fps,
        "source_frames": len(original),
        "output_frames": len(converted),
        "dropped_arrays": dropped,
        "ok": True,
      }
    except (ValueError, KeyError, OSError, EOFError, zipfile.BadZipFile) as error:
      row = {"source": str(path.resolve()), "ok": False, "error": str(error)}
    report.append(row)
    print(json.dumps(row), flush=True)
  manifest = cfg.output_dir / "resample_manifest.json"
  converted_count = sum(bool(row["ok"]) for row in report)
  manifest.write_text(
    json.dumps(
      {
        "target_fps": cfg.target_fps,
        "xml_sha256": robot.sha256,
        "selected_clips": len(paths),
        "converted_clips": converted_count,
        "failed_clips": len(paths) - converted_count,
        "clips": report,
      },
      indent=2,
    )
    + "\n"
  )
  if converted_count == 0:
    raise ValueError(f"No clips converted; inspect {manifest}")
  return manifest


def main() -> None:
  resample(tyro.cli(ResampleCfg))


if __name__ == "__main__":
  main()
