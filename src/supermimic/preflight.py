"""Bounded Q1 motion diagnostics without changing XML or reference data.

This reports Q1 XML limits and static collision geometry, not calibrated G1
filtering thresholds from the paper. No simulation or self-collision test runs.
"""

import argparse
import glob
import itertools
import json
import zipfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol, cast

import mujoco
import numpy as np
from numpy.typing import NDArray

from supermimic.robot import Q1Robot

Array = NDArray[np.float64]
MAX_CLIPS = 32
GeomDistance = Callable[
  [mujoco.MjModel, mujoco.MjData, int, int, float, Array | None], float
]


class GeometryModel(Protocol):
  """Native attributes beyond the project's minimal MuJoCo stub."""

  ngeom: int
  geom_type: NDArray[np.int32]
  geom_bodyid: NDArray[np.int32]
  geom_contype: NDArray[np.int32]
  geom_conaffinity: NDArray[np.int32]
  jnt_limited: NDArray[np.bool_]


class GeometryData(Protocol):
  geom_xmat: Array


class GeometryAPI(Protocol):
  mj_geomDistance: GeomDistance


@dataclass
class MotionReport:
  path: str
  compatible_with_loader: bool = False
  compatibility_scope: str = "archive schema and chosen expected FPS only"
  feasibility_assessment: str = "not_assessed"
  errors: list[str] = field(default_factory=list)
  warnings: list[str] = field(default_factory=list)
  checks: dict[str, object] = field(default_factory=dict)
  diagnostics: dict[str, object] = field(default_factory=dict)


def select_paths(
  paths: list[Path], directory: Path | None, pattern: str, limit: int
) -> list[Path]:
  """Select at most 32 files; never read all motion archives to find candidates."""
  if not 1 <= limit <= MAX_CLIPS:
    raise ValueError(f"limit must be between 1 and {MAX_CLIPS}")
  if bool(paths) == (directory is not None):
    raise ValueError("Choose explicit motion paths or a motion directory")
  if paths:
    if len(paths) > limit:
      raise ValueError("Explicit motion count exceeds limit; increase limit")
    selected = paths
  else:
    assert directory is not None
    if not directory.is_dir():
      raise ValueError(f"Motion directory does not exist: {directory}")
    if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
      raise ValueError("motion pattern must stay within the motion directory")
    matches = glob.iglob(str(directory / pattern), recursive=False)
    selected = sorted(Path(p) for p in itertools.islice(matches, limit))
  selected = list(dict.fromkeys(path.resolve() for path in selected))
  if not selected:
    raise ValueError("No motion files selected")
  return selected


def _foot_geoms(robot: Q1Robot) -> tuple[int, dict[str, list[int]]]:
  model = cast(GeometryModel, robot.model)
  floors = [
    geom
    for geom in range(model.ngeom)
    if model.geom_type[geom] == 0 and model.geom_bodyid[geom] == 0
  ]
  if len(floors) != 1:
    raise ValueError("Need exactly one world plane to verify ground clearance")
  floor = floors[0]
  feet: dict[str, list[int]] = {}
  for side in ("left", "right"):
    body_id = int(robot.model.body(f"{side}_ankle_roll_link").id)
    bodies = {body_id}
    for child in range(body_id + 1, robot.model.nbody):
      if robot.parents[child] in bodies:
        bodies.add(child)
    geoms = [
      geom
      for geom in range(model.ngeom)
      if int(model.geom_bodyid[geom]) in bodies
      and (
        (int(model.geom_contype[geom]) & int(model.geom_conaffinity[floor]))
        or (int(model.geom_contype[floor]) & int(model.geom_conaffinity[geom]))
      )
    ]
    if not geoms:
      raise ValueError(f"No {side} foot geoms can collide with the floor")
    feet[side] = geoms
  return floor, feet


def geometry_diagnostics(
  robot: Q1Robot, qpos: Array, max_frames: int, tolerance: float
) -> dict[str, object]:
  """Measure link origins and foot surfaces on a deterministic frame sample."""
  if max_frames < 1:
    raise ValueError("max FK frames must be positive")
  indexes = np.unique(
    np.linspace(0, len(qpos) - 1, min(max_frames, len(qpos)), dtype=np.int64)
  )
  data = mujoco.MjData(robot.model)
  geom_data = cast(GeometryData, data)
  geom_distance = cast(GeometryAPI, mujoco).mj_geomDistance
  minimum_link = float("inf")
  minimum_body, minimum_frame, link_below = "", 0, 0
  gaps: dict[str, list[float]] = {"left": [], "right": []}
  geometry_error: str | None = None
  try:
    floor, feet = _foot_geoms(robot)
  except (ValueError, KeyError) as exc:
    floor, feet, geometry_error = -1, {}, str(exc)
  for frame in indexes:
    data.qpos[:] = qpos[frame]
    mujoco.mj_forward(robot.model, data)
    heights = data.xpos[1:, 2]
    body = int(heights.argmin()) + 1
    height = float(heights[body - 1])
    link_below += int(height < -tolerance)
    if height < minimum_link:
      minimum_link, minimum_body, minimum_frame = (
        height,
        robot.model.body(body).name,
        int(frame),
      )
    if geometry_error is not None:
      continue
    normal = geom_data.geom_xmat[floor].reshape(3, 3)[:, 2]
    if normal[2] <= 0:
      geometry_error = "World plane normal does not point upward"
      continue
    try:
      for side, geoms in feet.items():
        distances = []
        for geom in geoms:
          distance = geom_distance(robot.model, data, floor, geom, 1e6, None)
          if not np.isfinite(distance) or distance >= 1e6:
            raise ValueError(f"No exact distance returned for foot geom {geom}")
          distances.append(distance)
        gaps[side].append(min(distances))
    except ValueError as exc:
      geometry_error = str(exc)
  ground: dict[str, object] = {
    "distance_computation_verified": geometry_error is None,
    "method": "MuJoCo mj_geomDistance against XML world plane (signed)",
    "scope": "static geometry; no self-collision, dynamics or torque feasibility",
    "coverage": "selected FK frames; no bound for unsampled frame clearance",
    "clearance_filter_applied": False,
    "penetration_numerical_tolerance_m": tolerance,
  }
  if geometry_error is not None:
    ground["reason"] = geometry_error
  else:
    ground["floor_geom_id"] = floor
    ground["feet"] = {
      side: {
        "collision_geom_ids": feet[side],
        "minimum_clearance_m": min(values),
        "minimum_frame": int(indexes[int(np.argmin(values))]),
        "penetration_frames": sum(value < -tolerance for value in values),
        "penetration_fraction_of_sample": float(
          np.mean(np.asarray(values) < -tolerance)
        ),
      }
      for side, values in gaps.items()
    }
  return {
    "sampled_frames": len(indexes),
    "total_frames": len(qpos),
    "all_frames_checked": len(indexes) == len(qpos),
    "frame_selection": "uniform index sample including first and last frame",
    "minimum_link_origin_height_m": minimum_link,
    "minimum_link": minimum_body,
    "minimum_link_frame": minimum_frame,
    "link_origin_below_zero_frames": link_below,
    "ground_clearance": ground,
  }


def inspect_motion(
  robot: Q1Robot,
  path: Path,
  expected_fps: float = 50.0,
  max_fk_frames: int = 1000,
  numerical_tolerance: float = 1e-6,
) -> MotionReport:
  """Inspect one named Q1 NPZ, retaining diagnostics on frequency mismatch."""
  if not np.isfinite(expected_fps) or expected_fps <= 0:
    raise ValueError("expected FPS must be finite and positive")
  if numerical_tolerance < 0 or not np.isfinite(numerical_tolerance):
    raise ValueError("numerical tolerance must be finite and nonnegative")
  report = MotionReport(str(path.resolve()))
  required = {"qpos", "fps", "robot_name", "root_quat_order", "robot_joint_names"}
  try:
    with np.load(path, allow_pickle=False) as archive:
      missing = required - set(archive.files)
      if missing:
        report.errors.append(f"Missing fields: {', '.join(sorted(missing))}")
        return report
      raw = np.asarray(archive["qpos"], dtype=np.float64)
      fps = float(archive["fps"].item())
      robot_name = str(archive["robot_name"].item())
      quat_order = str(archive["root_quat_order"].item())
      names_array = archive["robot_joint_names"]
      names = [str(name) for name in names_array.tolist()]
  except (OSError, ValueError, TypeError, EOFError, zipfile.BadZipFile) as exc:
    report.errors.append(f"Cannot read motion: {exc}")
    return report
  fps_valid = bool(np.isfinite(fps) and fps > 0)
  fps_matches = bool(fps_valid and np.isclose(fps, expected_fps))
  report.checks["fps"] = {
    "actual": fps if np.isfinite(fps) else None,
    "expected": expected_fps,
    "matches_loader": fps_matches,
  }
  report.checks["robot_name"] = robot_name
  report.checks["root_quat_order"] = quat_order
  if not fps_matches:
    report.errors.append("Motion FPS does not match loader control frequency")
  if robot_name != "q1_v2":
    report.errors.append("Expected robot_name=q1_v2")
  if quat_order != "wxyz":
    report.errors.append("Expected root_quat_order=wxyz")
  names_valid = (
    names_array.ndim == 1 and len(names) == 22 and set(names) == set(robot.joint_names)
  )
  report.checks["joint_names_match_xml"] = names_valid
  if not names_valid:
    report.errors.append("Joint names mismatch Q1 XML or contain duplicates")
  report.checks["qpos_shape"] = list(raw.shape)
  if raw.ndim != 2 or raw.shape[1] != 29 or len(raw) < 20:
    report.errors.append("Expected >=20 frames with qpos shape (frames, 29)")
    return report
  report.checks["nonfinite_qpos_values"] = int((~np.isfinite(raw)).sum())
  if not np.isfinite(raw).all():
    report.errors.append("Nonfinite qpos")
    return report
  with np.errstate(over="ignore", invalid="ignore"):
    norms = np.linalg.norm(raw[:, 3:7], axis=1)
  quaternion_valid = bool(
    np.isfinite(norms).all() and np.all(np.abs(norms - 1) <= 0.01)
  )
  report.checks["quaternion"] = {
    "minimum_norm": float(norms.min()) if np.isfinite(norms.min()) else None,
    "maximum_norm": float(norms.max()) if np.isfinite(norms.max()) else None,
    "max_unit_norm_error": (
      float(np.abs(norms - 1).max()) if np.isfinite(norms).all() else None
    ),
    "nonfinite_computed_norms": int((~np.isfinite(norms)).sum()),
    "loader_tolerance": 0.01,
    "unit_length_valid": quaternion_valid,
    "adjacent_sign_flips": (
      int(((raw[1:, 3:7] * raw[:-1, 3:7]).sum(axis=1) < 0).sum())
      if quaternion_valid
      else None
    ),
  }
  if not quaternion_valid:
    report.errors.append("Root quaternions violate loader unit-length tolerance")
  if not names_valid or not quaternion_valid or quat_order != "wxyz":
    return report
  if robot_name != "q1_v2":
    return report
  qpos = np.concatenate(
    (raw[:, :7], raw[:, [7 + names.index(name) for name in robot.joint_names]]),
    axis=1,
  )
  # Same normalization as the loader, applied to this local array only.
  qpos[:, 3:7] /= norms[:, None]
  joints = qpos[:, 7:]
  limits = np.asarray(robot.model.jnt_range[1:], dtype=np.float64)
  limited = np.asarray(cast(GeometryModel, robot.model).jnt_limited[1:], dtype=bool)
  excess = np.maximum(joints - limits[:, 1], limits[:, 0] - joints).clip(min=0)
  excess[:, ~limited] = 0
  violations = excess > numerical_tolerance
  per_joint: list[dict[str, object]] = []
  for joint, name in enumerate(robot.joint_names):
    diagnostic: dict[str, object] = {
      "name": name,
      "xml_limited": bool(limited[joint]),
      "xml_range_rad": limits[joint].tolist(),
      "motion_range_rad": [
        float(joints[:, joint].min()),
        float(joints[:, joint].max()),
      ],
      "out_of_range_frames": int(violations[:, joint].sum()),
      "maximum_excess_rad": float(excess[:, joint].max()),
    }
    if fps_valid:
      diagnostic["max_absolute_difference_speed_rad_s"] = float(
        np.abs(np.diff(joints[:, joint]) * fps).max()
      )
    per_joint.append(diagnostic)
  report.diagnostics["joint_limits"] = {
    "source": "provided Q1 XML; not paper G1 filtering thresholds",
    "numerical_tolerance_rad": numerical_tolerance,
    "violating_frames": int(violations.any(axis=1).sum()),
    "maximum_excess_rad": float(excess.max()),
    "per_joint": per_joint,
    "speed_method": "adjacent-frame difference at actual FPS; no speed cutoff",
  }
  if violations.any():
    report.warnings.append("Reference joints exceed provided Q1 XML ranges")
  if fps_valid:
    report.diagnostics["duration_s"] = (len(raw) - 1) / fps
  geometry = geometry_diagnostics(robot, qpos, max_fk_frames, numerical_tolerance)
  report.diagnostics["geometry"] = geometry
  ground = cast(dict[str, object], geometry["ground_clearance"])
  if ground["distance_computation_verified"]:
    feet = cast(dict[str, dict[str, object]], ground["feet"])
    if any(cast(int, foot["penetration_frames"]) > 0 for foot in feet.values()):
      report.warnings.append("Sampled foot collision shapes penetrate XML ground")
  else:
    report.warnings.append("Exact foot-ground clearance was not verified")
  report.compatible_with_loader = not report.errors
  return report


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--robot-xml", type=Path, required=True)
  parser.add_argument("--motion-path", type=Path, action="append", default=[])
  parser.add_argument("--motion-dir", type=Path)
  parser.add_argument("--motion-pattern", default="*.npz")
  parser.add_argument("--limit", type=int, default=5)
  parser.add_argument("--max-fk-frames", type=int, default=1000)
  parser.add_argument("--expected-fps", type=float, default=50.0)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  try:
    paths = select_paths(
      args.motion_path, args.motion_dir, args.motion_pattern, args.limit
    )
    if args.max_fk_frames < 1:
      raise ValueError("max FK frames must be positive")
    robot = Q1Robot(args.robot_xml)
    reports = [
      inspect_motion(robot, path, args.expected_fps, args.max_fk_frames)
      for path in paths
    ]
  except ValueError as exc:
    parser.error(str(exc))
  result = {
    "schema_version": 1,
    "robot_xml": str(robot.xml),
    "robot_asset_sha256": robot.asset_sha256,
    "selected_clip_count": len(reports),
    "selection_limit": args.limit,
    "selection_order": "explicit input or bounded filesystem glob discovery",
    "policy": "diagnostics only; no XML/motion edits or calibrated Q1 feasibility filter",
    "unverified": ["self-collision", "torque feasibility", "dynamic trackability"],
    "motions": [asdict(report) for report in reports],
  }
  rendered = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
  if args.output is not None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
      stream.write(rendered)
  else:
    print(rendered, end="")


if __name__ == "__main__":
  main()
