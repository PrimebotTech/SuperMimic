"""Targeted diagnostics checks using a synthetic Q1 articulation."""

import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from supermimic.preflight import (
  geometry_diagnostics,
  inspect_motion,
  main,
  select_paths,
)
from supermimic.robot import Q1Robot


@pytest.fixture
def preflight_robot(tmp_path: Path) -> Q1Robot:
  tree = ET.Element("mujoco", model="q1_v2")
  ET.SubElement(tree, "compiler", angle="radian", autolimits="true")
  world = ET.SubElement(tree, "worldbody")
  ET.SubElement(world, "geom", name="floor", type="plane", size="1 1 0.1")
  pelvis = ET.SubElement(world, "body", name="pelvis", pos="0 0 0.5")
  ET.SubElement(pelvis, "freejoint")
  ET.SubElement(pelvis, "geom", type="sphere", size="0.01", mass="1")
  actuators = ET.SubElement(tree, "actuator")
  for joint in range(22):
    name = (
      "left_ankle_roll_link"
      if joint == 0
      else "right_ankle_roll_link"
      if joint == 1
      else "torso_link"
      if joint == 21
      else f"link{joint}"
    )
    body = ET.SubElement(
      pelvis,
      "body",
      name=name,
      pos=f"0 {0.1 if joint == 0 else -0.1} -0.4",
    )
    ET.SubElement(body, "joint", name=f"joint{joint}", type="hinge", range="-1 1")
    ET.SubElement(body, "geom", type="sphere", size="0.05", mass="0.1")
    # A larger disabled visual geom must not influence foot clearance.
    if joint < 2:
      ET.SubElement(
        body,
        "geom",
        type="sphere",
        size="0.4",
        contype="0",
        conaffinity="0",
        density="0",
      )
    ET.SubElement(actuators, "motor", joint=f"joint{joint}", ctrlrange="-10 10")
  path = tmp_path / "robot.xml"
  ET.ElementTree(tree).write(path)
  return Q1Robot(path)


def save_motion(
  path: Path, robot: Q1Robot, qpos: np.ndarray, fps: float = 50, reverse: bool = False
) -> None:
  names = list(robot.joint_names)
  if reverse:
    names.reverse()
    qpos = np.concatenate((qpos[:, :7], qpos[:, 7:][:, ::-1]), axis=1)
  np.savez(
    path,
    qpos=qpos,
    fps=fps,
    robot_name="q1_v2",
    root_quat_order="wxyz",
    robot_joint_names=names,
  )


def resting_qpos(robot: Q1Robot, frames: int = 20) -> np.ndarray:
  return np.tile(robot.model.qpos0, (frames, 1))


@pytest.mark.parametrize("fps", [50, 120])
def test_joint_mapping_limits_speed_and_fps_diagnostic(
  preflight_robot: Q1Robot, tmp_path: Path, fps: int
):
  qpos = resting_qpos(preflight_robot)
  qpos[1:, 7] = 1.25
  path = tmp_path / "motion.npz"
  save_motion(path, preflight_robot, qpos, fps=fps, reverse=True)
  report = inspect_motion(preflight_robot, path, max_fk_frames=3)
  assert report.compatible_with_loader == (fps == 50)
  assert report.feasibility_assessment == "not_assessed"
  assert "expected FPS only" in report.compatibility_scope
  if fps == 120:
    assert len(report.errors) == 1 and "FPS" in report.errors[0]
  limits = cast(dict[str, object], report.diagnostics["joint_limits"])
  per_joint = cast(list[dict[str, object]], limits["per_joint"])
  assert per_joint[0]["name"] == "joint0"
  assert per_joint[0]["out_of_range_frames"] == 19
  assert per_joint[0]["maximum_excess_rad"] == 0.25
  assert per_joint[0]["max_absolute_difference_speed_rad_s"] == 1.25 * fps
  geometry = cast(dict[str, object], report.diagnostics["geometry"])
  assert geometry["sampled_frames"] == 3
  assert not geometry["all_frames_checked"]
  # Input trajectory and joint ordering remain untouched.
  with np.load(path) as archive:
    assert archive["qpos"][1, -1] == 1.25


def test_geometry_uses_collision_surfaces_and_reports_penetration(
  preflight_robot: Q1Robot,
):
  qpos = resting_qpos(preflight_robot)
  qpos[1, 2] = 0.4
  geometry = geometry_diagnostics(preflight_robot, qpos, 20, 1e-6)
  ground = cast(dict[str, object], geometry["ground_clearance"])
  assert ground["distance_computation_verified"]
  assert not ground["clearance_filter_applied"]
  feet = cast(dict[str, dict[str, object]], ground["feet"])
  for foot in feet.values():
    assert foot["minimum_clearance_m"] == pytest.approx(-0.05)
    assert foot["penetration_frames"] == 1
    assert len(cast(list[int], foot["collision_geom_ids"])) == 1
  assert geometry["minimum_link_origin_height_m"] == pytest.approx(0)
  assert geometry["all_frames_checked"]


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_nonfinite_rejected_without_nonfinite_json(
  preflight_robot: Q1Robot, tmp_path: Path, bad_value: float
):
  qpos = resting_qpos(preflight_robot)
  qpos[4, 7] = bad_value
  path = tmp_path / "bad.npz"
  save_motion(path, preflight_robot, qpos)
  report = inspect_motion(preflight_robot, path)
  assert report.checks["nonfinite_qpos_values"] == 1
  assert "Nonfinite qpos" in report.errors
  json.dumps(report.checks, allow_nan=False)
  assert "geometry" not in report.diagnostics


def test_invalid_quaternion_stops_geometry(preflight_robot: Q1Robot, tmp_path: Path):
  qpos = resting_qpos(preflight_robot)
  qpos[0, 3:7] = 0
  path = tmp_path / "bad_quaternion.npz"
  save_motion(path, preflight_robot, qpos)
  report = inspect_motion(preflight_robot, path)
  assert any("quaternion" in error for error in report.errors)
  assert "geometry" not in report.diagnostics


def test_finite_invalid_quaternion_norm_overflow_retains_strict_json(
  preflight_robot: Q1Robot, tmp_path: Path
):
  qpos = resting_qpos(preflight_robot)
  qpos[0, 3] = 1e308
  path = tmp_path / "huge_quaternion.npz"
  save_motion(path, preflight_robot, qpos)
  report = inspect_motion(preflight_robot, path)
  assert not report.compatible_with_loader
  quaternion = cast(dict[str, object], report.checks["quaternion"])
  assert quaternion["nonfinite_computed_norms"] == 1
  json.dumps(asdict(report), allow_nan=False)


def test_selection_is_bounded_and_explicit_counts_are_not_silently_truncated(
  tmp_path: Path,
):
  paths = [tmp_path / f"motion{index}.npz" for index in range(3)]
  for path in paths:
    path.touch()
  assert len(select_paths([], tmp_path, "*.npz", 2)) == 2
  with pytest.raises(ValueError, match="count exceeds"):
    select_paths(paths, None, "*.npz", 2)
  with pytest.raises(ValueError, match="between"):
    select_paths([], tmp_path, "*.npz", 1000)


def test_cli_creates_strict_json_report(
  preflight_robot: Q1Robot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
  path, output = tmp_path / "good.npz", tmp_path / "report.json"
  save_motion(path, preflight_robot, resting_qpos(preflight_robot))
  monkeypatch.setattr(
    sys,
    "argv",
    [
      "supermimic-preflight",
      "--robot-xml",
      str(preflight_robot.xml),
      "--motion-path",
      str(path),
      "--output",
      str(output),
    ],
  )
  main()
  report = json.loads(output.read_text())
  assert report["selected_clip_count"] == 1
  assert report["motions"][0]["compatible_with_loader"]
  assert report["motions"][0]["diagnostics"]["geometry"]["all_frames_checked"]
