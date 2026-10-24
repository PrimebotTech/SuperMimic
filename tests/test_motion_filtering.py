"""CPU geometry tests, conservative labels and crash-safe screening identity."""

import json
import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import Q1_JOINT_NAMES, link_name

from supermimic.filter_motions import (
  CollisionGeometry,
  FilterConfig,
  classify_labels,
  export_manifests,
  read_motion,
  run_screen,
  screen_motion,
)


def write_fixture(path: Path, overlap: bool = False) -> Path:
  tree = ET.Element("mujoco", model="q1_v2")
  ET.SubElement(tree, "compiler", angle="radian", autolimits="true")
  world = ET.SubElement(tree, "worldbody")
  ET.SubElement(world, "geom", type="plane", size="5 5 .1")
  root = ET.SubElement(world, "body", name="pelvis", pos="0 0 .52")
  ET.SubElement(root, "freejoint")
  ET.SubElement(root, "geom", type="sphere", size=".02", mass="2")
  actuators = ET.SubElement(tree, "actuator")
  for index, name in enumerate(Q1_JOINT_NAMES):
    is_foot = "ankle_roll" in name
    x = 0 if overlap and index in (0, 6) else 0.2 + index * 0.10
    position = f"{x} .3 {-0.5 if is_foot else 0}"
    body = ET.SubElement(root, "body", name=link_name(name), pos=position)
    ET.SubElement(body, "joint", name=name, axis="0 0 1", range="-2 2")
    attrs = {
      "type": "box" if is_foot else "sphere",
      "size": ".05 .03 .02" if is_foot else ".04",
      "mass": ".1",
      # Neither proxy can collide through normal masks, but independent signed
      # distance must still detect overlap between physical geometries.
      "contype": "2",
      "conaffinity": "0",
    }
    ET.SubElement(body, "geom", attrs)
    ET.SubElement(actuators, "motor", joint=name, ctrlrange="-20 20")
  ET.ElementTree(tree).write(path)
  return path


@pytest.fixture
def geometry(tmp_path: Path) -> CollisionGeometry:
  return CollisionGeometry(write_fixture(tmp_path / "q1.xml"))


def write_motion(path: Path, geometry: CollisionGeometry, qpos: Any = None) -> Path:
  path.parent.mkdir(parents=True, exist_ok=True)
  if qpos is None:
    qpos = np.tile(geometry.model.qpos0, (80, 1))
  np.savez_compressed(
    path,
    qpos=qpos,
    robot_name="q1_v2",
    root_quat_order="wxyz",
    fps=50.0,
    robot_joint_names=np.array(geometry.joint_names),
    motion_key=path.stem,
    source_qpos_sha256="fixture-source",
  )
  return path


def test_labels_require_precise_evidence_and_opaque_ids_are_reviewed():
  assert classify_labels("omomo/sub1_whitechair.npz", {}, {})[0] == "rejected"
  assert classify_labels("KIT/carry_box.npz", {}, {})[0] == "rejected"
  assert classify_labels("KIT/carry_boxing.npz", {}, {})[0] == "review"
  assert classify_labels("KIT/standing_stable.npz", {}, {})[0] == "accepted"
  assert classify_labels("KIT/push.npz", {}, {})[0] == "review"
  assert classify_labels("KIT/walk_near_smalltable.npz", {}, {})[0] == "review"
  assert classify_labels("KIT/carrying_suitcase.npz", {}, {})[0] == "rejected"
  assert classify_labels("CMU_GMR/120_07.npz", {}, {})[0] == "review"
  assert classify_labels("lafan1/fallAndGetUp1_subject5.npz", {}, {})[0] == "accepted"
  annotations = {
    "CMU_GMR/120_07": {
      "classification": "non_interaction",
      "evidence": "Human-reviewed walking",
    }
  }
  assert (
    classify_labels("CMU_GMR/120_07__clip002.npz", {}, annotations)[0] == "accepted"
  )
  with pytest.raises(ValueError, match="evidence"):
    classify_labels(
      "CMU_GMR/120_07__clip002.npz",
      {},
      {"CMU_GMR/120_07": {"classification": "non_interaction"}},
    )


def test_real_signed_geometry_ignores_masks_and_excludes_parent_pairs(geometry):
  calibration = geometry.calibration()
  assert calibration["excluded_pairs"]["adjacent_welded_clusters"] > 0
  pairs = geometry.pairs
  first = int(geometry.model.body("left_hip_pitch_link").id)
  second = int(geometry.model.body("right_hip_pitch_link").id)
  pair_index = next(
    i
    for i, (a, b) in enumerate(pairs)
    if (
      geometry.model.geom_bodyid[a] == first and geometry.model.geom_bodyid[b] == second
    )
  )
  a, b = pairs[pair_index]
  geometry.data.geom_xpos[b] = geometry.data.geom_xpos[a]
  sample = geometry.penetration()
  assert sample.absolute_depth == pytest.approx(0.08)
  assert sample.maximum_excess == pytest.approx(0.08)
  assert sample.worst_pair == pair_index
  assert sample.severe_pair == pair_index


def test_neutral_model_overlap_does_not_reject_its_own_fixture(tmp_path):
  geometry = CollisionGeometry(write_fixture(tmp_path / "q1.xml", overlap=True))
  assert geometry.baseline.max() == pytest.approx(0.08)
  sample = geometry.penetration()
  assert sample.absolute_depth == pytest.approx(0.08)
  assert sample.maximum_excess == 0.0
  assert sample.severe_pair is None


def test_severe_pair_is_preserved_when_another_pair_has_larger_excess(geometry):
  pairs = geometry.pairs.copy()
  indices = []
  for names in (
    ("left_hip_pitch_link", "right_hip_pitch_link"),
    ("left_hip_roll_link", "right_hip_roll_link"),
  ):
    bodies = [geometry.model.body(name).id for name in names]
    indices.append(
      next(
        i
        for i, (a, b) in enumerate(pairs)
        if (
          geometry.model.geom_bodyid[a] == bodies[0]
          and geometry.model.geom_bodyid[b] == bodies[1]
        )
      )
    )
  geometry.pairs = pairs[indices]
  geometry.bound_sum = geometry.model.geom_rbound[geometry.pairs].sum(1)
  geometry.baseline = np.array([0.063, 0.0])
  for (a, b), separation in zip(geometry.pairs, (0.001, 0.061), strict=True):
    geometry.data.geom_xpos[b] = geometry.data.geom_xpos[a] + [separation, 0, 0]
  sample = geometry.penetration()
  assert sample.absolute_depth == pytest.approx(0.079)
  assert sample.maximum_excess == pytest.approx(0.019)
  assert sample.worst_pair == 1
  assert sample.severe_pair == 0
  assert sample.severe_depth == pytest.approx(0.079)
  assert sample.severe_excess == pytest.approx(0.016)


def test_welded_child_is_excluded_from_its_clusters_adjacent_body(tmp_path):
  xml = write_fixture(tmp_path / "q1.xml")
  tree = ET.parse(xml)
  parent = tree.find(".//body[@name='left_hip_pitch_link']")
  assert parent is not None
  child = ET.SubElement(parent, "body", name="fixed_child", pos="0 0 .1")
  ET.SubElement(child, "geom", type="sphere", size=".02", mass=".1")
  tree.write(xml)
  geometry = CollisionGeometry(xml)
  fixed = geometry.model.body("fixed_child").id
  pelvis = geometry.model.body("pelvis").id
  for first, second in geometry.pairs:
    bodies = {geometry.model.geom_bodyid[first], geometry.model.geom_bodyid[second]}
    assert bodies != {fixed, pelvis}


def test_named_joint_reorder_and_nonfinite_contract(tmp_path, geometry):
  path = write_motion(tmp_path / "walk.npz", geometry)
  with np.load(path, allow_pickle=False) as data:
    content = {key: data[key].copy() for key in data.files}
  original = content["qpos"].copy()
  original[:, 7:] = np.linspace(-0.1, 0.1, 22)
  content["qpos"] = np.concatenate((original[:, :7], original[:, 7:][:, ::-1]), 1)
  content["robot_joint_names"] = content["robot_joint_names"][::-1]
  np.savez(path, **content)
  decoded, _, file_hash = read_motion(path, geometry, 50)
  np.testing.assert_allclose(decoded, original)
  assert len(file_hash) == 64
  content["qpos"][0, 0] = np.nan
  np.savez(path, **content)
  with pytest.raises(ValueError, match="Nonfinite"):
    read_motion(path, geometry, 50)


def test_stationary_suspension_and_ground_penetration_are_not_corrected(
  tmp_path, geometry
):
  root = tmp_path / "motions"
  nominal = np.tile(geometry.model.qpos0, (80, 1))
  grounded = write_motion(root / "lafan1/walk__clip000.npz", geometry, nominal)
  assert screen_motion(grounded, root, geometry, FilterConfig())["status"] == "accepted"
  suspended = nominal.copy()
  suspended[:, 2] += 0.12
  path = write_motion(root / "lafan1/walk__clip001.npz", geometry, suspended)
  before = path.read_bytes()
  report = screen_motion(path, root, geometry, FilterConfig())
  assert report["status"] == "rejected"
  assert "stationary_unsupported_pose" in {r["code"] for r in report["reasons"]}
  assert path.read_bytes() == before
  penetrated = nominal.copy()
  penetrated[:, 2] -= 0.06
  path = write_motion(root / "lafan1/walk__clip002.npz", geometry, penetrated)
  report = screen_motion(path, root, geometry, FilterConfig())
  assert report["status"] == "rejected"
  assert "ground_penetration" in {r["code"] for r in report["reasons"]}


def test_grounded_short_ballistic_jump_is_retained(tmp_path, geometry):
  root = tmp_path / "motions"
  qpos = np.tile(geometry.model.qpos0, (80, 1))
  times = np.arange(80) / 50 - 0.2
  height = 3 * times - 0.5 * 9.81 * times**2
  qpos[:, 2] += np.where(times >= 0, height.clip(min=0), 0)
  path = write_motion(root / "lafan1/jump__clip000.npz", geometry, qpos)
  report = screen_motion(path, root, geometry, FilterConfig())
  assert report["status"] == "accepted"
  flight = report["measurements"]["unsupported_segments"][0]
  assert flight["has_grounded_before"] and flight["has_grounded_after"]
  assert flight["fitted_com_acceleration_m_s2"] == pytest.approx(-9.81)


def test_four_stationary_samples_do_not_prove_point_four_seconds(tmp_path, geometry):
  root = tmp_path / "motions"
  qpos = np.tile(geometry.model.qpos0, (80, 1))
  qpos[:20, 2] += 0.12
  path = write_motion(root / "lafan1/walk__clip000.npz", geometry, qpos)
  report = screen_motion(path, root, geometry, FilterConfig())
  assert report["status"] == "review"
  reasons = {reason["code"]: reason for reason in report["reasons"]}
  assert "stationary_unsupported_pose" not in reasons
  assert reasons["stationary_duration_uncertain"]["seconds"] == pytest.approx(0.3)


def test_poor_com_fit_does_not_claim_reliable_flight_acceleration(tmp_path, geometry):
  root = tmp_path / "motions"
  qpos = np.tile(geometry.model.qpos0, (80, 1))
  times = np.arange(80) / 50
  qpos[:, 0] += 0.4 * times
  qpos[:, 2] += 0.25 + 0.10 * np.sin(4 * np.pi * times)
  path = write_motion(root / "lafan1/walk__clip000.npz", geometry, qpos)
  report = screen_motion(path, root, geometry, FilterConfig())
  assert report["status"] == "review"
  flight = report["measurements"]["unsupported_segments"][0]
  assert flight["quadratic_com_fit_rmse_m"] > 0.02
  assert not flight["quadratic_fit_reliable"]


def test_conservative_whole_sequence_excludes_an_accepted_sibling(tmp_path):
  database = sqlite3.connect(":memory:")
  database.execute("CREATE TABLE results (path TEXT, payload TEXT)")
  for index, status in enumerate(("accepted", "rejected")):
    row = {
      "path": f"lafan1/walk__clip{index:03}.npz",
      "status": status,
      "sequence_id": "lafan1/walk",
    }
    database.execute("INSERT INTO results VALUES (?,?)", (row["path"], json.dumps(row)))
  export_manifests(database, tmp_path, Path("/generic/motions"))
  assert "clip000" in (tmp_path / "accepted.txt").read_text()
  assert not (tmp_path / "accepted_sequences.txt").read_text()
  assert (
    json.loads((tmp_path / "sequences.json").read_text())["lafan1/walk"]["status"]
    == "rejected"
  )


@pytest.mark.parametrize(
  "expected",
  [
    ["lafan1/walk__clip000.npz", "lafan1/walk__clip001.npz"],
    ["lafan1/walk__clip001.npz"],
  ],
)
def test_partial_or_nonzero_start_chunks_are_not_complete_sequences(tmp_path, expected):
  database = sqlite3.connect(":memory:")
  database.execute("CREATE TABLE results (path TEXT, payload TEXT)")
  relative = expected[-1]
  row = {"path": relative, "status": "accepted", "sequence_id": "lafan1/walk"}
  database.execute("INSERT INTO results VALUES (?,?)", (relative, json.dumps(row)))
  export_manifests(
    database, tmp_path, Path("/generic/motions"), {"lafan1/walk": expected}
  )
  assert not (tmp_path / "accepted_sequences.txt").read_text()
  result = json.loads((tmp_path / "sequences.json").read_text())["lafan1/walk"]
  assert result["status"] == "review"
  assert not result["complete_available_input"]


def test_resumption_is_idempotent_and_refuses_changed_screen_identity(tmp_path):
  xml = write_fixture(tmp_path / "q1.xml")
  geometry = CollisionGeometry(xml)
  root, output = tmp_path / "motions", tmp_path / "screen"
  write_motion(root / "lafan1/walk__clip000.npz", geometry)
  config = FilterConfig()
  result = run_screen(xml, root, output, config, workers=1, batch_size=1)
  assert result["phase"] == "complete"
  assert result["counts"] == {"accepted": 1}
  with (output / "results.jsonl").open("a") as stream:
    stream.write('{"partial":')
  result = run_screen(xml, root, output, config, workers=1)
  assert result["completed_files"] == 1
  assert len((output / "results.jsonl").read_text().splitlines()) == 1
  with pytest.raises(ValueError, match="identity differs"):
    run_screen(xml, root, output, FilterConfig(sample_hz=5), workers=1)


def test_output_never_changes_motion_library(tmp_path):
  xml = write_fixture(tmp_path / "q1.xml")
  geometry = CollisionGeometry(xml)
  root = tmp_path / "motions"
  write_motion(root / "lafan1/walk__clip000.npz", geometry)
  with pytest.raises(ValueError, match="immutable"):
    run_screen(xml, root, root / "filter", FilterConfig(), workers=1)
