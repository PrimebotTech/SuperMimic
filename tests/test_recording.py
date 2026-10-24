"""Full-sequence assembly, failure scoring, and published-checkpoint selection."""

from pathlib import Path

import numpy as np
import pytest

from supermimic.record import assemble, diagnostic_summary, sequences
from supermimic.video_watcher import pending_checkpoints


def test_full_lafan_sequence_requires_contiguous_chunks(tmp_path: Path):
  for name in ("walk__clip000", "walk__clip001", "gap__clip000", "gap__clip002"):
    (tmp_path / f"{name}.npz").touch()
  groups = sequences(tmp_path)
  assert list(groups) == ["walk"]
  assert [p.stem for p in groups["walk"]] == ["walk__clip000", "walk__clip001"]


def test_assembly_keeps_entire_original_motion(tmp_path: Path):
  paths = []
  for index in range(2):
    qpos = np.zeros((20, 29), dtype=np.float32)
    qpos[:, 0] = np.arange(20) + index * 20
    qpos[:, 3] = 1
    path = tmp_path / f"walk__clip{index:03d}.npz"
    np.savez(
      path,
      qpos=qpos,
      robot_name="q1_v2",
      fps=50,
      robot_joint_names=np.array([f"joint_{j}" for j in range(22)]),
    )
    paths.append(path)
  output = tmp_path / "full.npz"
  assemble(paths, output)
  with np.load(output) as data:
    np.testing.assert_array_equal(data["qpos"][:, 0], np.arange(40))


def test_failed_video_continues_but_score_counts_remaining_steps_as_zero():
  poses = np.zeros((5, 29), dtype=np.float32)
  poses[:, 3] = 1
  result = diagnostic_summary(
    poses, poses.copy(), np.full(4, 0.1), np.array([False, True, False, True])
  )
  assert result["first_failure_frame"] == 2
  assert result["first_failure_seconds"] == pytest.approx(0.04)
  assert result["normalized_return"] == pytest.approx(0.5)
  assert result["recorded_frames"] == 5
  assert result["root_orientation_rmse_rad"] == pytest.approx(0)


def test_watcher_only_reads_published_matching_checkpoints(tmp_path: Path):
  videos = tmp_path / "videos"
  for name in (
    "checkpoint_30.pt",
    "checkpoint_5000.pt",
    "checkpoint_10000.pt",
    ".checkpoint_15000.pt.tmp",
    "checkpoint_bad.pt",
  ):
    (tmp_path / name).touch()
  assert [p.name for p in pending_checkpoints(tmp_path, videos, 5000)] == [
    "checkpoint_5000.pt",
    "checkpoint_10000.pt",
  ]
  done = videos / "checkpoint_5000"
  done.mkdir(parents=True)
  (done / "diagnostics.json").touch()
  assert [p.name for p in pending_checkpoints(tmp_path, videos, 5000)] == [
    "checkpoint_10000.pt",
  ]


def test_watcher_finds_checkpoints_after_a_fresh_recovery_attempt(tmp_path: Path):
  old, new = tmp_path / "attempt_00", tmp_path / "attempt_01"
  old.mkdir()
  new.mkdir()
  (old / "checkpoint_5000.pt").touch()
  (new / "checkpoint_10000.pt").touch()
  (new / ".checkpoint_15000.pt.tmp").touch()
  assert pending_checkpoints(tmp_path, tmp_path / "videos", 5000) == [
    old / "checkpoint_5000.pt",
    new / "checkpoint_10000.pt",
  ]
