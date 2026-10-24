"""Monitoring must follow the live attempt and distinguish startup from a stall."""

import json
import os
from pathlib import Path

import pytest

from supermimic.training_monitor import Config, progress_report


@pytest.fixture
def monitored_run(tmp_path: Path) -> tuple[Config, Path]:
  active = tmp_path / "attempt_100"
  active.mkdir()
  (tmp_path / "supervisor.json").write_text(
    json.dumps({"state": "running", "attempt": str(active), "started": 1000})
  )
  obsolete = tmp_path / "attempt_99"
  obsolete.mkdir()
  (obsolete / "metrics.jsonl").write_text(
    json.dumps({"iteration": 99999, "update_seconds": 99}) + "\n"
  )
  return Config(run_root=tmp_path, minimum_disk_gib=0), active


def test_startup_does_not_report_previous_attempt(monitored_run) -> None:
  cfg, active = monitored_run
  report = progress_report(cfg, 1100)
  assert report["attempt"] == str(active)
  assert report["phase"] == "initializing"
  assert "iteration" not in report
  assert report["health"] == "ok"
  assert progress_report(cfg, 3000)["health"] == "warning"


def test_live_attempt_partial_json_and_startup_timing(monitored_run) -> None:
  cfg, active = monitored_run
  metrics = active / "metrics.jsonl"
  rows = [
    {"iteration": 5001, "update_seconds": 100, "startup_update": True},
    {"iteration": 5002, "update_seconds": 1.2},
    {"iteration": 5003, "update_seconds": 1.4},
  ]
  metrics.write_text("\n".join(json.dumps(row) for row in rows) + '\n{"iteration":')
  os.utime(metrics, (1200, 1200))
  report = progress_report(cfg, 1210)
  assert report["iteration"] == 5003
  assert report["recent_update_seconds"] == pytest.approx(1.3)
  assert report["recent_metrics_finite"]
  assert report["stale_seconds"] == 10
  assert report["health"] == "ok"
  assert progress_report(cfg, 1510)["health"] == "warning"


def test_nonfinite_report_and_only_completed_videos(monitored_run) -> None:
  cfg, active = monitored_run
  metrics = active / "metrics.jsonl"
  metrics.write_text(json.dumps({"iteration": 5001, "actor_grad_norm": float("nan")}))
  os.utime(metrics, (1200, 1200))
  completed = cfg.run_root / "videos" / "checkpoint_5000"
  completed.mkdir(parents=True)
  (completed / "diagnostics.json").write_text("{}")
  (cfg.run_root / "videos" / "checkpoint_10000").mkdir()
  report = progress_report(cfg, 1210)
  assert not report["recent_metrics_finite"]
  assert report["health"] == "warning"
  assert report["completed_video_iterations"] == [5000]


def test_unavailable_supervisor_is_reported(monitored_run) -> None:
  cfg, _ = monitored_run
  (cfg.run_root / "supervisor.json").write_text("{")
  report = progress_report(cfg, 1210)
  assert report["supervisor"] == "starting"
  assert report["health"] == "warning"
  assert "supervisor status unavailable" in report["warnings"][0]


def test_direct_run_reports_metrics_without_a_private_supervisor(tmp_path: Path):
  metrics = tmp_path / "metrics.jsonl"
  metrics.write_text(json.dumps({"iteration": 17, "update_seconds": 1.25}) + "\n")
  os.utime(metrics, (1200, 1200))
  cfg = Config(run_root=tmp_path, minimum_disk_gib=0)
  report = progress_report(cfg, 1210)
  assert report["supervisor"] == "unmanaged"
  assert report["attempt"] == str(tmp_path)
  assert report["iteration"] == 17
  assert report["recent_update_seconds"] == 1.25
  assert report["health"] == "ok"
  assert progress_report(cfg, 1510)["health"] == "warning"
