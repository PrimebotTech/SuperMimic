"""Persistent headless progress and health reports for the active training attempt."""

import json
import math
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro


@dataclass
class Config:
  run_root: Path
  iterations: int = 250000
  poll_seconds: int = 30
  stale_seconds: int = 300
  initialization_seconds: int = 1800
  minimum_disk_gib: float = 10.0
  once: bool = False


def progress_report(cfg: Config, now: float) -> dict[str, Any]:
  """Never attribute an old attempt's metrics to a newly starting process."""
  warnings: list[str] = []
  status_path = cfg.run_root / "supervisor.json"
  status: dict[str, Any]
  try:
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
  except (OSError, json.JSONDecodeError) as error:
    status = {}
    warnings.append(f"supervisor status unavailable: {error}")
  active = Path(status["attempt"]) if status.get("attempt") else None
  if not status_path.exists() and (cfg.run_root / "metrics.jsonl").is_file():
    active = cfg.run_root
    status = {"state": "unmanaged"}
  metrics_path = active / "metrics.jsonl" if active else None
  report: dict[str, Any] = {
    "time": now,
    "supervisor": status.get("state", "starting"),
    "attempt": str(active) if active else None,
    "phase": "initializing",
    "restart": status.get("restart"),
    "warnings": warnings,
  }
  metrics = []
  if metrics_path is not None and metrics_path.exists():
    # Discard any partial first or last line; training appends one JSON per update.
    with metrics_path.open("rb") as stream:
      stream.seek(max(0, metrics_path.stat().st_size - 100_000))
      lines = stream.read().decode().splitlines()
    for line in lines:
      try:
        value = json.loads(line)
      except json.JSONDecodeError:
        continue
      if isinstance(value, dict) and "iteration" in value:
        metrics.append(value)
  if metrics and metrics_path is not None:
    latest = metrics[-1]
    seconds = [
      m["update_seconds"]
      for m in metrics[-200:]
      if m.get("update_seconds", 0) > 0
      and math.isfinite(m["update_seconds"])
      and m["iteration"] > 1
      and not m.get("startup_update", False)
    ]
    mean = sum(seconds) / len(seconds) if seconds else None
    finite = all(
      math.isfinite(value)
      for metric in metrics
      for value in metric.values()
      if isinstance(value, (int, float))
    )
    stale = max(0, now - metrics_path.stat().st_mtime)
    report.update(
      phase="training",
      iteration=latest["iteration"],
      recent_update_seconds=mean,
      collection_reward=latest.get("collection_reward"),
      actor_grad_norm=latest.get("actor_grad_norm"),
      recent_metrics_finite=finite,
      stale_seconds=stale,
      remaining_hours=(
        max(0, cfg.iterations - latest["iteration"]) * mean / 3600 if mean else None
      ),
    )
    if not finite:
      warnings.append("nonfinite training metrics in the recent log")
    if status.get("state") in ("running", "unmanaged") and stale > cfg.stale_seconds:
      warnings.append(f"no new training metrics for {stale:.0f} seconds")
  elif status.get("state") == "running":
    initializing = max(0, now - status.get("started", now))
    report["initialization_seconds"] = initializing
    if initializing > cfg.initialization_seconds:
      warnings.append(f"initialization exceeds {cfg.initialization_seconds} seconds")
  if status.get("state") == "crashed":
    warnings.append("training crashed; check supervisor recovery status")
  free_gib = shutil.disk_usage(cfg.run_root).free / 1024**3
  report["disk_free_gib"] = free_gib
  if free_gib < cfg.minimum_disk_gib:
    warnings.append(f"run filesystem has only {free_gib:.2f} GiB free")
  videos = cfg.run_root / "videos"
  completed = sorted(
    int(path.parent.name.removeprefix("checkpoint_"))
    for path in videos.glob("checkpoint_*/diagnostics.json")
    if path.parent.name.removeprefix("checkpoint_").isdigit()
  )
  report["completed_video_iterations"] = completed
  report["health"] = "warning" if warnings else "ok"
  return report


def main(cfg: Config) -> None:
  cfg.run_root.mkdir(parents=True, exist_ok=True)
  while True:
    try:
      report = progress_report(cfg, time.time())
      try:
        result = subprocess.run(
          [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
          ],
          capture_output=True,
          text=True,
          timeout=10,
          check=True,
        )
        report["gpus"] = result.stdout.strip().splitlines()
      except (OSError, subprocess.SubprocessError) as error:
        report["gpus"] = []
        report["warnings"].append(f"GPU query failed: {error}")
        report["health"] = "warning"
      temporary = cfg.run_root / ".monitor.json.tmp"
      temporary.write_text(json.dumps(report, indent=2) + "\n")
      temporary.replace(cfg.run_root / "monitor.json")
      with (cfg.run_root / "monitor.jsonl").open("a") as stream:
        stream.write(json.dumps(report) + "\n")
      print(json.dumps(report), flush=True)
      if cfg.once or report["supervisor"] in ("complete", "stopped"):
        return
    except (OSError, ValueError) as error:
      print(json.dumps({"monitor_error": str(error)}), flush=True)
    time.sleep(cfg.poll_seconds)


def cli() -> None:
  main(tyro.cli(Config))


if __name__ == "__main__":
  cli()
