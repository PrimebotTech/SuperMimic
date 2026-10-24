"""Durable eight-rank fresh or warm-start training with retained lineage."""

import argparse
import fcntl
import importlib
import json
import os
import pickle
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from benchmark_q1_eight_gpu import command, environment


def latest_checkpoint(root: Path) -> Path | None:
  paths = [p for p in root.glob("attempt_*/checkpoint_*.pt") if p.stem[11:].isdigit()]
  torch = importlib.import_module("torch")
  for path in sorted(paths, key=lambda p: int(p.stem[11:]), reverse=True):
    try:
      state = torch.load(path, map_location="cpu", weights_only=True)
      shards = path.parent / state["rank_directory"]
      if state["world_size"] == 8 and all(
        (shards / f"rank_{rank}.pt").is_file() for rank in range(8)
      ):
        return path
    except (
      OSError,
      ValueError,
      RuntimeError,
      KeyError,
      EOFError,
      pickle.UnpicklingError,
    ):
      continue
  return None


def write_status(root: Path, value: dict[str, Any]) -> None:
  temporary = root / ".supervisor.json.tmp"
  temporary.write_text(json.dumps(value, indent=2) + "\n")
  temporary.replace(root / "supervisor.json")
  with (root / "supervisor_events.jsonl").open("a") as stream:
    stream.write(json.dumps(value) + "\n")
  print(json.dumps(value), flush=True)


def attempt_options(
  initial: dict[str, Any], attempt: Path, checkpoint: Path | None
) -> dict[str, Any]:
  result = initial | {"output_dir": str(attempt)}
  if checkpoint is not None:
    result.pop("warm_start", None)
    result.pop("warm_start_progress", None)
    result["resume"] = str(checkpoint)
  return result


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--dry-run", action="store_true")
  parser.add_argument("--max-restarts", type=int, default=3)
  args = parser.parse_args()
  document = json.loads(args.config.read_text())
  root = Path(document["run_root"]).resolve()
  original = Path(document["original_run_root"]).resolve()
  if root == original or root.is_relative_to(original):
    raise ValueError("Filtered training must use a fresh root outside original run")
  options = document["train"].copy()
  options["iterations"] = 250000
  source = document.get("checkpoint")
  if source:
    options.update(warm_start=source, warm_start_progress=True)
  else:
    options.pop("warm_start", None)
    options.pop("warm_start_progress", None)
  options.pop("resume", None)
  if args.dry_run:
    print(
      json.dumps(
        {
          "first_attempt": options,
          "recovery": "remove warm_start/progress; use resume; preserve lineage",
        },
        indent=2,
      )
    )
    return
  root.mkdir(parents=True, exist_ok=True)
  lineage = root / "lineage.json"
  stopping = False
  process: subprocess.Popen[Any] | None = None

  def request_stop(signum: int, frame: Any) -> None:
    nonlocal stopping
    stopping = True
    if process is not None and process.poll() is None:
      os.killpg(process.pid, signal.SIGTERM)

  signal.signal(signal.SIGINT, request_stop)
  signal.signal(signal.SIGTERM, request_stop)
  with (root / "supervisor.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if lineage.exists():
      if json.loads(lineage.read_text())["initial_train"] != options:
        raise ValueError("Supervisor configuration differs from retained lineage")
    else:
      lineage.write_text(
        json.dumps(
          {
            "source_checkpoint": source,
            "initialization": "weights-only" if source else "random from iteration 0",
            "initial_train": options,
            "created": time.time(),
          },
          indent=2,
        )
        + "\n"
      )
    for restart in range(args.max_restarts + 1):
      if stopping:
        write_status(root, {"state": "stopped", "ended": time.time()})
        return
      latest = latest_checkpoint(root)
      if latest is not None and int(latest.stem[11:]) >= 250000:
        write_status(root, {"state": "complete", "checkpoint": str(latest)})
        return
      if restart and latest is None:
        raise RuntimeError("Crash produced no committed checkpoint; recovery stopped")
      number = (
        max(
          (int(p.name[8:]) for p in root.glob("attempt_*") if p.name[8:].isdigit()),
          default=-1,
        )
        + 1
      )
      attempt = root / f"attempt_{number:02d}"
      train = attempt_options(options, attempt, latest)
      config = root / f"attempt_{number:02d}.json"
      config.write_text(json.dumps(document | {"train": train}, indent=2) + "\n")
      status = {
        "state": "running",
        "attempt": str(attempt),
        "resume": str(latest) if latest else None,
        "source_checkpoint": source,
        "started": time.time(),
      }
      write_status(root, status)
      with (root / f"attempt_{number:02d}.stdout.log").open("w") as log:
        process = subprocess.Popen(
          command(config, "durable"),
          cwd=document["project_dir"],
          env=environment(document),
          stdout=log,
          stderr=subprocess.STDOUT,
          start_new_session=True,
        )
        code = process.wait()
      complete = (attempt / "checkpoint_250000.pt").is_file()
      status.update(
        state="complete"
        if complete
        else "stopped"
        if stopping or code == 0
        else "crashed",
        exit_code=code,
        ended=time.time(),
      )
      write_status(root, status)
      if complete or stopping or code == 0:
        return
      if restart < args.max_restarts:
        time.sleep(10)
    raise RuntimeError("Training exhausted the bounded recovery attempts")


if __name__ == "__main__":
  main()
