"""Compare execution options from identical random initialization on eight GPUs."""

import argparse
import json
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path

from benchmark_q1_eight_gpu import command, environment, summarize


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--dry-run", action="store_true")
  args = parser.parse_args()
  document = json.loads(args.config.read_text())
  updates = int(document.get("benchmark_updates", 120))
  warmup = int(document.get("benchmark_warmup_updates", 20))
  if not 1 <= warmup < updates <= 300:
    raise ValueError("Require 1 <= warmup < updates <= 300")
  root = Path(document["benchmark_root"])
  if not args.dry_run:
    root.mkdir(parents=True, exist_ok=False)
  variants = document.get(
    "variants",
    {
      "baseline": {},
      "critic_fused": {"compile_critic": True, "fused_optimizers": True},
      "imagination_graph": {"capture_imagination": True},
      "combined": {
        "capture_imagination": True,
        "compile_critic": True,
        "fused_optimizers": True,
      },
    },
  )
  results = {}
  for label, overrides in variants.items():
    directory = root / label
    options = (
      document["train"]
      | {
        "iterations": updates,
        "profile_iterations": updates,
        "profile_every": 1,
        "save_interval": 250000,
        "initial_save_iteration": 0,
        "checkpoint_replay": False,
        "resume": None,
        "warm_start": None,
        "warm_start_progress": False,
        "capture_imagination": False,
        "compile_critic": False,
        "fused_optimizers": False,
        "output_dir": str(directory / "train"),
      }
      | overrides
    )
    config = directory / "benchmark.json"
    payload = document | {"train": options, "source_iteration": 0}
    invocation = command(config, "benchmark")
    if args.dry_run:
      print(
        json.dumps(
          {"case": label, "train": options, "command": shlex.join(invocation)}
        ),
        flush=True,
      )
      continue
    directory.mkdir()
    config.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"case": label, "state": "starting"}), flush=True)
    start = time.monotonic()
    with (directory / "stdout.log").open("w") as log:
      process = subprocess.Popen(
        invocation,
        cwd=document["project_dir"],
        env=environment(document),
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
      )
      try:
        code = process.wait(timeout=document.get("benchmark_timeout_seconds", 2400))
      except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
          process.wait(timeout=45)
        except subprocess.TimeoutExpired:
          os.killpg(process.pid, signal.SIGKILL)
          process.wait()
        code = 124
    report = summarize(directory, warmup, time.monotonic() - start, code)
    report["initialization"] = "all networks and training state initialized at zero"
    report["execution_options"] = overrides
    results[label] = report
    (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    temporary = root / ".results.tmp"
    temporary.write_text(json.dumps(results, indent=2) + "\n")
    temporary.replace(root / "results.json")
    print(json.dumps({"case": label, "summary": report}), flush=True)


if __name__ == "__main__":
  main()
