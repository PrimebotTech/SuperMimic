"""Bounded eight-rank 30000 warm-start comparison, with no saved checkpoints."""

import argparse
import json
import os
import shlex
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def environment(document: dict[str, Any]) -> dict[str, str]:
  result = os.environ | {k: str(v) for k, v in document.get("environment", {}).items()}
  result["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  result["CUDA_VISIBLE_DEVICES"] = document.get("gpu_ids", "0,1,2,3,4,5,6,7")
  for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    result.setdefault(name, "1")
  return result


def command(config: Path, mode: str) -> list[str]:
  return [
    sys.executable,
    "-m",
    "torch.distributed.run",
    "--standalone",
    "--nnodes",
    "1",
    "--nproc-per-node",
    "8",
    str(Path(__file__).with_name("q1_rank_runner.py")),
    "--config",
    str(config),
    "--mode",
    mode,
  ]


def summarize(
  directory: Path, warmup: int, elapsed: float, code: int
) -> dict[str, Any]:
  path = directory / "train" / "metrics.jsonl"
  metrics = (
    [json.loads(line) for line in path.read_text().splitlines()]
    if path.exists()
    else []
  )
  steady = metrics[warmup:]
  times = [row["update_seconds"] for row in steady]
  steps = [
    right["global_env_steps"] - left["global_env_steps"]
    for left, right in zip(metrics[warmup - 1 :], metrics[warmup:], strict=False)
  ]
  ranks: dict[str, Any] = {}
  for rank in range(8):
    log = directory / f"rank_{rank}_metrics.jsonl"
    rows = (
      [json.loads(line) for line in log.read_text().splitlines()]
      if log.exists()
      else []
    )
    stable = rows[warmup:]
    ranks[str(rank)] = {
      "observed_updates": len(rows),
      "all_finite": bool(rows) and all(row["finite"] for row in rows),
      "peak_torch_gib": max((row["torch_peak_gib"] for row in rows), default=None),
      "minimum_free_gib": min((row["cuda_free_gib"] for row in rows), default=None),
      "mean_cuda_compute_seconds": {
        phase: statistics.mean(row["cuda_compute_seconds"][phase] for row in stable)
        for phase in ("simulation", "model", "policy")
      }
      if stable
      else {},
    }
  mean_time = statistics.mean(times) if times else None
  return {
    "continuable": False,
    "exit_code": code,
    "process_wall_seconds_including_import_compile_warmup": elapsed,
    "startup_first_update_seconds": metrics[0]["update_seconds"] if metrics else None,
    "warmup_updates_excluded": warmup,
    "steady_updates": len(steady),
    "mean_update_seconds": mean_time,
    "median_update_seconds": statistics.median(times) if times else None,
    "mean_global_env_steps_per_update": statistics.mean(steps) if steps else None,
    "end_to_end_env_steps_per_second": statistics.mean(steps) / mean_time
    if steps and mean_time
    else None,
    "mean_phase_seconds": {
      phase: statistics.mean(row[f"{phase}_seconds"] for row in steady)
      for phase in ("collect", "model", "policy")
    }
    if steady
    else {},
    "all_rank_metrics_finite": all(rank["all_finite"] for rank in ranks.values()),
    "rank_reports": ranks,
    "measurement_scope": "profiling/instrumented complete updates; not bare simulation throughput",
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--dry-run", action="store_true")
  args = parser.parse_args()
  document = json.loads(args.config.read_text())
  source_iteration = int(document.get("source_iteration", 30000))
  updates = int(document.get("benchmark_updates", 80))
  warmup = int(document.get("benchmark_warmup_updates", 20))
  if not 1 <= warmup < updates <= 100:
    raise ValueError("Use 1 <= warmup < bounded benchmark updates <= 100")
  root = Path(document["benchmark_root"]).resolve()
  original = Path(document["original_run_root"]).resolve()
  if root == original or root.is_relative_to(original):
    raise ValueError("Benchmark root must be outside the original run")
  shared = document["train"].copy()
  shared.update(
    warm_start=document["checkpoint"],
    warm_start_progress=True,
    iterations=source_iteration + updates,
    replay_steps=1000,
    model_batches=10,
    critic_epochs=5,
    critic_minibatches=4,
    physics_timestep=0.005,
    compile_networks=True,
    compile_collection=True,
    cache_reference_heads=True,
    cache_goals=True,
    fuse_model_loss=True,
    capture_reset=True,
    checkpoint_replay=False,
    profile_every=1,
    profile_iterations=updates,
    save_interval=250000,
    initial_save_iteration=0,
    num_envs=4096,
    model_batch_size=2048,
    policy_batch_size=4096,
    clean_policy_actions=False,
    action_lag=True,
    normalize_policy_objective=False,
    normalize_model_objective=False,
    reward_arm_joint_weight=0.0,
    reward_arm_pd_target_weight=0.0,
    reward_body_orientation_std=None,
  )
  shared.pop("resume", None)
  variants = {
    "baseline": {"runtime_checks": "per_minibatch"},
    "fast": {"runtime_checks": "per_update"},
    "larger_env": {
      "runtime_checks": "per_update",
      "num_envs": 8192,
    },
    "zip_batch": {
      "runtime_checks": "per_update",
      "num_envs": 8192,
      "model_batch_size": 4096,
      "policy_batch_size": 4096,
    },
  }
  if document.get("include_large_case", False):
    variants["large_env_batch"] = {
      "runtime_checks": "per_update",
      "num_envs": 32768,
      "model_batch_size": 16384,
      "policy_batch_size": 16384,
    }
  if document.get("include_correctness_case", False):
    variants["new_semantics_correctness"] = {
      "runtime_checks": "per_update",
      "num_envs": 8192,
      "clean_policy_actions": True,
      "action_lag": False,
      "normalize_policy_objective": True,
      "normalize_model_objective": True,
      "reward_arm_joint_weight": 0.05,
    }
  variants = document.get("variants", variants)
  results = {}
  for label, override in variants.items():
    directory = root / label
    options = shared | override | {"output_dir": str(directory / "train")}
    for count in ("num_envs", "model_batch_size", "policy_batch_size"):
      if int(options[count]) <= 0 or int(options[count]) % 8:
        raise ValueError(f"{count} must be positive and divisible by eight")
    config = directory / "benchmark.json"
    payload = document | {"train": options, "source_iteration": source_iteration}
    invocation = command(config, "benchmark")
    print(
      json.dumps({"case": label, "train": options, "command": shlex.join(invocation)}),
      flush=True,
    )
    if args.dry_run:
      continue
    directory.mkdir(parents=True, exist_ok=False)
    config.write_text(json.dumps(payload, indent=2) + "\n")
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
        code = process.wait(
          timeout=float(document.get("benchmark_timeout_seconds", 1800))
        )
      except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
          process.wait(timeout=60)
        except subprocess.TimeoutExpired:
          os.killpg(process.pid, signal.SIGKILL)
          process.wait()
        code = 124
    result = summarize(directory, warmup, time.monotonic() - start, code)
    result["workload"] = {
      "global_num_envs": options["num_envs"],
      "global_model_batch_size": options["model_batch_size"],
      "global_policy_batch_size": options["policy_batch_size"],
      "model_optimizers_per_update": 10,
      "policy_optimizers_per_update": 1,
      "critic_optimizers_per_update": 20,
      "learner_updates_per_vector_control_step": 0.4,
      "model_window_samples_per_collected_env_step": 10
      * options["model_batch_size"]
      / (2.5 * options["num_envs"]),
      "policy_start_samples_per_collected_env_step": options["policy_batch_size"]
      / (2.5 * options["num_envs"]),
    }
    results[label] = result
    (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"case": label, "result": result}), flush=True)
    if code != 0 or not result["all_rank_metrics_finite"]:
      break
  if not args.dry_run:
    (root / "comparison.json").write_text(json.dumps(results, indent=2) + "\n")
    if any(
      result["exit_code"] or not result["all_rank_metrics_finite"]
      for result in results.values()
    ):
      raise SystemExit(1)


if __name__ == "__main__":
  main()
