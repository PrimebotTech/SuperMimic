"""Rank entrypoint; benchmark instrumentation never changes deployed files."""

import argparse
import importlib
import json
import math
import os
import socket
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest import mock


def cpu_ids(value: str) -> set[int]:
  result: set[int] = set()
  for segment in value.split(","):
    bounds = segment.split("-")
    if len(bounds) == 1:
      result.add(int(bounds[0]))
    elif len(bounds) == 2:
      result.update(range(int(bounds[0]), int(bounds[1]) + 1))
    else:
      raise ValueError(f"Invalid CPU range: {segment}")
  if not result:
    raise ValueError("CPU affinity cannot be empty")
  return result


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--mode", choices=("benchmark", "durable"), required=True)
  args = parser.parse_args()
  document = json.loads(args.config.read_text())
  rank = int(os.environ.get("RANK", "0"))
  affinity = document.get("cpu_affinity", {}).get(str(rank))
  if affinity:
    os.sched_setaffinity(0, cpu_ids(affinity))
  # CPU binding precedes Torch/Warp imports and their host worker creation.
  torch = importlib.import_module("torch")
  training = importlib.import_module("supermimic.train")
  options = document["train"].copy()
  for name in (
    "robot_xml",
    "motion_dir",
    "output_dir",
    "resume",
    "warm_start",
    "motion_cache_dir",
  ):
    if options.get(name) is not None:
      options[name] = Path(options[name])
  cfg = training.TrainCfg(**options)
  if args.mode == "durable":
    training.train(cfg)
    return
  report_dir = args.config.parent
  mapping = {
    "rank": rank,
    "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
    "hostname": socket.gethostname(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "cpu_affinity": sorted(os.sched_getaffinity(0)),
    "continuable": False,
    "checkpoint_serialization": "disabled only in this benchmark process",
  }
  (report_dir / f"rank_{rank}_mapping.json").write_text(
    json.dumps(mapping, indent=2) + "\n"
  )
  phases: dict[str, list[Any]] = {"simulation": [], "model": [], "policy": []}
  local_metrics: dict[str, Any] = {}
  previous = time.monotonic()
  update = 0

  def timed(name: str, function: Any) -> Any:
    def wrapped(instance: Any, *positional: Any, **keywords: Any) -> Any:
      start = torch.cuda.Event(enable_timing=True)
      end = torch.cuda.Event(enable_timing=True)
      start.record()
      result = function(instance, *positional, **keywords)
      end.record()
      phases[name].append((start, end))
      if name in ("model", "policy"):
        local_metrics.update(result)
      return result

    return wrapped

  original_mean = training.Distributed.mean

  def recorded_mean(group: Any, values: list[float]) -> list[float]:
    nonlocal previous, update
    result = original_mean(group, values)
    torch.cuda.synchronize(group.device)
    if update == 0:
      properties = torch.cuda.get_device_properties(group.device)
      mapping.update(
        device=group.device,
        device_name=properties.name,
        device_uuid=str(getattr(properties, "uuid", "unavailable")),
      )
      (report_dir / f"rank_{rank}_mapping.json").write_text(
        json.dumps(mapping, indent=2) + "\n"
      )
    now = time.monotonic()
    names = sorted(set(local_metrics) | {"collection_reward"})
    metrics = dict(zip(names, values, strict=True)) if len(names) == len(values) else {}
    record = {
      "update_index": update,
      "rank": rank,
      "startup": update == 0,
      "local_update_wall_seconds": now - previous,
      "num_envs_per_rank": cfg.num_envs // group.world_size,
      "expected_control_steps": 2 + (document["source_iteration"] + update) % 2,
      "metrics": metrics,
      "finite": all(math.isfinite(value) for value in values),
      "cuda_compute_seconds": {
        name: sum(start.elapsed_time(end) / 1000 for start, end in events)
        for name, events in phases.items()
      },
      "torch_allocated_gib": torch.cuda.memory_allocated(group.device) / 1024**3,
      "torch_reserved_gib": torch.cuda.memory_reserved(group.device) / 1024**3,
      "torch_peak_gib": torch.cuda.max_memory_allocated(group.device) / 1024**3,
      "cuda_free_gib": torch.cuda.mem_get_info(group.device)[0] / 1024**3,
    }
    with (report_dir / f"rank_{rank}_metrics.jsonl").open("a") as stream:
      stream.write(json.dumps(record) + "\n")
    for events in phases.values():
      events.clear()
    previous = time.monotonic()
    update += 1
    return result

  def suppressed_checkpoint(path: Path, *unused: Any, **keywords: Any) -> Path:
    with (report_dir / f"rank_{rank}_omitted_checkpoints.jsonl").open("a") as stream:
      stream.write(json.dumps({"path": str(path), "continuable": False}) + "\n")
    return path

  def omitted_replay(instance: Any) -> dict[str, Any]:
    return {"benchmark_replay_omitted": True}

  with ExitStack() as patches:
    patches.enter_context(
      mock.patch.object(training, "save_checkpoint", suppressed_checkpoint)
    )
    patches.enter_context(
      mock.patch.object(training.TemporalReplay, "state_dict", omitted_replay)
    )
    patches.enter_context(
      mock.patch.object(training.Distributed, "mean", recorded_mean)
    )
    for owner, method, name in (
      (training.Q1Simulator, "step", "simulation"),
      (training.SuperMimic, "fit_model_tensors", "model"),
      (training.SuperMimic, "improve_policy", "policy"),
    ):
      patches.enter_context(
        mock.patch.object(owner, method, timed(name, getattr(owner, method)))
      )
    training.train(cfg)


if __name__ == "__main__":
  main()
