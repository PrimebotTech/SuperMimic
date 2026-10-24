"""Expand a complete checkpoint by partitioning its environment replay tensors."""

import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import tyro

from supermimic.checkpoint import atomic_save
from supermimic.replay import TemporalReplay


@dataclass
class Config:
  source: Path
  output: Path
  world_size: int = 8


def same_state(first: Any, second: Any) -> bool:
  if isinstance(first, torch.Tensor):
    return isinstance(second, torch.Tensor) and torch.equal(first, second)
  if isinstance(first, dict):
    return (
      isinstance(second, dict)
      and first.keys() == second.keys()
      and all(same_state(value, second[key]) for key, value in first.items())
    )
  return first == second


def partition_rank(
  state: dict[str, Any], factor: int, child: int, seed: int
) -> dict[str, Any]:
  """Split along environment axes; preserve every chronological transition."""
  replay = state.get("replay")
  if replay is None:
    raise ValueError("Resharding requires a complete replay checkpoint")
  if state["pending_bins"] or state["pending_progress"]:
    raise ValueError("Flush curriculum updates before changing the rank count")
  if replay["num_envs"] % factor or not 0 <= child < factor:
    raise ValueError("Environment count must divide evenly across child ranks")
  if state["local_env_steps"] % factor:
    raise ValueError("Environment step counts must divide across child ranks")
  count = replay["num_envs"] // factor
  start, end = child * count, (child + 1) * count
  partition = {
    "capacity": replay["capacity"],
    "count": replay["count"],
    "num_envs": count,
  }
  for name in TemporalReplay._storage_names():
    tensor = replay[name]
    if tensor.shape[:2] != (replay["count"], replay["num_envs"]):
      raise ValueError(f"Invalid replay tensor: {name}")
    # A cloned slice prevents torch.save from writing the parent's full storage.
    partition[name] = tensor[:, start:end].clone()
  rng = dict(state["rng"])
  if child:
    rng["reshard_seed"] = seed
  return state | {
    "replay": partition,
    "rng": rng,
    "local_env_steps": state["local_env_steps"] // factor,
    "resharding": {"parent_rank": state["rank"], "environment_range": [start, end]},
  }


def reshard_checkpoint(cfg: Config) -> Path:
  if cfg.output.exists():
    raise FileExistsError("Use a fresh checkpoint path for resharding")
  learner = torch.load(cfg.source, map_location="cpu", weights_only=True)
  if learner.get("checkpoint_version") != 1:
    raise ValueError("Expected a version 1 checkpoint with rank shards")
  previous_world = learner["world_size"]
  if cfg.world_size <= previous_world or cfg.world_size % previous_world:
    raise ValueError("New world size must be a larger multiple of the saved size")
  factor = cfg.world_size // previous_world
  identity = dict(learner["identity"])
  for name in (
    "num_envs_per_rank",
    "model_batch_size_per_rank",
    "policy_batch_size_per_rank",
  ):
    if identity[name] % factor:
      raise ValueError(f"Rank count does not divide {name}")
    identity[name] //= factor
  identity["world_size"] = cfg.world_size
  directory = learner["rank_directory"]
  if (
    not isinstance(directory, str)
    or directory in ("", ".", "..")
    or Path(directory).name != directory
  ):
    raise ValueError("Invalid original checkpoint shard directory")
  source_digest = hashlib.sha256(cfg.source.read_bytes()).hexdigest()
  snapshot = uuid.uuid4().hex
  destination = cfg.output.with_name(f"{cfg.output.stem}.{snapshot}.ranks")
  curriculum = None
  total_steps = 0
  manifest = []
  for parent in range(previous_world):
    state = torch.load(
      cfg.source.parent / directory / f"rank_{parent}.pt",
      map_location="cpu",
      weights_only=True,
    )
    if state["snapshot"] != learner["snapshot"] or state["rank"] != parent:
      raise ValueError("Original rank shard belongs to another checkpoint")
    if curriculum is None:
      curriculum = state["curriculum"]
    elif not same_state(curriculum, state["curriculum"]):
      raise ValueError("Original ranks disagree about shared curriculum state")
    total_steps += state["local_env_steps"]
    for child in range(factor):
      rank = parent * factor + child
      seed = (
        int.from_bytes(
          hashlib.sha256(f"{source_digest}:{rank}".encode()).digest()[:8], "little"
        )
        % 2**63
      )
      partition = partition_rank(state, factor, child, seed)
      partition.update(snapshot=snapshot, rank=rank)
      atomic_save(partition, destination / f"rank_{rank}.pt")
      manifest.append(
        {"rank": rank, "rng_seed": seed if child else None} | partition["resharding"]
      )
      del partition
    del state
  if total_steps != learner["global_env_steps"]:
    raise ValueError("Rank environment steps disagree with learner checkpoint")
  lineage = {
    "source": str(cfg.source.resolve()),
    "source_sha256": source_digest,
    "previous_world_size": previous_world,
    "world_size": cfg.world_size,
    "iteration": learner["iteration"],
    "global_env_steps": total_steps,
    "preserved": "networks,optimizers,iteration,curriculum,all replay transitions",
    "rng": "first child retains parent stream; other children use recorded seeds",
    "ranks": manifest,
  }
  atomic_save(
    learner
    | {
      "identity": identity,
      "world_size": cfg.world_size,
      "snapshot": snapshot,
      "rank_directory": destination.name,
      "resharding": lineage,
    },
    cfg.output,
  )
  cfg.output.with_suffix(".resharding.json").write_text(
    json.dumps(lineage, indent=2) + "\n"
  )
  print(json.dumps(lineage), flush=True)
  return cfg.output


def main() -> None:
  reshard_checkpoint(tyro.cli(Config))


if __name__ == "__main__":
  main()
