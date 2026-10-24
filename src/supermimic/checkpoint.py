"""Atomic learner snapshots with one optional replay shard per rank."""

import os
import random
import signal
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch

from supermimic.distributed import Distributed


def atomic_save(value: Any, path: Path) -> None:
  """Publish only a flushed, complete file; keep the previous file on failure."""
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
  try:
    with temporary.open("xb") as stream:
      torch.save(value, stream)
      stream.flush()
      os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
      os.fsync(directory)
    finally:
      os.close(directory)
  finally:
    temporary.unlink(missing_ok=True)


def capture_rng(device: str) -> dict[str, Any]:
  """Represent NumPy's RNG using tensors accepted by weights_only loading."""
  name, keys, position, gaussian, cached = np.random.get_state()
  result = {
    "python": random.getstate(),
    "numpy": (name, torch.tensor(keys.astype(np.int64)), position, gaussian, cached),
    "torch": torch.get_rng_state(),
  }
  if device.startswith("cuda"):
    result["cuda"] = torch.cuda.get_rng_state(device).cpu()
  return result


def restore_rng(state: dict[str, Any], device: str) -> None:
  # New shards need independent streams after a change in process count.
  # The saved parent stream remains in the checkpoint for provenance.
  if "reshard_seed" in state:
    seed = state["reshard_seed"]
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
    if device.startswith("cuda"):
      with torch.cuda.device(device):
        torch.cuda.manual_seed(seed)
    return
  random.setstate(state["python"])
  name, keys, position, gaussian, cached = state["numpy"]
  np.random.set_state(
    (name, keys.numpy().astype(np.uint32), position, gaussian, cached)
  )
  torch.set_rng_state(state["torch"])
  if device.startswith("cuda") and "cuda" in state:
    torch.cuda.set_rng_state(state["cuda"], device)


def save_checkpoint(
  path: Path,
  learner: dict[str, Any],
  rank_state: dict[str, Any],
  group: Distributed,
) -> Path:
  """Commit rank shards first; the main PT file is the transaction manifest."""
  snapshot = group.objects(uuid.uuid4().hex if group.main else None)[0]
  directory = path.with_name(f"{path.stem}.{snapshot}.ranks")
  with group.checked("checkpoint rank shard"):
    atomic_save(
      rank_state | {"snapshot": snapshot, "rank": group.rank},
      directory / f"rank_{group.rank}.pt",
    )
  with group.checked("checkpoint publication"):
    if group.main:
      atomic_save(
        learner
        | {
          "checkpoint_version": 1,
          "world_size": group.world_size,
          "snapshot": snapshot,
          "rank_directory": directory.name,
        },
        path,
      )
  return path


def load_checkpoint(path: Path, group: Distributed) -> tuple[dict, dict | None]:
  """Load on CPU before GPU allocation, rejecting unpublished/mismatched shards."""
  learner = torch.load(path, map_location="cpu", weights_only=True)
  if not isinstance(learner, dict) or "actor" not in learner:
    raise ValueError(f"{path}: not a SuperMimic learner checkpoint")
  if "checkpoint_version" not in learner:
    if group.world_size != 1:
      raise ValueError("Legacy snapshots have no rank state; resume with one rank")
    return learner, None
  if learner["checkpoint_version"] != 1:
    raise ValueError("Unsupported checkpoint version")
  if learner["world_size"] != group.world_size:
    raise ValueError("Resume with the checkpoint's world_size to retain rank state")
  directory = learner["rank_directory"]
  if (
    not isinstance(directory, str)
    or directory in ("", ".", "..")
    or Path(directory).name != directory
  ):
    raise ValueError("Invalid checkpoint shard directory")
  rank_state = torch.load(
    path.parent / directory / f"rank_{group.rank}.pt",
    map_location="cpu",
    weights_only=True,
  )
  if not isinstance(rank_state, dict) or (
    rank_state.get("snapshot") != learner["snapshot"]
    or rank_state.get("rank") != group.rank
  ):
    raise ValueError("Replay/RNG shard belongs to another checkpoint or rank")
  return learner, rank_state


class StopRequest:
  """Signals request a checkpoint at a safe optimizer/collective boundary."""

  def __init__(self) -> None:
    self.requested = False
    self.previous: dict[signal.Signals, Any] = {}

  def _request(self, signum: int, frame: Any) -> None:
    self.requested = True

  def __enter__(self) -> "StopRequest":
    for signum in (signal.SIGINT, signal.SIGTERM):
      self.previous[signum] = signal.signal(signum, self._request)
    return self

  def __exit__(self, *args: Any) -> None:
    for signum, previous in self.previous.items():
      signal.signal(signum, previous)
