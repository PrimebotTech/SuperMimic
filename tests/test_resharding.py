"""An expanded checkpoint retains learner state and every environment history."""

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from supermimic.checkpoint import atomic_save, capture_rng, load_checkpoint, restore_rng
from supermimic.distributed import Distributed
from supermimic.replay import TemporalReplay
from supermimic.reshard import Config, partition_rank, reshard_checkpoint, same_state


@pytest.fixture
def original_checkpoint(tmp_path: Path) -> Path:
  path = tmp_path / "old" / "checkpoint_24731.pt"
  states = {}
  for parent in range(2):
    replay = TemporalReplay(12, 8, 3, 2, 4, "cpu")
    for step in range(16):
      replay.add(
        torch.arange(8).float()[:, None].expand(8, 3) + parent * 100 + step * 1000,
        torch.full((8, 3), float(step + 1)),
        torch.zeros(8, 22),
        torch.zeros(8, 2),
        torch.zeros(8, 4),
        torch.zeros(8),
        torch.full((8,), step == 6, dtype=torch.bool),
        torch.arange(8),
        torch.full((8,), step, dtype=torch.long),
      )
    states[parent] = {
      "snapshot": "original",
      "rank": parent,
      "rng": capture_rng("cpu"),
      "curriculum": {"counts": torch.arange(3).float()},
      "pending_bins": [],
      "pending_progress": [],
      "local_env_steps": 128,
      "replay": replay.state_dict(),
    }
    atomic_save(states[parent], path.parent / "ranks" / f"rank_{parent}.pt")
  learner = {
    "checkpoint_version": 1,
    "world_size": 2,
    "snapshot": "original",
    "rank_directory": "ranks",
    "identity": {
      "world_size": 2,
      "num_envs_per_rank": 8,
      "model_batch_size_per_rank": 16,
      "policy_batch_size_per_rank": 32,
    },
    "iteration": 24731,
    "global_env_steps": 256,
    "actor": {"weights": torch.arange(3)},
    "model_optimizer": {"state": {0: {"step": torch.tensor(731)}}},
  }
  atomic_save(learner, path)
  return path


def test_resharding_preserves_learner_and_complete_replay(original_checkpoint):
  source = original_checkpoint
  output = source.parent.parent / "new" / source.name
  reshard_checkpoint(Config(source, output, 8))
  original = torch.load(source, weights_only=True)
  loaded = []
  for rank in range(8):
    learner, state = load_checkpoint(output, Distributed(rank, 8, rank))
    assert state is not None
    assert same_state(learner["actor"], original["actor"])
    assert same_state(learner["model_optimizer"], original["model_optimizer"])
    assert learner["iteration"] == 24731
    assert learner["identity"]["num_envs_per_rank"] == 2
    assert state["local_env_steps"] == 32
    restored = TemporalReplay(12, 2, 3, 2, 4, "cpu")
    restored.load_state_dict(state["replay"])
    assert restored.boundaries[-1].all()
    loaded.append(state)
  for parent in range(2):
    old = torch.load(source.parent / "ranks" / f"rank_{parent}.pt", weights_only=True)
    for name in TemporalReplay._storage_names():
      joined = torch.cat(
        [s["replay"][name] for s in loaded[parent * 4 : parent * 4 + 4]], 1
      )
      assert torch.equal(joined, old["replay"][name])
  assert sum(s["local_env_steps"] for s in loaded) == original["global_env_steps"]
  with pytest.raises(FileExistsError):
    reshard_checkpoint(Config(source, output, 8))


def test_resharded_rng_streams_are_repeatable_and_independent():
  original = capture_rng("cpu")
  draws = []
  for seed in (101, 202, 101):
    restore_rng(original | {"reshard_seed": seed}, "cpu")
    draws.append((random.random(), float(np.random.rand()), torch.rand(4)))
  assert draws[0][:2] == draws[2][:2] and draws[0][:2] != draws[1][:2]
  assert torch.equal(draws[0][2], draws[2][2])
  assert not torch.equal(draws[0][2], draws[1][2])


def test_resharding_rejects_unflushed_curriculum(original_checkpoint):
  old = torch.load(
    original_checkpoint.parent / "ranks" / "rank_0.pt", weights_only=True
  )
  with pytest.raises(ValueError, match="Flush"):
    partition_rank(old | {"pending_bins": [1]}, 4, 0, 101)
