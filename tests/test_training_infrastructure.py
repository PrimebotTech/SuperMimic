"""Regression checks for launch, resumable state and shared curriculum."""

import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from supermimic.checkpoint import (
  atomic_save,
  capture_rng,
  load_checkpoint,
  restore_rng,
  save_checkpoint,
)
from supermimic.data import ReturnResetSampler
from supermimic.distributed import Distributed
from supermimic.replay import TemporalReplay
from supermimic.train import TrainCfg, rank_configuration


def test_auto_device_and_global_counts(tmp_path: Path):
  cfg = TrainCfg(
    tmp_path / "q1.xml",
    tmp_path,
    tmp_path / "run",
    num_envs=4096,
    model_batch_size=2048,
    policy_batch_size=4096,
  )
  for rank in range(8):
    resolved = rank_configuration(cfg, Distributed(rank, 8, rank), 8)
    assert resolved.device == f"cuda:{rank}"
    assert resolved.num_envs == 512
    assert resolved.model_batch_size == 256
    assert resolved.policy_batch_size == 512
  assert rank_configuration(cfg, Distributed.detached(), 0).device == "cpu"
  assert (
    rank_configuration(replace(cfg, device="cpu"), Distributed(2, 8, 2), 8).device
    == "cpu"
  )
  with pytest.raises(ValueError, match="visible GPUs"):
    rank_configuration(cfg, Distributed(2, 8, 2), 2)
  with pytest.raises(ValueError, match="does not divide"):
    rank_configuration(replace(cfg, num_envs=4097), Distributed(0, 8, 0), 8)


def test_reset_sampling_does_not_mutate_shared_history():
  first = ReturnResetSampler(torch.tensor([100, 160]), 0.02)
  second = ReturnResetSampler(torch.tensor([100, 160]), 0.02)
  for _ in range(11):
    first.sample(3)
  second.sample(1)
  torch.testing.assert_close(first.past_weights, second.past_weights)
  for sampler in (first, second):
    sampler.update(torch.tensor([0, 1, 1]), torch.tensor([0.1, 0.4, 0.8]))
    sampler.advance_past()
  torch.testing.assert_close(first.past_weights, second.past_weights)
  second.load_state_dict(first.state_dict())
  for name in ("counts", "normalized", "past_weights"):
    torch.testing.assert_close(getattr(first, name), getattr(second, name))


def test_atomic_checkpoint_failure_preserves_previous_file(tmp_path: Path, monkeypatch):
  path = tmp_path / "checkpoint.pt"
  atomic_save({"value": 1}, path)

  def fail_save(value, stream):
    stream.write(b"incomplete")
    raise OSError("simulated interrupted write")

  monkeypatch.setattr(torch, "save", fail_save)
  with pytest.raises(OSError, match="interrupted"):
    atomic_save({"value": 2}, path)
  assert torch.load(path, weights_only=True) == {"value": 1}
  assert list(tmp_path.iterdir()) == [path]


def test_rank_snapshot_and_rng_roundtrip(tmp_path: Path):
  rng = capture_rng("cpu")
  expected = random.random(), np.random.rand(), torch.rand(3)
  path = tmp_path / "checkpoint.pt"
  group = Distributed.detached()
  save_checkpoint(path, {"actor": {"weight": torch.ones(2)}}, {"rng": rng}, group)
  learner, rank = load_checkpoint(path, group)
  assert rank is not None
  assert learner["actor"]["weight"].device.type == "cpu"
  restore_rng(rank["rng"], "cpu")
  assert random.random() == expected[0]
  assert np.random.rand() == expected[1]
  torch.testing.assert_close(torch.rand(3), expected[2])
  with pytest.raises(ValueError, match="world_size"):
    load_checkpoint(path, Distributed(0, 2, 0))
  shard = path.parent / learner["rank_directory"] / "rank_0.pt"
  atomic_save(rank | {"snapshot": "wrong"}, shard)
  with pytest.raises(ValueError, match="another checkpoint"):
    load_checkpoint(path, group)


def add_replay_step(replay: TemporalReplay, step: int) -> None:
  replay.add(
    torch.full((1, 79), float(step)),
    torch.full((1, 79), float(step + 1)),
    torch.zeros(1, 22),
    torch.zeros(1, 26),
    torch.full((1, 72), float(step)),
    torch.zeros(1),
    torch.zeros(1, dtype=torch.bool),
    torch.zeros(1, dtype=torch.long),
    torch.tensor([step]),
  )


def test_replay_resume_preserves_wrapped_rows_and_breaks_old_new_episode():
  old = TemporalReplay(8, 1, 79, 26, 72, "cpu")
  for step in range(11):
    add_replay_step(old, step)
  saved = old.state_dict()
  torch.testing.assert_close(saved["states"][:, 0, 0], torch.arange(3, 11).float())
  restored = TemporalReplay(8, 1, 79, 26, 72, "cpu")
  restored.load_state_dict(saved)
  add_replay_step(restored, 100)
  add_replay_step(restored, 101)
  batch = restored.sample(512, window=3, history=1)
  assert (batch.states[:, 1:, 0] - batch.states[:, :-1, 0] == 1).all()
  assert batch.states.max() <= 10
  with pytest.raises(ValueError, match="count"):
    restored.load_state_dict(saved | {"count": 9})
  with pytest.raises(ValueError, match="tensor"):
    restored.load_state_dict(saved | {"states": torch.zeros(1)})


def test_optimizer_state_resumes_equivalent_next_update(tmp_path: Path):
  """Synthetic unit exercise of Adam state, without a simulator/training run."""
  model = torch.nn.Linear(2, 1)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  model(torch.ones(2, 2)).square().mean().backward()
  optimizer.step()
  path = tmp_path / "learner.pt"
  atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
  loaded = torch.load(path, map_location="cpu", weights_only=True)
  other = torch.nn.Linear(2, 1)
  other.load_state_dict(loaded["model"])
  resumed = torch.optim.Adam(other.parameters(), lr=0.5)
  resumed.load_state_dict(loaded["optimizer"])
  for network, opt in ((model, optimizer), (other, resumed)):
    opt.zero_grad()
    network(torch.tensor([[2.0, 3.0]])).square().mean().backward()
    opt.step()
  for first, second in zip(model.parameters(), other.parameters(), strict=True):
    torch.testing.assert_close(first, second, rtol=0, atol=0)
