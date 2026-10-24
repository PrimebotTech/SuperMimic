"""Collectives over real processes: the parts a single-process test cannot reach."""

import os
import socket
import tempfile
from pathlib import Path

import pytest
import torch
from torch.multiprocessing.spawn import spawn

from supermimic.checkpoint import load_checkpoint, save_checkpoint
from supermimic.data import ReturnResetSampler
from supermimic.distributed import Distributed


def _layer() -> torch.nn.Linear:
  torch.manual_seed(0)
  return torch.nn.Linear(4, 3)


def _average_worker(rank: int, world_size: int, directory: str) -> None:
  group = Distributed(rank, world_size, rank)
  group.initialize("cpu")
  try:
    layer = _layer()
    # Each rank starts from a different initialization, so a missing broadcast
    # shows up as diverged weights rather than as luck.
    with torch.no_grad():
      layer.weight.add_(rank)
    group.broadcast_parameters([layer])
    # Each rank sees its own slice of the batch, as in training.
    inputs = torch.full((2, 4), float(rank + 1))
    layer(inputs).square().sum().backward()
    group.average_gradients(layer.parameters(), bucket_bytes=16)
    assert layer.weight.grad is not None
    torch.save(
      {
        "weight": layer.weight.detach().clone(),
        "grad": layer.weight.grad.clone(),
        "sum": group.sum_(torch.tensor([float(rank + 1)])).clone(),
        "mean": group.mean([float(rank + 1), 10.0 * (rank + 1)]),
      },
      Path(directory) / f"rank_{rank}.pt",
    )
  finally:
    group.shutdown()


def test_broadcast_and_gradient_average_agree_across_real_processes():
  """Averaged gradients must equal the gradient of the whole batch."""
  world_size = 3
  os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
  os.environ["MASTER_PORT"] = "29517"
  with tempfile.TemporaryDirectory() as directory:
    spawn(
      _average_worker,
      (world_size, directory),
      world_size,
      join=True,
      start_method="spawn",
    )
    ranks = [
      torch.load(Path(directory) / f"rank_{r}.pt", weights_only=True)
      for r in range(world_size)
    ]
  # Rank 0's initialization wins, so every rank trains the same network.
  for other in ranks[1:]:
    torch.testing.assert_close(other["weight"], ranks[0]["weight"])
  torch.testing.assert_close(ranks[0]["weight"], _layer().weight)
  # The averaged gradient is the mean of the per-rank gradients, which for this
  # loss is the gradient of the concatenated batch.
  expected = []
  for rank in range(world_size):
    layer = _layer()
    layer(torch.full((2, 4), float(rank + 1))).square().sum().backward()
    assert layer.weight.grad is not None
    expected.append(layer.weight.grad)
  mean = torch.stack(expected).mean(0)
  for rank in ranks:
    torch.testing.assert_close(rank["grad"], mean)
  # Bookkeeping reductions: a sum for the curriculum, a mean for the log.
  for rank in ranks:
    torch.testing.assert_close(rank["sum"], torch.tensor([6.0]))
    assert rank["mean"] == pytest.approx([2.0, 20.0])


def _infrastructure_worker(rank: int, world_size: int, directory: str) -> None:
  group = Distributed(rank, world_size, rank)
  group.initialize("cpu", timeout_seconds=30)
  try:
    with pytest.raises(RuntimeError, match="rank 1: test readiness"):
      group.require(rank != 1, "test readiness")
    with pytest.raises(RuntimeError, match="simulated disk failure"):
      with group.checked("write"):
        if rank == 1:
          raise OSError("simulated disk failure")
    layer = _layer()
    if rank == 0:
      layer(torch.ones(2, 4)).sum().backward()
    with pytest.raises(RuntimeError, match="received gradients"):
      group.average_gradients(layer.parameters())
    with torch.no_grad():
      layer.weight.add_(rank)
    with pytest.raises(RuntimeError, match="differ across ranks"):
      group.assert_parameters_synced([layer])
    group.broadcast_parameters([layer])
    group.assert_parameters_synced([layer])
    sampler = ReturnResetSampler(torch.tensor([100, 200]), 0.02)
    for _ in range(rank * 7 + 1):
      sampler.sample(3)
    sampler.update(torch.tensor([rank]), torch.tensor([rank * 0.5]), group.sum_)
    sampler.advance_past()
    path = Path(directory) / "checkpoint.pt"
    save_checkpoint(
      path, {"actor": layer.state_dict()}, {"curriculum": sampler.state_dict()}, group
    )
    before = path.read_bytes()
    # One rank cannot serialize its shard: the last committed learner survives.
    invalid = {"bad": lambda: None} if rank == 1 else {}
    with pytest.raises(RuntimeError, match="checkpoint rank shard"):
      save_checkpoint(path, {"actor": {}}, invalid, group)
    assert path.read_bytes() == before
    learner, local = load_checkpoint(path, group)
    assert local is not None and local["rank"] == rank
    assert learner["world_size"] == world_size
    torch.save(local["curriculum"], Path(directory) / f"curriculum_{rank}.pt")
  finally:
    group.shutdown()


def test_rank_failures_and_atomic_snapshot_commit(tmp_path: Path, monkeypatch):
  with socket.socket() as listener:
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
  monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
  monkeypatch.setenv("MASTER_PORT", str(port))
  spawn(_infrastructure_worker, (2, str(tmp_path)), 2, join=True)
  first, second = [
    torch.load(tmp_path / f"curriculum_{r}.pt", weights_only=True) for r in range(2)
  ]
  for name in ("normalized", "counts", "past_weights"):
    torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
