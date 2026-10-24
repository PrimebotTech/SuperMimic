"""Deferred validation preserves updates and coordinates real-rank failures."""

import socket
from pathlib import Path
from unittest import mock

import pytest
import torch
import torch.distributed as dist
from torch.multiprocessing.spawn import spawn

from supermimic.distributed import Distributed


@pytest.fixture
def distributed_address(monkeypatch):
  with socket.socket() as listener:
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
  monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
  monkeypatch.setenv("MASTER_PORT", str(port))


def _network() -> torch.nn.Sequential:
  torch.manual_seed(31)
  model = torch.nn.Sequential(
    torch.nn.Linear(4, 3), torch.nn.Tanh(), torch.nn.Linear(3, 1)
  ).double()
  # The final head must keep input Jacobians without receiving weight gradients.
  model[-1].requires_grad_(False)
  return model


def _parity_worker(rank: int, directory: str) -> None:
  legacy = Distributed(rank, 2, rank)
  deferred = Distributed(rank, 2, rank, validation_mode="per_update")
  deferred.initialize("cpu", timeout_seconds=30)
  try:
    models = [_network(), _network()]
    optimizers = [torch.optim.Adam(m.parameters(), lr=2e-4) for m in models]
    reductions: list[list[int]] = [[], []]
    for update in range(3):
      inputs = torch.arange(16, dtype=torch.float64).reshape(4, 4)
      inputs = inputs / 7 + rank + 0.2 * update
      for i, group in enumerate((legacy, deferred)):
        model, optimizer = models[i], optimizers[i]
        group.begin_update()
        optimizer.zero_grad(set_to_none=True)
        loss = (model(inputs) - 2).square().mean()
        group.require_finite(loss, "Nonfinite loss")
        loss.backward()
        with mock.patch.object(dist, "all_reduce", wraps=dist.all_reduce) as reduce:
          group.average_gradients(model.parameters(), bucket_bytes=64)
          reductions[i].append(reduce.call_count)
        norm = torch.nn.utils.clip_grad_norm_(
          model.parameters(), 1.0, error_if_nonfinite=i == 0
        )
        group.require_finite(norm, "Nonfinite gradient norm")
        optimizer.step()
        group.validate_update()
      for left, right in zip(
        models[0].parameters(), models[1].parameters(), strict=True
      ):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        if left.requires_grad:
          assert left.grad is not None and right.grad is not None
          torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
        else:
          assert left.grad is None and right.grad is None
      left_state, right_state = (optimizer.state_dict() for optimizer in optimizers)
      for key, state in left_state["state"].items():
        for name, value in state.items():
          torch.testing.assert_close(value, right_state["state"][key][name])
    torch.save(
      {"reductions": reductions, "weights": models[1].state_dict()},
      Path(directory) / f"rank_{rank}.pt",
    )
  finally:
    deferred.shutdown()


def test_deferred_updates_equal_legacy_over_two_cpu_ranks(
  tmp_path: Path, distributed_address
):
  spawn(_parity_worker, (str(tmp_path),), 2, join=True)
  first, second = [
    torch.load(tmp_path / f"rank_{rank}.pt", weights_only=True) for rank in range(2)
  ]
  for key, value in first["weights"].items():
    torch.testing.assert_close(value, second["weights"][key], rtol=0, atol=0)
  for result in (first, second):
    legacy, deferred = result["reductions"]
    assert deferred[0] == legacy[0]
    assert all(
      fast == old - 1 for fast, old in zip(deferred[1:], legacy[1:], strict=True)
    )


def _failure_worker(rank: int, directory: str, scenario: str) -> None:
  group = Distributed(rank, 2, rank, validation_mode="per_update")
  group.initialize("cpu", timeout_seconds=30)
  try:
    torch.manual_seed(31)
    model = torch.nn.Linear(2, 1)
    if scenario == "unexpected":
      model.bias.requires_grad_(False)
    if scenario != "initial":
      group.begin_update()
      model(torch.ones(2, 2)).sum().backward()
      group.average_gradients(model.parameters(), bucket_bytes=8)
      group.validate_update()
    model.zero_grad(set_to_none=True)
    group.begin_update()
    if scenario == "unexpected" and rank == 1:
      model.bias.requires_grad_(True)
    loss = model(torch.full((2, 2), float(rank + 1))).sum()
    if scenario == "nonfinite" and rank == 1:
      loss = loss * float("nan")
    group.require_finite(loss, "Nonfinite loss")
    if scenario != "initial" or rank == 0:
      loss.backward()
    if scenario == "missing" and rank == 1:
      model.weight.grad = None
    group.average_gradients(model.parameters(), bucket_bytes=8)
    if scenario in ("missing", "initial") and rank == 1:
      assert model.weight.grad is None
    expected = "Nonfinite loss" if scenario == "nonfinite" else "gradients|presence"
    with pytest.raises(RuntimeError, match=expected) as error:
      group.validate_update()
    # Failure cleanup must permit a new update without carrying old flags.
    group.begin_update()
    group.require_finite(torch.tensor(1.0), "Clean next update")
    group.validate_update()
    group.validate_update()
    torch.save({"error": str(error.value)}, Path(directory) / f"rank_{rank}.pt")
  finally:
    group.shutdown()


@pytest.mark.parametrize("scenario", ["missing", "unexpected", "initial", "nonfinite"])
def test_deferred_failure_reaches_every_rank_without_changing_collective_shapes(
  tmp_path: Path, distributed_address, scenario: str
):
  spawn(_failure_worker, (str(tmp_path), scenario), 2, join=True)
  results = [
    torch.load(tmp_path / f"rank_{rank}.pt", weights_only=True) for rank in range(2)
  ]
  assert results[0]["error"] == results[1]["error"]
  assert "rank 1:" in results[0]["error"]


def test_deferred_single_process_checks_vectors_and_update_lifecycle():
  group = Distributed.detached(validation_mode="per_update")
  with pytest.raises(RuntimeError, match="begin_update"):
    group.require_finite(torch.tensor(1.0), "Outside update")
  group.begin_update()
  with pytest.raises(RuntimeError, match="not been validated"):
    group.begin_update()
  group.require_finite(torch.tensor([1.0, float("inf")]), "Nonfinite vector")
  with pytest.raises(RuntimeError, match="Nonfinite vector"):
    group.validate_update()
  group.begin_update()
  group.require_finite(torch.tensor([1.0, 2.0]), "Finite vector")
  group.validate_update()
  group.validate_update()
