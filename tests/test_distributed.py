"""Rank bookkeeping and the gradient bucketing, without needing a second GPU."""

import os
from unittest import mock

import pytest
import torch

from supermimic.data import ReturnResetSampler
from supermimic.distributed import Distributed, _buckets


def test_detached_group_is_a_complete_no_op():
  """A single process must behave exactly as it did before any of this existed."""
  group = Distributed.detached()
  assert not group.enabled and group.main and group.share(7, "Count") == 7
  layer = torch.nn.Linear(3, 2)
  layer(torch.ones(1, 3)).sum().backward()
  assert layer.weight.grad is not None
  before = layer.weight.grad.clone()
  group.average_gradients(layer.parameters())
  group.broadcast_parameters([layer])
  torch.testing.assert_close(layer.weight.grad, before)
  assert group.mean([1.0, 2.0]) == [1.0, 2.0]
  tensor = torch.ones(2)
  assert group.sum_(tensor) is tensor


def test_rank_assignment_comes_from_the_launcher():
  with mock.patch.dict(os.environ, {"RANK": "3", "WORLD_SIZE": "8"}, clear=False):
    group = Distributed.from_environment()
  assert (group.rank, group.world_size, group.local_rank) == (3, 8, 3)
  assert group.enabled and not group.main
  with mock.patch.dict(
    os.environ, {"RANK": "1", "WORLD_SIZE": "4", "LOCAL_RANK": "1"}, clear=False
  ):
    assert Distributed.from_environment().local_rank == 1
  with mock.patch.dict(os.environ, {"RANK": "8", "WORLD_SIZE": "8"}, clear=False):
    with pytest.raises(ValueError, match="rank 8 of 8"):
      Distributed.from_environment()
  for key in ("RANK", "WORLD_SIZE"):
    environment = {k: v for k, v in os.environ.items() if k != key}
    with mock.patch.dict(os.environ, environment, clear=True):
      assert not Distributed.from_environment().enabled


def test_totals_split_evenly_or_not_at_all():
  """A silent floor would change the batch the paper's constants were set for."""
  group = Distributed(0, 8, 0)
  assert group.share(4096, "Environment count") == 512
  with pytest.raises(ValueError, match="Environment count 100 does not divide"):
    group.share(100, "Environment count")


def test_gradient_buckets_stay_under_their_budget_and_lose_nothing():
  grads = [torch.zeros(n) for n in (100, 100, 100, 700, 50)]
  # Four bytes an element, so a 1200-byte budget holds exactly 300 of them.
  buckets = _buckets(grads, 1200)
  assert [sum(g.numel() for g in b) for b in buckets] == [300, 700, 50]
  assert all(len(b) == 1 or 4 * sum(g.numel() for g in b) <= 1200 for b in buckets)
  assert [g.data_ptr() for b in buckets for g in b] == [g.data_ptr() for g in grads]
  # One gradient larger than the budget still travels, in a bucket of its own.
  assert [len(b) for b in _buckets(grads, 4)] == [1, 1, 1, 1, 1]
  assert _buckets([], 1000) == []


def test_curriculum_reduction_shares_one_estimate_across_ranks():
  """Each rank sees its own environments; the hard-sampling share is global.

  Without the reduction, a rank only ever learns about the bins its own
  environments visited, so the 40% hard share would chase different bins on
  every GPU.
  """
  lengths = torch.tensor([200, 200])
  first, second = (ReturnResetSampler(lengths, 0.02, momentum=0.5) for _ in range(2))
  shared = ReturnResetSampler(lengths, 0.02, momentum=0.5)
  left = (torch.tensor([0, 0]), torch.tensor([0.2, 0.4]))
  right = (torch.tensor([1, 5]), torch.tensor([0.6, 0.8]))
  # Two ranks, each reducing its own half: both must end at the same estimate,
  # and at the one a single process would reach from the union of the samples.
  totals = [
    torch.stack(
      (
        torch.zeros(len(shared.clips)).scatter_add_(0, b, torch.ones_like(v)),
        torch.zeros(len(shared.clips)).scatter_add_(0, b, v),
      )
    )
    for b, v in (left, right)
  ]
  summed = totals[0] + totals[1]
  first.update(*left, reduce=lambda _: summed)
  second.update(*right, reduce=lambda _: summed)
  shared.update(
    torch.cat((left[0], right[0])), torch.cat((left[1], right[1])), reduce=None
  )
  torch.testing.assert_close(first.normalized, second.normalized)
  torch.testing.assert_close(first.normalized, shared.normalized)
  torch.testing.assert_close(first.counts, shared.counts)
