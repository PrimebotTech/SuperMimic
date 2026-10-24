"""Absorbing inputs for extinct branches of a differentiable model rollout."""

import torch
from torch import Tensor

MIN_SURVIVAL = 1e-6
TERMINAL_HANDLING = "survival_weighted_absorbing_v1"


def absorbing_inputs(
  survival: Tensor,
  state: Tensor,
  history: Tensor,
  context: Tensor,
  initial_state: Tensor,
  initial_history: Tensor,
  initial_context: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
  # A zero-probability branch must not feed ever larger states through the
  # model: even 0 * inf contaminates BPTT. Truncate probability mass below
  # 1e-6; the termination head keeps its analytic gradients on live branches.
  active = survival.detach() > MIN_SURVIVAL
  return (
    torch.where(active[:, None], state, initial_state),
    torch.where(active[:, None, None], history, initial_history),
    torch.where(active[:, None], context, initial_context),
    active,
  )


def surviving_reward_mean(rewards: Tensor, weights: Tensor) -> Tensor:
  return (rewards * weights).sum() / weights.sum().clamp_min(1)
