"""Forecast checks for terminal boundaries, fixed actions, and physical units."""

from typing import cast

import pytest
import torch

from supermimic.diagnostics import (
  EpisodeTrace,
  autoregressive_prediction,
  physical_errors,
  score_horizon,
  window_starts,
)
from supermimic.models import WorldModel
from supermimic.robot import Q1Robot


class ActionIntegrator:
  scale = torch.ones(79)

  def advance_context(
    self, context: torch.Tensor, action: torch.Tensor
  ) -> torch.Tensor:
    return context

  def __call__(
    self, state: torch.Tensor, action: torch.Tensor, context: torch.Tensor
  ) -> torch.Tensor:
    result = state.clone()
    result[:, 0] += action[:, 0] + context[:, 0]
    result[:, 57:] = action
    return result

  def termination_logits(
    self, robot: Q1Robot, state: torch.Tensor, target: torch.Tensor
  ) -> torch.Tensor:
    return torch.where(state[:, 0] >= target[:, 0], 2.0, -2.0)


def test_windows_include_terminal_endpoint_and_cannot_cross_it():
  labels = torch.tensor([False, False, True, False, False, False])
  assert window_starts(labels, 1).tolist() == list(range(6))
  assert window_starts(labels, 3).tolist() == [0, 3]
  assert window_starts(labels, 4).tolist() == []
  assert window_starts(labels, 32).tolist() == []
  with pytest.raises(ValueError, match="positive horizon"):
    window_starts(labels, 0)


def test_forecast_carries_predictions_and_holds_executed_actions_fixed():
  initial = torch.zeros(2, 79)
  initial[:, 3] = 1
  initial[:, 0] = torch.tensor([10.0, 20.0])
  actions = torch.zeros(2, 3, 22)
  actions[:, :, 0] = torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
  context = torch.tensor([[0.5], [1.0]])
  forecast = autoregressive_prediction(ActionIntegrator(), initial, actions, context)
  torch.testing.assert_close(forecast[:, 0], torch.tensor([17.5, 17.0]))
  torch.testing.assert_close(forecast[:, 57:], actions[:, -1])
  torch.testing.assert_close(initial[:, 0], torch.tensor([10.0, 20.0]))


def test_physical_errors_use_training_scale_and_ignore_quaternion_sign():
  actual = torch.zeros(1, 79)
  actual[:, 3] = 1
  predicted = actual.clone()
  predicted[:, 3] = -1
  predicted[:, 0] = 3
  predicted[:, 7] = 2
  predicted[:, 29] = 6
  predicted[:, 35] = 20
  scale = torch.ones(79)
  scale[0], scale[7], scale[29], scale[35] = 3, 2, 6, 20
  errors = physical_errors(predicted, actual, scale)
  assert errors.normalized_physical_rmse == pytest.approx((4 / 57) ** 0.5)
  assert errors.root_position_rmse_m == pytest.approx(3 / 3**0.5)
  assert errors.root_orientation_rmse_rad == pytest.approx(0)
  assert errors.joint_angle_rmse_rad == pytest.approx(2 / 22**0.5)
  assert errors.root_linear_velocity_rmse_m_per_s == pytest.approx(6 / 3**0.5)
  assert errors.joint_velocity_rmse_rad_per_s == pytest.approx(20 / 22**0.5)


def test_horizon_scoring_preserves_terminal_labels_and_marks_missing_windows():
  states = torch.zeros(9, 79)
  states[:, 0] = torch.arange(9)
  states[:, 3] = 1
  actions = torch.zeros(8, 22)
  actions[:, 0] = 1
  states[1:, 57:] = actions
  references = states[1:].clone()
  references[:, 0] = 8
  episode = EpisodeTrace(
    motion="synthetic_numerical_test",
    start_frame=0,
    states=states,
    actions=actions,
    references=references,
    context=torch.zeros(1),
    terminated=torch.tensor([False] * 7 + [True]),
    end_reason="terminated",
  )
  # The scoring interface needs only this numerical test double's callable and
  # termination method; it does not invoke robot simulation in these unit tests.
  model = cast(WorldModel, ActionIntegrator())
  robot = cast(Q1Robot, object())
  result = score_horizon(robot, model, [episode], 8)
  assert result.status == "measured" and result.windows == 1
  assert result.errors is not None
  assert result.errors.normalized_physical_rmse == pytest.approx(0)
  assert result.terminal_labels == 1
  assert result.termination_bce == pytest.approx(0.126928, abs=1e-6)
  assert result.mean_probability_on_nonterminal is None
  missing = score_horizon(robot, model, [episode, episode], 32)
  assert missing.status == "not_measured" and missing.windows == 0
  assert missing.errors is None and missing.termination_bce is None
  assert missing.reason is not None
