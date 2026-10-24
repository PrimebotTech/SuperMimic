"""Opt-in reward semantics, preserved failures, and physical input gradients."""

import argparse
import json
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from test_q1_reward_scaling import state_at_rest, write_robot

from supermimic.models import WorldModel, frozen
from supermimic.randomization import nominal_context
from supermimic.reward_tuning import (
  RewardTuningCfg,
  compare_fixed_rollout,
  reward_tuning_terms,
  tuned_reward,
)
from supermimic.robot import Q1Robot, RewardScaling, rotation_error


@pytest.fixture
def robot(tmp_path: Path) -> Q1Robot:
  robot = Q1Robot(write_robot(tmp_path / "q1.xml")).double()
  robot.reward_scaling = RewardScaling.q1_proportional(0.419)
  return robot


def test_disabled_tuning_is_bit_exact_for_values_and_input_gradients(robot: Q1Robot):
  state, target = state_at_rest(robot), state_at_rest(robot)
  actual = state.clone()
  actual[:, 21:29] += 0.6
  actual.requires_grad_()
  action = state.new_full((1, 22), 0.1, requires_grad=True)
  baseline = robot.reward(state, actual, target, action, 0.02)
  tuned = tuned_reward(robot, state, actual, target, action, 0.02)
  assert torch.equal(baseline, tuned)
  baseline_gradients = torch.autograd.grad(
    baseline.sum(), (actual, action), retain_graph=True
  )
  tuned_gradients = torch.autograd.grad(tuned.sum(), (actual, action))
  for original, experimental in zip(baseline_gradients, tuned_gradients, strict=True):
    assert torch.equal(original, experimental)


@pytest.mark.parametrize("cached", [False, True])
def test_disabled_tuning_does_not_add_kinematics_work(
  robot: Q1Robot,
  monkeypatch: pytest.MonkeyPatch,
  cached: bool,
):
  state = state_at_rest(robot)
  heads = robot.kinematics(state) if cached else None
  original = robot.kinematics
  calls = 0

  def counted(value: torch.Tensor):
    nonlocal calls
    calls += 1
    return original(value)

  monkeypatch.setattr(robot, "kinematics", counted)
  tuned_reward(robot, state, state, state, state[:, 57:79], 0.02, target_heads=heads)
  assert calls == (1 if cached else 2)


@pytest.mark.parametrize(
  "tuning",
  [
    RewardTuningCfg(),
    RewardTuningCfg(arm_joint_weight=0.05),
    RewardTuningCfg(arm_pd_target_weight=0.05),
    RewardTuningCfg(arm_joint_weight=0.05, arm_pd_target_weight=0.05),
    RewardTuningCfg(body_orientation_std=0.6),
  ],
)
def test_perfect_static_score_and_strict_height_failure_are_preserved(
  robot: Q1Robot,
  tuning: RewardTuningCfg,
):
  state = state_at_rest(robot)
  action = state[:, 57:79]
  torch.testing.assert_close(
    tuned_reward(robot, state, state, state, action, 0.02, tuning=tuning),
    state.new_tensor([0.1]),
  )
  failed = state.clone()
  failed[:, 2] += robot.reward_scaling.termination_height + 0.001
  before = robot.reward_identity()
  assert robot.terminated(failed, state).item()
  tuned_reward(robot, state, failed, state, action, 0.02, tuning=tuning)
  assert robot.terminated(failed, state).item()
  assert robot.reward_identity() == before


def test_joint_aux_supplies_finite_difference_verified_arm_correction(robot: Q1Robot):
  state, target = state_at_rest(robot), state_at_rest(robot)
  state[:, 21:29] = torch.linspace(-1.4, 1.4, 8, dtype=state.dtype)
  state.requires_grad_()
  tuning = RewardTuningCfg(arm_joint_weight=0.05)
  value = reward_tuning_terms(robot, state, target, state[:, 57:79], tuning)
  gradient = torch.autograd.grad(value["arm_joint_penalty"].sum(), state)[0]
  direction = target[:, 21:29] - state.detach()[:, 21:29]
  assert (gradient[:, 21:29] * direction > 0).all()
  assert torch.count_nonzero(gradient[:, :21]) == 0
  epsilon = 1e-6
  values = []
  for sign in [1, -1]:
    perturbed = state.detach().clone()
    perturbed[:, 21] += sign * epsilon
    values.append(
      reward_tuning_terms(robot, perturbed, target, state[:, 57:79], tuning)[
        "arm_joint_penalty"
      ]
    )
  torch.testing.assert_close(gradient[:, 21], (values[0] - values[1]) / (2 * epsilon))


def test_pd_prior_uses_randomized_zero_and_has_correct_action_gradient(robot: Q1Robot):
  state, target = state_at_rest(robot), state_at_rest(robot)
  context = nominal_context(robot.context_layout, 1, "cpu").double()
  context[:, robot.context_layout.motor_offset] = 0.17
  target[:, 21:29] = 0.8
  tuning = RewardTuningCfg(arm_pd_target_weight=0.05, action_scale=0.3)
  action = state.new_zeros((1, 22), requires_grad=True)
  penalty = reward_tuning_terms(robot, state, target, action, tuning, context)[
    "arm_pd_target_penalty"
  ]
  gradient = torch.autograd.grad(penalty.sum(), action)[0]
  assert (gradient[:, 14:] > 0).all()
  assert torch.count_nonzero(gradient[:, :14]) == 0
  compensated = action.detach().clone()
  compensated[:, 14:] = (0.8 - 0.17) / tuning.action_scale
  correct = reward_tuning_terms(robot, state, target, compensated, tuning, context)[
    "arm_pd_target_penalty"
  ]
  ignored = reward_tuning_terms(robot, state, target, compensated, tuning)[
    "arm_pd_target_penalty"
  ]
  torch.testing.assert_close(correct, state.new_zeros(1))
  assert ignored.item() < 0


def test_frozen_model_keeps_action_input_gradients_without_weight_gradients(
  robot: Q1Robot,
):
  torch.manual_seed(311)
  model = WorldModel(robot, (16, 16)).double()
  state, target = state_at_rest(robot), state_at_rest(robot)
  target[:, 21:29] += 0.8
  action = state.new_zeros((1, 22), requires_grad=True)
  context = nominal_context(robot.context_layout, 1, "cpu").double()
  with frozen(model):
    predicted = model(state, action, context)
    reward = tuned_reward(
      robot,
      state,
      predicted,
      target,
      action,
      0.02,
      tuning=RewardTuningCfg(arm_joint_weight=0.05),
    )
    gradient = torch.autograd.grad(reward.sum(), action)[0]
  assert torch.isfinite(gradient).all()
  assert gradient[:, 14:].norm() > 0
  assert all(parameter.grad is None for parameter in model.parameters())
  assert all(parameter.requires_grad for parameter in model.parameters())


def test_enabled_tuning_never_inflates_baseline_score(robot: Q1Robot):
  state, target = state_at_rest(robot), state_at_rest(robot)
  actual = state.clone()
  actual[:, 21:29] = 0.8
  action = state.new_full((1, 22), 0.2)
  baseline = robot.reward(state, actual, target, action, 0.02)
  tuned = tuned_reward(
    robot,
    state,
    actual,
    target,
    action,
    0.02,
    tuning=RewardTuningCfg(arm_joint_weight=0.05, arm_pd_target_weight=0.05),
  )
  assert (tuned < baseline).all()
  assert asdict(RewardTuningCfg())["arm_joint_weight"] == 0


def test_orientation_override_replaces_body_gaussian_and_preserves_root(robot: Q1Robot):
  state, target = state_at_rest(robot), state_at_rest(robot)
  actual = state.clone()
  actual[:, 21:29] = 0.8
  action = state[:, 57:79]
  heads = robot.kinematics(target)
  rotations = robot.kinematics(actual)[1]
  error = rotation_error(rotations[:, 1:], heads[1][:, 1:]).mean(-1)
  replacement = (-error / 0.6**2).exp()
  original = (-error / robot.reward_scaling.orientation_std**2).exp()
  baseline = robot.reward(state, actual, target, action, 0.02, heads)
  tuned = tuned_reward(
    robot,
    state,
    actual,
    target,
    action,
    0.02,
    tuning=RewardTuningCfg(body_orientation_std=0.6),
    target_heads=heads,
  )
  torch.testing.assert_close(tuned - baseline, 0.02 * (replacement - original))
  assert robot.reward_scaling.orientation_std == 0.4
  assert tuned.item() <= 0.1


def test_cpu_comparison_selects_arm_joints_by_name_after_xml_reordering(tmp_path: Path):
  xml = write_robot(tmp_path / "q1.xml")
  tree = ET.parse(xml)
  root = tree.find("worldbody/body")
  assert root is not None
  bodies = root.findall("body")
  for body in bodies:
    root.remove(body)
  for body in reversed(bodies):
    root.append(body)
  tree.write(xml)
  robot = Q1Robot(xml)
  arms = [
    i
    for i, name in enumerate(robot.joint_names)
    if "shoulder" in name or "elbow" in name
  ]
  assert arms != list(range(14, 22))
  checkpoint = tmp_path / "checkpoint.pt"
  torch.save(
    {
      "config": {"robot_xml": str(xml), "action_scale": 0.5},
      "identity": {"asset_sha256": robot.asset_sha256},
      "robot": robot.state_dict(),
    },
    checkpoint,
  )
  qpos = np.repeat(robot.model.qpos0[None], 30, axis=0)
  qpos[:, [7 + i for i in arms]] = 0.8
  reference, rollout, config = (
    tmp_path / name for name in ["reference.npz", "rollout.npz", "config.json"]
  )
  np.savez(
    reference,
    qpos=qpos,
    robot_name="q1_v2",
    root_quat_order="wxyz",
    fps=50.0,
    robot_joint_names=np.array(robot.joint_names),
  )
  states = state_at_rest(robot).repeat(30, 1).numpy()
  np.savez(rollout, states=states)
  config.write_text(json.dumps(asdict(RewardTuningCfg(arm_joint_weight=0.05))))
  result = compare_fixed_rollout(
    argparse.Namespace(
      checkpoint=checkpoint,
      xml=xml,
      reference=reference,
      rollout=rollout,
      config=config,
      frames=25,
    )
  )
  assert result["physical_arm_qpos_rmse_rad"] == pytest.approx(0.8)
  assert result["arm_joint_names"] == [robot.joint_names[i] for i in arms]
  assert result["physical_improvement_demonstrated"] is False


@pytest.mark.parametrize(
  "field,value",
  [
    ("arm_joint_weight", -0.01),
    ("arm_pd_target_weight", float("nan")),
    ("arm_joint_weight", float("inf")),
    ("arm_pd_target_weight", -1.0),
    ("arm_error_std", 0.0),
    ("arm_error_std", float("inf")),
    ("action_scale", 0.0),
    ("action_scale", -0.5),
    ("body_orientation_std", 0.0),
    ("body_orientation_std", float("nan")),
  ],
)
def test_invalid_tuning_units_are_rejected(field: str, value: float):
  with pytest.raises(ValueError):
    RewardTuningCfg(**{field: value})
