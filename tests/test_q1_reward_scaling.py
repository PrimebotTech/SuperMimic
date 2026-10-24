"""Physical-unit invariance, gradients, and checkpoint reward compatibility."""

import io
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import torch
from conftest import Q1_JOINT_NAMES, link_name
from torch import nn

from supermimic.robot import Q1Robot, RewardScaling, normalize_quat


def write_robot(path: Path, length_scale: float = 1.0) -> Path:
  tree = ET.Element("mujoco", model="q1_v2")
  ET.SubElement(tree, "compiler", angle="radian", autolimits="true")
  world = ET.SubElement(tree, "worldbody")
  root = ET.SubElement(world, "body", name="pelvis", pos=f"0 0 {0.793 * length_scale}")
  ET.SubElement(root, "freejoint")
  ET.SubElement(root, "geom", type="sphere", size="0.02", mass="2")
  actuators = ET.SubElement(tree, "actuator")
  body = root
  for index, joint in enumerate(Q1_JOINT_NAMES):
    body = ET.SubElement(
      root if index % 3 == 0 else body,
      "body",
      name=link_name(joint),
      pos=f"{0.02 * length_scale} 0 {-0.03 * length_scale}",
    )
    ET.SubElement(
      body,
      "joint",
      name=joint,
      axis="0 1 0" if index % 2 else "1 0 0",
      pos=f"{0.01 * length_scale} 0 0",
      range="-2 2",
    )
    ET.SubElement(
      body,
      "geom",
      type="sphere",
      size="0.01",
      mass="4" if link_name(joint) == "torso_link" else "0.1",
    )
    ET.SubElement(actuators, "motor", joint=joint, ctrlrange="-20 20")
  ET.ElementTree(tree).write(path)
  return path


@pytest.fixture
def robot(tmp_path: Path) -> Q1Robot:
  return Q1Robot(write_robot(tmp_path / "q1.xml")).double()


def state_at_rest(robot: Q1Robot) -> torch.Tensor:
  state = torch.zeros(1, robot.state_dim, dtype=torch.float64)
  state[:, :29] = torch.from_numpy(robot.model.qpos0)
  return state


@pytest.mark.parametrize("adapted", [False, True])
def test_translation_and_velocity_have_normalized_physical_units(
  robot: Q1Robot, adapted: bool
):
  if adapted:
    robot.reward_scaling = RewardScaling.q1_proportional(0.419)
  scaling = robot.reward_scaling
  state = state_at_rest(robot)
  displaced = state.clone()
  displaced[:, 0] += 0.7 * scaling.position_std
  velocity = state.clone()
  velocity[:, 29] = 0.7 * scaling.linear_velocity_std
  action = state[:, 57:79]
  torch.testing.assert_close(
    robot.reward(state, displaced, state, action, 0.02),
    state.new_tensor([0.02 * (3.5 + 1.5 * math.exp(-(0.7**2)))]),
  )
  torch.testing.assert_close(
    robot.reward(state, velocity, state, action, 0.02),
    state.new_tensor([0.02 * (4.0 + math.exp(-(0.7**2)))]),
  )
  torch.testing.assert_close(
    robot.reward(state, state, state, action, 0.02), state.new_tensor([0.1])
  )


def test_geometric_scaling_preserves_full_fk_reward_and_gradients(tmp_path: Path):
  ratio = 0.419 / 0.793
  baseline = Q1Robot(write_robot(tmp_path / "baseline.xml")).double()
  adapted = Q1Robot(
    write_robot(tmp_path / "adapted.xml", ratio), reward_scaling="q1_proportional"
  ).double()
  generator = torch.Generator().manual_seed(311)
  state = torch.randn(2, 79, generator=generator, dtype=torch.float64) * 0.1
  state[:, 3:7] = normalize_quat(state[:, 3:7])
  target = torch.randn(2, 79, generator=generator, dtype=torch.float64) * 0.1
  target[:, 3:7] = normalize_quat(target[:, 3:7])
  state[:, 57:79] = 0
  multiplier = state.new_ones(79)
  multiplier[:3] = ratio
  multiplier[29:32] = ratio
  baseline_state = state.clone().requires_grad_()
  adapted_state = (state * multiplier).requires_grad_()
  action = state.new_zeros(2, 22)
  expected = baseline.reward(state, baseline_state, target, action, 0.02)
  actual = adapted.reward(
    state * multiplier, adapted_state, target * multiplier, action, 0.02
  )
  torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-9)
  expected_grad = torch.autograd.grad(expected.sum(), baseline_state)[0]
  actual_grad = torch.autograd.grad(actual.sum(), adapted_state)[0]
  assert torch.isfinite(actual_grad).all()
  assert actual_grad[:, :3].abs().sum() > 0
  torch.testing.assert_close(
    actual_grad * multiplier, expected_grad, rtol=1e-6, atol=1e-9
  )


def test_joint_sums_preserve_29_joint_regularization_and_action_gradient(
  robot: Q1Robot,
):
  robot.reward_scaling = RewardScaling.q1_proportional(0.419)
  state = state_at_rest(robot)
  action = torch.full((1, 22), 0.2, dtype=torch.float64, requires_grad=True)
  result = robot.reward(state, state, state, action, 0.02)
  torch.testing.assert_close(result, state.new_tensor([0.02 * (5 - 0.1 * 29 * 0.2**2)]))
  gradient = torch.autograd.grad(result.sum(), action)[0]
  torch.testing.assert_close(
    gradient, torch.full_like(action, -0.02 * 0.1 * (29 / 22) * 2 * 0.2)
  )
  beyond_limits = state.clone()
  beyond_limits[:, 7:29] = robot.joint_limits[:, 1] + 0.1
  beyond_limits.requires_grad_()
  result = robot.reward(state, beyond_limits, beyond_limits, action.detach() * 0, 0.02)
  torch.testing.assert_close(result, state.new_tensor([0.02 * (5 - 2 * 29 * 0.1)]))
  limit_gradient = torch.autograd.grad(result.sum(), beyond_limits)[0]
  torch.testing.assert_close(
    limit_gradient[:, 7:29], torch.full((1, 22), -0.02 * 2 * 29 / 22).double()
  )


@pytest.mark.parametrize("adapted", [False, True])
def test_height_threshold_scales_and_tilt_stays_unchanged(
  robot: Q1Robot, adapted: bool
):
  if adapted:
    robot.reward_scaling = RewardScaling.q1_proportional(0.419)
  state = state_at_rest(robot)
  near = state.clone()
  far = state.clone()
  threshold = robot.reward_scaling.termination_height
  near[:, 2] += threshold * 0.999
  far[:, 2] += threshold * 1.001
  assert not robot.terminated(near, state).item()
  assert robot.terminated(far, state).item()
  # A 90 degree rotation produces a gravity-z difference of 1 > 0.8.
  tilted = state.clone()
  tilted[:, 3:7] = state.new_tensor([math.sqrt(0.5), math.sqrt(0.5), 0, 0])
  assert robot.terminated(tilted, state).item()
  assert robot.reward_scaling.orientation_std == 0.4
  assert robot.reward_scaling.angular_velocity_std == 3.14
  assert robot.reward_scaling.termination_tilt == 0.8


@pytest.mark.parametrize("nested", [False, True])
def test_weights_only_checkpoint_restores_reward_and_legacy_semantics(
  robot: Q1Robot, tmp_path: Path, nested: bool
):
  robot.reward_scaling = RewardScaling.q1_proportional(0.419)
  parent = nn.ModuleDict({"robot": robot}) if nested else robot
  saved = parent.state_dict()
  stream = io.BytesIO()
  torch.save(saved, stream)
  stream.seek(0)
  restored = Q1Robot(write_robot(tmp_path / "restored.xml")).double()
  recipient = nn.ModuleDict({"robot": restored}) if nested else restored
  recipient.load_state_dict(torch.load(stream, weights_only=True))
  assert restored.reward_identity() == robot.reward_identity()
  legacy = {
    key: value for key, value in saved.items() if not key.endswith("_extra_state")
  }
  recipient.load_state_dict(legacy)
  assert restored.reward_scaling == RewardScaling()
  state = state_at_rest(restored)
  far = state.clone()
  far[:, 2] += 0.15
  assert not restored.terminated(far, state).item()


def test_adapted_observation_remains_proprioceptive(robot: Q1Robot):
  state = state_at_rest(robot)
  context = state.new_zeros(1, robot.context_dim)
  expected = robot.observation(state, context)
  robot.reward_scaling = RewardScaling.q1_proportional(0.419)
  state[:, :3] += 99
  state[:, 29:32] += 77
  torch.testing.assert_close(robot.observation(state, context), expected)
  assert expected.shape == (1, 72)


@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf])
def test_invalid_morphology_scale_is_rejected(value: float):
  with pytest.raises(ValueError, match="finite and positive"):
    RewardScaling.q1_proportional(value)
  with pytest.raises(ValueError, match="finite and positive"):
    RewardScaling.q1_proportional(0.419, reference_pelvis_height=value)


def test_actual_q1_xml_metadata_and_asset_hash_stay_consistent():
  supplied_xml = os.environ.get("SUPERMIMIC_Q1_XML")
  if supplied_xml is None:
    pytest.skip("Set SUPERMIMIC_Q1_XML to check a real Q1 V2 asset")
  xml = Path(supplied_xml)
  legacy = Q1Robot(xml)
  adapted = Q1Robot(xml, reward_scaling="q1_proportional")
  assert legacy.sha256 == adapted.sha256
  assert legacy.asset_sha256 == adapted.asset_sha256
  assert adapted.reward_scaling == RewardScaling.q1_proportional(0.419)
  assert adapted.reward_scaling.joint_sum_multiplier == 29 / 22
  assert (adapted.obs_dim, adapted.physical_dim, adapted.model.nu) == (72, 57, 22)
