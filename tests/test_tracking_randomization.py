"""Tracking randomization acts on the requested bodies and lifecycle."""

import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from test_q1_supermimic import robot as robot

from supermimic.randomization import RandomizationCfg
from supermimic.robot import Q1Robot
from supermimic.simulator import Q1Simulator


@pytest.fixture
def contact_robot(robot: Q1Robot, tmp_path: Path) -> Q1Robot:
  tree = ET.parse(robot.xml)
  world = tree.getroot().find("worldbody")
  assert world is not None
  ET.SubElement(
    world,
    "geom",
    name="floor",
    type="plane",
    size="0 0 1",
    friction="0.6 0.005 0.0001",
    contype="0",
    conaffinity="0",
  )
  path = tmp_path / "q1_contact.xml"
  tree.write(path)
  return Q1Robot(path)


def initial_state(robot: Q1Robot, count: int) -> torch.Tensor:
  state = torch.zeros(count, robot.state_dim)
  state[:, :29] = torch.tensor(robot.model.qpos0, dtype=torch.float32)
  return state


def test_startup_parameters_survive_episode_and_nominal_resets(contact_robot: Q1Robot):
  simulator = Q1Simulator(contact_robot, 8, "cpu")
  ids, state = torch.arange(8), initial_state(contact_robot, 8)
  simulator.reset(ids, state, randomize=True)
  startup = simulator.context.clone()
  assert not torch.equal(startup[0], startup[1])
  simulator.reset(ids[:2], state[:2], randomize=True)
  torch.testing.assert_close(simulator.context, startup, atol=0, rtol=0)
  simulator.reset(ids, state, randomize=False)
  assert not simulator.push_enabled.any()
  simulator.reset(ids, state, randomize=True)
  torch.testing.assert_close(simulator.context, startup, atol=0, rtol=0)


def test_only_torso_mass_base_com_and_shared_feet_change(contact_robot: Q1Robot):
  simulator = Q1Simulator(contact_robot, 2, "cpu")
  simulator.reset(torch.arange(2), initial_state(contact_robot, 2), randomize=True)
  layout, cfg = contact_robot.context_layout, simulator.randomization
  for i, model in enumerate(simulator.cpu_models):
    mass = torch.tensor(model.body_mass, dtype=torch.float32)
    expected = simulator.default_mass.clone()
    expected[contact_robot.torso_id] += simulator.context[i, layout.torso_added_mass][0]
    torch.testing.assert_close(mass, expected)
    center = torch.tensor(model.body_ipos[contact_robot.root_id], dtype=torch.float32)
    offset = center - simulator.default_ipos[contact_robot.root_id]
    for axis, (low, high) in enumerate(cfg.base_com):
      assert low <= offset[axis] <= high
    friction = torch.tensor(model.geom_friction[:, 0], dtype=torch.float32)
    foot = friction[simulator.foot_geoms]
    assert (foot == foot[0]).all()
    assert cfg.friction[0] <= foot[0] <= cfg.friction[1]
    assert friction[simulator.floor_geoms].max() <= foot[0]
    untouched = torch.ones(len(friction), dtype=torch.bool)
    untouched[simulator.foot_geoms] = False
    untouched[simulator.floor_geoms] = False
    torch.testing.assert_close(
      friction[untouched],
      torch.tensor(contact_robot.model.geom_friction[:, 0], dtype=torch.float32)[
        untouched
      ],
    )
  assert not simulator.context[:, layout.reserved_scalar].any()
  assert (simulator.context[:, layout.reserved_joints] == 1).all()
  assert (simulator.context[:, layout.reserved_bodies] == 1).all()


def test_push_is_world_velocity_impulse_and_preserves_physical_label(
  contact_robot: Q1Robot,
):
  simulator = Q1Simulator(contact_robot, 2, "cpu")
  simulator.reset(torch.arange(2), initial_state(contact_robot, 2), randomize=True)
  simulator.push_timer[:] = torch.tensor([0.0, 3.0])
  state = simulator.step(torch.zeros(2, 22))
  assert simulator.pushed.tolist() == [True, False]
  delta = state[:, 29:35] - simulator.transition_state[:, 29:35]
  for axis, (low, high) in enumerate(simulator.randomization.push_velocity):
    assert low <= delta[0, axis] <= high
  assert delta[0].abs().sum() > 0
  torch.testing.assert_close(delta[1], torch.zeros(6), atol=0, rtol=0)
  assert 1.0 <= simulator.push_timer[0] <= 3.0
  simulator.reset(torch.arange(2), initial_state(contact_robot, 2), randomize=False)
  simulator.push_timer.zero_()
  state = simulator.step(torch.zeros(2, 22))
  torch.testing.assert_close(state, simulator.transition_state, atol=0, rtol=0)


def test_reference_ranges_and_gaussian_sensor_scales(contact_robot: Q1Robot):
  cfg = RandomizationCfg()
  cfg.validate()
  assert cfg.friction == (0.3, 1.2)
  assert cfg.encoder_bias == (-0.02, 0.02)
  assert cfg.kp_factor == (0.8, 1.2)
  assert cfg.kd_factor == (0.9, 1.1)
  assert cfg.torso_added_mass == (-1.5, 1.5)
  assert cfg.push_interval_seconds == (1.0, 3.0)
  assert set(asdict(cfg)) == {
    "friction",
    "encoder_bias",
    "kp_factor",
    "kd_factor",
    "base_com",
    "torso_added_mass",
    "push_interval_seconds",
    "push_velocity",
  }
  scale = contact_robot.noise_scale()
  torch.testing.assert_close(
    scale,
    torch.tensor([0.0058] * 22 + [1.0] * 22 + [0.115] * 3 + [0.029] * 3 + [0] * 22),
  )
