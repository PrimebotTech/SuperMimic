"""Mass and COM randomization must update per-world MuJoCo constraint constants."""

import copy
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, cast

import mujoco
import mujoco_warp as mw
import numpy as np
import pytest
import torch
from conftest import Q1_JOINT_NAMES, link_name

from supermimic.robot import Q1Robot, normalize_quat
from supermimic.simulator import Q1Simulator


@pytest.fixture
def constant_robot(tmp_path: Path) -> Q1Robot:
  supplied_xml = os.environ.get("SUPERMIMIC_Q1_XML")
  if supplied_xml is not None:
    return Q1Robot(Path(supplied_xml))
  # Keep the numerical check portable when the user's Q1 asset is unavailable.
  tree = ET.Element("mujoco", model="q1_constants")
  ET.SubElement(tree, "compiler", angle="radian", autolimits="true")
  ET.SubElement(tree, "option", timestep="0.001")
  world = ET.SubElement(tree, "worldbody")
  root = ET.SubElement(world, "body", name="pelvis", pos="0 0 0.8")
  ET.SubElement(root, "freejoint")
  ET.SubElement(root, "geom", type="sphere", size="0.04", mass="5")
  actuators = ET.SubElement(tree, "actuator")
  body = root
  for j, joint in enumerate(Q1_JOINT_NAMES):
    body = ET.SubElement(
      root if j % 3 == 0 else body,
      "body",
      name=link_name(joint),
      pos="0.02 0.01 -0.03",
    )
    ET.SubElement(
      body,
      "joint",
      name=joint,
      axis="0 1 0" if j % 2 else "1 0 0",
      range="-2 2",
      armature="0.05",
    )
    ET.SubElement(
      body,
      "geom",
      type="sphere",
      size="0.02",
      pos="0.01 0.02 0.03",
      mass="4" if link_name(joint) == "torso_link" else "0.2",
      contype="0",
      conaffinity="0",
    )
    ET.SubElement(actuators, "motor", joint=joint, ctrlrange="-20 20")
  path = tmp_path / "q1_constants.xml"
  ET.ElementTree(tree).write(path)
  return Q1Robot(path)


def reset_states(robot: Q1Robot) -> torch.Tensor:
  state = torch.zeros(2, robot.state_dim)
  state[:, :29] = torch.tensor(robot.model.qpos0, dtype=torch.float32)
  state[:, :3] += torch.tensor([[0.1, 0.0, 0.05], [-0.1, 0.05, 0.1]])
  state[:, 3:7] = normalize_quat(torch.tensor([[0.96, 0.12, -0.17, 0.2]]).repeat(2, 1))
  state[:, 7:29] += 0.03
  state[:, 29:57] = torch.linspace(-0.1, 0.15, 56).reshape(2, 28)
  return state


def assert_constants_match_cpu(simulator: Q1Simulator) -> None:
  robot = simulator.robot
  layout = robot.context_layout
  root = robot.root_id
  for index in range(simulator.num_envs):
    expected = copy.copy(robot.model)
    data = mujoco.MjData(expected)
    context = simulator.context[index].cpu().numpy()
    expected.body_mass[robot.torso_id] += context[layout.torso_added_mass][0]
    expected.body_ipos[root] += context[layout.root_com]
    mujoco.mj_setConst(expected, data)
    # mj_setConst leaves body_subtreemass alone, so compare the simulator's
    # ancestor-mask sum against a plain walk up the parent chain instead.
    subtree = np.zeros(expected.nbody)
    for body in reversed(range(expected.nbody)):
      subtree[body] += expected.body_mass[body]
      if body:
        subtree[expected.body_parentid[body]] += subtree[body]
    if simulator.device.startswith("cuda"):
      actual_subtree = simulator.subtreemass[index].cpu().numpy()
    else:
      actual_subtree = cast(Any, simulator.cpu_models[index]).body_subtreemass
    np.testing.assert_allclose(actual_subtree, subtree, rtol=1e-6, atol=1e-6)
    for field in ("body_invweight0", "dof_invweight0", "actuator_acc0"):
      if simulator.device.startswith("cuda"):
        actual = getattr(simulator.warp_model, field).numpy()[index]
      else:
        actual = getattr(simulator.cpu_models[index], field)
      np.testing.assert_allclose(
        actual, getattr(expected, field), rtol=3e-4, atol=2e-5, err_msg=field
      )
    if simulator.device.startswith("cuda"):
      actual_mean = simulator.warp_model.stat.meaninertia.numpy()[index]
    else:
      actual_mean = cast(Any, simulator.cpu_models[index]).stat.meaninertia
    np.testing.assert_allclose(
      actual_mean, cast(Any, expected).stat.meaninertia, rtol=3e-4, atol=2e-5
    )


def test_cpu_com_constants_preserve_reset_state(constant_robot: Q1Robot):
  torch.manual_seed(813)
  simulator = Q1Simulator(constant_robot, 2, "cpu")
  state = reset_states(constant_robot)
  simulator.reset(torch.arange(2), state, randomize=True)
  torch.testing.assert_close(simulator.state(), state, rtol=1e-6, atol=1e-6)
  assert_constants_match_cpu(simulator)
  assert not np.array_equal(
    cast(Any, simulator.cpu_models[0]).body_invweight0,
    cast(Any, constant_robot.model).body_invweight0,
  )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
@pytest.mark.parametrize("captured", [False, True])
def test_gpu_per_world_com_constants_and_partial_reset(
  constant_robot: Q1Robot, captured: bool
):
  torch.manual_seed(813)
  simulator = Q1Simulator(
    constant_robot.to("cuda:0"), 2, "cuda:0", capture_reset=captured
  )
  state = reset_states(constant_robot).to("cuda:0")
  simulator.reset(torch.arange(2, device="cuda:0"), state, randomize=True)
  torch.testing.assert_close(simulator.state(), state, rtol=1e-6, atol=1e-6)
  slices = simulator.robot.context_layout
  for span in (slices.root_com, slices.torso_added_mass, slices.kp_factor):
    assert not torch.equal(simulator.context[0, span], simulator.context[1, span])
  assert_constants_match_cpu(simulator)
  weights = simulator.warp_model.body_invweight0.numpy()
  assert not np.array_equal(weights[0], weights[1])
  before = simulator.state().clone()
  before_context = simulator.context.clone()
  simulator.reset(torch.tensor([0], device="cuda:0"), state[:1], randomize=True)
  torch.testing.assert_close(simulator.state()[1], before[1], rtol=0, atol=0)
  torch.testing.assert_close(simulator.context[1], before_context[1], rtol=0, atol=0)
  torch.testing.assert_close(simulator.state()[0], state[0], rtol=1e-6, atol=1e-6)
  assert_constants_match_cpu(simulator)
  # Return randomized worlds to the XML's nominal mass and COM, with no
  # stale constraint weights left behind.
  simulator.reset(torch.arange(2, device="cuda:0"), state, randomize=False)
  assert_constants_match_cpu(simulator)
  assert simulator.substeps == 20
  assert simulator.dt == 0.02


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_reset_without_extra_forward_preserves_next_physics_steps(
  constant_robot: Q1Robot,
):
  device = "cuda:0"
  first = Q1Simulator(constant_robot.to(device), 2, device, physics_timestep=0.005)
  second = Q1Simulator(
    Q1Robot(constant_robot.xml).to(device),
    2,
    device,
    physics_timestep=0.005,
    capture_reset=True,
  )
  state = reset_states(constant_robot).to(device)
  ids = torch.arange(2, device=device)
  first.reset(ids, state, randomize=False)
  second.reset(ids, state, randomize=False)
  with second._stream():
    mw.forward(second.warp_model, second.warp_data)
  actions = torch.linspace(-0.05, 0.05, 22, device=device)[None].expand(2, -1)
  for step in range(5):
    if step == 2:
      first.reset(ids[:1], state[:1], randomize=False)
      second.reset(ids[:1], state[:1], randomize=False)
      with second._stream():
        mw.forward(second.warp_model, second.warp_data)
    torch.testing.assert_close(
      first.step(actions), second.step(actions), rtol=1e-6, atol=1e-6
    )
