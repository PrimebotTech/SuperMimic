"""Numerical and gradient checks for the independent Q1 implementation."""

import json
import math
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import joblib
import mujoco
import numpy as np
import pytest
import torch
from conftest import Q1_JOINT_NAMES, link_name

from supermimic.algorithm import SuperMimic, gae_returns
from supermimic.collection import Q1Collector
from supermimic.data import Q1Motions, ReturnResetSampler
from supermimic.import_pkl import ImportCfg, import_pkl
from supermimic.models import Critic, MotionPolicy, WorldModel, frozen, goal_features
from supermimic.randomization import RandomizationCfg
from supermimic.replay import ReplayBatch, TemporalReplay
from supermimic.robot import Q1Robot, normalize_quat, quat_apply, quat_inverse, quat_mul
from supermimic.simulator import (
  Q1Simulator,
  torque_speed_limits,
  torque_speed_parameters,
)
from supermimic.train import TrainCfg, select_motions


@pytest.fixture
def robot(tmp_path: Path) -> Q1Robot:
  """Small synthetic Q1-shaped articulation, with nonzero hinge anchors."""
  tree = ET.Element("mujoco", model="q1_v2")
  ET.SubElement(tree, "compiler", angle="radian", autolimits="true")
  ET.SubElement(tree, "option", timestep="0.001", gravity="0 0 0")
  world = ET.SubElement(tree, "worldbody")
  root = ET.SubElement(world, "body", name="pelvis", pos="0 0 0.8")
  ET.SubElement(root, "freejoint")
  ET.SubElement(
    root, "geom", type="sphere", size="0.02", mass="2", contype="0", conaffinity="0"
  )
  actuators = ET.SubElement(tree, "actuator")
  body = root
  for j, joint in enumerate(Q1_JOINT_NAMES):
    body = ET.SubElement(
      root if j % 3 == 0 else body,
      "body",
      name=link_name(joint),
      pos="0.02 0.01 -0.03",
      quat="0.99 0.1 0 0",
    )
    ET.SubElement(
      body,
      "joint",
      name=joint,
      type="hinge",
      axis="0 1 0" if j % 2 else "1 0 0",
      pos="0.01 0.02 0.01",
      range="-2 2",
      armature="0.01",
    )
    ET.SubElement(
      body,
      "geom",
      type="sphere",
      size="0.01",
      mass="4" if link_name(joint) == "torso_link" else "0.1",
      contype="0",
      conaffinity="0",
    )
    ET.SubElement(actuators, "motor", joint=joint, ctrlrange="-20 20")
  path = tmp_path / "q1.xml"
  ET.ElementTree(tree).write(path)
  return Q1Robot(path)


def sample_state(robot: Q1Robot, count: int = 1) -> torch.Tensor:
  generator = torch.Generator().manual_seed(12)
  s = torch.randn(count, robot.state_dim, generator=generator) * 0.2
  s[:, 3:7] = normalize_quat(s[:, 3:7])
  return s


@torch.no_grad()
def test_compiled_collection_preserves_randomized_physics_and_history(robot: Q1Robot):
  actor = MotionPolicy(robot, (16, 16))
  compiled = Q1Collector(robot, actor, 0.02)
  compiled.enable_compile(backend="aot_eager")
  first = Q1Simulator(robot, 2, "cpu", physics_timestep=0.005)
  second = Q1Simulator(robot, 2, "cpu", physics_timestep=0.005)
  state = sample_state(robot, 2)
  state[:, :29] = torch.tensor(robot.model.qpos0)
  ids = torch.arange(2)
  torch.manual_seed(20261023)
  first.reset(ids, state)
  torch.manual_seed(20261023)
  second.reset(ids, state)
  target = sample_state(robot, 2)[:, None].repeat(1, 10, 1)
  history = robot.observation(first.state(), first.context)[:, None].repeat(1, 10, 1)
  actual_history = history.clone()
  for _ in range(5):
    before = first.state()
    action = actor(history, goal_features(robot, target))
    rng = torch.get_rng_state()
    expected_state = first.step(action)
    torch.set_rng_state(rng)
    actual_action = compiled.action(actual_history, target)
    actual_state = second.step(actual_action)
    torch.testing.assert_close(actual_action, action)
    torch.testing.assert_close(actual_state, expected_state)
    expected = (
      robot.reward(before, expected_state, target[:, 0], action, 0.02),
      robot.terminated(expected_state, target[:, 0]),
      robot.observation(expected_state, first.context),
    )
    actual = compiled.heads(before, actual_state, target[:, 0], action, second.context)
    for a, b in zip(actual, expected, strict=True):
      torch.testing.assert_close(a, b)
    noise = torch.randn_like(expected[2]) * robot.noise_scale()
    history = torch.cat((history[:, 1:], (expected[2] + noise)[:, None]), 1)
    actual_history = torch.cat((actual_history[:, 1:], (actual[2] + noise)[:, None]), 1)
    torch.testing.assert_close(actual_history, history)
  assert all(p.grad is None for p in actor.parameters())


def test_compiled_imagination_preserves_values_and_analytic_gradients(
  robot: Q1Robot, tmp_path: Path
):
  model = WorldModel(robot, (16, 16))
  actor = MotionPolicy(robot, (16, 16))
  critic = Critic(robot, (16, 16))
  algorithm = SuperMimic(
    robot, Q1Motions(robot, [tmp_path / "unused.npz"]), model, actor, critic
  )
  state = sample_state(robot, 2).requires_grad_()
  history = torch.randn(2, 10, robot.obs_dim, requires_grad=True)
  references = sample_state(robot, 20).reshape(2, 10, -1)
  context = torch.zeros(2, robot.context_dim)
  with frozen(model, critic):
    expected = algorithm.imagined_step(state, history, references, context)
    expected_grads = torch.autograd.grad(
      expected[1].sum() + expected[5].sum(),
      (state, history, *actor.parameters()),
    )
    algorithm.enable_compile(backend="aot_eager")
    actual = algorithm.imagined_step(state, history, references, context)
    actual_grads = torch.autograd.grad(
      actual[1].sum() + actual[5].sum(),
      (state, history, *actor.parameters()),
    )
  for a, b in zip(actual, expected, strict=True):
    torch.testing.assert_close(a, b)
  for a, b in zip(actual_grads, expected_grads, strict=True):
    torch.testing.assert_close(a, b)
  assert all(p.grad is None for p in model.parameters())
  assert all(p.grad is None for p in critic.parameters())


def test_fk_matches_mujoco_link_positions_and_velocities(robot: Q1Robot):
  state = sample_state(robot).double()
  robot.double()
  p, r, v, w = robot.kinematics(state)
  data = mujoco.MjData(robot.model)
  data.qpos[:] = state[0, :29].numpy()
  velocity = np.concatenate(
    (
      state[0, 29:32].numpy(),
      quat_apply(quat_inverse(state[:, 3:7]), state[:, 32:35])[0].numpy(),
      state[0, 35:57].numpy(),
    )
  )
  mujoco.mj_forward(robot.model, data)
  before_p, before_r = data.xpos.copy(), data.xquat.copy()
  np.testing.assert_allclose(p[0].numpy(), before_p, atol=1e-6)
  np.testing.assert_allclose(np.abs((r[0].numpy() * before_r).sum(-1)), 1, atol=1e-6)
  eps = 1e-6
  mujoco.mj_integratePos(robot.model, data.qpos, velocity, eps)
  mujoco.mj_forward(robot.model, data)
  np.testing.assert_allclose(v[0].numpy(), (data.xpos - before_p) / eps, atol=1e-5)
  # Rotation derivatives expressed in the world frame.
  next_q = torch.tensor(data.xquat)
  delta = quat_mul(next_q, quat_inverse(torch.tensor(before_r)))
  numerical_w = 2 * delta[:, 1:] / eps
  np.testing.assert_allclose(w[0].numpy(), numerical_w.numpy(), atol=1e-5)


def test_perfect_reward_and_pose_gradient(robot: Q1Robot):
  state = sample_state(robot, 2)
  action = state[:, 57:79]
  reward = robot.reward(state, state, state, action, 0.02)
  torch.testing.assert_close(reward, torch.full((2,), 0.1))
  displaced = state.clone()
  displaced[:, 7] += 0.2
  displaced.requires_grad_()
  robot.reward(state, displaced, state, action, 0.02).sum().backward()
  assert displaced.grad is not None
  assert torch.isfinite(displaced.grad).all()
  assert displaced.grad[:, 7].abs().sum() > 0


def test_reference_head_cache_preserves_reward_and_input_gradients(robot: Q1Robot):
  state = sample_state(robot, 3).requires_grad_()
  next_state = (state.detach() + 0.01).requires_grad_()
  action = torch.randn(3, 22, requires_grad=True)
  target = sample_state(robot, 3)
  heads = robot.kinematics(target)
  expected = robot.reward(state, next_state, target, action, 0.02)
  cached = robot.reward(state, next_state, target, action, 0.02, heads)
  inputs = (state, next_state, action)
  expected_grads = torch.autograd.grad(expected.sum(), inputs)
  cached_grads = torch.autograd.grad(cached.sum(), inputs)
  torch.testing.assert_close(cached, expected)
  for first, second in zip(cached_grads, expected_grads, strict=True):
    torch.testing.assert_close(first, second)


def test_batched_goal_encoding_preserves_actions_and_actor_gradients(robot: Q1Robot):
  actor = MotionPolicy(robot, (16, 16))
  history = torch.randn(2, 3, 10, robot.obs_dim, requires_grad=True)
  goals = torch.randn(2, 3, 10 * robot.physical_dim)
  expected = torch.stack([actor(history[:, t], goals[:, t]) for t in range(3)], 1)
  encoding = actor.encode_goal(goals.flatten(0, 1)).reshape(2, 3, -1)
  cached = torch.stack(
    [actor.decode(history[:, t], encoding[:, t]) for t in range(3)], 1
  )
  inputs = (history, *actor.parameters())
  expected_grads = torch.autograd.grad(expected.sum(), inputs)
  cached_grads = torch.autograd.grad(cached.sum(), inputs)
  torch.testing.assert_close(cached, expected)
  for first, second in zip(cached_grads, expected_grads, strict=True):
    torch.testing.assert_close(first, second)


def test_cached_rollout_preserves_complete_policy_update(
  robot: Q1Robot, tmp_path: Path
):
  qpos = np.repeat(robot.model.qpos0[None], 20, axis=0).astype(np.float32)
  path = tmp_path / "reference.npz"
  np.savez(
    path,
    qpos=qpos,
    robot_name="q1_v2",
    root_quat_order="wxyz",
    fps=50,
    robot_joint_names=np.array(robot.joint_names),
  )
  motions = Q1Motions(robot, [path])
  first = SuperMimic(
    robot,
    motions,
    WorldModel(robot, (16, 16)),
    MotionPolicy(robot, (16, 16)),
    Critic(robot, (16, 16)),
    horizon=4,
  )
  second = SuperMimic(
    robot,
    motions,
    WorldModel(robot, (16, 16)),
    MotionPolicy(robot, (16, 16)),
    Critic(robot, (16, 16)),
    horizon=4,
    cache_reference_heads=True,
    cache_goals=True,
  )
  for name in ("model", "actor", "critic"):
    getattr(second, name).load_state_dict(getattr(first, name).state_dict())
  states = motions.load(0)[:2]
  context = torch.zeros(2, robot.context_dim)
  batch = ReplayBatch(
    states[:, None],
    states[:, None],
    torch.zeros(2, 1, 22),
    context,
    torch.zeros(2, 1),
    robot.observation(states, context)[:, None].repeat(1, 10, 1),
    torch.zeros(2, dtype=torch.long),
    torch.arange(2),
  )
  torch.manual_seed(20261023)
  expected = first.improve_policy(batch, critic_epochs=1, critic_minibatches=2)
  second.enable_compile(backend="aot_eager")
  torch.manual_seed(20261023)
  actual = second.improve_policy(batch, critic_epochs=1, critic_minibatches=2)
  assert actual == pytest.approx(expected, rel=1e-4, abs=1e-6)
  for name in ("model", "actor", "critic"):
    for a, b in zip(
      getattr(first, name).parameters(), getattr(second, name).parameters(), strict=True
    ):
      torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
  assert all(p.grad is None for p in second.model.parameters())


def test_observation_excludes_root_position_and_linear_velocity(robot: Q1Robot):
  state = sample_state(robot, 2)
  context = torch.zeros(2, robot.context_dim)
  observation = robot.observation(state, context)
  changed = state.clone()
  changed[:, :3] += 100
  changed[:, 29:32] += 20
  torch.testing.assert_close(observation, robot.observation(changed, context))
  assert observation.shape == (2, 72)
  refs = state[:, None].repeat(1, 10, 1)
  assert goal_features(robot, refs).shape == (2, 570)


def test_replay_wrap_and_terminal_next_state(robot: Q1Robot):
  replay = TemporalReplay(7, 1, 79, 26, 72, "cpu")
  for step in range(10):
    replay.add(
      torch.full((1, 79), float(step)),
      torch.full((1, 79), float(step + 1)),
      torch.zeros(1, 22),
      torch.zeros(1, 26),
      torch.full((1, 72), float(step)),
      torch.tensor([float(step == 7)]),
      torch.tensor([step in (2, 7)]),
      torch.zeros(1, dtype=torch.long),
      torch.tensor([step]),
    )
  batch = replay.sample(128, window=2, history=3)
  assert set(batch.frames.tolist()).issubset({5, 6})
  assert (batch.history[:, 1:, 0] - batch.history[:, :-1, 0] == 1).all()
  assert (batch.next_states[:, :, 0] == batch.states[:, :, 0] + 1).all()
  assert (batch.terminated[:, -1] == 1).any()


def test_gae_bootstrap_and_terminal_mask():
  rewards = torch.tensor([[1.0, 2.0]])
  values = torch.tensor([[7.0, 8.0, 10.0]], requires_grad=True)
  result = gae_returns(rewards, torch.zeros_like(rewards), values, gamma=0.9, lam=1)
  torch.testing.assert_close(result, torch.tensor([[10.9, 11.0]]))
  terminal = gae_returns(rewards, torch.tensor([[0.0, 1.0]]), values, gamma=0.9, lam=1)
  torch.testing.assert_close(terminal, torch.tensor([[2.8, 2.0]]))
  assert not result.requires_grad


def test_frozen_world_model_preserves_action_jacobian(robot: Q1Robot):
  model = WorldModel(robot, (32, 32))
  state = sample_state(robot, 2)
  action = torch.randn(2, 22, requires_grad=True)
  with frozen(model):
    for _ in range(8):
      state = model(state, action, torch.zeros(2, robot.context_dim))
    state[:, :57].sum().backward()
  assert action.grad is not None and action.grad.abs().sum() > 0
  assert all(p.grad is None for p in model.parameters())
  assert all(p.requires_grad for p in model.parameters())


@pytest.mark.parametrize("fuse_model_loss", [False, True])
def test_model_loss_sums_normalized_features_and_retains_bce(
  robot: Q1Robot, tmp_path: Path, fuse_model_loss: bool
):
  qpos = np.zeros((20, 29), dtype=np.float32)
  qpos[:, 3] = 1
  path = tmp_path / "constant_reference.npz"
  np.savez(
    path,
    qpos=qpos,
    robot_name="q1_v2",
    root_quat_order="wxyz",
    fps=50,
    robot_joint_names=robot.joint_names,
  )
  motions = Q1Motions(robot, [path])
  model = WorldModel(robot, (8,))
  with torch.no_grad():
    for parameter in model.parameters():
      parameter.zero_()
    last = model.termination[-1]
    assert isinstance(last, torch.nn.Linear)
    last.bias.fill_(math.log(3))  # Constant P(termination) = 0.75.
  algorithm = SuperMimic(
    robot,
    motions,
    model,
    MotionPolicy(robot, (8,)),
    Critic(robot, (8,)),
    fuse_model_loss=fuse_model_loss,
  )
  if fuse_model_loss:
    algorithm.enable_compile(backend="aot_eager")
  states = torch.zeros(2, 2, robot.state_dim)
  states[:, :, 3] = 1
  targets = states.clone()
  # Metres, m/s and rad/s become [1,2,3], [2,1,4] in normalized state.
  targets[:, 0, [0, 29, 35]] = torch.tensor([[1.0, 6.0, 30.0], [2.0, 3.0, 40.0]])
  # The second step has normalized errors [0.5,1,2], [1.5,2,1].
  targets[:, 1, [0, 29, 35]] = torch.tensor([[0.5, 3.0, 20.0], [1.5, 6.0, 10.0]])
  batch = ReplayBatch(
    states=states,
    next_states=targets,
    actions=torch.zeros(2, 2, 22),
    context=torch.zeros(2, robot.context_dim),
    terminated=torch.tensor([[0.0, 1.0], [1.0, 1.0]]),
    history=torch.zeros(2, 10, robot.obs_dim),
    clips=torch.zeros(2, dtype=torch.long),
    frames=torch.zeros(2, dtype=torch.long),
  )
  metrics = algorithm.fit_model(batch)
  # Average the two samples, sum the two steps: 17.5 + 6.25 = 23.75.
  state_loss = 23.75
  termination_loss = -0.5 * math.log(0.25) - 1.5 * math.log(0.75)
  assert metrics["state_loss"] == pytest.approx(state_loss)
  assert metrics["termination_loss"] == pytest.approx(termination_loss)
  assert metrics["model_loss"] == pytest.approx(state_loss + termination_loss)


def test_fused_model_loss_preserves_bptt_gradients_and_optimizer_updates(
  robot: Q1Robot, tmp_path: Path
):
  path = tmp_path / "reference.npz"
  qpos = np.repeat(robot.model.qpos0[None], 32, axis=0).astype(np.float32)
  qpos[:, 7] = 0.15 * np.sin(np.arange(32) * 0.1)
  np.savez(
    path,
    qpos=qpos,
    robot_name="q1_v2",
    root_quat_order="wxyz",
    fps=50,
    robot_joint_names=robot.joint_names,
  )
  motions = Q1Motions(robot, [path])
  algorithms = [
    SuperMimic(
      robot,
      motions,
      WorldModel(robot, (16, 16)),
      MotionPolicy(robot, (16, 16)),
      Critic(robot, (16, 16)),
      fuse_model_loss=fused,
    )
    for fused in (False, True)
  ]
  algorithms[1].model.load_state_dict(algorithms[0].model.state_dict())
  for algorithm in algorithms:
    algorithm.enable_compile(backend="aot_eager")
  states = sample_state(robot, 16).reshape(2, 8, -1)
  labels = states.clone()
  labels[:, ::2, 3:7].neg_()
  labels[:, :, 7:29] += 0.03
  actions = torch.randn(2, 8, 22) * 0.1
  context = torch.randn(2, robot.context_dim) * 0.02
  for _ in range(2):
    results, gradients = [], []
    for algorithm in algorithms:
      inputs = [x.detach().clone().requires_grad_() for x in (states, actions, context)]
      batch = ReplayBatch(
        states=inputs[0],
        next_states=labels,
        actions=inputs[1],
        context=inputs[2],
        terminated=torch.tensor([[0.0, 1.0] * 4, [1.0, 0.0] * 4]),
        history=torch.zeros(2, 10, robot.obs_dim),
        clips=torch.zeros(2, dtype=torch.long),
        frames=torch.arange(2),
      )
      results.append(algorithm.fit_model(batch))
      assert all(x.grad is not None for x in inputs)
      gradients.append([x.grad for x in inputs])
    assert results[1] == pytest.approx(results[0], rel=1e-5, abs=1e-6)
    for first, second in zip(*gradients, strict=True):
      torch.testing.assert_close(first, second, rtol=1e-4, atol=1e-6)
    for first, second in zip(
      algorithms[0].model.parameters(), algorithms[1].model.parameters(), strict=True
    ):
      torch.testing.assert_close(first, second, rtol=1e-4, atol=1e-6)


def test_named_motion_reorders_joints_and_rejects_wrong_robot(
  robot: Q1Robot, tmp_path: Path
):
  names = list(reversed(robot.joint_names))
  qpos = np.zeros((30, 29), dtype=np.float32)
  qpos[:, 3] = 1
  qpos[:, 7:] = np.arange(22)
  path = tmp_path / "motion.npz"
  np.savez(
    path,
    qpos=qpos,
    robot_name="q1_v2",
    root_quat_order="wxyz",
    fps=50,
    robot_joint_names=names,
  )
  motions = Q1Motions(robot, [path])
  torch.testing.assert_close(motions.load(0)[0, 7:29], torch.arange(21, -1, -1).float())
  assert motions.lengths().tolist() == [30]
  wrong = tmp_path / "wrong.npz"
  np.savez(
    wrong,
    qpos=qpos,
    robot_name="qd1",
    root_quat_order="wxyz",
    fps=50,
    robot_joint_names=names,
  )
  with pytest.raises(ValueError, match="robot_name"):
    Q1Motions(robot, [wrong]).load(0)


def test_motion_selection_spreads_across_the_library_and_is_reproducible(
  tmp_path: Path,
):
  """A seeded draw must cover every corpus, and every rank must draw the same.

  Ranks pick their own clips independently, so a draw that depended on process
  state would have each GPU train on a different subset while reporting one
  selection in identity.json.
  """
  for corpus in ("bones", "cmu", "lafan1"):
    (tmp_path / corpus).mkdir()
    for index in range(40):
      (tmp_path / corpus / f"clip_{index:03d}.npz").touch()
  available = sorted(tmp_path.glob("*/*.npz"))
  assert len(available) == 120

  def select(
    selection: Literal["random", "first"], seed: int = 0, count: int = 12
  ) -> list[Path]:
    cfg = TrainCfg(
      robot_xml=tmp_path / "unused.xml",
      motion_dir=tmp_path,
      output_dir=tmp_path / "unused",
      motion_pattern="*/*.npz",
      max_motions=count,
      motion_selection=selection,
      motion_seed=seed,
    )
    return select_motions(cfg, available)

  drawn = select("random")
  assert len(drawn) == 12
  assert drawn == select("random"), "two ranks must agree on the selection"
  assert drawn == sorted(drawn), "the selection keeps the library's own order"
  assert {path.parent.name for path in drawn} == {"bones", "cmu", "lafan1"}
  assert select("random", seed=7) != drawn
  # The alphabetical prefix is what this guards against: twelve neighbouring
  # takes out of one corpus.
  assert {path.parent.name for path in select("first")} == {"bones"}
  # A limit at or above the library keeps everything, under either rule.
  assert select("random", count=120) == available
  assert select("random", count=0) == available


def test_motions_decode_once_and_bound_their_own_memory(robot: Q1Robot, tmp_path: Path):
  """Reference gathers must not re-decode: `sequence` touches every live clip.

  A residency cache smaller than the selected set re-decoded the whole set on
  every control step, because each step gathers from every clip the sampler
  picked. Decoding once up front removes the dependency on cache size.
  """
  paths = []
  for index in range(4):
    qpos = np.zeros((30, 29), dtype=np.float32)
    qpos[:, 3] = 1
    qpos[:, 7:] = index
    path = tmp_path / f"clip_{index}.npz"
    np.savez(
      path,
      qpos=qpos,
      robot_name="q1_v2",
      root_quat_order="wxyz",
      fps=50,
      robot_joint_names=robot.joint_names,
    )
    paths.append(path)
  motions = Q1Motions(robot, paths, cache_dir=tmp_path / "cache")
  decoded: list[int] = []
  original = Q1Motions._decode

  def counting_decode(self: Q1Motions, clip: int) -> torch.Tensor:
    decoded.append(clip)
    return original(self, clip)

  Q1Motions._decode = counting_decode  # type: ignore[method-assign]
  try:
    clips = torch.tensor([3, 1, 0, 2, 1])
    frames = torch.tensor([5, 0, 29, 10, 28])
    offsets = torch.tensor([-2, 0, 3])
    first = motions.sequence(clips, frames, offsets)
    second = motions.sequence(clips, frames, offsets)
  finally:
    Q1Motions._decode = original  # type: ignore[method-assign]
  assert decoded == [0, 1, 2, 3], "one decode per clip, no re-reads"
  torch.testing.assert_close(first, second)
  # Another rank/process instance reuses the CPU file without decoding again.
  reused = Q1Motions(robot, paths, cache_dir=tmp_path / "cache")
  Q1Motions._decode = counting_decode  # type: ignore[method-assign]
  try:
    torch.testing.assert_close(reused.sequence(clips, frames, offsets), first)
  finally:
    Q1Motions._decode = original  # type: ignore[method-assign]
  assert decoded == [0, 1, 2, 3]
  assert reused._states is not None
  assert reused._states.shape == (120, robot.physical_dim)
  assert not first[:, :, 57:].any()
  # Clamping happens at each clip's own end, and only after it.
  torch.testing.assert_close(first[:, :, 7], clips.float()[:, None].expand(5, 3))
  torch.testing.assert_close(
    motions.sequence(torch.tensor([0]), torch.tensor([29]), offsets)[0, :, 0],
    motions.load(0)[[27, 29, 29], 0],
  )
  # A selection too large to decode is refused with its own size, instead of
  # silently thrashing a cache at every step.
  budgeted = Q1Motions(robot, paths, memory_budget_gb=1e-6, storage="device")
  with pytest.raises(ValueError, match="above the 1e-06 GiB budget"):
    budgeted.prepare()
  limited = Q1Motions(
    robot, paths, memory_budget_gb=1e-6, cache_dir=tmp_path / "too-small"
  )
  with pytest.raises(ValueError, match="One clip exceeds"):
    limited.prepare()


def test_cpu_reset_preserves_velocity_and_applies_motor_order(robot: Q1Robot):
  sim = Q1Simulator(robot, 2, "cpu")
  state = sample_state(robot, 2)
  state[:, 57:] = 0
  sim.reset(torch.arange(2), state, randomize=False)
  torch.testing.assert_close(sim.state(), state, atol=1e-6, rtol=1e-6)
  next_state = sim.step(torch.ones(2, 22) * 0.1)
  assert torch.isfinite(next_state).all()
  assert not torch.equal(next_state[:, 7:29], state[:, 7:29])


def test_unrandomized_reset_disables_pushes_per_environment(robot: Q1Robot):
  actions = [torch.full((2, 22), 0.2), torch.full((2, 22), -0.3), torch.zeros(2, 22)]
  rollouts = []
  for seed in (0, 1234):
    simulator = Q1Simulator(robot, 2, "cpu")
    state = sample_state(robot, 2)
    state[:, 57:] = 0
    torch.manual_seed(seed)
    simulator.reset(torch.arange(2), state, randomize=False)
    rollouts.append(torch.stack([simulator.step(a).clone() for a in actions]))
  torch.testing.assert_close(rollouts[0], rollouts[1], atol=0, rtol=0)
  simulator = Q1Simulator(robot, 2, "cpu")
  state = sample_state(robot, 2)
  simulator.reset(torch.tensor([0]), state[:1], randomize=False)
  simulator.reset(torch.tensor([1]), state[1:], randomize=True)
  assert simulator.push_enabled.tolist() == [False, True]
  assert not simulator.lagged.any()


def test_q1_motor_curves_preserve_braking_and_follow_joint_names(robot: Q1Robot):
  peaks, knees, cutoffs = torque_speed_parameters(robot.joint_names, "cpu")
  assert int((peaks == 34).sum()) == 10
  assert int((peaks == 21).sum()) == 12
  velocity = torch.stack(
    (torch.zeros_like(peaks), (knees + cutoffs) / 2, cutoffs, -cutoffs)
  )
  lower, upper = torque_speed_limits(velocity, peaks, knees, cutoffs)
  torch.testing.assert_close(upper[0], peaks)
  torch.testing.assert_close(lower[0], -peaks)
  torch.testing.assert_close(upper[1], peaks / 2)
  torch.testing.assert_close(lower[1], -peaks)
  assert not upper[2].any() and not lower[3].any()
  torch.testing.assert_close(lower[2], -peaks)
  torch.testing.assert_close(upper[3], peaks)
  reversed_curves = torque_speed_parameters(tuple(reversed(robot.joint_names)), "cpu")
  for original, reordered in zip((peaks, knees, cutoffs), reversed_curves, strict=True):
    torch.testing.assert_close(original.flip(0), reordered)
  with pytest.raises(ValueError, match="unambiguous"):
    torque_speed_parameters(("unknown_joint",), "cpu")


def test_cpu_controller_applies_speed_limits_before_xml_limits(
  robot: Q1Robot, monkeypatch
):
  sim = Q1Simulator(robot, 1, "cpu", randomization=RandomizationCfg())
  assert sim.action_scale == 0.5 and sim.use_tn
  state = torch.zeros(1, robot.state_dim)
  state[:, :29] = torch.tensor(robot.model.qpos0, dtype=torch.float32)
  state[:, 35:57] = (sim.torque_knees + sim.torque_cutoffs) / 2
  sim.reset(torch.tensor([0]), state, randomize=False)
  controls = []

  def record_control(model, data):
    controls.append(data.ctrl.copy())

  # Isolate actuator clipping from the physical velocity changing during a step.
  monkeypatch.setattr(mujoco, "mj_step", record_control)
  sim.step(torch.full((1, 22), 8.0))
  expected = torch.where(sim.torque_peaks == 34, 17.0, 10.5).numpy()
  np.testing.assert_allclose(controls[-1], expected, rtol=1e-6)
  sim.use_tn = False
  sim.step(torch.full((1, 22), 8.0))
  np.testing.assert_allclose(controls[-1], robot.model.actuator_ctrlrange[:, 1])


def test_calibration_uses_same_episode_zero_for_observation_and_control(robot: Q1Robot):
  simulator = Q1Simulator(robot, 1, "cpu", randomization=RandomizationCfg())
  action = torch.linspace(-0.4, 0.4, 22)[None]
  offset = torch.linspace(-0.01, 0.01, 22)[None]
  state = torch.zeros(1, robot.state_dim)
  state[:, 3] = 1
  state[:, 2] = 0.8
  state[:, 7:29] = robot.default_q + offset + simulator.action_scale * action
  simulator.reset(torch.tensor([0]), state, randomize=False)
  simulator.context[:, robot.context_layout.motor_offset] = offset
  observed = robot.observation(state, simulator.context)
  torch.testing.assert_close(observed[:, :22], simulator.action_scale * action)
  # The synthetic robot has no gravity or contact: this matching setpoint must
  # produce zero torque and preserve the stationary articulation.
  next_state = simulator.step(action)
  torch.testing.assert_close(next_state[:, :57], state[:, :57], atol=1e-6, rtol=1e-6)
  nominal = robot.observation(state, torch.zeros_like(simulator.context))
  torch.testing.assert_close(observed[:, 22:], nominal[:, 22:])


def test_return_sampler_uses_valid_one_second_bins():
  sampler = ReturnResetSampler(torch.tensor([30, 120]), 0.02)
  clips, frames, bins = sampler.sample(100)
  assert ((frames >= 0) & (frames < sampler.lengths[clips] - 1)).all()
  sampler.update(bins[:3], torch.tensor([0.1, 0.2, 0.3]))
  assert sampler.counts.sum() == 3
  assert sampler.bin_ids(
    torch.tensor([0, 1, 1]), torch.tensor([0, 50, 118])
  ).tolist() == [0, 2, 3]


def test_reset_sampler_normalizes_by_attainable_return():
  """A short tail bin must not look worse than a long one it cannot compete with."""
  sampler = ReturnResetSampler(torch.tensor([400, 160]), 0.02)
  clips = torch.zeros(400, dtype=torch.long)
  frames = torch.arange(400)
  reachable = (400 - 1 - frames).clamp(min=1).float()
  torch.testing.assert_close(
    sampler.frame_ceiling(clips, frames), 0.1 * (1 - 0.99**reachable) / 0.01
  )
  # The ceiling belongs to the frame, not to the second-wide bin it falls in:
  # across the final second it drops by more than half.
  assert (
    sampler.frame_ceiling(clips, torch.tensor([398]))[0]
    < 0.5 * sampler.frame_ceiling(clips, torch.tensor([349]))[0]
  )
  # A policy that reaches every frame's own ceiling leaves the curriculum
  # uniform instead of piling its 40% share onto the clip tails.
  ceilings = sampler.frame_ceiling(clips, frames)
  sampler.update(sampler.bin_ids(clips, frames), ceilings / ceilings)
  hard = torch.softmax(-25 * sampler.normalized, dim=0)
  torch.testing.assert_close(hard, torch.full_like(hard, 1 / len(hard)))


def test_reset_sampler_does_not_punish_frames_near_a_clip_end():
  """Perfect tracking must not make the last second of a clip look hardest.

  Normalizing a frame's discounted return by the ceiling of its bin's *start*
  frame reports near-zero progress for frames near a clip end, where little
  reward remains to be earned. On a 500-frame clip that handed the final second
  over 99% of the hard-sampling mass even under a flawless policy.
  """
  sampler = ReturnResetSampler(torch.tensor([500]), 0.02)
  clips, frames = torch.zeros(499, dtype=torch.long), torch.arange(499)
  returns = sampler.frame_ceiling(clips, frames)
  bins = sampler.bin_ids(clips, frames)
  for _ in range(50):
    sampler.update(bins, returns / sampler.frame_ceiling(clips, frames))
  hard = torch.softmax(-25 * sampler.normalized, dim=0)
  assert hard[-1] < 1.01 / len(hard)
  # The old normalization is what this guards against: aggregating the same
  # flawless returns first and only then dividing by each bin's *start* ceiling
  # put almost the whole 40% hard share on the final bin.
  per_bin = torch.zeros(len(hard)).scatter_add_(0, bins, returns) / torch.zeros(
    len(hard)
  ).scatter_add_(0, bins, torch.ones_like(returns))
  bin_start_ceiling = sampler.frame_ceiling(clips[: len(hard)], sampler.starts)
  stale = torch.softmax(-25 * per_bin / bin_start_ceiling, dim=0)
  assert stale[-1] > 0.99


def test_reset_sampler_return_estimate_decays_towards_recent_returns():
  sampler = ReturnResetSampler(torch.tensor([200]), 0.02, momentum=0.9)
  optimistic = sampler.normalized[0].clone()
  triple = torch.zeros(3, dtype=torch.long)
  sampler.update(triple, torch.zeros(3))
  torch.testing.assert_close(sampler.normalized[0], optimistic * 0.9**3)
  assert sampler.counts[0] == 3
  one_at_a_time = ReturnResetSampler(torch.tensor([200]), 0.02, momentum=0.9)
  for _ in range(3):
    one_at_a_time.update(torch.zeros(1, dtype=torch.long), torch.zeros(1))
  torch.testing.assert_close(one_at_a_time.normalized[0], sampler.normalized[0])
  # Stale returns decay away: an all-time mean would stay pinned near zero.
  for _ in range(200):
    sampler.update(triple, torch.full((3,), 4.0))
  torch.testing.assert_close(sampler.normalized[0], torch.tensor(4.0))


def pickled_record(qpos: np.ndarray, names: Sequence[str], fps: float = 50.0) -> dict:
  """Mirror the source writer: XYZW root rotation plus explicit joint angles."""
  return {
    "root_trans_offset": qpos[:, :3].astype(np.float32),
    # pose_aa is deliberately wrong here: the importer must not read it.
    "pose_aa": np.full((len(qpos), 27, 3), 99.0, dtype=np.float32),
    "root_quat": qpos[:, 3:7][:, [1, 2, 3, 0]].astype(np.float32),
    "dof_pos": qpos[:, 7:].astype(np.float32),
    "joint_names": list(names),
    "robot_name": "q1_v2",
    "fps": fps,
    "metadata": {"root_quat_order": "xyzw", "dof_unit": "rad"},
  }


def test_pkl_import_round_trips_qpos_through_the_npz_contract(
  robot: Q1Robot, tmp_path: Path
):
  generator = torch.Generator().manual_seed(5)
  qpos = np.zeros((40, 29), dtype=np.float32)
  qpos[:, :3] = torch.randn(40, 3, generator=generator).numpy()
  # An asymmetric quaternion: a wrong XYZW/WXYZ permutation cannot pass.
  rotation = torch.randn(40, 4, generator=generator)
  qpos[:, 3:7] = normalize_quat(rotation).numpy()
  qpos[:, 7:] = torch.rand(40, 22, generator=generator).numpy()
  names = list(reversed(robot.joint_names))
  source = tmp_path / "clip.pkl"
  joblib.dump({"clip_a": pickled_record(qpos, names)}, source)
  output = tmp_path / "imported"
  manifest = json.loads(
    import_pkl(
      ImportCfg(robot_xml=robot.xml, output_dir=output, motion_paths=[source])
    ).read_text()
  )
  assert manifest["written_clips"] == 1 and manifest["failed_files"] == 0
  # The loader resolves the source joint order by name, so the reloaded state
  # must match the original qpos column for column.
  motion = Q1Motions(robot, [output / "clip_a.npz"]).load(0)
  expected = np.concatenate(
    (qpos[:, :7], qpos[:, [7 + names.index(n) for n in robot.joint_names]]), axis=-1
  )
  torch.testing.assert_close(
    motion[:, :29], torch.from_numpy(expected), atol=1e-6, rtol=1e-6
  )


def test_pkl_import_parallel_matches_serial_and_refuses_duplicate_keys(
  robot: Q1Robot, tmp_path: Path
):
  """Worker processes must not race, and must not overwrite each other's clips."""
  qpos = np.zeros((40, 29), dtype=np.float32)
  qpos[:, 3] = 1
  names = robot.joint_names
  sources = tmp_path / "sources"
  sources.mkdir()
  for index in range(6):
    joblib.dump(
      {f"clip_{index}": pickled_record(qpos, names)}, sources / f"motion_{index}.pkl"
    )
  # A seventh file reuses an earlier motion key, which is the one case where two
  # source files compete for a single output name.
  joblib.dump({"clip_0": pickled_record(qpos, names)}, sources / "motion_6.pkl")

  reports = []
  for workers in (1, 3):
    manifest = json.loads(
      import_pkl(
        ImportCfg(
          robot_xml=robot.xml,
          motion_dir=sources,
          output_dir=tmp_path / f"imported_{workers}",
          max_motions=0,
          workers=workers,
        )
      ).read_text()
    )
    assert manifest["written_clips"] == 6
    assert manifest["failed_files"] == 1
    failed = [row for row in manifest["files"] if not row["ok"]]
    assert "motion_6.pkl" in failed[0]["source"]
    assert "FileExistsError" in failed[0]["error"]
    reports.append([(row["source"], row["ok"]) for row in manifest["files"]])
  # Ordered results keep the manifest reproducible across worker counts.
  assert reports[0] == reports[1]


def test_pkl_import_rejects_padding_and_implicit_conventions(
  robot: Q1Robot, tmp_path: Path
):
  qpos = np.zeros((40, 29), dtype=np.float32)
  qpos[:, 3] = 1
  names = robot.joint_names

  # Repeated-pose padding is not a tracked reference.
  padded = pickled_record(qpos, names)
  padded["metadata"]["padded_from_frames"] = 3
  # Degrees, a missing quaternion convention and pose_aa-only records are all
  # refused instead of being guessed at.
  degrees = pickled_record(qpos, names)
  degrees["metadata"]["dof_unit"] = "deg"
  implicit = pickled_record(qpos, names)
  del implicit["metadata"]["root_quat_order"]
  pose_only = pickled_record(qpos, names)
  del pose_only["root_quat"]
  for index, record in enumerate((padded, degrees, implicit, pose_only)):
    source = tmp_path / f"bad{index}.pkl"
    joblib.dump({f"bad{index}": record}, source)
    with pytest.raises(ValueError, match="No clips imported"):
      import_pkl(
        ImportCfg(
          robot_xml=robot.xml,
          output_dir=tmp_path / f"out{index}",
          motion_paths=[source],
        )
      )
