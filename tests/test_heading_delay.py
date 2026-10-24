"""Physical invariance, actuator causality and compiled training integration."""

import math
from collections.abc import Generator
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch
from test_reward_training import environment as environment
from test_reward_training import training_config
from test_runtime_training import make_batch

from supermimic.algorithm import SuperMimic
from supermimic.data import Q1Motions
from supermimic.imagination_graph import ImaginationGraph
from supermimic.models import Critic, MotionPolicy, WorldModel
from supermimic.play import PlayCfg, play
from supermimic.randomization import nominal_context
from supermimic.robot import Q1Robot, normalize_quat, quat_apply, quat_mul
from supermimic.simulator import Q1Simulator
from supermimic.state_representation import HeadingState, quaternion_exp
from supermimic.train import train


@pytest.fixture(autouse=True)
def isolated_compiler_cache() -> Generator[None]:
  torch._dynamo.reset()
  yield
  torch._dynamo.reset()


@pytest.fixture
def delayed_environment(
  environment: tuple[Q1Robot, Q1Motions],
) -> tuple[Q1Robot, Q1Motions]:
  old, motions = environment
  robot = Q1Robot(old.xml, modeled_action_delay=True)
  return robot, Q1Motions(robot, motions.paths)


def rotate_world(state: torch.Tensor) -> torch.Tensor:
  yaw = state.new_tensor([math.cos(0.7), 0, 0, math.sin(0.7)]).expand(len(state), -1)
  moved = state.clone()
  moved[:, :3] = quat_apply(yaw, state[:, :3]) + state.new_tensor([5, -3, 0])
  moved[:, 3:7] = quat_mul(yaw, state[:, 3:7])
  moved[:, 29:32] = quat_apply(yaw, state[:, 29:32])
  moved[:, 32:35] = quat_apply(yaw, state[:, 32:35])
  return moved


def test_heading_features_and_learned_dynamics_are_world_yaw_equivariant(
  delayed_environment: tuple[Q1Robot, Q1Motions],
):
  robot, _ = delayed_environment
  robot.double()
  torch.manual_seed(21)
  state = torch.randn(4, 79, dtype=torch.float64) * 0.2
  state[:, 3:7] = normalize_quat(state[:, 3:7])
  moved = rotate_world(state)
  features = HeadingState().double()
  torch.testing.assert_close(features.features(moved), features.features(state))
  model = WorldModel(
    robot, (16, 16), activation="silu", representation="heading"
  ).double()
  # A zero dynamics output would make the equivariance assertion trivial.
  last = model.dynamics[-1]
  assert isinstance(last, torch.nn.Linear)
  with torch.no_grad():
    torch.nn.init.normal_(last.weight, std=0.03)
  action = torch.randn(4, 22, dtype=torch.float64) * 0.1
  context = nominal_context(robot.context_layout, 4).double()
  torch.testing.assert_close(
    model(moved, action, context), rotate_world(model(state, action, context))
  )
  opposite = state.clone()
  opposite[:, 3:7].neg_()
  torch.testing.assert_close(features.features(opposite), features.features(state))


def test_zero_initialized_model_integrates_velocity_and_preserves_finite_jacobians(
  delayed_environment: tuple[Q1Robot, Q1Motions],
):
  robot, _ = delayed_environment
  model = WorldModel(robot, (8,), activation="silu", representation="heading").double()
  state = torch.zeros(1, 79, dtype=torch.float64)
  state[:, 3] = 1
  state[:, 29:32] = state.new_tensor([0.3, -0.2, 0.1])
  state[:, 32:35] = state.new_tensor([0.0, 0.0, 0.4])
  state[:, 35:57] = 0.2
  action = torch.randn(1, 22, dtype=torch.float64) * 0.1
  context = nominal_context(robot.context_layout, 1).double()
  successor = model(state, action, context)
  torch.testing.assert_close(successor[:, :3], state[:, :3] + 0.02 * state[:, 29:32])
  torch.testing.assert_close(
    successor[:, 7:29], state[:, 7:29] + 0.02 * state[:, 35:57]
  )
  torch.testing.assert_close(successor[:, 3:7], quaternion_exp(state[:, 32:35] * 0.02))
  torch.testing.assert_close(successor[:, 57:79], action)
  assert torch.autograd.gradcheck(
    lambda s: model(s, action, context), (state.requires_grad_(),)
  )
  assert model.heading_state is not None
  label = state.detach().clone()
  label[:, 3:7].neg_()
  loss = model.heading_state.error(state, label).square().sum()
  gradient = torch.autograd.grad(loss, state)[0]
  assert torch.isfinite(gradient).all()
  torch.testing.assert_close(loss, loss.new_zeros(()))


def test_fixed_delay_holds_previous_noisy_command_and_reset_clears_memory(
  delayed_environment: tuple[Q1Robot, Q1Motions],
  monkeypatch: pytest.MonkeyPatch,
):
  robot, motions = delayed_environment
  simulator = Q1Simulator(robot, 1, "cpu", physics_timestep=0.002, use_tn=False)
  neutral = motions.load(0)[:1].clone()
  neutral[:, :29] = torch.tensor(robot.model.qpos0)
  neutral[:, 29:] = 0
  simulator.reset(torch.tensor([0]), neutral, randomize=False)
  simulator.context[:, robot.context_layout.delay_s] = 0.016
  controls = []
  monkeypatch.setattr(mujoco, "mj_step", lambda m, d: controls.append(d.ctrl.copy()))
  first = torch.full((1, 22), 0.1)
  simulator.step(first)
  np.testing.assert_allclose(np.array(controls[:8]), 0, atol=1e-6)
  assert np.abs(controls[8]).sum() > 0
  assert len(controls) == 10
  torch.testing.assert_close(
    simulator.context[:, robot.context_layout.previous_executed_action], first
  )
  simulator.step(-first)
  for old, held in zip(controls[8:10] * 4, controls[10:18], strict=True):
    np.testing.assert_allclose(old, held, atol=1e-6)
  assert np.sign(controls[18][0]) == -np.sign(controls[8][0])
  torch.testing.assert_close(
    simulator.context[:, robot.context_layout.delay_s], torch.tensor([[0.016]])
  )
  simulator.reset(torch.tensor([0]), motions.load(0)[:1], randomize=False)
  torch.testing.assert_close(simulator.context[:, -23:], torch.zeros(1, 23))
  assert not simulator.last_action.any()


def test_tracking_randomization_has_no_random_delay_or_privileged_actor_input(
  delayed_environment: tuple[Q1Robot, Q1Motions],
):
  robot, motions = delayed_environment
  simulator = Q1Simulator(robot, 64, "cpu", physics_timestep=0.002)
  simulator.reset(torch.arange(64), motions.load(0)[:1].expand(64, -1), randomize=True)
  delay = simulator.context[:, robot.context_layout.delay_s].clone()
  ticks = delay / 0.002
  torch.testing.assert_close(ticks, ticks.round())
  assert ticks.min() >= 0 and ticks.max() <= 8
  assert not ticks.any()
  before = robot.observation(simulator.state(), simulator.context)
  changed = simulator.context.clone()
  changed[:, robot.context_layout.delay_s] = 0
  changed[:, robot.context_layout.previous_executed_action] = 0.9
  torch.testing.assert_close(robot.observation(simulator.state(), changed), before)
  assert before.shape == (64, 72)
  simulator.step(torch.zeros(64, 22))
  torch.testing.assert_close(simulator.context[:, robot.context_layout.delay_s], delay)


def delayed_learner(robot: Q1Robot, motions: Q1Motions) -> SuperMimic:
  return SuperMimic(
    robot,
    motions,
    WorldModel(robot, (16, 16), activation="silu", representation="heading"),
    MotionPolicy(robot, (16, 16), activation="silu"),
    Critic(robot, (16, 16), activation="silu"),
    horizon=4,
    clean_policy_actions=True,
    normalize_policy_objective=True,
    normalize_model_objective=True,
    cache_reference_heads=True,
    cache_goals=True,
    fuse_model_loss=True,
  )


def test_delay_context_advances_with_analytic_gradients_and_matches_compiled_updates(
  delayed_environment: tuple[Q1Robot, Q1Motions],
):
  robot, motions = delayed_environment
  learners = []
  for compiled in (False, True):
    torch.manual_seed(112)
    learner = delayed_learner(robot, motions)
    if compiled:
      learner.enable_compile(backend="aot_eager")
    learners.append(learner)
  batch = make_batch(robot, motions, 3)
  batch.context[:, robot.context_layout.delay_s] = 0.008
  for update in range(2):
    for learner in learners:
      learner.fit_model_tensors(batch)
      torch.manual_seed(34 + update)
      learner.improve_policy(batch, 1, 2)
    for name in ("actor", "model", "critic"):
      for key, value in getattr(learners[0], name).state_dict().items():
        torch.testing.assert_close(
          value, getattr(learners[1], name).state_dict()[key], rtol=1e-4, atol=3e-6
        )
  action = torch.randn(3, 22, requires_grad=True)
  context = learners[0].model.advance_context(batch.context, action)
  torch.testing.assert_close(
    torch.autograd.grad(context[:, -22:].sum(), action)[0], torch.ones_like(action)
  )
  graph = ImaginationGraph(learners[0], lambda r, d, v: r.detach())
  output = graph.run(batch)
  torch.testing.assert_close(output.context[:, -22:], output.states[:, 57:79])
  assert any(
    p.grad is not None and p.grad.abs().sum() > 0
    for p in learners[0].actor.parameters()
  )
  assert all(p.grad is None for p in learners[0].model.parameters())


def test_fresh_heading_delay_checkpoint_preserves_pre_action_replay_context(
  environment: tuple[Q1Robot, Q1Motions],
  tmp_path: Path,
):
  robot, motions = environment
  cfg = replace(
    training_config(robot, motions, tmp_path / "aligned"),
    iterations=2,
    physics_timestep=0.002,
    modeled_action_delay=True,
    action_lag=False,
    clean_policy_actions=True,
    network_activation="silu",
    dynamics_representation="heading",
    normalize_policy_objective=True,
    normalize_model_objective=True,
  )
  path = train(cfg)
  checkpoint = torch.load(path, weights_only=True)
  assert checkpoint["identity"]["network_architecture"] == {
    "activation": "silu",
    "representation": "heading",
    "modeled_action_delay": True,
  }
  shard = path.parent / checkpoint["rank_directory"] / "rank_0.pt"
  replay = torch.load(shard, weights_only=True)["replay"]
  torch.testing.assert_close(replay["context"][..., -22:], replay["states"][..., 57:79])
  assert checkpoint["identity"]["context_dim"] == robot.context_dim + 23
  result = play(
    PlayCfg(
      checkpoint=path,
      motion=motions.paths[0],
      output=tmp_path / "played.npz",
      viewer="off",
    )
  )
  assert result["played_steps"] > 0
  with pytest.raises(ValueError, match="representation changed"):
    train(
      replace(
        cfg,
        output_dir=tmp_path / "bad_resume",
        resume=cfg.output_dir / "checkpoint_2.pt",
        network_activation="elu",
      )
    )
