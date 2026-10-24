"""Extinct imagined branches cannot supply exploding critic targets."""

import pytest
import torch
from test_reward_training import environment as environment
from test_reward_training import small_learner
from test_runtime_training import assert_state_equal, make_batch
from torch import Tensor

from supermimic.algorithm import gae_returns
from supermimic.data import Q1Motions
from supermimic.imagination_graph import ImaginationGraph
from supermimic.imagination_terminal import MIN_SURVIVAL, absorbing_inputs
from supermimic.reward_tuning import RewardTuningCfg
from supermimic.robot import Q1Robot


def test_absorption_keeps_live_gradients_and_resets_extinct_inputs():
  state = torch.tensor([[2.0], [1e30]], requires_grad=True)
  history = state[:, None]
  context = state * 2
  initial = torch.ones_like(state)
  survival = torch.tensor([0.2, MIN_SURVIVAL / 2], requires_grad=True)
  x, hist, ctx, active = absorbing_inputs(
    survival, state, history, context, initial, initial[:, None], initial
  )
  assert torch.equal(active, torch.tensor([True, False]))
  assert x.tolist() == [[2.0], [1.0]]
  assert hist[:, 0].tolist() == [[2.0], [1.0]]
  assert ctx.tolist() == [[4.0], [1.0]]
  (x.sum() + hist.sum() + ctx.sum()).backward()
  assert state.grad is not None and state.grad.tolist() == [[4.0], [0.0]]


def test_critic_weights_probability_mass_without_rank_local_normalization(
  environment: tuple[Q1Robot, Q1Motions],
):
  robot, motions = environment
  learner = small_learner(robot, motions, RewardTuningCfg())
  batch = make_batch(robot, motions, 1)
  x, ctx = batch.states[:, 0], batch.context
  g = learner.reference_goals(
    motions.sequence(batch.clips, batch.frames, torch.arange(43))
  )[:, 0]
  prediction = learner.critic(x, g, ctx)
  target = prediction.detach() + torch.tensor([1.0, 2.0, float("inf")])
  weight = torch.tensor([1.0, 0.25, 0.0])
  loss = learner._critic_loss_step(x, g, ctx, target, weight)
  assert float(loss.detach()) == pytest.approx(2 / 3)
  loss.backward()
  assert all(
    p.grad is not None and torch.isfinite(p.grad).all()
    for p in learner.critic.parameters()
  )
  full_grad = []
  for p in learner.critic.parameters():
    assert p.grad is not None
    full_grad.append(p.grad.clone())
  learner.critic.zero_grad(set_to_none=True)
  # Fixed-size rank partitions and averaging produce the same gradient.
  for i in range(3):
    local = learner._critic_loss_step(
      x[i : i + 1],
      g[i : i + 1],
      ctx[i : i + 1],
      target[i : i + 1],
      weight[i : i + 1],
    )
    (local / 3).backward()
  for p, expected in zip(learner.critic.parameters(), full_grad, strict=True):
    assert p.grad is not None
    torch.testing.assert_close(p.grad, expected, atol=1e-7, rtol=1e-5)


def test_terminated_rollout_does_not_reenter_exploding_terminal_state(
  environment: tuple[Q1Robot, Q1Motions], monkeypatch: pytest.MonkeyPatch
):
  robot, motions = environment
  learner = small_learner(robot, motions, RewardTuningCfg())
  learner.horizon = 32
  graph = ImaginationGraph(learner, gae_returns)
  batch = make_batch(robot, motions, 1)
  inputs = graph.prepare(batch)
  visited: list[Tensor] = []
  original = learner.imagined_step

  def terminal_step(state, history, references, context, *args):
    visited.append(state.detach().clone())
    successor, reward, termination, observation, goal, value = original(
      state, history, references, context, *args
    )
    # Simulate a terminal successor outside the model's fitted distribution.
    successor = torch.cat(
      (successor[:, :35], successor[:, 35:57] * 1e25, successor[:, 57:]), -1
    )
    return successor, reward, torch.ones_like(termination), observation, goal, value

  monkeypatch.setattr(learner, "imagined_step", terminal_step)
  output = graph.forward_backward(inputs)
  assert len(visited) == 32
  for state in visited:
    assert torch.equal(state, inputs.state)
  weights = output.weights.reshape(len(inputs.state), 32)
  assert (weights[:, 0] == 1).all() and (weights[:, 1:] == 0).all()
  assert (output.rewards[:, 1:] == 0).all()
  assert (output.targets.reshape_as(weights)[:, 1:] == 0).all()
  assert torch.isfinite(output.targets).all()
  loss = learner._fit_critic(
    output.states,
    output.goals,
    output.context,
    output.targets,
    output.weights,
    1,
    1,
  )
  assert torch.isfinite(loss)
  assert all(
    p.grad is not None and torch.isfinite(p.grad).all()
    for p in learner.actor.parameters()
  )


def test_eager_and_graph_updates_match_when_survival_becomes_extinct(
  environment: tuple[Q1Robot, Q1Motions], monkeypatch: pytest.MonkeyPatch
):
  robot, motions = environment
  learners = []
  for captured in (False, True):
    torch.manual_seed(387)
    learner = small_learner(robot, motions, RewardTuningCfg())
    learner.horizon = 32
    # Probability reaches the absorption threshold after six transitions.
    monkeypatch.setattr(
      learner.model,
      "termination_logits",
      lambda robot, state, target: state.new_full((len(state),), 3.0),
    )
    if captured:
      learner.imagination_graph = ImaginationGraph(learner, gae_returns)
    learners.append(learner)
  batch = make_batch(robot, motions, 1)
  metrics = []
  for learner in learners:
    torch.manual_seed(731)
    result = (
      learner.improve_policy(batch, 2, 2)
      if learner.imagination_graph is None
      else learner._improve_policy_captured(batch, 2, 2)
    )
    metrics.append(result)
  assert_state_equal(metrics[0], metrics[1])
  assert metrics[0]["imagination_survival_fraction"] < 0.04
  for name in ("actor", "critic"):
    assert_state_equal(
      getattr(learners[0], name).state_dict(),
      getattr(learners[1], name).state_dict(),
    )
