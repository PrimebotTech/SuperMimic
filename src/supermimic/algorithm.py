"""Autoregressive model fitting and analytic short-horizon actor-critic."""

import torch
import torch.nn.functional as F
from torch import Tensor

from supermimic.compile_runtime import configure_compiler
from supermimic.data import Q1Motions
from supermimic.distributed import Distributed
from supermimic.imagination_graph import ImaginationGraph
from supermimic.imagination_terminal import (
  absorbing_inputs,
  surviving_reward_mean,
)
from supermimic.models import (
  Critic,
  MotionPolicy,
  WorldModel,
  frozen,
  goal_features,
)
from supermimic.replay import ReplayBatch
from supermimic.reward_tuning import BASELINE_TUNING, RewardTuningCfg, tuned_reward
from supermimic.robot import Q1Robot


def gae_returns(
  rewards: Tensor,
  termination: Tensor,
  values: Tensor,
  gamma: float = 0.99,
  lam: float = 0.95,
) -> Tensor:
  """Equation 3; all targets are detached once, without an EMA critic."""
  result = []
  future = values[:, -1].detach()
  for t in reversed(range(rewards.shape[1])):
    future = rewards[:, t].detach() + gamma * (1 - termination[:, t].detach()) * (
      (1 - lam) * values[:, t + 1].detach() + lam * future
    )
    result.append(future)
  return torch.stack(result[::-1], dim=1)


class SuperMimic:
  def __init__(
    self,
    robot: Q1Robot,
    motions: Q1Motions,
    model: WorldModel,
    actor: MotionPolicy,
    critic: Critic,
    learning_rate: float = 2e-4,
    horizon: int = 32,
    distributed: Distributed | None = None,
    cache_reference_heads: bool = False,
    cache_goals: bool = False,
    fuse_model_loss: bool = False,
    reward_tuning: RewardTuningCfg = BASELINE_TUNING,
    clean_policy_actions: bool = False,
    normalize_policy_objective: bool = False,
    normalize_model_objective: bool = False,
    capture_imagination: bool = False,
    compile_critic: bool = False,
    fused_optimizers: bool = False,
  ):
    self.robot, self.motions = robot, motions
    self.model, self.actor, self.critic = model, actor, critic
    self.horizon = horizon
    self.reward_tuning = reward_tuning
    self.clean_policy_actions = clean_policy_actions
    self.normalize_policy_objective = normalize_policy_objective
    self.normalize_model_objective = normalize_model_objective
    self.cache_reference_heads = cache_reference_heads
    self.reference_kinematics = robot.kinematics
    self.cache_goals = cache_goals
    self.fuse_model_loss = fuse_model_loss
    self.compile_critic = compile_critic
    self.critic_loss_step = self._critic_loss_step
    self.encode_goal = actor.encode_goal
    self.reference_goals = self._reference_goals
    # Gradients are averaged across ranks before each clip, so the clip acts on
    # the global gradient; a lone process reduces nothing.
    self.distributed = distributed or Distributed.detached()
    self.gamma = 0.99
    fused = True if fused_optimizers and next(model.parameters()).is_cuda else None
    self.model_optimizer = torch.optim.Adam(
      model.parameters(), lr=learning_rate, fused=fused
    )
    self.actor_optimizer = torch.optim.Adam(
      actor.parameters(), lr=learning_rate, betas=(0.9, 0.95), fused=fused
    )
    self.critic_optimizer = torch.optim.Adam(
      critic.parameters(), lr=learning_rate, fused=fused
    )
    self.imagined_step = self._imagined_step
    self.model_step = self._model_step
    self.model_loss_step = self._model_loss_step
    self.compiled_backend: str | None = None
    self.imagination_graph = (
      ImaginationGraph(self, gae_returns) if capture_imagination else None
    )

  def enable_compile(self, backend: str = "inductor", max_fusion_size: int = 8) -> None:
    """Fuse tensor work without compiling optimizers or distributed collectives."""
    if backend == "inductor":
      configure_compiler()
    options: dict[str, str | int | bool] = (
      {"triton.cudagraphs": False}
      if backend == "inductor" and self.imagination_graph is not None
      else {}
    )
    self.imagined_step = torch.compile(
      self._imagined_step,
      backend=backend,
      dynamic=False,
      fullgraph=True,
      options={"max_fusion_size": max_fusion_size, **options}
      if backend == "inductor"
      else None,
    )
    self.model_step = torch.compile(
      self._model_step,
      backend=backend,
      dynamic=False,
      fullgraph=True,
      options=options or None,
    )
    if self.fuse_model_loss:
      self.model_loss_step = torch.compile(
        self._model_loss_step,
        backend=backend,
        dynamic=False,
        fullgraph=True,
        options=options or None,
      )
    if self.cache_reference_heads:
      self.reference_kinematics = torch.compile(
        self.robot.kinematics,
        backend=backend,
        dynamic=False,
        fullgraph=True,
        options=options or None,
      )
    if self.cache_goals:
      self.encode_goal = torch.compile(
        self.actor.encode_goal,
        backend=backend,
        dynamic=False,
        fullgraph=True,
        options=options or None,
      )
      self.reference_goals = torch.compile(
        self._reference_goals,
        backend=backend,
        dynamic=False,
        fullgraph=True,
        options=options or None,
      )
    if self.compile_critic:
      self.critic_loss_step = torch.compile(
        self._critic_loss_step,
        backend=backend,
        dynamic=False,
        fullgraph=True,
        options=options or None,
      )
    self.compiled_backend = backend

  def _reference_goals(self, references: Tensor) -> Tensor:
    windows = references.unfold(1, 10, 1)[:, 1 : self.horizon + 1].permute(0, 1, 3, 2)
    flat = windows.reshape(-1, 10, self.robot.state_dim)
    return goal_features(self.robot, flat).reshape(len(references), self.horizon, -1)

  def _model_step(
    self, state: Tensor, action: Tensor, context: Tensor, target: Tensor
  ) -> tuple[Tensor, Tensor]:
    next_state = self.model(state, action, context)
    return next_state, self.model.termination_logits(self.robot, next_state, target)

  def _imagined_step(
    self,
    state: Tensor,
    history: Tensor,
    references: Tensor,
    context: Tensor,
    target_heads: tuple[Tensor, Tensor, Tensor, Tensor] | None = None,
    cached_goal: Tensor | None = None,
    encoding: Tensor | None = None,
    active: Tensor | None = None,
  ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    goal = goal_features(self.robot, references) if cached_goal is None else cached_goal
    value = self.critic(state, goal, context)
    action = (
      self.actor(history, goal)
      if encoding is None
      else self.actor.decode(history, encoding)
    )
    next_state = self.model(state, action, context)
    reward_state = (
      torch.cat((state[..., :57], history[..., -1, -22:]), -1)
      if self.clean_policy_actions
      else state
    )
    reward = tuned_reward(
      self.robot,
      reward_state,
      next_state,
      references[:, 0],
      action,
      self.motions.dt,
      tuning=self.reward_tuning,
      context=context,
      target_heads=target_heads,
    )
    termination = self.model.termination_logits(
      self.robot, next_state, references[:, 0]
    ).sigmoid()
    observation = self.robot.observation(next_state, context)
    if active is not None:
      reward = torch.where(active, reward, 0)
      termination = torch.where(active, termination, 1)
      value = torch.where(active, value, 0)
    return next_state, reward, termination, observation, goal, value

  def _model_losses(
    self, state: Tensor, logits: Tensor, label: Tensor, terminated: Tensor
  ) -> tuple[Tensor, Tensor]:
    if self.model.heading_state is not None:
      error = self.model.heading_state.error(state, label)
      return error.square().sum(-1).mean(), F.binary_cross_entropy_with_logits(
        logits, terminated
      )
    # Match quaternion hemisphere before the Euclidean state loss.
    rotation = torch.where(
      (state[:, 3:7] * label[:, 3:7]).sum(-1, keepdim=True) < 0,
      -label[:, 3:7],
      label[:, 3:7],
    )
    label = torch.cat((label[:, :3], rotation, label[:, 7:]), -1)
    error = (state[:, :57] - label[:, :57]) / self.model.scale[:57]
    # Eq. 1 sums squared features; the replay expectation averages samples.
    return error.square().sum(-1).mean(), F.binary_cross_entropy_with_logits(
      logits, terminated
    )

  def _model_loss_step(
    self,
    state: Tensor,
    action: Tensor,
    context: Tensor,
    target: Tensor,
    label: Tensor,
    terminated: Tensor,
  ) -> tuple[Tensor, Tensor, Tensor]:
    state, logits = self._model_step(state, action, context, target)
    state_loss, term_loss = self._model_losses(state, logits, label, terminated)
    return state, state_loss, term_loss

  def fit_model(self, batch: ReplayBatch) -> dict[str, float]:
    return {name: float(value) for name, value in self.fit_model_tensors(batch).items()}

  def fit_model_tensors(self, batch: ReplayBatch) -> dict[str, Tensor]:
    """Keep diagnostics on device until the complete model update ends."""
    state = batch.states[:, 0]
    context = batch.context
    targets = self.motions.sequence(
      batch.clips,
      batch.frames,
      torch.arange(1, batch.actions.shape[1] + 1, device=batch.frames.device),
    )
    state_losses, term_losses = [], []
    self.model_optimizer.zero_grad(set_to_none=True)
    for t in range(batch.actions.shape[1]):
      if self.fuse_model_loss:
        state, state_loss, term_loss = self.model_loss_step(
          state,
          batch.actions[:, t],
          context,
          targets[:, t],
          batch.next_states[:, t],
          batch.terminated[:, t],
        )
      else:
        state, logits = self.model_step(
          state, batch.actions[:, t], context, targets[:, t]
        )
        state_loss, term_loss = self._model_losses(
          state, logits, batch.next_states[:, t], batch.terminated[:, t]
        )
      state_losses.append(state_loss)
      term_losses.append(term_loss)
      context = self.model.advance_context(context, batch.actions[:, t])
    state_loss = torch.stack(state_losses).sum()
    term_loss = torch.stack(term_losses).sum()
    if self.normalize_model_objective:
      state_loss = state_loss / batch.actions.shape[1]
      term_loss = term_loss / batch.actions.shape[1]
    loss = state_loss + term_loss
    self.distributed.require_finite(loss, "Nonfinite world-model loss")
    loss.backward()
    self.distributed.average_gradients(self.model.parameters())
    model_norm = torch.nn.utils.clip_grad_norm_(
      self.model.parameters(),
      1.0,
      error_if_nonfinite=self.distributed.validation_mode == "per_minibatch",
    )
    self.distributed.require_finite(model_norm, "Nonfinite world-model gradient norm")
    self.model_optimizer.step()
    return {
      "model_loss": loss.detach(),
      "state_loss": state_loss.detach(),
      "termination_loss": term_loss.detach(),
    }

  def improve_policy(
    self, batch: ReplayBatch, critic_epochs: int = 5, critic_minibatches: int = 4
  ) -> dict[str, float]:
    if self.imagination_graph is not None and batch.states.is_cuda:
      return self._improve_policy_captured(batch, critic_epochs, critic_minibatches)
    h = self.horizon
    references = self.motions.sequence(
      batch.clips, batch.frames, torch.arange(h + 11, device=batch.frames.device)
    )
    remaining = self.motions.remaining(batch.clips, batch.frames)
    reference_heads = None
    cached_goals, encodings = None, None
    if self.cache_goals:
      cached_goals = self.reference_goals(references)
      encodings = self.encode_goal(cached_goals.flatten(0, 1)).reshape(
        len(references), h, -1
      )
    if self.cache_reference_heads:
      with torch.no_grad():
        flat = references[:, 1 : h + 1].reshape(-1, self.robot.state_dim)
        reference_heads = tuple(
          head.reshape(len(references), h, *head.shape[1:])
          for head in self.reference_kinematics(flat)
        )
    state, history = batch.states[:, 0].detach(), batch.history.detach()
    context = batch.context
    initial_state, initial_history, initial_context = state, history, context
    survival = state.new_ones(len(state))
    objective = state.new_zeros(len(state))
    states, goals, rewards, terminations, values = [], [], [], [], []
    contexts = []
    weights = []
    self.actor_optimizer.zero_grad(set_to_none=True)
    self.model_optimizer.zero_grad(set_to_none=True)
    self.critic_optimizer.zero_grad(set_to_none=True)
    with frozen(self.model, self.critic):
      for t in range(h):
        state, history, context, active = absorbing_inputs(
          survival,
          state,
          history,
          context,
          initial_state,
          initial_history,
          initial_context,
        )
        weights.append(torch.where(active, survival.detach(), 0))
        next_state, reward, termination, observation, goal, value = self.imagined_step(
          state,
          history,
          references[:, t + 1 : t + (2 if self.cache_goals else 11)].contiguous(),
          context,
          (
            reference_heads[0][:, t].contiguous(),
            reference_heads[1][:, t].contiguous(),
            reference_heads[2][:, t].contiguous(),
            reference_heads[3][:, t].contiguous(),
          )
          if reference_heads is not None
          else None,
          cached_goals[:, t] if cached_goals is not None else None,
          encodings[:, t].contiguous() if encodings is not None else None,
          active,
        )
        states.append(state.detach())
        contexts.append(context.detach())
        goals.append(goal.detach())
        values.append(value)
        # A reference ending is a known finite horizon, not a fall label.
        termination = torch.where(
          remaining <= t + 1, torch.ones_like(termination), termination
        )
        objective = objective + self.gamma**t * survival * reward
        survival = survival * (1 - termination)
        rewards.append(reward)
        terminations.append(termination)
        observation = (
          observation + torch.randn_like(observation) * self.robot.noise_scale()
        )
        history = torch.cat((history[:, 1:], observation[:, None]), dim=1)
        state = next_state
        context = self.model.advance_context(context, next_state[:, 57:79])
      final_goal = goal_features(self.robot, references[:, h + 1 : h + 11])
      state, history, context, active = absorbing_inputs(
        survival,
        state,
        history,
        context,
        initial_state,
        initial_history,
        initial_context,
      )
      survival = torch.where(active, survival, 0)
      values.append(torch.where(active, self.critic(state, final_goal, context), 0))
      objective = objective + self.gamma**h * survival * values[-1]
      loss = -objective.mean()
      if self.normalize_policy_objective:
        loss = loss / h
      self.distributed.require_finite(loss, "Nonfinite imagined return")
      loss.backward()
    actor_norm = self._update_actor()
    returns = gae_returns(
      torch.stack(rewards, 1), torch.stack(terminations, 1), torch.stack(values, 1)
    )
    x = torch.stack(states, 1).flatten(0, 1)
    g = torch.stack(goals, 1).flatten(0, 1)
    ctx = torch.stack(contexts, 1).flatten(0, 1)
    weights_tensor = torch.stack(weights, 1)
    critic_loss = self._fit_critic(
      x,
      g,
      ctx,
      returns.flatten(),
      weights_tensor.flatten(),
      critic_epochs,
      critic_minibatches,
    )
    return {
      "actor_loss": float(loss.detach()),
      "actor_grad_norm": float(actor_norm),
      "critic_loss": float(critic_loss),
      "imagined_reward": float(
        surviving_reward_mean(torch.stack(rewards, 1).detach(), weights_tensor)
      ),
      "imagination_survival_fraction": float(weights_tensor.mean()),
      "critic_target_abs_max": float(returns.abs().max()),
    }

  def _improve_policy_captured(
    self, batch: ReplayBatch, critic_epochs: int, critic_minibatches: int
  ) -> dict[str, float]:
    assert self.imagination_graph is not None
    self.model_optimizer.zero_grad(set_to_none=True)
    self.critic_optimizer.zero_grad(set_to_none=True)
    # Graph execution has no collectives or host validity reads. Coordinate
    # allocation/capture failures before all ranks enter gradient averaging.
    with self.distributed.checked("captured imagination"):
      output = self.imagination_graph.run(batch)
    self.distributed.require_finite(output.loss, "Nonfinite imagined return")
    actor_norm = self._update_actor()
    critic_loss = self._fit_critic(
      output.states,
      output.goals,
      output.context,
      output.targets,
      output.weights,
      critic_epochs,
      critic_minibatches,
    )
    return {
      "actor_loss": float(output.loss),
      "actor_grad_norm": float(actor_norm),
      "critic_loss": float(critic_loss),
      "imagined_reward": float(
        surviving_reward_mean(output.rewards, output.weights.view_as(output.rewards))
      ),
      "imagination_survival_fraction": float(output.weights.mean()),
      "critic_target_abs_max": float(output.targets.abs().max()),
    }

  def _update_actor(self) -> Tensor:
    self.distributed.average_gradients(self.actor.parameters())
    actor_norm = torch.nn.utils.clip_grad_norm_(
      self.actor.parameters(),
      1.0,
      error_if_nonfinite=self.distributed.validation_mode == "per_minibatch",
    )
    self.distributed.require_finite(actor_norm, "Nonfinite actor gradient norm")
    for p in self.actor.parameters():
      if p.grad is not None:
        p.grad.clamp_(-0.1, 0.1)
    self.actor_optimizer.step()
    return actor_norm

  def _critic_loss_step(
    self, x: Tensor, g: Tensor, ctx: Tensor, y: Tensor, weight: Tensor
  ) -> Tensor:
    error = torch.where(weight > 0, self.critic(x, g, ctx) - y, 0)
    # A fixed sample denominator preserves the global objective when gradients
    # are averaged across ranks with different surviving probability mass.
    return (error.square() * weight).mean()

  def _fit_critic(
    self,
    x: Tensor,
    g: Tensor,
    ctx: Tensor,
    y: Tensor,
    weight: Tensor,
    critic_epochs: int,
    critic_minibatches: int,
  ) -> Tensor:
    critic_losses = []
    for _ in range(critic_epochs):
      order = torch.randperm(len(x), device=x.device)
      for ids in order.tensor_split(min(critic_minibatches, len(order))):
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss = self.critic_loss_step(
          x[ids], g[ids], ctx[ids], y[ids], weight[ids]
        )
        self.distributed.require_finite(critic_loss, "Nonfinite critic loss")
        critic_loss.backward()
        self.distributed.average_gradients(self.critic.parameters())
        critic_norm = torch.nn.utils.clip_grad_norm_(
          self.critic.parameters(),
          1.0,
          error_if_nonfinite=self.distributed.validation_mode == "per_minibatch",
        )
        self.distributed.require_finite(critic_norm, "Nonfinite critic gradient norm")
        self.critic_optimizer.step()
        critic_losses.append(critic_loss.detach())
    return torch.stack(critic_losses).mean()
