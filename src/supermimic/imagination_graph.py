"""Opt-in whole-horizon actor backward, with all optimizer work outside.

The returned tensors belong to one graph and are overwritten by its next
replay. Consume them in the critic update before another call to run().
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor

from supermimic.imagination_terminal import absorbing_inputs
from supermimic.models import frozen, goal_features
from supermimic.replay import ReplayBatch

if TYPE_CHECKING:
  from supermimic.algorithm import SuperMimic


@dataclass
class ImaginationInput:
  state: Tensor
  history: Tensor
  context: Tensor
  references: Tensor
  remaining: Tensor

  def tensors(self) -> tuple[Tensor, ...]:
    return (self.state, self.history, self.context, self.references, self.remaining)


@dataclass
class ImaginationOutput:
  loss: Tensor
  rewards: Tensor
  states: Tensor
  goals: Tensor
  context: Tensor
  targets: Tensor
  weights: Tensor


class ImaginationGraph:
  def __init__(
    self,
    learner: "SuperMimic",
    calculate_returns: Callable[[Tensor, Tensor, Tensor], Tensor],
  ):
    self.learner = learner
    self.calculate_returns = calculate_returns
    self.graph: torch.cuda.CUDAGraph | None = None
    self.static_input: ImaginationInput | None = None
    self.static_output: ImaginationOutput | None = None
    self.gradients: list[tuple[torch.nn.Parameter, Tensor]] = []
    self.parameter_layout: tuple[tuple[int, bool], ...] | None = None
    self.capture_peak_bytes = 0
    # Eager new_tensor(list) stages CPU constants; prepare noise outside capture.
    self.noise_scale = learner.robot.noise_scale()

  def prepare(self, batch: ReplayBatch) -> ImaginationInput:
    learner = self.learner
    references = learner.motions.sequence(
      batch.clips,
      batch.frames,
      torch.arange(learner.horizon + 11, device=batch.frames.device),
    )
    return ImaginationInput(
      batch.states[:, 0].detach(),
      batch.history.detach(),
      batch.context,
      references,
      learner.motions.remaining(batch.clips, batch.frames),
    )

  def forward_backward(self, inputs: ImaginationInput) -> ImaginationOutput:
    """Tensor-only execution: no validation, collectives, clips or optimizers."""
    learner = self.learner
    h, references = learner.horizon, inputs.references
    state, history, context = inputs.state, inputs.history, inputs.context
    initial_state, initial_history, initial_context = state, history, context
    cached_goals, encodings = None, None
    reference_heads = None
    if learner.cache_goals:
      cached_goals = learner.reference_goals(references)
      encodings = learner.encode_goal(cached_goals.flatten(0, 1)).reshape(
        len(references), h, -1
      )
    if learner.cache_reference_heads:
      with torch.no_grad():
        flat = references[:, 1 : h + 1].reshape(-1, learner.robot.state_dim)
        reference_heads = tuple(
          head.reshape(len(references), h, *head.shape[1:])
          for head in learner.reference_kinematics(flat)
        )
    survival = state.new_ones(len(state))
    objective = state.new_zeros(len(state))
    states, goals, rewards, terminations, values = [], [], [], [], []
    contexts = []
    weights = []
    with frozen(learner.model, learner.critic):
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
        successor, reward, termination, observation, goal, value = (
          learner.imagined_step(
            state,
            history,
            references[:, t + 1 : t + (2 if learner.cache_goals else 11)].contiguous(),
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
        )
        states.append(state.detach())
        contexts.append(context.detach())
        goals.append(goal.detach())
        values.append(value)
        termination = torch.where(
          inputs.remaining <= t + 1, torch.ones_like(termination), termination
        )
        objective = objective + learner.gamma**t * survival * reward
        survival = survival * (1 - termination)
        rewards.append(reward)
        terminations.append(termination)
        observation = observation + torch.randn_like(observation) * self.noise_scale
        history = torch.cat((history[:, 1:], observation[:, None]), dim=1)
        state = successor
        context = learner.model.advance_context(context, successor[:, 57:79])
      final_goal = goal_features(learner.robot, references[:, h + 1 : h + 11])
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
      values.append(torch.where(active, learner.critic(state, final_goal, context), 0))
      objective = objective + learner.gamma**h * survival * values[-1]
      loss = -objective.mean()
      if learner.normalize_policy_objective:
        loss = loss / h
      loss.backward()
    rewards_tensor = torch.stack(rewards, 1)
    returns = self.calculate_returns(
      rewards_tensor, torch.stack(terminations, 1), torch.stack(values, 1)
    )
    states_tensor = torch.stack(states, 1).flatten(0, 1)
    return ImaginationOutput(
      loss.detach(),
      rewards_tensor.detach(),
      states_tensor,
      torch.stack(goals, 1).flatten(0, 1),
      torch.stack(contexts, 1).flatten(0, 1),
      returns.flatten(),
      torch.stack(weights, 1).flatten(),
    )

  def _layout(self) -> tuple[tuple[int, bool], ...]:
    return tuple(
      (parameter.data_ptr(), parameter.requires_grad)
      for module in (self.learner.actor, self.learner.model, self.learner.critic)
      for parameter in module.parameters()
    )

  def _capture(self, inputs: ImaginationInput) -> None:
    device = inputs.state.device
    assert device.index is not None
    self.static_input = ImaginationInput(
      *(tensor.clone() for tensor in inputs.tensors())
    )
    # Torch 2.7's Stream.__new__ annotation differs between type checkers.
    stream = cast(torch.cuda.Stream, torch.cuda.Stream(device=device))  # ty: ignore[redundant-cast]
    stream.wait_stream(torch.cuda.current_stream(device))
    # Warmups create compiled forward/backward kernels and autograd grad buffers.
    # They do not update weights; fork_rng restores all consumed exploration RNG.
    with torch.random.fork_rng(devices=[device.index]):
      with torch.cuda.stream(stream):
        for _ in range(3):
          self.learner.actor_optimizer.zero_grad(set_to_none=True)
          self.forward_backward(self.static_input)
        self.gradients = [
          (parameter, parameter.grad)
          for parameter in self.learner.actor.parameters()
          if parameter.grad is not None
        ]
      torch.cuda.current_stream(device).wait_stream(stream)
      torch.cuda.synchronize(device)
      graph = torch.cuda.CUDAGraph()
      with torch.cuda.graph(graph, stream=stream):
        # Zero captured buffers inside the graph so external averaging/clipping
        # cannot accumulate into the next raw actor gradient.
        for _, gradient in self.gradients:
          gradient.zero_()
        self.static_output = self.forward_backward(self.static_input)
      self.graph = graph
    self.parameter_layout = self._layout()
    self.capture_peak_bytes = torch.cuda.max_memory_allocated(device)

  def run(self, batch: ReplayBatch) -> ImaginationOutput:
    inputs = self.prepare(batch)
    if inputs.state.device.type != "cuda":
      self.learner.actor_optimizer.zero_grad(set_to_none=True)
      return self.forward_backward(inputs)
    if self.learner.compiled_backend != "inductor":
      raise ValueError("CUDA imagination capture requires Inductor compilation")
    if self.graph is None:
      self._capture(inputs)
    assert self.graph is not None and self.static_input is not None
    assert self.static_output is not None
    if self._layout() != self.parameter_layout:
      raise RuntimeError("Captured imagination weights moved or trainability changed")
    for source, destination in zip(
      inputs.tensors(), self.static_input.tensors(), strict=True
    ):
      if source.shape != destination.shape or source.dtype != destination.dtype:
        raise ValueError("Captured imagination requires fixed input shapes and dtypes")
      destination.copy_(source)
    for parameter, gradient in self.gradients:
      parameter.grad = gradient
    self.graph.replay()
    return self.static_output
