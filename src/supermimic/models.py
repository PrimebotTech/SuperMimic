"""Networks independently specified from paper Tables A.4 and A.5."""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Literal

import torch
from torch import Tensor, nn

from supermimic.robot import (
  Q1Robot,
  normalize_quat,
)
from supermimic.state_representation import HeadingState

Activation = Literal["elu", "silu"]
Representation = Literal["legacy", "heading"]


def mlp(
  inputs: int, hidden: tuple[int, ...], outputs: int, activation: Activation = "elu"
) -> nn.Sequential:
  layers: list[nn.Module] = []
  for width in hidden:
    layers.extend(
      (nn.Linear(inputs, width), nn.SiLU() if activation == "silu" else nn.ELU())
    )
    inputs = width
  layers.append(nn.Linear(inputs, outputs))
  return nn.Sequential(*layers)


@contextmanager
def frozen(*modules: nn.Module) -> Generator[None, None, None]:
  """Freeze weights while retaining Jacobians with respect to tensor inputs."""
  parameters = [p for m in modules for p in m.parameters()]
  old = [p.requires_grad for p in parameters]
  try:
    for p in parameters:
      p.requires_grad_(False)
    yield
  finally:
    for p, flag in zip(parameters, old, strict=True):
      p.requires_grad_(flag)


def state_scale(robot: Q1Robot) -> Tensor:
  return torch.tensor([1.0] * 29 + [3.0] * 6 + [10.0] * 22 + [1.0] * 22)


def goal_features(robot: Q1Robot, references: Tensor) -> Tensor:
  """Reference-only goal: never inject the robot's privileged root pose."""
  rotations = references[..., 3:7]
  # A deterministic hemisphere removes arbitrary reference quaternion signs.
  rotations = torch.where(rotations[..., :1] < 0, -rotations, rotations)
  return torch.cat(
    (
      references[..., :3],
      rotations,
      references[..., 7:29] - robot.default_q,
      references[..., 29:32] / 3,
      references[..., 32:35] / 3,
      references[..., 35:57] / 10,
    ),
    dim=-1,
  ).flatten(1)


class WorldModel(nn.Module):
  scale: Tensor
  context_scale: Tensor
  termination_scale: Tensor

  def __init__(
    self,
    robot: Q1Robot,
    hidden: tuple[int, ...] = (2048, 2048, 2048),
    *,
    activation: Activation = "elu",
    representation: Representation = "legacy",
  ):
    super().__init__()
    self.register_buffer("scale", state_scale(robot))
    self.representation = representation
    self.heading_state = HeadingState() if representation == "heading" else None
    self.modeled_action_delay = robot.modeled_action_delay
    if self.modeled_action_delay:
      context_scale = torch.ones(robot.context_dim)
      context_scale[robot.context_layout.delay_s] = 0.016
      self.register_buffer("context_scale", context_scale)
    self.dynamics = mlp(
      robot.state_dim + 22 + robot.context_dim,
      hidden,
      56 if self.heading_state is not None else robot.physical_dim,
      activation,
    )
    self.termination = mlp(
      4 if self.heading_state is not None else 2,
      (64, 64) if self.heading_state is not None else (32, 32),
      1,
      activation,
    )
    if self.heading_state is not None:
      self.register_buffer(
        "termination_scale",
        torch.tensor(
          [
            robot.reward_scaling.termination_height,
            robot.reward_scaling.termination_tilt,
          ]
        ),
      )
    terminal_last = self.termination[-1]
    assert isinstance(terminal_last, nn.Linear)
    nn.init.constant_(terminal_last.bias, -4)
    last = self.dynamics[-1]
    assert isinstance(last, nn.Linear)
    if self.heading_state is not None:
      nn.init.zeros_(last.weight)
    else:
      nn.init.normal_(last.weight, std=0.001)
    nn.init.zeros_(last.bias)

  # Torch 2.7 types forward as a mutable Callable; overriding it is its public API.
  def forward(self, state: Tensor, action: Tensor, context: Tensor) -> Tensor:  # ty: ignore[invalid-mutable-override]
    if self.heading_state is not None:
      model_context = (
        context / self.context_scale if self.modeled_action_delay else context
      )
      residual = self.dynamics(
        torch.cat((self.heading_state.features(state), action, model_context), -1)
      )
      return self.heading_state.integrate(state, residual, action)
    # Absolute world XY is irrelevant to flat-ground dynamics.
    x = torch.cat((torch.zeros_like(state[:, :2]), state[:, 2:] / self.scale[2:]), -1)
    residual = self.dynamics(torch.cat((x, action, context), -1))
    physical = state[:, :57] + 0.02 * self.scale[:57] * residual
    return torch.cat(
      (physical[:, :3], normalize_quat(physical[:, 3:7]), physical[:, 7:], action), -1
    )

  def advance_context(self, context: Tensor, executed_action: Tensor) -> Tensor:
    if not self.modeled_action_delay:
      return context
    return torch.cat((context[..., :-22], executed_action), -1)

  def termination_logits(self, robot: Q1Robot, state: Tensor, target: Tensor) -> Tensor:
    if self.heading_state is not None:
      features = robot.termination_features(state, target) / self.termination_scale
      return self.termination(torch.cat((features, features.abs()), -1)).squeeze(-1)
    features = robot.termination_features(state, target) / state.new_tensor([0.25, 0.8])
    return self.termination(features).squeeze(-1)


class MotionPolicy(nn.Module):
  obs_scale: Tensor

  def __init__(
    self,
    robot: Q1Robot,
    hidden: tuple[int, ...] = (2048, 1024, 512, 256),
    *,
    activation: Activation = "elu",
  ):
    super().__init__()
    self.encoder = mlp(10 * robot.physical_dim, hidden, 32, activation)
    self.decoder = mlp(32 + 10 * robot.obs_dim, hidden, 22, activation)
    self.register_buffer(
      "obs_scale", torch.tensor([1.0] * 22 + [10.0] * 22 + [3.0] * 3 + [1.0] * 25)
    )
    last = self.decoder[-1]
    assert isinstance(last, nn.Linear)
    nn.init.normal_(last.weight, std=0.001)
    nn.init.zeros_(last.bias)

  def forward(self, history: Tensor, goal: Tensor) -> Tensor:  # ty: ignore[invalid-mutable-override]
    return self.decode(history, self.encode_goal(goal))

  def encode_goal(self, goal: Tensor) -> Tensor:
    return self.encoder(goal).tanh()

  def decode(self, history: Tensor, encoding: Tensor) -> Tensor:
    return self.decoder(
      torch.cat((encoding, (history / self.obs_scale).flatten(1)), -1)
    )


class Critic(nn.Module):
  scale: Tensor

  def __init__(
    self,
    robot: Q1Robot,
    hidden: tuple[int, ...] = (256, 128),
    *,
    activation: Activation = "elu",
  ):
    super().__init__()
    self.register_buffer("scale", state_scale(robot))
    self.network = mlp(
      robot.state_dim + 10 * robot.physical_dim + robot.context_dim,
      hidden,
      1,
      activation,
    )

  def forward(self, state: Tensor, goal: Tensor, context: Tensor) -> Tensor:  # ty: ignore[invalid-mutable-override]
    return self.network(torch.cat((state / self.scale, goal, context), -1)).squeeze(-1)
