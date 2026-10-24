"""Tracking-task domain randomization, sampled once per simulated robot."""

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor

VELOCITY_RANGE = (
  (-0.5, 0.5),
  (-0.5, 0.5),
  (-0.2, 0.2),
  (-0.52, 0.52),
  (-0.52, 0.52),
  (-0.78, 0.78),
)
OBSERVATION_NOISE_STD = (
  (0.0058,) * 22 + (1.0,) * 22 + (0.115,) * 3 + (0.029,) * 3 + (0.0,) * 22
)


@dataclass(frozen=True)
class RandomizationCfg:
  """Tracking events: pelvis COM, torso mass and shared foot friction."""

  friction: tuple[float, float] = (0.3, 1.2)
  encoder_bias: tuple[float, float] = (-0.02, 0.02)
  kp_factor: tuple[float, float] = (0.8, 1.2)
  kd_factor: tuple[float, float] = (0.9, 1.1)
  base_com: tuple[tuple[float, float], ...] = (
    (-0.025, 0.025),
    (-0.05, 0.05),
    (-0.05, 0.05),
  )
  torso_added_mass: tuple[float, float] = (-1.5, 1.5)
  push_interval_seconds: tuple[float, float] = (1.0, 3.0)
  push_velocity: tuple[tuple[float, float], ...] = VELOCITY_RANGE

  def validate(self) -> None:
    pairs = {
      "friction": self.friction,
      "encoder_bias": self.encoder_bias,
      "kp_factor": self.kp_factor,
      "kd_factor": self.kd_factor,
      "torso_added_mass": self.torso_added_mass,
      "push_interval_seconds": self.push_interval_seconds,
    }
    if len(self.base_com) != 3 or len(self.push_velocity) != 6:
      raise ValueError("Expected three COM axes and six push velocity axes")
    pairs.update({f"base_com_{i}": pair for i, pair in enumerate(self.base_com)})
    pairs.update({f"push_velocity_{i}": p for i, p in enumerate(self.push_velocity)})
    for name, (low, high) in pairs.items():
      if not math.isfinite(low) or not math.isfinite(high) or not low <= high:
        raise ValueError(f"{name} range is empty: {(low, high)}")
    if self.friction[0] < 0 or min(self.kp_factor[0], self.kd_factor[0]) <= 0:
      raise ValueError("Friction must be nonnegative and PD factors positive")
    if self.push_interval_seconds[0] <= 0:
      raise ValueError("Push intervals must be positive")


class ContextLayout(NamedTuple):
  """Fixed tensor interface shared with the trained networks.

  Joints keep the XML order and links keep the XML body order minus the world
  body. Reserved columns keep network input dimensions fixed. Motor offset
  stores minus encoder bias; payload is applied to torso_link.
  """

  friction: slice
  reserved_scalar: slice
  motor_offset: slice
  kp_factor: slice
  kd_factor: slice
  reserved_joints: slice
  root_com: slice
  torso_added_mass: slice
  reserved_bodies: slice
  delay_s: slice
  previous_executed_action: slice
  size: int

  @staticmethod
  def build(
    joints: int, links: int, *, modeled_action_delay: bool = False
  ) -> "ContextLayout":
    if joints < 1 or links < 1:
      raise ValueError("A randomization layout needs at least one joint and link")
    widths = (
      1,
      1,
      joints,
      joints,
      joints,
      joints,
      3,
      1,
      links,
      1 if modeled_action_delay else 0,
      joints if modeled_action_delay else 0,
    )
    bounds, start = [], 0
    for width in widths:
      bounds.append(slice(start, start + width))
      start += width
    return ContextLayout(*bounds, size=start)


def nominal_context(
  layout: ContextLayout, count: int, device: str | None = None
) -> Tensor:
  """The unrandomized context: unit friction, unit factors, no displacement."""
  context = torch.zeros(count, layout.size, device=device)
  context[:, layout.friction] = 1.0
  context[:, layout.kp_factor] = 1.0
  context[:, layout.kd_factor] = 1.0
  context[:, layout.reserved_joints] = 1.0
  context[:, layout.reserved_bodies] = 1.0
  return context


def sample_context(
  cfg: RandomizationCfg,
  layout: ContextLayout,
  count: int,
  device: str | None = None,
) -> Tensor:
  """Draw startup friction, encoder bias, PD gains, base COM and torso mass."""
  context = nominal_context(layout, count, device)
  for span, (low, high) in (
    (layout.friction, cfg.friction),
    (layout.kp_factor, cfg.kp_factor),
    (layout.kd_factor, cfg.kd_factor),
    (layout.torso_added_mass, cfg.torso_added_mass),
  ):
    context[:, span].uniform_(low, high)
  context[:, layout.motor_offset].uniform_(*cfg.encoder_bias).neg_()
  for axis, pair in enumerate(cfg.base_com):
    context[:, layout.root_com.start + axis].uniform_(*pair)
  return context


def sample_push(cfg: RandomizationCfg, count: int, device: str) -> Tensor:
  velocity = torch.empty(count, 6, device=device)
  for axis, bounds in enumerate(cfg.push_velocity):
    velocity[:, axis].uniform_(*bounds)
  return velocity
