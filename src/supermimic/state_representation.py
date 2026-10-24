"""Heading-invariant physical features and differentiable rigid-body integration.

States keep the simulator's WXYZ quaternion and world angular velocity contract.
Heading features and explicit integration are implementation choices; the paper
specifies neither these encodings nor hidden-layer activation functions.
"""

import math

import torch
from torch import Tensor, nn

from supermimic.robot import normalize_quat, quat_apply, quat_inverse, quat_mul


def heading_quat(rotation: Tensor) -> Tensor:
  w, x, y, z = normalize_quat(rotation).unbind(-1)
  yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
  zero = torch.zeros_like(yaw)
  return torch.stack(((yaw / 2).cos(), zero, zero, (yaw / 2).sin()), -1)


def rotation_columns(rotation: Tensor) -> Tensor:
  """First two rotation-matrix columns; independent of quaternion sign."""
  w, x, y, z = normalize_quat(rotation).unbind(-1)
  return torch.stack(
    (
      1 - 2 * (y.square() + z.square()),
      2 * (x * y + w * z),
      2 * (x * z - w * y),
      2 * (x * y - w * z),
      1 - 2 * (x.square() + z.square()),
      2 * (y * z + w * x),
    ),
    -1,
  )


def quaternion_exp(vector: Tensor) -> Tensor:
  # Avoid the undefined sqrt derivative at zero while preserving the limit.
  angle = vector.square().sum(-1, keepdim=True).clamp_min(1e-12).sqrt()
  coefficient = 0.5 * torch.sinc(angle / (2 * math.pi))
  return torch.cat(((angle / 2).cos(), coefficient * vector), -1)


def quaternion_log(rotation: Tensor) -> Tensor:
  rotation = normalize_quat(rotation)
  rotation = torch.where(rotation[..., :1] < 0, -rotation, rotation)
  vector = rotation[..., 1:]
  length = vector.square().sum(-1, keepdim=True).clamp_min(1e-12).sqrt()
  angle = 2 * torch.atan2(length, rotation[..., :1])
  return vector * (angle / length)


class HeadingState(nn.Module):
  input_scale: Tensor
  output_scale: Tensor
  error_scale: Tensor

  def __init__(self):
    super().__init__()
    self.register_buffer(
      "input_scale",
      torch.tensor([1.0] * 29 + [1.0] * 3 + [2.0] * 3 + [5.0] * 22 + [1.0] * 22),
    )
    self.register_buffer(
      "output_scale",
      torch.tensor(
        [0.3] * 3 + [0.5] * 3 + [2.0] * 22 + [0.01] * 3 + [0.02] * 3 + [0.02] * 22
      ),
    )
    self.register_buffer(
      "error_scale",
      torch.tensor(
        [0.05] * 3 + [0.1] * 3 + [0.5] * 3 + [1.0] * 3 + [0.1] * 22 + [2.0] * 22
      ),
    )

  def features(self, state: Tensor) -> Tensor:
    rotation = normalize_quat(state[..., 3:7])
    heading = heading_quat(rotation)
    local = quat_mul(quat_inverse(heading), rotation)
    return (
      torch.cat(
        (
          state[..., 2:3],
          rotation_columns(local),
          state[..., 7:29],
          quat_apply(quat_inverse(heading), state[..., 29:32]),
          quat_apply(quat_inverse(rotation), state[..., 32:35]),
          state[..., 35:57],
          state[..., 57:79],
        ),
        -1,
      )
      / self.input_scale
    )

  def integrate(self, state: Tensor, prediction: Tensor, action: Tensor) -> Tensor:
    dv, dw, dqd, dp, dr, dq = (prediction * self.output_scale).split(
      (3, 3, 22, 3, 3, 22), -1
    )
    rotation = normalize_quat(state[..., 3:7])
    heading = heading_quat(rotation)
    velocity = state[..., 29:32] + quat_apply(heading, dv)
    body_gyro = quat_apply(quat_inverse(rotation), state[..., 32:35]) + dw
    joint_velocity = state[..., 35:57] + dqd
    successor_rotation = normalize_quat(
      quat_mul(rotation, quaternion_exp(0.02 * body_gyro + dr))
    )
    return torch.cat(
      (
        state[..., :3] + 0.02 * velocity + quat_apply(heading, dp),
        successor_rotation,
        state[..., 7:29] + 0.02 * joint_velocity + dq,
        velocity,
        quat_apply(successor_rotation, body_gyro),
        joint_velocity,
        action,
      ),
      -1,
    )

  def error(self, predicted: Tensor, target: Tensor) -> Tensor:
    rotation_delta = quat_mul(
      predicted[..., 3:7], quat_inverse(normalize_quat(target[..., 3:7]))
    )
    return (
      torch.cat(
        (
          predicted[..., :3] - target[..., :3],
          quaternion_log(rotation_delta),
          predicted[..., 29:35] - target[..., 29:35],
          predicted[..., 7:29] - target[..., 7:29],
          predicted[..., 35:57] - target[..., 35:57],
        ),
        -1,
      )
      / self.error_scale
    )
