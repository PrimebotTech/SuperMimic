"""Q1 XML validation and differentiable link kinematics.

Quaternions are WXYZ. Root linear and angular velocities are world-frame
velocities at the root link origin, rather than its center of mass.
"""

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import mujoco
import numpy as np
import torch
from torch import Tensor, nn

from supermimic.randomization import OBSERVATION_NOISE_STD, ContextLayout


def normalize_quat(q: Tensor) -> Tensor:
  return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def quat_mul(a: Tensor, b: Tensor) -> Tensor:
  aw, av = a[..., :1], a[..., 1:]
  bw, bv = b[..., :1], b[..., 1:]
  return torch.cat(
    (
      aw * bw - (av * bv).sum(-1, keepdim=True),
      aw * bv + bw * av + torch.linalg.cross(av, bv),
    ),
    dim=-1,
  )


def quat_apply(q: Tensor, v: Tensor) -> Tensor:
  qv, v = torch.broadcast_tensors(q[..., 1:], v)
  cross = 2 * torch.linalg.cross(qv, v)
  return v + q[..., :1] * cross + torch.linalg.cross(qv, cross)


def quat_inverse(q: Tensor) -> Tensor:
  return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def rotation_error(a: Tensor, b: Tensor) -> Tensor:
  """Squared SO(3) log norm, invariant to quaternion sign."""
  d = quat_mul(a, quat_inverse(b))
  return (2 * torch.atan2(d[..., 1:].norm(dim=-1), d[..., 0].abs())).square()


@dataclass(frozen=True)
class RewardScaling:
  """Resolved, checkpointed reward units; defaults preserve legacy training.

  Proportional scaling compares the same fractional tracking error at the same
  50 Hz control rate. It does not assume dynamically similar gravity or motors.
  """

  mode: Literal["legacy", "q1_proportional"] = "legacy"
  position_std: float = 0.3
  orientation_std: float = 0.4
  linear_velocity_std: float = 1.0
  angular_velocity_std: float = 3.14
  joint_sum_multiplier: float = 1.0
  termination_height: float = 0.25
  termination_tilt: float = 0.8
  length_scale: float = 1.0
  nominal_pelvis_height: float | None = None
  reference_pelvis_height: float | None = None
  joint_count: int = 22
  reference_joint_count: int = 29

  def __post_init__(self) -> None:
    if self.mode not in ("legacy", "q1_proportional"):
      raise ValueError("Unknown reward scaling mode")
    for value in (
      self.position_std,
      self.orientation_std,
      self.linear_velocity_std,
      self.angular_velocity_std,
      self.joint_sum_multiplier,
      self.termination_height,
      self.termination_tilt,
      self.length_scale,
    ):
      if not math.isfinite(value) or value <= 0:
        raise ValueError("Reward scales must be finite and positive")
    for height in (self.nominal_pelvis_height, self.reference_pelvis_height):
      if height is not None and (not math.isfinite(height) or height <= 0):
        raise ValueError("Nominal pelvis heights must be finite and positive")
    if self.joint_count <= 0 or self.reference_joint_count <= 0:
      raise ValueError("Reward joint counts must be positive")

  @classmethod
  def q1_proportional(
    cls,
    nominal_pelvis_height: float,
    reference_pelvis_height: float = 0.793,
    joint_count: int = 22,
    reference_joint_count: int = 29,
  ) -> "RewardScaling":
    """Scale dimensional terms by stature and joint sums by actuator count."""
    if (
      not math.isfinite(nominal_pelvis_height)
      or nominal_pelvis_height <= 0
      or not math.isfinite(reference_pelvis_height)
      or reference_pelvis_height <= 0
    ):
      raise ValueError("Nominal pelvis heights must be finite and positive")
    if joint_count <= 0 or reference_joint_count <= 0:
      raise ValueError("Reward joint counts must be positive")
    ratio = nominal_pelvis_height / reference_pelvis_height
    return cls(
      mode="q1_proportional",
      position_std=0.3 * ratio,
      linear_velocity_std=ratio,
      joint_sum_multiplier=reference_joint_count / joint_count,
      termination_height=0.25 * ratio,
      length_scale=ratio,
      nominal_pelvis_height=nominal_pelvis_height,
      reference_pelvis_height=reference_pelvis_height,
      joint_count=joint_count,
      reference_joint_count=reference_joint_count,
    )


class Q1Robot(nn.Module):
  """Build all joint maps and link transforms from the supplied XML."""

  body_pos: Tensor
  body_quat: Tensor
  joint_axis: Tensor
  joint_anchor: Tensor
  default_q: Tensor
  arm_links: Tensor
  joint_limits: Tensor
  actuator_qindex: Tensor
  actuator_gear: Tensor
  ctrl_limits: Tensor

  def __init__(
    self,
    xml: Path,
    *,
    reward_scaling: RewardScaling | Literal["q1_proportional"] | None = None,
    modeled_action_delay: bool = False,
  ):
    super().__init__()
    self.xml = xml.resolve()
    self.model = mujoco.MjModel.from_xml_path(str(self.xml))
    m = self.model
    if m.body(1).name != "pelvis" or m.nu != 22 or m.nq != 29 or m.nv != 28:
      raise ValueError("Expected Q1 V2: pelvis root, 22 actuators, nq=29, nv=28")
    if m.nmocap or m.na or m.neq:
      raise ValueError("Q1 training requires a free robot without mocap/equalities")
    if int(m.jnt_type[0]) != mujoco.mjtJoint.mjJNT_FREE:
      raise ValueError("The first Q1 joint must be free")
    if np.any(m.jnt_type[1:] != mujoco.mjtJoint.mjJNT_HINGE):
      raise ValueError("Q1's 22 controlled joints must be hinges")
    joint_ids = m.actuator_trnid[:, 0]
    if sorted(joint_ids.tolist()) != list(range(1, 23)):
      raise ValueError("Each Q1 hinge must have exactly one actuator")
    if np.any(m.actuator_trntype != mujoco.mjtTrn.mjTRN_JOINT):
      raise ValueError("Q1 actuators must use joint transmissions")
    if np.any(m.actuator_gaintype != mujoco.mjtGain.mjGAIN_FIXED):
      raise ValueError("Expected fixed-gain Q1 motors")
    if np.any(m.actuator_biastype != mujoco.mjtBias.mjBIAS_NONE):
      raise ValueError("Expected torque motors; position servos need another adapter")
    if np.any(m.actuator_gear[:, 1:] != 0) or np.any(m.actuator_gear[:, 0] == 0):
      raise ValueError("Expected scalar, nonzero Q1 motor gears")
    if not np.all(m.actuator_ctrllimited):
      raise ValueError("Q1 motors must declare torque control limits")
    if reward_scaling == "q1_proportional":
      self.reward_scaling = RewardScaling.q1_proportional(
        float(m.qpos0[2]), joint_count=m.nu
      )
    elif reward_scaling is None:
      self.reward_scaling = RewardScaling()
    elif isinstance(reward_scaling, RewardScaling):
      self.reward_scaling = reward_scaling
    else:
      raise ValueError("Expected RewardScaling or 'q1_proportional'")
    if self.reward_scaling.joint_count != m.nu:
      raise ValueError("Reward scaling actuator count differs from the Q1 XML")
    self.joint_names = tuple(m.joint(i).name for i in range(1, 23))
    self.body_names = tuple(m.body(i).name for i in range(1, m.nbody))
    self.state_dim = 7 + 22 + 6 + 22 + 22
    self.obs_dim = 3 * 22 + 6
    self.physical_dim = self.state_dim - 22
    # One per-episode randomization vector, laid out over the XML's own joint
    # and body order so a checkpoint's context columns stay readable.
    self.context_layout = ContextLayout.build(
      len(self.joint_names),
      len(self.body_names),
      modeled_action_delay=modeled_action_delay,
    )
    self.modeled_action_delay = modeled_action_delay
    self.context_dim = self.context_layout.size
    self.root_id = int(m.body(self.body_names[0]).id)
    self.torso_id = int(m.body("torso_link").id)
    # Named arm-body mask used by tracking diagnostics.
    self.register_buffer(
      "arm_links",
      torch.tensor(
        [
          any(part in name for part in ("shoulder", "elbow", "wrist"))
          for name in self.body_names
        ]
      ),
    )
    self.sha256 = hashlib.sha256(self.xml.read_bytes()).hexdigest()
    # Include mesh contents; an XML hash alone does not identify the model.
    asset_hash = hashlib.sha256(self.xml.read_bytes())
    asset_hash.update(m.mesh_vert.tobytes())
    asset_hash.update(m.mesh_face.tobytes())
    self.asset_sha256 = asset_hash.hexdigest()
    self.parents = m.body_parentid.tolist()
    self.body_joints = [
      list(range(int(m.body_jntadr[b]), int(m.body_jntadr[b] + m.body_jntnum[b])))
      if m.body_jntnum[b]
      else []
      for b in range(m.nbody)
    ]
    self.qindexes = (m.jnt_qposadr - 7).tolist()
    self.register_buffer("body_pos", torch.tensor(m.body_pos, dtype=torch.float32))
    self.register_buffer("body_quat", torch.tensor(m.body_quat, dtype=torch.float32))
    self.register_buffer("joint_axis", torch.tensor(m.jnt_axis, dtype=torch.float32))
    self.register_buffer("joint_anchor", torch.tensor(m.jnt_pos, dtype=torch.float32))
    self.register_buffer("default_q", torch.tensor(m.qpos0[7:], dtype=torch.float32))
    self.register_buffer(
      "joint_limits", torch.tensor(m.jnt_range[1:], dtype=torch.float32)
    )
    self.register_buffer("actuator_qindex", torch.tensor(m.jnt_qposadr[joint_ids] - 7))
    self.register_buffer(
      "actuator_gear", torch.tensor(m.actuator_gear[:, 0], dtype=torch.float32)
    )
    self.register_buffer(
      "ctrl_limits", torch.tensor(m.actuator_ctrlrange, dtype=torch.float32)
    )

  def reward_identity(self) -> dict[str, str | float | int | None]:
    """Expose resolved values for config/identity and comparison across runs."""
    return asdict(self.reward_scaling)

  def get_extra_state(self) -> dict[str, str | float | int | None]:
    return self.reward_identity()

  def set_extra_state(self, state: dict[str, Any]) -> None:
    scaling = RewardScaling(**state)
    if scaling.joint_count != self.model.nu:
      raise ValueError("Checkpoint reward actuator count differs from the Q1 XML")
    self.reward_scaling = scaling

  def _load_from_state_dict(
    self,
    state_dict: dict[str, Any],
    prefix: str,
    local_metadata: dict[str, Any],
    strict: bool,
    missing_keys: list[str],
    unexpected_keys: list[str],
    error_msgs: list[str],
  ) -> None:
    # Older checkpoints lack reward metadata. Migrate them to their original
    # semantics even when the caller initially constructed an adapted robot.
    if f"{prefix}_extra_state" not in state_dict:
      state_dict = dict(state_dict)
      state_dict[f"{prefix}_extra_state"] = asdict(RewardScaling())
    super()._load_from_state_dict(
      state_dict,
      prefix,
      local_metadata,
      strict,
      missing_keys,
      unexpected_keys,
      error_msgs,
    )

  def observation(self, state: Tensor, context: Tensor) -> Tensor:
    q, dq = state[..., 7:29], state[..., 35:57]
    rot = normalize_quat(state[..., 3:7])
    gyro = quat_apply(quat_inverse(rot), state[..., 32:35])
    gravity = quat_apply(quat_inverse(rot), state.new_tensor([0, 0, -1]))
    return torch.cat(
      (
        q - self.default_q - context[..., self.context_layout.motor_offset],
        dq,
        gyro,
        gravity,
        state[..., 57:79],
      ),
      dim=-1,
    )

  def noise_scale(self) -> Tensor:
    return self.default_q.new_tensor(OBSERVATION_NOISE_STD)

  def kinematics(self, state: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return link-origin position, WXYZ rotation, world linear/angular velocity."""
    shape = state.shape[:-1]
    zero = state.new_zeros((*shape, 3))
    identity = state.new_tensor([1, 0, 0, 0]).expand(*shape, 4)
    positions, rotations, velocities, angular = [zero], [identity], [zero], [zero]
    for body in range(1, self.model.nbody):
      if body == 1:
        p, r, v, w = (
          state[..., :3],
          normalize_quat(state[..., 3:7]),
          state[..., 29:32],
          state[..., 32:35],
        )
      else:
        parent = self.parents[body]
        offset = quat_apply(rotations[parent], self.body_pos[body])
        p = positions[parent] + offset
        r = quat_mul(rotations[parent], self.body_quat[body].expand(*shape, 4))
        w = angular[parent]
        v = velocities[parent] + torch.linalg.cross(w, offset)
        for joint in self.body_joints[body]:
          index = self.qindexes[joint]
          half = 0.5 * (state[..., 7 + index] - self.default_q[index])
          local_rot = torch.cat(
            (
              half.cos().unsqueeze(-1),
              half.sin().unsqueeze(-1) * self.joint_axis[joint],
            ),
            -1,
          )
          axis = quat_apply(r, self.joint_axis[joint])
          anchor = p + quat_apply(r, self.joint_anchor[joint])
          new_r = quat_mul(r, local_rot)
          new_p = anchor - quat_apply(new_r, self.joint_anchor[joint])
          joint_w = axis * state[..., 35 + index].unsqueeze(-1)
          v = (
            v
            + torch.linalg.cross(w, new_p - p)
            + torch.linalg.cross(joint_w, new_p - anchor)
          )
          w = w + joint_w
          p, r = new_p, new_r
      positions.append(p)
      rotations.append(r)
      velocities.append(v)
      angular.append(w)
    return (
      torch.stack(positions, dim=-2),
      torch.stack(rotations, dim=-2),
      torch.stack(velocities, dim=-2),
      torch.stack(angular, dim=-2),
    )

  def reward(
    self,
    state: Tensor,
    next_state: Tensor,
    target: Tensor,
    action: Tensor,
    dt: float,
    target_heads: tuple[Tensor, Tensor, Tensor, Tensor] | None = None,
  ) -> Tensor:
    p, r, v, w = self.kinematics(next_state)
    tp, tr, tv, tw = self.kinematics(target) if target_heads is None else target_heads
    scales = self.reward_scaling
    score = (
      (
        -(p[..., 1:, :] - tp[..., 1:, :]).square().sum(-1).mean(-1)
        / scales.position_std**2
      ).exp()
      + (
        -rotation_error(r[..., 1:, :], tr[..., 1:, :]).mean(-1)
        / scales.orientation_std**2
      ).exp()
      + (
        -(v[..., 1:, :] - tv[..., 1:, :]).square().sum(-1).mean(-1)
        / scales.linear_velocity_std**2
      ).exp()
      + (
        -(w[..., 1:, :] - tw[..., 1:, :]).square().sum(-1).mean(-1)
        / scales.angular_velocity_std**2
      ).exp()
      + 0.5
      * (
        -(p[..., 1, :] - tp[..., 1, :]).square().sum(-1) / scales.position_std**2
      ).exp()
      + 0.5
      * (-rotation_error(r[..., 1, :], tr[..., 1, :]) / scales.orientation_std**2).exp()
    )
    limits = (
      (next_state[..., 7:29] - self.joint_limits[:, 1]).clamp_min(0)
      + (self.joint_limits[:, 0] - next_state[..., 7:29]).clamp_min(0)
    ).sum(-1)
    smooth = (action - state[..., 57:79]).square().sum(-1)
    return dt * (
      score
      - 0.1 * scales.joint_sum_multiplier * smooth
      - 2 * scales.joint_sum_multiplier * limits
    )

  def termination_features(self, state: Tensor, target: Tensor) -> Tensor:
    gravity = state.new_tensor([0, 0, -1])
    g = quat_apply(quat_inverse(normalize_quat(state[..., 3:7])), gravity)
    tg = quat_apply(quat_inverse(normalize_quat(target[..., 3:7])), gravity)
    return torch.stack(
      ((state[..., 2] - target[..., 2]).abs(), (g[..., 2] - tg[..., 2]).abs()), -1
    )

  def terminated(self, state: Tensor, target: Tensor) -> Tensor:
    f = self.termination_features(state, target)
    return (f[..., 0] > self.reward_scaling.termination_height) | (
      f[..., 1] > self.reward_scaling.termination_tilt
    )
