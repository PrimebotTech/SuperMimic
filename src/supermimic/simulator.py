"""Direct Q1 XML simulation; CPU MuJoCo and GPU MuJoCo Warp collectors."""

import copy

import mujoco
import mujoco_warp as mw
import numpy as np
import torch
import warp as wp
from torch import Tensor

from supermimic.randomization import (
  RandomizationCfg,
  nominal_context,
  sample_context,
  sample_push,
)
from supermimic.robot import Q1Robot, quat_apply, quat_inverse
from supermimic.warp_runtime import configure_cuda_compilation

# Q1 PD calibration, read off the walk/run PPO environment that drives the real
# robot. Keys are matched as substrings of the XML joint name, as that
# environment does, and every Q1 joint must match exactly one of them.
STIFFNESS = {
  "hip_pitch_joint": 40.0,
  "hip_roll_joint": 40.0,
  "hip_yaw_joint": 40.0,
  "knee_joint": 40.0,
  "ankle_pitch_joint": 20.0,
  "ankle_roll_joint": 10.0,
  "waist_yaw_joint": 80.0,
  "waist_roll_joint": 80.0,
  "shoulder": 80.0,
  "elbow": 80.0,
}
DAMPING = {
  "hip_pitch_joint": 1.0,
  "hip_roll_joint": 1.0,
  "hip_yaw_joint": 1.0,
  "knee_joint": 1.0,
  "ankle_pitch_joint": 1.0,
  "ankle_roll_joint": 1.0,
  "waist_yaw_joint": 2.0,
  "waist_roll_joint": 2.0,
  "shoulder": 2.0,
  "elbow": 2.0,
}


def pd_gains(joint_names: tuple[str, ...]) -> tuple[list[float], list[float]]:
  """Resolve the PD table by joint name, refusing ambiguous or missing joints."""
  kp, kd = [], []
  for name in joint_names:
    keys = [key for key in STIFFNESS if key in name]
    if len(keys) != 1:
      raise ValueError(f"Joint {name!r} matches {len(keys)} PD gain keys, expected 1")
    kp.append(STIFFNESS[keys[0]])
    kd.append(DAMPING[keys[0]])
  return kp, kd


def torque_speed_parameters(
  joint_names: tuple[str, ...], device: str
) -> tuple[Tensor, Tensor, Tensor]:
  """Q1 walk/run motor curves, resolved by joint name rather than DOF index."""
  peaks, knees, cutoffs = [], [], []
  for name in joint_names:
    big = any(part in name for part in ("hip_", "knee_", "waist_"))
    small = any(part in name for part in ("ankle_", "shoulder_", "elbow_"))
    if big == small:
      raise ValueError(f"Joint {name!r} has no unambiguous Q1 motor curve")
    peak, knee, cutoff = (34.0, 7.5, 17.0) if big else (21.0, 8.0, 16.0)
    peaks.append(peak)
    knees.append(knee)
    cutoffs.append(cutoff)
  return (
    torch.tensor(peaks, device=device),
    torch.tensor(knees, device=device),
    torch.tensor(cutoffs, device=device),
  )


def torque_speed_limits(
  velocity: Tensor, peaks: Tensor, knees: Tensor, cutoffs: Tensor
) -> tuple[Tensor, Tensor]:
  """Directional torque bounds: full braking torque remains at high speed."""
  slope = peaks / (cutoffs - knees)
  upper = (peaks - slope * (velocity - knees)).clamp(min=0).minimum(peaks)
  lower = (-peaks - slope * (velocity + knees)).clamp(max=0).maximum(-peaks)
  return lower, upper


class Q1Simulator:
  def __init__(
    self,
    robot: Q1Robot,
    num_envs: int,
    device: str,
    action_scale: float = 0.5,
    randomization: RandomizationCfg | None = None,
    use_tn: bool = True,
    physics_timestep: float | None = None,
    capture_reset: bool = False,
  ):
    self.robot, self.num_envs, self.device = robot, num_envs, device
    self.action_scale = action_scale
    self.use_tn = use_tn
    self.capture_reset = capture_reset
    self._const_graph = None
    self.randomization = randomization or RandomizationCfg()
    self.randomization.validate()
    self.layout = robot.context_layout
    self.dt = 0.02
    if physics_timestep is not None:
      if physics_timestep <= 0 or physics_timestep > self.dt:
        raise ValueError("Physics timestep must be positive and at most 0.02 s")
      robot.model.opt.timestep = physics_timestep
    self.substeps = round(self.dt / robot.model.opt.timestep)
    if not np.isclose(self.substeps * robot.model.opt.timestep, self.dt):
      raise ValueError("The XML timestep must divide the 50 Hz control period")
    self.context = nominal_context(self.layout, num_envs, device)
    self.startup_context = self.context.clone()
    self.startup_sampled = torch.zeros(num_envs, dtype=torch.bool, device=device)
    self.push_enabled = torch.zeros(num_envs, dtype=torch.bool, device=device)
    self.push_timer = torch.zeros(num_envs, device=device)
    self.pushed = torch.zeros(num_envs, dtype=torch.bool, device=device)
    self.lagged = torch.zeros(num_envs, 1, dtype=torch.bool, device=device)
    self.last_action = torch.zeros(num_envs, 22, device=device)
    # Q1 control calibration, separate from the XML's torque motors. All torque
    # Gear, armature, nominal mass and collision assets come from the XML.
    # Q1 torque-speed curves impose additional bounds within its motor limits.
    kp, kd = pd_gains(robot.joint_names)
    self.kp = torch.tensor(kp, device=device)
    self.kd = torch.tensor(kd, device=device)
    self.torque_peaks, self.torque_knees, self.torque_cutoffs = torque_speed_parameters(
      robot.joint_names, device
    )
    torso_mass = float(robot.model.body_mass[robot.torso_id])
    if torso_mass + self.randomization.torso_added_mass[0] <= 0:
      raise ValueError(
        f"A {self.randomization.torso_added_mass[0]:+g} kg payload leaves the "
        f"{torso_mass:g} kg torso_link massless"
      )
    self.foot_geoms = torch.tensor(
      [
        i
        for i, body in enumerate(robot.model.geom_bodyid)
        if robot.model.body(int(body)).name
        in ("left_ankle_roll_link", "right_ankle_roll_link")
      ],
      device=device,
      dtype=torch.long,
    )
    self.floor_geoms = torch.tensor(
      [
        i
        for i, body in enumerate(robot.model.geom_bodyid)
        if body == 0 and robot.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE
      ],
      device=device,
      dtype=torch.long,
    )
    # Randomizing body mass invalidates body_subtreemass, and only
    # set_const_fixed recomputes it -- a call that also syncs body_gravcomp to
    # the host, which would stall every reset. Summing mass over an ancestor
    # mask reproduces the same recursion on device: ancestors[child, body]
    # marks body as an ancestor of child, or child itself.
    ancestors = np.zeros((robot.model.nbody, robot.model.nbody), dtype=np.float32)
    for body in range(robot.model.nbody):
      ancestor = body
      while True:
        ancestors[body, ancestor] = 1.0
        if ancestor == 0:
          break
        ancestor = int(robot.model.body_parentid[ancestor])
    if not np.allclose(
      robot.model.body_mass @ ancestors, robot.model.body_subtreemass, atol=1e-6
    ):
      raise ValueError("The ancestor mask does not reproduce MuJoCo's subtree masses")
    self.ancestors = torch.tensor(ancestors, device=device)
    self.default_mass = torch.tensor(
      robot.model.body_mass, device=device, dtype=torch.float32
    )
    self.default_ipos = torch.tensor(
      robot.model.body_ipos, device=device, dtype=torch.float32
    )
    self.default_solref = torch.tensor(
      robot.model.geom_solref, device=device, dtype=torch.float32
    )
    self.cpu_models: list[mujoco.MjModel] = []
    self.cpu_data: list[mujoco.MjData] = []
    self._step_graph = None
    if device.startswith("cuda"):
      configure_cuda_compilation()
      with wp.ScopedDevice(device):
        self.warp_model = mw.put_model(robot.model)
        self.warp_data = mw.put_data(
          robot.model,
          mujoco.MjData(robot.model),
          nworld=num_envs,
          nconmax=256,
          # Contact rows plus joint friction can exceed 512 on fallen Q1 poses.
          njmax=2048,
        )
        self.ipos = self.default_ipos.repeat(num_envs, 1, 1)
        self.friction = torch.tensor(
          robot.model.geom_friction, device=device, dtype=torch.float32
        ).repeat(num_envs, 1, 1)
        self.solref = self.default_solref.repeat(num_envs, 1, 1)
        self.mass = self.default_mass.repeat(num_envs, 1)
        self.subtreemass = (self.default_mass @ self.ancestors).repeat(num_envs, 1)
        self.warp_model.body_ipos = wp.from_torch(self.ipos, dtype=wp.vec3)
        self.warp_model.geom_friction = wp.from_torch(self.friction, dtype=wp.vec3)
        self.warp_model.geom_solref = wp.from_torch(self.solref, dtype=wp.vec2)
        self.warp_model.body_mass = wp.from_torch(self.mass)
        self.warp_model.body_subtreemass = wp.from_torch(self.subtreemass)
        # COM offsets and link masses change qpos0 inertia and constraint
        # weights. Every output written by set_const_0 needs a private row for
        # each world; put_model otherwise broadcasts one shared row and
        # concurrent writes race.
        for field in (
          "dof_invweight0",
          "body_invweight0",
          "actuator_acc0",
          "actuator_biasprm",
          "tendon_length0",
          "tendon_invweight0",
          "cam_pos0",
          "cam_poscom0",
          "cam_mat0",
          "light_pos0",
          "light_poscom0",
          "light_dir0",
        ):
          values = getattr(self.warp_model, field)
          setattr(
            self.warp_model,
            field,
            wp.array(np.repeat(values.numpy(), num_envs, axis=0), dtype=values.dtype),
          )
        self.warp_model.stat.meaninertia = wp.array(
          np.repeat(self.warp_model.stat.meaninertia.numpy(), num_envs),
          dtype=wp.float32,
        )
        self.qpos = wp.to_torch(self.warp_data.qpos)
        self.qvel = wp.to_torch(self.warp_data.qvel)
    else:
      self.cpu_models = [copy.copy(robot.model) for _ in range(num_envs)]
      self.cpu_data = [mujoco.MjData(m) for m in self.cpu_models]
      self.qpos = torch.tensor(
        np.stack([d.qpos for d in self.cpu_data]), dtype=torch.float32
      )
      self.qvel = torch.zeros(num_envs, robot.model.nv)
    self.transition_state = self.state()

  def _stream(self) -> wp.ScopedStream:
    return wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.device)))

  def _reclaim_cuda_cache(self) -> None:
    # Warp graph launches allocate temporary physics buffers independently of
    # Torch. Both reset and step need headroom after a large BPTT update.
    free, _ = torch.cuda.mem_get_info(self.device)
    cached = torch.cuda.memory_reserved(self.device) - torch.cuda.memory_allocated(
      self.device
    )
    if free < 2 * 1024**3 and cached > 512 * 1024**2:
      torch.cuda.empty_cache()

  @torch.no_grad()
  def reset(self, ids: Tensor, states: Tensor, randomize: bool = True) -> None:
    layout = self.layout
    if randomize:
      new_ids = ids[~self.startup_sampled[ids]]
      if len(new_ids):
        self.startup_context[new_ids] = sample_context(
          self.randomization,
          layout,
          len(new_ids),
          self.device,
        )
        self.startup_sampled[new_ids] = True
      context = self.startup_context[ids].clone()
    else:
      context = nominal_context(layout, len(ids), self.device)
    self.context[ids] = context
    self.push_enabled[ids] = randomize
    self.push_timer[ids] = torch.empty(len(ids), device=self.device).uniform_(
      *self.randomization.push_interval_seconds
    )
    self.pushed[ids] = False
    self.lagged[ids] = False
    self.last_action[ids] = 0
    if self.robot.modeled_action_delay:
      self.context[ids, layout.previous_executed_action] = 0
    mass = self.default_mass.repeat(len(ids), 1)
    root = self.robot.root_id
    torso = self.robot.torso_id
    mass[:, torso : torso + 1] += context[:, layout.torso_added_mass]
    subtree = mass @ self.ancestors
    center = self.default_ipos[root] + context[:, layout.root_com]
    friction = torch.tensor(
      self.robot.model.geom_friction[:, 0], device=self.device, dtype=torch.float32
    ).repeat(len(ids), 1)
    if randomize:
      friction[:, self.foot_geoms] = context[:, layout.friction]
      # Contact friction takes max(foot, floor); let the requested minimum bite.
      friction[:, self.floor_geoms] = friction[:, self.floor_geoms].clamp_max(
        self.randomization.friction[0]
      )
    velocity = torch.cat(
      (
        states[:, 29:32],
        quat_apply(quat_inverse(states[:, 3:7]), states[:, 32:35]),
        states[:, 35:57],
      ),
      dim=-1,
    )
    if self.device.startswith("cuda"):
      # Torch's freed BPTT tensors remain in its caching allocator, invisible
      # to Warp's independent allocator. Reclaim only under device pressure.
      self._reclaim_cuda_cache()
      with self._stream():
        self.qpos[ids] = states[:, :29]
        self.qvel[ids] = velocity
        wp.to_torch(self.warp_data.qacc_warmstart)[ids] = 0
        wp.to_torch(self.warp_data.ctrl)[ids] = 0
        wp.to_torch(self.warp_data.time)[ids] = 0
        self.mass[ids] = mass
        self.subtreemass[ids] = subtree
        self.ipos[ids, root] = center
        self.friction[ids, :, 0] = friction
        # This API restores qpos and leaves qvel intact. Scratch kinematics
        # refer to qpos0; step() recomputes forward at the actual reset pose
        # before integrating. An extra forward here repeats collision/solver
        # work and allocates large temporary buffers for no observable state.
        if self.capture_reset and self._const_graph is not None:
          wp.capture_launch(self._const_graph)
        else:
          mw.set_const_0(self.warp_model, self.warp_data)
          if self.capture_reset:
            capture_stream = wp.Stream(self.device)
            with wp.ScopedStream(capture_stream):
              with wp.ScopedCapture(
                device=self.device, stream=capture_stream
              ) as capture:
                mw.set_const_0(self.warp_model, self.warp_data)
            self._const_graph = capture.graph
    else:
      for i, index in enumerate(ids.tolist()):
        m, d = self.cpu_models[index], self.cpu_data[index]
        mujoco.mj_resetData(m, d)
        m.body_mass[:] = mass[i].numpy()
        m.body_subtreemass[:] = subtree[i].numpy()
        m.body_ipos[root] = center[i].numpy()
        m.geom_friction[:, 0] = friction[i].numpy()
        mujoco.mj_setConst(m, d)
        d.qpos[:] = states[i, :29].numpy()
        d.qvel[:] = velocity[i].numpy()
        mujoco.mj_forward(m, d)
      self._sync_cpu()

  def _sync_cpu(self) -> None:
    self.qpos.copy_(torch.from_numpy(np.stack([d.qpos for d in self.cpu_data])))
    self.qvel.copy_(torch.from_numpy(np.stack([d.qvel for d in self.cpu_data])))

  def state(self) -> Tensor:
    return torch.cat(
      (
        self.qpos,
        self.qvel[:, :3],
        quat_apply(self.qpos[:, 3:7], self.qvel[:, 3:6]),
        self.qvel[:, 6:],
        self.last_action,
      ),
      -1,
    )

  @torch.no_grad()
  def step(self, actions: Tensor) -> Tensor:
    if not torch.isfinite(actions).all():
      raise FloatingPointError("Nonfinite Q1 actions")
    layout = self.layout
    kp = self.kp * self.context[:, layout.kp_factor]
    kd = self.kd * self.context[:, layout.kd_factor]
    # The episode's own calibrated zero, so the setpoint and the observation of
    # Table A.1 are measured against the same pose.
    zero = self.robot.default_q + self.context[:, layout.motor_offset]
    target = zero + self.action_scale * actions
    previous = zero + self.action_scale * self.last_action
    if self.robot.modeled_action_delay:
      delay = (
        (self.context[:, layout.delay_s] / self.robot.model.opt.timestep).round().long()
      )
    else:
      delay = torch.zeros(self.num_envs, 1, dtype=torch.long, device=self.device)
    index = self.robot.actuator_qindex
    if self.device.startswith("cuda"):
      self._reclaim_cuda_cache()
      with self._stream():
        ctrl = wp.to_torch(self.warp_data.ctrl)
        for substep in range(self.substeps):
          applied = torch.where(substep >= delay, target, previous)
          torque = kp * (applied - self.qpos[:, 7:29]) - kd * self.qvel[:, 6:]
          if self.use_tn:
            lower, upper = torque_speed_limits(
              self.qvel[:, 6:],
              self.torque_peaks,
              self.torque_knees,
              self.torque_cutoffs,
            )
            torque = torque.maximum(lower).minimum(upper)
          motor = torque[:, index] / self.robot.actuator_gear
          ctrl.copy_(
            motor.clamp(self.robot.ctrl_limits[:, 0], self.robot.ctrl_limits[:, 1])
          )
          if self._step_graph is None:
            # Warm kernels once before capturing; capture itself does not step.
            mw.step(self.warp_model, self.warp_data)
            capture_stream = wp.Stream(self.device)
            with wp.ScopedStream(capture_stream):
              with wp.ScopedCapture(
                device=self.device, stream=capture_stream
              ) as capture:
                mw.step(self.warp_model, self.warp_data)
            self._step_graph = capture.graph
          else:
            wp.capture_launch(self._step_graph)
    else:
      target_np, previous_np = target.numpy(), previous.numpy()
      kp_np, kd_np = kp.numpy(), kd.numpy()
      delay_np = delay.numpy()
      indexes = index.cpu().numpy()
      peaks = self.torque_peaks.numpy()
      knees = self.torque_knees.numpy()
      slope = peaks / (self.torque_cutoffs.numpy() - knees)
      for i, (m, d) in enumerate(zip(self.cpu_models, self.cpu_data, strict=True)):
        for substep in range(self.substeps):
          applied = target_np[i] if substep >= delay_np[i, 0] else previous_np[i]
          torque = kp_np[i] * (applied - d.qpos[7:]) - kd_np[i] * d.qvel[6:]
          if self.use_tn:
            upper = np.clip(peaks - slope * (d.qvel[6:] - knees), 0, peaks)
            lower = np.clip(-peaks - slope * (d.qvel[6:] + knees), -peaks, 0)
            torque = np.clip(torque, lower, upper)
          d.ctrl[:] = np.clip(
            torque[indexes] / m.actuator_gear[:, 0],
            m.actuator_ctrlrange[:, 0],
            m.actuator_ctrlrange[:, 1],
          )
          mujoco.mj_step(m, d)
      self._sync_cpu()
    self.last_action.copy_(actions)
    if self.robot.modeled_action_delay:
      self.context[:, layout.previous_executed_action].copy_(actions)
    self.transition_state = self.state()
    # Pushes are exogenous: replay fits this physical transition, then starts
    # a new sampling segment from the perturbed state returned to the actor.
    self._push_velocity()
    result = self.state()
    if not torch.isfinite(result).all():
      raise FloatingPointError("Nonfinite Q1 simulator state")
    return result

  def _push_velocity(self) -> None:
    self.push_timer.sub_(self.dt)
    self.pushed = self.push_enabled & (self.push_timer <= 0)
    impulse = sample_push(self.randomization, self.num_envs, self.device)
    impulse *= self.pushed[:, None]
    self.qvel[:, :3].add_(impulse[:, :3])
    self.qvel[:, 3:6].add_(quat_apply(quat_inverse(self.qpos[:, 3:7]), impulse[:, 3:6]))
    interval = torch.empty_like(self.push_timer).uniform_(
      *self.randomization.push_interval_seconds
    )
    self.push_timer.copy_(torch.where(self.pushed, interval, self.push_timer))
    if self.device == "cpu":
      for i, data in enumerate(self.cpu_data):
        data.qvel[:] = self.qvel[i].numpy()
