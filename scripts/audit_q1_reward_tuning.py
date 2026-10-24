"""Bounded CPU reward/control audit; never train or modify input artifacts."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, cast

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"

import mujoco
import numpy as np
import torch
from torch import Tensor

import supermimic.robot as robot_module
from supermimic.data import Q1Motions
from supermimic.models import MotionPolicy, WorldModel, goal_features
from supermimic.robot import Q1Robot, rotation_error
from supermimic.simulator import Q1Simulator

CANDIDATES = (
  "baseline",
  "body_orientation_std_0p6",
  "arm_orientation_split_weight2",
  "arm_orientation_split_weight2_std0p6",
  "arm_joint_aux_0p05",
  "arm_pd_target_aux_0p05",
)


def digest(path: Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as stream:
    while block := stream.read(1024 * 1024):
      h.update(block)
  return h.hexdigest()


def reward_terms(
  robot: Q1Robot,
  state: Tensor,
  next_state: Tensor,
  target: Tensor,
  action: Tensor,
  mode: str,
  action_scale: float,
) -> dict[str, Tensor]:
  p, r, v, w = robot.kinematics(next_state)
  tp, tr, tv, tw = robot.kinematics(target)
  s = robot.reward_scaling
  orient = rotation_error(r[..., 1:, :], tr[..., 1:, :])
  orientation_std = 0.6 if mode == "body_orientation_std_0p6" else s.orientation_std
  orientation = (-orient.mean(-1) / orientation_std**2).exp()
  if mode.startswith("arm_orientation_split"):
    mask = robot.arm_links
    arm_std = 0.6 if mode.endswith("std0p6") else s.orientation_std
    arm = (-orient[..., mask].mean(-1) / arm_std**2).exp()
    other = (-orient[..., ~mask].mean(-1) / s.orientation_std**2).exp()
    arm_count, other_count = int(mask.sum()), int((~mask).sum())
    orientation = (2 * arm_count * arm + other_count * other) / (
      2 * arm_count + other_count
    )
  terms = {
    "body_position": (
      -(p[..., 1:, :] - tp[..., 1:, :]).square().sum(-1).mean(-1) / s.position_std**2
    ).exp(),
    "body_orientation": orientation,
    "body_linear_velocity": (
      -(v[..., 1:, :] - tv[..., 1:, :]).square().sum(-1).mean(-1)
      / s.linear_velocity_std**2
    ).exp(),
    "body_angular_velocity": (
      -(w[..., 1:, :] - tw[..., 1:, :]).square().sum(-1).mean(-1)
      / s.angular_velocity_std**2
    ).exp(),
    "root_position": 0.5
    * (-(p[..., 1, :] - tp[..., 1, :]).square().sum(-1) / s.position_std**2).exp(),
    "root_orientation": 0.5
    * (-rotation_error(r[..., 1, :], tr[..., 1, :]) / s.orientation_std**2).exp(),
    "action_smoothness": -0.1
    * s.joint_sum_multiplier
    * (action - state[..., 57:79]).square().sum(-1),
    "joint_limits": -2
    * s.joint_sum_multiplier
    * (
      (next_state[..., 7:29] - robot.joint_limits[:, 1]).clamp_min(0)
      + (robot.joint_limits[:, 0] - next_state[..., 7:29]).clamp_min(0)
    ).sum(-1),
  }
  if mode == "arm_joint_aux_0p05":
    terms["arm_joint_aux"] = -0.05 * (
      (next_state[..., 21:29] - target[..., 21:29]) / 0.8
    ).square().mean(-1)
  if mode == "arm_pd_target_aux_0p05":
    # This is an explicit action prior; no target enters actor observations.
    zero = robot.default_q[14:]
    terms["arm_pd_target_aux"] = -0.05 * (
      (zero + action_scale * action[..., 14:] - target[..., 21:29]) / 0.8
    ).square().mean(-1)
  return terms


def summary(x: Tensor) -> dict[str, Any]:
  rows = x.detach().cpu()
  return {
    "mean": float(rows.mean()),
    "first": rows[0].tolist(),
    "last": rows[-1].tolist(),
    "min": float(rows.min()),
    "max": float(rows.max()),
  }


def gradient_summary(gradient: Tensor, direction: Tensor) -> dict[str, Any]:
  g, d = gradient.detach(), direction.detach()
  cos = (g * d).sum(-1) / (g.norm(dim=-1) * d.norm(dim=-1)).clamp_min(1e-20)
  return {
    "mean_l2": float(g.norm(dim=-1).mean()),
    "first_l2": float(g[0].norm()),
    "last_l2": float(g[-1].norm()),
    "first_components": g[0].tolist(),
    "cosine_with_error_correction": summary(cos),
    "positive_direction_fraction": float(((g * d).sum(-1) > 0).float().mean()),
  }


def make_simulator(robot: Q1Robot, config: dict[str, Any]) -> Q1Simulator:
  return Q1Simulator(
    robot,
    1,
    "cpu",
    config["action_scale"],
    use_tn=config["use_tn"],
    physics_timestep=config["physics_timestep"],
  )


def pose_metrics(robot: Q1Robot, actual: Tensor, reference: Tensor) -> dict[str, Any]:
  model: Any = robot.model
  geom_types: Any = mujoco.mjtGeom
  rows: dict[str, Any] = {}
  for label, state in [("actual", actual), ("reference", reference)]:
    data: Any = mujoco.MjData(model)
    data.qpos[:] = state[:29].numpy()
    mujoco.mj_forward(model, data)
    bottoms = []
    for name in ("left_ankle_roll_link", "right_ankle_roll_link"):
      body = model.body(name).id
      for geom in range(model.ngeom):
        if model.geom_bodyid[geom] != body or not (
          model.geom_contype[geom] or model.geom_conaffinity[geom]
        ):
          continue
        pos, size = data.geom_xpos[geom], model.geom_size[geom]
        row = data.geom_xmat[geom].reshape(3, 3)[2]
        kind = model.geom_type[geom]
        if kind == geom_types.mjGEOM_MESH:
          mesh = model.geom_dataid[geom]
          start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
          vertices = model.mesh_vert[start : start + count]
          bottoms.append(float((vertices @ row + pos[2]).min()))
          continue
        if kind == geom_types.mjGEOM_BOX:
          support = np.dot(np.abs(row), size)
        elif kind == geom_types.mjGEOM_SPHERE:
          support = size[0]
        elif kind == geom_types.mjGEOM_CAPSULE:
          support = size[0] + size[1] * abs(row[2])
        elif kind == geom_types.mjGEOM_CYLINDER:
          support = size[0] * np.linalg.norm(row[:2]) + size[1] * abs(row[2])
        elif kind == geom_types.mjGEOM_ELLIPSOID:
          support = np.linalg.norm(row * size)
        else:
          raise ValueError(f"Unsupported Q1 foot collision geometry {kind}")
        bottoms.append(float(pos[2] - support))
    wrists = {}
    for name in ("left_wrist_yaw_link", "right_wrist_yaw_link"):
      try:
        body = model.body(name).id
      except KeyError:
        continue
      wrists[name] = {
        "world": data.xpos[body].copy(),
        "root_relative": data.xpos[body] - data.xpos[1],
      }
    rows[label] = {"foot_collision_bottom_m": min(bottoms), "wrists": wrists}
  errors = {}
  for name in rows["actual"]["wrists"]:
    errors[name] = {
      key + "_error_m": float(
        np.linalg.norm(
          rows["actual"]["wrists"][name][key] - rows["reference"]["wrists"][name][key]
        )
      )
      for key in ("world", "root_relative")
    }
  return {
    "actual_foot_bottom_m": rows["actual"]["foot_collision_bottom_m"],
    "reference_foot_bottom_m": rows["reference"]["foot_collision_bottom_m"],
    "root_xy_error_m": float((actual[:2] - reference[:2]).norm()),
    "wrist_errors": errors,
  }


def reward_audit(
  robot: Q1Robot,
  model: WorldModel,
  reference: Tensor,
  recorded: Tensor,
  config: dict[str, Any],
  frames: int,
) -> dict[str, Any]:
  state = recorded[:frames].clone()
  actual = recorded[1 : frames + 1].clone()
  target = reference[1 : frames + 1]
  actions = actual[:, 57:79].clone()
  context = make_simulator(robot, config).context.expand(frames, -1)
  context = model.advance_context(context, state[:, 57:79])
  model.requires_grad_(False)
  with torch.no_grad():
    predicted = model(state, actions, context)
  result: dict[str, Any] = {
    "frames": frames,
    "dt": 0.02,
    "gradient_units": "per-control-step reward, including dt=0.02 once",
    "model_arm_qpos_error_rad": summary(
      (predicted[:, 21:29] - actual[:, 21:29]).square().mean(-1).sqrt()
    ),
    "model_arm_qvel_error_rad_per_second": summary(
      (predicted[:, 49:57] - actual[:, 49:57]).square().mean(-1).sqrt()
    ),
    "candidates": {},
  }
  desired = (target[:, 7:29] - robot.default_q) / config["action_scale"] - actions
  qdirection = target[:, 21:29] - actual[:, 21:29]
  for mode in CANDIDATES:
    next_state = actual.clone().requires_grad_()
    action = actions.clone().requires_grad_()
    terms = reward_terms(
      robot, state, next_state, target, action, mode, config["action_scale"]
    )
    rows: dict[str, Any] = {}
    for name, value in terms.items():
      qgrad, agrad = torch.autograd.grad(
        0.02 * value.sum(), (next_state, action), retain_graph=True, allow_unused=True
      )
      rows[name] = {"contribution_before_dt": summary(value)}
      if qgrad is not None:
        rows[name]["actual_state_arm_q_gradient"] = gradient_summary(
          qgrad[:, 21:29], qdirection
        )
      if agrad is not None:
        rows[name]["direct_arm_action_gradient"] = gradient_summary(
          agrad[:, 14:], desired[:, 14:]
        )
    total = 0.02 * cast(Tensor, sum(terms.values()))
    qgrad, direct = torch.autograd.grad(total.sum(), (next_state, action))
    model_action = actions.clone().requires_grad_()
    model_next = model(state, model_action, context)
    model_terms = reward_terms(
      robot, state, model_next, target, model_action, mode, config["action_scale"]
    )
    model_total = 0.02 * cast(Tensor, sum(model_terms.values()))
    tracking = 0.02 * cast(
      Tensor, sum(v for k, v in model_terms.items() if k != "action_smoothness")
    )
    mgrad = torch.autograd.grad(model_total.sum(), model_action, retain_graph=True)[0]
    tgrad = torch.autograd.grad(tracking.sum(), model_action)[0]
    result["candidates"][mode] = {
      "actual_reward_per_step": summary(total),
      "terms": rows,
      "actual_state_arm_q_gradient": gradient_summary(qgrad[:, 21:29], qdirection),
      "direct_arm_action_gradient": gradient_summary(direct[:, 14:], desired[:, 14:]),
      "world_model_total_arm_action_gradient": gradient_summary(
        mgrad[:, 14:], desired[:, 14:]
      ),
      "world_model_tracking_arm_action_gradient": gradient_summary(
        tgrad[:, 14:], desired[:, 14:]
      ),
    }
    if mode == "baseline":
      expected = robot.reward(state, actual, target, actions, 0.02)
      result["baseline_reward_reconstruction_max_error"] = float(
        (total.detach() - expected).abs().max()
      )
  # Finite differences of the actual CPU physics at three recorded states.
  finite_difference: dict[str, Any] = {}
  epsilon = 0.001
  for frame in sorted(set([0, frames // 2, frames - 1])):
    sim = make_simulator(robot, config)
    plus_states, minus_states = [], []
    for joint in range(14, 22):
      pair = []
      for sign in [1, -1]:
        sim.reset(torch.tensor([0]), state[frame : frame + 1], False)
        sim.last_action.copy_(state[frame : frame + 1, 57:79])
        perturb = actions[frame : frame + 1].clone()
        perturb[0, joint] += sign * epsilon
        with torch.no_grad():
          pair.append(sim.step(perturb)[0].clone())
      plus_states.append(pair[0])
      minus_states.append(pair[1])
    grad_rows = {}
    for mode in CANDIDATES:
      gradients, track_gradients = [], []
      for arm in range(8):
        values, tracking_values = [], []
        for sign, next_states in [(1, plus_states), (-1, minus_states)]:
          perturb = actions[frame : frame + 1].clone()
          perturb[0, 14 + arm] += sign * epsilon
          terms = reward_terms(
            robot,
            state[frame : frame + 1],
            next_states[arm][None],
            target[frame : frame + 1],
            perturb,
            mode,
            config["action_scale"],
          )
          values.append(float(0.02 * sum(terms.values())))
          tracking_values.append(
            float(0.02 * sum(v for k, v in terms.items() if k != "action_smoothness"))
          )
        gradients.append((values[0] - values[1]) / (2 * epsilon))
        track_gradients.append(
          (tracking_values[0] - tracking_values[1]) / (2 * epsilon)
        )
      grad_rows[mode] = {
        "total": gradient_summary(
          torch.tensor([gradients]), desired[frame : frame + 1, 14:]
        ),
        "tracking": gradient_summary(
          torch.tensor([track_gradients]), desired[frame : frame + 1, 14:]
        ),
      }
    finite_difference[str(frame)] = grad_rows
  result["actual_physics_action_finite_difference"] = {
    "epsilon_action": epsilon,
    "frames_zero_based": finite_difference,
    "note": "Local sensitivity; saturated motors can have zero tracking derivative.",
  }
  return result


def control_audit(
  robot: Q1Robot,
  reference: Tensor,
  recorded: Tensor,
  config: dict[str, Any],
  frames: int,
) -> dict[str, Any]:
  output: dict[str, Any] = {}
  mujoco_api: Any = mujoco
  original_step = mujoco.mj_step
  for mode in (
    "recorded_actions",
    "reference_all_targets",
    "reference_arm_targets_recorded_legs",
  ):
    sim = make_simulator(robot, config)
    sim.reset(torch.tensor([0]), reference[:1], False)
    states = [sim.state()[0].clone()]
    logs: list[dict[str, np.ndarray]] = []
    current: dict[str, np.ndarray] = {}

    def traced_step(
      model: Any,
      data: Any,
      *args: Any,
      audit_sim: Q1Simulator = sim,
      audit_current: dict[str, np.ndarray] = current,
      audit_logs: list[dict[str, np.ndarray]] = logs,
      **kwargs: Any,
    ) -> Any:
      q, v = data.qpos[7:].copy(), data.qvel[6:].copy()
      raw = (
        audit_sim.kp.numpy() * (audit_current["target"] - q) - audit_sim.kd.numpy() * v
      )
      applied = np.empty(22)
      applied[robot.actuator_qindex.numpy()] = data.ctrl * model.actuator_gear[:, 0]
      audit_logs.append(
        {
          "q": q,
          "v": v,
          "target": audit_current["target"].copy(),
          "raw": raw,
          "applied": applied,
        }
      )
      return original_step(model, data, *args, **kwargs)

    mujoco_api.mj_step = traced_step
    try:
      with torch.no_grad():
        for frame in range(frames):
          action = recorded[frame + 1 : frame + 2, 57:79].clone()
          if mode == "reference_all_targets":
            action = (reference[frame + 1, 7:29] - robot.default_q)[None]
            action /= config["action_scale"]
          elif mode == "reference_arm_targets_recorded_legs":
            action[:, 14:] = reference[frame + 1, 21:29] - robot.default_q[14:]
            action[:, 14:] /= config["action_scale"]
          current["target"] = (
            robot.default_q + config["action_scale"] * action[0]
          ).numpy()
          states.append(sim.step(action)[0].clone())
    finally:
      mujoco_api.mj_step = original_step
    actual = torch.stack(states)
    error = actual[:, 7:29] - reference[: frames + 1, 7:29]
    raw, applied = (
      np.array([x["raw"] for x in logs]),
      np.array([x["applied"] for x in logs]),
    )
    output[mode] = {
      "recorded_physical57_max_error": float(
        (actual[:, :57] - recorded[: frames + 1, :57]).abs().max()
      ),
      "arm_saturation_fraction": (np.abs(raw[:, 14:] - applied[:, 14:]) > 1e-5)
      .mean(0)
      .tolist(),
      "first_substep": {k: v[14:].tolist() for k, v in logs[0].items()},
      "arm_qpos_rmse_by_frame": error[:, 14:].square().mean(-1).sqrt().tolist(),
      "height_error_by_frame": (actual[:, 2] - reference[: frames + 1, 2])
      .abs()
      .tolist(),
      "last_arm_angles": actual[-1, 21:29].tolist(),
      "pose_metrics": {
        str(t): pose_metrics(robot, actual[t], reference[t])
        for t in sorted(set([0, 1, min(5, frames), min(10, frames), frames]))
      },
    }
  f = robot.termination_features(recorded[1:], reference[1:])
  terminated = robot.terminated(recorded[1:], reference[1:])
  indices = terminated.nonzero().flatten()
  first = int(indices[0]) + 1 if len(indices) else None
  height = (f[:, 0] > robot.reward_scaling.termination_height).nonzero().flatten()
  gravity = (f[:, 1] > robot.reward_scaling.termination_tilt).nonzero().flatten()
  output["termination"] = {
    "first_failure_frame": first,
    "first_height_failure_frame": int(height[0]) + 1 if len(height) else None,
    "first_gravity_z_failure_frame": int(gravity[0]) + 1 if len(gravity) else None,
    "first_failure_features": f[first - 1].tolist() if first else None,
    "first_failure_pose_metrics": pose_metrics(robot, recorded[first], reference[first])
    if first
    else None,
    "first_failure_reference_z_step_m": float(
      reference[first, 2] - reference[first - 1, 2]
    )
    if first
    else None,
  }
  output["reference_first_next_arm_angles"] = reference[1, 21:29].tolist()
  violation = (reference[:, 7:29] - robot.joint_limits[:, 1]).clamp_min(0) + (
    robot.joint_limits[:, 0] - reference[:, 7:29]
  ).clamp_min(0)
  output["reference_joint_limit_max_violation_rad"] = float(violation.max())
  delta = reference[1:, 7:29] - reference[:-1, 7:29]
  output["reference_max_joint_step_rad"] = float(delta.abs().max())
  output["reference_max_joint_step_frame"] = (
    int(delta.abs().max(-1).values.argmax()) + 1
  )
  return output


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--reference", type=Path, required=True)
  parser.add_argument("--rollout", type=Path, required=True)
  parser.add_argument("--xml", type=Path)
  parser.add_argument("--frames", type=int, default=25)
  args = parser.parse_args()
  if not 2 <= args.frames <= 50:
    raise ValueError("The CPU audit is bounded to 2..50 control steps")
  torch.set_num_threads(1)
  checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
  config, identity = checkpoint["config"], checkpoint["identity"]
  robot = Q1Robot(
    args.xml or Path(config["robot_xml"]),
    modeled_action_delay=config.get("modeled_action_delay", False),
  )
  robot.load_state_dict(checkpoint["robot"])
  if robot.asset_sha256 != identity["asset_sha256"]:
    raise ValueError("XML/mesh differs from the actual checkpoint")
  reference = Q1Motions(robot, [args.reference], device="cpu", storage="device").load(0)
  with np.load(args.rollout, allow_pickle=False) as archive:
    recorded = torch.from_numpy(archive["states"].copy())
    failures = archive["step_failures"].copy()
  model = WorldModel(
    robot,
    tuple(identity["model_hidden"]),
    activation=config.get("network_activation", "elu"),
    representation=config.get("dynamics_representation", "legacy"),
  )
  model.load_state_dict(checkpoint["model"])
  actor = MotionPolicy(
    robot,
    tuple(identity["actor_hidden"]),
    activation=config.get("network_activation", "elu"),
  )
  actor.load_state_dict(checkpoint["actor"])
  context = make_simulator(robot, config).context
  history = robot.observation(recorded[:1], context)[:, None].expand(-1, 10, -1)
  with torch.no_grad():
    first_action = actor(history, goal_features(robot, reference[1:11][None]))
  result = {
    "provenance": {
      "checkpoint": str(args.checkpoint),
      "checkpoint_sha256": digest(args.checkpoint),
      "reference_sha256": digest(args.reference),
      "rollout_sha256": digest(args.rollout),
      "xml_sha256": robot.sha256,
      "asset_sha256": robot.asset_sha256,
      "robot_source_sha256": digest(Path(robot_module.__file__)),
      "cuda_available": torch.cuda.is_available(),
    },
    "reward_scaling": robot.reward_identity(),
    "joint_names": list(robot.joint_names),
    "arm_body_names": [
      name
      for name, arm in zip(robot.body_names, robot.arm_links.tolist(), strict=True)
      if arm
    ],
    "actor_first_action": first_action[0].tolist(),
    "actor_first_action_saved_max_error": float(
      (first_action[0] - recorded[1, 57:79]).abs().max()
    ),
    "reset_physical57_max_error": float(
      (reference[0, :57] - recorded[0, :57]).abs().max()
    ),
    "control_audit": control_audit(robot, reference, recorded, config, args.frames),
    "reward_audit": reward_audit(
      robot, model, reference, recorded, config, args.frames
    ),
    "failure_flags_match": bool(
      np.array_equal(robot.terminated(recorded[1:], reference[1:]).numpy(), failures)
    ),
    "limitations": [
      "Same-state gradients are not training-quality evidence.",
      "One-step frozen-model gradients omit H=32 value/dynamics/termination terms.",
      "Reference PD targets are a bounded diagnostic, not a deployed controller.",
      "Candidates preserve ideal tracking score 5 and existing termination thresholds.",
      "The baseline is the checkpoint's existing Q1 adaptation; candidates are optional experiments.",
    ],
  }
  print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
  main()
