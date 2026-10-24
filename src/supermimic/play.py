"""Watch a trained Q1 policy track a reference motion, on screen or to disk."""

import json
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

import mujoco
import mujoco.viewer
import numpy as np
import torch
import tyro

from supermimic.data import Q1Motions
from supermimic.models import MotionPolicy, goal_features
from supermimic.robot import Q1Robot
from supermimic.simulator import Q1Simulator


class PlayResult(TypedDict):
  motion: str
  loops: int
  played_steps: int
  reference_steps: int
  failed: bool
  max_height_error: float
  max_tilt_error: float
  normalized_return: float
  output: str | None


@dataclass
class PlayCfg:
  checkpoint: Path
  """A checkpoint written by supermimic-train."""
  motion: Path | None = None
  """A Q1 motion NPZ to track; defaults to one the checkpoint trained on."""
  motion_index: int = 0
  """Which of the checkpoint's own motions to track when --motion is absent."""
  output: Path | None = None
  """Write the rollout, and the reference beside it, to this NPZ."""
  viewer: Literal["auto", "on", "off"] = "auto"
  """Open a MuJoCo window; "auto" does so only when a display is available."""
  device: str = "cpu"
  speed: float = 1.0
  """Playback rate relative to real time. Only the window is paced."""
  loops: int = 1
  randomize: bool = False
  """Draw one episode of domain randomization instead of the nominal model."""
  stop_on_failure: bool = True
  """Stop at the paper's termination thresholds instead of playing on."""


def display_available() -> bool:
  return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


@torch.no_grad()
def play(cfg: PlayCfg) -> PlayResult:
  if cfg.speed <= 0 or cfg.loops < 1:
    raise ValueError("Playback speed must be positive and loops at least one")
  if cfg.motion_index < 0:
    raise ValueError("Motion index must be nonnegative")
  show = display_available() if cfg.viewer == "auto" else cfg.viewer == "on"
  if show and cfg.device != "cpu":
    raise ValueError("The window draws the CPU model; play on cpu or pass --viewer off")
  if not show and cfg.output is None:
    raise ValueError(
      "Nothing would come of this run: pass --output to save the rollout, "
      "or --viewer on to open a window"
    )
  if cfg.output is not None and cfg.output.exists():
    raise FileExistsError(f"{cfg.output} exists; pick a path that does not")
  torch.set_num_threads(min(torch.get_num_threads(), 4))
  checkpoint = torch.load(cfg.checkpoint, map_location=cfg.device, weights_only=True)
  config, identity = checkpoint["config"], checkpoint["identity"]
  robot = Q1Robot(
    Path(config["robot_xml"]),
    modeled_action_delay=config.get("modeled_action_delay", False),
  ).to(cfg.device)
  if robot.asset_sha256 != identity["asset_sha256"]:
    raise ValueError("Q1 XML/mesh identity differs from the training checkpoint")
  robot.load_state_dict(checkpoint["robot"])
  actor = MotionPolicy(
    robot,
    tuple(identity["actor_hidden"]),
    activation=config.get("network_activation", "elu"),
  ).to(cfg.device)
  actor.load_state_dict(checkpoint["actor"])
  actor.eval()
  if cfg.motion is not None:
    path = cfg.motion
  else:
    trained = identity["motion_paths"]
    if cfg.motion_index >= len(trained):
      raise ValueError(f"The checkpoint holds {len(trained)} motions; index is lower")
    path = Path(trained[cfg.motion_index])
  motions = Q1Motions(robot, [path])
  motion = motions.load(0).to(cfg.device)
  simulator = Q1Simulator(
    robot,
    1,
    cfg.device,
    config["action_scale"],
    use_tn=config.get("use_tn", False),
    physics_timestep=config.get("physics_timestep"),
  )
  ids = torch.tensor([0], device=cfg.device)
  dt = motions.dt
  executed: list[np.ndarray] = []
  reference: list[np.ndarray] = []
  states: list[np.ndarray] = []
  rewards: list[float] = []
  failures: list[bool] = []
  total_reward, max_height_error, max_tilt_error = 0.0, 0.0, 0.0
  failed, played, loops = False, 0, 0

  simulator.reset(ids, motion[:1], cfg.randomize)
  states.append(simulator.state()[0].cpu().numpy())
  window = (
    mujoco.viewer.launch_passive(simulator.cpu_models[0], simulator.cpu_data[0])
    if show
    else nullcontext()
  )
  with window as handle:
    for _ in range(cfg.loops):
      if loops:
        simulator.reset(ids, motion[:1], cfg.randomize)
      loops += 1
      observation = robot.observation(simulator.state(), simulator.context)
      history = observation[:, None].expand(-1, 10, -1).clone()
      for frame in range(len(motion) - 1):
        if handle is not None and not handle.is_running():
          break
        clock = time.perf_counter()
        offsets = torch.arange(frame + 1, frame + 11, device=cfg.device).clamp(
          max=len(motion) - 1
        )
        references = motion[offsets][None]
        state = simulator.state()
        action = actor(history, goal_features(robot, references))
        next_state = simulator.step(action)
        reward = float(robot.reward(state, next_state, references[:, 0], action, dt))
        total_reward += reward
        rewards.append(reward)
        states.append(next_state[0].cpu().numpy())
        failures.append(bool(robot.terminated(next_state, references[:, 0]).item()))
        errors = robot.termination_features(next_state, references[:, 0])
        max_height_error = max(max_height_error, float(errors[0, 0]))
        max_tilt_error = max(max_tilt_error, float(errors[0, 1]))
        executed.append(next_state[0, :29].cpu().numpy())
        reference.append(references[0, 0, :29].cpu().numpy())
        played += 1
        if handle is not None:
          _draw_reference(handle, reference[-1][:3])
          handle.sync()
          remaining = dt / cfg.speed - (time.perf_counter() - clock)
          if remaining > 0:
            time.sleep(remaining)
        if failures[-1]:
          failed = True
          if cfg.stop_on_failure:
            break
        observation = robot.observation(next_state, simulator.context)
        history = torch.cat((history[:, 1:], observation[:, None]), dim=1)
      if failed and cfg.stop_on_failure:
        break
      if handle is not None and not handle.is_running():
        break

  if cfg.output is not None:
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    # The same schema the trainer reads, so a rollout can be replayed as a
    # motion, with the tracked reference stored alongside it for comparison.
    np.savez_compressed(
      cfg.output,
      robot_name="q1_v2",
      root_quat_order="wxyz",
      fps=1 / dt,
      robot_joint_names=np.array(robot.joint_names),
      qpos=np.stack(executed).astype(np.float32),
      reference_qpos=np.stack(reference).astype(np.float32),
      states=np.stack(states).astype(np.float32),
      step_rewards=np.array(rewards, dtype=np.float32),
      step_failures=np.array(failures, dtype=np.bool_),
    )
  result: PlayResult = {
    "motion": path.name,
    "loops": loops,
    "played_steps": played,
    "reference_steps": len(motion) - 1,
    "failed": failed,
    "max_height_error": max_height_error,
    "max_tilt_error": max_tilt_error,
    "normalized_return": total_reward / (5 * dt * played) if played else 0.0,
    "output": str(cfg.output.resolve()) if cfg.output is not None else None,
  }
  print(json.dumps(result), flush=True)
  return result


def _draw_reference(handle: Any, position: np.ndarray) -> None:
  """Mark the reference root, so drift is visible without a second robot."""
  scene = handle.user_scn
  if not scene.maxgeom:
    return
  scene.ngeom = 1
  mujoco.mjv_initGeom(
    scene.geoms[0],
    int(mujoco.mjtGeom.mjGEOM_SPHERE),
    np.full(3, 0.04),
    position.astype(np.float64),
    np.eye(3).flatten(),
    np.array([0.25, 0.8, 1.0, 0.45], dtype=np.float32),
  )


def main() -> None:
  play(tyro.cli(PlayCfg))


if __name__ == "__main__":
  main()
