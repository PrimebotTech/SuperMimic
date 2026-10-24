"""Deterministic full-clip Q1 evaluation with the paper's failure thresholds."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import torch
import tyro

from supermimic.data import Q1Motions
from supermimic.models import MotionPolicy, goal_features
from supermimic.robot import Q1Robot
from supermimic.simulator import Q1Simulator


class ClipResult(TypedDict):
  motion: str
  failed: bool
  evaluated_steps: int
  reference_steps: int
  max_height_error: float
  max_tilt_error: float
  normalized_return: float


class EvaluationResult(TypedDict):
  robot: str
  checkpoint: str
  motions: int
  failures: int
  mean_normalized_return: float
  results: list[ClipResult]


@dataclass
class EvalCfg:
  checkpoint: Path
  output: Path
  device: str = "cpu"
  max_motions: int = 8


@torch.no_grad()
def evaluate(cfg: EvalCfg) -> EvaluationResult:
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
  paths = [Path(p) for p in identity["motion_paths"]]
  if cfg.max_motions < 0:
    raise ValueError("Motion limit must be nonnegative")
  if cfg.max_motions:
    paths = paths[: cfg.max_motions]
  motions = Q1Motions(robot, paths, device=cfg.device)
  simulator = Q1Simulator(
    robot,
    1,
    cfg.device,
    config["action_scale"],
    use_tn=config.get("use_tn", False),
    physics_timestep=config.get("physics_timestep"),
  )
  rows: list[ClipResult] = []
  for clip, path in enumerate(paths):
    motion = motions.load(clip).to(cfg.device)
    ids = torch.tensor([0], device=cfg.device)
    simulator.reset(ids, motion[:1], randomize=False)
    obs = robot.observation(simulator.state(), simulator.context)
    history = obs[:, None].expand(-1, 10, -1).clone()
    total_reward, max_height_error, max_tilt_error = 0.0, 0.0, 0.0
    failed = False
    steps = 0
    for frame in range(len(motion) - 1):
      offsets = torch.arange(frame + 1, frame + 11, device=cfg.device).clamp(
        max=len(motion) - 1
      )
      references = motion[offsets][None]
      state = simulator.state()
      action = actor(history, goal_features(robot, references))
      next_state = simulator.step(action)
      total_reward += float(
        robot.reward(state, next_state, references[:, 0], action, 0.02)
      )
      errors = robot.termination_features(next_state, references[:, 0])
      max_height_error = max(max_height_error, float(errors[0, 0]))
      max_tilt_error = max(max_tilt_error, float(errors[0, 1]))
      steps += 1
      if bool(robot.terminated(next_state, references[:, 0]).item()):
        failed = True
        break
      obs = robot.observation(next_state, simulator.context)
      history = torch.cat((history[:, 1:], obs[:, None]), dim=1)
    row: ClipResult = {
      "motion": path.name,
      "failed": failed,
      "evaluated_steps": steps,
      "reference_steps": len(motion) - 1,
      "max_height_error": max_height_error,
      "max_tilt_error": max_tilt_error,
      "normalized_return": total_reward / (0.1 * (len(motion) - 1)),
    }
    rows.append(row)
    print(json.dumps(row), flush=True)
  result: EvaluationResult = {
    "robot": "q1_v2",
    "checkpoint": str(cfg.checkpoint.resolve()),
    "motions": len(rows),
    "failures": sum(r["failed"] for r in rows),
    "mean_normalized_return": sum(r["normalized_return"] for r in rows) / len(rows),
    "results": rows,
  }
  cfg.output.parent.mkdir(parents=True, exist_ok=True)
  with cfg.output.open("x") as stream:
    stream.write(json.dumps(result, indent=2) + "\n")
  return result


def main() -> None:
  evaluate(tyro.cli(EvalCfg))


if __name__ == "__main__":
  main()
