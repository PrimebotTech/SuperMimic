"""Full-network CUDA parity before enabling whole-horizon graph execution."""

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch

from supermimic.algorithm import SuperMimic
from supermimic.data import Q1Motions
from supermimic.distributed import Distributed
from supermimic.models import Critic, MotionPolicy, WorldModel
from supermimic.randomization import nominal_context
from supermimic.replay import ReplayBatch
from supermimic.reward_tuning import RewardTuningCfg
from supermimic.robot import Q1Robot


def batch(robot: Q1Robot, motions: Q1Motions, count: int) -> ReplayBatch:
  device = motions.device
  clips = torch.zeros(count, dtype=torch.long, device=device)
  length = int(motions.lengths()[0])
  frames = torch.arange(count, device=device) % max(1, length - 55) + 10
  reference = motions.sequence(clips, frames, torch.arange(9, device=device))
  states, successors = reference[:, :-1].clone(), reference[:, 1:].clone()
  actions = torch.randn(count, 8, 22, device=device) * 0.05
  states[:, 1:, 57:] = actions[:, :-1]
  successors[:, :, 57:] = actions
  context = nominal_context(robot.context_layout, count, device)
  history = robot.observation(
    motions.sequence(clips, frames, torch.arange(-9, 1, device=device)),
    context[:, None],
  )
  history[:, -1, -22:] = -0.03
  return ReplayBatch(
    states,
    successors,
    actions,
    context,
    torch.zeros(count, 8, device=device),
    history,
    clips,
    frames,
  )


def saved(learner: SuperMimic) -> dict[str, Any]:
  result: dict[str, Any] = {}
  for name in ("model", "actor", "critic"):
    module = getattr(learner, name)
    result[name] = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
    result[name + "_grad"] = [
      None if p.grad is None else p.grad.detach().cpu().clone()
      for p in module.parameters()
    ]
    optimizer = getattr(learner, name + "_optimizer")
    result[name + "_adam"] = {
      key: {
        k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else copy.copy(v)
        for k, v in state.items()
      }
      for key, state in optimizer.state_dict()["state"].items()
    }
  result["rng"] = torch.cuda.get_rng_state().clone()
  return result


def compare(left: Any, right: Any, name: str, maximum: dict[str, float]) -> None:
  if isinstance(left, torch.Tensor):
    assert isinstance(right, torch.Tensor)
    if name == "rng":
      assert torch.equal(left, right), "CUDA random streams differ"
    else:
      torch.testing.assert_close(left, right, rtol=1e-4, atol=3e-6)
      if left.numel():
        maximum[name] = float((left.double() - right.double()).abs().max())
  elif isinstance(left, dict):
    assert left.keys() == right.keys()
    for key in left:
      compare(left[key], right[key], f"{name}.{key}" if name else str(key), maximum)
  elif isinstance(left, list):
    assert len(left) == len(right)
    for i, (a, b) in enumerate(zip(left, right, strict=True)):
      compare(a, b, f"{name}.{i}", maximum)
  elif isinstance(left, float):
    torch.testing.assert_close(
      torch.tensor(left), torch.tensor(right), rtol=1e-4, atol=3e-6
    )
  else:
    assert left == right


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--robot-xml", type=Path, required=True)
  parser.add_argument("--motion", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--batch-size", type=int, default=512)
  parser.add_argument("--updates", type=int, default=2)
  parser.add_argument("--combined", action="store_true")
  parser.add_argument("--heading-delay", action="store_true")
  args = parser.parse_args()
  torch.cuda.set_device(0)
  torch.backends.cuda.matmul.allow_tf32 = True
  robot = Q1Robot(
    args.robot_xml,
    reward_scaling="q1_proportional",
    modeled_action_delay=args.heading_delay,
  ).to("cuda:0")
  motions = Q1Motions(
    robot, [args.motion], device="cuda:0", storage="device-fp16", memory_budget_gb=2
  )
  motions.prepare()
  learners = []
  for captured in (False, True):
    torch.manual_seed(123)
    learner = SuperMimic(
      robot,
      motions,
      WorldModel(
        robot,
        activation="silu" if args.heading_delay else "elu",
        representation="heading" if args.heading_delay else "legacy",
      ).cuda(),
      MotionPolicy(robot, activation="silu" if args.heading_delay else "elu").cuda(),
      Critic(robot, activation="silu" if args.heading_delay else "elu").cuda(),
      cache_reference_heads=True,
      cache_goals=True,
      fuse_model_loss=True,
      distributed=Distributed.detached(validation_mode="per_update"),
      clean_policy_actions=True,
      normalize_policy_objective=True,
      normalize_model_objective=True,
      reward_tuning=RewardTuningCfg(arm_joint_weight=0.05),
      capture_imagination=captured,
      compile_critic=args.combined and captured,
      fused_optimizers=args.combined and captured,
    )
    learner.distributed.initialize("cuda:0")
    learner.enable_compile()
    learners.append(learner)
  torch.manual_seed(512)
  inputs = batch(robot, motions, args.batch_size)
  if args.heading_delay:
    inputs.context[:, robot.context_layout.delay_s] = torch.linspace(
      0, 0.016, args.batch_size, device="cuda:0"
    )[:, None]
    inputs.context[:, robot.context_layout.previous_executed_action] = inputs.states[
      :, 0, -22:
    ]
  initial_rng = torch.cuda.get_rng_state().clone()
  maximum: dict[str, float] = {}
  streams = []
  for update in range(args.updates):
    snapshots, metrics = [], []
    print(json.dumps({"state": "update", "update": update}), flush=True)
    for learner in learners:
      torch.cuda.set_rng_state(initial_rng)
      learner.distributed.begin_update()
      model_metrics = learner.fit_model_tensors(inputs)
      policy_metrics = learner.improve_policy(inputs, 5, 4)
      learner.distributed.validate_update()
      torch.cuda.synchronize()
      metrics.append({k: float(v) for k, v in model_metrics.items()} | policy_metrics)
      snapshots.append(saved(learner))
    compare(metrics[0], metrics[1], "metrics", maximum)
    compare(snapshots[0], snapshots[1], "", maximum)
    streams.append(snapshots[1]["rng"])
    initial_rng = snapshots[1]["rng"]
  assert all(
    not torch.equal(a, b) for a, b in zip(streams, streams[1:], strict=False)
  ), "Consecutive updates reused the random stream"
  report = {
    "passed": True,
    "batch_size": args.batch_size,
    "horizon": 32,
    "updates": args.updates,
    "combined": args.combined,
    "random_states_equal": True,
    "checks": "loss, gradients, updated weights, Adam moments, RNG",
    "maximum_absolute_difference": max(maximum.values(), default=0),
    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
    "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps(report), flush=True)


if __name__ == "__main__":
  main()
