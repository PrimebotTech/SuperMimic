"""CPU training integration, compiled gradients, and new-experiment warm starts."""

import copy
import hashlib
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from test_q1_reward_scaling import write_robot

from supermimic.algorithm import SuperMimic
from supermimic.collection import Q1Collector
from supermimic.data import Q1Motions
from supermimic.models import Critic, MotionPolicy, WorldModel, frozen, goal_features
from supermimic.randomization import nominal_context
from supermimic.reward_tuning import RewardTuningCfg, tuned_reward
from supermimic.robot import Q1Robot
from supermimic.simulator import Q1Simulator
from supermimic.train import TrainCfg, train
from supermimic.warm_start import load_warm_start, restore_warm_start


@pytest.fixture
def environment(tmp_path: Path) -> tuple[Q1Robot, Q1Motions]:
  xml = write_robot(tmp_path / "q1.xml")
  tree = ET.parse(xml)
  ET.SubElement(tree.getroot(), "option", timestep="0.005", gravity="0 0 0")
  for geom in tree.findall(".//geom"):
    geom.set("contype", "0")
    geom.set("conaffinity", "0")
  for joint in tree.findall(".//joint"):
    joint.set("armature", "0.01")
  tree.write(xml)
  robot = Q1Robot(xml)
  directory = tmp_path / "motions"
  directory.mkdir()
  qpos = np.repeat(robot.model.qpos0[None], 150, axis=0).astype(np.float32)
  qpos[:, 21:29] += 0.03 * np.sin(np.arange(150)[:, None] / 25)
  path = directory / "reference.npz"
  np.savez(
    path,
    qpos=qpos,
    robot_name="q1_v2",
    root_quat_order="wxyz",
    robot_joint_names=np.array(robot.joint_names),
    fps=50.0,
  )
  return robot, Q1Motions(robot, [path], storage="device", device="cpu")


def small_learner(
  robot: Q1Robot, motions: Q1Motions, tuning: RewardTuningCfg
) -> SuperMimic:
  return SuperMimic(
    robot,
    motions,
    WorldModel(robot, (16, 16)),
    MotionPolicy(robot, (16, 16)),
    Critic(robot, (16, 16)),
    horizon=4,
    reward_tuning=tuning,
    cache_reference_heads=True,
    cache_goals=True,
  )


@pytest.mark.parametrize(
  "tuning",
  [
    RewardTuningCfg(),
    RewardTuningCfg(arm_joint_weight=0.05),
    RewardTuningCfg(arm_pd_target_weight=0.05),
    RewardTuningCfg(body_orientation_std=0.6),
  ],
)
def test_collection_and_imagination_share_tuned_formula_and_compiled_gradients(
  environment: tuple[Q1Robot, Q1Motions],
  tuning: RewardTuningCfg,
):
  robot, motions = environment
  torch.manual_seed(331)
  learner = small_learner(robot, motions, tuning)
  state = motions.load(0)[:2].clone().requires_grad_()
  refs = motions.load(0)[1:11][None].expand(2, -1, -1)
  context = nominal_context(robot.context_layout, 2, "cpu")
  context[:, robot.context_layout.motor_offset] = 0.12
  history = (
    robot.observation(state, context)[:, None]
    .expand(-1, 10, -1)
    .clone()
    .requires_grad_()
  )
  goal = goal_features(robot, refs)
  heads = robot.kinematics(refs[:, 0])
  encoding = learner.actor.encode_goal(goal)
  collector = Q1Collector(robot, learner.actor, 0.02, reward_tuning=tuning)
  with frozen(learner.model, learner.critic):
    eager = learner._imagined_step(state, history, refs, context, heads, goal, encoding)
    action = learner.actor.decode(history, encoding)
    expected = tuned_reward(
      robot,
      state,
      eager[0],
      refs[:, 0],
      action,
      0.02,
      tuning=tuning,
      context=context,
      target_heads=heads,
    )
    torch.testing.assert_close(eager[1], expected, rtol=0, atol=0)
    collected = collector.heads(state, eager[0], refs[:, 0], action, context)
    torch.testing.assert_close(collected[0], expected, rtol=0, atol=0)
    torch.testing.assert_close(collected[1], robot.terminated(eager[0], refs[:, 0]))
    before = torch.autograd.grad(
      eager[1].sum(), (state, history, encoding), retain_graph=True
    )
    learner.enable_compile(backend="aot_eager")
    compiled = learner.imagined_step(
      state, history, refs, context, heads, goal, encoding
    )
    after = torch.autograd.grad(
      compiled[1].sum(), (state, history, encoding), retain_graph=True
    )
    for a, b in zip(eager, compiled, strict=True):
      torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)
    for a, b in zip(before, after, strict=True):
      torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-7)
    collector.enable_compile(backend="aot_eager")
    compiled_collection = collector.heads(state, eager[0], refs[:, 0], action, context)
    torch.testing.assert_close(compiled_collection[0], expected, rtol=1e-5, atol=1e-7)
  assert all(p.grad is None for p in learner.model.parameters())
  assert all(p.grad is None for p in learner.critic.parameters())


def physical_identity(robot: Q1Robot) -> dict[str, Any]:
  simulator = Q1Simulator(robot, 1, "cpu", physics_timestep=0.005, use_tn=True)
  return {
    "asset_sha256": robot.asset_sha256,
    "joint_names": robot.joint_names,
    "state_dim": robot.state_dim,
    "obs_dim": robot.obs_dim,
    "context_dim": robot.context_dim,
    "model_hidden": (16, 16),
    "actor_hidden": (16, 16),
    "control_dt": 0.02,
    "kp": simulator.kp.tolist(),
    "kd": simulator.kd.tolist(),
    "default_q": robot.default_q.tolist(),
    "physics_timestep": 0.005,
    "torque_speed_peak_nm": simulator.torque_peaks.tolist(),
    "torque_speed_knee_rad_s": simulator.torque_knees.tolist(),
    "torque_speed_cutoff_rad_s": simulator.torque_cutoffs.tolist(),
  }


def test_warm_start_copies_only_actor_dynamics_and_preserves_new_shared_robot(
  environment: tuple[Q1Robot, Q1Motions],
  tmp_path: Path,
):
  robot, motions = environment
  old = small_learner(robot, motions, RewardTuningCfg())
  new = small_learner(robot, motions, RewardTuningCfg(arm_joint_weight=0.05))
  new.actor.shared_robot = robot
  source = {
    "actor": copy.deepcopy(old.actor.state_dict()),
    "model": copy.deepcopy(old.model.state_dict()),
    "identity": physical_identity(robot),
    "config": {"action_scale": 0.5, "use_tn": True},
    "iteration": 25000,
    "world_size": 8,
    "checkpoint_version": 1,
    "rank_directory": "absent",
    "critic": {"bad": torch.ones(3)},
    "actor_optimizer": {"bad": True},
  }
  source["actor"]["shared_robot._extra_state"] = {"bad_old_reward": True}
  path = tmp_path / "source.pt"
  torch.save(source, path)
  termination = copy.deepcopy(new.model.termination.state_dict())
  critic = copy.deepcopy(new.critic.state_dict())
  identity = robot.reward_identity()
  loaded = load_warm_start(path)
  restore_warm_start(
    loaded,
    physical_identity(robot),
    new.model,
    new.actor,
    action_scale=0.5,
    use_tn=True,
  )
  for name, tensor in old.model.dynamics.state_dict().items():
    assert torch.equal(new.model.dynamics.state_dict()[name], tensor)
  for name, tensor in old.actor.state_dict().items():
    assert torch.equal(new.actor.state_dict()[name], tensor)
  for name, tensor in termination.items():
    assert torch.equal(new.model.termination.state_dict()[name], tensor)
  for name, tensor in critic.items():
    assert torch.equal(new.critic.state_dict()[name], tensor)
  assert robot.reward_identity() == identity
  for optimizer in [new.actor_optimizer, new.model_optimizer, new.critic_optimizer]:
    assert optimizer.state_dict()["state"] == {}


@pytest.mark.parametrize(
  "field,changed",
  [
    ("asset_sha256", "other"),
    ("obs_dim", 73),
    ("model_hidden", (32, 32)),
    ("actor_hidden", (32, 32)),
    ("physics_timestep", 0.001),
  ],
)
def test_warm_start_rejects_changed_physical_or_input_semantics(
  environment: tuple[Q1Robot, Q1Motions],
  field: str,
  changed: Any,
):
  robot, motions = environment
  learner = small_learner(robot, motions, RewardTuningCfg())
  identity = physical_identity(robot)
  source = {
    "identity": identity | {field: changed},
    "config": {"action_scale": 0.5, "use_tn": True},
    "model": learner.model.state_dict(),
    "actor": learner.actor.state_dict(),
  }
  with pytest.raises(ValueError, match="identity changed|timestep changed"):
    restore_warm_start(
      source, identity, learner.model, learner.actor, action_scale=0.5, use_tn=True
    )
  with pytest.raises(ValueError, match="action scale changed"):
    restore_warm_start(
      source | {"identity": identity},
      identity,
      learner.model,
      learner.actor,
      action_scale=0.3,
      use_tn=True,
    )


def training_config(robot: Q1Robot, motions: Q1Motions, output: Path) -> TrainCfg:
  return TrainCfg(
    robot.xml,
    motions.paths[0].parent,
    output,
    device="cpu",
    num_envs=2,
    iterations=1,
    replay_steps=64,
    model_batch_size=2,
    policy_batch_size=2,
    model_batches=1,
    critic_epochs=1,
    critic_minibatches=1,
    randomize=False,
    action_noise=0.0,
    small_networks=True,
    save_interval=1,
    physics_timestep=0.005,
    checkpoint_replay=True,
  )


def test_new_reward_new_data_warm_start_is_a_fresh_cpu_training_run(
  environment: tuple[Q1Robot, Q1Motions],
  tmp_path: Path,
):
  robot, motions = environment
  cfg = training_config(robot, motions, tmp_path / "baseline")
  original = train(cfg)
  saved = torch.load(original, map_location="cpu", weights_only=True)
  source = copy.deepcopy(saved)
  source["iteration"], source["global_env_steps"] = 25000, 12345678
  source["world_size"], source["rank_directory"] = 8, "absent_rank_shards"
  for name in ("actor", "model", "critic"):
    for state in source[f"{name}_optimizer"]["state"].values():
      state["step"] = torch.tensor(999.0)
  changed = tmp_path / "weights_only.pt"
  torch.save(source, changed)
  new_data = tmp_path / "new_data"
  new_data.mkdir()
  new_motion = new_data / "new_reference.npz"
  new_motion.write_bytes(motions.paths[0].read_bytes())
  experiment = replace(
    cfg,
    output_dir=tmp_path / "experiment",
    motion_dir=new_data,
    warm_start=changed,
    reward_arm_joint_weight=0.05,
  )
  result = torch.load(train(experiment), map_location="cpu", weights_only=True)
  initialization = result["identity"]["initialization"]
  assert initialization["source_iteration"] == 25000
  assert (
    initialization["source_checkpoint_sha256"]
    == hashlib.sha256(changed.read_bytes()).hexdigest()
  )
  assert (
    initialization["initial_iteration"]
    == initialization["initial_environment_steps"]
    == 0
  )
  assert result["iteration"] == 1
  assert result["global_env_steps"] < 12345678
  assert result["identity"]["motion_paths"] != saved["identity"]["motion_paths"]
  assert result["identity"]["reward_tuning"]["arm_joint_weight"] == 0.05
  assert result["config"]["reward_arm_joint_weight"] == 0.05
  for name in ("actor", "model", "critic"):
    assert all(
      int(state["step"]) == 1 for state in result[f"{name}_optimizer"]["state"].values()
    )
  with pytest.raises(ValueError, match="reward tuning changed"):
    train(
      replace(
        cfg,
        output_dir=tmp_path / "invalid_resume",
        iterations=2,
        resume=original,
        reward_arm_joint_weight=0.05,
      )
    )
  assert not (tmp_path / "invalid_resume").exists()
  with pytest.raises(ValueError, match="not both"):
    train(
      replace(
        cfg, output_dir=tmp_path / "invalid_both", resume=original, warm_start=original
      )
    )
