"""Explicit new-phase semantics, analytic gradients and retained progress."""

import copy
from collections.abc import Generator
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_reward_training import environment as environment
from test_reward_training import small_learner, training_config
from test_runtime_training import make_batch
from torch._dynamo import reset

from supermimic.collection import Q1Collector
from supermimic.data import Q1Motions
from supermimic.models import frozen, goal_features
from supermimic.randomization import RandomizationCfg, nominal_context
from supermimic.reward_tuning import RewardTuningCfg, tuned_reward
from supermimic.robot import Q1Robot
from supermimic.simulator import Q1Simulator
from supermimic.train import train
from supermimic.warm_start import restore_warm_start


@pytest.fixture(autouse=True)
def isolated_compiler_cache() -> Generator[None]:
  """Each semantic comparison gets its own Dynamo specialization budget."""
  reset()
  yield
  reset()


@pytest.mark.parametrize("compiled", [False, True])
def test_clean_reward_history_preserves_executed_physical_state_and_gradients(
  environment: tuple[Q1Robot, Q1Motions], compiled: bool
):
  robot, motions = environment
  torch.manual_seed(821)
  learner = small_learner(robot, motions, RewardTuningCfg(arm_pd_target_weight=0.05))
  learner.clean_policy_actions = True
  batch = make_batch(robot, motions, 1)
  state = batch.states[:, 0].clone()
  state[:, 57:79] = 0.7  # Previous executed command differs from policy history.
  history = batch.history.clone()
  history[:, -1, -22:] = -0.3
  state.requires_grad_()
  history.requires_grad_()
  references = motions.sequence(batch.clips, batch.frames, torch.arange(1, 11))
  goal = goal_features(robot, references)
  encoding = learner.actor.encode_goal(goal)
  heads = robot.kinematics(references[:, 0])
  if compiled:
    learner.enable_compile(backend="aot_eager")
  with frozen(learner.model, learner.critic):
    imagined = learner.imagined_step(
      state, history, references, batch.context, heads, goal, encoding
    )
    action = learner.actor.decode(history, encoding)
    reward_state = torch.cat((state[:, :57], history[:, -1, -22:]), -1)
    expected = tuned_reward(
      robot,
      reward_state,
      imagined[0],
      references[:, 0],
      action,
      0.02,
      tuning=learner.reward_tuning,
      context=batch.context,
      target_heads=heads,
    )
    torch.testing.assert_close(imagined[1], expected, rtol=1e-5, atol=1e-7)
    actual_grad = torch.autograd.grad(imagined[1].sum(), history, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.sum(), history, retain_graph=True)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-4, atol=1e-7)
    collector = Q1Collector(
      robot,
      learner.actor,
      0.02,
      reward_tuning=learner.reward_tuning,
      clean_policy_actions=True,
    )
    if compiled:
      collector.enable_compile(backend="aot_eager")
    executed = action + 0.2
    successor = learner.model(state, executed, batch.context)
    before = successor.detach().clone()
    reward, _, observation = collector.heads(
      state,
      successor,
      references[:, 0],
      action,
      batch.context,
      history[:, -1, -22:],
    )
    expected_collection = tuned_reward(
      robot,
      reward_state,
      successor,
      references[:, 0],
      action,
      0.02,
      tuning=learner.reward_tuning,
      context=batch.context,
    )
    torch.testing.assert_close(reward, expected_collection, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(observation[:, -22:], action, rtol=0, atol=0)
    assert torch.equal(successor.detach(), before)
    torch.testing.assert_close(successor[:, 57:79], executed, rtol=0, atol=0)
    torch.testing.assert_close(
      observation[:, :-22],
      robot.observation(successor, batch.context)[:, :-22],
      rtol=1e-5,
      atol=1e-7,
    )


@pytest.mark.parametrize("component", ["model", "policy"])
@pytest.mark.parametrize("compiled", [False, True])
def test_independent_objective_normalization_scales_analytic_gradients(
  environment: tuple[Q1Robot, Q1Motions],
  component: str,
  compiled: bool,
  monkeypatch: pytest.MonkeyPatch,
):
  robot, motions = environment
  torch.manual_seed(491)
  batch = make_batch(robot, motions, 3 if component == "model" else 1)
  original = torch.nn.utils.clip_grad_norm_
  captured: list[list[torch.Tensor]] = []

  def clip(parameters, *args, **kwargs):
    parameters = list(parameters)
    captured.append([p.grad.detach().clone() for p in parameters if p.grad is not None])
    return original(parameters, *args, **kwargs)

  monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
  results = []
  gradients = []
  for normalized in (False, True):
    torch.manual_seed(992)
    learner = small_learner(robot, motions, RewardTuningCfg())
    learner.clean_policy_actions = True
    learner.fuse_model_loss = True
    if component == "model":
      learner.normalize_model_objective = normalized
      assert not learner.normalize_policy_objective
    else:
      learner.normalize_policy_objective = normalized
      assert not learner.normalize_model_objective
    if compiled:
      learner.enable_compile(backend="aot_eager")
    captured.clear()
    torch.manual_seed(777)
    if component == "model":
      result = learner.fit_model_tensors(batch)["model_loss"]
    else:
      result = torch.tensor(learner.improve_policy(batch, 1, 1)["actor_loss"])
    results.append(result)
    gradients.append(captured[0])
  divisor = 3 if component == "model" else 4
  torch.testing.assert_close(results[1], results[0] / divisor, rtol=1e-5, atol=1e-7)
  assert any(g.abs().max() > 0 for g in gradients[0])
  for baseline, normalized in zip(*gradients, strict=True):
    torch.testing.assert_close(normalized, baseline / divisor, rtol=1e-4, atol=1e-7)


def test_progress_warm_start_retains_counters_and_resets_learning_state(
  environment: tuple[Q1Robot, Q1Motions],
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
):
  robot, motions = environment
  cfg = training_config(robot, motions, tmp_path / "source_run")
  original = torch.load(train(cfg), map_location="cpu", weights_only=True)
  source = copy.deepcopy(original)
  source["iteration"], source["global_env_steps"] = 30000, 96000000
  source["world_size"], source["rank_directory"] = 8, "absent_rank_shards"
  for name in ("model", "actor", "critic"):
    for state in source[f"{name}_optimizer"]["state"].values():
      state["step"] = torch.tensor(999.0)
  path = tmp_path / "checkpoint_30000.pt"
  torch.save(source, path)
  seen = []

  def restore(source, identity, model, actor, **kwargs):
    termination = copy.deepcopy(model.termination.state_dict())
    restore_warm_start(source, identity, model, actor, **kwargs)
    for prefix, module in (("encoder", actor.encoder), ("decoder", actor.decoder)):
      for name, tensor in module.state_dict().items():
        assert torch.equal(tensor, source["actor"][f"{prefix}.{name}"])
    for name, tensor in model.dynamics.state_dict().items():
      assert torch.equal(tensor, source["model"][f"dynamics.{name}"])
    for name, tensor in model.termination.state_dict().items():
      assert torch.equal(tensor, termination[name])
    seen.append(True)

  monkeypatch.setattr("supermimic.train.restore_warm_start", restore)
  new_data = tmp_path / "new_data"
  new_data.mkdir()
  (new_data / "new_reference.npz").write_bytes(motions.paths[0].read_bytes())
  experiment = replace(
    cfg,
    output_dir=tmp_path / "phase",
    motion_dir=new_data,
    warm_start=path,
    warm_start_progress=True,
    iterations=30001,
    clean_policy_actions=True,
    action_lag=False,
    action_noise=0.1,
  )
  result_path = train(experiment)
  result = torch.load(result_path, map_location="cpu", weights_only=True)
  assert seen == [True]
  assert result["iteration"] == 30001
  assert result["global_env_steps"] == 96000000 + 84
  initialization = result["identity"]["initialization"]
  assert initialization["initial_iteration"] == 30000
  assert initialization["initial_environment_steps"] == 96000000
  assert initialization["preserved_progress"] is True
  assert set(initialization["preserved_counters"]) == {"iteration", "environment_steps"}
  assert "iteration" not in initialization["reset"]
  assert "environment_steps" not in initialization["reset"]
  assert set(initialization["reset"]) == {
    "critic",
    "model.termination",
    "all_optimizers",
    "replay",
    "curriculum",
    "rng",
  }
  for name in ("model", "actor", "critic"):
    assert all(
      int(state["step"]) == 1 for state in result[f"{name}_optimizer"]["state"].values()
    )
  assert result["identity"]["training_semantics"]["clean_policy_actions"] is True
  assert result["identity"]["training_semantics"]["action_lag"] is False
  # Replay keeps noisy executed commands in physical state/action columns.
  rank = torch.load(
    result_path.parent / result["rank_directory"] / "rank_0.pt",
    map_location="cpu",
    weights_only=True,
  )
  replay = rank["replay"]
  assert torch.equal(replay["next_states"][..., 57:79], replay["actions"])
  valid = ~replay["boundaries"][:-1]
  assert valid.any()
  difference = replay["observations"][1:, :, -22:] - replay["actions"][:-1]
  assert difference[valid].abs().mean() > 0.01


@pytest.mark.parametrize(
  "field,value",
  [
    ("clean_policy_actions", True),
    ("action_lag", True),
    ("normalize_policy_objective", True),
    ("normalize_model_objective", True),
  ],
)
def test_strict_resume_rejects_changed_training_semantics(
  environment: tuple[Q1Robot, Q1Motions], tmp_path: Path, field: str, value: bool
):
  robot, motions = environment
  cfg = training_config(robot, motions, tmp_path / "source")
  source = train(cfg)
  changed = replace(
    cfg, output_dir=tmp_path / "invalid", resume=source, iterations=2, **{field: value}
  )
  with pytest.raises(ValueError, match="training semantics changed"):
    train(changed)
  assert not changed.output_dir.exists()


def test_disabled_lag_retains_other_domain_randomization(
  environment: tuple[Q1Robot, Q1Motions],
):
  robot, motions = environment
  simulator = Q1Simulator(
    robot,
    4,
    "cpu",
    randomization=RandomizationCfg(),
    physics_timestep=0.005,
  )
  simulator.reset(torch.arange(4), motions.load(0)[:4], randomize=True)
  assert not simulator.lagged.any()
  nominal = nominal_context(robot.context_layout, 4, "cpu")
  assert not torch.equal(simulator.context, nominal)
  assert not torch.equal(
    simulator.context[:, robot.context_layout.kp_factor],
    nominal[:, robot.context_layout.kp_factor],
  )
