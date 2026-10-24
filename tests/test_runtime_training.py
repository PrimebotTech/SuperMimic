"""Deferred validation preserves finite updates and rejects invalid snapshots."""

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from test_reward_training import environment as environment
from test_reward_training import small_learner, training_config

from supermimic.algorithm import SuperMimic
from supermimic.data import Q1Motions
from supermimic.distributed import Distributed
from supermimic.randomization import nominal_context
from supermimic.replay import ReplayBatch
from supermimic.reward_tuning import RewardTuningCfg
from supermimic.robot import Q1Robot
from supermimic.train import train


def make_batch(robot: Q1Robot, motions: Q1Motions, window: int) -> ReplayBatch:
  clips = torch.zeros(3, dtype=torch.long)
  frames = torch.tensor([17, 28, 41])
  reference = motions.sequence(clips, frames, torch.arange(window + 1))
  actions = torch.randn(3, window, 22) * 0.05
  states, successors = reference[:, :-1].clone(), reference[:, 1:].clone()
  successors[:, :, 57:79] = actions
  if window > 1:
    states[:, 1:, 57:79] = actions[:, :-1]
  context = nominal_context(robot.context_layout, len(clips), "cpu")
  history_states = motions.sequence(clips, frames, torch.arange(-9, 1))
  history = robot.observation(history_states, context[:, None])
  return ReplayBatch(
    states,
    successors,
    actions,
    context,
    torch.zeros(3, window),
    history,
    clips,
    frames,
  )


def assert_state_equal(first: Any, second: Any) -> None:
  if isinstance(first, torch.Tensor):
    assert isinstance(second, torch.Tensor)
    assert torch.equal(first, second)
  elif isinstance(first, dict):
    assert first.keys() == second.keys()
    for key in first:
      assert_state_equal(first[key], second[key])
  elif isinstance(first, (tuple, list)):
    assert type(first) is type(second) and len(first) == len(second)
    for left, right in zip(first, second, strict=True):
      assert_state_equal(left, right)
  else:
    assert first == second


@pytest.mark.parametrize("fused_loss", [False, True])
def test_deferred_finite_model_actor_critic_updates_match_strict_exactly(
  environment: tuple[Q1Robot, Q1Motions], fused_loss: bool
):
  robot, motions = environment
  learners = []
  for mode in ("per_minibatch", "per_update"):
    torch.manual_seed(931)
    learner = small_learner(robot, motions, RewardTuningCfg())
    learner.distributed = Distributed.detached(validation_mode=mode)
    learner.distributed.initialize("cpu")
    learner.fuse_model_loss = fused_loss
    learners.append(learner)
  torch.manual_seed(731)
  model_batch, policy_batch = (
    make_batch(robot, motions, 3),
    make_batch(robot, motions, 1),
  )
  for update in range(2):
    metrics = []
    for learner in learners:
      torch.manual_seed(451 + update)
      learner.distributed.begin_update()
      model_metrics = {}
      for _ in range(2):
        model_metrics = learner.fit_model_tensors(model_batch)
      policy_metrics = learner.improve_policy(policy_batch, 2, 2)
      learner.distributed.validate_update()
      metrics.append((model_metrics, policy_metrics))
    assert_state_equal(metrics[0], metrics[1])
    strict, deferred = learners
    for name in ("model", "actor", "critic"):
      assert_state_equal(
        getattr(strict, name).state_dict(), getattr(deferred, name).state_dict()
      )
      assert_state_equal(
        getattr(strict, f"{name}_optimizer").state_dict(),
        getattr(deferred, f"{name}_optimizer").state_dict(),
      )
      for left, right in zip(
        getattr(strict, name).parameters(),
        getattr(deferred, name).parameters(),
        strict=True,
      ):
        if left.grad is None:
          assert right.grad is None
        else:
          assert right.grad is not None and torch.equal(left.grad, right.grad)


@pytest.mark.parametrize("component", ["model", "actor", "critic"])
def test_deferred_nonfinite_gradient_is_rejected_at_update_boundary(
  environment: tuple[Q1Robot, Q1Motions], component: str
):
  robot, motions = environment
  learner = small_learner(robot, motions, RewardTuningCfg())
  learner.distributed = Distributed.detached(validation_mode="per_update")
  learner.distributed.initialize("cpu")
  parameter = next(getattr(learner, component).parameters())
  hook = parameter.register_hook(lambda grad: torch.full_like(grad, float("nan")))
  learner.distributed.begin_update()
  try:
    if component == "model":
      metrics = learner.fit_model_tensors(make_batch(robot, motions, 3))
      assert torch.isfinite(metrics["model_loss"])
    else:
      learner.improve_policy(make_batch(robot, motions, 1), 1, 1)
    # Loss checks alone cannot detect a finite forward pass with bad gradients.
    assert parameter.grad is not None and torch.isnan(parameter.grad).any()
    with pytest.raises(RuntimeError, match="Nonfinite.*gradient norm"):
      learner.distributed.validate_update()
  finally:
    hook.remove()


def test_deferred_invalid_training_update_cannot_publish_log_or_checkpoint(
  environment: tuple[Q1Robot, Q1Motions],
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
):
  robot, motions = environment
  cfg = replace(
    training_config(robot, motions, tmp_path / "invalid"),
    runtime_checks="per_update",
  )
  original = SuperMimic.fit_model_tensors
  calls = []

  def invalid_update(learner: SuperMimic, batch: ReplayBatch):
    metrics = original(learner, batch)
    learner.distributed.require_finite(
      torch.tensor(float("nan")), "Injected nonfinite update"
    )
    calls.append(learner)
    return metrics

  monkeypatch.setattr(SuperMimic, "fit_model_tensors", invalid_update)
  with pytest.raises(RuntimeError, match="Injected nonfinite update"):
    train(cfg)
  assert calls
  assert calls[-1].actor_optimizer.state
  assert calls[-1].critic_optimizer.state
  assert not list(cfg.output_dir.glob("checkpoint*"))
  assert not (cfg.output_dir / "metrics.jsonl").exists()


def test_finite_bounded_training_checkpoints_match_between_runtime_modes(
  environment: tuple[Q1Robot, Q1Motions],
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
):
  robot, motions = environment
  checks = {"per_minibatch": 0, "per_update": 0}
  original = Distributed.checked

  @contextmanager
  def checked(group: Distributed, label: str):
    if label == "model replay sampling":
      checks[group.validation_mode] += 1
    with original(group, label):
      yield

  monkeypatch.setattr(Distributed, "checked", checked)
  cfg = replace(
    training_config(robot, motions, tmp_path / "strict"), iterations=2, model_batches=10
  )
  strict_path = train(replace(cfg, runtime_checks="per_minibatch"))
  deferred_path = train(
    replace(cfg, output_dir=tmp_path / "deferred", runtime_checks="per_update")
  )
  strict = torch.load(strict_path, map_location="cpu", weights_only=True)
  deferred = torch.load(deferred_path, map_location="cpu", weights_only=True)
  for name in (
    "model",
    "actor",
    "critic",
    "robot",
    "model_optimizer",
    "actor_optimizer",
    "critic_optimizer",
    "iteration",
    "global_env_steps",
    "identity",
  ):
    assert_state_equal(strict[name], deferred[name])
  assert strict["config"]["runtime_checks"] == "per_minibatch"
  assert deferred["config"]["runtime_checks"] == "per_update"
  assert checks == {"per_minibatch": 20, "per_update": 2}
