"""Execution options preserve gradients and leave CPU optimizer behavior intact."""

from collections.abc import Generator

import pytest
import torch
from test_reward_training import environment as environment
from test_reward_training import small_learner
from test_runtime_training import assert_state_equal, make_batch
from torch._dynamo import reset

from supermimic.algorithm import SuperMimic
from supermimic.data import Q1Motions
from supermimic.reward_tuning import RewardTuningCfg
from supermimic.robot import Q1Robot


@pytest.fixture(autouse=True)
def isolated_compiler_cache() -> Generator[None]:
  reset()
  yield
  reset()


def test_compiled_critic_loss_preserves_complete_optimizer_updates(
  environment: tuple[Q1Robot, Q1Motions],
):
  robot, motions = environment
  learners = []
  for compiled in (False, True):
    torch.manual_seed(121)
    learner = small_learner(robot, motions, RewardTuningCfg(arm_joint_weight=0.05))
    learner.compile_critic = compiled
    if compiled:
      learner.enable_compile(backend="aot_eager")
    learners.append(learner)
  torch.manual_seed(77)
  batch = make_batch(robot, motions, 1)
  for update in range(2):
    results = []
    for learner in learners:
      torch.manual_seed(200 + update)
      results.append(learner.improve_policy(batch, 2, 2))
    assert_state_equal(results[0], results[1])
    for name in ("actor", "model", "critic"):
      assert_state_equal(
        getattr(learners[0], name).state_dict(),
        getattr(learners[1], name).state_dict(),
      )
      assert_state_equal(
        getattr(learners[0], f"{name}_optimizer").state_dict(),
        getattr(learners[1], f"{name}_optimizer").state_dict(),
      )


def test_fused_optimizer_request_falls_back_for_cpu(
  environment: tuple[Q1Robot, Q1Motions],
):
  robot, motions = environment
  original = small_learner(robot, motions, RewardTuningCfg())
  learner = SuperMimic(
    robot,
    motions,
    original.model,
    original.actor,
    original.critic,
    fused_optimizers=True,
  )
  for name in ("actor", "model", "critic"):
    assert getattr(learner, f"{name}_optimizer").defaults["fused"] is None
