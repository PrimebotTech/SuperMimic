"""Pure tensor collection heads, with physical stepping and noise kept outside."""

import torch
from torch import Tensor

from supermimic.compile_runtime import configure_compiler
from supermimic.models import MotionPolicy, goal_features
from supermimic.reward_tuning import BASELINE_TUNING, RewardTuningCfg, tuned_reward
from supermimic.robot import Q1Robot


class Q1Collector:
  def __init__(
    self,
    robot: Q1Robot,
    actor: MotionPolicy,
    dt: float,
    *,
    reward_tuning: RewardTuningCfg = BASELINE_TUNING,
    clean_policy_actions: bool = False,
  ):
    self.robot, self.actor, self.dt = robot, actor, dt
    self.reward_tuning = reward_tuning
    self.clean_policy_actions = clean_policy_actions
    self.action = self._action
    self.heads = self._heads

  def enable_compile(self, backend: str = "inductor") -> None:
    if backend == "inductor":
      configure_compiler()
    options: dict[str, str | int | bool] | None = (
      {"max_fusion_size": 8} if backend == "inductor" else None
    )
    self.action = torch.compile(
      self._action, backend=backend, dynamic=False, fullgraph=True, options=options
    )
    self.heads = torch.compile(
      self._heads, backend=backend, dynamic=False, fullgraph=True, options=options
    )

  def _action(self, history: Tensor, references: Tensor) -> Tensor:
    return self.actor(history, goal_features(self.robot, references))

  def _heads(
    self,
    state: Tensor,
    next_state: Tensor,
    target: Tensor,
    action: Tensor,
    context: Tensor,
    previous_action: Tensor | None = None,
  ) -> tuple[Tensor, Tensor, Tensor]:
    observation = self.robot.observation(next_state, context)
    reward_state = state
    if self.clean_policy_actions:
      if previous_action is None:
        raise ValueError("Clean-action rewards require the previous policy action")
      reward_state = torch.cat((state[..., :57], previous_action), -1)
      observation = torch.cat((observation[..., :-22], action), -1)
    return (
      tuned_reward(
        self.robot,
        reward_state,
        next_state,
        target,
        action,
        self.dt,
        tuning=self.reward_tuning,
        context=context,
      ),
      self.robot.terminated(next_state, target),
      observation,
    )
