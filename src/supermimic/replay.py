"""Per-environment temporal replay with terminal next states kept before reset."""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass
class ReplayBatch:
  states: Tensor
  next_states: Tensor
  actions: Tensor
  context: Tensor
  terminated: Tensor
  history: Tensor
  clips: Tensor
  frames: Tensor


class TemporalReplay:
  def __init__(
    self,
    capacity: int,
    num_envs: int,
    state_dim: int,
    context_dim: int,
    obs_dim: int,
    device: str,
  ):
    self.capacity = capacity
    self.num_envs = num_envs
    self.cursor = 0
    self.count = 0
    self.device = device
    self.states = torch.empty(capacity, num_envs, state_dim, device=device)
    self.next_states = torch.empty_like(self.states)
    self.context = torch.empty(capacity, num_envs, context_dim, device=device)
    self.actions = torch.empty(capacity, num_envs, 22, device=device)
    self.observations = torch.empty(capacity, num_envs, obs_dim, device=device)
    self.terminated = torch.empty(capacity, num_envs, device=device)
    self.boundaries = torch.ones(capacity, num_envs, dtype=torch.bool, device=device)
    self.clips = torch.empty(capacity, num_envs, dtype=torch.long, device=device)
    self.frames = torch.empty_like(self.clips)

  def state_dict(self) -> dict[str, Any]:
    """Save only initialized rows, in chronological order, on CPU."""
    oldest = self.cursor if self.count == self.capacity else 0
    rows = (oldest + torch.arange(self.count, device=self.device)) % self.capacity
    result: dict[str, Any] = {
      "capacity": self.capacity,
      "num_envs": self.num_envs,
      "count": self.count,
    }
    for name in self._storage_names():
      result[name] = getattr(self, name)[rows].cpu()
    return result

  @staticmethod
  def _storage_names() -> tuple[str, ...]:
    return (
      "states",
      "next_states",
      "context",
      "actions",
      "observations",
      "terminated",
      "boundaries",
      "clips",
      "frames",
    )

  def load_state_dict(self, state: dict[str, Any]) -> None:
    if state["capacity"] != self.capacity or state["num_envs"] != self.num_envs:
      raise ValueError("Replay capacity/environment count differs from checkpoint")
    count = state["count"]
    if not isinstance(count, int) or not 0 <= count <= self.capacity:
      raise ValueError("Invalid replay row count")
    for name in self._storage_names():
      target, source = getattr(self, name), state[name]
      if source.shape != (count, *target.shape[1:]) or source.dtype != target.dtype:
        raise ValueError(f"Invalid replay shard tensor: {name}")
    for name in self._storage_names():
      getattr(self, name)[:count].copy_(state[name])
    self.count, self.cursor = count, count % self.capacity
    # Physics is reset on resume. No sample may join the old and new episodes.
    if count:
      self.boundaries[(self.cursor - 1) % self.capacity] = True

  def add(
    self,
    states: Tensor,
    next_states: Tensor,
    actions: Tensor,
    context: Tensor,
    observation: Tensor,
    terminated: Tensor,
    boundary: Tensor,
    clips: Tensor,
    frames: Tensor,
  ) -> None:
    i = self.cursor
    for storage, value in (
      (self.states, states),
      (self.next_states, next_states),
      (self.actions, actions),
      (self.context, context),
      (self.observations, observation),
      (self.terminated, terminated),
      (self.boundaries, boundary),
      (self.clips, clips),
      (self.frames, frames),
    ):
      storage[i].copy_(value.detach())
    self.cursor = (self.cursor + 1) % self.capacity
    self.count = min(self.capacity, self.count + 1)

  def sample(self, batch_size: int, window: int = 8, history: int = 10) -> ReplayBatch:
    total = history - 1 + window
    if self.count < total:
      raise ValueError("Replay has too few temporal samples")
    oldest = self.cursor if self.count == self.capacity else 0
    starts = torch.arange(self.count - total + 1, device=self.device)
    rows = (
      oldest + starts[:, None] + torch.arange(total, device=self.device)
    ) % self.capacity
    # Permit termination at the last supervised transition, never before it.
    valid = ~self.boundaries[rows[:, :-1]].any(dim=1)
    candidates = valid.nonzero()
    if not len(candidates):
      raise ValueError("No replay windows without reset boundaries")
    pick = candidates[torch.randint(len(candidates), (batch_size,), device=self.device)]
    indexes, envs = rows[pick[:, 0]], pick[:, 1]
    env_grid = envs[:, None]
    step_rows = indexes[:, history - 1 :]
    return ReplayBatch(
      self.states[step_rows, env_grid],
      self.next_states[step_rows, env_grid],
      self.actions[step_rows, env_grid],
      self.context[step_rows[:, 0], envs],
      self.terminated[step_rows, env_grid],
      self.observations[indexes[:, :history], env_grid],
      self.clips[step_rows[:, 0], envs],
      self.frames[step_rows[:, 0], envs],
    )
