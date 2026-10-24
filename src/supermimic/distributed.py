"""Data-parallel training across one process per GPU, without torch's DDP.

DDP is the wrong tool for this algorithm. Eq. 2 differentiates an imagined
rollout through a *frozen* world model to reach the actor, so the world model's
parameters take part in the actor's graph while receiving no gradient of their
own; DDP's reducer would either flag them as unused or synchronize gradients
that must not exist. Three modules under three optimizers, each with its own
loss graph and its own clipping rule, would also need three wrappers and three
reducers. Reducing gradients explicitly between the backward pass and the clip
is simpler and gives the clip the global gradient, which a per-rank reducer
running after the clip could not.

Nothing here is required for a single-process run: `Distributed.detached()`
leaves every call a no-op, so the single-GPU and CPU paths stay unchanged.
"""

import os
from collections.abc import Generator, Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Literal

import torch
import torch.distributed as dist
from torch import Tensor


@dataclass
class _GradientPlan:
  """A fixed gradient layout, established once for an optimizer's parameters."""

  parameters: tuple[torch.nn.Parameter, ...]
  present: tuple[bool, ...]
  buckets: list[list[Tensor]]


@dataclass
class Distributed:
  """One rank's view of the process group."""

  rank: int
  world_size: int
  local_rank: int
  # Set by `initialize`; collectives on bookkeeping tensors stage through it.
  device: str = "cpu"
  validation_mode: Literal["per_minibatch", "per_update"] = "per_minibatch"
  _update_failure: Tensor | None = field(default=None, init=False, repr=False)
  _failure_checks: list[tuple[Tensor, str]] = field(
    default_factory=list, init=False, repr=False
  )
  _failure_messages: list[str] = field(default_factory=list, init=False, repr=False)
  _gradient_plans: dict[tuple[tuple[int, ...], int], _GradientPlan] = field(
    default_factory=dict, init=False, repr=False
  )

  def __post_init__(self) -> None:
    if self.validation_mode not in ("per_minibatch", "per_update"):
      raise ValueError(f"Unknown validation mode: {self.validation_mode}")

  @staticmethod
  def detached(
    validation_mode: Literal["per_minibatch", "per_update"] = "per_minibatch",
  ) -> "Distributed":
    return Distributed(0, 1, 0, validation_mode=validation_mode)

  @staticmethod
  def from_environment(
    validation_mode: Literal["per_minibatch", "per_update"] = "per_minibatch",
  ) -> "Distributed":
    """Read the launcher's assignment, defaulting to a lone process."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
      return Distributed.detached(validation_mode)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if not 0 <= rank < world_size or local_rank < 0:
      raise ValueError(f"Launcher assigned rank {rank} of {world_size}")
    return Distributed(rank, world_size, local_rank, validation_mode=validation_mode)

  @property
  def enabled(self) -> bool:
    return self.world_size > 1

  @property
  def main(self) -> bool:
    """Whether this rank owns the run directory, the logs and the stdout."""
    return self.rank == 0

  def initialize(self, device: str, timeout_seconds: int = 300) -> None:
    """Join the process group on the backend that matches the device.

    A real run is one rank per GPU over NCCL. CPU ranks over Gloo exist so the
    collective *ordering* -- the part that deadlocks rather than slows down --
    can be exercised without reserving GPUs, including in CI.
    """
    self.device = device
    if timeout_seconds <= 0:
      raise ValueError("Collective timeout must be positive")
    if not self.enabled:
      return
    if device.startswith("cuda"):
      # Bind before the process group: NCCL picks up the current device, and
      # two ranks sharing one would deadlock on the first collective.
      index = torch.device(device).index
      if index is None or index >= torch.cuda.device_count():
        raise ValueError(f"Rank {self.rank} cannot access visible device {device}")
      torch.cuda.set_device(device)
    dist.init_process_group(
      "nccl" if device.startswith("cuda") else "gloo",
      rank=self.rank,
      world_size=self.world_size,
      timeout=timedelta(seconds=timeout_seconds),
    )

  def shutdown(self) -> None:
    if self.enabled and dist.is_initialized():
      dist.destroy_process_group()

  def barrier(self) -> None:
    if self.enabled:
      if self.device.startswith("cuda"):
        index = torch.device(self.device).index
        assert index is not None
        dist.barrier(device_ids=[index])
      else:
        dist.barrier()

  def any(self, value: bool) -> bool:
    """Agree on stopping even when just one worker received a signal."""
    flag = torch.tensor(int(value), device=self.device, dtype=torch.int32)
    return bool(self.sum_(flag).item())

  def objects(self, value: Any) -> list[Any]:
    """Gather small metadata/errors; never use this for replay tensors."""
    if not self.enabled:
      return [value]
    result: list[Any] = [None] * self.world_size
    dist.all_gather_object(result, value)
    return result

  def require(self, ready: bool, message: str) -> None:
    """Fail on every rank before entering a collective with missing peers."""
    if not self.any(not ready):
      return
    failures = self.objects(None if ready else message)
    if any(failure is not None for failure in failures):
      detail = "; ".join(
        f"rank {rank}: {failure}"
        for rank, failure in enumerate(failures)
        if failure is not None
      )
      raise RuntimeError(detail)

  def begin_update(self) -> None:
    """Start one learner update's deferred device-side validity checks."""
    if self.validation_mode == "per_minibatch":
      return
    if self._update_failure is not None:
      raise RuntimeError("Previous learner update has not been validated")
    self._update_failure = torch.zeros((), dtype=torch.int32, device=self.device)
    self._failure_checks.clear()
    self._failure_messages.clear()

  def require_finite(self, value: Tensor, message: str) -> None:
    """Check now in legacy mode, or defer one device scalar until update end."""
    if self.validation_mode == "per_minibatch":
      self.require(bool(torch.isfinite(value).all()), message)
      return
    if self._update_failure is None:
      raise RuntimeError("begin_update() is required for deferred validation")
    failure = (~torch.isfinite(value.detach()).all()).to(
      device=self.device, dtype=torch.int32
    )
    self._update_failure.bitwise_or_(failure)
    self._failure_checks.append((failure, message))

  def _mark_update_failure(self, message: str) -> None:
    if self._update_failure is None:
      raise RuntimeError("begin_update() is required for deferred validation")
    self._update_failure.fill_(1)
    self._failure_messages.append(message)

  def validate_update(self) -> None:
    """Coordinate deferred errors once, before any update log or checkpoint."""
    if self.validation_mode == "per_minibatch" or self._update_failure is None:
      return
    failure = self._update_failure.clone()
    if self.enabled:
      dist.all_reduce(failure, op=dist.ReduceOp.MAX)
    try:
      if not bool(failure.item()):
        return
      messages = list(self._failure_messages)
      if self._failure_checks:
        flags = torch.stack([flag for flag, _ in self._failure_checks]).tolist()
        messages.extend(
          message
          for flag, (_, message) in zip(flags, self._failure_checks, strict=True)
          if flag
        )
      local = "; ".join(dict.fromkeys(messages)) or None
      failures = self.objects(local)
      detail = "; ".join(
        f"rank {rank}: {message}"
        for rank, message in enumerate(failures)
        if message is not None
      )
      raise RuntimeError(detail or "Invalid learner update")
    finally:
      self._update_failure = None
      self._failure_checks.clear()
      self._failure_messages.clear()

  @contextmanager
  def checked(self, label: str) -> Generator[None]:
    """Coordinate failures in a phase that contains no other collectives."""
    error: Exception | None = None
    try:
      yield
    except Exception as exc:
      error = exc
    if not self.enabled and error is not None:
      raise error
    self.require(error is None, f"{label}: {error}")

  def share(self, count: int, what: str) -> int:
    """Split a total evenly across ranks, refusing a remainder.

    Environment and batch counts are quoted as totals so that a run on eight
    GPUs and a run on one are the same experiment at the same hyperparameters.
    A silent floor would quietly change the batch the paper's constants were
    chosen for, so an indivisible total is an error the user can act on.
    """
    if count % self.world_size:
      raise ValueError(f"{what} {count} does not divide across {self.world_size} ranks")
    return count // self.world_size

  def broadcast_parameters(self, modules: Iterable[torch.nn.Module]) -> None:
    """Make every rank start from rank 0's initialization.

    Identical seeds already produce identical weights, but that holds only as
    long as every rank draws from the RNG in exactly the same order. Paying one
    broadcast at startup removes the invariant instead of trusting it.
    """
    if not self.enabled:
      return
    with torch.no_grad():
      for module in modules:
        for tensor in list(module.parameters()) + list(module.buffers()):
          dist.broadcast(tensor.data, src=0)

  def assert_parameters_synced(self, modules: Iterable[torch.nn.Module]) -> None:
    """Check exact agreement before publishing a rank-0 learner snapshot."""
    if not self.enabled:
      return
    different = torch.zeros((), dtype=torch.int32, device=self.device)
    with torch.no_grad():
      for module in modules:
        for tensor in list(module.parameters()) + list(module.buffers()):
          reference = tensor.detach().clone()
          dist.broadcast(reference, src=0)
          different.bitwise_or_(torch.any(tensor != reference).int())
    dist.all_reduce(different, op=dist.ReduceOp.MAX)
    if different.item():
      raise RuntimeError("Learner parameters/buffers differ across ranks")

  def average_gradients(
    self,
    parameters: Iterable[torch.nn.Parameter],
    bucket_bytes: int = 128 * 1024 * 1024,
  ) -> None:
    """Average the gradients that exist, in flat buckets, before any clipping.

    Parameters without a gradient are skipped rather than reduced as zeros:
    under Eq. 2 the frozen model's parameters legitimately have none, and
    reducing a zero in their place would be a silent error if that ever changed.
    Deferred mode establishes that layout once. If it later changes, temporary
    zeros preserve collective shapes, and validation rejects the whole update
    before publishing it. Frozen parameters never receive synthetic gradients.
    """
    if not self.enabled:
      return
    if bucket_bytes <= 0:
      raise ValueError("Gradient bucket size must be positive")
    parameters = list(parameters)
    if self.validation_mode == "per_update":
      self._average_cached_gradients(tuple(parameters), bucket_bytes)
      return
    presence = torch.tensor(
      [p.grad is not None for p in parameters], device=self.device, dtype=torch.int32
    )
    dist.all_reduce(presence)
    if ((presence != 0) & (presence != self.world_size)).any():
      raise RuntimeError("Ranks disagree on which parameters received gradients")
    grads = [p.grad for p in parameters if p.grad is not None]
    self._reduce_gradient_buckets(_buckets(grads, bucket_bytes))

  def _average_cached_gradients(
    self, parameters: tuple[torch.nn.Parameter, ...], bucket_bytes: int
  ) -> None:
    if self._update_failure is None:
      raise RuntimeError("begin_update() is required for deferred validation")
    key = (tuple(id(parameter) for parameter in parameters), bucket_bytes)
    plan = self._gradient_plans.get(key)
    if plan is None:
      presence = torch.tensor(
        [p.grad is not None for p in parameters],
        device=self.device,
        dtype=torch.int32,
      )
      dist.all_reduce(presence)
      counts = presence.tolist()
      if any(count not in (0, self.world_size) for count in counts):
        self._mark_update_failure(
          "Ranks disagree on which parameters received gradients"
        )
      present = tuple(count > 0 for count in counts)
      active: list[Tensor] = [
        parameter
        for parameter, expected in zip(parameters, present, strict=True)
        if expected
      ]
      plan = _GradientPlan(parameters, present, _buckets(active, bucket_bytes))
      self._gradient_plans[key] = plan
    if any(
      (parameter.grad is not None) != expected
      for parameter, expected in zip(parameters, plan.present, strict=True)
    ):
      self._mark_update_failure("Gradient presence changed after its initial layout")
    gradients = [
      [p.grad if p.grad is not None else torch.zeros_like(p) for p in bucket]
      for bucket in plan.buckets
    ]
    self._reduce_gradient_buckets(gradients)

  def _reduce_gradient_buckets(self, buckets: list[list[Tensor]]) -> None:
    pending = []
    for bucket in buckets:
      flat = torch.cat([grad.reshape(-1) for grad in bucket])
      pending.append((dist.all_reduce(flat, async_op=True), flat, bucket))
    for handle, flat, bucket in pending:
      handle.wait()
      flat /= self.world_size
      offset = 0
      for grad in bucket:
        grad.copy_(flat[offset : offset + grad.numel()].view_as(grad))
        offset += grad.numel()

  def sum_(self, tensor: Tensor) -> Tensor:
    """Sum a tensor across ranks in place, staging it onto the backend's device."""
    if not self.enabled:
      return tensor
    if tensor.device.type == self.device.split(":")[0]:
      dist.all_reduce(tensor)
      return tensor
    # NCCL has no host path; stage the bookkeeping tensor through the bound
    # device rather than opening a second process group for it.
    staged = tensor.to(self.device)
    dist.all_reduce(staged)
    return tensor.copy_(staged.cpu())

  def mean(self, values: Sequence[float]) -> list[float]:
    """Average a row of scalars across ranks, for logging that reads globally."""
    if not self.enabled or not values:
      return list(values)
    total = torch.tensor(values, dtype=torch.float64, device=self.device)
    dist.all_reduce(total)
    return (total / self.world_size).tolist()


def _buckets(grads: list[Tensor], bucket_bytes: int) -> list[list[Tensor]]:
  """Group gradients into buckets of roughly one transfer each."""
  buckets: list[list[Tensor]] = []
  current: list[Tensor] = []
  size = 0
  for grad in grads:
    span = grad.numel() * grad.element_size()
    if current and size + span > bucket_bytes:
      buckets.append(current)
      current, size = [], 0
    current.append(grad)
    size += span
  if current:
    buckets.append(current)
  return buckets
