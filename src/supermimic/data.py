"""Read named Q1 qpos motions without importing a retargeting framework."""

import fcntl
import hashlib
import json
import os
import uuid
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import mujoco
import numpy as np
import torch
from torch import Tensor

from supermimic.robot import Q1Robot, normalize_quat, quat_apply


class Q1Motions:
  """Physical Q1 states in shared CPU storage or a bounded device buffer."""

  def __init__(
    self,
    robot: Q1Robot,
    paths: list[Path],
    dt: float = 0.02,
    device: str = "cpu",
    memory_budget_gb: float = 8.0,
    storage: Literal["shared", "device", "device-fp16"] = "shared",
    cache_dir: Path | None = None,
  ):
    if not paths:
      raise ValueError("No Q1 motion files selected")
    if memory_budget_gb <= 0:
      raise ValueError("Allow a positive motion state budget")
    if storage not in ("shared", "device", "device-fp16"):
      raise ValueError("Motion storage must be shared or device")
    self.robot = robot
    self.paths = paths
    self.dt = dt
    self.device = device
    self.memory_budget_gb = memory_budget_gb
    self.storage = storage
    self.cache_dir = cache_dir or Path.home() / ".cache/supermimic/motions"
    self._states: Tensor | None = None

  def prepare(self) -> None:
    """Decode selected physical states once; pad previous actions on gather.

    Shared storage uses a locked CPU mmap file, reused by every rank. Device
    storage keeps the full selection on the requested device within its budget.
    Both validate the complete selection before creating a training run.
    """
    if self._states is not None:
      return
    lengths = self.lengths()
    frames = int(lengths.sum())
    needed = (
      frames * self.robot.physical_dim * (2 if self.storage == "device-fp16" else 4)
    )
    if self.storage != "shared" and needed > self.memory_budget_gb * 1024**3:
      raise ValueError(
        f"Decoding {len(self.paths)} clips ({frames} frames) needs "
        f"{needed / 1024**3:.2f} GiB of state, above the "
        f"{self.memory_budget_gb:g} GiB budget: select fewer motions or raise "
        f"the budget"
      )
    bounds = torch.cat((torch.zeros(1, dtype=torch.long), lengths.cumsum(0)))
    if self.storage == "shared":
      states = self._mapped_states(lengths, bounds)
      self.device = "cpu"
    elif self.storage == "device-fp16":
      source = self._mapped_states(lengths, bounds)
      states = torch.empty(
        frames, self.robot.physical_dim, device=self.device, dtype=torch.float16
      )
      for start in range(0, frames, 1_000_000):
        end = min(start + 1_000_000, frames)
        chunk = source[start:end].to(self.device, dtype=torch.float16)
        if not torch.isfinite(chunk).all():
          raise ValueError("Reference states exceed finite FP16 storage range")
        states[start:end].copy_(chunk)
    else:
      states = torch.empty(frames, self.robot.physical_dim, device=self.device)
      for clip in range(len(self.paths)):
        physical = self._decode(clip)[:, : self.robot.physical_dim]
        states[bounds[clip] : bounds[clip + 1]] = physical.to(self.device)
    self._bounds = bounds
    self._starts = bounds[:-1].to(self.device)
    self._limits = (lengths - 1).to(self.device)
    self._states = states

  def _mapped_states(self, lengths: Tensor, bounds: Tensor) -> Tensor:
    """One atomic, locked CPU mapping shared by all processes on this node.

    All ranks keep the same global clip IDs and curriculum. The operating
    system shares reference pages; only gathered samples travel to each GPU.
    No rank stores the entire library in VRAM or repeats the frame decoding.
    """
    files = []
    for path in self.paths:
      stat = path.stat()
      files.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
    signature = hashlib.sha256(
      json.dumps(
        ["q1-physical57-v1", self.robot.asset_sha256, self.dt, files, lengths.tolist()]
      ).encode()
    ).hexdigest()
    self.cache_dir.mkdir(parents=True, exist_ok=True)
    path = self.cache_dir / f"{signature}.npy"
    with (self.cache_dir / f"{signature}.lock").open("a") as lock:
      fcntl.flock(lock, fcntl.LOCK_EX)
      if not path.exists():
        temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.npy")
        try:
          print(
            f"[motion-cache] decoding {len(self.paths)} clips, "
            f"{int(lengths.sum())} frames into shared CPU storage",
            flush=True,
          )
          output = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.float32,
            shape=(int(lengths.sum()), self.robot.physical_dim),
          )
          for clip in range(len(self.paths)):
            if int(lengths[clip]) * self.robot.state_dim * 4 > (
              self.memory_budget_gb * 1024**3
            ):
              raise ValueError("One clip exceeds the motion decoding memory budget")
            physical = self._decode(clip)[:, : self.robot.physical_dim]
            output[int(bounds[clip]) : int(bounds[clip + 1])] = physical.numpy()
            if (clip + 1) % 1000 == 0 or clip + 1 == len(self.paths):
              print(f"[motion-cache] {clip + 1}/{len(self.paths)} clips", flush=True)
          output.flush()
          del output
          with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
          os.replace(temporary, path)
        finally:
          temporary.unlink(missing_ok=True)
      mapped = np.load(path, mmap_mode="c", allow_pickle=False)
      if mapped.shape != (int(lengths.sum()), self.robot.physical_dim):
        raise ValueError(f"Invalid shared motion cache: {path}")
      if mapped.dtype != np.float32:
        raise ValueError(f"Invalid shared motion dtype: {path}")
    return torch.from_numpy(mapped)

  def _state_with_action(self, physical: Tensor) -> Tensor:
    if physical.dtype == torch.float16:
      physical = physical.float()
      rotation = physical[..., 3:7]
      rotation = rotation / rotation.norm(dim=-1, keepdim=True).clamp_min(1e-8)
      physical = torch.cat((physical[..., :3], rotation, physical[..., 7:]), -1)
    return torch.cat((physical, physical.new_zeros((*physical.shape[:-1], 22))), -1)

  def load(self, clip: int) -> Tensor:
    """One clip's state trajectory, with zero previous-action fields appended."""
    self.prepare()
    assert self._states is not None
    return self._state_with_action(
      self._states[self._bounds[clip] : self._bounds[clip + 1]]
    )

  def _decode(self, clip: int) -> Tensor:
    path = self.paths[clip]
    with np.load(path, allow_pickle=False) as data:
      if str(data["robot_name"].item()) != "q1_v2":
        raise ValueError(f"{path}: expected robot_name=q1_v2")
      if str(data["root_quat_order"].item()) != "wxyz":
        raise ValueError(f"{path}: expected WXYZ root quaternions")
      if not np.isclose(float(data["fps"].item()), 1 / self.dt):
        raise ValueError(f"{path}: resample to {1 / self.dt:g} Hz before training")
      names = data["robot_joint_names"].tolist()
      if len(names) != 22 or set(names) != set(self.robot.joint_names):
        raise ValueError(f"{path}: Q1 joint names mismatch or duplicate")
      raw = data["qpos"].astype(np.float64)
    if raw.ndim != 2 or raw.shape[1] != 29 or len(raw) < 20:
      raise ValueError(f"{path}: expected at least 20 frames of 29-dimensional qpos")
    if not np.isfinite(raw).all():
      raise ValueError(f"{path}: nonfinite qpos")
    norms = np.linalg.norm(raw[:, 3:7], axis=-1)
    if np.any(np.abs(norms - 1) > 0.01):
      raise ValueError(f"{path}: root quaternion is not unit length")
    raw[:, 3:7] /= norms[:, None]
    qpos = np.concatenate(
      (raw[:, :7], raw[:, [7 + names.index(n) for n in self.robot.joint_names]]),
      axis=-1,
    )
    # MuJoCo handles quaternion signs and free-joint velocity conventions.
    qvel = np.empty((len(qpos), 28), dtype=np.float64)
    for t in range(len(qpos)):
      lo, hi = max(0, t - 1), min(len(qpos) - 1, t + 1)
      mujoco.mj_differentiatePos(
        self.robot.model, qvel[t], (hi - lo) * self.dt, qpos[lo], qpos[hi]
      )
    s = torch.zeros(len(qpos), self.robot.state_dim)
    s[:, :29] = torch.from_numpy(qpos.astype(np.float32))
    s[:, 29:32] = torch.from_numpy(qvel[:, :3].astype(np.float32))
    s[:, 32:35] = quat_apply(
      normalize_quat(s[:, 3:7]), torch.from_numpy(qvel[:, 3:6].astype(np.float32))
    )
    s[:, 35:57] = torch.from_numpy(qvel[:, 6:].astype(np.float32))
    return s

  def sequence(self, clips: Tensor, frames: Tensor, offsets: Tensor) -> Tensor:
    """Gather reference sequences, clamping only after a clip's last frame."""
    self.prepare()
    assert self._states is not None
    caller = clips.device
    clips = clips.to(self.device)
    rows = frames.to(self.device)[:, None] + offsets.to(self.device)
    rows = rows.clamp(min=0).minimum(self._limits[clips][:, None])
    physical = self._states[self._starts[clips][:, None] + rows]
    return self._state_with_action(physical).to(caller)

  def lengths(self) -> Tensor:
    if not hasattr(self, "_lengths"):
      lengths = []
      for path in self.paths:
        # Read only the NPY header inside NPZ, without decompressing trajectories.
        with zipfile.ZipFile(path) as archive, archive.open("qpos.npy") as stream:
          version = np.lib.format.read_magic(stream)
          if version == (1, 0):
            shape, _, _ = np.lib.format.read_array_header_1_0(stream)
          elif version == (2, 0):
            shape, _, _ = np.lib.format.read_array_header_2_0(stream)
          else:
            raise ValueError(f"{path}: unsupported NPY header {version}")
          if len(shape) != 2 or shape[1] != 29 or shape[0] < 20:
            raise ValueError(f"{path}: invalid Q1 motion shape {shape}")
          lengths.append(shape[0])
      self._lengths = torch.tensor(lengths)
    return self._lengths

  def remaining(self, clips: Tensor, frames: Tensor) -> Tensor:
    """Reference horizon on the storage device, without a GPU-to-CPU gather."""
    self.prepare()
    return (self._limits[clips.to(self.device)] - frames.to(self.device)).to(
      frames.device
    )


class ReturnResetSampler:
  """Appendix D.1's 50/40/10 reset mixture over one-second motion bins."""

  def __init__(
    self, lengths: Tensor, dt: float, gamma: float = 0.99, momentum: float = 0.999
  ):
    self.lengths = lengths.cpu()
    self.momentum = momentum
    self.frames_per_bin = round(1 / dt)
    self.clips = torch.cat(
      [
        torch.full(((int(n) - 2) // self.frames_per_bin + 1,), c)
        for c, n in enumerate(lengths)
      ]
    )
    self.starts = torch.cat(
      [
        torch.arange((int(n) - 2) // self.frames_per_bin + 1) * self.frames_per_bin
        for n in lengths
      ]
    )
    sizes = (self.lengths - 2) // self.frames_per_bin + 1
    self.bin_offsets = torch.cat(
      (torch.zeros(1, dtype=torch.long), sizes.cumsum(0)[:-1])
    )
    self.gamma, self.dt = gamma, dt
    # Appendix D.1 normalizes by the largest *attainable* discounted return.
    # An episode runs to the clip end, so the ceiling belongs to the frame the
    # episode actually started on, not to the bin that frame falls in: a bin is
    # a second wide, and within the last seconds of a clip the attainable return
    # drops by an order of magnitude across that second. Normalizing a frame
    # against its bin's start would therefore report near-zero progress for
    # perfect tracking near a clip end and collapse the softmax onto it.
    self.normalized = torch.ones(len(self.clips))
    self.counts = torch.zeros(len(self.clips))
    self.past_weights = torch.ones(len(self.clips)) / len(self.clips)

  def frame_ceiling(self, clips: Tensor, frames: Tensor) -> Tensor:
    """The largest discounted return an episode started at this frame can earn."""
    reachable = (self.lengths[clips.cpu()] - 1 - frames.cpu()).clamp(min=1).float()
    return 5 * self.dt * (1 - self.gamma**reachable) / (1 - self.gamma)

  def sample(self, count: int) -> tuple[Tensor, Tensor, Tensor]:
    hard = torch.softmax(-25 * self.normalized, dim=0)
    past = self.past_weights / self.past_weights.sum()
    weights = 0.5 / len(hard) + 0.4 * hard + 0.1 * past
    bins = torch.multinomial(weights, count, replacement=True)
    clips = self.clips[bins]
    available = (self.lengths[clips] - 1 - self.starts[bins]).clamp(
      max=self.frames_per_bin
    )
    frames = self.starts[bins] + (torch.rand(count) * available).long()
    return clips, frames, bins

  def advance_past(self) -> None:
    """Advance historical difficulty once per iteration, equally on every rank."""
    self.past_weights += torch.softmax(-25 * self.normalized, dim=0)

  def state_dict(self) -> dict[str, Any]:
    return {
      "lengths": self.lengths.clone(),
      "normalized": self.normalized.clone(),
      "counts": self.counts.clone(),
      "past_weights": self.past_weights.clone(),
      "dt": self.dt,
      "gamma": self.gamma,
      "momentum": self.momentum,
    }

  def load_state_dict(self, state: dict[str, Any]) -> None:
    if not torch.equal(state["lengths"], self.lengths) or any(
      state[name] != getattr(self, name) for name in ("dt", "gamma", "momentum")
    ):
      raise ValueError("Curriculum motion lengths/hyperparameters changed on resume")
    for name in ("normalized", "counts", "past_weights"):
      value = state[name]
      if value.shape != self.normalized.shape or not torch.isfinite(value).all():
        raise ValueError(f"Invalid curriculum state: {name}")
      setattr(self, name, value.cpu().clone())

  def update(
    self,
    bins: Tensor,
    normalized: Tensor,
    reduce: Callable[[Tensor], Tensor] | None = None,
  ) -> None:
    """Decay each bin's running progress estimate towards what was observed.

    ``normalized`` is each sample's discounted return already divided by
    ``frame_ceiling`` for the frame it started on, so one bin can mix frames
    with very different attainable returns without biasing the estimate.

    The paper keeps a *running* estimate. An all-time mean would pin every bin
    to the returns of the earliest, untrained policy and freeze the curriculum,
    so this applies one exponential step per observed sample: the decay rate
    follows the number of samples rather than how many of them happen to arrive
    in the same batch.

    ``reduce`` sums the per-bin counts and totals across data-parallel ranks.
    One shared curriculum is the point: every rank samples resets from the same
    motion library, so letting each keep private estimates would have them
    disagree about which bins are hard and spend the 40% hard share on
    whichever bins their own environments happened to visit.
    """
    bins, normalized = bins.cpu(), normalized.detach().cpu().float()
    counts = torch.zeros_like(self.counts).scatter_add_(
      0, bins, torch.ones_like(normalized)
    )
    totals = torch.zeros_like(self.normalized).scatter_add_(0, bins, normalized)
    if reduce is not None:
      shared = reduce(torch.stack((counts, totals)))
      counts, totals = shared[0], shared[1]
    touched = counts > 0
    retained = self.momentum ** counts[touched]
    observed = totals[touched] / counts[touched]
    self.normalized[touched] = (
      retained * self.normalized[touched] + (1 - retained) * observed
    )
    self.counts += counts

  def bin_ids(self, clips: Tensor, frames: Tensor) -> Tensor:
    return self.bin_offsets[clips.cpu()] + frames.cpu() // self.frames_per_bin
