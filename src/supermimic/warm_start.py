"""New-experiment initialization from compatible actor/dynamics weights only."""

import hashlib
from pathlib import Path
from typing import Any

import torch

from supermimic.models import MotionPolicy, WorldModel


def load_warm_start(path: Path) -> dict[str, Any]:
  """Load the learner file on CPU, without reading any replay or rank shard."""
  source = torch.load(path, map_location="cpu", weights_only=True)
  if not isinstance(source, dict) or not all(
    isinstance(source.get(key), dict)
    for key in ("actor", "model", "identity", "config")
  ):
    raise ValueError("Expected a SuperMimic actor/dynamics learner checkpoint")
  return source


def restore_warm_start(
  source: dict[str, Any],
  identity: dict[str, Any],
  model: WorldModel,
  actor: MotionPolicy,
  *,
  action_scale: float,
  use_tn: bool,
) -> None:
  """Keep new reward/robot state, termination head, critic and optimizers.

  Data/reward units may change termination labels or their distribution, so this
  initialization conservatively relearns the termination head from fresh replay.
  Only the actor's expected keys and model.dynamics are copied; legacy nested
  robot state cannot replace the new experiment's reward/termination metadata.
  """
  previous = source["identity"]
  legacy = {
    "activation": "elu",
    "representation": "legacy",
    "modeled_action_delay": False,
  }
  if previous.get("network_architecture", legacy) != identity.get(
    "network_architecture", legacy
  ):
    raise ValueError("Warm-start network representation changed; start a fresh run")
  for name in (
    "asset_sha256",
    "joint_names",
    "state_dim",
    "obs_dim",
    "context_dim",
    "model_hidden",
    "actor_hidden",
    "control_dt",
    "kp",
    "kd",
    "default_q",
  ):
    if previous.get(name) != identity[name]:
      raise ValueError(f"Warm-start physical/network identity changed: {name}")
  timestep = previous.get("physics_timestep", previous.get("xml_timestep"))
  if timestep != identity["physics_timestep"]:
    raise ValueError("Warm-start physics timestep changed")
  if source["config"].get("action_scale") != action_scale:
    raise ValueError("Warm-start action scale changed")
  if source["config"].get("use_tn", False) != use_tn:
    raise ValueError("Warm-start torque-speed control changed")
  if use_tn:
    for name in (
      "torque_speed_peak_nm",
      "torque_speed_knee_rad_s",
      "torque_speed_cutoff_rad_s",
    ):
      if previous.get(name) != identity[name]:
        raise ValueError(f"Warm-start motor curve changed: {name}")
  for name, current, saved in (
    ("model.scale", model.scale, source["model"].get("scale")),
    ("actor.obs_scale", actor.obs_scale, source["actor"].get("obs_scale")),
  ):
    if not isinstance(saved, torch.Tensor) or not torch.equal(
      current.detach().cpu(), saved
    ):
      raise ValueError(f"Warm-start input normalization changed: {name}")
  for prefix, module in (("encoder", actor.encoder), ("decoder", actor.decoder)):
    weights = {
      name: source["actor"][f"{prefix}.{name}"] for name in module.state_dict()
    }
    module.load_state_dict(weights)
  dynamics_weights = {
    name: source["model"][f"dynamics.{name}"] for name in model.dynamics.state_dict()
  }
  model.dynamics.load_state_dict(dynamics_weights)


def warm_start_identity(
  path: Path, source: dict[str, Any], *, preserve_progress: bool = False
) -> dict[str, Any]:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while block := stream.read(1024 * 1024):
      digest.update(block)
  iteration = int(source["iteration"]) if preserve_progress else 0
  steps = int(source["global_env_steps"]) if preserve_progress else 0
  reset = [
    "critic",
    "model.termination",
    "all_optimizers",
    "replay",
    "curriculum",
    "rng",
  ]
  if not preserve_progress:
    reset.extend(("iteration", "environment_steps"))
  return {
    "mode": "weights_only_warm_start",
    "source_checkpoint": str(path.resolve()),
    "source_checkpoint_sha256": digest.hexdigest(),
    "source_iteration": int(source["iteration"]),
    "copied": ["actor", "model.dynamics"],
    "reset": reset,
    "preserved_progress": preserve_progress,
    "preserved_counters": ["iteration", "environment_steps"]
    if preserve_progress
    else [],
    "initial_iteration": iteration,
    "initial_environment_steps": steps,
    "termination_reset_reason": "Fresh labels/distribution for the new data/reward experiment",
  }
