"""Portable TOML launcher using the same TrainCfg and training implementation."""

import math
import os
import shlex
import subprocess
import sys
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

import tyro

from supermimic.train import TrainCfg

Scalar = str | int | float | bool
PATH_FIELDS = {
  "robot_xml",
  "motion_dir",
  "output_dir",
  "resume",
  "warm_start",
  "motion_cache_dir",
}


@dataclass
class Config:
  config: Path
  """TOML with a [train] table; relative paths resolve against this file."""
  gpus: int = 1
  """Number of visible GPUs/ranks; global env/batch counts are not multiplied."""
  dry_run: bool = False
  """Print the resolved command without creating a run or initializing CUDA."""
  compiler_cache_dir: Path | None = None
  """Compiler/temp root; defaults to local /tmp/supermimic-<uid>."""


def load_options(path: Path) -> dict[str, Scalar]:
  with path.open("rb") as stream:
    document = tomllib.load(stream)
  if set(document) != {"train"} or not isinstance(document["train"], dict):
    raise ValueError("Expected one TOML [train] table")
  values = document["train"]
  known = {field.name for field in fields(TrainCfg)}
  unknown = set(values) - known
  if unknown:
    raise ValueError(f"Unknown training options: {', '.join(sorted(unknown))}")
  for required in ("robot_xml", "motion_dir", "output_dir"):
    if required not in values:
      raise ValueError(f"Missing training option: {required}")
  resolved: dict[str, Scalar] = {}
  for key, value in values.items():
    if type(value) not in (str, int, float, bool):
      raise ValueError(f"Expected a scalar training option: {key}")
    if isinstance(value, float) and not math.isfinite(value):
      raise ValueError(f"Expected a finite training option: {key}")
    if key in PATH_FIELDS:
      if not isinstance(value, str) or not value:
        raise ValueError(f"Expected a nonempty path: {key}")
      location = Path(value).expanduser()
      if not location.is_absolute():
        location = path.resolve().parent / location
      value = str(location.resolve())
    resolved[key] = value
  return resolved


def build_command(options: dict[str, Scalar], gpus: int) -> list[str]:
  """Construct an argument vector; never interpret configuration as shell code."""
  if gpus < 1:
    raise ValueError("Choose at least one GPU/rank")
  defaults = {field.name: field.default for field in fields(TrainCfg)}
  for name in ("num_envs", "model_batch_size", "policy_batch_size"):
    value = options.get(name, defaults[name])
    if type(value) is not int or value < 1 or value % gpus:
      raise ValueError(f"{name} must be positive and divisible by {gpus}")
  command = [sys.executable]
  if gpus > 1:
    command += [
      "-m",
      "torch.distributed.run",
      "--standalone",
      "--nnodes",
      "1",
      "--nproc-per-node",
      str(gpus),
    ]
  command += ["-m", "supermimic.train"]
  for key, value in options.items():
    flag = key.replace("_", "-")
    if isinstance(value, bool):
      command.append(f"--{flag}" if value else f"--no-{flag}")
    else:
      command += [f"--{flag}", str(value)]
  return command


def runtime_environment(cache_dir: Path | None = None) -> dict[str, str]:
  """Keep compiler file operations off an inherited network temporary directory."""
  environment = os.environ.copy()
  root = (cache_dir or Path("/tmp") / f"supermimic-{os.getuid()}").expanduser()
  root = root.resolve()
  locations = {
    "TMPDIR": root / "tmp",
    "XDG_CACHE_HOME": root / "xdg",
    "TORCHINDUCTOR_CACHE_DIR": root / "inductor",
    "TRITON_CACHE_DIR": root / "triton",
  }
  for name, location in locations.items():
    location.mkdir(parents=True, exist_ok=True)
    environment[name] = str(location)
  for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    environment.setdefault(name, "1")
  return environment


def main() -> None:
  cfg = tyro.cli(Config)
  command = build_command(load_options(cfg.config), cfg.gpus)
  print(shlex.join(command), flush=True)
  if cfg.dry_run:
    return
  environment = runtime_environment(cfg.compiler_cache_dir)
  print(f"Compiler cache: {environment['TORCHINDUCTOR_CACHE_DIR']}", flush=True)
  subprocess.run(command, env=environment, check=True)


if __name__ == "__main__":
  main()
