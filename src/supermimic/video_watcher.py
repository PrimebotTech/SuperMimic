"""Record each atomically published 5000-update checkpoint on CPU."""

import fcntl
import json
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import tyro


@dataclass
class Config:
  run_dir: Path
  motion_dir: Path
  interval: int = 5000
  poll_seconds: int = 60
  seed: int = 20261023
  ffmpeg: str = "ffmpeg"
  render_device: int | None = None
  once: bool = False


def pending_checkpoints(run: Path, videos: Path, interval: int) -> list[Path]:
  if interval <= 0:
    raise ValueError("Checkpoint interval must be positive")
  pending = []
  for path in run.rglob("checkpoint_*.pt"):
    match = re.fullmatch(r"checkpoint_(\d+)\.pt", path.name)
    if match and int(match[1]) > 0 and int(match[1]) % interval == 0:
      if not (videos / path.stem / "diagnostics.json").is_file():
        pending.append((int(match[1]), path))
  return [path for _, path in sorted(pending)]


def main(cfg: Config) -> None:
  videos = cfg.run_dir / "videos"
  videos.mkdir(parents=True, exist_ok=True)
  with (videos / "watcher.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
      for checkpoint in pending_checkpoints(cfg.run_dir, videos, cfg.interval):
        iteration = int(checkpoint.stem.split("_")[1])
        stage = videos / f".{checkpoint.stem}.{uuid.uuid4().hex}"
        env = os.environ.copy()
        env.update(
          CUDA_VISIBLE_DEVICES="",
          MUJOCO_GL="osmesa",
          PYOPENGL_PLATFORM="osmesa",
          OMP_NUM_THREADS="1",
          OPENBLAS_NUM_THREADS="1",
          LP_NUM_THREADS="2",
        )
        if cfg.render_device is not None:
          env.update(
            MUJOCO_GL="egl",
            PYOPENGL_PLATFORM="egl",
            MUJOCO_EGL_DEVICE_ID=str(cfg.render_device),
          )
        args = [
          "nice",
          "-n",
          "10",
          sys.executable,
          "-m",
          "supermimic.record",
          "--checkpoint",
          str(checkpoint),
          "--output-dir",
          str(stage),
          "--motion-dir",
          str(cfg.motion_dir),
          "--seed",
          str(cfg.seed + iteration),
          "--ffmpeg",
          cfg.ffmpeg,
        ]
        try:
          with (videos / f"{checkpoint.stem}.log").open("a") as log:
            subprocess.run(args, env=env, stdout=log, stderr=log, check=True)
          os.rename(stage, videos / checkpoint.stem)
          print(
            json.dumps(
              {"recorded": str(checkpoint), "video": str(videos / checkpoint.stem)}
            ),
            flush=True,
          )
        except (subprocess.CalledProcessError, OSError) as error:
          print(
            json.dumps(
              {
                "recording_failed": str(checkpoint),
                "error": str(error),
                "stage": str(stage),
              }
            ),
            flush=True,
          )
      if cfg.once:
        break
      time.sleep(cfg.poll_seconds)


def cli() -> None:
  main(tyro.cli(Config))


if __name__ == "__main__":
  cli()
