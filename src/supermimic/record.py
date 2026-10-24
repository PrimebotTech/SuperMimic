"""Record one complete original LAFAN sequence and its tracking diagnostics."""

import json
import os
import random
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import mujoco
import numpy as np
import torch
import tyro
from PIL import Image, ImageDraw

from supermimic.play import PlayCfg, play
from supermimic.robot import Q1Robot


@dataclass
class RecordCfg:
  checkpoint: Path
  output_dir: Path
  motion_dir: Path
  seed: int = 0
  sequence: str | None = None
  width: int = 640
  height: int = 360
  ffmpeg: str = "ffmpeg"


def sequences(directory: Path) -> dict[str, list[Path]]:
  groups: dict[str, list[tuple[int, Path]]] = {}
  for path in directory.glob("*.npz"):
    match = re.fullmatch(r"(.+)__clip(\d+)", path.stem)
    if match:
      groups.setdefault(match[1], []).append((int(match[2]), path))
    else:
      groups.setdefault(path.stem, []).append((0, path))
  result = {}
  for name, chunks in groups.items():
    chunks.sort()
    if [i for i, _ in chunks] != list(range(len(chunks))):
      continue
    result[name] = [path for _, path in chunks]
  if not result:
    raise ValueError("No complete contiguous LAFAN sequences found")
  return result


def assemble(paths: list[Path], output: Path) -> None:
  frames = []
  names = None
  for path in paths:
    with np.load(path, allow_pickle=False) as data:
      if str(data["robot_name"].item()) != "q1_v2":
        raise ValueError("Only Q1 V2 references are supported")
      if not np.isclose(float(data["fps"].item()), 50):
        raise ValueError("Expected 50 Hz LAFAN references")
      current = data["robot_joint_names"].tolist()
      if names is not None and current != names:
        raise ValueError("Joint ordering differs across LAFAN chunks")
      names = current
      frames.append(data["qpos"].copy())
  np.savez_compressed(
    output,
    qpos=np.concatenate(frames),
    robot_name="q1_v2",
    root_quat_order="wxyz",
    fps=50.0,
    robot_joint_names=np.array(names),
  )


def diagnostic_summary(
  actual: np.ndarray,
  reference: np.ndarray,
  rewards: np.ndarray,
  failures: np.ndarray,
  dt: float = 0.02,
) -> dict:
  if len(actual) != len(reference) or len(rewards) != len(actual) - 1:
    raise ValueError("Rollout must include every reference frame and initial state")
  if not np.isfinite(actual).all():
    raise ValueError("Nonfinite full-clip rollout")
  indices = np.flatnonzero(failures)
  first = int(indices[0]) + 1 if len(indices) else None
  cutoff = first if first is not None else len(rewards)
  error = actual[:, :3] - reference[:, :3]
  qa, qr = actual[:, 3:7], reference[:, 3:7]
  qa = qa / np.linalg.norm(qa, axis=-1, keepdims=True)
  qr = qr / np.linalg.norm(qr, axis=-1, keepdims=True)
  angle = 2 * np.arccos(np.abs((qa * qr).sum(-1)).clip(0, 1))
  return {
    "reference_frames": len(reference),
    "recorded_frames": len(actual),
    "duration_seconds": (len(reference) - 1) * dt,
    "failed": first is not None,
    "first_failure_frame": first,
    "first_failure_seconds": first * dt if first is not None else None,
    "normalized_return": float(rewards[:cutoff].sum() / (5 * dt * len(rewards))),
    "continued_after_failure_for_video": True,
    "root_position_rmse_m": float(np.sqrt((error**2).mean())),
    "root_xy_max_error_m": float(np.linalg.norm(error[:, :2], axis=-1).max()),
    "root_orientation_rmse_rad": float(np.sqrt((angle**2).mean())),
    "joint_angle_rmse_rad": float(
      np.sqrt(((actual[:, 7:29] - reference[:, 7:29]) ** 2).mean())
    ),
    "evaluation_randomization": False,
  }


def render_video(
  cfg: RecordCfg,
  xml: Path,
  actual: np.ndarray,
  reference: np.ndarray,
  first_failure: int | None,
) -> int:
  executable = shutil.which(cfg.ffmpeg)
  if executable is None:
    raise FileNotFoundError(f"FFmpeg executable missing: {cfg.ffmpeg}")
  model = mujoco.MjModel.from_xml_path(str(xml))
  visual = cast(Any, model).vis.global_
  visual.offwidth = max(visual.offwidth, cfg.width)
  visual.offheight = max(visual.offheight, cfg.height)
  data = mujoco.MjData(model)
  camera = cast(Any, mujoco).MjvCamera()
  camera.distance, camera.azimuth, camera.elevation = 2.7, 125, -15
  output = cfg.output_dir / "tracking.mp4"
  command = [
    executable,
    "-nostdin",
    "-hide_banner",
    "-loglevel",
    "error",
    "-n",
    "-f",
    "rawvideo",
    "-pix_fmt",
    "rgb24",
    "-s",
    f"{2 * cfg.width}x{cfg.height + 32}",
    "-r",
    "50",
    "-i",
    "pipe:0",
    "-an",
    "-c:v",
    "libx264",
    "-preset",
    "veryfast",
    "-crf",
    "23",
    "-threads",
    "2",
    "-pix_fmt",
    "yuv420p",
    "-movflags",
    "+faststart",
    str(output),
  ]
  process = subprocess.Popen(command, stdin=subprocess.PIPE)
  assert process.stdin is not None
  try:
    with cast(Any, mujoco).Renderer(
      model, height=cfg.height, width=cfg.width
    ) as renderer:
      for frame, (pose, target) in enumerate(zip(actual, reference, strict=True)):
        # Keep both bodies visible; the caption reports their global XY drift.
        camera.lookat[:] = pose[:3]
        camera.lookat[2] = max(0.25, float(camera.lookat[2]))
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        left = renderer.render().copy()
        camera.lookat[:] = target[:3]
        data.qpos[:] = target
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        right = renderer.render().copy()
        image = Image.new("RGB", (2 * cfg.width, cfg.height + 32))
        image.paste(Image.fromarray(left), (0, 32))
        image.paste(Image.fromarray(right), (cfg.width, 32))
        draw = ImageDraw.Draw(image)
        failed = first_failure is not None and frame >= first_failure
        drift = float(np.linalg.norm(pose[:2] - target[:2]))
        text = (
          f"POLICY  t={frame / 50:.2f}s  XY={drift:.2f}m  "
          f"{'FAILED' if failed else 'TRACKING'}"
        )
        draw.text((8, 8), text, fill="red" if failed else "white")
        draw.text((cfg.width + 8, 8), "LAFAN REFERENCE", fill="white")
        if frame == 0:
          image.save(cfg.output_dir / "preview.png")
        process.stdin.write(image.tobytes())
        if frame % 1000 == 0:
          print(
            json.dumps({"rendered_frames": frame, "total": len(actual)}), flush=True
          )
  except BaseException:
    process.stdin.close()
    process.terminate()
    process.wait()
    raise
  process.stdin.close()
  if process.wait() != 0:
    raise RuntimeError("FFmpeg failed to encode full-clip video")
  return len(actual)


def record(cfg: RecordCfg) -> dict:
  torch.set_num_threads(1)
  if cfg.width % 2 or cfg.height % 2:
    raise ValueError("Video dimensions must be even")
  groups = sequences(cfg.motion_dir)
  name = cfg.sequence or random.Random(cfg.seed).choice(sorted(groups))
  paths = groups[name]
  cfg.output_dir.mkdir(parents=True, exist_ok=False)
  selection = {
    "sequence": name,
    "chunks": [str(p) for p in paths],
    "seed": cfg.seed,
    "checkpoint": str(cfg.checkpoint),
  }
  (cfg.output_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
  assembled = cfg.output_dir / "reference.npz"
  assemble(paths, assembled)
  play(
    PlayCfg(
      checkpoint=cfg.checkpoint,
      motion=assembled,
      output=cfg.output_dir / "rollout.npz",
      viewer="off",
      stop_on_failure=False,
    )
  )
  with np.load(cfg.output_dir / "rollout.npz") as rollout, np.load(assembled) as ref:
    actual = rollout["states"][:, :29].copy()
    reference = ref["qpos"].copy()
    result = diagnostic_summary(
      actual, reference, rollout["step_rewards"], rollout["step_failures"]
    )
  checkpoint = torch.load(cfg.checkpoint, map_location="cpu", weights_only=True)
  robot = Q1Robot(Path(checkpoint["config"]["robot_xml"]))
  result["video_frames"] = render_video(
    cfg, robot.xml, actual, reference, result["first_failure_frame"]
  )
  result.update(selection)
  (cfg.output_dir / "diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result), flush=True)
  return result


def main() -> None:
  if os.environ.get("MUJOCO_GL") not in ("osmesa", "egl"):
    print("For headless rendering set MUJOCO_GL=osmesa or egl before launching")
  record(tyro.cli(RecordCfg))


if __name__ == "__main__":
  main()
