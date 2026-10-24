"""Summarize a centralized Q1 motion library from the importers' own manifests.

The index is what the training commands point at, and what answers "how much
Q1-retargeted motion is there": every count here is read back from the written
clips rather than from the retargeting pipeline's own bookkeeping.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tyro


@dataclass
class IndexCfg:
  library: Path
  output: Path
  target_fps: float = 50.0


def build_index(cfg: IndexCfg) -> dict[str, Any]:
  if not np.isclose(cfg.target_fps, 50.0):
    raise ValueError("The Q1 contract is 50 Hz")
  if not cfg.library.is_dir():
    raise ValueError(f"Library root does not exist: {cfg.library}")
  sources: list[dict[str, Any]] = []
  for manifest_path in sorted(cfg.library.glob("*/import_manifest.json")):
    manifest = json.loads(manifest_path.read_text())
    frames = sum(
      clip["output_frames"]
      for row in manifest["files"]
      for clip in row.get("clips", ())
    )
    clips = sum(len(row.get("clips", ())) for row in manifest["files"])
    if clips != manifest["written_clips"]:
      raise ValueError(f"{manifest_path} disagrees with itself about clip count")
    sources.append(
      {
        "name": manifest_path.parent.name,
        "directory": str(manifest_path.parent),
        "source_files": manifest["selected_files"],
        "failed_files": manifest["failed_files"],
        "clips": clips,
        "frames": frames,
        "hours": round(frames / cfg.target_fps / 3600, 3),
        "xml_sha256": manifest["xml_sha256"],
      }
    )
  if not sources:
    raise ValueError(f"No imported corpora under {cfg.library}")
  # One library must describe one robot, or a training run mixing its
  # directories would silently mix two kinematic trees.
  hashes = {source["xml_sha256"] for source in sources}
  if len(hashes) != 1:
    raise ValueError(f"Corpora were imported against different XMLs: {sorted(hashes)}")
  return {
    "library": str(cfg.library),
    "robot": "q1_v2",
    "xml_sha256": hashes.pop(),
    "fps": cfg.target_fps,
    "clips": sum(source["clips"] for source in sources),
    "frames": sum(source["frames"] for source in sources),
    "hours": round(
      sum(source["frames"] for source in sources) / cfg.target_fps / 3600, 3
    ),
    "failed_files": sum(source["failed_files"] for source in sources),
    "sources": sources,
  }


def main() -> None:
  cfg = tyro.cli(IndexCfg)
  index = build_index(cfg)
  cfg.output.write_text(json.dumps(index, indent=2) + "\n")
  print(
    json.dumps(
      {key: index[key] for key in ("clips", "frames", "hours", "failed_files")}
    )
  )


if __name__ == "__main__":
  main()
