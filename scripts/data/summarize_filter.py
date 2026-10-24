"""Summarize completed screening rows without importing simulation or Torch."""

import argparse
import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def summarize(path: Path) -> dict[str, Any]:
  statuses: Counter[str] = Counter()
  frames: Counter[str] = Counter()
  sources: dict[str, Counter[str]] = defaultdict(Counter)
  sources_frames: dict[str, Counter[str]] = defaultdict(Counter)
  reasons: dict[str, Counter[str]] = defaultdict(Counter)
  deepest: list[tuple[float, str, dict[str, Any]]] = []
  for line in path.open():
    row = json.loads(line)
    status, relative = row["status"], row["path"]
    source = Path(relative).parts[0]
    statuses[status] += 1
    count = row.get("frames", 0)
    frames[status] += count
    sources[source][status] += 1
    sources_frames[source][status] += count
    unique = {
      (reason.get("severity", "review"), reason["code"])
      for reason in row.get("reasons", [])
    }
    for severity, code in unique:
      reasons[severity][code] += 1
    for reason in row.get("reasons", []):
      if reason["code"] == "self_collision":
        evidence = {
          "path": relative,
          "status": status,
          "archive_sha256": row.get("file_sha256"),
          "evidence": reason,
        }
        heapq.heappush(deepest, (reason.get("excess_m", 0.0), relative, evidence))
        if len(deepest) > 20:
          heapq.heappop(deepest)
  return {
    "completed_clips": sum(statuses.values()),
    "completed_frames": sum(frames.values()),
    "status_counts": dict(statuses),
    "frames_by_status": dict(frames),
    "source_status_counts": {source: dict(value) for source, value in sources.items()},
    "source_frame_counts": {
      source: dict(value) for source, value in sources_frames.items()
    },
    "reason_clip_counts_by_severity": {
      severity: dict(value) for severity, value in reasons.items()
    },
    "largest_self_collision_examples": [
      item[2] for item in sorted(deepest, reverse=True)
    ],
    "scope": (
      "Counts cover committed JSONL rows only. Rejection reasons can overlap; "
      "each reason is counted at most once per clip. Accepted means screened "
      "candidates, not proven dynamic feasibility."
    ),
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--results", type=Path, required=True)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  report = summarize(args.results)
  if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
