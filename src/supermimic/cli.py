"""One discoverable entry point; command modules own their existing options."""

import argparse
import sys
from importlib import import_module

from supermimic import __version__

COMMANDS = {
  "train": ("train", "main", "Train directly with explicit CLI options"),
  "launch": ("launch", "main", "Launch single/multiple ranks from TOML"),
  "eval": ("evaluate", "main", "Evaluate checkpoint clips deterministically"),
  "play": ("play", "main", "View or save a policy rollout"),
  "record": ("record", "main", "Record one complete original LAFAN sequence"),
  "watch-videos": ("video_watcher", "cli", "Record periodic checkpoints on CPU"),
  "monitor": ("training_monitor", "cli", "Monitor a run or supervised attempts"),
  "reshard": ("reshard", "main", "Expand a complete checkpoint's rank count"),
  "resample": ("resample", "main", "Resample named Q1 trajectories to 50 Hz"),
  "import-pkl": ("import_pkl", "main", "Import trusted Q1 retargeting records"),
  "index": ("library_index", "main", "Summarize a motion library"),
  "preflight": ("preflight", "main", "Check motion schema and static geometry"),
  "filter": ("filter_motions", "main", "Screen motions into reviewable manifests"),
  "diagnose": ("diagnostics", "main", "Measure 1/8/32-step model forecasts"),
  "reward-tuning": (
    "reward_tuning",
    "main",
    "Compare opt-in rewards on fixed CPU states",
  ),
}


def main() -> None:
  description = "Independent SuperMimic training and evaluation for Q1 V2."
  descriptions = "\n".join(
    f"  {name:14} {entry[2]}" for name, entry in COMMANDS.items()
  )
  parser = argparse.ArgumentParser(
    description=description,
    epilog=f"Commands:\n{descriptions}\n\nUse <command> --help for its options.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  parser.add_argument("--version", action="version", version=__version__)
  parser.add_argument("command", choices=COMMANDS, help="Command to run")
  parser.add_argument("arguments", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
  args = parser.parse_args()
  module, entry, _ = COMMANDS[args.command]
  function = getattr(import_module(f"supermimic.{module}"), entry)
  original = sys.argv
  try:
    sys.argv = [f"{parser.prog} {args.command}", *args.arguments]
    function()
  finally:
    sys.argv = original
