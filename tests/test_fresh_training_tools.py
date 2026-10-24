"""Dry-run entry points preserve fresh initialization and training workloads."""

import json
import os
import subprocess
import sys
from pathlib import Path

from supermimic.launch import build_command, load_options

ROOT = Path(__file__).resolve().parents[1]


def test_fresh_preset_supports_single_and_multi_gpu_commands():
  options = load_options(ROOT / "configs/q1-fresh.example.toml")
  assert "warm_start" not in options and "resume" not in options
  assert options["iterations"] == 250000
  assert options["capture_imagination"] and options["compile_networks"]
  assert not options["compile_critic"] and not options["fused_optimizers"]
  multi = build_command(options, 8)
  assert multi[multi.index("--nproc-per-node") + 1] == "8"
  assert "--capture-imagination" in multi
  single = build_command(
    options | {"num_envs": 1024, "model_batch_size": 512, "policy_batch_size": 512},
    1,
  )
  assert "torch.distributed.run" not in single
  assert single[single.index("--num-envs") + 1] == "1024"


def test_fresh_benchmark_dry_run_uses_no_source_weights_or_gpu(tmp_path: Path):
  document = json.loads((ROOT / "tools/q1_fresh_benchmark.example.json").read_text())
  output = tmp_path / "benchmark"
  document["benchmark_root"] = str(output)
  config = tmp_path / "config.json"
  config.write_text(json.dumps(document))
  result = subprocess.run(
    [
      sys.executable,
      str(ROOT / "tools/benchmark_q1_fresh.py"),
      "--config",
      str(config),
      "--dry-run",
    ],
    capture_output=True,
    text=True,
    check=True,
    env=os.environ | {"CUDA_VISIBLE_DEVICES": ""},
  )
  cases = [json.loads(line) for line in result.stdout.splitlines()]
  assert len(cases) == 4 and not output.exists()
  for case in cases:
    train = case["train"]
    assert train["warm_start"] is None and train["resume"] is None
    assert train["iterations"] == 120
    for key in (
      "num_envs",
      "model_batch_size",
      "policy_batch_size",
      "model_batches",
      "critic_epochs",
      "critic_minibatches",
    ):
      assert train[key] == document["train"][key]


def test_fresh_supervisor_dry_run_removes_inherited_warm_start(tmp_path: Path):
  document = json.loads((ROOT / "tools/q1_fresh_benchmark.example.json").read_text())
  document["run_root"] = str(tmp_path / "fresh")
  document["train"].update(warm_start="old.pt", warm_start_progress=True)
  config = tmp_path / "config.json"
  config.write_text(json.dumps(document))
  result = subprocess.run(
    [
      sys.executable,
      str(ROOT / "tools/supervise_q1_filtered_training.py"),
      "--config",
      str(config),
      "--dry-run",
    ],
    capture_output=True,
    text=True,
    check=True,
    env=os.environ | {"CUDA_VISIBLE_DEVICES": ""},
  )
  initial = json.loads(result.stdout)["first_attempt"]
  assert "warm_start" not in initial and "warm_start_progress" not in initial
  assert "resume" not in initial and initial["iterations"] == 250000
  assert not (tmp_path / "fresh").exists()
