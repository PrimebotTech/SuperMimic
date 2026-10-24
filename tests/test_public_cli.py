"""Portable entry-point and launcher behavior without CUDA or real Q1 assets."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from supermimic import cli, launch
from supermimic.launch import Config, build_command, load_options, runtime_environment


def test_dispatch_forwards_arguments_and_restores_process_state(monkeypatch):
  original = ["supermimic", "train", "--device", "cpu", "--iterations", "2"]
  seen = []
  monkeypatch.setattr(sys, "argv", original)
  monkeypatch.setattr(
    cli,
    "import_module",
    lambda name: SimpleNamespace(main=lambda: seen.append((name, sys.argv.copy()))),
  )
  cli.main()
  assert seen == [
    ("supermimic.train", ["supermimic train", "--device", "cpu", "--iterations", "2"])
  ]
  assert sys.argv is original


def test_unknown_command_is_rejected_before_import(monkeypatch):
  monkeypatch.setattr(sys, "argv", ["supermimic", "typo"])
  with pytest.raises(SystemExit) as error:
    cli.main()
  assert error.value.code == 2


def test_toml_paths_resolve_against_configuration_location(tmp_path: Path):
  path = tmp_path / "configs" / "train.toml"
  path.parent.mkdir()
  path.write_text(
    '[train]\nrobot_xml="../assets/q1.xml"\nmotion_dir="../data"\n'
    'output_dir="../logs/run"\nwarm_start="../logs/source/checkpoint_2.pt"\n'
    "num_envs=8\nrandomize=false\n"
  )
  options = load_options(path)
  assert options["robot_xml"] == str(tmp_path / "assets/q1.xml")
  assert options["motion_dir"] == str(tmp_path / "data")
  assert options["warm_start"] == str(tmp_path / "logs/source/checkpoint_2.pt")
  command = build_command(options, 2)
  assert "torch.distributed.run" in command
  assert command[command.index("--nproc-per-node") + 1] == "2"
  assert "--no-randomize" in command
  assert command[command.index("--num-envs") + 1] == "8"


@pytest.mark.parametrize("gpus", [0, 3])
def test_launcher_rejects_invalid_global_count_partition(gpus: int):
  with pytest.raises(ValueError):
    build_command({"num_envs": 8}, gpus)


def test_unknown_training_options_fail_before_launch(tmp_path: Path):
  path = tmp_path / "bad.toml"
  path.write_text("[train]\nnum_envz=8\n")
  with pytest.raises(ValueError, match="num_envz"):
    load_options(path)


def test_network_temp_environment_is_isolated_for_training(monkeypatch, tmp_path):
  inherited = tmp_path / "network"
  monkeypatch.setenv("TMPDIR", str(inherited / "tmp"))
  monkeypatch.setenv("XDG_CACHE_HOME", str(inherited / "xdg"))
  monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(inherited / "inductor"))
  monkeypatch.setenv("TRITON_CACHE_DIR", str(inherited / "triton"))
  monkeypatch.setenv("OMP_NUM_THREADS", "2")
  original = launch.os.environ.copy()
  local = tmp_path / "local"
  environment = runtime_environment(local)
  for name in (
    "TMPDIR",
    "XDG_CACHE_HOME",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
  ):
    assert Path(environment[name]).is_relative_to(local)
    assert Path(environment[name]).is_dir()
  assert environment["OMP_NUM_THREADS"] == "2"
  assert launch.os.environ == original
  assert not inherited.exists()


def test_dry_run_does_not_create_compiler_cache(monkeypatch, tmp_path):
  config = tmp_path / "train.toml"
  config.write_text(
    '[train]\nrobot_xml="q1.xml"\nmotion_dir="motions"\noutput_dir="run"\n'
  )
  cache = tmp_path / "cache"
  monkeypatch.setattr(
    launch.tyro,
    "cli",
    lambda _: Config(config=config, dry_run=True, compiler_cache_dir=cache),
  )
  launch.main()
  assert not cache.exists()
  assert not (tmp_path / "run").exists()
