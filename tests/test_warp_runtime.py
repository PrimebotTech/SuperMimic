"""Multi-process compiler settings must retain the normal binary cache."""

import pytest
import warp as wp

from supermimic.warp_runtime import configure_cuda_compilation


@pytest.mark.parametrize("world_size", ["2", "8"])
def test_multi_rank_disables_only_pch(monkeypatch, world_size):
  monkeypatch.setenv("WORLD_SIZE", world_size)
  monkeypatch.setattr(wp.config, "use_precompiled_headers", True)
  monkeypatch.setattr(wp.config, "cache_kernels", True)
  monkeypatch.setattr(wp.config, "kernel_cache_dir", "/shared/kernel-cache")
  configure_cuda_compilation()
  assert not wp.config.use_precompiled_headers
  assert wp.config.cache_kernels
  assert wp.config.kernel_cache_dir == "/shared/kernel-cache"


@pytest.mark.parametrize("world_size", [None, "1"])
@pytest.mark.parametrize("pch", [True, False])
def test_single_rank_preserves_pch_setting(monkeypatch, world_size, pch):
  if world_size is None:
    monkeypatch.delenv("WORLD_SIZE", raising=False)
  else:
    monkeypatch.setenv("WORLD_SIZE", world_size)
  monkeypatch.setattr(wp.config, "use_precompiled_headers", pch)
  configure_cuda_compilation()
  assert wp.config.use_precompiled_headers is pch
