"""Configure Warp's CUDA compiler for torchrun processes."""

import os

import warp as wp


def configure_cuda_compilation() -> None:
  """Avoid NVRTC precompiled-header races between ranks sharing a cache.

  Warp 1.12 uses PID/thread-specific build directories and publishes binaries
  safely, but CUDA 12.9's PCH files live in the common kernel cache directory.
  Concurrent compilers can delete the same .pch and fail with NVRTC error 6.
  Disable only PCH for multi-process jobs; compiled kernel caching stays on.
  Call before launching any Warp CUDA kernels. CPU simulation never calls it.
  """
  if int(os.environ.get("WORLD_SIZE", "1")) > 1:
    wp.config.use_precompiled_headers = False
