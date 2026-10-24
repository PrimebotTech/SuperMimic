"""Scoped compatibility for Torch 2.7's scalar broadcast indices in Triton 3.3."""

from importlib.metadata import version
from typing import Any, cast

import torch._inductor.config
import triton.language as tl
from triton.language import semantic
from triton.language.core import (
  _shape_check_impl,
  _unwrap_iterable,
  builtin,
)

to_tensor: Any = getattr(semantic, "to_tensor", None)


@builtin
def tensor_broadcast_to(value: Any, *shape: Any, _builder: Any = None) -> Any:
  """Coerce a scalar index to a tensor before requesting a dense broadcast."""
  checked = _shape_check_impl(_unwrap_iterable(shape))
  return semantic.broadcast_impl_shape(to_tensor(value, _builder), checked, _builder)


def configure_compiler() -> None:
  if version("torch").startswith("2.7.") and version("triton").startswith("3.3."):
    # Keep compilation in this process so worker imports retain the same patch.
    # No installed dependency files or GPU numerical formulas are changed.
    torch._inductor.config.compile_threads = 1
    cast(Any, tl).broadcast_to = tensor_broadcast_to
