"""Minimal Triton stub for environments where the optional dependency is unavailable.

DNABERT-2's remote model code imports Triton at module import time, but the model
falls back to a PyTorch attention implementation when Triton is unavailable.
This stub is only meant to satisfy those imports on non-CUDA environments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from . import language

__version__ = "0.0.0"

@dataclass(slots=True)
class Config:
    params: dict[str, Any] = field(default_factory=dict)
    num_warps: int | None = None
    num_stages: int | None = None
    pre_hook: Callable[..., Any] | None = None
    post_hook: Callable[..., Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __init__(self, params: dict[str, Any], num_warps: int | None = None, num_stages: int | None = None, **kwargs: Any):
        object.__setattr__(self, "params", params)
        object.__setattr__(self, "num_warps", num_warps)
        object.__setattr__(self, "num_stages", num_stages)
        object.__setattr__(self, "pre_hook", kwargs.pop("pre_hook", None))
        object.__setattr__(self, "post_hook", kwargs.pop("post_hook", None))
        object.__setattr__(self, "extra", kwargs)


def _identity_decorator(fn: Callable[..., Any] | None = None, **_kwargs: Any):
    if fn is None:
        def decorator(inner: Callable[..., Any]) -> Callable[..., Any]:
            return inner

        return decorator
    return fn


def autotune(*_args: Any, **_kwargs: Any):
    return _identity_decorator


def heuristics(*_args: Any, **_kwargs: Any):
    return _identity_decorator


def jit(fn: Callable[..., Any] | None = None, **_kwargs: Any):
    return _identity_decorator(fn)


def cdiv(x: int, y: int) -> int:
    return -(-x // y)


def next_power_of_2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()
