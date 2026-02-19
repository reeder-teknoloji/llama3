"""
Drop-in replacement for fairscale model-parallel primitives when
torch.distributed is not available or broken (e.g. Windows CPU-only
builds where gloo cannot create transport devices).

For **world_size == 1** every fairscale "parallel" layer is equivalent to
its standard PyTorch counterpart:

    ColumnParallelLinear  ➜  nn.Linear   (bias=False)
    RowParallelLinear     ➜  nn.Linear   (bias=False)
    VocabParallelEmbedding ➜ nn.Embedding

The state-dict keys (`.weight`) are identical, so checkpoint loading
works transparently.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any, Callable, Optional

# ── model parallel state (trivial for single-process) ─────────────

_MODEL_PARALLEL_IS_INITIALIZED = True
_MODEL_PARALLEL_SIZE = 1
_MODEL_PARALLEL_RANK = 0


def get_model_parallel_world_size() -> int:
    return _MODEL_PARALLEL_SIZE


def get_model_parallel_rank() -> int:
    return _MODEL_PARALLEL_RANK


def initialize_model_parallel(model_parallel_size: int = 1) -> None:
    global _MODEL_PARALLEL_SIZE
    _MODEL_PARALLEL_SIZE = model_parallel_size


def model_parallel_is_initialized() -> bool:
    return _MODEL_PARALLEL_IS_INITIALIZED


# ── drop-in layer replacements ────────────────────────────────────

class ColumnParallelLinear(nn.Linear):
    """nn.Linear stand-in for fairscale ColumnParallelLinear (MP=1)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        gather_output: bool = True,
        init_method: Optional[Callable[..., Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(in_features, out_features, bias=bias)
        # init_method is ignored — weights come from checkpoint.


class RowParallelLinear(nn.Linear):
    """nn.Linear stand-in for fairscale RowParallelLinear (MP=1)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        input_is_parallel: bool = False,
        init_method: Optional[Callable[..., Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(in_features, out_features, bias=bias)


class VocabParallelEmbedding(nn.Embedding):
    """nn.Embedding stand-in for fairscale VocabParallelEmbedding (MP=1)."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        init_method: Optional[Callable[..., Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(num_embeddings, embedding_dim)
