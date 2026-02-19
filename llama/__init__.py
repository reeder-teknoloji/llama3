# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

# ── EARLIEST POSSIBLE FIX ────────────────────────────────────────────
# Force safe environment defaults at import time.
# On Windows CPU-only builds, gloo cannot create transport devices,
# so for world_size=1 we bypass torch.distributed entirely.
# These env vars are set defensively for any code path that might
# still read them.
import os as _os

_os.environ["MASTER_ADDR"] = "127.0.0.1"
_os.environ.setdefault("MASTER_PORT", "29500")
_os.environ.setdefault("RANK", "0")
_os.environ.setdefault("WORLD_SIZE", "1")
_os.environ.setdefault("LOCAL_RANK", "0")
# ─────────────────────────────────────────────────────────────────────

from .generation import Llama
from .model import ModelArgs, Transformer
from .tokenizer import Dialog, Tokenizer
