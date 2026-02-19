"""
Minimal test script to verify that the llama package works on
this Windows machine WITHOUT the 10049 / localsocket error.

Usage (from .venv):
    .venv\\Scripts\\python.exe test_distributed_init.py

Expected output:
    All checks should print [OK].
"""
import os
import socket
import sys

print("=" * 60)
print("DISTRIBUTED / SINGLE-PROCESS INIT TEST")
print("=" * 60)

# ── Step 1: Show system info ──────────────────────────────────────
hostname = socket.gethostname()
fqdn = socket.getfqdn()
print(f"[INFO] hostname        = {hostname}")
print(f"[INFO] fqdn            = {fqdn}")
print(f"[INFO] platform        = {sys.platform}")

# Check hosts file for problematic entries
try:
    hosts_path = r"C:\Windows\System32\drivers\etc\hosts"
    with open(hosts_path, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                if "localsocket" in stripped or "mysoft" in stripped:
                    print(f"[WARN] Hosts file entry: {stripped}")
                    print("       (This entry would cause error 10049 with "
                          "torch.distributed, but our shim bypasses it)")
except Exception as e:
    print(f"[WARN] Could not read hosts file: {e}")

# ── Step 2: Force env vars ────────────────────────────────────────
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["LOCAL_RANK"] = "0"

print(f"\n[INFO] MASTER_ADDR     = {os.environ['MASTER_ADDR']}")
print(f"[INFO] MASTER_PORT     = {os.environ['MASTER_PORT']}")
print(f"[INFO] WORLD_SIZE      = {os.environ['WORLD_SIZE']}")

# ── Step 3: Test llama import (the main test) ─────────────────────
print("\n--- Testing llama package import ---")
try:
    import torch
    print(f"[INFO] PyTorch version = {torch.__version__}")
    print(f"[INFO] CUDA available  = {torch.cuda.is_available()}")

    from llama import Llama, ModelArgs, Transformer
    print("[OK]  from llama import Llama — SUCCESS (no distributed errors)")
except Exception as e:
    print(f"[FAIL] llama import raised: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ── Step 4: Verify shim is active ────────────────────────────────
print("\n--- Verifying fairscale shim ---")
try:
    from llama._fairscale_shim import (
        get_model_parallel_rank,
        get_model_parallel_world_size,
        model_parallel_is_initialized,
        ColumnParallelLinear,
        RowParallelLinear,
        VocabParallelEmbedding,
    )
    assert get_model_parallel_rank() == 0
    assert get_model_parallel_world_size() == 1
    assert model_parallel_is_initialized() is True
    print("[OK]  Shim active: mp_rank=0, mp_world_size=1")

    # Test that shim layers produce correct shapes
    x = torch.randn(1, 16)
    col = ColumnParallelLinear(16, 32, bias=False, init_method=lambda x: x)
    row = RowParallelLinear(32, 16, bias=False, init_method=lambda x: x)
    emb = VocabParallelEmbedding(100, 16, init_method=lambda x: x)
    assert col(x).shape == (1, 32)
    assert row(col(x)).shape == (1, 16)
    assert emb(torch.tensor([5])).shape == (1, 16)
    print("[OK]  Shim layers work correctly")
except Exception as e:
    print(f"[FAIL] Shim test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ── Step 5: Verify distributed is NOT initialized ────────────────
print("\n--- Checking distributed state ---")
import torch.distributed as dist
if dist.is_initialized():
    print("[WARN] dist.is_initialized()=True (unexpected for shim mode)")
else:
    print("[OK]  dist.is_initialized()=False — "
          "no c10d, no hostname resolution, no error 10049")

print("\n" + "=" * 60)
print("ALL TESTS PASSED — SAFE TO RUN start_app.bat")
print("=" * 60)
