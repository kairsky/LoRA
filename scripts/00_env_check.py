"""Phase 0 fail-fast environment check for LoRA Lab.

Verifies the stack is ready for QLoRA on an RTX 5090 (Blackwell / sm_120) under
native Windows before we waste time loading a 9B model:

  1. Python + OS info
  2. PyTorch sees CUDA and reports the right compute capability (12, 0)
  3. Available VRAM is enough
  4. bitsandbytes can actually run a 4-bit matmul on the GPU
  5. transformers / peft / trl import cleanly

Run:
    python scripts/00_env_check.py

Exit code is non-zero if any *blocking* check fails, so CI / a launcher can
gate training on it.
"""

from __future__ import annotations

import platform
import sys

# Blackwell (RTX 50xx) reports compute capability 12.0 / sm_120.
EXPECTED_CAPABILITY = (12, 0)
MIN_VRAM_GB = 20.0  # QLoRA of a 10B model is comfortable well under this, but warn early.

OK = "[ OK ]"
WARN = "[WARN]"
FAIL = "[FAIL]"


class CheckState:
    def __init__(self) -> None:
        self.blocking_failures: list[str] = []
        self.warnings: list[str] = []

    def fail(self, msg: str) -> None:
        self.blocking_failures.append(msg)
        print(f"{FAIL} {msg}")

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"{WARN} {msg}")

    def ok(self, msg: str) -> None:
        print(f"{OK} {msg}")


def check_python(state: CheckState) -> None:
    print("\n== Python / OS ==")
    state.ok(f"Python {platform.python_version()} on {platform.system()} {platform.release()}")
    if sys.version_info < (3, 11):  # noqa: UP036 - runtime guard, not dead code
        state.fail(f"Python >=3.11 required, found {platform.python_version()}")


def check_torch(state: CheckState):
    print("\n== PyTorch / CUDA ==")
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        state.fail(f"Cannot import torch: {exc}")
        return None

    state.ok(f"torch {torch.__version__}")
    cuda_build = getattr(torch.version, "cuda", None)
    state.ok(f"torch built against CUDA {cuda_build}")

    if not torch.cuda.is_available():
        state.fail(
            "torch.cuda.is_available() is False. On a 5090 you need the cu128 "
            "nightly wheel: pip install --pre torch --index-url "
            "https://download.pytorch.org/whl/nightly/cu128"
        )
        return torch

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    state.ok(f"GPU 0: {name} (compute capability {cap[0]}.{cap[1]})")

    if cap[0] < EXPECTED_CAPABILITY[0]:
        state.warn(
            f"Compute capability {cap} < expected {EXPECTED_CAPABILITY}. "
            "This does not look like a Blackwell card; is the right GPU selected?"
        )
    elif cap != EXPECTED_CAPABILITY:
        state.warn(f"Compute capability {cap} != expected {EXPECTED_CAPABILITY} (probably fine).")

    # Confirm the installed torch actually supports this arch (Blackwell needs sm_120 kernels).
    arch_list = getattr(torch.cuda, "get_arch_list", lambda: [])()
    if arch_list:
        state.ok(f"torch arch list: {arch_list}")
        sm_tag = f"sm_{cap[0]}{cap[1]}"
        if not any(sm_tag in a for a in arch_list):
            state.warn(
                f"{sm_tag} not in torch arch list. Kernels may fall back / fail on Blackwell. "
                "Use the cu128 nightly build."
            )

    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if total_gb < MIN_VRAM_GB:
        state.warn(f"VRAM {total_gb:.1f} GB < {MIN_VRAM_GB} GB; tune max_seq_len / batch.")
    else:
        state.ok(f"VRAM total: {total_gb:.1f} GB")

    # A tiny real matmul on device confirms kernels actually launch on this arch.
    try:
        x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        _ = (x @ x).sum().item()
        torch.cuda.synchronize()
        state.ok("bfloat16 matmul on GPU succeeded")
    except Exception as exc:  # noqa: BLE001
        state.fail(f"GPU matmul failed (arch/kernel mismatch?): {exc}")

    return torch


def check_bitsandbytes(state: CheckState, torch) -> None:
    print("\n== bitsandbytes (4-bit QLoRA backend) ==")
    if torch is None or not torch.cuda.is_available():
        state.warn("Skipping bitsandbytes GPU test (no CUDA).")
        return
    try:
        import bitsandbytes as bnb
    except Exception as exc:  # noqa: BLE001
        state.fail(f"Cannot import bitsandbytes: {exc}")
        return

    state.ok(f"bitsandbytes {getattr(bnb, '__version__', '?')}")
    try:
        import torch.nn as nn

        # Build a Linear4bit layer, move to GPU, run a forward pass.
        linear = bnb.nn.Linear4bit(
            64,
            64,
            bias=False,
            compute_dtype=torch.bfloat16,
            quant_type="nf4",
        ).cuda()
        with torch.no_grad():
            out = linear(torch.randn(4, 64, device="cuda", dtype=torch.bfloat16))
        assert out.shape == (4, 64)
        _ = nn  # silence unused if edited later
        state.ok("NF4 Linear4bit forward pass on GPU succeeded")
    except Exception as exc:  # noqa: BLE001
        state.fail(
            f"bitsandbytes 4-bit forward failed: {exc}. "
            "On Windows ensure a recent bitsandbytes wheel with CUDA 12.x support."
        )


def check_libs(state: CheckState) -> None:
    print("\n== transformers / peft / trl / datasets ==")
    for mod in ("transformers", "peft", "trl", "datasets", "accelerate"):
        try:
            m = __import__(mod)
            state.ok(f"{mod} {getattr(m, '__version__', '?')}")
        except Exception as exc:  # noqa: BLE001
            state.fail(f"Cannot import {mod}: {exc}")


def main() -> int:
    print("=" * 68)
    print("LoRA Lab :: Phase 0 environment check")
    print("=" * 68)

    state = CheckState()
    check_python(state)
    torch = check_torch(state)
    check_bitsandbytes(state, torch)
    check_libs(state)

    print("\n" + "=" * 68)
    if state.warnings:
        print(f"{len(state.warnings)} warning(s):")
        for w in state.warnings:
            print(f"  - {w}")
    if state.blocking_failures:
        print(f"\n{FAIL} {len(state.blocking_failures)} blocking failure(s):")
        for f in state.blocking_failures:
            print(f"  - {f}")
        print("\nFix these before training. Aborting.")
        return 1

    print(f"{OK} Environment looks ready for QLoRA on Qwen3.5-9B.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
