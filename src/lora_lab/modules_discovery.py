"""Auto-discover which linear layers to attach LoRA to.

For a classic dense transformer you can hard-code
``["q_proj", "k_proj", "v_proj", "o_proj", ...]``. But Qwen3.5-9B is a
multimodal MoE model with a hybrid (Gated DeltaNet + Gated Attention)
architecture, so the layer names are non-obvious and differ from Qwen2/3.

This module walks ``model.named_modules()`` and collects the *leaf* linear
layers that live in the language model, excluding:

  - the vision tower (we freeze and never adapt it),
  - the MoE router/gate (adapting the router is unstable and rarely helpful),
  - the ``lm_head`` / embeddings (kept full-precision, optionally in modules_to_save).

It returns the set of unique suffix names PEFT expects in ``target_modules``.
The tree printer is the main *learning* tool: run it once and you literally see
where adapters get injected inside the MoE.
"""

from __future__ import annotations

import re
from collections import Counter

import torch.nn as nn

# Substrings that mark a module as belonging to the frozen vision side or to
# layers we deliberately skip.
DEFAULT_EXCLUDE = (
    "vision",
    "visual",
    "image_",
    "patch_embed",
    "lm_head",
    "embed_tokens",
    "router",  # MoE routing gate
)

# Linear-like layer class names, including bitsandbytes' 4-bit/8-bit variants.
_LINEAR_CLASSNAMES = ("Linear", "Linear4bit", "Linear8bitLt", "Params4bit")

# Ablation groups: restrict LoRA to a functional slice of the LM. Substrings are
# matched against the FULL module path (covers Qwen dense, MoE experts and the
# hybrid DeltaNet/attention naming).
GROUP_INCLUDES: dict[str, tuple[str, ...] | None] = {
    "all": None,
    "attention": ("attn", "attention", "linear_attn"),
    "mlp": ("mlp", "expert", "feed_forward", "ffn"),
}


def _is_linear(module: nn.Module) -> bool:
    if isinstance(module, nn.Linear):
        return True
    return type(module).__name__ in _LINEAR_CLASSNAMES


def find_lora_targets(
    model: nn.Module,
    include: tuple[str, ...] | None = None,
    exclude: tuple[str, ...] = DEFAULT_EXCLUDE,
    group: str = "all",
) -> list[str]:
    """Return module names suitable for ``LoraConfig.target_modules``.

    ``group``: one of ``GROUP_INCLUDES`` ("all" / "attention" / "mlp") for
    ablations. ``include``: explicit substrings, overrides ``group``.

    Unrestricted discovery returns unique *suffix* names (PEFT matches the last
    dotted component). When a restriction applies, FULL module paths are
    returned instead: a suffix like ``gate_proj`` can exist in both attention
    and MLP blocks, and suffix matching would silently re-broaden the ablation.
    """
    if include is None:
        include = GROUP_INCLUDES[group]

    names: set[str] = set()
    for name, module in model.named_modules():
        if not _is_linear(module):
            continue
        if any(bad in name for bad in exclude):
            continue
        if include and not any(good in name for good in include):
            continue
        suffix = name.split(".")[-1]
        # Skip purely numeric names (e.g. ModuleList indices).
        if not suffix or suffix.isdigit():
            continue
        names.add(name if include else suffix)
    return sorted(names)


def summarize_targets(model: nn.Module, targets: list[str]) -> dict[str, int]:
    """Count how many layers each target suffix matches (sanity check)."""
    counts: Counter[str] = Counter()
    for name, module in model.named_modules():
        if not _is_linear(module):
            continue
        suffix = name.split(".")[-1]
        if suffix in targets:
            counts[suffix] += 1
    return dict(sorted(counts.items()))


def print_linear_tree(model: nn.Module, max_depth: int = 4, max_lines: int = 200) -> None:
    """Print a compact tree of linear layers to *see* the architecture.

    Collapses repeated numeric layer indices (``layers.0``, ``layers.1``, ...)
    into ``layers.{i}`` so a 40-layer MoE stays readable.
    """
    seen_patterns: set[str] = set()
    printed = 0
    for name, module in model.named_modules():
        if not _is_linear(module):
            continue
        depth = name.count(".")
        if depth > max_depth:
            continue
        pattern = re.sub(r"\.\d+", ".{i}", name)
        if pattern in seen_patterns:
            continue
        seen_patterns.add(pattern)
        shape = ""
        if hasattr(module, "in_features") and hasattr(module, "out_features"):
            shape = f"  [{module.in_features} -> {module.out_features}]"
        indent = "  " * pattern.count(".")
        print(f"{indent}{pattern}  ({type(module).__name__}){shape}")
        printed += 1
        if printed >= max_lines:
            print("  ... (truncated)")
            break
