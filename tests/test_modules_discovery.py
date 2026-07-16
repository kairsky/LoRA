"""Target-module discovery: exclusions and ablation groups.

A fake Qwen-like tree checks that vision/router/lm_head never get adapters and
that the "attention"/"mlp" ablation groups select disjoint, correct slices.
"""

import torch.nn as nn

from lora_lab.modules_discovery import find_lora_targets


def _linear() -> nn.Linear:
    return nn.Linear(2, 2)


class FakeQwen(nn.Module):
    """Mimics the naming of a Qwen-style VLM with one MoE layer."""

    def __init__(self):
        super().__init__()
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.q_proj = _linear()
        layer.self_attn.k_proj = _linear()
        layer.self_attn.v_proj = _linear()
        layer.self_attn.o_proj = _linear()
        layer.mlp = nn.Module()
        expert = nn.Module()
        expert.gate_proj = _linear()
        expert.up_proj = _linear()
        expert.down_proj = _linear()
        layer.mlp.experts = nn.ModuleList([expert])
        layer.mlp.router = _linear()  # MoE routing gate - must never be adapted
        self.layers = nn.ModuleList([layer])
        self.visual = nn.Module()
        self.visual.proj = _linear()  # vision tower - frozen, never adapted
        self.lm_head = _linear()


def test_all_group_returns_suffixes_and_excludes_special():
    targets = find_lora_targets(FakeQwen())
    assert targets == sorted(
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    assert "router" not in targets
    assert "lm_head" not in targets
    assert "proj" not in targets  # visual.proj


def test_attention_group_selects_only_attention_paths():
    targets = find_lora_targets(FakeQwen(), group="attention")
    # Restricted groups return FULL paths (suffix matching could re-broaden).
    assert targets == [
        "layers.0.self_attn.k_proj",
        "layers.0.self_attn.o_proj",
        "layers.0.self_attn.q_proj",
        "layers.0.self_attn.v_proj",
    ]


def test_mlp_group_selects_only_expert_paths():
    targets = find_lora_targets(FakeQwen(), group="mlp")
    assert targets == [
        "layers.0.mlp.experts.0.down_proj",
        "layers.0.mlp.experts.0.gate_proj",
        "layers.0.mlp.experts.0.up_proj",
    ]
    # The router lives under mlp but stays excluded.
    assert not any("router" in t for t in targets)


def test_groups_are_disjoint():
    attn = set(find_lora_targets(FakeQwen(), group="attention"))
    mlp = set(find_lora_targets(FakeQwen(), group="mlp"))
    assert not (attn & mlp)
