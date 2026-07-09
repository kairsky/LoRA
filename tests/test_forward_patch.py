"""TRL's SFTTrainer leaves a training-only ``forward`` monkey-patch on the model;
``_strip_instance_forward_patches`` must remove it (and be a no-op otherwise)."""

import torch
import torch.nn as nn

from lora_lab.train import _strip_instance_forward_patches


class Inner(nn.Module):
    def forward(self, x):
        return x + 1


class Outer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Inner()

    def forward(self, x):
        return self.model(x)


def test_strip_removes_trl_style_patch():
    outer = Outer()
    # Simulate TRL patching forward on both the wrapper and the inner model.
    outer.forward = lambda x: "patched-outer"
    outer.model.forward = lambda x: "patched-inner"
    assert outer(torch.tensor(1)) == "patched-outer"

    _strip_instance_forward_patches(outer)

    # Class forwards are restored: Outer -> Inner -> x + 1.
    assert outer(torch.tensor(1)).item() == 2


def test_strip_is_noop_on_clean_model():
    outer = Outer()
    _strip_instance_forward_patches(outer)
    _strip_instance_forward_patches(outer)  # idempotent
    assert outer(torch.tensor(1)).item() == 2
