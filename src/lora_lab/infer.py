"""Single-image inference with a trained LoRA adapter.

Loads the (optionally 4-bit) base model, attaches the adapter, and generates
JSON for one image. Use this for spot-checking; use ``eval.py`` for metrics.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from .config import ModelConfig
from .data import _downscale
from .generate_utils import safe_generate
from .model import build_model


def load_for_inference(model_cfg: ModelConfig, adapter_dir: str | Path | None):
    model, processor, _ = build_model(model_cfg, verbose=False)
    if adapter_dir is not None:
        model.load_adapter(str(adapter_dir), adapter_name="trained")
        model.set_adapter("trained")
    model.eval()
    return model, processor


@torch.no_grad()
def extract(
    model,
    processor,
    image: str | Path | Image.Image,
    instruction: str,
    max_new_tokens: int = 768,
    image_max_pixels: int | None = 1_048_576,
    schema: dict | None = None,
    constrained: bool = False,
) -> str:
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = _downscale(image.convert("RGB"), image_max_pixels)
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    gen_kwargs: dict = {"max_new_tokens": max_new_tokens, "do_sample": False}
    if constrained:
        from .constrained import build_json_logits_processor

        lp = build_json_logits_processor(model, processor, schema)
        if lp is not None:
            gen_kwargs["logits_processor"] = lp
    out = safe_generate(model, inputs, gen_kwargs)
    gen = out[0][inputs["input_ids"].shape[1] :]
    return processor.tokenizer.decode(gen, skip_special_tokens=True)
