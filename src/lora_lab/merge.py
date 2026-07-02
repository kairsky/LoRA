"""Merge a LoRA adapter into the base weights and export a standalone model.

IMPORTANT: you cannot cleanly merge LoRA into a 4-bit (QLoRA) base and keep the
quantization. The correct recipe is to reload the base model in half precision
(bf16/fp16), attach the adapter, then ``merge_and_unload()``. That is what this
module does, so it needs enough RAM/VRAM to hold the base in bf16 (~20 GB for a
10B model). For inference-only use it is usually better to just keep the adapter
separate (see ``infer.py``).
"""

from __future__ import annotations

from pathlib import Path

import torch


def merge_adapter(
    base_model_name: str,
    adapter_dir: str | Path,
    output_dir: str | Path,
    modality: str = "multimodal",
    dtype: str = "bfloat16",
    trust_remote_code: bool = False,
) -> Path:
    from peft import PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoProcessor,
    )

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    loader = AutoModelForImageTextToText if modality == "multimodal" else AutoModelForCausalLM

    print(f"[merge] loading base '{base_model_name}' in {dtype} (no quantization)...")
    base = loader.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        device_map="auto",
    )

    print(f"[merge] attaching adapter '{adapter_dir}'...")
    model = PeftModel.from_pretrained(base, str(adapter_dir))

    print("[merge] merge_and_unload()...")
    model = model.merge_and_unload()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    try:
        AutoProcessor.from_pretrained(str(adapter_dir)).save_pretrained(str(output_dir))
    except Exception:
        AutoProcessor.from_pretrained(base_model_name).save_pretrained(str(output_dir))
    print(f"[merge] saved merged model -> {output_dir}")
    return output_dir


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Merge a LoRA adapter into base weights.")
    p.add_argument("--base", required=True, help="Base model name or path")
    p.add_argument("--adapter", required=True, help="Adapter directory")
    p.add_argument("--out", required=True, help="Output directory for merged model")
    p.add_argument("--modality", default="multimodal", choices=["multimodal", "text"])
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = p.parse_args()
    merge_adapter(args.base, args.adapter, args.out, args.modality, args.dtype)
