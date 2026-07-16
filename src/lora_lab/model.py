"""Model layer: load (optionally 4-bit) base model + processor, then wrap with LoRA.

This is the only module that knows about quantization and PEFT. The QLoRA recipe:

    1. Load the base model in 4-bit NF4 (double-quantized) with a bf16 compute dtype.
    2. ``prepare_model_for_kbit_training`` (enables gradient checkpointing, casts
       layernorms to fp32, makes inputs require grad).
    3. Freeze the vision tower so we only adapt the language model.
    4. Discover ``target_modules`` (unless pinned in the config) and inject LoRA.
    5. ``print_trainable_parameters`` -> you should see <1% of params trainable.
"""

from __future__ import annotations

import torch

from .config import ModelConfig
from .modules_discovery import find_lora_targets, print_linear_tree, summarize_targets

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


def _load_processor(model_cfg: ModelConfig):
    from transformers import AutoProcessor, AutoTokenizer

    try:
        proc = AutoProcessor.from_pretrained(
            model_cfg.name_or_path, trust_remote_code=model_cfg.trust_remote_code
        )
    except Exception:
        # Text-only fallback: expose a tokenizer under the same attribute name.
        proc = AutoTokenizer.from_pretrained(
            model_cfg.name_or_path, trust_remote_code=model_cfg.trust_remote_code
        )
    tok = getattr(proc, "tokenizer", proc)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    return proc


def _load_base_model(model_cfg: ModelConfig):
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        BitsAndBytesConfig,
    )

    compute_dtype = _DTYPES[model_cfg.torch_dtype]
    quant_config = None
    if model_cfg.quant.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model_cfg.quant.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=model_cfg.quant.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=_DTYPES[model_cfg.quant.bnb_4bit_compute_dtype],
        )

    kwargs = dict(
        quantization_config=quant_config,
        torch_dtype=compute_dtype,
        attn_implementation=model_cfg.attn_implementation,
        trust_remote_code=model_cfg.trust_remote_code,
        device_map={"": 0},  # single-GPU; accelerate/DeepSpeed handles multi-GPU
    )

    if model_cfg.modality == "multimodal":
        loader = AutoModelForImageTextToText
    else:
        loader = AutoModelForCausalLM

    try:
        model = loader.from_pretrained(model_cfg.name_or_path, **kwargs)
    except Exception:
        # Some VLMs still register only under AutoModelForCausalLM on older code.
        model = AutoModelForCausalLM.from_pretrained(model_cfg.name_or_path, **kwargs)
    return model


def _freeze_vision(model) -> int:
    frozen = 0
    for name, param in model.named_parameters():
        if any(tok in name.lower() for tok in ("vision", "visual", "image_")):
            param.requires_grad_(False)
            frozen += 1
    return frozen


def build_model(model_cfg: ModelConfig, verbose: bool = True):
    """Return ``(peft_model, processor, target_modules)`` ready for training."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    processor = _load_processor(model_cfg)
    model = _load_base_model(model_cfg)

    if model_cfg.quant.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=model_cfg.gradient_checkpointing
        )
    elif model_cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if model_cfg.freeze_vision:
        n = _freeze_vision(model)
        if verbose:
            print(f"[model] froze {n} vision parameter tensors")

    # Resolve LoRA target modules.
    if model_cfg.lora.target_modules == "auto":
        targets = find_lora_targets(model, group=model_cfg.lora.target_groups)
        if verbose:
            print("\n[model] linear layer tree (LoRA injection points):")
            print_linear_tree(model)
            if model_cfg.lora.target_groups != "all":
                print(
                    f"\n[model] auto target_modules (group={model_cfg.lora.target_groups}): "
                    f"{len(targets)} modules"
                )
            else:
                print(f"\n[model] auto target_modules: {targets}")
                print(f"[model] target match counts: {summarize_targets(model, targets)}")
    else:
        targets = list(model_cfg.lora.target_modules)

    if not targets:
        raise RuntimeError(
            "No LoRA target modules found. Inspect print_linear_tree() output and set "
            "model.lora.target_modules explicitly in the config."
        )

    lora_kwargs = dict(
        r=model_cfg.lora.r,
        lora_alpha=model_cfg.lora.lora_alpha,
        lora_dropout=model_cfg.lora.lora_dropout,
        bias=model_cfg.lora.bias,
        target_modules=targets,
        modules_to_save=model_cfg.lora.modules_to_save or None,
        task_type=model_cfg.lora.task_type,
        use_dora=model_cfg.lora.use_dora,
        use_rslora=model_cfg.lora.use_rslora,
        init_lora_weights=model_cfg.lora.init_lora_weights,
    )
    # LoftQ needs an explicit LoftQConfig; it re-initializes adapters to best
    # approximate the quantization error of the base weights.
    if model_cfg.lora.init_lora_weights == "loftq":
        try:
            from peft import LoftQConfig

            lora_kwargs["loftq_config"] = LoftQConfig(loftq_bits=4)
        except Exception as exc:  # noqa: BLE001
            print(f"[model] LoftQ requested but unavailable ({exc}); using default init")
            lora_kwargs["init_lora_weights"] = True

    lora_config = LoraConfig(**lora_kwargs)
    if verbose:
        variants = [
            n
            for n, on in (
                ("DoRA", model_cfg.lora.use_dora),
                ("rsLoRA", model_cfg.lora.use_rslora),
            )
            if on
        ]
        init = model_cfg.lora.init_lora_weights
        if init is not True:
            variants.append(f"init={init}")
        print(f"[model] LoRA variants: {variants or ['vanilla LoRA']}")
    model = get_peft_model(model, lora_config)

    # Gradient checkpointing needs input grads to flow through the frozen base.
    if model_cfg.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.config.use_cache = False

    if verbose:
        model.print_trainable_parameters()
    return model, processor, targets
