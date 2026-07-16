"""DPO stage: preference-tune the extractor on (chosen, rejected) JSON pairs.

Sits ON TOP of SFT: pass the SFT adapter as the starting point (or train a
fresh adapter from the base model). Uses TRL's ``DPOTrainer`` with
``ref_model=None`` - with a PEFT model TRL computes the reference logits by
temporarily disabling the adapter, so no second model copy is needed (crucial
for QLoRA on one GPU).
"""

from __future__ import annotations

from pathlib import Path

from .config import RunConfig, snapshot_config
from .model import build_model
from .pairs import load_pairs
from .registry import get_dataset_builder
from .train import _available_reporters


def _to_trl_dataset(pairs: list[dict], data_cfg, instruction: str):
    """TRL conversational preference format with an images column."""
    from datasets import Dataset

    train_ds, eval_ds = get_dataset_builder(data_cfg.type)(data_cfg)
    by_split = {"train": train_ds, "validation": eval_ds}

    def _msgs(text: str) -> list[dict]:
        return [{"role": "assistant", "content": [{"type": "text", "text": text}]}]

    rows = []
    for pair in pairs:
        image = by_split[pair.get("split", "train")][pair["index"]]["image"]
        rows.append(
            {
                "images": [image],
                "prompt": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": instruction},
                        ],
                    }
                ],
                "chosen": _msgs(pair["chosen"]),
                "rejected": _msgs(pair["rejected"]),
            }
        )
    return Dataset.from_list(rows)


def run_dpo(cfg: RunConfig, pairs_path: str | Path, sft_adapter: str | Path | None = None):
    """Train a DPO adapter; returns the adapter directory."""
    from transformers import set_seed
    from trl import DPOConfig, DPOTrainer

    set_seed(cfg.train.seed)
    run_dir = cfg.run_dir
    snapshot_config(cfg, run_dir)
    print(f"[dpo] run_dir = {run_dir}")

    model, processor, _targets = build_model(cfg.model)
    if sft_adapter is not None:
        # Continue from the SFT adapter instead of a fresh (zero) adapter.
        model.load_adapter(str(sft_adapter), adapter_name="sft")
        model.set_adapter("sft")
        print(f"[dpo] starting from SFT adapter: {sft_adapter}")

    pairs = load_pairs(pairs_path)
    dataset = _to_trl_dataset(pairs, cfg.data, cfg.data.instruction)
    print(f"[dpo] preference pairs = {len(dataset)}")

    args = DPOConfig(
        output_dir=str(run_dir),
        beta=cfg.train.dpo_beta,
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        learning_rate=cfg.train.learning_rate,
        num_train_epochs=cfg.train.num_train_epochs,
        max_steps=cfg.train.max_steps,
        optim=cfg.train.optim,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        warmup_ratio=cfg.train.warmup_ratio,
        max_grad_norm=cfg.train.max_grad_norm,
        bf16=cfg.train.bf16,
        fp16=cfg.train.fp16,
        logging_steps=cfg.train.logging_steps,
        save_steps=cfg.train.save_steps,
        save_total_limit=cfg.train.save_total_limit,
        report_to=_available_reporters(cfg.train.report_to),
        seed=cfg.train.seed,
        gradient_checkpointing=cfg.model.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        run_name=cfg.train.run_name or cfg.run_id,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # PEFT: reference = same model with adapter disabled
        args=args,
        train_dataset=dataset,
        processing_class=processor,
    )
    trainer.train()

    adapter_dir = run_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    print(f"[dpo] saved DPO adapter -> {adapter_dir}")
    return adapter_dir
