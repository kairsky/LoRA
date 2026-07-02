"""Training orchestrator: glue config -> model -> data -> TRL SFTTrainer.

Uses TRL's ``SFTTrainer`` in "bring your own collator" mode
(``skip_prepare_dataset`` + ``remove_unused_columns=False``) because our samples
are multimodal and already turned into masked tensors by
``MultimodalJSONCollator``. TRL/transformers version differences are handled
defensively so this keeps working on the fast-moving bleeding edge.
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import TrainerCallback

from .config import RunConfig, snapshot_config
from .data import build_dataset
from .model import build_model


class VramCallback(TrainerCallback):
    """Log peak VRAM so you can *see* the memory cost of seq_len/batch choices."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and torch.cuda.is_available():
            logs["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / (1024**3), 2)

    def on_step_end(self, args, state, control, **kwargs):
        # Reset the peak periodically so the number reflects recent steps.
        if torch.cuda.is_available() and state.global_step % args.logging_steps == 0:
            torch.cuda.reset_peak_memory_stats()


def _build_sft_config(cfg: RunConfig, run_dir: Path):
    """Construct an SFTConfig (preferred) or fall back to TrainingArguments."""
    common = dict(
        output_dir=str(run_dir),
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.train.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        learning_rate=cfg.train.learning_rate,
        num_train_epochs=cfg.train.num_train_epochs,
        max_steps=cfg.train.max_steps,
        optim=cfg.train.optim,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        warmup_ratio=cfg.train.warmup_ratio,
        weight_decay=cfg.train.weight_decay,
        max_grad_norm=cfg.train.max_grad_norm,
        bf16=cfg.train.bf16,
        fp16=cfg.train.fp16,
        logging_steps=cfg.train.logging_steps,
        save_steps=cfg.train.save_steps,
        save_total_limit=cfg.train.save_total_limit,
        dataloader_num_workers=cfg.train.dataloader_num_workers,
        report_to=cfg.train.report_to,
        seed=cfg.train.seed,
        gradient_checkpointing=cfg.model.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        run_name=cfg.train.run_name or cfg.run_id,
    )
    try:
        from trl import SFTConfig

        return SFTConfig(
            dataset_kwargs={"skip_prepare_dataset": True},
            **common,
        )
    except Exception:
        from transformers import TrainingArguments

        return TrainingArguments(**common)


def _build_trainer(model, args, train_ds, eval_ds, collator, processor):
    try:
        from trl import SFTTrainer

        return SFTTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
            processing_class=processor,
        )
    except Exception:
        from transformers import Trainer

        return Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
            processing_class=processor,
        )


def run(cfg: RunConfig) -> Path:
    """Execute a full SFT run; returns the path to the saved adapter."""
    from transformers import set_seed

    set_seed(cfg.train.seed)
    run_dir = cfg.run_dir
    snapshot_config(cfg, run_dir)
    print(f"[train] run_dir = {run_dir}")

    model, processor, targets = build_model(cfg.model)
    train_ds, eval_ds, train_collator, _eval_collator = build_dataset(cfg.data, processor)
    print(f"[train] train samples = {len(train_ds)}, eval samples = {len(eval_ds)}")

    args = _build_sft_config(cfg, run_dir)
    trainer = _build_trainer(model, args, train_ds, eval_ds, train_collator, processor)
    trainer.add_callback(VramCallback())

    trainer.train()

    adapter_dir = run_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    print(f"[train] saved LoRA adapter -> {adapter_dir}")
    return adapter_dir
