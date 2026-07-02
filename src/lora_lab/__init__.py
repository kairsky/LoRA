"""LoRA Lab: a config-driven playground for learning LoRA/QLoRA deeply.

The package is intentionally split into thin, single-responsibility layers so
that swapping the model, dataset, or training method is a config change rather
than a code rewrite:

    config.py             -> declarative pydantic schemas (the "what")
    registry.py           -> name -> builder mapping for datasets/models/metrics
    data.py               -> dataset loading, chat template, multimodal collator
    modules_discovery.py  -> auto-find LoRA target_modules inside the LM
    model.py              -> quantization + PEFT wrapping (the "how" for models)
    train.py              -> SFTTrainer orchestration
    eval.py               -> objective metrics (JSON validity, field F1, EM)
    infer.py / merge.py   -> inference with an adapter and adapter merging
"""

__version__ = "0.1.0"
