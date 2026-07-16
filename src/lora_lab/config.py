"""Declarative, validated configuration schemas (pydantic v2).

A full run is composed of three independent configs so each axis scales on its
own: ``ModelConfig`` (what to adapt + how to quantize), ``DataConfig`` (what
task/dataset), ``TrainConfig`` (optimization). They are merged into a
``RunConfig`` and snapshotted verbatim into ``outputs/<run_id>/`` for
reproducibility.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class QuantConfig(BaseModel):
    """bitsandbytes 4-bit (QLoRA) settings. Set ``load_in_4bit=False`` for a
    plain (non-quantized) LoRA run to compare memory/quality."""

    load_in_4bit: bool = True
    bnb_4bit_quant_type: Literal["nf4", "fp4"] = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: Literal["bfloat16", "float16"] = "bfloat16"


class LoraConfigModel(BaseModel):
    """LoRA hyper-parameters. ``target_modules="auto"`` triggers runtime
    discovery of the linear layers inside the language model (see
    ``modules_discovery``).

    The variant flags below are the main "understand adaptation deeply" levers:
      - ``use_dora``  : DoRA (weight-decomposed LoRA) - separates magnitude/direction.
      - ``use_rslora``: rank-stabilized scaling (alpha / sqrt(r) instead of alpha / r).
      - ``init_lora_weights``: adapter init strategy - True | "gaussian" | "pissa" |
        "pissa_niter_[N]" | "olora" | "loftq". PiSSA/OLoRA/LoftQ give better starts.
      - ``lora_plus_lr_ratio``: if set, use LoRA+ (higher LR for the B matrices).
    """

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: Literal["none", "all", "lora_only"] = "none"
    target_modules: list[str] | Literal["auto"] = "auto"
    modules_to_save: list[str] = Field(default_factory=list)
    task_type: str = "CAUSAL_LM"
    use_dora: bool = False
    use_rslora: bool = False
    init_lora_weights: bool | str = True
    lora_plus_lr_ratio: float | None = None


class ModelConfig(BaseModel):
    name_or_path: str
    modality: Literal["text", "multimodal"] = "multimodal"
    attn_implementation: Literal["sdpa", "eager", "flash_attention_2"] = "sdpa"
    torch_dtype: Literal["bfloat16", "float16"] = "bfloat16"
    freeze_vision: bool = True
    gradient_checkpointing: bool = True
    trust_remote_code: bool = False
    quant: QuantConfig = Field(default_factory=QuantConfig)
    lora: LoraConfigModel = Field(default_factory=LoraConfigModel)


class DataConfig(BaseModel):
    """A dataset registered under ``type``. ``schema_fields`` lists the target
    JSON keys used both to build the prompt and to score field-level F1."""

    type: str
    dataset_id: str | None = None  # HF hub id or local path, dataset-specific
    schema_fields: list[str] = Field(default_factory=list)
    # A real JSON Schema (inline dict) or a path to a .json file. Used for strict
    # output validation (eval) and, optionally, grammar-constrained decoding.
    json_schema: dict[str, Any] | str | None = None
    instruction: str = (
        "Extract the fields as strict, minified JSON. Output only JSON, no prose."
    )
    max_seq_len: int = 2048
    image_max_pixels: int | None = 1_048_576  # cap vision tokens / VRAM
    train_split: str = "train"
    eval_split: str = "validation"
    max_train_samples: int | None = None
    max_eval_samples: int | None = None


class TrainConfig(BaseModel):
    output_dir: str = "outputs"
    run_name: str | None = None
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    num_train_epochs: float = 3.0
    max_steps: int = -1
    optim: str = "paged_adamw_8bit"
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 0.3
    bf16: bool = True
    fp16: bool = False
    logging_steps: int = 10
    # Loss-based eval during training. "no" disables; "steps" runs every
    # ``eval_steps`` and enables ``load_best_model_at_end``.
    eval_strategy: Literal["no", "steps", "epoch"] = "no"
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 2
    load_best_model_at_end: bool = False
    metric_for_best_model: str = "loss"  # i.e. eval_loss
    # Generation-based metrics DURING training (json_valid_rate / field_f1 on a
    # small eval slice). 0 disables. Costs one batched generate every N steps,
    # but shows the moment the model "clicks" into the JSON format.
    gen_eval_steps: int = 0
    gen_eval_samples: int = 8
    gen_eval_batch_size: int = 4
    dataloader_num_workers: int = 0  # 0 is safest on native Windows
    report_to: list[str] = Field(default_factory=list)  # e.g. ["tensorboard"]
    seed: int = 42


class RunConfig(BaseModel):
    model: ModelConfig
    data: DataConfig
    train: TrainConfig = Field(default_factory=TrainConfig)
    run_id: str = ""

    @model_validator(mode="after")
    def _assign_run_id(self) -> RunConfig:
        if not self.run_id:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            model_tag = self.model.name_or_path.split("/")[-1]
            self.run_id = f"{stamp}_{self.data.type}_{model_tag}"
        return self

    @property
    def run_dir(self) -> Path:
        return Path(self.train.output_dir) / self.run_id


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_run_config(
    model_cfg: str | Path,
    data_cfg: str | Path,
    train_cfg: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> RunConfig:
    """Compose a ``RunConfig`` from three YAML files (+ optional dotted overrides).

    ``overrides`` keys use dotted paths, e.g. ``{"model.lora.r": 32}``.
    """
    payload: dict[str, Any] = {
        "model": _load_yaml(model_cfg),
        "data": _load_yaml(data_cfg),
        "train": _load_yaml(train_cfg) if train_cfg else {},
    }
    if overrides:
        for dotted, value in overrides.items():
            _set_dotted(payload, dotted, value)
    return RunConfig(**payload)


def _set_dotted(d: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = d
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def load_snapshot(path: str | Path, overrides: dict[str, Any] | None = None) -> RunConfig:
    """Rebuild a ``RunConfig`` from a run's ``config.snapshot.yaml``.

    ``path`` may be the snapshot file or the run directory containing it. The
    snapshot stores ``run_id``, so the rebuilt config points at the SAME run
    directory - this is what makes ``--resume`` find the checkpoints.
    """
    p = Path(path)
    if p.is_dir():
        p = p / "config.snapshot.yaml"
    payload = _load_yaml(p)
    if overrides:
        for dotted, value in overrides.items():
            _set_dotted(payload, dotted, value)
    return RunConfig(**payload)


def snapshot_config(cfg: RunConfig, run_dir: Path) -> Path:
    """Write the resolved config to ``run_dir/config.snapshot.yaml``."""
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "config.snapshot.yaml"
    with open(out, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.model_dump(mode="json"), fh, allow_unicode=True, sort_keys=False)
    return out
