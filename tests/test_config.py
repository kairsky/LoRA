from pathlib import Path

from lora_lab.config import load_run_config, load_snapshot, snapshot_config

ROOT = Path(__file__).resolve().parents[1]


def _paths():
    return (
        ROOT / "configs/model/qwen3_5_9b_mm_qlora.yaml",
        ROOT / "configs/data/cord.yaml",
        ROOT / "configs/train/sft_qlora.yaml",
    )


def test_compose_defaults():
    m, d, t = _paths()
    cfg = cfg = load_run_config(m, d, t)
    assert cfg.model.name_or_path == "Qwen/Qwen3.5-9B"
    assert cfg.data.type == "cord"
    assert cfg.run_id  # auto-assigned
    assert str(cfg.run_dir).endswith(cfg.run_id)


def test_dotted_overrides_and_types():
    m, d, t = _paths()
    cfg = load_run_config(
        m,
        d,
        t,
        overrides={
            "model.lora.r": 64,
            "model.lora.use_dora": True,
            "train.max_steps": 5,
        },
    )
    assert cfg.model.lora.r == 64
    assert cfg.model.lora.use_dora is True
    assert cfg.train.max_steps == 5


def test_json_schema_field_present():
    m, d, t = _paths()
    cfg = load_run_config(m, d, t)
    # cord.yaml points json_schema at the schema file.
    assert isinstance(cfg.data.json_schema, str)
    assert cfg.data.json_schema.endswith("cord_schema.json")


def test_eval_and_gen_eval_fields():
    m, d, t = _paths()
    cfg = load_run_config(m, d, t)
    assert cfg.train.eval_strategy == "steps"
    assert cfg.train.load_best_model_at_end is True
    assert cfg.train.gen_eval_steps == 100
    # save/eval alignment required by load_best_model_at_end.
    assert cfg.train.save_steps % cfg.train.eval_steps == 0


def test_snapshot_roundtrip_preserves_run_id(tmp_path):
    m, d, t = _paths()
    cfg = load_run_config(m, d, t, overrides={"model.lora.r": 24})
    snapshot_config(cfg, tmp_path)

    restored = load_snapshot(tmp_path)  # accepts the run dir itself
    assert restored.run_id == cfg.run_id  # same run dir -> resume finds checkpoints
    assert restored.model.lora.r == 24

    extended = load_snapshot(tmp_path, overrides={"train.max_steps": 999})
    assert extended.train.max_steps == 999
    assert extended.run_id == cfg.run_id
