# LoRA Lab: QLoRA-адаптация Qwen3.5-9B (image -> JSON)

Config-driven пет-проект для глубокого изучения **LoRA/QLoRA**. Задача — мультимодальный
*Structured Extractor*: картинка документа (чек) -> строгий JSON. Обучение через
**QLoRA** (4-bit NF4, заморозка vision-энкодера, LoRA на языковой части) на одной
**RTX 5090 (32 GB)** под нативным Windows.

Одна кодовая база гоняет и крошечную модель (`--smoke`), и `Qwen/Qwen3.5-9B` —
отличается только YAML-конфиг.

## Структура

```
configs/            # декларативные конфиги (model / data / train)
src/lora_lab/       # ядро: config, data, model, modules_discovery, train, eval, infer, merge
scripts/            # 00_env_check, 01_prepare_data, 02_train, 03_evaluate
notebooks/          # explore_lora_internals.ipynb — заглянуть внутрь адаптации
outputs/            # артефакты запусков (в .gitignore)
```

Ключевые слои и их контракты:

| Модуль | Ответственность |
|---|---|
| `config.py` | pydantic-схемы `RunConfig = model + data + train`, композиция YAML, снапшот |
| `registry.py` | `name -> builder` для датасетов/метрик |
| `data.py` | chat template, маска `-100`, мультимодальный collator |
| `modules_discovery.py` | авто-поиск `target_modules` внутри LM (MoE/attention) |
| `model.py` | `BitsAndBytesConfig` (NF4) + PEFT + заморозка vision |
| `train.py` | оркестрация TRL `SFTTrainer` + VRAM-коллбэк |
| `eval.py` | JSON-valid %, field-level F1, exact-match |
| `infer.py` / `merge.py` | инференс с адаптером / слияние адаптера |

## Установка (RTX 5090 / Blackwell / sm_120)

Порядок важен: сначала Blackwell-совместимый PyTorch (cu128 nightly), затем проект.

> **Версия Python: используй 3.12** (или 3.11). Колёса `torch` cu128 nightly и
> `bitsandbytes` отстают от новых интерпретаторов на месяцы — на Python 3.14 ты
> почти наверняка получишь `No matching distribution found for torch`. Ставь
> [Python 3.12](https://www.python.org/downloads/) и создавай venv именно им:

```powershell
# Явно берём Python 3.12 через py-launcher (Windows)
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version   # должно быть 3.12.x

# 1) PyTorch с поддержкой Blackwell
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# 2) Проект (тянет transformers@main, peft, trl, bitsandbytes, ...)
pip install -e .
```

> На нативном Windows `bitsandbytes` требует свежего колеса с поддержкой CUDA 12.x,
> а `flash_attention_2` почти не собирается — поэтому в конфигах по умолчанию
> `attn_implementation: sdpa`. Если ловишь странности — рассмотри WSL2.

## Использование

```powershell
# Фаза 0: проверка окружения (fail-fast). Без прохождения дальше нет смысла.
python scripts/00_env_check.py

# Посмотреть, как выглядят сэмплы и целевой JSON (без GPU)
python scripts/01_prepare_data.py --data configs/data/cord.yaml --n 3

# Быстрый smoke-прогон всего пайплайна на крошечной модели (несколько шагов)
python scripts/02_train.py --smoke

# То же, но на синтетических чеках (генерируются локально, без скачиваний)
python scripts/02_train.py --smoke --data configs/data/synthetic_invoices.yaml

# Полное обучение на Qwen3.5-9B
python scripts/02_train.py `
  --model configs/model/qwen3_5_9b_mm_qlora.yaml `
  --data  configs/data/cord.yaml `
  --train configs/train/sft_qlora.yaml

# Продолжить прерванный прогон с последнего чекпоинта
python scripts/02_train.py --resume outputs/<run_id>

# Оценка: baseline (без адаптера) vs адаптер на одном тест-сете
python scripts/03_evaluate.py --adapter outputs/<run_id>/adapter --limit 100
```

Переопределение любых полей конфига на лету:

```powershell
python scripts/02_train.py --set model.lora.r=32 model.lora.lora_alpha=64 train.max_steps=200
```

## Продвинутые возможности

### Варианты адаптации (для сравнения на одной метрике)

Тумблеры в `configs/model/*.yaml` (секция `lora`) или через `--set`:

```powershell
python scripts/02_train.py --set model.lora.use_dora=true          # DoRA
python scripts/02_train.py --set model.lora.use_rslora=true         # rank-stabilized LoRA
python scripts/02_train.py --set model.lora.init_lora_weights=pissa # PiSSA-инициализация
python scripts/02_train.py --set model.lora.lora_plus_lr_ratio=16   # LoRA+ (LR B > LR A)
```

### JSON Schema + строгая валидация + constrained decoding

`configs/data/cord.yaml` ссылается на настоящую JSON Schema (`configs/data/cord_schema.json`).
Она используется для метрики `schema_valid_rate` и (опционально) для grammar-constrained
декодинга, гарантирующего валидный JSON:

```powershell
pip install -e .[constrained]     # ставит outlines
python scripts/03_evaluate.py --adapter outputs/<run_id>/adapter --constrained --out outputs/eval_run
```

### Метрики во время обучения

Помимо loss-eval (`eval_strategy: steps` + `load_best_model_at_end`), каждые
`gen_eval_steps` шагов коллбэк генерирует на маленьком срезе eval-сета и логирует
`gen_json_valid_rate` / `gen_field_f1` — видно момент, когда модель «щёлкает»
формат JSON. Логи идут в TensorBoard (`report_to: ["tensorboard"]`):

```powershell
tensorboard --logdir outputs
```

### Богатый eval

`scripts/03_evaluate.py` считает `json_valid_rate`, `schema_valid_rate`, `field_f1`,
`exact_match` и **per-field** precision/recall/F1, батчит генерацию (`--batch-size`),
нормализует числа при сравнении и пишет `predictions_*.jsonl` + `metrics.json` (`--out`).

### DPO: preference-обучение поверх SFT

Вторая стадия: пары «правильный JSON vs правдоподобно-неправильный» и
`DPOTrainer` (TRL) поверх SFT-адаптера. Reference-модель не нужна — с PEFT
адаптер просто временно отключается (экономия VRAM на одном GPU).

```powershell
# 1) Собрать пары. Оффлайн-режим: rejected = испорченный gold
#    (сбитые цифры, потерянный/переименованный ключ, проза вокруг JSON, обрыв)
python scripts/05_make_pairs.py --data configs/data/cord.yaml --n 256 --out outputs/pairs/cord.jsonl

#    Либо из реальных ошибок модели (после scripts/03_evaluate.py --out ...):
python scripts/05_make_pairs.py --from-predictions outputs/eval/predictions_baseline.jsonl `
  --split validation --out outputs/pairs/cord_hard.jsonl

# 2) DPO поверх SFT-адаптера
python scripts/06_dpo.py `
  --model configs/model/qwen3_5_9b_mm_qlora.yaml `
  --data  configs/data/cord.yaml `
  --pairs outputs/pairs/cord.jsonl `
  --from-adapter outputs/<sft_run_id>/adapter

# Смоук всей DPO-ветки: tiny-модель + синтетические чеки, полностью оффлайн
python scripts/06_dpo.py --smoke
```

Гиперпараметры — `configs/train/dpo.yaml` (LR в ~40 раз ниже SFT, `dpo_beta`).
В логах смотри `rewards/margins` и `rewards/accuracies` — они должны расти.

### Sweeps по гиперпараметрам и аблэйшены методов

```powershell
python scripts/04_sweep.py --sweep configs/sweep/lora_rank.yaml       # сетка r x alpha
python scripts/04_sweep.py --sweep configs/sweep/methods.yaml        # LoRA vs DoRA vs rsLoRA vs PiSSA vs LoRA+
python scripts/04_sweep.py --sweep configs/sweep/quant.yaml          # QLoRA (NF4) vs bf16 LoRA
python scripts/04_sweep.py --sweep configs/sweep/target_modules.yaml # attention-only vs MLP-only vs все
```

Два вида осей: `grid` (декартово произведение числовых параметров) и `variants`
(взаимоисключающие тумблеры методов). Каждая комбинация обучается+оценивается,
таблица складывается в `outputs/sweeps/<name>/results.csv`.

Аблэйшен `target_modules` работает через `model.lora.target_groups: all|attention|mlp`
(фильтр в `modules_discovery.py`) — доступен и в обычном обучении через `--set`.

### Демо и экспорт

```powershell
# Локальное Gradio-демо: загрузил фото чека -> получил JSON
pip install -e .[demo]
python scripts/07_demo.py --model configs/model/tiny_mm_smoke.yaml --adapter outputs/<run_id>/adapter

# Запушить адаптер на HF Hub (лёгкий, десятки МБ) или merged-модель
python scripts/08_export.py --adapter outputs/<run_id>/adapter --push <user>/qwen3.5-cord-lora --private
python scripts/08_export.py --adapter outputs/<run_id>/adapter --merge --base Qwen/Qwen3.5-9B --out outputs/merged
```

В демо есть кнопка «Random synthetic receipt» — можно проверять модель без
своих картинок. CI (GitHub Actions) гоняет ruff + pytest на каждый push.

### Тесты

```powershell
pip install -e .[dev]
python -m pytest tests -q
```

Проверяют критичное: маскирование лейблов в коллаторе (только ответ учится в loss),
метрики/нормализацию, JSON Schema и композицию конфигов. Гоняются на CPU за секунды.

## Оси масштабирования

- **Модель**: `name_or_path` + `modality` (tiny VLM <-> Qwen3.5-9B <-> любая VLM).
- **Датасет**: новый билдер в `data.py` + `@register_dataset(...)` + JSON-схема в `configs/data/`.
  Рабочий пример второго домена — `synthetic_invoices` (`src/lora_lab/synthetic.py`):
  чеки рендерятся локально через PIL, оффлайн и полностью тестируемо на CPU.
- **Метод**: сегодня SFT (LoRA/QLoRA); DPO/ORPO (TRL) — как альтернативный оркестратор.
- **Квантизация**: `nf4 / fp4 / без` через `quant.load_in_4bit`.
- **Железо**: один GPU -> `accelerate`/DeepSpeed для мультиGPU без правок кода.

## Что изучить руками

Открой `notebooks/explore_lora_internals.ipynb`: разложение `dW = (alpha/r) * B @ A`,
доля обучаемых параметров (<1%), куда именно вешаются адаптеры в MoE, влияние `r`/`alpha`,
а после обучения — SVD-анализ реального адаптера: какие слои изменились сильнее всего
(attention vs MLP/эксперты) и сколько сингулярных компонент реально несут энергию
(хватило бы меньшего `r`?).
