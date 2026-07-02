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

# Полное обучение на Qwen3.5-9B
python scripts/02_train.py `
  --model configs/model/qwen3_5_9b_mm_qlora.yaml `
  --data  configs/data/cord.yaml `
  --train configs/train/sft_qlora.yaml

# Оценка: baseline (без адаптера) vs адаптер на одном тест-сете
python scripts/03_evaluate.py --adapter outputs/<run_id>/adapter --limit 100
```

Переопределение любых полей конфига на лету:

```powershell
python scripts/02_train.py --set model.lora.r=32 model.lora.lora_alpha=64 train.max_steps=200
```

## Оси масштабирования

- **Модель**: `name_or_path` + `modality` (tiny VLM <-> Qwen3.5-9B <-> любая VLM).
- **Датасет**: новый билдер в `data.py` + `@register_dataset(...)` + JSON-схема в `configs/data/`.
- **Метод**: сегодня SFT (LoRA/QLoRA); DPO/ORPO (TRL) — как альтернативный оркестратор.
- **Квантизация**: `nf4 / fp4 / без` через `quant.load_in_4bit`.
- **Железо**: один GPU -> `accelerate`/DeepSpeed для мультиGPU без правок кода.

## Что изучить руками

Открой `notebooks/explore_lora_internals.ipynb`: разложение `dW = (alpha/r) * B @ A`,
доля обучаемых параметров (<1%), куда именно вешаются адаптеры в MoE, влияние `r`/`alpha`.
