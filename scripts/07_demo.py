"""Gradio demo: upload a receipt photo -> get strict JSON.

Loads the (4-bit) base model once, optionally with a trained adapter, and
serves a local web UI. Requires the demo extra: ``pip install -e .[demo]``.

Examples:
    # Trained adapter on the tiny smoke model:
    python scripts/07_demo.py --model configs/model/tiny_mm_smoke.yaml `
        --adapter outputs/<run_id>/adapter

    # Baseline 9B (no adapter), constrained decoding toggle available in the UI:
    python scripts/07_demo.py --adapter none
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import lora_lab.data  # noqa: E402, F401  (registers dataset builders)
from lora_lab.config import load_run_config  # noqa: E402
from lora_lab.eval import extract_json  # noqa: E402
from lora_lab.infer import extract, load_for_inference  # noqa: E402
from lora_lab.schema import load_schema  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local Gradio demo for the JSON extractor.")
    p.add_argument("--model", default="configs/model/qwen3_5_9b_mm_qlora.yaml")
    p.add_argument("--data", default="configs/data/cord.yaml")
    p.add_argument("--adapter", default="none", help="Adapter dir or 'none' for baseline.")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="Create a public gradio link.")
    return p.parse_args()


def main() -> int:
    try:
        import gradio as gr
    except ImportError:
        print("gradio is not installed. Run: pip install -e .[demo]")
        return 1

    args = parse_args()
    cfg = load_run_config(args.model, args.data)
    adapter = None if args.adapter.lower() == "none" else args.adapter

    print("[demo] loading model (first call may take a while)...")
    model, processor = load_for_inference(cfg.model, adapter)
    schema = load_schema(cfg.data.json_schema)

    def run_extract(image, constrained: bool):
        if image is None:
            return {}, "Upload an image first."
        raw = extract(
            model,
            processor,
            image,
            cfg.data.instruction,
            image_max_pixels=cfg.data.image_max_pixels,
            schema=schema,
            constrained=constrained,
        )
        parsed = extract_json(raw)
        pretty = parsed if parsed is not None else {}
        status = "valid JSON" if parsed is not None else "NOT valid JSON - showing raw output"
        return pretty, f"{status}\n\n--- raw model output ---\n{raw}"

    def random_synthetic():
        from random import randrange

        from lora_lab.synthetic import generate_invoice

        sample = generate_invoice(randrange(1_000_000))
        return sample["image"], json.dumps(json.loads(sample["target_json"]), indent=2)

    title = f"LoRA Lab - {cfg.model.name_or_path}" + (" + adapter" if adapter else " (baseline)")
    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}\nReceipt image -> strict JSON. Dataset: `{cfg.data.type}`.")
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Receipt image")
                constrained = gr.Checkbox(
                    label="Grammar-constrained decoding (needs .[constrained])", value=False
                )
                with gr.Row():
                    btn = gr.Button("Extract JSON", variant="primary")
                    rnd = gr.Button("Random synthetic receipt")
                gold = gr.Textbox(label="Gold JSON (synthetic only)", lines=6)
            with gr.Column():
                out_json = gr.JSON(label="Parsed JSON")
                out_raw = gr.Textbox(label="Status / raw output", lines=12)
        btn.click(run_extract, inputs=[image, constrained], outputs=[out_json, out_raw])
        rnd.click(random_synthetic, inputs=None, outputs=[image, gold])

    demo.launch(server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
