"""Export a trained run: push the adapter to the Hugging Face Hub, or merge
the adapter into the base weights first and push/save the standalone model.

The adapter is ~tens of MB, the merged 9B model is ~18 GB - push the adapter
unless you specifically need a standalone model.

Examples:
    # Push just the adapter (recommended):
    python scripts/08_export.py --adapter outputs/<run_id>/adapter `
        --push <user>/qwen3.5-9b-cord-lora --private

    # Merge into bf16 base and save locally:
    python scripts/08_export.py --adapter outputs/<run_id>/adapter `
        --merge --base Qwen/Qwen3.5-9B --out outputs/merged

    # Merge and push the standalone model:
    python scripts/08_export.py --adapter outputs/<run_id>/adapter `
        --merge --base Qwen/Qwen3.5-9B --out outputs/merged --push <user>/qwen3.5-9b-cord
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lora_lab.merge import merge_adapter  # noqa: E402


def push_folder(folder: str | Path, repo_id: str, private: bool) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=repo_id)
    url = f"https://huggingface.co/{repo_id}"
    print(f"[export] pushed {folder} -> {url}")
    return url


def main() -> int:
    p = argparse.ArgumentParser(description="Export/push a trained adapter or merged model.")
    p.add_argument("--adapter", required=True, help="Trained adapter directory.")
    p.add_argument("--merge", action="store_true", help="Merge into the bf16 base first.")
    p.add_argument("--base", default="Qwen/Qwen3.5-9B", help="Base model for --merge.")
    p.add_argument("--out", default=None, help="Output dir for the merged model.")
    p.add_argument("--modality", default="multimodal", choices=["multimodal", "text"])
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--push", default=None, metavar="REPO_ID", help="Hub repo to upload to.")
    p.add_argument("--private", action="store_true", help="Create the Hub repo as private.")
    args = p.parse_args()

    folder = Path(args.adapter)
    if args.merge:
        if not args.out:
            p.error("--merge requires --out")
        folder = merge_adapter(
            args.base, args.adapter, args.out, modality=args.modality, dtype=args.dtype
        )

    if args.push:
        push_folder(folder, args.push, private=args.private)
    elif not args.merge:
        p.error("Nothing to do: pass --push and/or --merge")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
