# Full QLoRA run on the 9B target, end to end:
#   1) fail-fast environment check
#   2) SFT training (resume-able: scripts/02_train.py --resume outputs/<run_id>)
#   3) baseline vs adapter evaluation on the same held-out slice
#
# Usage (from the repo root, venv activated):
#   .\scripts\full_run.ps1
#   .\scripts\full_run.ps1 -EvalLimit 200
#   .\scripts\full_run.ps1 -Model configs/model/tiny_mm_smoke.yaml   # dry-run the runbook itself
param(
    [string]$Model = "configs/model/qwen3_5_9b_mm_qlora.yaml",
    [string]$Data = "configs/data/cord.yaml",
    [string]$Train = "configs/train/sft_qlora.yaml",
    [int]$EvalLimit = 100,
    [string]$EvalOut = "outputs/eval_9b"
)
$ErrorActionPreference = "Stop"

Write-Host "== [1/3] Environment check ==" -ForegroundColor Cyan
python scripts/00_env_check.py
if ($LASTEXITCODE -ne 0) { throw "Environment check failed - fix it before burning GPU hours." }

Write-Host "`n== [2/3] QLoRA SFT training ==" -ForegroundColor Cyan
python scripts/02_train.py --model $Model --data $Data --train $Train
if ($LASTEXITCODE -ne 0) {
    throw "Training failed. If it crashed mid-run, continue with: python scripts/02_train.py --resume outputs/<run_id>"
}

# The freshest run directory that actually contains a saved adapter.
$run = Get-ChildItem outputs -Directory |
    Where-Object { Test-Path (Join-Path $_.FullName "adapter") } |
    Sort-Object Name -Descending | Select-Object -First 1
if (-not $run) { throw "No run with a saved adapter found under outputs/." }
$adapter = Join-Path $run.FullName "adapter"
Write-Host "`n== [3/3] Evaluation: baseline vs $adapter ==" -ForegroundColor Cyan

python scripts/03_evaluate.py --model $Model --data $Data --adapter $adapter `
    --limit $EvalLimit --out $EvalOut
if ($LASTEXITCODE -ne 0) { throw "Evaluation failed." }

Write-Host "`nDone. Metrics: $EvalOut/metrics.json - paste them into the README results table." -ForegroundColor Green
