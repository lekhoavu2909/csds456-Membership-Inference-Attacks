#!/usr/bin/env bash
# ================================================================
# Phase 3 — DP-SGD sweep
# Runs DP fine-tuning for every (σ, C) combination on the target
# data, then exports predictions so the attack classifier can be
# re-evaluated.
# ================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${ROOT_DIR}/.cache/huggingface"
export XDG_CACHE_HOME="${ROOT_DIR}/.cache"
export TRANSFORMERS_CACHE="${ROOT_DIR}/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="${ROOT_DIR}/.cache/huggingface/datasets"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}"

PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

DATA_DIR="${1:-${ROOT_DIR}/data/core_promoter_70bp_target}"

# noise multipliers (σ) and clipping norms (C) to sweep
SIGMAS=(0.5 1.0 1.5 2.0)
CLIPS=(0.5 1.0)

for sigma in "${SIGMAS[@]}"; do
  for clip in "${CLIPS[@]}"; do
    TAG="dp_s${sigma}_c${clip}"
    OUTPUT_DIR="${ROOT_DIR}/runs/phase3_dp/${TAG}"
    echo ""
    echo "========================================"
    echo "  DP-SGD: σ=${sigma}  C=${clip}"
    echo "  output: ${OUTPUT_DIR}"
    echo "========================================"
    "${PYTHON_BIN}" -m dnabert2_phase3.train_dp \
      --data-dir "${DATA_DIR}" \
      --output-dir "${OUTPUT_DIR}" \
      --noise-multiplier "${sigma}" \
      --max-grad-norm "${clip}" \
      --lora-rank 16 \
      --epochs 10 \
      --batch-size 8 \
      --learning-rate 2e-5 \
      --seed 42
  done
done

echo ""
echo "=== DP-SGD sweep complete ==="
