#!/usr/bin/env bash
# ================================================================
# Phase 3 — L2 Regularization (weight decay) sweep
# Reuses the Phase 1 training pipeline with different --weight-decay
# values.  Only trains at LoRA rank 16 (our standard comparison).
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

# weight decay values (λ) to sweep
LAMBDAS=(0.001 0.01 0.05 0.1)

for lambda_val in "${LAMBDAS[@]}"; do
  TAG="l2_wd${lambda_val}"
  OUTPUT_DIR="${ROOT_DIR}/runs/phase3_l2/${TAG}"
  echo ""
  echo "========================================"
  echo "  L2 weight decay: λ=${lambda_val}"
  echo "  output: ${OUTPUT_DIR}"
  echo "========================================"
  "${PYTHON_BIN}" -m dnabert2_phase1.cli train \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --lora-ranks 16 \
    --weight-decay "${lambda_val}" \
    --epochs 10 \
    --seed 42
done

echo ""
echo "=== L2 sweep complete ==="
