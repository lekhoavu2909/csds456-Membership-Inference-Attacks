#!/usr/bin/env bash
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

"${PYTHON_BIN}" src/dnabert2_phase2/attack_classifier.py \
  --shadow-dirs \
    "${ROOT_DIR}/runs/shadow1" \
    "${ROOT_DIR}/runs/shadow2" \
    "${ROOT_DIR}/runs/shadow3" \
    "${ROOT_DIR}/runs/shadow4" \
  --target-dir "${ROOT_DIR}/runs/target" \
  --evaluate-dir "${ROOT_DIR}/runs/target" \
  --output-path "${ROOT_DIR}/runs/attack_classifier/attack_clf.joblib"