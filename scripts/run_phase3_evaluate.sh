#!/usr/bin/env bash
# ================================================================
# Phase 3 — Evaluate all defended models with the attack classifier
#
# For each Phase 3 run (DP and L2), this exports predictions and
# then runs the attack classifier trained on shadow data against
# the defended target model.
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
ATTACK_CLF="${ROOT_DIR}/runs/attack_classifier/attack_clf.joblib"
RESULTS_DIR="${ROOT_DIR}/runs/phase3_results"
mkdir -p "${RESULTS_DIR}"

# ── helper: export predictions for a defended run ────────────────
export_preds() {
  local RUN_DIR="$1"
  local RANK="${2:-16}"
  echo "  exporting predictions for ${RUN_DIR} (rank ${RANK})..."
  "${PYTHON_BIN}" -m dnabert2_phase1.cli export-predictions \
    --run-dir "${RUN_DIR}" \
    --data-dir "${DATA_DIR}" \
    --rank "${RANK}" \
    --batch-size 4 \
    --device cpu
}

# ── helper: run attack against a defended run ────────────────────
run_attack() {
  local RUN_DIR="$1"
  local TAG="$2"
  local RANK="${3:-16}"
  echo "  running attack on ${TAG}..."
  "${PYTHON_BIN}" src/dnabert2_phase3/evaluate_defense.py \
    --run-dir "${RUN_DIR}" \
    --rank "${RANK}" \
    --attack-clf "${ATTACK_CLF}" \
    --output "${RESULTS_DIR}/${TAG}.json"
}

echo "=== Evaluating DP-SGD defended models ==="
for run_dir in "${ROOT_DIR}"/runs/phase3_dp/dp_*; do
  if [[ -d "${run_dir}" ]]; then
    tag=$(basename "${run_dir}")
    export_preds "${run_dir}"
    run_attack "${run_dir}" "${tag}"
  fi
done

echo ""
echo "=== Evaluating L2 defended models ==="
for run_dir in "${ROOT_DIR}"/runs/phase3_l2/l2_*; do
  if [[ -d "${run_dir}" ]]; then
    tag=$(basename "${run_dir}")
    export_preds "${run_dir}"
    run_attack "${run_dir}" "${tag}"
  fi
done

echo ""
echo "=== Compiling results ==="
"${PYTHON_BIN}" src/dnabert2_phase3/compile_results.py \
  --results-dir "${RESULTS_DIR}" \
  --output "${RESULTS_DIR}/phase3_summary.json"

echo "Done. Results in ${RESULTS_DIR}/phase3_summary.json"
