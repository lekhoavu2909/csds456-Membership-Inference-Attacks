#!/usr/bin/env bash
#SBATCH --job-name=dnabert2_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data/core_promoter_70bp}"
RUNS_DIR="${RUNS_DIR:-${ROOT_DIR}/runs}"
JOB_RUN_DIR="${JOB_RUN_DIR:-${RUNS_DIR}/phase1_${SLURM_JOB_ID:-local}}"

mkdir -p "${RUNS_DIR}/slurm" "${JOB_RUN_DIR}"

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

echo "Running Phase 1 from: ${ROOT_DIR}"
echo "Data dir: ${DATA_DIR}"
echo "Run dir: ${JOB_RUN_DIR}"
echo "Python: ${PYTHON_BIN}"

"${PYTHON_BIN}" -m dnabert2_phase1.cli train   --data-dir "${DATA_DIR}"   --output-dir "${JOB_RUN_DIR}"   --lora-ranks 8 16 32

cat > "${JOB_RUN_DIR}/job_info.txt" <<EOF
job_id=${SLURM_JOB_ID:-local}
job_name=${SLURM_JOB_NAME:-dnabert2_p1}
data_dir=${DATA_DIR}
output_dir=${JOB_RUN_DIR}
repo_root=${ROOT_DIR}
EOF
