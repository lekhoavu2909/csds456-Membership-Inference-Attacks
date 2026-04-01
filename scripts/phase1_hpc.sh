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
ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

exec "${ROOT_DIR}/scripts/train_phase1.sh" "$@"
