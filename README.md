# DNABERT-2 Phase 1 Baseline

This repo implements Phase 1 of the project report:

- DNABERT-2 fine-tuning
- LoRA ranks `8`, `16`, `32`
- 70 bp Core Promoter Detection baseline
- metrics: train/test accuracy, MCC, and train-test gap

## What this repo produces

- Reproducible training runs
- Saved checkpoints and configs
- Split manifest for downstream phases
- Train/test prediction exports for membership inference work

## Official model setup

Base model:

- `zhihan1996/DNABERT-2-117M`

The official DNABERT-2 repository documents fine-tuning with CSV files containing `sequence,label` columns. This repo follows that contract and adds a cleaner project structure around it.

## Data formats

### Local CSV mode

Provide a directory with:

- `train.csv`
- `dev.csv`
- `test.csv`

Each file must contain:

- `sequence`
- `label`

### Hugging Face mode

You can also export data from a Hugging Face dataset using the included adapter script. The repo keeps this optional because the report’s Phase 1 baseline is defined by the local 70 bp promoter task.

## Phase 1 training

Example:

```bash
python -m dnabert2_phase1.cli train \
  --data-dir data/core_promoter_70bp \
  --output-dir runs/phase1_baseline \
  --lora-ranks 8 16 32
```

This will:

- train one run per LoRA rank
- save metrics and predictions under each run directory
- keep the best checkpoint according to validation loss

## Hugging Face export

Example:

```bash
python -m dnabert2_phase1.cli prepare-hf \
  --dataset-name leanmmlindsey/GUE \
  --dataset-config EPI_GM12878 \
  --output-dir data/epi_gm12878
```

## Outputs

Each run writes:

- `config.json`
- `split_manifest.csv`
- `metrics/*.json`
- `predictions/*.csv`
- `checkpoints/`
- `best_model/`

