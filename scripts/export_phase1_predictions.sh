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

RUN_DIR="${1:-${ROOT_DIR}/runs/phase1_local}"
DATA_DIR="${2:-${ROOT_DIR}/data/core_promoter_70bp}"
RANK="${3:-8}"
BATCH_SIZE="${EXPORT_BATCH_SIZE:-8}"
DEVICE="${EXPORT_DEVICE:-cpu}"

"${PYTHON_BIN}" - <<PY
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, set_seed

from dnabert2_phase1.config import Phase1Config
from dnabert2_phase1.data import load_local_splits
from dnabert2_phase1.modeling import load_sequence_classifier, load_tokenizer
from dnabert2_phase1.runner import tokenize_frame, _save_json

run_dir = Path(r"${RUN_DIR}")
data_dir = Path(r"${DATA_DIR}")
rank = int("${RANK}")
batch_size = int("${BATCH_SIZE}")
device = torch.device("${DEVICE}")

model_dir = run_dir / f"lora_r{rank}" / "best_model"
if not (model_dir / "adapter_config.json").exists():
    raise FileNotFoundError(f"missing best_model for rank {rank}: {model_dir}")

frames, _ = load_local_splits(data_dir, expected_length=70)
config = Phase1Config(data_dir=str(data_dir), output_dir=str(run_dir))

set_seed(config.seed)
tokenizer = load_tokenizer(str(model_dir))
model = load_sequence_classifier(
    str(model_dir),
    num_labels=2,
    lora_r=rank,
    lora_alpha=2 * rank,
    lora_dropout=0.05,
)
model.to(device)
model.eval()

collator = DataCollatorWithPadding(tokenizer=tokenizer)


def predict_frame(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    tokenized = tokenize_frame(frame, tokenizer, config.max_length)
    loader = DataLoader(tokenized, batch_size=batch_size, collate_fn=collator)

    rows = []
    offset = 0
    for batch in loader:
        labels = batch.pop("labels").cpu().numpy()
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        predicted = probs.argmax(axis=-1)
        for i in range(len(labels)):
            rows.append(
                {
                    "split": split_name,
                    "row_index": offset + i,
                    "sequence": frame.iloc[offset + i]["sequence"],
                    "label": int(labels[i]),
                    "predicted_label": int(predicted[i]),
                    "prob_class_0": float(probs[i, 0]),
                    "prob_class_1": float(probs[i, 1]),
                    "max_probability": float(probs[i].max()),
                    "entropy": float(-(probs[i] * np.log(np.clip(probs[i], 1e-12, 1.0))).sum()),
                }
            )
        offset += len(labels)
    return pd.DataFrame(rows)

out_dir = run_dir / f"lora_r{rank}" / "prediction_exports"
out_dir.mkdir(parents=True, exist_ok=True)

train_pred = predict_frame(frames["train"], "train")
dev_pred = predict_frame(frames["dev"], "dev")
test_pred = predict_frame(frames["test"], "test")

train_pred.to_csv(out_dir / "train_predictions.csv", index=False)
dev_pred.to_csv(out_dir / "dev_predictions.csv", index=False)
test_pred.to_csv(out_dir / "test_predictions.csv", index=False)
_save_json(
    {
        "run_dir": str(run_dir),
        "data_dir": str(data_dir),
        "rank": rank,
        "batch_size": batch_size,
        "device": str(device),
        "exports": ["train_predictions.csv", "dev_predictions.csv", "test_predictions.csv"],
    },
    out_dir / "export_summary.json",
)
print(f"wrote predictions to {out_dir}")
PY
