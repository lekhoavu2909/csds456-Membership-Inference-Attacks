from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef


@dataclass(slots=True)
class ClassificationMetrics:
    accuracy: float
    mcc: float
    loss: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {k: v for k, v in payload.items() if v is not None}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))


def _ensure_array(predictions: np.ndarray | list | tuple) -> np.ndarray:
    if isinstance(predictions, tuple):
        return _ensure_array(predictions[0])
    if isinstance(predictions, list):
        if not predictions:
            return np.asarray(predictions)
        return _ensure_array(predictions[0])
    return np.asarray(predictions)


def compute_classification_metrics(predictions: np.ndarray | list | tuple, labels: np.ndarray, loss: float | None = None) -> ClassificationMetrics:
    preds = _ensure_array(predictions)
    labels = np.asarray(labels)
    if preds.ndim == 3:
        # Some models emit per-token logits; use CLS (position 0) for sequence classification
        preds = preds[:, 0, :]
    if preds.ndim == 2:
        preds = preds.argmax(axis=-1)
    return ClassificationMetrics(
        accuracy=float(accuracy_score(labels, preds)),
        mcc=float(matthews_corrcoef(labels, preds)),
        loss=loss,
    )


def trainer_compute_metrics(eval_pred) -> dict[str, float]:
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    metrics = compute_classification_metrics(logits, labels)
    return {
        "accuracy": metrics.accuracy,
        "mcc": metrics.mcc,
    }
