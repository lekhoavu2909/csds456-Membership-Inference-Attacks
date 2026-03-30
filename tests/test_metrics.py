from __future__ import annotations

import numpy as np

from dnabert2_phase1.metrics import compute_classification_metrics


def test_metrics_binary():
    logits = np.array([[0.1, 0.9], [0.8, 0.2]])
    labels = np.array([1, 0])
    metrics = compute_classification_metrics(logits, labels)
    assert metrics.accuracy == 1.0
    assert metrics.mcc == 1.0

