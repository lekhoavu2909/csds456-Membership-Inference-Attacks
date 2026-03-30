from __future__ import annotations

import pandas as pd

from dnabert2_phase1.data import build_split_manifest, normalize_sequence, validate_disjoint_splits


def test_normalize_sequence():
    assert normalize_sequence(" acgu n ") == "ACGTN"


def test_validate_disjoint_splits():
    frames = {
        "train": pd.DataFrame({"sequence": ["AAAA"], "label": [0]}),
        "dev": pd.DataFrame({"sequence": ["CCCC"], "label": [1]}),
        "test": pd.DataFrame({"sequence": ["GGGG"], "label": [0]}),
    }
    validate_disjoint_splits(frames)
    manifest = build_split_manifest(frames)
    assert set(manifest["split"]) == {"train", "dev", "test"}

