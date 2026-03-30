from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
from typing import Iterable
from collections import Counter

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset


DNA_PATTERN = re.compile(r"^[ACGTN]+$")


@dataclass(slots=True)
class SplitSummary:
    rows: int
    positives: int
    negatives: int
    min_length: int
    max_length: int
    mean_length: float


@dataclass(slots=True)
class SplitPreparationReport:
    total_rows: int
    unique_sequences: int
    resolved_conflicts: int
    per_split: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, object]:
        return {
            "total_rows": self.total_rows,
            "unique_sequences": self.unique_sequences,
            "resolved_conflicts": self.resolved_conflicts,
            "per_split": self.per_split,
        }


def normalize_sequence(value: object) -> str:
    sequence = str(value).strip().upper().replace("U", "T")
    sequence = re.sub(r"\s+", "", sequence)
    if not sequence:
        raise ValueError("sequence is empty")
    return sequence


def read_split_csv(
    path: str | Path,
    *,
    sequence_column: str = "sequence",
    label_column: str = "label",
    expected_length: int | None = None,
) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"missing split file: {path}")

    frame = pd.read_csv(path)
    required = {sequence_column, label_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    frame = frame.loc[:, [sequence_column, label_column]].copy()
    frame.rename(columns={sequence_column: "sequence", label_column: "label"}, inplace=True)
    frame["sequence"] = frame["sequence"].map(normalize_sequence)
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    if frame["label"].isnull().any():
        raise ValueError(f"{path} contains null labels")

    lengths = frame["sequence"].str.len()
    if expected_length is not None and not (lengths == expected_length).all():
        bad = frame.loc[lengths != expected_length, "sequence"].head(5).tolist()
        raise ValueError(
            f"{path} has sequences that do not match expected length {expected_length}: {bad}"
        )

    invalid = frame.loc[~frame["sequence"].str.fullmatch(DNA_PATTERN.pattern), "sequence"].head(5).tolist()
    if invalid:
        raise ValueError(f"{path} contains invalid DNA characters: {invalid}")

    return frame


def split_summary(frame: pd.DataFrame) -> SplitSummary:
    lengths = frame["sequence"].str.len()
    positives = int((frame["label"] == 1).sum())
    negatives = int((frame["label"] == 0).sum())
    return SplitSummary(
        rows=int(len(frame)),
        positives=positives,
        negatives=negatives,
        min_length=int(lengths.min()),
        max_length=int(lengths.max()),
        mean_length=float(lengths.mean()),
    )


def canonicalize_split_frames(frames: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], SplitPreparationReport]:
    split_order = list(frames)
    sequence_records: dict[str, dict[str, object]] = {}
    per_split_stats: dict[str, dict[str, int]] = {
        split: {"input_rows": len(frame), "unique_sequences": 0, "conflicting_labels": 0, "assigned_sequences": 0}
        for split, frame in frames.items()
    }
    resolved_conflicts = 0

    for split_name in split_order:
        frame = frames[split_name]
        per_split_stats[split_name]["unique_sequences"] = int(frame["sequence"].nunique())
        grouped = frame.groupby("sequence", sort=False)["label"].agg(list)
        per_split_stats[split_name]["conflicting_labels"] = int(sum(len(set(labels)) > 1 for labels in grouped))
        for sequence, labels in grouped.items():
            sequence = str(sequence)
            label_list = [int(label) for label in labels]
            if sequence not in sequence_records:
                sequence_records[sequence] = {
                    "first_split": split_name,
                    "label_counts": Counter(label_list),
                    "first_label": label_list[0],
                    "had_conflict": len(set(label_list)) > 1,
                }
            else:
                record = sequence_records[sequence]
                record["label_counts"].update(label_list)  # type: ignore[union-attr]
                record["had_conflict"] = bool(record["had_conflict"]) or len(set(label_list)) > 1

    resolved_rows: dict[str, list[dict[str, object]]] = {split: [] for split in split_order}
    for sequence, meta in sequence_records.items():
        counts: Counter[int] = meta["label_counts"]  # type: ignore[assignment]
        if bool(meta["had_conflict"]):
            resolved_conflicts += 1
        if counts[1] > counts[0]:
            label = 1
        elif counts[0] > counts[1]:
            label = 0
        else:
            label = int(meta["first_label"])  # type: ignore[arg-type]
        owner = str(meta["first_split"])
        resolved_rows[owner].append({"sequence": sequence, "label": label})

    cleaned_frames: dict[str, pd.DataFrame] = {}
    for split_name, rows in resolved_rows.items():
        frame = pd.DataFrame(rows, columns=["sequence", "label"])
        cleaned_frames[split_name] = frame.reset_index(drop=True)
        per_split_stats[split_name]["assigned_sequences"] = int(len(frame))

    report = SplitPreparationReport(
        total_rows=int(sum(len(frame) for frame in frames.values())),
        unique_sequences=int(len(sequence_records)),
        resolved_conflicts=int(resolved_conflicts),
        per_split=per_split_stats,
    )
    return cleaned_frames, report


def validate_disjoint_splits(frames: dict[str, pd.DataFrame]) -> None:
    seen: dict[str, set[str]] = {}
    for split_name, frame in frames.items():
        items = {str(seq) for seq in frame["sequence"]}
        seen[split_name] = items

    split_names = list(seen)
    for i, left in enumerate(split_names):
        for right in split_names[i + 1 :]:
            overlap = seen[left] & seen[right]
            if overlap:
                example = sorted(overlap)[0]
                raise ValueError(f"splits {left} and {right} overlap on sequence {example}")


def build_split_manifest(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for split_name, frame in frames.items():
        for row_id, row in frame.reset_index(drop=True).iterrows():
            sequence = str(row["sequence"])
            label = int(row["label"])
            records.append(
                {
                    "split": split_name,
                    "row_id": row_id,
                    "sequence": sequence,
                    "label": label,
                    "length": len(sequence),
                    "sha256": sha256(f"{sequence}::{label}".encode("utf-8")).hexdigest(),
                }
            )
    return pd.DataFrame.from_records(records)


def frames_to_datasets(frames: dict[str, pd.DataFrame]) -> DatasetDict:
    return DatasetDict({name: Dataset.from_pandas(frame.reset_index(drop=True)) for name, frame in frames.items()})


def load_local_splits(
    data_dir: str | Path,
    *,
    train_name: str = "train.csv",
    dev_name: str = "dev.csv",
    test_name: str = "test.csv",
    expected_length: int | None = None,
) -> tuple[dict[str, pd.DataFrame], SplitPreparationReport]:
    data_dir = Path(data_dir)
    raw_frames = {
        "train": read_split_csv(data_dir / train_name, expected_length=expected_length),
        "dev": read_split_csv(data_dir / dev_name, expected_length=expected_length),
        "test": read_split_csv(data_dir / test_name, expected_length=expected_length),
    }
    frames, report = canonicalize_split_frames(raw_frames)
    validate_disjoint_splits(frames)
    return frames, report


def _candidate_split_names() -> tuple[str, ...]:
    return ("train", "validation", "dev", "test")


def load_hf_dataset(name: str, config: str | None = None) -> DatasetDict | Dataset:
    return load_dataset(name, config) if config else load_dataset(name)


def export_hf_dataset_to_csv(
    dataset_name: str,
    output_dir: str | Path,
    *,
    dataset_config: str | None = None,
    sequence_column: str = "sequence",
    label_column: str = "label",
) -> dict[str, Path]:
    raw = load_hf_dataset(dataset_name, dataset_config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.DataFrame] = {}
    if isinstance(raw, DatasetDict):
        split_map = {
            "validation": "dev",
            "dev": "dev",
            "train": "train",
            "test": "test",
        }
        for split_name, split in raw.items():
            normalized_name = split_map.get(split_name)
            if normalized_name is None:
                continue
            frame = split.to_pandas()
            if sequence_column not in frame.columns or label_column not in frame.columns:
                raise ValueError(
                    f"split {split_name} is missing required columns {sequence_column!r}/{label_column!r}"
                )
            frames[normalized_name] = frame.loc[:, [sequence_column, label_column]].copy()
    else:
        frame = raw.to_pandas()
        if sequence_column not in frame.columns or label_column not in frame.columns:
            raise ValueError(
                f"dataset is missing required columns {sequence_column!r}/{label_column!r}"
            )
        if len(frame) < 3:
            raise ValueError("dataset is too small to split into train/dev/test")
        # Deterministic fallback split when the dataset is not already split.
        train_end = int(len(frame) * 0.8)
        dev_end = int(len(frame) * 0.9)
        frames = {
            "train": frame.iloc[:train_end].copy(),
            "dev": frame.iloc[train_end:dev_end].copy(),
            "test": frame.iloc[dev_end:].copy(),
        }

    written: dict[str, Path] = {}
    for split_name, frame in frames.items():
        out = output_dir / f"{split_name}.csv"
        frame.to_csv(out, index=False)
        written[split_name] = out
    return written
