from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import joblib

FEATURE_COLS = ["prob_class_0", "prob_class_1", "entropy"]


def load_shadow_predictions(shadow_dirs: list[Path], ranks: list[int] = [8, 16, 32]) -> pd.DataFrame:
    frames = []
    for shadow_dir in shadow_dirs:
        for rank in ranks:
            pred_dir = shadow_dir / f"lora_r{rank}" / "predictions"
            eval_pred_dir = shadow_dir / f"lora_r{rank}" / "prediction_exports"
            if not pred_dir.exists() or not eval_pred_dir.exists():
                continue
            train_preds = pd.read_csv(pred_dir / "train_predictions.csv")
            non_member_preds = pd.read_csv(eval_pred_dir / "train_predictions.csv")
            train_preds["member"] = 1
            non_member_preds["member"] = 0
            frames.extend([train_preds, non_member_preds])
    if not frames:
        raise ValueError("no prediction files found")
    return pd.concat(frames, ignore_index=True)


def load_target_predictions(target_dir: Path, evaluate_dir: Path, rank: int = 16) -> pd.DataFrame:
    pred_dir = target_dir / f"lora_r{rank}" / "predictions"
    eval_pred_dir = evaluate_dir / f"lora_r{rank}" / "prediction_exports"

    train_preds = pd.read_csv(pred_dir / "train_predictions.csv")
    non_member_preds = pd.read_csv(eval_pred_dir / "train_predictions.csv")

    train_preds["member"] = 1
    non_member_preds["member"] = 0

    return pd.concat([train_preds, non_member_preds], ignore_index=True)


def build_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["member"].values.astype(int)
    return X, y


def train_attack_classifier(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    return clf


def evaluate(clf: LogisticRegression, X: np.ndarray, y: np.ndarray, label: str = "") -> dict[str, float]:
    preds = clf.predict(X)
    probs = clf.predict_proba(X)[:, 1]
    acc = accuracy_score(y, preds)
    auc = roc_auc_score(y, probs)
    probs_flipped = 1 - probs
    auc_flipped = roc_auc_score(y, probs_flipped)
    print(f"  auc_roc_flipped: {auc_flipped:.4f}")
    fpr, tpr, _ = roc_curve(y, probs)
    tpr_at_low_fpr = float(tpr[np.searchsorted(fpr, 0.001)])

    metrics = {
        "accuracy": float(acc),
        "auc_roc": float(auc),
        "tpr_at_fpr_0.1pct": tpr_at_low_fpr,
    }

    if label:
        print(f"\n--- {label} ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return metrics


def save_classifier(clf: LogisticRegression, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, output_path)


def main(shadow_dirs: list[Path], target_dir: Path, evaluate_dir: Path, output_path: Path) -> None:
    print("loading shadow predictions...")
    df = load_shadow_predictions(shadow_dirs, ranks=[16])
    print(f"total samples: {len(df)} ({df['member'].sum()} members, {(df['member'] == 0).sum()} non-members)")

    X, y = build_features(df)

    print("training attack classifier...")
    clf = train_attack_classifier(X, y)

    print("evaluating on shadow data (sanity check, expected to look good)...")
    evaluate(clf, X, y, label="shadow data sanity check")

    print("\nloading target model predictions...")
    target_df = load_target_predictions(target_dir, evaluate_dir)
    print(target_df["member"].value_counts())
    X_target, y_target = build_features(target_df)

    print("evaluating on target model (real attack)...")
    evaluate(clf, X_target, y_target, label="target model attack")

    save_classifier(clf, output_path)
    print(f"\nclassifier saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--shadow-dirs", nargs="+", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--evaluate-dir", required=True)
    parser.add_argument("--output-path", default="runs/attack_classifier/attack_clf.joblib")
    args = parser.parse_args()

    main(
        shadow_dirs=[Path(d) for d in args.shadow_dirs],
        target_dir=Path(args.target_dir),
        evaluate_dir=Path(args.evaluate_dir),
        output_path=Path(args.output_path),
    )