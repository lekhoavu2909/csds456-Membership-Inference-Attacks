"""
Evaluate a pre-trained attack classifier against a defended model's
prediction exports.  Reads the same CSV format that Phase 1/2 produce.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

FEATURE_COLS = ["prob_class_0", "prob_class_1", "entropy"]


def load_predictions(run_dir: Path, rank: int) -> pd.DataFrame:
    """Load member (train) and non-member (test) predictions."""
    pred_dir = run_dir / f"lora_r{rank}" / "predictions"
    export_dir = run_dir / f"lora_r{rank}" / "prediction_exports"

    # train predictions from the training-time export (members)
    train_path = pred_dir / "train_predictions.csv"
    # test predictions from the export step (non-members)
    test_path = export_dir / "train_predictions.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"missing {train_path}")

    # fall back: if export_dir doesn't exist yet, use predictions dir test
    if test_path.exists():
        non_member_path = test_path
    else:
        alt = pred_dir / "test_predictions.csv"
        if alt.exists():
            non_member_path = alt
        else:
            raise FileNotFoundError(f"missing non-member predictions in {export_dir} or {pred_dir}")

    train_preds = pd.read_csv(train_path)
    non_member_preds = pd.read_csv(non_member_path)

    train_preds["member"] = 1
    non_member_preds["member"] = 0

    return pd.concat([train_preds, non_member_preds], ignore_index=True)


def evaluate_attack(clf, df: pd.DataFrame) -> dict[str, float]:
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["member"].values.astype(int)

    preds = clf.predict(X)
    probs = clf.predict_proba(X)[:, 1]

    acc = accuracy_score(y, preds)
    auc = roc_auc_score(y, probs)

    fpr, tpr, _ = roc_curve(y, probs)
    tpr_at_low_fpr = float(tpr[np.searchsorted(fpr, 0.001)])

    return {
        "attack_accuracy": round(float(acc), 4),
        "attack_auc_roc": round(float(auc), 4),
        "attack_tpr_at_fpr_0.1pct": round(tpr_at_low_fpr, 4),
    }


def load_task_metrics(run_dir: Path, rank: int) -> dict[str, float]:
    """Pull train/test accuracy and MCC from the metrics.json saved during training."""
    metrics_path = run_dir / f"lora_r{rank}" / "metrics.json"
    if not metrics_path.exists():
        return {}
    with open(metrics_path) as f:
        data = json.load(f)
    result = {}
    if "train" in data:
        result["train_accuracy"] = data["train"].get("accuracy", None)
        result["train_mcc"] = data["train"].get("mcc", None)
    if "test" in data:
        result["test_accuracy"] = data["test"].get("accuracy", None)
        result["test_mcc"] = data["test"].get("mcc", None)
    if "train_test_gap" in data:
        result["train_test_gap"] = data["train_test_gap"]
    elif result.get("train_accuracy") and result.get("test_accuracy"):
        result["train_test_gap"] = round(result["train_accuracy"] - result["test_accuracy"], 4)
    if "dp" in data:
        result["epsilon"] = data["dp"].get("epsilon", None)
        result["noise_multiplier"] = data["dp"].get("noise_multiplier", None)
        result["max_grad_norm"] = data["dp"].get("max_grad_norm", None)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--attack-clf", required=True, help="Path to trained attack_clf.joblib")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    clf = joblib.load(args.attack_clf)

    print(f"evaluating {run_dir.name} ...")
    df = load_predictions(run_dir, args.rank)
    attack_metrics = evaluate_attack(clf, df)
    task_metrics = load_task_metrics(run_dir, args.rank)

    result = {
        "run": run_dir.name,
        "rank": args.rank,
        **task_metrics,
        **attack_metrics,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    print(f"  attack_accuracy: {attack_metrics['attack_accuracy']:.4f}")
    print(f"  attack_auc_roc:  {attack_metrics['attack_auc_roc']:.4f}")
    if "test_mcc" in task_metrics:
        print(f"  test_mcc:        {task_metrics['test_mcc']:.4f}")
    print(f"  saved to {out_path}")


if __name__ == "__main__":
    main()
