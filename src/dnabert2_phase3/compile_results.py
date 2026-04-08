"""
Compile all Phase 3 individual result JSONs into a single summary
and print a human-readable comparison table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    all_results = []

    for json_file in sorted(results_dir.glob("*.json")):
        if json_file.name == "phase3_summary.json":
            continue
        with open(json_file) as f:
            data = json.load(f)
        all_results.append(data)

    if not all_results:
        print("no result files found")
        return

    # save combined summary
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))

    # print table
    header = f"{'Run':<25} {'ε':>8} {'Atk Acc':>8} {'Atk AUC':>8} {'Test MCC':>9} {'T-T Gap':>8}"
    print("\n" + "=" * len(header))
    print("Phase 3 — Privacy-Utility Tradeoff Summary")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for r in all_results:
        name = r.get("run", "?")
        eps = r.get("epsilon", "—")
        atk_acc = r.get("attack_accuracy", "—")
        atk_auc = r.get("attack_auc_roc", "—")
        test_mcc = r.get("test_mcc", "—")
        gap = r.get("train_test_gap", "—")

        eps_str = f"{eps:.2f}" if isinstance(eps, (int, float)) else str(eps)
        atk_acc_str = f"{atk_acc:.4f}" if isinstance(atk_acc, (int, float)) else str(atk_acc)
        atk_auc_str = f"{atk_auc:.4f}" if isinstance(atk_auc, (int, float)) else str(atk_auc)
        mcc_str = f"{test_mcc:.4f}" if isinstance(test_mcc, (int, float)) else str(test_mcc)
        gap_str = f"{gap:.4f}" if isinstance(gap, (int, float)) else str(gap)

        print(f"{name:<25} {eps_str:>8} {atk_acc_str:>8} {atk_auc_str:>8} {mcc_str:>9} {gap_str:>8}")

    print("=" * len(header))
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
