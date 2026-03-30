from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .config import Phase1Config
from .data import export_hf_dataset_to_csv
from .runner import run_phase1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dnabert2-phase1")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-hf", help="Export a Hugging Face dataset to train/dev/test CSVs")
    prepare.add_argument("--dataset-name", required=True)
    prepare.add_argument("--dataset-config", default=None)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--sequence-column", default="sequence")
    prepare.add_argument("--label-column", default="label")

    train = subparsers.add_parser("train", help="Run Phase 1 LoRA fine-tuning")
    train.add_argument("--data-dir", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--base-model-name", default="zhihan1996/DNABERT-2-117M")
    train.add_argument("--max-length", type=int, default=100)
    train.add_argument("--learning-rate", type=float, default=2e-5)
    train.add_argument("--train-batch-size", type=int, default=8)
    train.add_argument("--eval-batch-size", type=int, default=16)
    train.add_argument("--gradient-accumulation-steps", type=int, default=1)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--weight-decay", type=float, default=0.0)
    train.add_argument("--warmup-steps", type=int, default=50)
    train.add_argument("--logging-steps", type=int, default=25)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--no-fp16", action="store_true")
    train.add_argument("--lora-ranks", nargs="+", type=int, default=[8, 16, 32])
    train.add_argument("--lora-dropout", type=float, default=0.05)
    train.add_argument("--target-modules", default=".*Wqkv.*")
    train.add_argument("--modules-to-save", nargs="+", default=["classifier"])
    train.add_argument("--expected-sequence-length", type=int, default=70)
    train.add_argument("--train-csv", default="train.csv")
    train.add_argument("--dev-csv", default="dev.csv")
    train.add_argument("--test-csv", default="test.csv")
    train.add_argument("--hf-dataset-name", default=None)
    train.add_argument("--hf-dataset-config", default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "prepare-hf":
        export_hf_dataset_to_csv(
            args.dataset_name,
            args.output_dir,
            dataset_config=args.dataset_config,
            sequence_column=args.sequence_column,
            label_column=args.label_column,
        )
        return 0

    if args.command == "train":
        config = Phase1Config(
            base_model_name=args.base_model_name,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            max_length=args.max_length,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.epochs,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            logging_steps=args.logging_steps,
            seed=args.seed,
            fp16=not args.no_fp16,
            lora_ranks=args.lora_ranks,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            modules_to_save=args.modules_to_save,
            expected_sequence_length=args.expected_sequence_length,
            train_csv=args.train_csv,
            dev_csv=args.dev_csv,
            test_csv=args.test_csv,
            hf_dataset_name=args.hf_dataset_name,
            hf_dataset_config=args.hf_dataset_config,
        )
        run_phase1(config)
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
