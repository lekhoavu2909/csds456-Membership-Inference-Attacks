from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments, set_seed

from .config import Phase1Config
from .data import build_split_manifest, load_local_splits, split_summary
from .metrics import ClassificationMetrics, compute_classification_metrics, trainer_compute_metrics
from .modeling import load_sequence_classifier, load_tokenizer


def tokenize_frame(frame: pd.DataFrame, tokenizer, max_length: int) -> Dataset:
    dataset = Dataset.from_pandas(frame.reset_index(drop=True))

    def _tokenize(batch):
        return tokenizer(
            batch["sequence"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    tokenized = dataset.map(_tokenize, batched=True, remove_columns=["sequence"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


def _save_dataframe(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def export_predictions(
    model: torch.nn.Module,
    tokenizer,
    dataset: Dataset,
    frame: pd.DataFrame,
    out_path: Path,
    *,
    batch_size: int = 8,
    device: torch.device | str | None = None,
) -> ClassificationMetrics:
    model_device = torch.device(device) if device is not None else next(model.parameters()).device
    model = model.to(model_device)
    model.eval()

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)

    logits_chunks: list[np.ndarray] = []
    labels_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").detach().cpu().numpy()
            batch = {key: value.to(model_device) for key, value in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            logits_chunks.append(logits.detach().cpu().numpy())
            labels_chunks.append(labels)

    logits = np.concatenate(logits_chunks, axis=0) if logits_chunks else np.empty((0, 2))
    labels = np.concatenate(labels_chunks, axis=0) if labels_chunks else np.empty((0,), dtype=int)
    probs = torch.softmax(torch.tensor(logits), dim=-1).cpu().numpy()
    metrics = compute_classification_metrics(logits, labels)

    export = frame.reset_index(drop=True).copy()
    export["label"] = labels
    export["predicted_label"] = probs.argmax(axis=-1)
    export["prob_class_0"] = probs[:, 0]
    export["prob_class_1"] = probs[:, 1]
    export["max_probability"] = probs.max(axis=-1)
    export["entropy"] = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=-1)
    _save_dataframe(export, out_path)
    return metrics


def run_phase1(config: Phase1Config) -> list[dict[str, Any]]:
    set_seed(config.seed)
    run_root = Path(config.output_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    config.save(run_root / "config.json")

    frames, prep_report = load_local_splits(
        config.data_dir,
        train_name=config.train_csv,
        dev_name=config.dev_csv,
        test_name=config.test_csv,
        expected_length=config.expected_sequence_length,
    )
    prepared_dir = run_root / "prepared_data"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    for split_name, frame in frames.items():
        frame.to_csv(prepared_dir / f"{split_name}.csv", index=False)
    _save_json(prep_report.to_dict(), run_root / "data_preparation_report.json")

    manifest = build_split_manifest(frames)
    _save_dataframe(manifest, run_root / "split_manifest.csv")

    summaries = {split: asdict(split_summary(frame)) for split, frame in frames.items()}
    _save_json(summaries, run_root / "data_summary.json")

    tokenizer = load_tokenizer(config.base_model_name)

    results: list[dict[str, Any]] = []
    for rank in config.lora_ranks:
        rank_dir = run_root / f"lora_r{rank}"
        checkpoint_dir = rank_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        model = load_sequence_classifier(
            config.base_model_name,
            num_labels=2,
            lora_r=rank,
            lora_alpha=config.lora_alpha_multiplier * rank,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            modules_to_save=config.modules_to_save,
        )

        train_ds = tokenize_frame(frames["train"], tokenizer, config.max_length)
        dev_ds = tokenize_frame(frames["dev"], tokenizer, config.max_length)
        test_ds = tokenize_frame(frames["test"], tokenizer, config.max_length)

        training_args = TrainingArguments(
            output_dir=str(checkpoint_dir),
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            num_train_epochs=config.num_train_epochs,
            warmup_steps=config.warmup_steps,
            weight_decay=config.weight_decay,
            logging_steps=config.logging_steps,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model=config.metric_for_best_model,
            greater_is_better=config.greater_is_better,
            save_total_limit=config.save_total_limit,
            fp16=config.fp16 and torch.cuda.is_available(),
            report_to=[],
            remove_unused_columns=False,
            seed=config.seed,
            data_seed=config.seed,
        )

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=dev_ds,
            data_collator=data_collator,
            processing_class=tokenizer,
            compute_metrics=trainer_compute_metrics,
        )

        train_output = trainer.train()
        trainer.save_model(rank_dir / "best_model")
        tokenizer.save_pretrained(rank_dir / "best_model")

        dev_metrics = trainer.evaluate(dev_ds)
        prediction_batch_size = max(1, min(config.per_device_eval_batch_size, 4))
        model_device = next(trainer.model.parameters()).device
        train_metrics = export_predictions(
            trainer.model,
            tokenizer,
            train_ds,
            frames["train"],
            rank_dir / "predictions" / "train_predictions.csv",
            batch_size=prediction_batch_size,
            device=model_device,
        )
        test_metrics = export_predictions(
            trainer.model,
            tokenizer,
            test_ds,
            frames["test"],
            rank_dir / "predictions" / "test_predictions.csv",
            batch_size=prediction_batch_size,
            device=model_device,
        )

        trainer_state_path = rank_dir / "trainer_state.json"
        if trainer.state is not None:
            trainer.state.save_to_json(str(trainer_state_path))

        metrics_payload = {
            "rank": rank,
            "train": train_metrics.to_dict(),
            "dev": {"accuracy": float(dev_metrics.get("eval_accuracy", 0.0)), "mcc": float(dev_metrics.get("eval_mcc", 0.0)), "loss": float(dev_metrics.get("eval_loss", 0.0))},
            "test": test_metrics.to_dict(),
            "train_runtime": float(train_output.metrics.get("train_runtime", 0.0)),
        }
        _save_json(metrics_payload, rank_dir / "metrics.json")

        results.append(
            {
                "rank": rank,
                "run_dir": str(rank_dir),
                "train_accuracy": train_metrics.accuracy,
                "train_mcc": train_metrics.mcc,
                "dev_accuracy": metrics_payload["dev"]["accuracy"],
                "dev_mcc": metrics_payload["dev"]["mcc"],
                "test_accuracy": test_metrics.accuracy,
                "test_mcc": test_metrics.mcc,
            }
        )

    _save_json({"runs": results}, run_root / "summary.json")
    return results
