"""
Phase 3 — DP-SGD fine-tuning for DNABERT-2.

Replaces the HuggingFace Trainer with a manual training loop so we can
wrap the optimizer with Opacus's PrivacyEngine.  Everything else (data
loading, tokenization, prediction export, metrics) reuses Phase 1 code.

Only the classifier head is trained with DP-SGD (base encoder stays
frozen).  DNABERT-2's packed/unpadded attention flattens all sequences
into (total_tokens, hidden) before the encoder layers, which prevents
Opacus from computing per-sample gradients for any encoder parameter —
including LoRA adapters.  The classifier head receives the pooled output
(batch, hidden) so Opacus can track per-sample gradients normally.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    set_seed,
)

# ── reuse Phase 1 utilities ──────────────────────────────────────────
from dnabert2_phase1.data import build_split_manifest, load_local_splits
from dnabert2_phase1.metrics import compute_classification_metrics
from dnabert2_phase1.modeling import load_tokenizer, resolve_model_source
from dnabert2_phase1.runner import (
    _save_dataframe,
    _save_json,
    export_predictions,
    tokenize_frame,
)

# ── Opacus ───────────────────────────────────────────────────────────
from opacus.accountants import RDPAccountant
from opacus.grad_sample import GradSampleModule
from opacus.optimizers import DPOptimizer
from opacus.validators import ModuleValidator


# =====================================================================
# load base model (Opacus-compatible)
# =====================================================================

def load_dp_model(model_name: str, *, num_labels: int):
    """Load base model with frozen encoder and trainable classifier."""
    import transformers.modeling_utils as modeling_utils

    model_name = resolve_model_source(model_name)
    tokenizer = load_tokenizer(model_name)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.num_labels = num_labels
    config.pad_token_id = tokenizer.pad_token_id

    original_load_state_dict = modeling_utils.load_state_dict

    def load_state_dict_without_weights_only(checkpoint_file, *args, **kwargs):
        kwargs["weights_only"] = False
        return original_load_state_dict(checkpoint_file, *args, **kwargs)

    try:
        modeling_utils.load_state_dict = load_state_dict_without_weights_only
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=False,
        )
    finally:
        modeling_utils.load_state_dict = original_load_state_dict

    # Freeze entire base encoder, only train the classifier head
    for name, param in model.named_parameters():
        param.requires_grad = "classifier" in name

    # Let Opacus fix any incompatible layers (e.g. BatchNorm → GroupNorm)
    model = ModuleValidator.fix(model)

    return model


# =====================================================================
# manual training loop with DP-SGD
# =====================================================================

def train_with_dp(
    model: torch.nn.Module,
    train_loader: DataLoader,
    *,
    noise_multiplier: float,
    max_grad_norm: float,
    target_delta: float,
    n_train: int,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> dict[str, Any]:
    """Fine-tune *model* with DP-SGD via Opacus and return privacy metrics.

    Manually wraps model/optimizer (skipping DPDataLoader which has a
    torch 2.x compatibility bug in Opacus 1.5.4).
    """
    model = model.to(device)
    model.train()

    # Wrap model for per-sample gradient computation
    model = GradSampleModule(model)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=learning_rate,
    )

    # Wrap optimizer for DP noise injection
    batch_size = train_loader.batch_size
    sample_rate = batch_size / n_train
    accountant = RDPAccountant()

    optimizer = DPOptimizer(
        optimizer=optimizer,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        expected_batch_size=batch_size,
    )
    optimizer.attach_step_hook(
        accountant.get_optimizer_hook_fn(sample_rate=sample_rate),
    )

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            optimizer.zero_grad()
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        epsilon = accountant.get_epsilon(delta=target_delta)
        print(f"  epoch {epoch}/{epochs}  loss={avg_loss:.4f}  ε={epsilon:.2f} (δ={target_delta})")

    final_epsilon = accountant.get_epsilon(delta=target_delta)
    return {
        "epsilon": float(final_epsilon),
        "delta": float(target_delta),
        "noise_multiplier": float(noise_multiplier),
        "max_grad_norm": float(max_grad_norm),
    }


# =====================================================================
# full Phase 3 DP pipeline
# =====================================================================

def run_phase3_dp(
    data_dir: str,
    output_dir: str,
    noise_multiplier: float,
    max_grad_norm: float,
    *,
    base_model_name: str = "zhihan1996/DNABERT-2-117M",
    lora_rank: int = 16,
    max_length: int = 100,
    batch_size: int = 8,
    epochs: int = 10,
    learning_rate: float = 2e-5,
    seed: int = 42,
    expected_sequence_length: int = 70,
    device_str: str = "auto",
) -> None:

    set_seed(seed)
    run_root = Path(output_dir)
    run_root.mkdir(parents=True, exist_ok=True)

    # ── device ───────────────────────────────────────────────────────
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"using device: {device}")

    # ── data ─────────────────────────────────────────────────────────
    frames, prep_report = load_local_splits(
        data_dir,
        expected_length=expected_sequence_length,
    )
    _save_json(prep_report.to_dict(), run_root / "data_preparation_report.json")

    manifest = build_split_manifest(frames)
    _save_dataframe(manifest, run_root / "split_manifest.csv")

    tokenizer = load_tokenizer(base_model_name)
    train_ds = tokenize_frame(frames["train"], tokenizer, max_length)
    dev_ds = tokenize_frame(frames["dev"], tokenizer, max_length)
    test_ds = tokenize_frame(frames["test"], tokenizer, max_length)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator, drop_last=True)

    # ── delta ────────────────────────────────────────────────────────
    n_train = len(frames["train"])
    target_delta = 1.0 / n_train

    # ── model ────────────────────────────────────────────────────────
    rank_dir = run_root / f"lora_r{lora_rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)

    model = load_dp_model(base_model_name, num_labels=2)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ── train with DP ────────────────────────────────────────────────
    print(f"\n=== DP-SGD training: σ={noise_multiplier}, C={max_grad_norm} ===")
    dp_metrics = train_with_dp(
        model,
        train_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        target_delta=target_delta,
        n_train=n_train,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
    )

    # ── unwrap Opacus wrapper for inference ──────────────────────────
    if hasattr(model, "_module"):
        model = model._module

    # ── save model ───────────────────────────────────────────────────
    model.save_pretrained(rank_dir / "best_model")
    tokenizer.save_pretrained(rank_dir / "best_model")

    # ── export predictions ───────────────────────────────────────────
    pred_batch = max(1, min(batch_size, 4))
    train_metrics = export_predictions(
        model, tokenizer, train_ds, frames["train"],
        rank_dir / "predictions" / "train_predictions.csv",
        batch_size=pred_batch, device=device,
    )
    test_metrics = export_predictions(
        model, tokenizer, test_ds, frames["test"],
        rank_dir / "predictions" / "test_predictions.csv",
        batch_size=pred_batch, device=device,
    )

    # ── save metrics ─────────────────────────────────────────────────
    metrics_payload = {
        "rank": lora_rank,
        "dp": dp_metrics,
        "train": train_metrics.to_dict(),
        "test": test_metrics.to_dict(),
        "train_test_gap": round(train_metrics.accuracy - test_metrics.accuracy, 4),
    }
    _save_json(metrics_payload, rank_dir / "metrics.json")

    # ── save config ──────────────────────────────────────────────────
    config_payload = {
        "phase": 3,
        "defense": "dp-sgd",
        "noise_multiplier": noise_multiplier,
        "max_grad_norm": max_grad_norm,
        "epsilon": dp_metrics["epsilon"],
        "delta": dp_metrics["delta"],
        "lora_rank": lora_rank,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "seed": seed,
        "data_dir": data_dir,
        "output_dir": output_dir,
    }
    _save_json(config_payload, run_root / "config.json")

    print(f"\n=== Done ===")
    print(f"  ε = {dp_metrics['epsilon']:.2f}")
    print(f"  train acc = {train_metrics.accuracy:.4f}  MCC = {train_metrics.mcc:.4f}")
    print(f"  test  acc = {test_metrics.accuracy:.4f}  MCC = {test_metrics.mcc:.4f}")
    print(f"  train-test gap = {metrics_payload['train_test_gap']:.4f}")


# =====================================================================
# CLI
# =====================================================================

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 3: DP-SGD fine-tuning")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--noise-multiplier", type=float, required=True, help="Opacus noise multiplier σ")
    parser.add_argument("--max-grad-norm", type=float, required=True, help="Per-sample gradient clipping norm C")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--expected-sequence-length", type=int, default=70)
    args = parser.parse_args(argv)

    run_phase3_dp(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        noise_multiplier=args.noise_multiplier,
        max_grad_norm=args.max_grad_norm,
        lora_rank=args.lora_rank,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device_str=args.device,
        expected_sequence_length=args.expected_sequence_length,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
