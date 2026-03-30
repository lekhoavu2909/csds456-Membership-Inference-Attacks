from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


def resolve_model_source(model_name: str) -> str:
    local_dir = Path("models/DNABERT-2-117M")
    if model_name == "zhihan1996/DNABERT-2-117M" and local_dir.exists():
        return str(local_dir)
    return model_name


def load_tokenizer(model_name: str):
    model_name = resolve_model_source(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token or tokenizer.cls_token or tokenizer.unk_token
    return tokenizer


def infer_modules_to_save(model) -> list[str]:
    module_names = {name.split(".")[-1] for name, _ in model.named_modules()}
    candidates = ["classifier", "score", "pre_classifier"]
    return [name for name in candidates if name in module_names]


def load_sequence_classifier(
    model_name: str,
    *,
    num_labels: int,
    lora_r: int,
    lora_alpha: int | None = None,
    lora_dropout: float = 0.05,
    target_modules: str | Iterable[str] = ".*Wqkv.*",
    modules_to_save: Iterable[str] | None = None,
):
    model_name = resolve_model_source(model_name)
    tokenizer = load_tokenizer(model_name)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.num_labels = num_labels
    config.pad_token_id = tokenizer.pad_token_id
    config.bos_token_id = getattr(tokenizer, "bos_token_id", None)
    config.eos_token_id = getattr(tokenizer, "eos_token_id", None)
    config.sep_token_id = getattr(tokenizer, "sep_token_id", None)
    config.cls_token_id = getattr(tokenizer, "cls_token_id", None)
    config.device = torch.device("cpu")
    previous_default_device = None
    try:
        if hasattr(torch, "get_default_device") and hasattr(torch, "set_default_device"):
            previous_default_device = torch.get_default_device()
            torch.set_default_device("cpu")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=False,
        )
    finally:
        if previous_default_device is not None:
            torch.set_default_device(previous_default_device)
    modules_to_save = infer_modules_to_save(model)
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha if lora_alpha is not None else 2 * lora_r,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=target_modules,
        modules_to_save=list(modules_to_save) if modules_to_save is not None else None,
    )
    model = get_peft_model(model, lora_config)
    return model
