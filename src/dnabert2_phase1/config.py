from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json


@dataclass(slots=True)
class Phase1Config:
    base_model_name: str = "zhihan1996/DNABERT-2-117M"
    data_dir: str = ""
    output_dir: str = "runs/phase1"
    max_length: int = 100
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 10
    weight_decay: float = 0.0
    warmup_steps: int = 50
    logging_steps: int = 25
    save_total_limit: int = 2
    seed: int = 42
    fp16: bool = True
    lora_ranks: list[int] = field(default_factory=lambda: [8, 16, 32])
    lora_alpha_multiplier: int = 2
    lora_dropout: float = 0.05
    target_modules: str = ".*Wqkv.*"
    modules_to_save: list[str] = field(default_factory=lambda: ["classifier"])
    expected_sequence_length: int | None = 70
    train_csv: str = "train.csv"
    dev_csv: str = "dev.csv"
    test_csv: str = "test.csv"
    sequence_column: str = "sequence"
    label_column: str = "label"
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    hf_dataset_name: str | None = None
    hf_dataset_config: str | None = None
    hf_sequence_column: str = "sequence"
    hf_label_column: str = "label"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> "Phase1Config":
        payload = json.loads(Path(path).read_text())
        valid_fields = {f.name for f in __import__("dataclasses").fields(cls)}
        payload = {k: v for k, v in payload.items() if k in valid_fields}
        return cls(**payload)
