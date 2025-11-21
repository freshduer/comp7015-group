import json
import os
import time
import warnings
from dataclasses import MISSING, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", message="Some weights of.*were not initialized")
warnings.filterwarnings("ignore", message="You should probably TRAIN this model")
warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0, but all input tensors were scalars")
warnings.filterwarnings("ignore", message="Detected kernel version.*which is below the recommended minimum")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.parallel._functions")


@dataclass
class TrainingConfig:
    model_name: str
    output_dir: str
    num_epochs: int = 2
    batch_size: int = 16
    max_length: int = 256
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "linear"
    seed: int = 42
    metrics_file: Optional[str] = None

    @classmethod
    def from_file(cls, config_path: Union[str, os.PathLike[str]]):
        path = Path(config_path).expanduser()
        if not path.is_absolute():
            module_dir = Path(__file__).resolve().parent
            path = module_dir / path
        if not path.is_file():
            raise FileNotFoundError(f"Training config not found: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)

        allowed_fields = {field.name for field in fields(cls)}
        config_kwargs = {k: v for k, v in payload.items() if k in allowed_fields}

        required_fields = {
            field.name
            for field in fields(cls)
            if field.default is MISSING and field.default_factory is MISSING  # type: ignore[attr-defined]
        }
        missing = required_fields - config_kwargs.keys()
        if missing:
            raise ValueError(f"Missing fields in training config {path}: {sorted(missing)}")

        return cls(**config_kwargs)


def tokenize_datasets(
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    model_name: str,
    max_length: int,
) -> Tuple[Dataset, Dataset, Dataset, AutoTokenizer, DataCollatorWithPadding]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tok_fn(batch: Dict[str, str]):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    num_proc = max(1, os.cpu_count() or 1)
    train_tok = train_ds.map(tok_fn, batched=True, remove_columns=["text"], num_proc=num_proc)
    val_tok = val_ds.map(tok_fn, batched=True, remove_columns=["text"], num_proc=num_proc)
    test_tok = test_ds.map(tok_fn, batched=True, remove_columns=["text"], num_proc=num_proc)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return train_tok, val_tok, test_tok, tokenizer, collator


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


def _resolve_metrics_file(output_dir: str, metrics_file: Optional[str], model_name: str) -> str:
    if metrics_file is None:
        safe_model_name = model_name.replace("/", "-")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        result_dir = Path(__file__).resolve().parent / "result"
        return str(result_dir / f"{safe_model_name}-{timestamp}.json")
    if os.path.isabs(metrics_file):
        return metrics_file
    return os.path.join(output_dir, metrics_file)


def _reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _get_peak_memory_bytes():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated()
    return None


def _timed_evaluate(trainer: Trainer, dataset: Dataset):
    _reset_peak_memory()
    start = time.perf_counter()
    metrics = trainer.evaluate(eval_dataset=dataset)
    duration = time.perf_counter() - start
    peak_mem = _get_peak_memory_bytes()
    return metrics, duration, peak_mem


def train_model(
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    config: TrainingConfig,
):
    set_seed(config.seed)

    train_tok, val_tok, test_tok, tokenizer, collator = tokenize_datasets(
        train_ds,
        val_ds,
        test_ds,
        config.model_name,
        config.max_length,
    )

    model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=2)
    param_count = sum(p.numel() for p in model.parameters())
    estimated_model_size_mb = (param_count * 4) / (1024**2)

    args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size * 2,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
        group_by_length=True,
        dataloader_num_workers=2,
        seed=config.seed,
        data_seed=config.seed,
    )

    metrics_path = _resolve_metrics_file(config.output_dir, config.metrics_file, config.model_name)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    _reset_peak_memory()
    train_start = time.perf_counter()
    trainer.train()
    train_wall_time = time.perf_counter() - train_start
    train_peak_mem = _get_peak_memory_bytes()

    epoch_metrics = []
    for log_entry in trainer.state.log_history:
        if "eval_loss" in log_entry and "epoch" in log_entry:
            epoch_metrics.append(
                {
                    "epoch": int(log_entry["epoch"]),
                    "eval_loss": float(log_entry.get("eval_loss", 0.0)),
                    "eval_accuracy": float(log_entry.get("eval_accuracy", 0.0)),
                    "eval_f1": float(log_entry.get("eval_f1", 0.0)),
                }
            )

    metrics_val, val_eval_time, val_peak_mem = _timed_evaluate(trainer, val_tok)
    print("Val metrics:", metrics_val)

    test_metrics, test_eval_time, test_peak_mem = _timed_evaluate(trainer, test_tok)
    print("Test metrics:", test_metrics)

    best_model_dir = os.path.join(config.output_dir, "best")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    summary = {
        "val_metrics": metrics_val,
        "test_metrics": test_metrics,
        "train_wall_time_sec": train_wall_time,
        "val_eval_time_sec": val_eval_time,
        "test_eval_time_sec": test_eval_time,
        "train_peak_mem_bytes": train_peak_mem,
        "val_peak_mem_bytes": val_peak_mem,
        "test_peak_mem_bytes": test_peak_mem,
        "param_count": param_count,
        "estimated_model_size_mb": estimated_model_size_mb,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }

    metrics_payload = {
        "epochs": epoch_metrics,
        "summary": summary,
    }

    metrics_dir = os.path.dirname(metrics_path)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)
    print(f"Evaluation metrics saved to: {metrics_path}")
    print(f"Total epochs recorded: {len(epoch_metrics)}")

    metrics = {
        "val": metrics_val,
        "test": test_metrics,
        "epoch_metrics": epoch_metrics,
        "summary": summary,
    }

    return trainer, tokenizer, metrics

