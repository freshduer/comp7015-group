import os
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

from datasets import Dataset
from sklearn.model_selection import train_test_split


__all__ = [
    "load_reviews",
    "prepare_data_from_memory",
    "prepare_data_from_files",
]


def load_reviews(dir_path: Path, label: int) -> Tuple[Sequence[str], Sequence[int]]:
    texts, labels = [], []
    for fp in sorted(dir_path.glob("*.txt")):
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = fp.read_text(errors="ignore")
        texts.append(text)
        labels.append(label)
    return texts, labels


def prepare_data_from_memory(
    X_train: Sequence[str],
    X_val: Sequence[str],
    X_test: Sequence[str],
    y_train: Sequence[int],
    y_val: Sequence[int],
    y_test: Sequence[int],
) -> Tuple[Dataset, Dataset, Dataset]:
    train_ds = Dataset.from_dict({"text": list(X_train), "label": list(y_train)})
    val_ds = Dataset.from_dict({"text": list(X_val), "label": list(y_val)})
    test_ds = Dataset.from_dict({"text": list(X_test), "label": list(y_test)})
    return train_ds, val_ds, test_ds


def _resolve_data_dir(data_dir: Optional[os.PathLike]) -> Path:
    if data_dir:
        base_dir = Path(data_dir)
        if base_dir.exists():
            return base_dir
        raise FileNotFoundError(f"Dataset directory not found: {base_dir}")

    root_dir = Path(__file__).resolve().parents[1]
    candidates: Iterable[Path] = (
        Path("./data/aclImdb"),
        Path("./data/acllmdb"),
        root_dir / "data" / "aclImdb",
        root_dir / "data" / "acllmdb",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Dataset directory not found: data/aclImdb or data/acllmdb")


def prepare_data_from_files(
    data_dir: Optional[os.PathLike] = None,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    base_dir = _resolve_data_dir(data_dir)

    pos_texts, pos_labels = load_reviews(base_dir / "train" / "pos", 1)
    neg_texts, neg_labels = load_reviews(base_dir / "train" / "neg", 0)
    X_all = list(pos_texts) + list(neg_texts)
    y_all = list(pos_labels) + list(neg_labels)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all,
        y_all,
        test_size=0.2,
        random_state=seed,
        stratify=y_all,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=seed,
        stratify=y_temp,
    )

    return prepare_data_from_memory(X_train, X_val, X_test, y_train, y_val, y_test)

