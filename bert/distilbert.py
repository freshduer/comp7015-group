import argparse
import sys
from importlib import import_module
from pathlib import Path

import torch


def _load_module(primary: str, fallback: str):
    try:
        return import_module(primary)
    except ModuleNotFoundError:
        module_dir = Path(__file__).resolve().parent
        if str(module_dir) not in sys.path:
            sys.path.insert(0, str(module_dir))
        return import_module(fallback)


_data_utils = _load_module("bert.data_utils", "data_utils")
_training_utils = _load_module("bert.training_utils", "training_utils")

prepare_data_from_files = _data_utils.prepare_data_from_files
TrainingConfig = _training_utils.TrainingConfig
train_model = _training_utils.train_model


SEED = 42
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config/distilbert.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT on IMDB reviews.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the JSON config file that defines TrainingConfig parameters.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config).expanduser() if args.config else DEFAULT_CONFIG_PATH

    print("transformers/datasets ready. torch cuda:", torch.cuda.is_available())

    train_ds, val_ds, test_ds = prepare_data_from_files(seed=SEED)
    print("Raw splits ready. Train/Val/Test =", len(train_ds), len(val_ds), len(test_ds))

    config = TrainingConfig.from_file(config_path)

    trainer, tokenizer, metrics = train_model(
        train_ds,
        val_ds,
        test_ds,
        config,
    )

    print("Training completed!")
    print("Validation accuracy:", metrics["val"]["eval_accuracy"])
    print("Test accuracy:", metrics["test"]["eval_accuracy"])

    return trainer, tokenizer, metrics


if __name__ == "__main__":
    main()

