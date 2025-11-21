import json
from pathlib import Path

import matplotlib.pyplot as plt


RESULT_DIR = Path(__file__).resolve().parent / "result"
OUTPUT_DIR = RESULT_DIR / "plots"


def extract_lr_info(stem: str) -> tuple[str, str]:
    parts = stem.split("-lr", maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"Unexpected filename pattern: {stem}")
    lr_part = parts[1]
    if lr_part.endswith("-warmup"):
        lr_value = lr_part[:-7]
        suffix = "-warmup"
    else:
        lr_value = lr_part
        suffix = ""
    return lr_value, suffix


def parse_lr_for_sort(lr_str: str) -> float:
    try:
        return float(lr_str)
    except ValueError:
        return 0.0


def load_curves(model_name: str) -> list:
    curves = []
    for path in sorted(RESULT_DIR.glob(f"{model_name}-lr*.json")):
        lr_value, suffix = extract_lr_info(path.stem)
        lr_label = lr_value + suffix if suffix else lr_value
        with path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        epochs = [entry["epoch"] for entry in payload["epochs"]]
        f1_scores = [entry["eval_f1"] for entry in payload["epochs"]]
        curves.append({"lr": lr_label, "lr_value": lr_value, "epochs": epochs, "f1": f1_scores})
    if not curves:
        raise FileNotFoundError(f"No lr runs found for {model_name}")
    curves.sort(key=lambda item: (parse_lr_for_sort(item["lr_value"]), item["lr"]))
    return curves


def plot_model(model_name: str, curves: list, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for curve in curves:
        label = f"α = {curve['lr']}"
        ax.plot(curve["epochs"], curve["f1"], marker="o", label=label)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation F1 Score", fontsize=12)
    model_display = model_name.replace("-", " ").title()
    ax.set_title(f"{model_display}: Validation F1 Score vs. Epoch", fontsize=13, fontweight="bold")
    all_epochs = sorted({epoch for curve in curves for epoch in curve["epochs"]})
    ax.set_xticks(all_epochs)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(title="Learning Rate", fontsize=10, title_fontsize=11)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    bert_curves = load_curves("bert-base-uncased")
    distil_curves = load_curves("distilbert-base-uncased")
    plot_model(
        "bert-base-uncased",
        bert_curves,
        OUTPUT_DIR / "bert-base-uncased-f1.png",
    )
    plot_model(
        "distilbert-base-uncased",
        distil_curves,
        OUTPUT_DIR / "distilbert-base-uncased-f1.png",
    )


if __name__ == "__main__":
    main()

