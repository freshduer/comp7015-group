from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data_utils import prepare_data_from_files


def select_tokens(tokens: List[str], limit: int) -> Tuple[List[str], slice]:
    if len(tokens) <= limit:
        return tokens, slice(None)
    return tokens[:limit], slice(0, limit)


def plot_attention_heatmap(
    attn_matrix,
    tokens: List[str],
    layer_idx: int,
    head: int,
    output_path: Path,
    sample_idx: int,
    fig_width: float = None,
    fig_height: float = None,
):
    default_w = min(10, 0.4 * len(tokens) + 3)
    default_h = min(8, 0.4 * len(tokens) + 3)
    final_fig_width = fig_width if fig_width else default_w
    final_fig_height = fig_height if fig_height else default_h
    fig, ax = plt.subplots(figsize=(final_fig_width, final_fig_height))
    im = ax.imshow(attn_matrix, cmap="viridis")
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90, fontsize=14)
    ax.set_yticklabels(tokens, fontsize=14)
    ax.set_xlabel("Key Tokens", fontsize=22, fontweight="bold")
    ax.set_ylabel("Query Tokens", fontsize=22, fontweight="bold")
    ax.set_title(f"Attention Weights: Sample{sample_idx}", fontsize=24, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Attention heatmap saved to: {output_path}")


def main():
    # Configuration parameters
    model_dir = Path(__file__).resolve().parent / "models" / "bert-imdb" / "best"
    data_dir = None
    sample_index = 0
    max_text_chars = 100
    layer = -1
    head = 0
    num_samples = 5
    max_length = 256
    max_tokens = 64
    output = Path(__file__).resolve().parent / "result" / "plots" / "attention_heatmap.png"
    fig_width = None
    fig_height = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_ds = prepare_data_from_files(data_dir=data_dir)
    sample_idx = max(0, min(sample_index, len(test_ds) - 1))

    def pick_samples(num: int) -> List[Tuple[int, Dict[str, str]]]:
        samples = []
        offset = 0
        while len(samples) < num and offset < len(test_ds):
            idx = (sample_idx + offset) % len(test_ds)
            sample = test_ds[idx]
            if len(sample["text"]) <= max_text_chars:
                samples.append((idx, sample))
            offset += 1
        return samples

    samples = pick_samples(num_samples)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    output_dir = output.parent
    output_stem = output.stem
    output_suffix = output.suffix

    for sample_idx, sample in samples:
        text = sample["text"]

        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded, output_attentions=True)

        attentions = outputs.attentions
        if attentions is None:
            raise RuntimeError("Model did not return attentions. Ensure it supports attention outputs.")

        layer_idx = layer if layer >= 0 else len(attentions) + layer
        if not (0 <= layer_idx < len(attentions)):
            raise ValueError(f"Layer index {layer} is out of range.")

        layer_attn = attentions[layer_idx][0]  # shape: (heads, seq_len, seq_len)
        if not (0 <= head < layer_attn.shape[0]):
            raise ValueError(f"Head index {head} is out of range.")

        attn_matrix = layer_attn[head].detach().cpu().numpy()
        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
        tokens, span = select_tokens(tokens, max_tokens)
        attn_matrix = attn_matrix[span, span]

        output_path = output_dir / f"{output_stem}_sample{sample_idx}{output_suffix}"
        plot_attention_heatmap(
            attn_matrix,
            tokens,
            layer_idx,
            head,
            output_path,
            sample_idx,
            fig_width,
            fig_height,
        )


if __name__ == "__main__":
    main()

