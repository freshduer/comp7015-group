import time
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from data_utils import prepare_data_from_files


def measure_inference(model_path, tokenizer_path, test_dataset, batch_size=32, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading model from {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=256)
    
    test_tokenized = test_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    start_memory = torch.cuda.memory_allocated(device) if device.type == 'cuda' else 0
    start_time = time.time()
    
    total_samples = len(test_tokenized)
    num_batches = (total_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            batch = test_tokenized[i:i+batch_size]
            batch = data_collator(batch)
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            
            if i % 500 == 0 and i > 0:
                print(f"Processed {i}/{total_samples} samples")
    
    end_time = time.time()
    inference_time = end_time - start_time
    
    if device.type == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024**3)
        current_memory = torch.cuda.memory_allocated(device) / (1024**3)
    else:
        peak_memory = 0
        current_memory = 0
    
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return {
        'inference_time': inference_time,
        'peak_memory_gb': peak_memory,
        'samples_per_second': total_samples / inference_time,
        'total_samples': total_samples
    }


def plot_inference_comparison(bert_results, distilbert_results, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    models = ['BERT', 'DistilBERT']
    memories = [bert_results['peak_memory_gb'], distilbert_results['peak_memory_gb']]
    speeds = [bert_results['samples_per_second'], distilbert_results['samples_per_second']]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    x = np.arange(len(models))
    width = 0.6
    
    axes[0].bar(x, memories, width, alpha=0.8, color=['#1f77b4', '#ff7f0e'])
    axes[0].set_ylabel('Peak Memory Usage (GB)', fontsize=12)
    axes[0].set_title('Peak Memory Usage Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(memories):
        axes[0].text(i, v, f'{v:.2f}GB', ha='center', va='bottom', fontsize=10)
    
    axes[1].bar(x, speeds, width, alpha=0.8, color=['#1f77b4', '#ff7f0e'])
    axes[1].set_ylabel('Samples per Second', fontsize=12)
    axes[1].set_title('Inference Speed Comparison', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models)
    axes[1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(speeds):
        axes[1].text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    output_file = output_path / 'inference_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nInference comparison plot saved to {output_file}")
    
    plt.close()
    
    speedup_time = bert_results['inference_time'] / distilbert_results['inference_time']
    speedup_memory = bert_results['peak_memory_gb'] / distilbert_results['peak_memory_gb'] if distilbert_results['peak_memory_gb'] > 0 else 0
    
    print(f"\nPerformance Summary:")
    print(f"Time Speedup: {speedup_time:.2f}x (DistilBERT is {speedup_time:.2f}x faster)")
    print(f"Memory Reduction: {speedup_memory:.2f}x (DistilBERT uses {1/speedup_memory:.2f}x less memory)")


def main():
    base_dir = Path(__file__).parent
    bert_model_path = base_dir / "models" / "bert-imdb" / "best"
    distilbert_model_path = base_dir / "models" / "distilbert-imdb" / "best"
    output_dir = base_dir / "result" / "plots"
    
    if not bert_model_path.exists():
        raise FileNotFoundError(f"BERT model not found at {bert_model_path}")
    if not distilbert_model_path.exists():
        raise FileNotFoundError(f"DistilBERT model not found at {distilbert_model_path}")
    
    print("Loading test dataset...")
    _, _, test_ds = prepare_data_from_files(seed=42)
    print(f"Test dataset size: {len(test_ds)}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("\n" + "="*60)
    print("Testing BERT inference...")
    print("="*60)
    bert_results = measure_inference(
        str(bert_model_path),
        str(bert_model_path),
        test_ds,
        batch_size=32,
        device=device
    )
    
    print("\n" + "="*60)
    print("Testing DistilBERT inference...")
    print("="*60)
    distilbert_results = measure_inference(
        str(distilbert_model_path),
        str(distilbert_model_path),
        test_ds,
        batch_size=32,
        device=device
    )
    
    print("\n" + "="*60)
    print("Results:")
    print("="*60)
    print(f"BERT - Time: {bert_results['inference_time']:.2f}s, Memory: {bert_results['peak_memory_gb']:.2f}GB, Speed: {bert_results['samples_per_second']:.1f} samples/s")
    print(f"DistilBERT - Time: {distilbert_results['inference_time']:.2f}s, Memory: {distilbert_results['peak_memory_gb']:.2f}GB, Speed: {distilbert_results['samples_per_second']:.1f} samples/s")
    
    plot_inference_comparison(bert_results, distilbert_results, output_dir)


if __name__ == "__main__":
    main()

