import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results(result_dir):
    bert_data = []
    distilbert_data = []
    
    result_path = Path(result_dir)
    
    for json_file in result_path.glob("*.json"):
        if json_file.name.startswith("bert-base-uncased"):
            with open(json_file, 'r') as f:
                data = json.load(f)
                config = json_file.stem.replace("bert-base-uncased-", "")
                bert_data.append({
                    'config': config,
                    'train_time': data['summary']['train_wall_time_sec'],
                    'train_memory': data['summary']['train_peak_mem_bytes'] / (1024**3),
                    'val_memory': data['summary']['val_peak_mem_bytes'] / (1024**3),
                    'test_memory': data['summary']['test_peak_mem_bytes'] / (1024**3)
                })
        elif json_file.name.startswith("distilbert-base-uncased"):
            with open(json_file, 'r') as f:
                data = json.load(f)
                config = json_file.stem.replace("distilbert-base-uncased-", "")
                distilbert_data.append({
                    'config': config,
                    'train_time': data['summary']['train_wall_time_sec'],
                    'train_memory': data['summary']['train_peak_mem_bytes'] / (1024**3),
                    'val_memory': data['summary']['val_peak_mem_bytes'] / (1024**3),
                    'test_memory': data['summary']['test_peak_mem_bytes'] / (1024**3)
                })
    
    return bert_data, distilbert_data

def plot_comparison(bert_data, distilbert_data, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    bert_dict = {d['config']: d for d in bert_data}
    distilbert_dict = {d['config']: d for d in distilbert_data}
    
    target_config = 'lr2e-5'
    
    if target_config not in bert_dict:
        raise ValueError(f"BERT configuration '{target_config}' not found")
    if target_config not in distilbert_dict:
        raise ValueError(f"DistilBERT configuration '{target_config}' not found")
    
    bert_times = [bert_dict[target_config]['train_time'] / 60]
    distilbert_times = [distilbert_dict[target_config]['train_time'] / 60]
    
    bert_train_mem = [bert_dict[target_config]['train_memory']]
    distilbert_train_mem = [distilbert_dict[target_config]['train_memory']]
    
    x = np.arange(1)
    width = 0.35
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.bar(x - width/2, bert_times, width, label='BERT', alpha=0.8)
    ax1.bar(x + width/2, distilbert_times, width, label='DistilBERT', alpha=0.8)
    ax1.set_ylabel('Training Time (minutes)', fontsize=12)
    ax1.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['batch size = 16, learning rate = 2e-5'])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    ax2.bar(x - width/2, bert_train_mem, width, label='BERT', alpha=0.8)
    ax2.bar(x + width/2, distilbert_train_mem, width, label='DistilBERT', alpha=0.8)
    ax2.set_ylabel('Peak Memory Usage (GB)', fontsize=12)
    ax2.set_title('Peak Memory Usage Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['batch size = 16, learning rate = 2e-5'])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_file = output_path / 'training_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {output_file}")
    
    plt.close()

if __name__ == "__main__":
    result_dir = Path(__file__).parent / "result"
    output_dir = result_dir / "plots"
    
    bert_data, distilbert_data = load_results(result_dir)
    
    print(f"Loaded {len(bert_data)} BERT configurations")
    print(f"Loaded {len(distilbert_data)} DistilBERT configurations")
    
    plot_comparison(bert_data, distilbert_data, output_dir)

