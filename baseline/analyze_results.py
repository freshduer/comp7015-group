import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import glob
import re

RESULTS_DIR = 'results'
PLOTS_DIR = 'plots'

# Base Configuration
BASE_CONFIG = {
    'batch_size': 1024,
    'hidden_dim': 256,
    'n_layers': 2,
    'dropout': 0.5,
    'lr': 0.001
}

def parse_filename(filename):
    basename = os.path.basename(filename)
    name_no_ext = basename.replace('_results.json', '')
    
    parts = name_no_ext.split('_')
    embedding = parts[0] # 'glove' or 'random'
    
    if 'base' in parts:
        return embedding, 'base', None
    
    # Handle specific parameters
    # Patterns: {embedding}_{param}_{value}
    # But some params might have underscores? No, based on file list they are simple.
    # bs, dropout, hidden, layers, lr
    
    param_map = {
        'bs': 'batch_size',
        'dropout': 'dropout',
        'hidden': 'hidden_dim',
        'layers': 'n_layers',
        'lr': 'lr'
    }
    
    for key, param_name in param_map.items():
        if key in parts:
            # Find the index of the key
            idx = parts.index(key)
            value_str = parts[idx+1]
            
            # Convert value
            try:
                if param_name in ['batch_size', 'hidden_dim', 'n_layers']:
                    value = int(value_str)
                else:
                    value = float(value_str)
            except ValueError:
                # Handle scientific notation for LR if needed, though float() handles 1e-4
                value = float(value_str)
                
            return embedding, param_name, value
            
    return embedding, 'unknown', None

def load_results():
    data = []
    files = glob.glob(os.path.join(RESULTS_DIR, '*_results.json'))
    
    for f in files:
        try:
            with open(f, 'r') as json_file:
                content = json.load(json_file)
                
            embedding, param_name, param_value = parse_filename(f)
            
            # Extract metrics
            test_res = content.get('test_results', {})
            acc = test_res.get('accuracy')
            f1 = test_res.get('f1')
            
            if acc is None:
                continue
                
            entry = {
                'embedding': embedding,
                'param_name': param_name,
                'param_value': param_value,
                'accuracy': acc,
                'f1': f1,
                'filename': os.path.basename(f)
            }
            data.append(entry)
        except Exception as e:
            print(f"Error processing {f}: {e}")
            
    return pd.DataFrame(data)

def plot_hyperparam(df, param_name, x_label, title, log_scale=False):
    plt.figure(figsize=(10, 6))
    
    # Filter for the specific parameter AND the base model
    # The base model counts as a data point for ALL parameters
    
    embeddings = ['random', 'glove']
    colors = {'random': 'blue', 'glove': 'orange'}
    
    for emb in embeddings:
        # Get specific experiments for this param
        subset = df[(df['embedding'] == emb) & (df['param_name'] == param_name)].copy()
        
        # Get base experiment
        base = df[(df['embedding'] == emb) & (df['param_name'] == 'base')].copy()
        if not base.empty:
            base['param_value'] = BASE_CONFIG[param_name]
            subset = pd.concat([subset, base])
        
        # Sort by parameter value
        subset = subset.sort_values('param_value')
        
        plt.plot(subset['param_value'], subset['accuracy'], marker='o', label=f'{emb.capitalize()} Embeddings', color=colors[emb])
        
        # Annotate points
        for i, row in subset.iterrows():
            plt.annotate(f"{row['accuracy']:.4f}", (row['param_value'], row['accuracy']), 
                         textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    if log_scale:
        plt.xscale('log')
        
    plt.xlabel(x_label)
    plt.ylabel('Test Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    output_path = os.path.join(PLOTS_DIR, f'impact_{param_name}.png')
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
    plt.close()

def plot_embedding_comparison(df):
    # Compare base models
    base_df = df[df['param_name'] == 'base']
    
    if base_df.empty:
        print("No base results found for comparison.")
        return

    plt.figure(figsize=(8, 6))
    bars = plt.bar(base_df['embedding'], base_df['accuracy'], color=['blue', 'orange'])
    
    plt.xlabel('Embedding Type')
    plt.ylabel('Test Accuracy')
    plt.title('GloVe vs. Random Embeddings (Base Configuration)')
    plt.ylim(0.8, 0.95) # Zoom in a bit as values are likely high
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom')
                 
    output_path = os.path.join(PLOTS_DIR, 'embedding_comparison.png')
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
    plt.close()

def main():
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
        
    df = load_results()
    print(f"Loaded {len(df)} result files.")
    
    # 1. Embedding Comparison
    plot_embedding_comparison(df)
    
    # 2. Hyperparameter Impacts
    # Learning Rate
    plot_hyperparam(df, 'lr', 'Learning Rate', 'Impact of Learning Rate on Accuracy', log_scale=True)
    
    # Batch Size
    plot_hyperparam(df, 'batch_size', 'Batch Size', 'Impact of Batch Size on Accuracy')
    
    # Dropout
    plot_hyperparam(df, 'dropout', 'Dropout Rate', 'Impact of Dropout on Accuracy')
    
    # Hidden Dim
    plot_hyperparam(df, 'hidden_dim', 'Hidden Dimension', 'Impact of Hidden Dimension on Accuracy')
    
    # Layers
    plot_hyperparam(df, 'n_layers', 'Number of Layers', 'Impact of Network Depth on Accuracy')

    # Generate a summary CSV table
    summary_path = os.path.join(PLOTS_DIR, 'summary_table.csv')
    df.sort_values(['embedding', 'param_name', 'param_value']).to_csv(summary_path, index=False)
    print(f"Saved summary table to {summary_path}")

if __name__ == "__main__":
    main()
