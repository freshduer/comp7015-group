# BERT and DistilBERT Fine-tuning for IMDB Sentiment Analysis

This directory contains scripts and utilities for fine-tuning BERT and DistilBERT models on the IMDB movie review sentiment analysis task.

## Directory Structure

```
bert/
├── bert.py                      # Main script for fine-tuning BERT
├── distilbert.py                # Main script for fine-tuning DistilBERT
├── data_utils.py                # Data loading and preprocessing utilities
├── training_utils.py            # Training configuration and utilities
├── test_inference_performance.py  # Script to test inference performance
├── visualize_attention.py       # Script to visualize attention weights
├── plot_f1_curves.py            # Script to plot F1 score curves
├── plot_training_comparison.py  # Script to compare training metrics
├── config/
│   ├── bert.json                # BERT training configuration
│   └── distilbert.json          # DistilBERT training configuration
├── models/                      # Directory for saved models
│   ├── bert-imdb/              # BERT model checkpoints and best model
│   └── distilbert-imdb/        # DistilBERT model checkpoints and best model
├── result/                      # Training results and visualizations
│   ├── *.json                  # Training metrics JSON files
│   ├── plots/                  # Generated plots and figures
│   └── latex/                  # LaTeX tables for results
├── logs/                        # Training logs (stdout/stderr)
└── slurm/                       # SLURM job submission scripts
    ├── bert.sbatch             # SLURM script for BERT training
    ├── distilbert.sbatch       # SLURM script for DistilBERT training
    └── submit_all.sh           # Script to submit all training jobs
```

## Scripts Usage

### Training Scripts

#### 1. BERT Fine-tuning

Train a BERT model on IMDB dataset:

```bash
python bert.py [--config CONFIG_PATH]
```

- `--config`: (Optional) Path to JSON configuration file. If not specified, uses `config/bert.json` by default.

Example:
```bash
python bert.py
python bert.py --config config/bert.json
```

#### 2. DistilBERT Fine-tuning

Train a DistilBERT model on IMDB dataset:

```bash
python distilbert.py [--config CONFIG_PATH]
```

- `--config`: (Optional) Path to JSON configuration file. If not specified, uses `config/distilbert.json` by default.

Example:
```bash
python distilbert.py
python distilbert.py --config config/distilbert.json
```

### Configuration Files

Configuration files are JSON files that define training parameters:

```json
{
  "model_name": "bert-base-uncased",
  "output_dir": "models/bert-imdb",
  "num_epochs": 3,
  "batch_size": 16,
  "max_length": 256,
  "learning_rate": 5e-5,
  "seed": 42
}
```

Available parameters:
- `model_name`: HuggingFace model identifier
- `output_dir`: Directory to save model checkpoints
- `num_epochs`: Number of training epochs
- `batch_size`: Training batch size
- `max_length`: Maximum sequence length for tokenization
- `learning_rate`: Learning rate for optimizer
- `warmup_ratio`: (Optional) Warmup ratio for learning rate scheduler
- `lr_scheduler_type`: (Optional) Learning rate scheduler type
- `seed`: Random seed for reproducibility

### Analysis and Visualization Scripts

#### 3. Test Inference Performance

Measure inference time and memory usage:

```bash
python test_inference_performance.py
```

This script:
- Loads trained models from `models/` directory
- Measures inference time on test dataset
- Tracks GPU memory usage
- Generates comparison plots saved to `result/plots/inference_comparison.png`

#### 4. Visualize Attention Weights

Generate attention heatmaps for specific samples:

```bash
python visualize_attention.py
```

This script:
- Loads a trained model
- Extracts attention weights from specified layers and heads
- Generates heatmap visualizations
- Saves plots to `result/plots/attention_heatmap_*.png`

#### 5. Plot F1 Score Curves

Generate F1 score curves for different learning rate configurations:

```bash
python plot_f1_curves.py
```

This script:
- Reads training results from `result/*.json` files
- Plots F1 score curves for different learning rates
- Generates separate plots for BERT and DistilBERT
- Saves plots to `result/plots/bert-base-uncased-f1.png` and `result/plots/distilbert-base-uncased-f1.png`

#### 6. Plot Training Comparison

Compare training metrics between BERT and DistilBERT:

```bash
python plot_training_comparison.py
```

This script:
- Compares training time and memory usage
- Generates comparison visualizations
- Saves plots to `result/plots/training_comparison.png`

### SLURM Job Submission

For cluster environments with SLURM:

#### Submit BERT training job:
```bash
cd slurm
sbatch bert.sbatch
```

#### Submit DistilBERT training job:
```bash
cd slurm
sbatch distilbert.sbatch
```

#### Submit all training jobs:
```bash
cd slurm
bash submit_all.sh
```

## Data Requirements

The scripts expect IMDB dataset to be available in the project root directory structure:
- `data/aclImdb/train/pos/` - Positive training samples
- `data/aclImdb/train/neg/` - Negative training samples
- `data/aclImdb/test/pos/` - Positive test samples
- `data/aclImdb/test/neg/` - Negative test samples

The `data_utils.py` module automatically handles data loading and splitting into train/validation/test sets.

## Output Files

### Model Checkpoints
- Saved in `models/{model_name}-imdb/`
- `best/` directory contains the best model based on validation metrics
- `checkpoint-*/` directories contain intermediate checkpoints

### Training Results
- JSON files in `result/` directory contain detailed training metrics
- Format: `{model_name}-{config}.json`
- Includes per-epoch metrics and summary statistics

### Visualizations
- All plots are saved in `result/plots/`
- Includes F1 curves, attention heatmaps, and comparison charts

## Dependencies

Required Python packages:
- `torch`
- `transformers`
- `datasets`
- `scikit-learn`
- `matplotlib`
- `numpy`

Ensure these are installed in your environment before running the scripts.

