# LSTM Baseline for IMDb Sentiment Analysis

This directory contains the implementation of a Bidirectional LSTM with Attention mechanism for sentiment analysis on the IMDb dataset. It includes scripts for data preprocessing, model training, and comprehensive hyperparameter tuning.

## Project Structure

- `main.py`: The entry point for training and experimentation. It runs a suite of experiments with different hyperparameters.
- `lstm.py`: Defines the `LSTMModel` architecture (BiLSTM + Attention).
- `preprocess.py`: Handles data loading, cleaning, tokenization, and saving processed data.
- `analyze_results.py`: Generates plots and summary tables from the experiment results.
- `results/`: Stores the training logs (`.json`) and best model checkpoints (`.pt`).

## Prerequisites

Ensure you have the following dependencies installed:

```bash
pip install torch numpy scikit-learn
```

## Setup & Data Preparation

Before running the training script, you need to prepare the data and embeddings.

1.  **GloVe Embeddings**:
    Download the GloVe embeddings (6B tokens, 100d) and place `glove.6B.100d.txt` in the `../data/` directory.
    ```
    cd ../data && wget http://nlp.stanford.edu/data/glove.6B.zip && unzip glove.6B.zip
    ../data/glove.6B.100d.txt
    ```

2.  **IMDb Dataset**:
    Ensure the raw IMDb dataset is available. The preprocessing script expects the standard ACL IMDb folder structure.

3.  **Run Preprocessing**:
    Execute the preprocessing script to clean the data, build the vocabulary, and save the processed tensors.
    ```bash
    python preprocess.py
    ```
    This will create `aclImdb_prepared.npz` and `vocab.json` in `../data/processed/`.

## Running Experiments

To reproduce the baseline results and run the hyperparameter search:

```bash
python main.py
```

This script will:
1.  Load the processed data.
2.  Load GloVe embeddings (if available).
3.  Iterate through a predefined list of hyperparameter variations (Learning Rate, Batch Size, Hidden Dimension, Dropout, Layers).
4.  Train each configuration for 20 epochs.
5.  Save the best model (based on validation loss) and training history to the `results/` directory.

**Note:** The script supports resuming. If a result JSON file for a specific configuration already exists in `results/`, that experiment will be skipped.

## Results & Analysis

The `results/` directory will contain two files for each experiment:

-   `{config_name}_results.json`: Contains the full training history (loss, accuracy, precision, recall, f1) for every epoch, the final test set metrics, and the training duration.
-   `{config_name}_best_model.pt`: The state dictionary of the model with the lowest validation loss.

### JSON Structure
```json
{
    "config": { ... },          // Hyperparameters used
    "training_time": 123.45,    // Total training time in seconds
    "history": { ... },         // Metrics per epoch
    "test_results": {           // Final evaluation on test set
        "loss": 0.275,
        "accuracy": 0.886,
        "f1": 0.889,
        ...
    }
}
```

You can analyze these JSON files to compare the impact of different hyperparameters (e.g., GloVe vs. Random embeddings, Batch Size sensitivity).

### Generating Plots

To automatically generate visualization plots and a summary table of all experiments:

```bash
python analyze_results.py
```

This will create a `plots/` directory containing:
-   `embedding_comparison.png`: Bar chart comparing GloVe vs. Random embeddings.
-   `impact_*.png`: Line charts showing the effect of Learning Rate, Batch Size, Dropout, etc.
-   `summary_table.csv`: A CSV file summarizing the metrics for all configurations.
