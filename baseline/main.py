import os
import re
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import glob
import json
import time
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from lstm import LSTMModel

# Set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Configuration
PROCESSED_DATA_DIR = '../data/processed'
BATCH_SIZE = 1024
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
DROPOUT = 0.5
LR = 0.001
N_EPOCHS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = 'results'

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def load_processed_data():
    print("Loading processed data...")
    npz_path = os.path.join(PROCESSED_DATA_DIR, 'aclImdb_prepared.npz')
    vocab_path = os.path.join(PROCESSED_DATA_DIR, 'vocab.json')
    
    if not os.path.exists(npz_path) or not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Processed data not found at {PROCESSED_DATA_DIR}. Please run preprocess.py first.")
        
    with np.load(npz_path) as npz:
        X_train = torch.from_numpy(npz['X_train']).long()
        y_train = torch.from_numpy(npz['y_train']).float()
        X_val = torch.from_numpy(npz['X_val']).long()
        y_val = torch.from_numpy(npz['y_val']).float()
        X_test = torch.from_numpy(npz['X_test']).long()
        y_test = torch.from_numpy(npz['y_test']).float()
        
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)['word2idx']
        
    train_data = TensorDataset(X_train, y_train)
    val_data = TensorDataset(X_val, y_val)
    test_data = TensorDataset(X_test, y_test)
    
    return train_data, val_data, test_data, vocab

def calculate_metrics(preds, y):
    # Apply sigmoid and round to get binary predictions
    rounded_preds = torch.round(torch.sigmoid(preds))
    
    # Move to CPU for sklearn
    rounded_preds = rounded_preds.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    
    acc = accuracy_score(y, rounded_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y, rounded_preds, average='binary', zero_division=0)
    return acc, precision, recall, f1

def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    
    all_preds = []
    all_labels = []
    
    for batch in iterator:
        optimizer.zero_grad()
        text, label = batch
        text, label = text.to(DEVICE), label.to(DEVICE)
        
        predictions = model(text).squeeze(1)
        loss = criterion(predictions, label)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        all_preds.append(predictions)
        all_labels.append(label)
        
    # Concatenate all predictions and labels for the epoch
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    acc, prec, rec, f1 = calculate_metrics(all_preds, all_labels)
        
    return epoch_loss / len(iterator), acc, prec, rec, f1

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in iterator:
            text, label = batch
            text, label = text.to(DEVICE), label.to(DEVICE)
            
            predictions = model(text).squeeze(1)
            loss = criterion(predictions, label)
            
            epoch_loss += loss.item()
            
            all_preds.append(predictions)
            all_labels.append(label)
            
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    acc, prec, rec, f1 = calculate_metrics(all_preds, all_labels)
            
    return epoch_loss / len(iterator), acc, prec, rec, f1

def run_experiment(exp_name, model, train_loader, val_loader, test_loader, criterion, optimizer, n_epochs, hyperparams=None):
    print(f"\nStarting Experiment: {exp_name}")
    best_valid_loss = float('inf')
    history = {
        'train_loss': [], 'train_acc': [], 'train_prec': [], 'train_rec': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_prec': [], 'val_rec': [], 'val_f1': []
    }
    
    start_time = time.time()
    
    for epoch in range(n_epochs):
        train_loss, train_acc, train_prec, train_rec, train_f1 = train(model, train_loader, optimizer, criterion)
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, criterion)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_prec'].append(train_prec)
        history['train_rec'].append(train_rec)
        history['train_f1'].append(train_f1)
        
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_prec'].append(val_prec)
        history['val_rec'].append(val_rec)
        history['val_f1'].append(val_f1)
        
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f'{exp_name}_best_model.pt'))
        
        print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.3f} | Val Acc: {val_acc*100:.2f}%')
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training finished in {training_time:.2f}s")
    
    # Load best model for testing
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, f'{exp_name}_best_model.pt')))
    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_loader, criterion)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% | Test F1: {test_f1:.4f}')
    
    results = {
        'config': {
            'embedding_dim': model.embedding.embedding_dim,
            'hidden_dim': model.lstm.hidden_size,
            'n_layers': model.lstm.num_layers,
            'dropout': model.lstm.dropout,
            'bidirectional': model.lstm.bidirectional
        },
        'training_time': training_time,
        'history': history,
        'test_results': {
            'loss': test_loss,
            'accuracy': test_acc,
            'precision': test_prec,
            'recall': test_rec,
            'f1': test_f1
        }
    }
    
    if hyperparams:
        results['config'].update(hyperparams)
    
    with open(os.path.join(RESULTS_DIR, f'{exp_name}_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
    return results

def load_glove_embeddings(vocab, glove_path):
    print(f"Loading GloVe embeddings from {glove_path}...")
    embeddings_index = {}
    try:
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
    except FileNotFoundError:
        print("GloVe file not found. Using random embeddings instead.")
        return None

    embedding_matrix = torch.zeros(len(vocab), EMBEDDING_DIM)
    found = 0
    for word, i in vocab.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = torch.from_numpy(embedding_vector)
            found += 1
        else:
            # Initialize random for OOV
            embedding_matrix[i] = torch.randn(EMBEDDING_DIM)
            
    print(f"Found {found}/{len(vocab)} words in GloVe.")
    return embedding_matrix

def main():
    train_data, val_data, test_data, vocab = load_processed_data()
    
    # Load GloVe
    glove_path = '../data/glove.6B.100d.txt' 
    glove_embeddings = load_glove_embeddings(vocab, glove_path)
    
    # Base Configuration
    base_config = {
        'lr': 0.001,
        'batch_size': 1024,
        'dropout': 0.5,
        'hidden_dim': 256,
        'n_layers': 2
    }
    
    # Variations to explore
    variations = [
        ('base', {}),
        # Learning Rate
        ('lr_5e-5', {'lr': 0.00005}),
        ('lr_1e-4', {'lr': 0.0001}),
        ('lr_5e-4', {'lr': 0.0005}),
        ('lr_5e-3', {'lr': 0.005}),
        ('lr_1e-2', {'lr': 0.01}),
        # Batch Size
        ('bs_64', {'batch_size': 64}),
        ('bs_128', {'batch_size': 128}),
        ('bs_256', {'batch_size': 256}),
        ('bs_512', {'batch_size': 512}),
        # Dropout
        ('dropout_0.0', {'dropout': 0.0}),
        ('dropout_0.2', {'dropout': 0.2}),
        ('dropout_0.3', {'dropout': 0.3}),
        ('dropout_0.7', {'dropout': 0.7}),
        ('dropout_0.8', {'dropout': 0.8}),
        # Hidden Dim
        ('hidden_64', {'hidden_dim': 64}),
        ('hidden_128', {'hidden_dim': 128}),
        ('hidden_512', {'hidden_dim': 512}),
        # ('hidden_1024', {'hidden_dim': 1024}), # OOM Risk
        # Layers
        ('layers_1', {'n_layers': 1}),
        ('layers_3', {'n_layers': 3}),
        ('layers_4', {'n_layers': 4})
    ]
    
    embedding_schemes = [
        ('random', None),
        ('glove', glove_embeddings)
    ]
    
    for emb_name, emb_weights in embedding_schemes:
        print(f"\n{'='*40}")
        print(f"Running Experiments for Embedding Scheme: {emb_name}")
        print(f"{'='*40}")
        
        for var_name, changes in variations:
            # Prepare config
            config = base_config.copy()
            config.update(changes)
            
            exp_id = f"{emb_name}_{var_name}"
            
            # Check if experiment already exists
            if os.path.exists(os.path.join(RESULTS_DIR, f'{exp_id}_results.json')):
                print(f"Skipping {exp_id}, results already exist.")
                continue

            print(f"\n--- Starting Experiment: {exp_id} ---")
            print(f"Config: {config}")
            
            # Prepare DataLoaders (Batch size might change)
            train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, num_workers=8, pin_memory=True)
            val_loader = DataLoader(val_data, batch_size=config['batch_size'], num_workers=8, pin_memory=True)
            test_loader = DataLoader(test_data, batch_size=config['batch_size'], num_workers=8, pin_memory=True)
            
            # Initialize Model
            model = LSTMModel(
                vocab_size=len(vocab),
                embedding_dim=EMBEDDING_DIM,
                hidden_dim=config['hidden_dim'],
                output_dim=OUTPUT_DIM,
                n_layers=config['n_layers'],
                dropout=config['dropout'],
                pad_idx=vocab['<pad>'],
                pretrained_embeddings=emb_weights,
                bidirectional=True,
                use_attention=True
            ).to(DEVICE)
            
            optimizer = optim.Adam(model.parameters(), lr=config['lr'])
            criterion = nn.BCEWithLogitsLoss().to(DEVICE)
            
            # Run Training
            try:
                run_experiment(exp_id, model, train_loader, val_loader, test_loader, criterion, optimizer, N_EPOCHS, hyperparams=config)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"WARNING: Experiment {exp_id} failed due to CUDA OOM. Skipping...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    raise e

if __name__ == '__main__':
    main()
