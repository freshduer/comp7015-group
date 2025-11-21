import os
import re
import json
from pathlib import Path
from collections import Counter
from typing import List, Tuple
import numpy as np
from sklearn.model_selection import train_test_split

# Configuration
SEED = 42
np.random.seed(SEED)
MIN_FREQ = 2
MAX_LEN_LIMIT = 256
DATA_DIR = Path('../data/aclImdb')
OUTPUT_DIR = Path('../data/processed')

def clean_text(s: str) -> str:
    TAG_RE = re.compile(r'<[^>]+>')
    URL_RE = re.compile(r'https?://\S+|www\.\S+')
    s = TAG_RE.sub(' ', s)
    s = URL_RE.sub(' http ', s)
    s = s.replace('\n', ' ')
    s = s.lower()
    return s

def tokenize(s: str) -> List[str]:
    TOKEN_RE = re.compile(r"[a-zA-Z']+")
    return TOKEN_RE.findall(s)

def load_reviews(dir_path: Path, label: int) -> Tuple[List[str], List[int]]:
    texts, labels = [], []
    if not dir_path.exists():
        print(f"Warning: {dir_path} does not exist.")
        return [], []
    for fp in sorted(dir_path.glob('*.txt')):
        try:
            text = fp.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            text = fp.read_text(errors='ignore')
        texts.append(text)
        labels.append(label)
    return texts, labels

def build_vocab(texts: List[str], min_freq: int = 2):
    counter = Counter()
    for t in texts:
        toks = tokenize(clean_text(t))
        counter.update(toks)
    
    SPECIALS = ['<pad>', '<unk>']
    words = [w for w, c in counter.items() if c >= min_freq]
    words.sort(key=lambda w: (-counter[w], w))
    
    word2idx = {w: i for i, w in enumerate(SPECIALS)}
    for w in words:
        if w not in word2idx:
            word2idx[w] = len(word2idx)
            
    idx2word = [None] * len(word2idx)
    for w, i in word2idx.items():
        idx2word[i] = w
        
    return word2idx, idx2word, counter

def to_ids(text: str, w2i: dict, unk_idx: int) -> List[int]:
    toks = tokenize(clean_text(text))
    return [w2i.get(tok, unk_idx) for tok in toks]

def pad_sequence(seq: List[int], max_len: int, pad_idx: int) -> List[int]:
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [pad_idx] * (max_len - len(seq))

def main():
    print(f"Checking data directory: {DATA_DIR.resolve()}")
    if not DATA_DIR.exists():
        # Try fallback or error
        print(f"Error: {DATA_DIR} not found.")
        return

    print("Loading raw data...")
    train_pos = DATA_DIR / 'train' / 'pos'
    train_neg = DATA_DIR / 'train' / 'neg'
    test_pos = DATA_DIR / 'test' / 'pos'
    test_neg = DATA_DIR / 'test' / 'neg'
    
    # Load Train Data
    pos_train_texts, pos_train_labels = load_reviews(train_pos, 1)
    neg_train_texts, neg_train_labels = load_reviews(train_neg, 0)
    X_train_all = pos_train_texts + neg_train_texts
    y_train_all = pos_train_labels + neg_train_labels
    
    # Load Test Data
    pos_test_texts, pos_test_labels = load_reviews(test_pos, 1)
    neg_test_texts, neg_test_labels = load_reviews(test_neg, 0)
    X_test = pos_test_texts + neg_test_texts
    y_test = pos_test_labels + neg_test_labels
    
    print(f'Total Train samples: {len(X_train_all)}')
    print(f'Total Test samples: {len(X_test)}')

    # Split Train into Train/Val (80/20)
    print("Splitting train data into train/val (80/20)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all, y_train_all, test_size=0.2, random_state=SEED, stratify=y_train_all
    )
    print(f'Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}')

    # Build Vocab
    print("Building vocabulary...")
    word2idx, idx2word, counter = build_vocab(X_train, MIN_FREQ)
    print(f'Vocab size: {len(word2idx)}')

    # Numericalize
    print("Numericalizing...")
    PAD = word2idx['<pad>']
    UNK = word2idx['<unk>']
    
    train_ids = [to_ids(t, word2idx, UNK) for t in X_train]
    val_ids = [to_ids(t, word2idx, UNK) for t in X_val]
    test_ids = [to_ids(t, word2idx, UNK) for t in X_test]

    # Determine Max Len
    p95 = int(np.percentile([len(x) for x in train_ids], 95))
    MAX_LEN = min(MAX_LEN_LIMIT, max(16, p95))
    print(f'MAX_LEN determined: {MAX_LEN}')

    # Pad
    print("Padding...")
    X_train_pad = np.array([pad_sequence(s, MAX_LEN, PAD) for s in train_ids], dtype=np.int32)
    X_val_pad = np.array([pad_sequence(s, MAX_LEN, PAD) for s in val_ids], dtype=np.int32)
    X_test_pad = np.array([pad_sequence(s, MAX_LEN, PAD) for s in test_ids], dtype=np.int32)
    
    y_train_a = np.array(y_train, dtype=np.int64)
    y_val_a = np.array(y_val, dtype=np.int64)
    y_test_a = np.array(y_test, dtype=np.int64)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    vocab_path = OUTPUT_DIR / 'vocab.json'
    with vocab_path.open('w', encoding='utf-8') as f:
        json.dump({'word2idx': word2idx, 'idx2word': idx2word}, f, ensure_ascii=False)
        
    npz_path = OUTPUT_DIR / 'aclImdb_prepared.npz'
    np.savez_compressed(
        npz_path,
        X_train=X_train_pad, y_train=y_train_a,
        X_val=X_val_pad, y_val=y_val_a,
        X_test=X_test_pad, y_test=y_test_a,
        max_len=np.array([MAX_LEN], dtype=np.int32),
        pad_idx=np.array([PAD], dtype=np.int32),
        unk_idx=np.array([UNK], dtype=np.int32)
    )
    
    print(f"Saved processed data to {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
