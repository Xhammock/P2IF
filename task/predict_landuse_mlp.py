#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Land-use (functional zone) classification with a shallow MLP.
HGI-style: 512-d hidden layer (BatchNorm1d + tanh) + softmax output.

Predicts classes 0, 1, 4, 5 only (classes 2 and 3 omitted due to skew).
"""

import argparse
import os
import warnings
from typing import List, Optional, Tuple, Dict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from tqdm import tqdm
import json

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
CONFIG = {
    # File paths
    'embeddings_path': 'checkpoints/train_20260106_105040/best_embeddings.npz',
    'landuse_data_path': None,  # Optional .npy + shapefile spatial alignment
    'landuse_csv_path': 'data/landuse_aligned_grid.csv',  # Prefer CSV when available
    'shp_path': 'data/SZ_clip/SZ_clip.shp',  # Shapefile for spatial alignment
    'output_dir': None,  # Auto-created with timestamp if None

    # Embedding
    'embedding_key': 'h',  # e.g. h, concat, z, h_spatial, h_od, embeddings
    'concat_keys': ['h_spatial', 'h_od'],  # Used when embedding_key is h or concat

    # Data
    'valid_classes': [0, 1, 2, 3, 4, 5],
    'test_size': 0.0,  # 0 = train on all data, no held-out test split
    'use_full_data': True,  # Train and validate on the same full dataset
    'random_state': 42,

    # Model (HGI-style)
    'hidden_dim': 512,
    'use_batch_norm': True,
    'activation': 'tanh',
    'dropout': 0.0,

    # Training
    'batch_size': 128,
    'num_epochs': 150,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'early_stopping_patience': 50,
    'use_class_weight': False,
    'label_smoothing': 0.1,
    'gradient_clip': 1.0,  # None to disable
    'warmup_epochs': 10,
    'save_predictions': True,

    # Device (detected in main)
    'device': None,
}
# ============================================================================


# ============================================================================
# Model
# ============================================================================

class LanduseMLP(nn.Module):
    """Shallow MLP classifier (HGI-style).

    Architecture:
    - Input -> 512-d hidden (BatchNorm1d + tanh) -> output (softmax via CrossEntropyLoss)
    """

    def __init__(self, input_dim: int, num_classes: int = 4,
                 hidden_dim: int = 512, use_batch_norm: bool = True,
                 activation: str = 'tanh', dropout: float = 0.0):
        super().__init__()
        self.num_classes = num_classes

        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(
            hidden_dim) if use_batch_norm else nn.Identity()

        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Output (softmax applied inside CrossEntropyLoss)
        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.output(x)
        return x


# ============================================================================
# Dataset
# ============================================================================

class LanduseDataset(Dataset):
    """Land-use classification dataset"""

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return {
            'embedding': self.embeddings[idx],
            'label': self.labels[idx]
        }


# ============================================================================
# Data loading
# ============================================================================

def _concat_embeddings(data: "np.lib.npyio.NpzFile", keys: List[str]) -> np.ndarray:
    """Concatenate multiple embedding matrices."""
    arrays = []
    n_rows = None
    for k in keys:
        if k not in getattr(data, "files", []):
            raise KeyError(f"key='{k}' not found in npz")
        arr = data[k]
        if not isinstance(arr, np.ndarray) or arr.ndim != 2:
            raise ValueError(f"Data for key='{k}' is not a 2D ndarray")
        if n_rows is None:
            n_rows = arr.shape[0]
        elif arr.shape[0] != n_rows:
            raise ValueError(f"Concatenation failed: row count mismatch for key='{k}'")
        arrays.append(arr)
    return np.concatenate(arrays, axis=1)


def load_embeddings(path: str, embedding_key: Optional[str] = None,
                    concat_keys: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load embeddings and node IDs."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embeddings file not found: {path}")

    data = np.load(path, allow_pickle=True)
    keys = list(getattr(data, "files", []))
    requested = (embedding_key or "z").strip()

    if requested.lower() in {"h", "concat"}:
        concat_keys = concat_keys or ["h_spatial", "h_od"]
        embeddings = _concat_embeddings(data, concat_keys)
        print(
            f"[Info] Using concatenated embeddings: concat({'+'.join(concat_keys)}), shape={embeddings.shape}")
    else:
        if requested in keys:
            embeddings = data[requested]
        else:
            if "embeddings" in keys:
                embeddings = data["embeddings"]
                print(f"[Warning] key='{requested}' not found; using 'embeddings'")
            else:
                candidates = [(data[k].shape[1], k) for k in keys
                              if isinstance(data[k], np.ndarray) and data[k].ndim == 2]
                if not candidates:
                    raise KeyError("Could not find a usable embedding matrix")
                embeddings = data[max(candidates)[1]]
                print(f"[Warning] key='{requested}' not found; using '{max(candidates)[1]}'")
        print(f"[Info] Using embedding key: '{requested}', shape={embeddings.shape}")

    node_ids = data.get('node_ids', np.arange(len(embeddings)))
    if node_ids.dtype.kind in {'S', 'O', 'U'}:
        node_ids = np.array([int(float(x)) if x else 0 for x in node_ids])
    else:
        node_ids = node_ids.astype(int)

    if len(node_ids) != len(embeddings):
        print(f"[Warning] node_ids length mismatch; using indices")
        node_ids = np.arange(len(embeddings), dtype=int)

    return node_ids, embeddings


def load_landuse_data(npy_path: Optional[str] = None, csv_path: Optional[str] = None,
                      node_ids: Optional[np.ndarray] = None,
                      shp_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load land-use data.

    With npy_path and shp_path: spatial alignment (recommended).
    Otherwise: simple index alignment (may be inaccurate).
    """
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if 'GWBH' not in df.columns or 'landuse' not in df.columns:
            raise ValueError("CSV must contain 'GWBH' and 'landuse' columns")
        df = df[df['GWBH'].notna() & df['landuse'].notna()]
        df['GWBH'] = pd.to_numeric(df['GWBH'], errors='coerce')
        df = df[df['GWBH'].notna()]
        df['GWBH'] = df['GWBH'].astype(int)
        df['landuse'] = df['landuse'].astype(int)
        return df
    elif npy_path and os.path.exists(npy_path):
        if shp_path and os.path.exists(shp_path):
            print("[Info] Loading land-use data with spatial alignment...")
            import geopandas as gpd

            landuse_array = np.load(npy_path, allow_pickle=True)
            if landuse_array.ndim > 1:
                landuse_array = landuse_array.flatten()

            gdf = gpd.read_file(shp_path)
            gdf['GWBH_1000'] = pd.to_numeric(gdf['GWBH_1000'], errors='coerce')
            gdf = gdf[gdf['GWBH_1000'].notna()].copy()
            gdf['GWBH_1000'] = gdf['GWBH_1000'].astype(int)
            gdf['GWBH'] = gdf['GWBH'].astype(int)

            unique_gwbh_1000 = sorted(gdf['GWBH_1000'].unique())

            min_len = min(len(landuse_array), len(unique_gwbh_1000))
            if len(landuse_array) != len(unique_gwbh_1000):
                print(
                    f"[Warning] Land-use length ({len(landuse_array)}) != unique GWBH_1000 count "
                    f"({len(unique_gwbh_1000)}); using first {min_len} values")
            gwbh_1000_to_landuse = dict(
                zip(unique_gwbh_1000[:min_len], landuse_array[:min_len]))

            gdf['landuse'] = gdf['GWBH_1000'].map(gwbh_1000_to_landuse)
            gdf_with_landuse = gdf[gdf['landuse'].notna()].copy()

            if node_ids is not None:
                target_gwbh_set = set(node_ids)
                gdf_with_landuse = gdf_with_landuse[gdf_with_landuse['GWBH'].isin(
                    target_gwbh_set)].copy()

            df = pd.DataFrame({
                'GWBH': gdf_with_landuse['GWBH'].values,
                'landuse': gdf_with_landuse['landuse'].astype(int).values
            })
            return df
        else:
            print("[Warning] shp_path not provided; using simple index alignment (may be inaccurate)")
            landuse_array = np.load(npy_path, allow_pickle=True)
            if landuse_array.ndim > 1:
                landuse_array = landuse_array.flatten()
            if node_ids is not None:
                min_len = min(len(node_ids), len(landuse_array))
                node_ids = node_ids[:min_len]
                landuse_array = landuse_array[:min_len]
                df = pd.DataFrame({'GWBH': node_ids, 'landuse': landuse_array})
            else:
                df = pd.DataFrame(
                    {'GWBH': np.arange(len(landuse_array)), 'landuse': landuse_array})
            return df
    else:
        raise FileNotFoundError("Provide landuse_data_path or landuse_csv_path")


def merge_data(node_ids: np.ndarray, embeddings: np.ndarray,
               landuse_df: pd.DataFrame, valid_classes: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, int]]:
    """Merge data and filter to valid classes.

    Returns:
        X: embeddings
        y: labels remapped to 0, 1, 2, ...
        gwbh: grid IDs (GWBH)
        class_mapping: original class -> new index
    """
    emb_df = pd.DataFrame(
        {'GWBH': node_ids, 'embedding_idx': range(len(node_ids))})

    merged = landuse_df.merge(emb_df, on='GWBH', how='inner')

    if len(merged) == 0:
        raise ValueError("No rows matched")

    valid_mask = merged['landuse'].isin(valid_classes)
    merged = merged[valid_mask]

    if len(merged) == 0:
        raise ValueError(f"No valid data after filtering (valid classes: {valid_classes})")

    print(f"[Info] Matched {len(merged)} rows (original land-use rows: {len(landuse_df)})")

    class_mapping = {orig_cls: new_idx for new_idx,
                     orig_cls in enumerate(sorted(valid_classes))}

    y = merged['landuse'].map(class_mapping).values
    X = embeddings[merged['embedding_idx'].values]
    gwbh = merged['GWBH'].values

    unique, counts = np.unique(y, return_counts=True)
    print("[Info] Class distribution (new label -> original class):")
    for new_label, count in zip(unique, counts):
        orig_cls = sorted(valid_classes)[new_label]
        print(f"  new label {new_label} (original class {orig_cls}): {count} samples")

    return X, y, gwbh, class_mapping


# ============================================================================
# Training and evaluation
# ============================================================================

def train_epoch(model: nn.Module, train_loader: DataLoader, criterion: nn.Module,
                optimizer: optim.Optimizer, device: str, gradient_clip: Optional[float] = None) -> Tuple[float, Dict]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    all_preds, all_labels = [], []

    for batch in tqdm(train_loader, desc="Training"):
        embeddings = batch['embedding'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(embeddings)
        loss = criterion(logits, labels)
        loss.backward()

        if gradient_clip is not None and gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted'),
        'f1_macro': f1_score(all_labels, all_preds, average='macro'),
    }

    return avg_loss, metrics


def validate(model: nn.Module, val_loader: DataLoader, criterion: nn.Module,
             device: str) -> Tuple[float, Dict]:
    """Validate."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_preds, all_labels = [], []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            embeddings = batch['embedding'].to(device)
            labels = batch['label'].to(device)

            logits = model(embeddings)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            n_batches += 1

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    report = classification_report(
        all_labels, all_preds, output_dict=True, zero_division=0)

    results = {
        'loss': avg_loss,
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted'),
        'f1_macro': f1_score(all_labels, all_preds, average='macro'),
        'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist(),
        'report': report,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
    }

    return avg_loss, results


def get_predictions(model: nn.Module, data_loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get predictions for a dataset.

    Returns:
        predictions: predicted labels
        labels: true labels
        probabilities: class probabilities
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Predicting"):
            embeddings = batch['embedding'].to(device)
            labels = batch['label'].to(device)

            logits = model(embeddings)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_labels), np.concatenate(all_probs)


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
          num_epochs: int, learning_rate: float, weight_decay: float,
          early_stopping_patience: int, device: str, class_weights: Optional[torch.Tensor] = None,
          label_smoothing: float = 0.0, output_dir: str = './results',
          gradient_clip: Optional[float] = None, warmup_epochs: int = 0,
          save_predictions: bool = False, valid_classes: Optional[List[int]] = None) -> Dict:
    """Train the model."""
    os.makedirs(output_dir, exist_ok=True)

    if class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=label_smoothing)
        print(f"[Info] Using class weights: {class_weights.cpu().numpy()}")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    if label_smoothing > 0:
        print(f"[Info] Using label_smoothing: {label_smoothing}")

    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    if warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs
        )
        print(f"[Info] LR warmup for {warmup_epochs} epochs")

    if gradient_clip is not None and gradient_clip > 0:
        print(f"[Info] Gradient clipping: max_norm={gradient_clip}")

    best_val_f1_macro = float('-inf')
    best_val_loss = float('inf')
    patience_counter = 0
    best_val_results = None

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_metrics': {},
        'val_metrics': {},
    }

    print("\n" + "=" * 60)
    print("Starting training")
    print("=" * 60)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, gradient_clip
        )

        val_loss, val_results = validate(model, val_loader, criterion, device)

        if warmup_epochs > 0 and epoch < warmup_epochs:
            warmup_scheduler.step()
            print(f"  Warmup LR: {optimizer.param_groups[0]['lr']:.6f}")
        else:
            scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        for key, value in train_metrics.items():
            if key not in history['train_metrics']:
                history['train_metrics'][key] = []
            history['train_metrics'][key].append(value)
        for key, value in val_results.items():
            if isinstance(value, np.ndarray):
                continue
            if key not in history['val_metrics']:
                history['val_metrics'][key] = []
            if key == 'confusion_matrix' and isinstance(value, list):
                history['val_metrics'][key].append(value)
            elif isinstance(value, (list, dict)):
                history['val_metrics'][key].append(value)
            else:
                history['val_metrics'][key].append(value)

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(
            f"Train Acc: {train_metrics['accuracy']:.4f}, Val Acc: {val_results['accuracy']:.4f}")
        print(
            f"Train F1-macro: {train_metrics['f1_macro']:.4f}, Val F1-macro: {val_results['f1_macro']:.4f}")

        if val_results['f1_macro'] > best_val_f1_macro:
            best_val_f1_macro = val_results['f1_macro']
            best_val_loss = val_loss
            patience_counter = 0
            best_val_results = val_results.copy()

            val_results_to_save = {k: v for k, v in val_results.items()
                                   if not isinstance(v, np.ndarray)}

            best_model_path = os.path.join(output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_f1_macro': val_results['f1_macro'],
                'val_results': val_results_to_save,
            }, best_model_path)
            print(f"Saved best model (Val F1-macro: {val_results['f1_macro']:.4f})")
        else:
            patience_counter += 1
            if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping (patience={early_stopping_patience})")
                break

    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    train_preds, val_preds = None, None
    if save_predictions:
        print("\n[Info] Loading best model for predictions...")
        best_model_path = os.path.join(output_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])

            train_preds_labels, train_true_labels, train_probs = get_predictions(
                model, train_loader, device)
            val_preds_labels, val_true_labels, val_probs = get_predictions(
                model, val_loader, device)

            train_preds = (train_preds_labels, train_true_labels, train_probs)
            val_preds = (val_preds_labels, val_true_labels, val_probs)

    if best_val_results is not None:
        best_val_results_clean = {k: v for k, v in best_val_results.items()
                                  if not isinstance(v, np.ndarray)}
    else:
        best_val_results_clean = None

    return {
        'history': history,
        'best_val_results': best_val_results_clean,
        'train_predictions': train_preds,
        'val_predictions': val_preds,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Land-use classification with MLP')
    parser.add_argument('--embeddings', type=str, default=None)
    parser.add_argument('--landuse_data', type=str, default=None)
    parser.add_argument('--landuse_csv', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--embedding_key', type=str, default=None)
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda, cuda:0, cuda:1, cpu, etc.); auto-detect if omitted')
    parser.add_argument('--use_full_data', action='store_true',
                        help='Train on all data (no test split; may overfit but best in-sample fit)')
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate')

    args = parser.parse_args()

    # Apply CLI overrides to CONFIG
    if args.use_full_data:
        CONFIG['use_full_data'] = True
    if args.num_epochs is not None:
        CONFIG['num_epochs'] = args.num_epochs
    if args.batch_size is not None:
        CONFIG['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        CONFIG['learning_rate'] = args.learning_rate

    # Device detection and setup
    print("\n" + "=" * 60)
    print("Device detection")
    print("=" * 60)

    if args.device:
        CONFIG['device'] = args.device
        print(f"[Info] Using user-specified device: {CONFIG['device']}")
    elif CONFIG['device'] is None:
        print(f"[Info] PyTorch version: {torch.__version__}")
        print(
            f"[Info] PyTorch CUDA build: {torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A'}")

        try:
            cuda_available = torch.cuda.is_available()
            print(f"[Info] torch.cuda.is_available(): {cuda_available}")

            if cuda_available:
                gpu_count = torch.cuda.device_count()
                print(f"[Info] Found {gpu_count} GPU(s)")

                try:
                    test_tensor = torch.tensor([1.0]).to('cuda:0')
                    result = test_tensor * 2
                    del test_tensor, result
                    torch.cuda.empty_cache()

                    CONFIG['device'] = 'cuda:0'
                    print(f"[Info] CUDA OK; using device: {CONFIG['device']}")
                    print(f"[Info] GPU name: {torch.cuda.get_device_name(0)}")
                    print(
                        f"[Info] GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
                except RuntimeError as e:
                    print(f"[Warning] CUDA device test failed: {e}")
                    print("[Warning] Driver/CUDA mismatch possible; falling back to CPU")
                    CONFIG['device'] = 'cpu'
            else:
                CONFIG['device'] = 'cpu'
                print("[Info] CUDA not available; using CPU")
                print("[Hint] If you have a GPU, try:")
                print("       1. Verify CUDA drivers are installed")
                print("       2. Install a PyTorch build matching your CUDA driver")
                print("       3. Run: python scripts/check_gpu.py for diagnostics")
        except Exception as e:
            CONFIG['device'] = 'cpu'
            print(f"[Warning] CUDA detection failed ({e}); training on CPU")
            print("[Hint] Run: python scripts/check_gpu.py for details")

    # Verify selected device works
    try:
        if 'cuda' in CONFIG['device']:
            device_id = int(CONFIG['device'].split(
                ':')[1]) if ':' in CONFIG['device'] else 0

            if not torch.cuda.is_available():
                print("[Warning] CUDA unavailable; switching to CPU")
                CONFIG['device'] = 'cpu'
            elif device_id >= torch.cuda.device_count():
                print(f"[Warning] Device {CONFIG['device']} not found; using cuda:0")
                CONFIG['device'] = 'cuda:0'
            else:
                test_tensor = torch.tensor([1.0]).to(CONFIG['device'])
                result = test_tensor * 2
                del test_tensor, result
                torch.cuda.empty_cache()
                print(f"[Info] Device {CONFIG['device']} verified")
    except Exception as e:
        print(f"[Warning] Device {CONFIG['device']} unavailable ({e}); switching to CPU")
        CONFIG['device'] = 'cpu'

    print("=" * 60)

    # Paths and options from CONFIG / CLI
    embeddings_path = args.embeddings or CONFIG['embeddings_path']
    landuse_data_path = args.landuse_data or CONFIG['landuse_data_path']
    landuse_csv_path = args.landuse_csv or CONFIG['landuse_csv_path']
    embedding_key = args.embedding_key or CONFIG['embedding_key']
    valid_classes = CONFIG['valid_classes']

    # Auto-generate output directory if not set
    if args.output_dir:
        output_dir = args.output_dir
    elif CONFIG['output_dir']:
        output_dir = CONFIG['output_dir']
    else:
        checkpoint_name = os.path.basename(os.path.dirname(embeddings_path))
        if checkpoint_name.startswith('train_'):
            base_dir = f'results/{checkpoint_name}'
        else:
            base_dir = 'results'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(base_dir, f'landuse_mlp_{timestamp}')

    print("=" * 60)
    print("Configuration:")
    print(f"  Embeddings path: {embeddings_path}")
    print(f"  Land-use data path: {landuse_data_path}")
    print(f"  Land-use CSV path: {landuse_csv_path}")
    print(f"  Output directory: {output_dir}")
    print(f"  Embedding key: {embedding_key}")
    print(f"  Valid classes: {valid_classes}")
    print(f"  Device: {CONFIG['device']}")
    print("=" * 60)

    print("\n[Info] Loading embeddings...")
    node_ids, embeddings = load_embeddings(
        embeddings_path, embedding_key, CONFIG['concat_keys'])

    print("[Info] Loading land-use data...")
    landuse_df = load_landuse_data(
        landuse_data_path, landuse_csv_path, node_ids, CONFIG.get('shp_path'))

    print("[Info] Merging data and filtering valid classes...")
    X, y, gwbh_all, class_mapping = merge_data(
        node_ids, embeddings, landuse_df, valid_classes)

    if CONFIG['use_full_data']:
        print("[Info] Training on all data (no test split)")
        X_train = X
        y_train = y
        gwbh_train = gwbh_all
        X_test = X
        y_test = y
        gwbh_test = gwbh_all
    elif CONFIG['test_size'] == 0:
        print("[Info] test_size=0: training on all data (no test split)")
        X_train = X
        y_train = y
        gwbh_train = gwbh_all
        X_test = X
        y_test = y
        gwbh_test = gwbh_all
    else:
        # Stratified split with fixed seed so indices stay aligned
        indices = np.arange(len(X))
        train_indices, test_indices = train_test_split(
            indices, test_size=CONFIG['test_size'], random_state=CONFIG['random_state'],
            stratify=y
        )
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        gwbh_train, gwbh_test = gwbh_all[train_indices], gwbh_all[test_indices]

    print("[Info] Standardizing embeddings...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    class_weights = None
    if CONFIG['use_class_weight']:
        class_counts = np.bincount(y_train, minlength=len(valid_classes))
        total_samples = class_counts.sum()
        class_weights = torch.FloatTensor(
            [total_samples / (len(class_counts) * count) if count > 0 else 0.0
             for count in class_counts]
        ).to(CONFIG['device'])
        print(f"[Info] Class weights: {class_weights.cpu().numpy()}")

    train_dataset = LanduseDataset(X_train, y_train)
    val_dataset = LanduseDataset(X_test, y_test)

    num_workers = 4 if 'cuda' in CONFIG['device'] else 0
    pin_memory = 'cuda' in CONFIG['device']

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    embedding_dim = X.shape[1]
    num_classes = len(valid_classes)
    model = LanduseMLP(
        embedding_dim, num_classes,
        hidden_dim=CONFIG['hidden_dim'],
        use_batch_norm=CONFIG['use_batch_norm'],
        activation=CONFIG['activation'],
        dropout=CONFIG['dropout']
    ).to(CONFIG['device'])

    print(f"\n[Info] Model:")
    print(f"  Input dim: {embedding_dim}")
    print(f"  Hidden dim: {CONFIG['hidden_dim']}")
    print(f"  Num classes: {num_classes}")
    print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Model device: {next(model.parameters()).device}")
    if 'cuda' in CONFIG['device']:
        print(f"  GPU memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

    results = train(
        model, train_loader, val_loader,
        CONFIG['num_epochs'], CONFIG['learning_rate'], CONFIG['weight_decay'],
        CONFIG['early_stopping_patience'], CONFIG['device'],
        class_weights=class_weights, label_smoothing=CONFIG['label_smoothing'],
        output_dir=output_dir,
        gradient_clip=CONFIG.get('gradient_clip'),
        warmup_epochs=CONFIG.get('warmup_epochs', 0),
        save_predictions=CONFIG.get('save_predictions', True),
        valid_classes=valid_classes
    )

    best_results = results['best_val_results']

    metrics_df = pd.DataFrame({
        'dataset': ['validation'],
        'Accuracy': [best_results['accuracy']],
        'F1_weighted': [best_results['f1_weighted']],
        'F1_macro': [best_results['f1_macro']],
    })
    metrics_path = os.path.join(output_dir, 'metrics.csv')
    metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')
    print(f"\n[Info] Metrics saved to: {metrics_path}")

    cm = best_results['confusion_matrix']
    cm_df = pd.DataFrame(cm)
    class_names = [f'class_{orig_cls}' for orig_cls in sorted(valid_classes)]
    cm_df.index = class_names
    cm_df.columns = class_names
    cm_path = os.path.join(output_dir, 'confusion_matrix.csv')
    cm_df.to_csv(cm_path, encoding='utf-8-sig')
    print(f"[Info] Confusion matrix saved to: {cm_path}")

    report_path = os.path.join(output_dir, 'classification_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(best_results['report'], f, indent=2, ensure_ascii=False)
    print(f"[Info] Classification report saved to: {report_path}")

    mapping_path = os.path.join(output_dir, 'class_mapping.json')
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(class_mapping, f, indent=2, ensure_ascii=False)
    print(f"[Info] Class mapping saved to: {mapping_path}")

    print("\n[Info] Building grid prediction table (for map visualization)...")
    best_model_path = os.path.join(output_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=CONFIG['device'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        full_dataset = LanduseDataset(X, y)
        full_loader = DataLoader(
            full_dataset,
            batch_size=CONFIG['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

        all_preds_labels, all_true_labels, all_probs = get_predictions(
            model, full_loader, CONFIG['device'])

        inv_class_mapping = {v: k for k, v in class_mapping.items()}

        prediction_df = pd.DataFrame({
            'GWBH': gwbh_all,
            'true_class': [inv_class_mapping[label] for label in all_true_labels],
            'pred_class': [inv_class_mapping[label] for label in all_preds_labels],
            'pred_label': all_preds_labels,
            'true_label': all_true_labels,
        })

        for i, orig_cls in enumerate(sorted(valid_classes)):
            prediction_df[f'prob_class_{orig_cls}'] = all_probs[:, i]

        prediction_df['pred_confidence'] = all_probs.max(axis=1)
        prediction_df['correct'] = (
            all_preds_labels == all_true_labels).astype(int)

        prediction_df = prediction_df.sort_values(
            'GWBH').reset_index(drop=True)

        prediction_path = os.path.join(output_dir, 'grid_predictions.csv')
        prediction_df.to_csv(prediction_path, index=False,
                             encoding='utf-8-sig')
        print(f"[Info] Grid predictions saved to: {prediction_path}")
        print(f"   {len(prediction_df)} grids")
        print("   Columns: GWBH, true_class, pred_class, pred_label, true_label,"
              " prob_class_*, pred_confidence, correct")

        if results.get('train_predictions') is not None:
            train_preds_labels, train_true_labels, train_probs = results['train_predictions']
            val_preds_labels, val_true_labels, val_probs = results['val_predictions']

            train_pred_df = pd.DataFrame({
                'GWBH': gwbh_train,
                'true_class': [inv_class_mapping[label] for label in train_true_labels],
                'pred_class': [inv_class_mapping[label] for label in train_preds_labels],
                'pred_label': train_preds_labels,
                'true_label': train_true_labels,
            })
            for i, orig_cls in enumerate(sorted(valid_classes)):
                train_pred_df[f'prob_class_{orig_cls}'] = train_probs[:, i]
            train_pred_df['pred_confidence'] = train_probs.max(axis=1)
            train_pred_df['correct'] = (
                train_preds_labels == train_true_labels).astype(int)
            train_pred_df = train_pred_df.sort_values(
                'GWBH').reset_index(drop=True)
            train_pred_path = os.path.join(output_dir, 'train_predictions.csv')
            train_pred_df.to_csv(
                train_pred_path, index=False, encoding='utf-8-sig')
            print(f"[Info] Train predictions (with GWBH) saved to: {train_pred_path}")

            val_pred_df = pd.DataFrame({
                'GWBH': gwbh_test,
                'true_class': [inv_class_mapping[label] for label in val_true_labels],
                'pred_class': [inv_class_mapping[label] for label in val_preds_labels],
                'pred_label': val_preds_labels,
                'true_label': val_true_labels,
            })
            for i, orig_cls in enumerate(sorted(valid_classes)):
                val_pred_df[f'prob_class_{orig_cls}'] = val_probs[:, i]
            val_pred_df['pred_confidence'] = val_probs.max(axis=1)
            val_pred_df['correct'] = (
                val_preds_labels == val_true_labels).astype(int)
            val_pred_df = val_pred_df.sort_values(
                'GWBH').reset_index(drop=True)
            val_pred_path = os.path.join(output_dir, 'test_predictions.csv')
            val_pred_df.to_csv(val_pred_path, index=False,
                               encoding='utf-8-sig')
            print(f"[Info] Test predictions (with GWBH) saved to: {val_pred_path}")

    print("\n" + "=" * 60)
    print("Training complete")
    print("=" * 60)
    print(f"Best validation F1-macro: {best_results['f1_macro']:.4f}")
    print(f"Best validation accuracy: {best_results['accuracy']:.4f}")

    if CONFIG['use_full_data'] or CONFIG['test_size'] == 0:
        print("\nNote: trained on all data; validation set equals training set")
        print("   This may overfit but maximizes in-sample performance")
        if results.get('train_predictions') is not None:
            train_preds, train_labels, _ = results['train_predictions']
            train_acc = accuracy_score(train_labels, train_preds)
            train_f1 = f1_score(train_labels, train_preds, average='macro')
            print(f"   Train accuracy: {train_acc:.4f}")
            print(f"   Train F1-macro: {train_f1:.4f}")

    print(f"\nResults saved under: {output_dir}")
    print("Map visualization: grid_predictions.csv (all grid predictions)")


if __name__ == '__main__':
    main()
