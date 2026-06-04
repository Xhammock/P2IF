#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict downstream tasks from model embeddings using LightGBM.
Supports house price, vitality, and land-use classification.

Edit CONFIG below to run without extra CLI arguments.
"""

import argparse
import os
import warnings
from typing import List, Optional, Tuple
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration — edit parameters here
# ============================================================================
CONFIG = {
    # Task type: 'house_price', 'vitality', or 'landuse'
    'task': 'house_price',

    # File paths
    'embeddings_path': 'checkpoints/train_20260410_214956/best_embeddings.npz',
    'price_data_path': 'data/house_price_aligned_grid.csv',
    'vitality_data_path': 'data/vitality_weekday_aggregated.csv',
    'landuse_data_path': 'data/landuse_aligned_grid.csv',
    'output_dir': None,  # Auto-set from task if None

    # Embedding
    'embedding_key': 'h',
    'concat_keys': ['h_spatial', 'h_od'],

    # Train/test split
    'test_size': 0.2,
    'random_state': 42,

    # Feature engineering
    'use_pca': True,
    'pca_n_components': 128,
    'use_log_transform': True,  # Metrics in log space; predictions saved in original space

    # Land-use classification
    'valid_classes': [0, 1, 2, 3, 4, 5],  # None = all classes

    # LightGBM (regression)
    'lgb_params': {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
    },
    # LightGBM (classification)
    'lgb_params_classification': {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
    },
    'num_boost_round': 500,
    'early_stopping_rounds': 100,
    'log_evaluation_period': 50,
}
# ============================================================================


def _concat_embeddings(data: "np.lib.npyio.NpzFile", keys: List[str]) -> np.ndarray:
    """Concatenate multiple 2D embedding matrices along the feature dimension.

    Requirements:
    - Each key maps to a 2D ndarray of shape (N, D_i)
    - All arrays share the same N
    """
    arrays: List[np.ndarray] = []
    n_rows: Optional[int] = None
    for k in keys:
        if k not in getattr(data, "files", []):
            raise KeyError(
                f"key='{k}' not found in npz; available keys: {list(getattr(data, 'files', []))}")
        arr = data[k]
        if not isinstance(arr, np.ndarray) or arr.ndim != 2:
            raise ValueError(
                f"Data for key='{k}' is not a 2D ndarray. "
                f"type={type(arr)}, ndim={getattr(arr, 'ndim', None)}, shape={getattr(arr, 'shape', None)}"
            )
        if n_rows is None:
            n_rows = arr.shape[0]
        elif arr.shape[0] != n_rows:
            raise ValueError(
                f"Concatenation failed: key='{k}' has {arr.shape[0]} rows, expected {n_rows}."
            )
        arrays.append(arr)
    return np.concatenate(arrays, axis=1)


def load_embeddings(
    path: str,
    embedding_key: Optional[str] = None,
    concat_keys: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Load node IDs and embedding matrix from an npz file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embeddings file not found: {path}")

    data = np.load(path, allow_pickle=True)
    keys = list(getattr(data, "files", []))

    requested = (embedding_key or "h").strip()

    if requested.lower() in {"h", "concat"}:
        concat_keys = concat_keys or ["h_spatial", "h_od"]
        embeddings = _concat_embeddings(data, concat_keys)
        key_to_use = f"concat({'+'.join(concat_keys)})"
        print(f"[Info] Using concatenated embeddings: {key_to_use}, shape={embeddings.shape}")
    else:
        if requested in keys:
            key_to_use = requested
        else:
            if "embeddings" in keys:
                key_to_use = "embeddings"
                print(f"[Warning] key='{requested}' not found; using 'embeddings'")
            else:
                candidates = []
                for k in keys:
                    try:
                        arr = data[k]
                        if isinstance(arr, np.ndarray) and arr.ndim == 2:
                            candidates.append((arr.shape[1], k))
                    except Exception:
                        continue
                if not candidates:
                    raise KeyError(f"Could not find a usable 2D embedding matrix in {keys}")
                candidates.sort(reverse=True)
                key_to_use = candidates[0][1]
                print(f"[Warning] key='{requested}' not found; using '{key_to_use}'")

        embeddings = data[key_to_use]
        if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
            raise ValueError(
                f"Data for key='{key_to_use}' is not a 2D ndarray. "
                f"type={type(embeddings)}, ndim={getattr(embeddings, 'ndim', None)}, shape={getattr(embeddings, 'shape', None)}"
            )
        print(f"[Info] Using embedding key: '{key_to_use}', shape={embeddings.shape}")

    node_ids = data.get('node_ids', np.arange(len(embeddings)))

    if node_ids.dtype.kind in {'S', 'O', 'U'}:
        node_ids = np.array([int(float(x)) if x else 0 for x in node_ids])
    else:
        node_ids = node_ids.astype(int)

    if len(node_ids) != len(embeddings):
        print(
            f"[Warning] node_ids length ({len(node_ids)}) != embeddings rows ({len(embeddings)}); "
            f"using 0..N-1 as node_ids")
        node_ids = np.arange(len(embeddings), dtype=int)

    return node_ids, embeddings


def load_price_data(path: str) -> pd.DataFrame:
    """Load house price data."""
    df = pd.read_csv(path)
    df = df[df['GWBH'].notna() & (df['GWBH'] != '')]
    df = df[df['avgprice'].notna()]
    df['GWBH'] = pd.to_numeric(df['GWBH'], errors='coerce')
    df = df[df['GWBH'].notna()]
    df['GWBH'] = df['GWBH'].astype(int)
    return df


def load_vitality_data(path: str) -> pd.DataFrame:
    """Load vitality (foot traffic) data."""
    df = pd.read_csv(path)
    df = df[df['GWBH'].notna() & (df['GWBH'] != '')]
    df = df[df['total_people'].notna()]
    df['GWBH'] = pd.to_numeric(df['GWBH'], errors='coerce')
    df = df[df['GWBH'].notna()]
    df['GWBH'] = df['GWBH'].astype(int)
    return df


def load_landuse_data(path: str, valid_classes: Optional[List[int]] = None) -> pd.DataFrame:
    """Load land-use data."""
    df = pd.read_csv(path)
    df = df[df['GWBH'].notna() & (df['GWBH'] != '')]
    df = df[df['landuse'].notna()]
    df['GWBH'] = pd.to_numeric(df['GWBH'], errors='coerce')
    df = df[df['GWBH'].notna()]
    df['GWBH'] = df['GWBH'].astype(int)
    df['landuse'] = df['landuse'].astype(int)

    if valid_classes is not None:
        df = df[df['landuse'].isin(valid_classes)]
        print(f"[Info] Kept classes {valid_classes}; {len(df)} records remain")

    print(f"[Info] Class distribution: {df['landuse'].value_counts().sort_index().to_dict()}")

    return df


def merge_data(
    node_ids: np.ndarray,
    embeddings: np.ndarray,
    target_df: pd.DataFrame,
    target_column: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match embeddings to targets via GWBH/node_ids.

    Returns:
        X: Feature matrix
        y: Target values
        gwbhs: Matched GWBH list
    """
    emb_df = pd.DataFrame({
        'GWBH': node_ids,
        'embedding_idx': range(len(node_ids))
    })

    merged = target_df.merge(emb_df, on='GWBH', how='inner')

    if len(merged) == 0:
        raise ValueError("No rows matched! Check that GWBH formats are consistent.")

    data_name = "target data" if target_column == 'total_people' else "price data"
    print(
        f"[Info] Matched {len(merged)} rows ({data_name}: {len(target_df)}, embeddings: {len(node_ids)})")

    matched_indices = merged['embedding_idx'].values
    X = embeddings[matched_indices]
    y = merged[target_column].values
    gwbhs = merged['GWBH'].values

    return X, y, gwbhs


def train_and_evaluate_classification(
    X: np.ndarray,
    y: np.ndarray,
    gwbhs: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    lgb_params: dict = None,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 30,
    log_evaluation_period: int = 50,
    use_pca: bool = True,
    pca_n_components: int = 128,
) -> dict:
    """Train and evaluate LightGBM for land-use classification.

    Args:
        X: Feature matrix
        y: Target labels
        gwbhs: GWBH list for per-grid predictions
    """
    X_train, X_test, y_train, y_test, gwbh_train, gwbh_test = train_test_split(
        X, y, gwbhs, test_size=test_size, random_state=random_state
    )

    print(f"\n[Info] Train size: {len(X_train)}, test size: {len(X_test)}")
    print(f"[Info] Original feature dim: {X.shape[1]}")
    print(
        f"[Info] Num classes: {len(np.unique(y))}, classes: {sorted(np.unique(y).tolist())}")
    print(
        f"[Info] Train class distribution: {pd.Series(y_train).value_counts().sort_index().to_dict()}")
    print(
        f"[Info] Test class distribution: {pd.Series(y_test).value_counts().sort_index().to_dict()}")

    pca = None
    if use_pca:
        print(f"\n[Info] Applying PCA to {pca_n_components} dimensions...")
        pca = PCA(n_components=pca_n_components, random_state=random_state)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        print(f"[Info] Reduced feature dim: {X_train.shape[1]}")
        print(f"[Info] Cumulative explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    unique_classes = sorted(np.unique(y))
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    y_train_mapped = np.array([class_to_idx[c] for c in y_train])
    y_test_mapped = np.array([class_to_idx[c] for c in y_test])

    print(f"\n[Info] Class mapping: {class_to_idx}")

    num_classes = len(unique_classes)
    train_data = lgb.Dataset(X_train, label=y_train_mapped)
    test_data = lgb.Dataset(X_test, label=y_test_mapped, reference=train_data)

    if lgb_params is None:
        lgb_params = {
            'objective': 'multiclass',
            'num_class': num_classes,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
        }
    else:
        lgb_params = lgb_params.copy()
        lgb_params['num_class'] = num_classes

    params = {**lgb_params, 'random_state': random_state}

    print("\n[Info] Training LightGBM classifier...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[train_data, test_data],
        valid_names=['train', 'eval'],
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=log_evaluation_period)
        ]
    )

    y_train_pred_proba = model.predict(
        X_train, num_iteration=model.best_iteration)
    y_test_pred_proba = model.predict(
        X_test, num_iteration=model.best_iteration)

    y_train_pred_mapped = np.argmax(y_train_pred_proba, axis=1)
    y_test_pred_mapped = np.argmax(y_test_pred_proba, axis=1)

    y_train_pred = np.array([idx_to_class[i] for i in y_train_pred_mapped])
    y_test_pred = np.array([idx_to_class[i] for i in y_test_pred_mapped])

    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_f1_macro = f1_score(y_train, y_train_pred, average='macro')
    train_f1_weighted = f1_score(y_train, y_train_pred, average='weighted')

    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1_macro = f1_score(y_test, y_test_pred, average='macro')
    test_f1_weighted = f1_score(y_test, y_test_pred, average='weighted')

    test_report = classification_report(y_test, y_test_pred, output_dict=True)
    test_cm = confusion_matrix(y_test, y_test_pred)

    train_report = classification_report(
        y_train, y_train_pred, output_dict=True)
    train_cm = confusion_matrix(y_train, y_train_pred)

    results = {
        'train': {
            'accuracy': train_accuracy,
            'f1_macro': train_f1_macro,
            'f1_weighted': train_f1_weighted
        },
        'test': {
            'accuracy': test_accuracy,
            'f1_macro': test_f1_macro,
            'f1_weighted': test_f1_weighted
        },
        'model': model,
        'pca': pca,
        'class_mapping': class_to_idx,
        'idx_to_class': idx_to_class,
        'train_predictions': {
            'GWBH': gwbh_train,
            'y_true': y_train,
            'y_pred': y_train_pred,
        },
        'test_predictions': {
            'GWBH': gwbh_test,
            'y_true': y_test,
            'y_pred': y_test_pred,
        },
        'train_report': train_report,
        'test_report': test_report,
        'train_confusion_matrix': train_cm,
        'test_confusion_matrix': test_cm,
    }

    return results


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    gwbhs: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    lgb_params: dict = None,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 30,
    log_evaluation_period: int = 50,
    use_pca: bool = True,
    pca_n_components: int = 128,
    use_log_transform: bool = True
) -> dict:
    """Train and evaluate LightGBM for regression.

    Args:
        X: Feature matrix
        y: Target values
        gwbhs: GWBH list for per-grid predictions
    """
    X_train, X_test, y_train, y_test, gwbh_train, gwbh_test = train_test_split(
        X, y, gwbhs, test_size=test_size, random_state=random_state
    )

    print(f"\n[Info] Train size: {len(X_train)}, test size: {len(X_test)}")
    print(f"[Info] Original feature dim: {X.shape[1]}")
    print(
        f"[Info] Target range: [{y.min():.2f}, {y.max():.2f}], mean: {y.mean():.2f}, std: {y.std():.2f}")

    y_train_original = y_train.copy()
    y_test_original = y_test.copy()

    pca = None
    if use_pca:
        print(f"\n[Info] Applying PCA to {pca_n_components} dimensions...")
        pca = PCA(n_components=pca_n_components, random_state=random_state)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        print(f"[Info] Reduced feature dim: {X_train.shape[1]}")
        print(f"[Info] Cumulative explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    if use_log_transform:
        print(f"\n[Info] Applying log1p to targets (metrics in log space)...")
        y_train = np.log1p(y_train)
        y_test = np.log1p(y_test)
        print(f"[Info] Log-transformed train range: [{y_train.min():.4f}, {y_train.max():.4f}]")

    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    if lgb_params is None:
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
        }
    params = {**lgb_params, 'random_state': random_state}

    print("\n[Info] Training LightGBM model...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[train_data, test_data],
        valid_names=['train', 'eval'],
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=log_evaluation_period)
        ]
    )

    y_train_pred_log = model.predict(
        X_train, num_iteration=model.best_iteration)
    y_test_pred_log = model.predict(X_test, num_iteration=model.best_iteration)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred_log))
    train_mae = mean_absolute_error(y_train, y_train_pred_log)
    train_r2 = r2_score(y_train, y_train_pred_log)

    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_log))
    test_mae = mean_absolute_error(y_test, y_test_pred_log)
    test_r2 = r2_score(y_test, y_test_pred_log)

    if use_log_transform:
        y_train_pred = np.expm1(y_train_pred_log)
        y_test_pred = np.expm1(y_test_pred_log)
    else:
        y_train_pred = y_train_pred_log
        y_test_pred = y_test_pred_log

    train_abs_error = np.abs(y_train_original - y_train_pred)
    test_abs_error = np.abs(y_test_original - y_test_pred)

    results = {
        'train': {
            'RMSE': train_rmse,
            'MAE': train_mae,
            'R²': train_r2
        },
        'test': {
            'RMSE': test_rmse,
            'MAE': test_mae,
            'R²': test_r2
        },
        'model': model,
        'pca': pca,
        'train_predictions': {
            'GWBH': gwbh_train,
            'y_true': y_train_original,
            'y_pred': y_train_pred,
            'abs_error': train_abs_error
        },
        'test_predictions': {
            'GWBH': gwbh_test,
            'y_true': y_test_original,
            'y_pred': y_test_pred,
            'abs_error': test_abs_error
        },
        'use_log_transform': use_log_transform
    }

    return results


def print_results(results: dict, task: str = 'house_price', use_log_transform: bool = True):
    """Print evaluation metrics."""
    print("\n" + "="*60)

    if task == 'landuse':
        task_name = "Land-use classification"
        print(f"{task_name} evaluation")
        print("="*60)

        print("\nTrain metrics:")
        print(f"  Accuracy:             {results['train']['accuracy']:.4f}")
        print(f"  F1 (macro):           {results['train']['f1_macro']:.4f}")
        print(f"  F1 (weighted):        {results['train']['f1_weighted']:.4f}")

        print("\nTest metrics:")
        print(f"  Accuracy:             {results['test']['accuracy']:.4f}")
        print(f"  F1 (macro):           {results['test']['f1_macro']:.4f}")
        print(f"  F1 (weighted):        {results['test']['f1_weighted']:.4f}")

        print("\nClassification report (test):")
        print("-" * 60)
        test_report = results['test_report']
        for class_label in sorted([k for k in test_report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]):
            metrics = test_report[class_label]
            print(f"  Class {class_label}:")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall:    {metrics['recall']:.4f}")
            print(f"    F1-Score:  {metrics['f1-score']:.4f}")
            print(f"    Support:   {int(metrics['support'])}")

    else:
        task_name = "Vitality prediction" if task == 'vitality' else "House price prediction"
        print(f"{task_name} evaluation")
        if use_log_transform:
            space_name = "vitality" if task == 'vitality' else "price"
            print(f"(Note: metrics computed in log space, not restored to original {space_name})")
        print("="*60)

        print("\nTrain metrics:")
        print(f"  RMSE: {results['train']['RMSE']:.4f}")
        print(f"  MAE:  {results['train']['MAE']:.4f}")
        print(f"  R²:   {results['train']['R²']:.4f}")

        print("\nTest metrics:")
        print(f"  RMSE: {results['test']['RMSE']:.4f}")
        print(f"  MAE:  {results['test']['MAE']:.4f}")
        print(f"  R²:   {results['test']['R²']:.4f}")

    print("="*60 + "\n")


def save_results(results: dict, output_dir: str, task: str = 'house_price'):
    """Save metrics, predictions, and model to disk."""
    os.makedirs(output_dir, exist_ok=True)

    if task == 'landuse':
        metrics_df = pd.DataFrame({
            'Dataset': ['Train', 'Test'],
            'Accuracy': [results['train']['accuracy'], results['test']['accuracy']],
            'F1_Macro': [results['train']['f1_macro'], results['test']['f1_macro']],
            'F1_Weighted': [results['train']['f1_weighted'], results['test']['f1_weighted']]
        })
        metrics_path = os.path.join(output_dir, 'metrics.csv')
        metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')
        print(f"[Info] Metrics saved to: {metrics_path}")

        class_mapping_path = os.path.join(output_dir, 'class_mapping.json')
        with open(class_mapping_path, 'w', encoding='utf-8') as f:
            json.dump({str(k): v for k, v in results['class_mapping'].items(
            )}, f, indent=2, ensure_ascii=False)
        print(f"[Info] Class mapping saved to: {class_mapping_path}")

        test_report_path = os.path.join(
            output_dir, 'classification_report.json')
        with open(test_report_path, 'w', encoding='utf-8') as f:
            json.dump(results['test_report'], f, indent=2, ensure_ascii=False)
        print(f"[Info] Classification report saved to: {test_report_path}")

        classes = sorted(results['class_mapping'].keys())
        cm_df = pd.DataFrame(
            results['test_confusion_matrix'],
            index=[f'True_{c}' for c in classes],
            columns=[f'Pred_{c}' for c in classes]
        )
        cm_path = os.path.join(output_dir, 'confusion_matrix.csv')
        cm_df.to_csv(cm_path, encoding='utf-8-sig')
        print(f"[Info] Confusion matrix saved to: {cm_path}")

        train_pred_df = pd.DataFrame({
            'GWBH': results['train_predictions']['GWBH'],
            'y_true': results['train_predictions']['y_true'],
            'y_pred': results['train_predictions']['y_pred'],
            'correct': (results['train_predictions']['y_true'] == results['train_predictions']['y_pred']).astype(int)
        })
        train_pred_path = os.path.join(output_dir, 'train_predictions.csv')
        train_pred_df.to_csv(train_pred_path, index=False,
                             encoding='utf-8-sig')
        print(f"[Info] Train predictions saved to: {train_pred_path}")

        test_pred_df = pd.DataFrame({
            'GWBH': results['test_predictions']['GWBH'],
            'y_true': results['test_predictions']['y_true'],
            'y_pred': results['test_predictions']['y_pred'],
            'correct': (results['test_predictions']['y_true'] == results['test_predictions']['y_pred']).astype(int)
        })
        test_pred_path = os.path.join(output_dir, 'test_predictions.csv')
        test_pred_df.to_csv(test_pred_path, index=False, encoding='utf-8-sig')
        print(f"[Info] Test predictions saved to: {test_pred_path}")

        all_pred_df = pd.concat([
            train_pred_df.assign(split='train'),
            test_pred_df.assign(split='test')
        ], ignore_index=True)
        all_pred_path = os.path.join(output_dir, 'all_predictions.csv')
        all_pred_df.to_csv(all_pred_path, index=False, encoding='utf-8-sig')
        print(f"[Info] All grid predictions saved to: {all_pred_path} (split column: train/test)")

    else:
        metrics_df = pd.DataFrame({
            'Dataset': ['Train', 'Test'],
            'RMSE': [results['train']['RMSE'], results['test']['RMSE']],
            'MAE': [results['train']['MAE'], results['test']['MAE']],
            'R²': [results['train']['R²'], results['test']['R²']]
        })
        metrics_path = os.path.join(output_dir, 'metrics.csv')
        metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')
        print(f"[Info] Metrics saved to: {metrics_path}")

        use_log_transform = results.get('use_log_transform', False)

        train_pred_df = pd.DataFrame({
            'GWBH': results['train_predictions']['GWBH'],
            'y_true': results['train_predictions']['y_true'],
            'y_pred': results['train_predictions']['y_pred'],
            'abs_error': results['train_predictions']['abs_error']
        })
        train_pred_path = os.path.join(output_dir, 'train_predictions.csv')
        train_pred_df.to_csv(train_pred_path, index=False,
                             encoding='utf-8-sig')
        print(f"[Info] Train predictions saved to: {train_pred_path}")

        test_pred_df = pd.DataFrame({
            'GWBH': results['test_predictions']['GWBH'],
            'y_true': results['test_predictions']['y_true'],
            'y_pred': results['test_predictions']['y_pred'],
            'abs_error': results['test_predictions']['abs_error']
        })
        test_pred_path = os.path.join(output_dir, 'test_predictions.csv')
        test_pred_df.to_csv(test_pred_path, index=False, encoding='utf-8-sig')
        print(f"[Info] Test predictions saved to: {test_pred_path}")

        all_pred_df = pd.concat([
            train_pred_df.assign(split='train'),
            test_pred_df.assign(split='test')
        ], ignore_index=True)
        all_pred_path = os.path.join(output_dir, 'all_predictions.csv')
        all_pred_df.to_csv(all_pred_path, index=False, encoding='utf-8-sig')
        print(f"[Info] All grid predictions saved to: {all_pred_path} (split column: train/test)")

        space_name = "vitality" if task == 'vitality' else "price"
        if use_log_transform:
            print(f"[Note] Training used log transform; saved predictions are in original {space_name} space for mapping")
        else:
            print(f"[Info] Predictions are in original {space_name} space for mapping")

    model_path = os.path.join(output_dir, 'lightgbm_model.txt')
    results['model'].save_model(model_path)
    print(f"[Info] Model saved to: {model_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Downstream prediction from embeddings (house price, vitality, land use)')
    parser.add_argument(
        '--task',
        type=str,
        choices=['house_price', 'vitality', 'landuse'],
        default=None,
        help='Task: house_price, vitality, or landuse (default: CONFIG)'
    )
    parser.add_argument(
        '--embeddings',
        type=str,
        default=None,
        help='Path to embeddings .npz (default: CONFIG)'
    )
    parser.add_argument(
        '--price_data',
        type=str,
        default=None,
        help='House price CSV path (default: CONFIG)'
    )
    parser.add_argument(
        '--vitality_data',
        type=str,
        default=None,
        help='Vitality CSV path (default: CONFIG)'
    )
    parser.add_argument(
        '--landuse_data',
        type=str,
        default=None,
        help='Land-use CSV path (default: CONFIG)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (default: CONFIG)'
    )
    parser.add_argument(
        '--embedding_key',
        type=str,
        default=None,
        help='Embedding key in npz (default: CONFIG)'
    )
    parser.add_argument(
        '--concat_keys',
        type=str,
        nargs='+',
        default=None,
        help='Keys to concatenate when embedding_key is h/concat (default: CONFIG)'
    )

    args = parser.parse_args()

    task = args.task or CONFIG['task']
    embeddings_path = args.embeddings or CONFIG['embeddings_path']
    price_data_path = args.price_data or CONFIG['price_data_path']
    vitality_data_path = args.vitality_data or CONFIG['vitality_data_path']
    landuse_data_path = args.landuse_data or CONFIG['landuse_data_path']
    output_dir = args.output_dir or CONFIG['output_dir']
    embedding_key = args.embedding_key or CONFIG['embedding_key']
    concat_keys = args.concat_keys or CONFIG['concat_keys']
    test_size = CONFIG['test_size']
    random_state = CONFIG['random_state']
    lgb_params = CONFIG['lgb_params']
    lgb_params_classification = CONFIG['lgb_params_classification']
    num_boost_round = CONFIG['num_boost_round']
    early_stopping_rounds = CONFIG['early_stopping_rounds']
    log_evaluation_period = CONFIG['log_evaluation_period']
    use_pca = CONFIG['use_pca']
    pca_n_components = CONFIG['pca_n_components']
    use_log_transform = CONFIG['use_log_transform']
    valid_classes = CONFIG['valid_classes']

    if output_dir is None:
        checkpoint_name = os.path.basename(os.path.dirname(embeddings_path))
        if task == 'vitality':
            task_name = 'vitality_prediction'
        elif task == 'landuse':
            task_name = 'landuse_lgb'
        else:
            task_name = 'house_price_prediction'
        output_dir = f'results/{checkpoint_name}/{task_name}'

    if task == 'vitality':
        target_data_path = vitality_data_path
        target_column = 'total_people'
        task_display_name = 'Vitality prediction'
    elif task == 'landuse':
        target_data_path = landuse_data_path
        target_column = 'landuse'
        task_display_name = 'Land-use classification'
    else:
        target_data_path = price_data_path
        target_column = 'avgprice'
        task_display_name = 'House price prediction'

    print("="*60)
    print("Configuration:")
    print(f"  Task: {task_display_name}")
    print(f"  Embeddings: {embeddings_path}")
    print(f"  Target data: {target_data_path}")
    print(f"  Output dir: {output_dir}")
    print(f"  Embedding key: {embedding_key}")
    if embedding_key.lower() in {'h', 'concat'}:
        print(f"  Concat keys: {concat_keys}")
    print(f"  Test size: {test_size}")
    print(f"  Random seed: {random_state}")
    print(f"  Use PCA: {use_pca} (to {pca_n_components} dims)")
    if task != 'landuse':
        print(f"  Log transform: {use_log_transform} (metrics in log space)")
    if task == 'landuse':
        print(f"  Filter classes: {valid_classes if valid_classes else 'all'}")
    print("="*60)

    print("\n[Info] Loading embeddings...")
    node_ids, embeddings = load_embeddings(
        embeddings_path,
        embedding_key=embedding_key,
        concat_keys=concat_keys
    )

    if task == 'vitality':
        print("[Info] Loading vitality data...")
        target_df = load_vitality_data(target_data_path)
    elif task == 'landuse':
        print("[Info] Loading land-use data...")
        target_df = load_landuse_data(
            target_data_path, valid_classes=valid_classes)
    else:
        print("[Info] Loading house price data...")
        target_df = load_price_data(target_data_path)

    print("[Info] Merging data...")
    X, y, gwbhs = merge_data(node_ids, embeddings, target_df, target_column)

    if task == 'landuse':
        results = train_and_evaluate_classification(
            X, y, gwbhs,
            test_size=test_size,
            random_state=random_state,
            lgb_params=lgb_params_classification,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            log_evaluation_period=log_evaluation_period,
            use_pca=use_pca,
            pca_n_components=pca_n_components,
        )
    else:
        results = train_and_evaluate(
            X, y, gwbhs,
            test_size=test_size,
            random_state=random_state,
            lgb_params=lgb_params,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            log_evaluation_period=log_evaluation_period,
            use_pca=use_pca,
            pca_n_components=pca_n_components,
            use_log_transform=use_log_transform
        )

    print_results(results, task=task,
                  use_log_transform=use_log_transform if task != 'landuse' else False)

    save_results(results, output_dir, task=task)


if __name__ == '__main__':
    main()
