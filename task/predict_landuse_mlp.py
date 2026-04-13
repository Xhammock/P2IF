#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用浅层MLP对功能区（土地利用）进行分类预测
参考HGI方法：512维隐藏层（BatchNorm1d + tanh激活）+ softmax输出层

只预测类别0, 1, 4, 5（去掉类别2和3，因为数据偏态）
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
# 配置参数
# ============================================================================
CONFIG = {
    # 文件路径配置
    'embeddings_path': 'checkpoints/train_20260106_105040/best_embeddings.npz',
    'landuse_data_path': '深圳土地利用分类/landuse_1km格网.npy',
    'landuse_csv_path': '土地利用_对齐网格_新.csv',  # 优先使用CSV文件
    'shp_path': 'data/SZ_clip/SZ_clip.shp',  # 用于空间对齐的shp文件
    'output_dir': None,  # 如果为None，会自动根据时间戳创建目录

    # Embedding配置
    'embedding_key': 'h',  # 可选: 'h', 'concat', 'z', 'h_spatial', 'h_od', 'embeddings'等
    'concat_keys': ['h_spatial', 'h_od'],  # 当embedding_key='h'或'concat'时使用

    # 数据配置
    'valid_classes': [0, 1, 2, 3, 4, 5],  # 使用的类别
    'test_size': 0.0,  # 如果设置为0，则使用全部数据训练，不划分测试集
    'use_full_data': True,  # 如果为True，使用全部数据训练，测试集也使用训练数据
    'random_state': 42,

    # 模型配置（参考HGI方法）
    'hidden_dim': 512,  # 512维隐藏层
    'use_batch_norm': True,  # 使用1D批归一化
    'activation': 'tanh',  # tanh激活函数
    'dropout': 0.0,  # 可选：dropout率

    # 训练配置
    'batch_size': 128,
    'num_epochs': 150,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'early_stopping_patience': 50,
    'use_class_weight': False,  # 是否使用类别权重
    'label_smoothing': 0.1,  # CrossEntropyLoss的label_smoothing
    'gradient_clip': 1.0,  # 梯度裁剪（None表示不裁剪）
    'warmup_epochs': 10,  # 学习率预热的epoch数
    'save_predictions': True,  # 是否保存预测结果

    # 设备配置
    'device': None,  # 将在main函数中安全检测
}
# ============================================================================


# ============================================================================
# 模型定义
# ============================================================================

class LanduseMLP(nn.Module):
    """浅层MLP分类器（参考HGI方法）

    结构：
    - 输入层 -> 512维隐藏层（BatchNorm1d + tanh）-> 输出层（softmax）
    """

    def __init__(self, input_dim: int, num_classes: int = 4,
                 hidden_dim: int = 512, use_batch_norm: bool = True,
                 activation: str = 'tanh', dropout: float = 0.0):
        super().__init__()
        self.num_classes = num_classes

        # 隐藏层
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(
            hidden_dim) if use_batch_norm else nn.Identity()

        # 激活函数
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"不支持的激活函数: {activation}")

        # Dropout（可选）
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 输出层（softmax在CrossEntropyLoss中自动应用）
        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.output(x)
        return x


# ============================================================================
# 数据集定义
# ============================================================================

class LanduseDataset(Dataset):
    """土地利用分类数据集"""

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
# 数据加载函数
# ============================================================================

def _concat_embeddings(data: "np.lib.npyio.NpzFile", keys: List[str]) -> np.ndarray:
    """拼接多个嵌入矩阵"""
    arrays = []
    n_rows = None
    for k in keys:
        if k not in getattr(data, "files", []):
            raise KeyError(f"npz 中不存在 key='{k}'")
        arr = data[k]
        if not isinstance(arr, np.ndarray) or arr.ndim != 2:
            raise ValueError(f"key='{k}' 对应的数据不是二维 ndarray")
        if n_rows is None:
            n_rows = arr.shape[0]
        elif arr.shape[0] != n_rows:
            raise ValueError(f"拼接失败：key='{k}' 的行数不一致")
        arrays.append(arr)
    return np.concatenate(arrays, axis=1)


def load_embeddings(path: str, embedding_key: Optional[str] = None,
                    concat_keys: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """加载embeddings"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到嵌入文件: {path}")

    data = np.load(path, allow_pickle=True)
    keys = list(getattr(data, "files", []))
    requested = (embedding_key or "z").strip()

    if requested.lower() in {"h", "concat"}:
        concat_keys = concat_keys or ["h_spatial", "h_od"]
        embeddings = _concat_embeddings(data, concat_keys)
        print(
            f"[信息] 使用拼接嵌入：concat({'+'.join(concat_keys)})，shape={embeddings.shape}")
    else:
        if requested in keys:
            embeddings = data[requested]
        else:
            if "embeddings" in keys:
                embeddings = data["embeddings"]
                print(f"[警告] key='{requested}'不存在，使用 'embeddings'")
            else:
                candidates = [(data[k].shape[1], k) for k in keys
                              if isinstance(data[k], np.ndarray) and data[k].ndim == 2]
                if not candidates:
                    raise KeyError(f"无法找到可用的嵌入矩阵")
                embeddings = data[max(candidates)[1]]
                print(f"[警告] key='{requested}'不存在，使用 '{max(candidates)[1]}'")
        print(f"[信息] 使用嵌入key: '{requested}'，shape={embeddings.shape}")

    node_ids = data.get('node_ids', np.arange(len(embeddings)))
    if node_ids.dtype.kind in {'S', 'O', 'U'}:
        node_ids = np.array([int(float(x)) if x else 0 for x in node_ids])
    else:
        node_ids = node_ids.astype(int)

    if len(node_ids) != len(embeddings):
        print(f"[警告] node_ids长度不一致，使用索引")
        node_ids = np.arange(len(embeddings), dtype=int)

    return node_ids, embeddings


def load_landuse_data(npy_path: Optional[str] = None, csv_path: Optional[str] = None,
                      node_ids: Optional[np.ndarray] = None,
                      shp_path: Optional[str] = None) -> pd.DataFrame:
    """
    加载土地利用数据

    如果提供了npy_path和shp_path，将使用空间对齐方法（推荐）
    否则使用简单的索引对齐（可能不准确）
    """
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if 'GWBH' not in df.columns or 'landuse' not in df.columns:
            raise ValueError(f"CSV必须包含'GWBH'和'landuse'列")
        df = df[df['GWBH'].notna() & df['landuse'].notna()]
        df['GWBH'] = pd.to_numeric(df['GWBH'], errors='coerce')
        df = df[df['GWBH'].notna()]
        df['GWBH'] = df['GWBH'].astype(int)
        df['landuse'] = df['landuse'].astype(int)
        return df
    elif npy_path and os.path.exists(npy_path):
        # 如果提供了shp_path，使用空间对齐方法
        if shp_path and os.path.exists(shp_path):
            print("[信息] 使用空间对齐方法加载土地利用数据...")
            import geopandas as gpd

            # 加载土地利用数据
            landuse_array = np.load(npy_path, allow_pickle=True)
            if landuse_array.ndim > 1:
                landuse_array = landuse_array.flatten()

            # 读取shp文件，建立空间索引
            gdf = gpd.read_file(shp_path)
            gdf['GWBH_1000'] = pd.to_numeric(gdf['GWBH_1000'], errors='coerce')
            gdf = gdf[gdf['GWBH_1000'].notna()].copy()
            gdf['GWBH_1000'] = gdf['GWBH_1000'].astype(int)
            gdf['GWBH'] = gdf['GWBH'].astype(int)

            # 获取唯一的GWBH_1000值并排序
            unique_gwbh_1000 = sorted(gdf['GWBH_1000'].unique())

            # 建立GWBH_1000到landuse的映射
            min_len = min(len(landuse_array), len(unique_gwbh_1000))
            if len(landuse_array) != len(unique_gwbh_1000):
                print(
                    f"[警告] 土地利用数据长度({len(landuse_array)})与唯一GWBH_1000数量({len(unique_gwbh_1000)})不一致，使用前{min_len}个值")
            gwbh_1000_to_landuse = dict(
                zip(unique_gwbh_1000[:min_len], landuse_array[:min_len]))

            # 将土地利用数据分配到500m格网
            gdf['landuse'] = gdf['GWBH_1000'].map(gwbh_1000_to_landuse)
            gdf_with_landuse = gdf[gdf['landuse'].notna()].copy()

            # 如果提供了node_ids，只保留这些节点
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
            # 使用简单的索引对齐（旧方法，不推荐）
            print("[警告] 未提供shp_path，使用简单的索引对齐方法（可能不准确）")
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
        raise FileNotFoundError("必须提供landuse_data_path或landuse_csv_path")


def merge_data(node_ids: np.ndarray, embeddings: np.ndarray,
               landuse_df: pd.DataFrame, valid_classes: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, int]]:
    """合并数据并过滤有效类别

    Returns:
        X: embeddings
        y: 标签（重新映射到0, 1, 2, 3, ...）
        gwbh: 格网编号（GWBH）
        class_mapping: 原始类别到新类别的映射
    """
    emb_df = pd.DataFrame(
        {'GWBH': node_ids, 'embedding_idx': range(len(node_ids))})

    merged = landuse_df.merge(emb_df, on='GWBH', how='inner')

    if len(merged) == 0:
        raise ValueError("无法匹配任何数据")

    # 过滤有效类别
    valid_mask = merged['landuse'].isin(valid_classes)
    merged = merged[valid_mask]

    if len(merged) == 0:
        raise ValueError(f"过滤后没有有效数据（有效类别: {valid_classes}）")

    print(f"[信息] 数据匹配: {len(merged)} 条（原始: {len(landuse_df)} 条）")

    # 创建类别映射（原始类别 -> 新类别索引）
    class_mapping = {orig_cls: new_idx for new_idx,
                     orig_cls in enumerate(sorted(valid_classes))}

    # 重新映射标签
    y = merged['landuse'].map(class_mapping).values
    X = embeddings[merged['embedding_idx'].values]
    gwbh = merged['GWBH'].values  # 保存GWBH信息

    # 检查类别分布
    unique, counts = np.unique(y, return_counts=True)
    print(f"[信息] 类别分布（新标签 -> 原始类别）:")
    for new_label, count in zip(unique, counts):
        orig_cls = sorted(valid_classes)[new_label]
        print(f"  新标签{new_label} (原始类别{orig_cls}): {count} 条")

    return X, y, gwbh, class_mapping


# ============================================================================
# 训练和评估
# ============================================================================

def train_epoch(model: nn.Module, train_loader: DataLoader, criterion: nn.Module,
                optimizer: optim.Optimizer, device: str, gradient_clip: Optional[float] = None) -> Tuple[float, Dict]:
    """训练一个epoch"""
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

        # 梯度裁剪
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
    """验证"""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_preds, all_labels = [], []
    all_probs = []  # 保存预测概率

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

    # 计算详细指标
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
    """获取数据集的预测结果

    Returns:
        predictions: 预测标签
        labels: 真实标签
        probabilities: 预测概率
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="获取预测"):
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
    """训练模型"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 损失函数
    if class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=label_smoothing)
        print(f"[信息] 使用类别权重: {class_weights.cpu().numpy()}")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    if label_smoothing > 0:
        print(f"[信息] 使用 label_smoothing: {label_smoothing}")

    # 优化器
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # 学习率预热调度器
    if warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs
        )
        print(f"[信息] 使用学习率预热：{warmup_epochs} epochs")

    if gradient_clip is not None and gradient_clip > 0:
        print(f"[信息] 使用梯度裁剪：max_norm={gradient_clip}")

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
    print("开始训练")
    print("=" * 60)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, gradient_clip
        )

        val_loss, val_results = validate(model, val_loader, criterion, device)

        # 学习率调度
        if warmup_epochs > 0 and epoch < warmup_epochs:
            warmup_scheduler.step()
            print(f"  预热阶段 LR: {optimizer.param_groups[0]['lr']:.6f}")
        else:
            scheduler.step(val_loss)

        # 保存历史（排除numpy数组，避免JSON序列化错误）
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        for key, value in train_metrics.items():
            if key not in history['train_metrics']:
                history['train_metrics'][key] = []
            history['train_metrics'][key].append(value)
        for key, value in val_results.items():
            # 跳过numpy数组（predictions, labels, probabilities），这些会单独保存
            if isinstance(value, np.ndarray):
                continue
            if key not in history['val_metrics']:
                history['val_metrics'][key] = []
            # 确保confusion_matrix是list格式（如果已经是list则保持不变）
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

        # 保存最佳模型（基于macro-F1）
        if val_results['f1_macro'] > best_val_f1_macro:
            best_val_f1_macro = val_results['f1_macro']
            best_val_loss = val_loss
            patience_counter = 0
            best_val_results = val_results.copy()

            # 过滤掉numpy数组，避免保存时占用过多空间
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
            print(f"✅ 保存最佳模型 (Val F1-macro: {val_results['f1_macro']:.4f})")
        else:
            patience_counter += 1
            if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
                print(f"\n早停触发 (patience={early_stopping_patience})")
                break

    # 保存训练历史
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    # 加载最佳模型并获取预测结果
    train_preds, val_preds = None, None
    if save_predictions:
        print("\n[信息] 加载最佳模型并获取预测结果...")
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

    # 清理best_val_results中的numpy数组
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
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='使用MLP进行土地利用分类')
    parser.add_argument('--embeddings', type=str, default=None)
    parser.add_argument('--landuse_data', type=str, default=None)
    parser.add_argument('--landuse_csv', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--embedding_key', type=str, default=None)
    parser.add_argument('--device', type=str, default=None,
                        help='指定设备 (cuda, cuda:0, cuda:1, cpu等)，默认自动检测')
    parser.add_argument('--use_full_data', action='store_true',
                        help='使用全部数据训练（不划分测试集，会过拟合但效果最好）')
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='学习率')

    args = parser.parse_args()

    # 更新配置
    if args.use_full_data:
        CONFIG['use_full_data'] = True
    if args.num_epochs is not None:
        CONFIG['num_epochs'] = args.num_epochs
    if args.batch_size is not None:
        CONFIG['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        CONFIG['learning_rate'] = args.learning_rate

    # 设备检测和设置
    print("\n" + "=" * 60)
    print("设备检测")
    print("=" * 60)

    if args.device:
        # 如果用户指定了设备，直接使用
        CONFIG['device'] = args.device
        print(f"[信息] 使用用户指定的设备: {CONFIG['device']}")
    elif CONFIG['device'] is None:
        # 自动检测设备
        print(f"[信息] PyTorch版本: {torch.__version__}")
        print(
            f"[信息] PyTorch CUDA编译版本: {torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A'}")

        try:
            # 先检查CUDA是否可用
            cuda_available = torch.cuda.is_available()
            print(f"[信息] torch.cuda.is_available(): {cuda_available}")

            if cuda_available:
                gpu_count = torch.cuda.device_count()
                print(f"[信息] 检测到 {gpu_count} 个GPU")

                # 尝试创建一个小的tensor来验证CUDA是否真的可用
                try:
                    test_tensor = torch.tensor([1.0]).to('cuda:0')
                    result = test_tensor * 2
                    del test_tensor, result
                    torch.cuda.empty_cache()

                    # 使用 cuda:0 明确指定第一个GPU
                    CONFIG['device'] = 'cuda:0'
                    print(f"[信息] ✅ CUDA验证成功，使用设备: {CONFIG['device']}")
                    print(f"[信息] GPU名称: {torch.cuda.get_device_name(0)}")
                    print(
                        f"[信息] GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
                except RuntimeError as e:
                    print(f"[警告] CUDA设备测试失败: {e}")
                    print(f"[警告] 可能是CUDA驱动版本不匹配，切换到CPU")
                    CONFIG['device'] = 'cpu'
            else:
                CONFIG['device'] = 'cpu'
                print("[信息] 未检测到CUDA，使用CPU")
                print("[提示] 如果您的系统有GPU，可能需要:")
                print("       1. 检查CUDA驱动是否正确安装")
                print("       2. 安装与CUDA驱动兼容的PyTorch版本")
                print("       3. 运行: python scripts/check_gpu.py 进行诊断")
        except Exception as e:
            # 如果CUDA检测过程中出错，使用CPU
            CONFIG['device'] = 'cpu'
            print(f"[警告] CUDA检测过程出错 ({str(e)})，将使用CPU进行训练")
            print(f"[提示] 运行: python scripts/check_gpu.py 查看详细错误信息")

    # 验证设备是否可用
    try:
        if 'cuda' in CONFIG['device']:
            device_id = int(CONFIG['device'].split(
                ':')[1]) if ':' in CONFIG['device'] else 0

            if not torch.cuda.is_available():
                print(f"[警告] CUDA不可用，切换到CPU")
                CONFIG['device'] = 'cpu'
            elif device_id >= torch.cuda.device_count():
                print(f"[警告] 设备 {CONFIG['device']} 不存在，使用 cuda:0")
                CONFIG['device'] = 'cuda:0'
            else:
                # 测试设备是否真的可用
                test_tensor = torch.tensor([1.0]).to(CONFIG['device'])
                result = test_tensor * 2
                del test_tensor, result
                torch.cuda.empty_cache()
                print(f"[信息] ✅ 设备 {CONFIG['device']} 验证成功")
    except Exception as e:
        print(f"[警告] 设备 {CONFIG['device']} 不可用 ({str(e)})，切换到CPU")
        CONFIG['device'] = 'cpu'

    print("=" * 60)

    # 从CONFIG获取配置
    embeddings_path = args.embeddings or CONFIG['embeddings_path']
    landuse_data_path = args.landuse_data or CONFIG['landuse_data_path']
    landuse_csv_path = args.landuse_csv or CONFIG['landuse_csv_path']
    embedding_key = args.embedding_key or CONFIG['embedding_key']
    valid_classes = CONFIG['valid_classes']

    # 自动生成输出目录
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
    print("配置信息:")
    print(f"  Embeddings路径: {embeddings_path}")
    print(f"  土地利用数据路径: {landuse_data_path}")
    print(f"  土地利用CSV路径: {landuse_csv_path}")
    print(f"  输出目录: {output_dir}")
    print(f"  Embedding key: {embedding_key}")
    print(f"  有效类别: {valid_classes}")
    print(f"  设备: {CONFIG['device']}")
    print("=" * 60)

    # 加载数据
    print("\n[信息] 加载embeddings...")
    node_ids, embeddings = load_embeddings(
        embeddings_path, embedding_key, CONFIG['concat_keys'])

    print("[信息] 加载土地利用数据...")
    landuse_df = load_landuse_data(
        landuse_data_path, landuse_csv_path, node_ids, CONFIG.get('shp_path'))

    # 合并数据并过滤有效类别
    print("[信息] 合并数据并过滤有效类别...")
    X, y, gwbh_all, class_mapping = merge_data(
        node_ids, embeddings, landuse_df, valid_classes)

    # 划分数据集
    if CONFIG['use_full_data']:
        print("[信息] 使用全部数据进行训练（不划分测试集）")
        X_train = X
        y_train = y
        gwbh_train = gwbh_all
        X_test = X  # 测试集也使用训练数据
        y_test = y
        gwbh_test = gwbh_all
    elif CONFIG['test_size'] == 0:
        print("[信息] test_size=0，使用全部数据进行训练（不划分测试集）")
        X_train = X
        y_train = y
        gwbh_train = gwbh_all
        X_test = X  # 测试集也使用训练数据
        y_test = y
        gwbh_test = gwbh_all
    else:
        # 使用相同的随机种子和分层策略划分数据，确保索引一致
        indices = np.arange(len(X))
        train_indices, test_indices = train_test_split(
            indices, test_size=CONFIG['test_size'], random_state=CONFIG['random_state'],
            stratify=y
        )
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        gwbh_train, gwbh_test = gwbh_all[train_indices], gwbh_all[test_indices]

    # 标准化
    print("[信息] 对embedding进行标准化...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    # 计算类别权重
    class_weights = None
    if CONFIG['use_class_weight']:
        class_counts = np.bincount(y_train, minlength=len(valid_classes))
        total_samples = class_counts.sum()
        class_weights = torch.FloatTensor(
            [total_samples / (len(class_counts) * count) if count > 0 else 0.0
             for count in class_counts]
        ).to(CONFIG['device'])  # 将类别权重移动到指定设备
        print(f"[信息] 类别权重: {class_weights.cpu().numpy()}")

    # 创建数据集
    train_dataset = LanduseDataset(X_train, y_train)
    val_dataset = LanduseDataset(X_test, y_test)

    # 优化DataLoader参数以更好地利用GPU
    # 如果使用GPU，可以设置num_workers来并行加载数据
    num_workers = 4 if 'cuda' in CONFIG['device'] else 0
    pin_memory = 'cuda' in CONFIG['device']  # 使用pin_memory可以加速GPU传输

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

    # 创建模型
    embedding_dim = X.shape[1]
    num_classes = len(valid_classes)
    model = LanduseMLP(
        embedding_dim, num_classes,
        hidden_dim=CONFIG['hidden_dim'],
        use_batch_norm=CONFIG['use_batch_norm'],
        activation=CONFIG['activation'],
        dropout=CONFIG['dropout']
    ).to(CONFIG['device'])

    print(f"\n[信息] 模型结构:")
    print(f"  输入维度: {embedding_dim}")
    print(f"  隐藏层维度: {CONFIG['hidden_dim']}")
    print(f"  输出类别数: {num_classes}")
    print(f"  总参数数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  模型设备: {next(model.parameters()).device}")
    if 'cuda' in CONFIG['device']:
        print(f"  GPU内存使用: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

    # 训练
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

    # 保存结果
    best_results = results['best_val_results']

    # 保存指标到CSV
    metrics_df = pd.DataFrame({
        '数据集': ['测试集'],
        'Accuracy': [best_results['accuracy']],
        'F1_weighted': [best_results['f1_weighted']],
        'F1_macro': [best_results['f1_macro']],
    })
    metrics_path = os.path.join(output_dir, 'metrics.csv')
    metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')
    print(f"\n[信息] 指标已保存到: {metrics_path}")

    # 保存混淆矩阵
    cm = best_results['confusion_matrix']
    cm_df = pd.DataFrame(cm)
    class_names = [f'类别{orig_cls}' for orig_cls in sorted(valid_classes)]
    cm_df.index = class_names
    cm_df.columns = class_names
    cm_path = os.path.join(output_dir, 'confusion_matrix.csv')
    cm_df.to_csv(cm_path, encoding='utf-8-sig')
    print(f"[信息] 混淆矩阵已保存到: {cm_path}")

    # 保存分类报告
    report_path = os.path.join(output_dir, 'classification_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(best_results['report'], f, indent=2, ensure_ascii=False)
    print(f"[信息] 分类报告已保存到: {report_path}")

    # 保存类别映射
    mapping_path = os.path.join(output_dir, 'class_mapping.json')
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(class_mapping, f, indent=2, ensure_ascii=False)
    print(f"[信息] 类别映射已保存到: {mapping_path}")

    # 保存包含格网ID的预测结果表格（用于地图可视化）
    print("\n[信息] 生成包含格网ID的预测结果表格...")
    best_model_path = os.path.join(output_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=CONFIG['device'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # 创建完整数据集（用于预测全部数据）
        full_dataset = LanduseDataset(X, y)
        full_loader = DataLoader(
            full_dataset,
            batch_size=CONFIG['batch_size'],
            shuffle=False,
            num_workers=0,  # 预测时不需要多进程
            pin_memory=False
        )

        # 获取全部数据的预测结果
        all_preds_labels, all_true_labels, all_probs = get_predictions(
            model, full_loader, CONFIG['device'])

        # 创建类别映射的反向映射（新标签 -> 原始类别）
        inv_class_mapping = {v: k for k, v in class_mapping.items()}

        # 创建预测结果表格
        prediction_df = pd.DataFrame({
            'GWBH': gwbh_all,  # 格网编号
            '真实类别': [inv_class_mapping[label] for label in all_true_labels],  # 原始类别
            # 原始类别
            '预测类别': [inv_class_mapping[label] for label in all_preds_labels],
            '预测标签': all_preds_labels,  # 新标签（0,1,2,...）
            '真实标签': all_true_labels,  # 新标签（0,1,2,...）
        })

        # 添加每个类别的预测概率
        for i, orig_cls in enumerate(sorted(valid_classes)):
            prediction_df[f'类别{orig_cls}_概率'] = all_probs[:, i]

        # 添加预测置信度（最大概率）
        prediction_df['预测置信度'] = all_probs.max(axis=1)

        # 添加是否正确预测的标记
        prediction_df['预测正确'] = (
            all_preds_labels == all_true_labels).astype(int)

        # 按GWBH排序，方便查看和可视化
        prediction_df = prediction_df.sort_values(
            'GWBH').reset_index(drop=True)

        # 保存预测结果表格
        prediction_path = os.path.join(output_dir, 'grid_predictions.csv')
        prediction_df.to_csv(prediction_path, index=False,
                             encoding='utf-8-sig')
        print(f"[信息] 格网预测结果已保存到: {prediction_path}")
        print(f"   包含 {len(prediction_df)} 个格网的预测结果")
        print(f"   列: GWBH, 真实类别, 预测类别, 预测标签, 真实标签, 各类别概率, 预测置信度, 预测正确")

        # 如果使用全数据训练，也保存训练集和测试集的预测（带GWBH）
        if results.get('train_predictions') is not None:
            train_preds_labels, train_true_labels, train_probs = results['train_predictions']
            val_preds_labels, val_true_labels, val_probs = results['val_predictions']

            # 训练集预测表格
            train_pred_df = pd.DataFrame({
                'GWBH': gwbh_train,
                '真实类别': [inv_class_mapping[label] for label in train_true_labels],
                '预测类别': [inv_class_mapping[label] for label in train_preds_labels],
                '预测标签': train_preds_labels,
                '真实标签': train_true_labels,
            })
            for i, orig_cls in enumerate(sorted(valid_classes)):
                train_pred_df[f'类别{orig_cls}_概率'] = train_probs[:, i]
            train_pred_df['预测置信度'] = train_probs.max(axis=1)
            train_pred_df['预测正确'] = (
                train_preds_labels == train_true_labels).astype(int)
            train_pred_df = train_pred_df.sort_values(
                'GWBH').reset_index(drop=True)
            train_pred_path = os.path.join(output_dir, 'train_predictions.csv')
            train_pred_df.to_csv(
                train_pred_path, index=False, encoding='utf-8-sig')
            print(f"[信息] 训练集预测（带GWBH）已保存到: {train_pred_path}")

            # 测试集预测表格
            val_pred_df = pd.DataFrame({
                'GWBH': gwbh_test,
                '真实类别': [inv_class_mapping[label] for label in val_true_labels],
                '预测类别': [inv_class_mapping[label] for label in val_preds_labels],
                '预测标签': val_preds_labels,
                '真实标签': val_true_labels,
            })
            for i, orig_cls in enumerate(sorted(valid_classes)):
                val_pred_df[f'类别{orig_cls}_概率'] = val_probs[:, i]
            val_pred_df['预测置信度'] = val_probs.max(axis=1)
            val_pred_df['预测正确'] = (
                val_preds_labels == val_true_labels).astype(int)
            val_pred_df = val_pred_df.sort_values(
                'GWBH').reset_index(drop=True)
            val_pred_path = os.path.join(output_dir, 'test_predictions.csv')
            val_pred_df.to_csv(val_pred_path, index=False,
                               encoding='utf-8-sig')
            print(f"[信息] 测试集预测（带GWBH）已保存到: {val_pred_path}")

    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"最佳验证 F1-macro: {best_results['f1_macro']:.4f}")
    print(f"最佳验证 Accuracy: {best_results['accuracy']:.4f}")

    # 如果使用全部数据训练，显示特殊提示
    if CONFIG['use_full_data'] or CONFIG['test_size'] == 0:
        print("\n⚠️  注意：使用了全部数据进行训练，验证集=训练集")
        print("   这种配置会产生过拟合，但可以获得最佳的训练集表现")
        if results.get('train_predictions') is not None:
            train_preds, train_labels, _ = results['train_predictions']
            train_acc = accuracy_score(train_labels, train_preds)
            train_f1 = f1_score(train_labels, train_preds, average='macro')
            print(f"   训练集 Accuracy: {train_acc:.4f}")
            print(f"   训练集 F1-macro: {train_f1:.4f}")

    print(f"\n结果保存在: {output_dir}")
    print(f"⭐ 地图可视化文件: grid_predictions.csv（包含所有格网的预测结果）")


if __name__ == '__main__':
    main()
