#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用模型生成的表征向量（embeddings）进行预测，使用LightGBM模型。
支持房价预测、活力预测和土地利用分类三种任务。

直接在代码中修改下面的配置参数即可运行。
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
# 配置参数 - 在这里修改参数即可
# ============================================================================
CONFIG = {
    # 任务类型配置
    # 可选: 'house_price' (房价预测), 'vitality' (活力预测), 或 'landuse' (土地利用分类)
    'task': 'house_price',  # 修改这里切换任务类型

    # 文件路径配置
    'embeddings_path': 'checkpoints/train_20260410_214956/best_embeddings.npz',
    'price_data_path': '深圳房价数据_对齐网格.csv',  # 房价数据路径
    'vitality_data_path': '20190622-20190626_按GWBH聚合.csv',  # 活力数据路径
    'landuse_data_path': '土地利用_对齐网格_新.csv',  # 土地利用数据路径
    'output_dir': None,  # 如果为None，将根据task自动设置

    # Embedding配置
    # 可选: 'h', 'concat', 'z', 'h_spatial', 'h_od', 'embeddings'等
    'embedding_key': 'h',
    # 当embedding_key='h'或'concat'时，用于拼接的key列表
    'concat_keys': ['h_spatial', 'h_od'],

    # 数据划分配置
    'test_size': 0.2,  # 测试集比例
    'random_state': 42,  # 随机种子

    # 特征工程配置
    'use_pca': True,  # 是否使用PCA降维
    'pca_n_components': 128,  # PCA降维后的固定维度
    'use_log_transform': True,  # 是否对目标变量进行log变换（在log空间计算指标，不还原）

    # 土地利用分类配置
    'valid_classes': [0, 1, 2, 3, 4, 5],  # 要使用的类别，None表示使用全部类别

    # LightGBM模型参数（回归任务）
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
    # LightGBM模型参数（分类任务）
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
    'num_boost_round': 500,  # 最大迭代次数
    'early_stopping_rounds': 100,  # 早停轮数
    'log_evaluation_period': 50,  # 日志输出周期
}
# ============================================================================


def _concat_embeddings(data: "np.lib.npyio.NpzFile", keys: List[str]) -> np.ndarray:
    """将多个二维嵌入矩阵按特征维度拼接。

    要求：
    - 每个 key 对应二维 ndarray，形状 (N, D_i)
    - 所有数组的 N 相同
    """
    arrays: List[np.ndarray] = []
    n_rows: Optional[int] = None
    for k in keys:
        if k not in getattr(data, "files", []):
            raise KeyError(
                f"npz 中不存在 key='{k}'，可用 keys: {list(getattr(data, 'files', []))}")
        arr = data[k]
        if not isinstance(arr, np.ndarray) or arr.ndim != 2:
            raise ValueError(
                f"key='{k}' 对应的数据不是二维 ndarray。"
                f"实际类型={type(arr)}, ndim={getattr(arr, 'ndim', None)}, shape={getattr(arr, 'shape', None)}"
            )
        if n_rows is None:
            n_rows = arr.shape[0]
        elif arr.shape[0] != n_rows:
            raise ValueError(
                f"拼接失败：key='{k}' 的行数({arr.shape[0]})与其他嵌入行数({n_rows})不一致。"
            )
        arrays.append(arr)
    return np.concatenate(arrays, axis=1)


def load_embeddings(
    path: str,
    embedding_key: Optional[str] = None,
    concat_keys: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """从 npz 文件加载节点 ID 与嵌入矩阵"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到嵌入文件: {path}")

    data = np.load(path, allow_pickle=True)
    keys = list(getattr(data, "files", []))

    requested = (embedding_key or "h").strip()

    # 特殊模式：拼接得到 h（默认拼接 h_spatial + h_od）
    if requested.lower() in {"h", "concat"}:
        concat_keys = concat_keys or ["h_spatial", "h_od"]
        embeddings = _concat_embeddings(data, concat_keys)
        key_to_use = f"concat({'+'.join(concat_keys)})"
        print(f"[信息] 使用拼接嵌入作为输入：{key_to_use}，shape={embeddings.shape}")
    else:
        # 使用指定的key
        if requested in keys:
            key_to_use = requested
        else:
            # 如果指定的key不存在，尝试自动选择
            if "embeddings" in keys:
                key_to_use = "embeddings"
                print(f"[警告] 指定的key='{requested}'不存在，将使用 'embeddings'")
            else:
                # 找到最大的二维数组
                candidates = []
                for k in keys:
                    try:
                        arr = data[k]
                        if isinstance(arr, np.ndarray) and arr.ndim == 2:
                            candidates.append((arr.shape[1], k))
                    except Exception:
                        continue
                if not candidates:
                    raise KeyError(f"无法在 {keys} 中找到可用的二维嵌入矩阵")
                candidates.sort(reverse=True)
                key_to_use = candidates[0][1]
                print(f"[警告] 指定的key='{requested}'不存在，将使用 '{key_to_use}'")

        embeddings = data[key_to_use]
        if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
            raise ValueError(
                f"key='{key_to_use}' 对应的数据不是二维 ndarray。"
                f"实际类型={type(embeddings)}, ndim={getattr(embeddings, 'ndim', None)}, shape={getattr(embeddings, 'shape', None)}"
            )
        print(f"[信息] 使用嵌入key: '{key_to_use}', shape={embeddings.shape}")

    # 获取node_ids
    node_ids = data.get('node_ids', np.arange(len(embeddings)))

    # 将node_ids转换为整数类型（统一格式）
    if node_ids.dtype.kind in {'S', 'O', 'U'}:
        # 如果是字符串，尝试转换为整数
        node_ids = np.array([int(float(x)) if x else 0 for x in node_ids])
    else:
        # 如果是数字，转换为整数
        node_ids = node_ids.astype(int)

    # 确保长度一致
    if len(node_ids) != len(embeddings):
        print(
            f"[警告] node_ids 长度({len(node_ids)})与 embeddings 行数({len(embeddings)})不一致，将使用 0..N-1 作为 node_ids")
        node_ids = np.arange(len(embeddings), dtype=int)

    return node_ids, embeddings


def load_price_data(path: str) -> pd.DataFrame:
    """加载房价数据"""
    df = pd.read_csv(path)
    # 清理数据：移除GWBH为空的行
    df = df[df['GWBH'].notna() & (df['GWBH'] != '')]
    # 移除avgprice为空的行
    df = df[df['avgprice'].notna()]
    # 将GWBH转换为整数（先转float再转int，处理可能的NaN）
    df['GWBH'] = pd.to_numeric(df['GWBH'], errors='coerce')
    df = df[df['GWBH'].notna()]
    df['GWBH'] = df['GWBH'].astype(int)
    return df


def load_vitality_data(path: str) -> pd.DataFrame:
    """加载活力数据"""
    df = pd.read_csv(path)
    # 清理数据：移除GWBH为空的行
    df = df[df['GWBH'].notna() & (df['GWBH'] != '')]
    # 移除total_people为空的行
    df = df[df['total_people'].notna()]
    # 将GWBH转换为整数（先转float再转int，处理可能的NaN）
    df['GWBH'] = pd.to_numeric(df['GWBH'], errors='coerce')
    df = df[df['GWBH'].notna()]
    df['GWBH'] = df['GWBH'].astype(int)
    return df


def load_landuse_data(path: str, valid_classes: Optional[List[int]] = None) -> pd.DataFrame:
    """加载土地利用数据"""
    df = pd.read_csv(path)
    # 清理数据：移除GWBH为空的行
    df = df[df['GWBH'].notna() & (df['GWBH'] != '')]
    # 移除landuse为空的行
    df = df[df['landuse'].notna()]
    # 将GWBH转换为整数（先转float再转int，处理可能的NaN）
    df['GWBH'] = pd.to_numeric(df['GWBH'], errors='coerce')
    df = df[df['GWBH'].notna()]
    df['GWBH'] = df['GWBH'].astype(int)
    df['landuse'] = df['landuse'].astype(int)

    # 过滤类别
    if valid_classes is not None:
        df = df[df['landuse'].isin(valid_classes)]
        print(f"[信息] 过滤后保留类别: {valid_classes}, 剩余 {len(df)} 条记录")

    # 打印类别分布
    print(f"[信息] 类别分布: {df['landuse'].value_counts().sort_index().to_dict()}")

    return df


def merge_data(
    node_ids: np.ndarray,
    embeddings: np.ndarray,
    target_df: pd.DataFrame,
    target_column: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """将embeddings和目标数据通过GWBH/node_ids进行匹配

    Returns:
        X: 特征矩阵
        y: 目标值
        gwbhs: 匹配的GWBH列表
    """
    # 创建embeddings的DataFrame
    emb_df = pd.DataFrame({
        'GWBH': node_ids,
        'embedding_idx': range(len(node_ids))
    })

    # 合并数据
    merged = target_df.merge(emb_df, on='GWBH', how='inner')

    if len(merged) == 0:
        raise ValueError("无法匹配任何数据！请检查GWBH格式是否一致")

    data_name = "目标数据" if target_column == 'total_people' else "房价数据"
    print(
        f"[信息] 成功匹配 {len(merged)} 条数据（{data_name}: {len(target_df)}, embeddings: {len(node_ids)}）")

    # 获取匹配的embeddings和目标值
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
    """使用LightGBM训练分类模型并评估（用于土地利用分类）

    Args:
        X: 特征矩阵
        y: 目标标签
        gwbhs: GWBH列表，用于保存每个格网的预测结果
    """
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test, gwbh_train, gwbh_test = train_test_split(
        X, y, gwbhs, test_size=test_size, random_state=random_state
    )

    print(f"\n[信息] 训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
    print(f"[信息] 原始特征维度: {X.shape[1]}")
    print(
        f"[信息] 类别数量: {len(np.unique(y))}, 类别: {sorted(np.unique(y).tolist())}")
    print(
        f"[信息] 训练集类别分布: {pd.Series(y_train).value_counts().sort_index().to_dict()}")
    print(
        f"[信息] 测试集类别分布: {pd.Series(y_test).value_counts().sort_index().to_dict()}")

    # PCA降维到固定维度
    pca = None
    if use_pca:
        print(f"\n[信息] 应用PCA降维到 {pca_n_components} 维...")
        pca = PCA(n_components=pca_n_components, random_state=random_state)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        print(f"[信息] 降维后特征维度: {X_train.shape[1]}")
        print(f"[信息] 累计方差解释率: {pca.explained_variance_ratio_.sum():.4f}")

    # 创建类别映射（原始类别 -> 0,1,2,...）
    unique_classes = sorted(np.unique(y))
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    # 转换标签
    y_train_mapped = np.array([class_to_idx[c] for c in y_train])
    y_test_mapped = np.array([class_to_idx[c] for c in y_test])

    print(f"\n[信息] 类别映射: {class_to_idx}")

    # 创建LightGBM数据集
    num_classes = len(unique_classes)
    train_data = lgb.Dataset(X_train, label=y_train_mapped)
    test_data = lgb.Dataset(X_test, label=y_test_mapped, reference=train_data)

    # 设置参数
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

    # 添加random_state
    params = {**lgb_params, 'random_state': random_state}

    # 训练模型
    print("\n[信息] 开始训练LightGBM分类模型...")
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

    # 预测（预测概率）
    y_train_pred_proba = model.predict(
        X_train, num_iteration=model.best_iteration)
    y_test_pred_proba = model.predict(
        X_test, num_iteration=model.best_iteration)

    # 转换为类别预测（映射后的类别）
    y_train_pred_mapped = np.argmax(y_train_pred_proba, axis=1)
    y_test_pred_mapped = np.argmax(y_test_pred_proba, axis=1)

    # 转换回原始类别
    y_train_pred = np.array([idx_to_class[i] for i in y_train_pred_mapped])
    y_test_pred = np.array([idx_to_class[i] for i in y_test_pred_mapped])

    # 计算指标
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_f1_macro = f1_score(y_train, y_train_pred, average='macro')
    train_f1_weighted = f1_score(y_train, y_train_pred, average='weighted')

    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1_macro = f1_score(y_test, y_test_pred, average='macro')
    test_f1_weighted = f1_score(y_test, y_test_pred, average='weighted')

    # 生成分类报告和混淆矩阵
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
        # 保存每个样本的预测结果
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
    """使用LightGBM训练模型并评估

    Args:
        X: 特征矩阵
        y: 目标值
        gwbhs: GWBH列表，用于保存每个格网的预测结果
    """
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test, gwbh_train, gwbh_test = train_test_split(
        X, y, gwbhs, test_size=test_size, random_state=random_state
    )

    print(f"\n[信息] 训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
    print(f"[信息] 原始特征维度: {X.shape[1]}")
    print(
        f"[信息] 目标变量范围: [{y.min():.2f}, {y.max():.2f}], 均值: {y.mean():.2f}, 标准差: {y.std():.2f}")

    # 保存原始目标值（在log变换之前，用于后续保存和可视化）
    y_train_original = y_train.copy()
    y_test_original = y_test.copy()

    # PCA降维到固定维度
    pca = None
    if use_pca:
        print(f"\n[信息] 应用PCA降维到 {pca_n_components} 维...")
        pca = PCA(n_components=pca_n_components, random_state=random_state)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        print(f"[信息] 降维后特征维度: {X_train.shape[1]}")
        print(f"[信息] 累计方差解释率: {pca.explained_variance_ratio_.sum():.4f}")

    # Log变换（在log空间计算指标，不还原）
    if use_log_transform:
        print(f"\n[信息] 对目标变量应用log1p变换（在log空间计算指标）...")
        y_train = np.log1p(y_train)
        y_test = np.log1p(y_test)
        print(f"[信息] log变换后训练集范围: [{y_train.min():.4f}, {y_train.max():.4f}]")

    # 创建LightGBM数据集
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # 设置参数（使用传入的参数，如果没有则使用默认值）
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
    # 添加random_state
    params = {**lgb_params, 'random_state': random_state}

    # 训练模型
    print("\n[信息] 开始训练LightGBM模型...")
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

    # 预测（在log空间）
    y_train_pred_log = model.predict(
        X_train, num_iteration=model.best_iteration)
    y_test_pred_log = model.predict(X_test, num_iteration=model.best_iteration)

    # 计算指标（在log空间，不还原到原始价格空间）
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred_log))
    train_mae = mean_absolute_error(y_train, y_train_pred_log)
    train_r2 = r2_score(y_train, y_train_pred_log)

    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_log))
    test_mae = mean_absolute_error(y_test, y_test_pred_log)
    test_r2 = r2_score(y_test, y_test_pred_log)

    # 将预测值还原到原始空间（用于保存和可视化）
    if use_log_transform:
        y_train_pred = np.expm1(y_train_pred_log)
        y_test_pred = np.expm1(y_test_pred_log)
    else:
        y_train_pred = y_train_pred_log
        y_test_pred = y_test_pred_log

    # 计算绝对误差（在原始空间）
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
        'pca': pca,  # 保存PCA对象
        # 保存每个样本的预测结果（在原始空间）
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
    """打印评估结果"""
    print("\n" + "="*60)

    if task == 'landuse':
        task_name = "土地利用分类"
        print(f"{task_name}结果评估")
        print("="*60)

        print("\n训练集指标:")
        print(f"  准确率 (Accuracy):    {results['train']['accuracy']:.4f}")
        print(f"  F1分数 (Macro):       {results['train']['f1_macro']:.4f}")
        print(f"  F1分数 (Weighted):    {results['train']['f1_weighted']:.4f}")

        print("\n测试集指标:")
        print(f"  准确率 (Accuracy):    {results['test']['accuracy']:.4f}")
        print(f"  F1分数 (Macro):       {results['test']['f1_macro']:.4f}")
        print(f"  F1分数 (Weighted):    {results['test']['f1_weighted']:.4f}")

        print("\n详细分类报告 (测试集):")
        print("-" * 60)
        test_report = results['test_report']
        for class_label in sorted([k for k in test_report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]):
            metrics = test_report[class_label]
            print(f"  类别 {class_label}:")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall:    {metrics['recall']:.4f}")
            print(f"    F1-Score:  {metrics['f1-score']:.4f}")
            print(f"    Support:   {int(metrics['support'])}")

    else:
        task_name = "活力预测" if task == 'vitality' else "房价预测"
        print(f"{task_name}结果评估")
        if use_log_transform:
            space_name = "活力空间" if task == 'vitality' else "价格空间"
            print(f"(注意: 指标在log空间计算，未还原到原始{space_name})")
        print("="*60)

        print("\n训练集指标:")
        print(f"  RMSE: {results['train']['RMSE']:.4f}")
        print(f"  MAE:  {results['train']['MAE']:.4f}")
        print(f"  R²:   {results['train']['R²']:.4f}")

        print("\n测试集指标:")
        print(f"  RMSE: {results['test']['RMSE']:.4f}")
        print(f"  MAE:  {results['test']['MAE']:.4f}")
        print(f"  R²:   {results['test']['R²']:.4f}")

    print("="*60 + "\n")


def save_results(results: dict, output_dir: str, task: str = 'house_price'):
    """保存结果到文件"""
    os.makedirs(output_dir, exist_ok=True)

    if task == 'landuse':
        # 分类任务
        # 保存指标到CSV
        metrics_df = pd.DataFrame({
            'Dataset': ['Train', 'Test'],
            'Accuracy': [results['train']['accuracy'], results['test']['accuracy']],
            'F1_Macro': [results['train']['f1_macro'], results['test']['f1_macro']],
            'F1_Weighted': [results['train']['f1_weighted'], results['test']['f1_weighted']]
        })
        metrics_path = os.path.join(output_dir, 'metrics.csv')
        metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')
        print(f"[信息] 指标已保存到: {metrics_path}")

        # 保存类别映射
        class_mapping_path = os.path.join(output_dir, 'class_mapping.json')
        with open(class_mapping_path, 'w', encoding='utf-8') as f:
            json.dump({str(k): v for k, v in results['class_mapping'].items(
            )}, f, indent=2, ensure_ascii=False)
        print(f"[信息] 类别映射已保存到: {class_mapping_path}")

        # 保存分类报告
        test_report_path = os.path.join(
            output_dir, 'classification_report.json')
        with open(test_report_path, 'w', encoding='utf-8') as f:
            json.dump(results['test_report'], f, indent=2, ensure_ascii=False)
        print(f"[信息] 分类报告已保存到: {test_report_path}")

        # 保存混淆矩阵
        classes = sorted(results['class_mapping'].keys())
        cm_df = pd.DataFrame(
            results['test_confusion_matrix'],
            index=[f'True_{c}' for c in classes],
            columns=[f'Pred_{c}' for c in classes]
        )
        cm_path = os.path.join(output_dir, 'confusion_matrix.csv')
        cm_df.to_csv(cm_path, encoding='utf-8-sig')
        print(f"[信息] 混淆矩阵已保存到: {cm_path}")

        # 训练集预测结果
        train_pred_df = pd.DataFrame({
            'GWBH': results['train_predictions']['GWBH'],
            'y_true': results['train_predictions']['y_true'],
            'y_pred': results['train_predictions']['y_pred'],
            'correct': (results['train_predictions']['y_true'] == results['train_predictions']['y_pred']).astype(int)
        })
        train_pred_path = os.path.join(output_dir, 'train_predictions.csv')
        train_pred_df.to_csv(train_pred_path, index=False,
                             encoding='utf-8-sig')
        print(f"[信息] 训练集预测结果已保存到: {train_pred_path}")

        # 测试集预测结果
        test_pred_df = pd.DataFrame({
            'GWBH': results['test_predictions']['GWBH'],
            'y_true': results['test_predictions']['y_true'],
            'y_pred': results['test_predictions']['y_pred'],
            'correct': (results['test_predictions']['y_true'] == results['test_predictions']['y_pred']).astype(int)
        })
        test_pred_path = os.path.join(output_dir, 'test_predictions.csv')
        test_pred_df.to_csv(test_pred_path, index=False, encoding='utf-8-sig')
        print(f"[信息] 测试集预测结果已保存到: {test_pred_path}")

        # 合并训练集和测试集
        all_pred_df = pd.concat([
            train_pred_df.assign(split='train'),
            test_pred_df.assign(split='test')
        ], ignore_index=True)
        all_pred_path = os.path.join(output_dir, 'all_predictions.csv')
        all_pred_df.to_csv(all_pred_path, index=False, encoding='utf-8-sig')
        print(f"[信息] 所有格网预测结果已保存到: {all_pred_path} (包含split列标识训练/测试集)")

    else:
        # 回归任务
        # 保存指标到CSV
        metrics_df = pd.DataFrame({
            'Dataset': ['Train', 'Test'],
            'RMSE': [results['train']['RMSE'], results['test']['RMSE']],
            'MAE': [results['train']['MAE'], results['test']['MAE']],
            'R²': [results['train']['R²'], results['test']['R²']]
        })
        metrics_path = os.path.join(output_dir, 'metrics.csv')
        metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')
        print(f"[信息] 指标已保存到: {metrics_path}")

        # 保存每个格网的预测结果（训练集和测试集）
        use_log_transform = results.get('use_log_transform', False)

        # 训练集预测结果
        train_pred_df = pd.DataFrame({
            'GWBH': results['train_predictions']['GWBH'],
            'y_true': results['train_predictions']['y_true'],
            'y_pred': results['train_predictions']['y_pred'],
            'abs_error': results['train_predictions']['abs_error']
        })
        train_pred_path = os.path.join(output_dir, 'train_predictions.csv')
        train_pred_df.to_csv(train_pred_path, index=False,
                             encoding='utf-8-sig')
        print(f"[信息] 训练集预测结果已保存到: {train_pred_path}")

        # 测试集预测结果
        test_pred_df = pd.DataFrame({
            'GWBH': results['test_predictions']['GWBH'],
            'y_true': results['test_predictions']['y_true'],
            'y_pred': results['test_predictions']['y_pred'],
            'abs_error': results['test_predictions']['abs_error']
        })
        test_pred_path = os.path.join(output_dir, 'test_predictions.csv')
        test_pred_df.to_csv(test_pred_path, index=False, encoding='utf-8-sig')
        print(f"[信息] 测试集预测结果已保存到: {test_pred_path}")

        # 合并训练集和测试集，保存所有格网的预测结果（用于地图可视化）
        all_pred_df = pd.concat([
            train_pred_df.assign(split='train'),
            test_pred_df.assign(split='test')
        ], ignore_index=True)
        all_pred_path = os.path.join(output_dir, 'all_predictions.csv')
        all_pred_df.to_csv(all_pred_path, index=False, encoding='utf-8-sig')
        print(f"[信息] 所有格网预测结果已保存到: {all_pred_path} (包含split列标识训练/测试集)")

        space_name = "活力空间" if task == 'vitality' else "价格空间"
        if use_log_transform:
            print(f"[注意] 虽然训练时使用了log变换，但保存的预测值已还原到原始{space_name}，可直接用于地图可视化")
        else:
            print(f"[信息] 预测值在原始{space_name}，可直接用于地图可视化")

    # 保存模型
    model_path = os.path.join(output_dir, 'lightgbm_model.txt')
    results['model'].save_model(model_path)
    print(f"[信息] 模型已保存到: {model_path}")


def main():
    # 使用代码中的配置（可以通过命令行参数覆盖）
    parser = argparse.ArgumentParser(description='使用表征向量进行预测（支持房价、活力和土地利用分类）')
    parser.add_argument(
        '--task',
        type=str,
        choices=['house_price', 'vitality', 'landuse'],
        default=None,
        help='任务类型：house_price（房价预测）、vitality（活力预测）或 landuse（土地利用分类），如果不指定则使用CONFIG中的配置'
    )
    parser.add_argument(
        '--embeddings',
        type=str,
        default=None,
        help='embeddings文件路径（.npz格式），如果不指定则使用CONFIG中的配置'
    )
    parser.add_argument(
        '--price_data',
        type=str,
        default=None,
        help='房价数据文件路径（CSV格式），如果不指定则使用CONFIG中的配置'
    )
    parser.add_argument(
        '--vitality_data',
        type=str,
        default=None,
        help='活力数据文件路径（CSV格式），如果不指定则使用CONFIG中的配置'
    )
    parser.add_argument(
        '--landuse_data',
        type=str,
        default=None,
        help='土地利用数据文件路径（CSV格式），如果不指定则使用CONFIG中的配置'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='结果输出目录，如果不指定则使用CONFIG中的配置'
    )
    parser.add_argument(
        '--embedding_key',
        type=str,
        default=None,
        help='要使用的embedding key，如果不指定则使用CONFIG中的配置'
    )
    parser.add_argument(
        '--concat_keys',
        type=str,
        nargs='+',
        default=None,
        help='当embedding_key为h或concat时，用于拼接的key列表，如果不指定则使用CONFIG中的配置'
    )

    args = parser.parse_args()

    # 从CONFIG获取配置，命令行参数可以覆盖
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

    # 如果output_dir为None，根据task自动设置
    if output_dir is None:
        # 从embeddings路径提取checkpoint名称
        checkpoint_name = os.path.basename(os.path.dirname(embeddings_path))
        if task == 'vitality':
            task_name = 'vitality_prediction'
        elif task == 'landuse':
            task_name = 'landuse_lgb'
        else:
            task_name = 'house_price_prediction'
        output_dir = f'results/{checkpoint_name}/{task_name}'

    # 根据任务类型选择数据路径和目标列
    if task == 'vitality':
        target_data_path = vitality_data_path
        target_column = 'total_people'
        task_display_name = '活力预测'
    elif task == 'landuse':
        target_data_path = landuse_data_path
        target_column = 'landuse'
        task_display_name = '土地利用分类'
    else:
        target_data_path = price_data_path
        target_column = 'avgprice'
        task_display_name = '房价预测'

    print("="*60)
    print("配置信息:")
    print(f"  任务类型: {task_display_name}")
    print(f"  Embeddings路径: {embeddings_path}")
    print(f"  目标数据路径: {target_data_path}")
    print(f"  输出目录: {output_dir}")
    print(f"  Embedding key: {embedding_key}")
    if embedding_key.lower() in {'h', 'concat'}:
        print(f"  拼接keys: {concat_keys}")
    print(f"  测试集比例: {test_size}")
    print(f"  随机种子: {random_state}")
    print(f"  使用PCA降维: {use_pca} (降维到 {pca_n_components} 维)")
    if task != 'landuse':
        print(f"  使用Log变换: {use_log_transform} (在log空间计算指标)")
    if task == 'landuse':
        print(f"  过滤类别: {valid_classes if valid_classes else '全部类别'}")
    print("="*60)

    # 加载数据
    print("\n[信息] 加载embeddings...")
    node_ids, embeddings = load_embeddings(
        embeddings_path,
        embedding_key=embedding_key,
        concat_keys=concat_keys
    )

    # 根据任务类型加载不同的数据
    if task == 'vitality':
        print("[信息] 加载活力数据...")
        target_df = load_vitality_data(target_data_path)
    elif task == 'landuse':
        print("[信息] 加载土地利用数据...")
        target_df = load_landuse_data(
            target_data_path, valid_classes=valid_classes)
    else:
        print("[信息] 加载房价数据...")
        target_df = load_price_data(target_data_path)

    # 合并数据
    print("[信息] 合并数据...")
    X, y, gwbhs = merge_data(node_ids, embeddings, target_df, target_column)

    # 训练和评估
    if task == 'landuse':
        # 分类任务
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
        # 回归任务
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

    # 打印结果
    print_results(results, task=task,
                  use_log_transform=use_log_transform if task != 'landuse' else False)

    # 保存结果
    save_results(results, output_dir, task=task)


if __name__ == '__main__':
    main()
