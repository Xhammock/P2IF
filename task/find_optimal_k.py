#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用多种指标（肘部法则、轮廓系数、Davies-Bouldin Index、Calinski-Harabasz Index）
来确定 K-Means 聚类的最佳 k 值。

示例：
    python find_optimal_k.py \
        --embeddings checkpoints/train_20251202_200939/fused_embeddings.npz \
        --k_min 2 \
        --k_max 30 \
        --output_dir checkpoints/train_20251202_200939/k_selection
"""

import argparse
import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from sklearn.preprocessing import StandardScaler

# ========== 配置区域：可直接在此修改默认值 ==========
# 如果不想通过命令行传参，可以直接修改下面的配置
DEFAULT_CONFIG = {
    'embeddings': 'checkpoints/train_20251222_220544/best_embeddings.npz',  # 嵌入文件路径
    'embedding_key': 'h',  # 使用的嵌入key: 'h_spatial', 'h_od', 'z', 或 'h'/'concat'（拼接模式）
    'concat_keys': ['h_spatial', 'h_od'],  # 当 embedding_key='h' 时，用于拼接的key列表
    'k_min': 2,  # k值的最小值
    'k_max': 8,  # k值的最大值
    'output_dir': 'checkpoints/train_20251222_220544/k_selection',  # 输出目录路径
    'random_state': 42,  # 随机种子
    'standardize': False,  # 是否标准化特征（默认False，因为嵌入通常已归一化）
    'use_pca': True,  # 是否使用PCA降维
    'pca_components': 128,  # PCA降维后的维度
}
# ====================================================


def _pick_embedding_key(data: "np.lib.npyio.NpzFile") -> str:
    """从 npz 文件中选择一个最像“嵌入矩阵”的 key。

    选择规则（从高到低）：
    - 若存在 'embeddings'，优先使用
    - 若存在 'z'，其次使用（很多模型会用 z 表示融合后的表征）
    - 否则在所有二维数组中选 embedding 维度（shape[1]）最大的那个
    """
    keys = list(getattr(data, "files", []))
    if "embeddings" in keys:
        return "embeddings"
    if "z" in keys:
        return "z"

    candidates = []
    for k in keys:
        try:
            arr = data[k]
        except Exception:
            continue
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] > 0:
            candidates.append((arr.shape[1], k))

    if not candidates:
        raise KeyError(
            f"无法在 {keys} 中找到可用的二维嵌入矩阵。"
            f"请确认 npz 内包含形如 (N, D) 的数组，或用 --embedding_key 显式指定。"
        )

    candidates.sort(reverse=True)
    return candidates[0][1]


def _concat_embeddings(data: "np.lib.npyio.NpzFile", keys: list) -> np.ndarray:
    """将多个二维嵌入矩阵按特征维度拼接。"""
    arrays = []
    n_rows = None
    available = list(getattr(data, "files", []))
    for k in keys:
        if k not in available:
            raise KeyError(f"npz 中不存在 key='{k}'，可用 keys: {available}")
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
    concat_keys: Optional[list] = None
) -> np.ndarray:
    """从 npz 文件加载嵌入矩阵（二维 ndarray）"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到嵌入文件: {path}")

    data = np.load(path, allow_pickle=True)
    keys = list(getattr(data, "files", []))

    requested = (embedding_key or "embeddings").strip()

    # 特殊模式：拼接得到 h（默认拼接 h_spatial + h_od）
    if requested.lower() in {"h", "concat"}:
        concat_keys = concat_keys or ["h_spatial", "h_od"]
        embeddings = _concat_embeddings(data, concat_keys)
        print(
            f"[提示] 使用拼接嵌入作为输入：concat({'+'.join(concat_keys)})，shape={embeddings.shape}")
        return embeddings

    if requested in keys:
        key_to_use = requested
    else:
        key_to_use = _pick_embedding_key(data)
        print(
            f"[提示] npz 中未找到 key='{requested}'，将改用 key='{key_to_use}'。"
            f"可用 keys: {keys}"
        )

    embeddings = data[key_to_use]
    if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
        raise ValueError(
            f"key='{key_to_use}' 对应的数据不是二维 ndarray。"
            f"实际类型={type(embeddings)}, ndim={getattr(embeddings, 'ndim', None)}, shape={getattr(embeddings, 'shape', None)}"
        )
    return embeddings


def compute_metrics(embeddings: np.ndarray, k: int, random_state: int) -> Tuple[float, float, float, float]:
    """对给定的k值计算四种评估指标

    Returns:
        inertia: 惯性值/簇内平方和 (WCSS) - 用于肘部法则，越小越好
        silhouette: Silhouette Score (越大越好，范围[-1, 1])
        dbi: Davies-Bouldin Index (越小越好，>=0)
        ch: Calinski-Harabasz Index (越大越好，>=0)
    """
    kmeans = KMeans(
        n_clusters=k,
        n_init=10,
        max_iter=300,
        random_state=random_state
    )
    labels = kmeans.fit_predict(embeddings)
    inertia = kmeans.inertia_  # 簇内平方和（WCSS）

    # 计算三种指标
    silhouette = silhouette_score(embeddings, labels)
    dbi = davies_bouldin_score(embeddings, labels)
    ch = calinski_harabasz_score(embeddings, labels)

    return inertia, silhouette, dbi, ch


def find_optimal_k(
    embeddings: np.ndarray,
    k_min: int,
    k_max: int,
    random_state: int
) -> Tuple[list, list, list, list, list]:
    """遍历k范围，计算所有指标

    Returns:
        k_values: k值列表
        inertia_scores: 惯性值列表（用于肘部法则）
        silhouette_scores: Silhouette分数列表
        dbi_scores: DBI分数列表
        ch_scores: CH分数列表
    """
    k_values = list(range(k_min, k_max + 1))
    inertia_scores = []
    silhouette_scores = []
    dbi_scores = []
    ch_scores = []

    print(f"正在评估 k 值范围: {k_min} 到 {k_max}...")
    for k in k_values:
        inertia, silhouette, dbi, ch = compute_metrics(
            embeddings, k, random_state)
        inertia_scores.append(inertia)
        silhouette_scores.append(silhouette)
        dbi_scores.append(dbi)
        ch_scores.append(ch)
        print(
            f"  k={k:2d}: Inertia={inertia:.2f}, Silhouette={silhouette:.4f}, DBI={dbi:.4f}, CH={ch:.4f}")

    return k_values, inertia_scores, silhouette_scores, dbi_scores, ch_scores


def find_elbow_point(inertia_scores: list, k_values: list) -> int:
    """使用肘部法则找到最佳k值

    肘部法则：寻找惯性值下降速率突然变缓的点（肘部）
    通过计算相邻k值之间惯性值下降的百分比变化率来找到肘部

    Returns:
        推荐的k值（肘部点）
    """
    if len(inertia_scores) < 3:
        # 如果k值太少，无法计算肘部，返回中间值
        return k_values[len(k_values) // 2]

    # 计算相邻k值之间惯性值的下降率
    decreases = []
    for i in range(1, len(inertia_scores)):
        if inertia_scores[i-1] > 0:
            decrease_rate = (
                inertia_scores[i-1] - inertia_scores[i]) / inertia_scores[i-1]
            decreases.append(decrease_rate)
        else:
            decreases.append(0)

    # 计算下降率的变化（二阶导数）
    # 肘部通常是下降率变化最大的点
    if len(decreases) < 2:
        return k_values[1] if len(k_values) > 1 else k_values[0]

    # 计算下降率的加速度（变化率）
    accelerations = []
    for i in range(1, len(decreases)):
        acceleration = decreases[i-1] - decreases[i]  # 下降率本身在减小
        accelerations.append(acceleration)

    # 找到加速度最大的点（下降率变化最明显的地方）
    if accelerations:
        # +1 因为accelerations比k_values少一个元素
        elbow_idx = np.argmax(accelerations) + 1
        return k_values[min(elbow_idx, len(k_values) - 1)]
    else:
        return k_values[1] if len(k_values) > 1 else k_values[0]


def suggest_optimal_k(
    k_values: list,
    inertia_scores: list,
    silhouette_scores: list,
    dbi_scores: list,
    ch_scores: list
) -> dict:
    """根据四种指标综合建议最佳k值

    Returns:
        包含各指标推荐k值和综合建议的字典
    """
    # 肘部法则：找到肘部点
    best_k_elbow = find_elbow_point(inertia_scores, k_values)
    best_inertia_at_elbow = inertia_scores[k_values.index(best_k_elbow)]

    # Silhouette: 选最大值
    best_k_silhouette = k_values[np.argmax(silhouette_scores)]
    best_silhouette_score = max(silhouette_scores)

    # DBI: 选最小值
    best_k_dbi = k_values[np.argmin(dbi_scores)]
    best_dbi_score = min(dbi_scores)

    # CH: 选最大值
    best_k_ch = k_values[np.argmax(ch_scores)]
    best_ch_score = max(ch_scores)

    # 综合建议：如果多个指标指向相近的k值，选择较小的（更简单）
    recommendations = [best_k_elbow, best_k_silhouette, best_k_dbi, best_k_ch]
    k_range = max(recommendations) - min(recommendations)

    if k_range <= 3:  # 如果多个指标推荐的k值相差不超过3
        # 选择中位数或较小的值
        suggested_k = min(recommendations)
        reason = "多个指标一致指向相近的k值"
    else:
        # 如果差异较大，优先考虑肘部法则和轮廓系数
        # 取肘部法则和轮廓系数的平均值（如果差异不大）
        if abs(best_k_elbow - best_k_silhouette) <= 2:
            suggested_k = int(np.mean([best_k_elbow, best_k_silhouette]))
            reason = "肘部法则和轮廓系数指向相近的k值"
        else:
            # 优先采用轮廓系数（更常用且更稳定）
            suggested_k = best_k_silhouette
            reason = "各指标差异较大，优先采用轮廓系数（Silhouette Score）"

    return {
        'elbow': {'k': best_k_elbow, 'score': best_inertia_at_elbow},
        'silhouette': {'k': best_k_silhouette, 'score': best_silhouette_score},
        'dbi': {'k': best_k_dbi, 'score': best_dbi_score},
        'ch': {'k': best_k_ch, 'score': best_ch_score},
        'suggested': {'k': suggested_k, 'reason': reason}
    }


def visualize_k_selection(
    k_values: list,
    inertia_scores: list,
    silhouette_scores: list,
    dbi_scores: list,
    ch_scores: list,
    recommendations: dict,
    output_dir: str
):
    """绘制四种指标的曲线图"""
    os.makedirs(output_dir, exist_ok=True)

    # 创建2x3的子图布局（增加一个肘部法则图）
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('K-Means 最佳 k 值选择分析（肘部法则 + 轮廓系数 + 其他指标）',
                 fontsize=16, fontweight='bold')

    # 1. 肘部法则（Elbow Method）
    ax0 = axes[0, 0]
    ax0.plot(k_values, inertia_scores, 'o-',
             color='orange', linewidth=2, markersize=6)
    best_k_elbow = recommendations['elbow']['k']
    best_inertia_elbow = recommendations['elbow']['score']
    ax0.axvline(best_k_elbow, color='red', linestyle='--',
                linewidth=1.5, alpha=0.7)
    ax0.plot(best_k_elbow, best_inertia_elbow, 'ro', markersize=10)
    ax0.set_xlabel('k (聚类数量)', fontsize=11)
    ax0.set_ylabel('惯性值 (Inertia / WCSS)', fontsize=11)
    ax0.set_title(
        f'肘部法则 (Elbow Method, 最佳 k={best_k_elbow})', fontsize=12, fontweight='bold')
    ax0.grid(True, alpha=0.3)
    ax0.legend(['Inertia', f'肘部 k={best_k_elbow}'], fontsize=9)

    # 2. Silhouette Score（轮廓系数）
    ax1 = axes[0, 1]
    ax1.plot(k_values, silhouette_scores, 'o-',
             color='blue', linewidth=2, markersize=6)
    best_k_sil = recommendations['silhouette']['k']
    best_score_sil = recommendations['silhouette']['score']
    ax1.axvline(best_k_sil, color='red', linestyle='--',
                linewidth=1.5, alpha=0.7)
    ax1.plot(best_k_sil, best_score_sil, 'ro', markersize=10)
    ax1.set_xlabel('k (聚类数量)', fontsize=11)
    ax1.set_ylabel('Silhouette Score', fontsize=11)
    ax1.set_title(
        f'轮廓系数 (Silhouette Score, 最佳 k={best_k_sil})', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(['Score', f'最佳 k={best_k_sil}'], fontsize=9)

    # 3. Davies-Bouldin Index
    ax2 = axes[0, 2]
    ax2.plot(k_values, dbi_scores, 'o-',
             color='green', linewidth=2, markersize=6)
    best_k_dbi = recommendations['dbi']['k']
    best_score_dbi = recommendations['dbi']['score']
    ax2.axvline(best_k_dbi, color='red', linestyle='--',
                linewidth=1.5, alpha=0.7)
    ax2.plot(best_k_dbi, best_score_dbi, 'ro', markersize=10)
    ax2.set_xlabel('k (聚类数量)', fontsize=11)
    ax2.set_ylabel('Davies-Bouldin Index', fontsize=11)
    ax2.set_title(
        f'Davies-Bouldin Index (最佳 k={best_k_dbi}, 越小越好)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(['Score', f'最佳 k={best_k_dbi}'], fontsize=9)

    # 4. Calinski-Harabasz Index
    ax3 = axes[1, 0]
    ax3.plot(k_values, ch_scores, 'o-',
             color='purple', linewidth=2, markersize=6)
    best_k_ch = recommendations['ch']['k']
    best_score_ch = recommendations['ch']['score']
    ax3.axvline(best_k_ch, color='red', linestyle='--',
                linewidth=1.5, alpha=0.7)
    ax3.plot(best_k_ch, best_score_ch, 'ro', markersize=10)
    ax3.set_xlabel('k (聚类数量)', fontsize=11)
    ax3.set_ylabel('Calinski-Harabasz Index', fontsize=11)
    ax3.set_title(
        f'Calinski-Harabasz Index (最佳 k={best_k_ch})', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(['Score', f'最佳 k={best_k_ch}'], fontsize=9)

    # 5. 肘部法则和轮廓系数对比
    ax4 = axes[1, 1]
    # 归一化惯性值（用于对比）
    inertia_norm = 1 - (np.array(inertia_scores) - min(inertia_scores)) / \
        (max(inertia_scores) - min(inertia_scores) + 1e-8)  # 惯性越小越好，所以取反并归一化
    sil_norm = (np.array(silhouette_scores) - min(silhouette_scores)) / \
        (max(silhouette_scores) - min(silhouette_scores) + 1e-8)

    ax4.plot(k_values, inertia_norm, 'o-', label='Elbow (归一化, 取反)',
             color='orange', linewidth=2, markersize=6)
    ax4.plot(k_values, sil_norm, 's-', label='Silhouette (归一化)',
             color='blue', linewidth=2, markersize=6)

    ax4.axvline(best_k_elbow, color='orange', linestyle='--',
                linewidth=1.5, alpha=0.7, label=f'肘部 k={best_k_elbow}')
    ax4.axvline(best_k_sil, color='blue', linestyle='--',
                linewidth=1.5, alpha=0.7, label=f'轮廓 k={best_k_sil}')

    ax4.set_xlabel('k (聚类数量)', fontsize=11)
    ax4.set_ylabel('归一化分数', fontsize=11)
    ax4.set_title('肘部法则 vs 轮廓系数', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9)

    # 6. 综合对比（所有指标归一化后）
    ax5 = axes[1, 2]
    # 归一化到[0,1]以便对比
    dbi_norm = 1 - (np.array(dbi_scores) - min(dbi_scores)) / \
        (max(dbi_scores) - min(dbi_scores) + 1e-8)  # DBI越小越好，所以取反
    ch_norm = (np.array(ch_scores) - min(ch_scores)) / \
        (max(ch_scores) - min(ch_scores) + 1e-8)

    ax5.plot(k_values, inertia_norm, 'o-', label='Elbow (归一化)',
             color='orange', linewidth=2, markersize=5)
    ax5.plot(k_values, sil_norm, 's-', label='Silhouette (归一化)',
             color='blue', linewidth=2, markersize=5)
    ax5.plot(k_values, dbi_norm, '^-', label='DBI (归一化, 取反)',
             color='green', linewidth=2, markersize=5)
    ax5.plot(k_values, ch_norm, 'd-', label='CH (归一化)',
             color='purple', linewidth=2, markersize=5)

    suggested_k = recommendations['suggested']['k']
    ax5.axvline(suggested_k, color='red',
                linestyle='--', linewidth=2, alpha=0.8, label=f'建议 k={suggested_k}')

    ax5.set_xlabel('k (聚类数量)', fontsize=11)
    ax5.set_ylabel('归一化分数', fontsize=11)
    ax5.set_title(f'综合对比 (建议 k={suggested_k})', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=8)

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'k_selection_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n可视化结果已保存至: {output_path}")


def save_results(
    k_values: list,
    inertia_scores: list,
    silhouette_scores: list,
    dbi_scores: list,
    ch_scores: list,
    recommendations: dict,
    output_dir: str
):
    """保存评估结果到CSV文件"""
    os.makedirs(output_dir, exist_ok=True)

    # 创建结果DataFrame
    results = {
        'k': k_values,
        'inertia': inertia_scores,
        'silhouette_score': silhouette_scores,
        'dbi_score': dbi_scores,
        'ch_score': ch_scores
    }

    import pandas as pd
    df = pd.DataFrame(results)

    csv_path = os.path.join(output_dir, 'k_evaluation_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"评估结果已保存至: {csv_path}")

    # 保存推荐结果
    summary_path = os.path.join(output_dir, 'k_recommendations.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("K-Means 最佳 k 值选择结果\n")
        f.write("=" * 60 + "\n\n")

        f.write("各指标推荐的最佳 k 值：\n")
        f.write(
            f"  - 肘部法则 (Elbow Method): k={recommendations['elbow']['k']} (惯性值={recommendations['elbow']['score']:.2f})\n")
        f.write(
            f"  - 轮廓系数 (Silhouette Score): k={recommendations['silhouette']['k']} (分数={recommendations['silhouette']['score']:.4f})\n")
        f.write(
            f"  - Davies-Bouldin Index: k={recommendations['dbi']['k']} (分数={recommendations['dbi']['score']:.4f}, 越小越好)\n")
        f.write(
            f"  - Calinski-Harabasz Index: k={recommendations['ch']['k']} (分数={recommendations['ch']['score']:.4f})\n")
        f.write("\n")

        f.write("综合建议：\n")
        f.write(f"  推荐 k = {recommendations['suggested']['k']}\n")
        f.write(f"  理由: {recommendations['suggested']['reason']}\n")
        f.write("\n")

        f.write("说明：\n")
        f.write("  - 肘部法则 (Elbow Method): 通过寻找惯性值（WCSS）下降速率突然变缓的点来确定最佳k值\n")
        f.write("  - 轮廓系数 (Silhouette Score): 越大越好，范围[-1, 1]，衡量样本与其所属簇的相似度\n")
        f.write("  - Davies-Bouldin Index: 越小越好，衡量簇内紧密度与簇间分离度\n")
        f.write("  - Calinski-Harabasz Index: 越大越好，基于簇间与簇内方差比\n")

    print(f"推荐结果已保存至: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="确定 K-Means 聚类的最佳 k 值")
    parser.add_argument(
        '--embeddings',
        type=str,
        default=DEFAULT_CONFIG['embeddings'],
        help=f'fused_embeddings.npz 文件路径（默认：{DEFAULT_CONFIG["embeddings"]}）'
    )
    parser.add_argument(
        '--embedding_key',
        type=str,
        default=DEFAULT_CONFIG['embedding_key'],
        help=f"npz 内用于聚类的嵌入矩阵 key（默认：{DEFAULT_CONFIG['embedding_key']}）。可直接填 h_spatial / h_od 等；也可填 'h' 或 'concat' 表示按 --concat_keys 拼接。若该 key 不存在，将自动兜底选择可用的二维嵌入矩阵（例如 embeddings）。"
    )
    parser.add_argument(
        '--concat_keys',
        type=str,
        nargs='+',
        default=DEFAULT_CONFIG['concat_keys'],
        help=f"当 --embedding_key 为 'h'/'concat' 时，用于拼接的 key 列表（默认：{' '.join(DEFAULT_CONFIG['concat_keys'])}）。"
    )
    parser.add_argument(
        '--k_min',
        type=int,
        default=DEFAULT_CONFIG['k_min'],
        help=f'k值的最小值（默认：{DEFAULT_CONFIG["k_min"]}）'
    )
    parser.add_argument(
        '--k_max',
        type=int,
        default=DEFAULT_CONFIG['k_max'],
        help=f'k值的最大值（默认：{DEFAULT_CONFIG["k_max"]}）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=DEFAULT_CONFIG['output_dir'],
        help=f'输出目录路径（默认：{DEFAULT_CONFIG["output_dir"]}）'
    )
    parser.add_argument(
        '--random_state',
        type=int,
        default=DEFAULT_CONFIG['random_state'],
        help=f'随机种子（默认：{DEFAULT_CONFIG["random_state"]}）'
    )
    parser.add_argument(
        '--standardize',
        action='store_true',
        help=f'是否标准化特征（默认：{DEFAULT_CONFIG["standardize"]}，因为嵌入通常已归一化）'
    )
    parser.add_argument(
        '--use_pca',
        action='store_true',
        help=f'是否使用PCA降维（默认：{DEFAULT_CONFIG["use_pca"]}）'
    )
    parser.add_argument(
        '--pca_components',
        type=int,
        default=DEFAULT_CONFIG['pca_components'],
        help=f'PCA降维后的维度（默认：{DEFAULT_CONFIG["pca_components"]}，仅在 --use_pca 时生效）'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 使用配置中的默认值（命令行参数会覆盖配置）
    # 注意：对于 --standardize，如果配置中为 True，默认启用；命令行 --standardize 可以覆盖
    use_standardize = DEFAULT_CONFIG['standardize'] if not args.standardize else args.standardize
    # 对于 --use_pca，如果配置中为 True，默认启用；命令行 --use_pca 可以覆盖
    use_pca = DEFAULT_CONFIG['use_pca'] if not args.use_pca else args.use_pca

    # 加载嵌入
    print("正在加载嵌入向量...")
    embeddings = load_embeddings(
        args.embeddings, args.embedding_key, args.concat_keys)
    print(f"嵌入向量形状: {embeddings.shape}")

    # 可选：标准化特征
    if use_standardize:
        print("正在标准化特征...")
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)

    # 可选：PCA降维
    if use_pca:
        original_dim = embeddings.shape[1]
        target_dim = args.pca_components

        if original_dim <= target_dim:
            print(f"[警告] 原始维度({original_dim})小于等于目标维度({target_dim})，跳过PCA降维")
        else:
            print(f"正在使用PCA降维: {original_dim} -> {target_dim}...")
            pca = PCA(n_components=target_dim, random_state=args.random_state)
            embeddings = pca.fit_transform(embeddings)
            explained_var = sum(pca.explained_variance_ratio_)
            print(
                f"PCA降维完成，保留方差比例: {explained_var:.4f} ({explained_var*100:.2f}%)")
            print(f"降维后嵌入向量形状: {embeddings.shape}")

    # 计算所有k值的指标
    k_values, inertia_scores, silhouette_scores, dbi_scores, ch_scores = find_optimal_k(
        embeddings,
        args.k_min,
        args.k_max,
        args.random_state
    )

    # 综合建议最佳k值
    print("\n" + "=" * 60)
    print("最佳 k 值推荐：")
    print("=" * 60)
    recommendations = suggest_optimal_k(
        k_values, inertia_scores, silhouette_scores, dbi_scores, ch_scores)

    print(f"\n各指标推荐：")
    print(
        f"  - 肘部法则 (Elbow Method): k={recommendations['elbow']['k']} (惯性值={recommendations['elbow']['score']:.2f})")
    print(
        f"  - 轮廓系数 (Silhouette Score): k={recommendations['silhouette']['k']} (分数={recommendations['silhouette']['score']:.4f})")
    print(
        f"  - Davies-Bouldin Index: k={recommendations['dbi']['k']} (分数={recommendations['dbi']['score']:.4f}, 越小越好)")
    print(
        f"  - Calinski-Harabasz Index: k={recommendations['ch']['k']} (分数={recommendations['ch']['score']:.4f})")
    print(f"\n综合建议: k={recommendations['suggested']['k']}")
    print(f"理由: {recommendations['suggested']['reason']}")

    # 可视化
    print("\n正在生成可视化图表...")
    visualize_k_selection(k_values, inertia_scores, silhouette_scores,
                          dbi_scores, ch_scores, recommendations, args.output_dir)

    # 保存结果
    save_results(k_values, inertia_scores, silhouette_scores, dbi_scores,
                 ch_scores, recommendations, args.output_dir)

    print(f"\n所有结果已保存至: {args.output_dir}")


if __name__ == '__main__':
    main()
