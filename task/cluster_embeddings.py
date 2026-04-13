#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 K-Means 对融合后的节点嵌入进行聚类，并提供多种 (PCA / t-SNE / UMAP)
可视化方式，同时输出 node_id 对 cluster_id 的 CSV 映射。

示例：
    python cluster_embeddings.py \
        --embeddings checkpoints/train_20251202_200939/fused_embeddings.npz \
        --num_clusters 8 \
        --output_dir checkpoints/train_20251202_200939/cluster_results \
        --viz_methods pca tsne umap
"""

import argparse
import os
import warnings
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

try:
    import umap
except ImportError:  # pragma: no cover - 仅在缺少依赖时触发
    umap = None

# ========== 配置区域：可直接在此修改默认值 ==========
# 如果不想通过命令行传参，可以直接修改下面的配置
DEFAULT_CONFIG = {
    'embeddings': 'checkpoints/train_20251222_220544/best_embeddings.npz',  # 嵌入文件路径
    # 使用的嵌入key: 'h_spatial', 'h_od', 'z', 或 'h'/'concat'（拼接模式）
    'embedding_key': 'z',
    'concat_keys': ['h_spatial', 'h_od'],  # 当 embedding_key='h' 时，用于拼接的key列表
    'clustering_method': 'hierarchical',  # 聚类方法: 'kmeans' 或 'hierarchical'
    'num_clusters': 3,  # 聚类数量z
    'output_dir': 'checkpoints/train_20251222_220544/cluster_results',  # 输出目录
    'viz_methods': ['pca', 'tsne', 'umap'],  # 可视化方法列表
    'random_state': 42,  # 随机种子
    'tsne_perplexity': 30.0,  # t-SNE 的 perplexity
    'umap_neighbors': 15,  # UMAP 的 n_neighbors
    'umap_min_dist': 0.1,  # UMAP 的 min_dist
    # 层次聚类参数
    'linkage': 'ward',  # 层次聚类的链接方法: 'ward', 'complete', 'average', 'single'
    'metric': 'euclidean',  # 距离度量（当 linkage 不是 'ward' 时使用）
    'plot_dendrogram': False,  # 是否绘制树状图（大数据时可能很慢）
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

    requested = (embedding_key or "embeddings").strip()

    # 特殊模式：拼接得到 h（默认拼接 h_spatial + h_od）
    if requested.lower() in {"h", "concat"}:
        concat_keys = concat_keys or ["h_spatial", "h_od"]
        embeddings = _concat_embeddings(data, concat_keys)
        key_to_use = f"concat({'+'.join(concat_keys)})"
        print(f"[提示] 使用拼接嵌入作为输入：{key_to_use}，shape={embeddings.shape}")
    else:
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

    node_ids = data.get('node_ids', np.arange(len(embeddings)))

    # 如果 node_ids 是字节串，转换为字符串
    if node_ids.dtype.kind in {'S', 'O'}:
        node_ids = np.array([str(x) for x in node_ids])

    # 尽量保证 node_ids 与 embeddings 对齐
    if len(node_ids) != len(embeddings):
        print(
            f"[警告] node_ids 长度({len(node_ids)})与 embeddings 行数({len(embeddings)})不一致，将使用 0..N-1 作为 node_ids。"
        )
        node_ids = np.arange(len(embeddings))

    return node_ids, embeddings


def run_kmeans(embeddings: np.ndarray, num_clusters: int, random_state: int) -> np.ndarray:
    """运行 KMeans 并返回聚类标签"""
    print(f"使用 K-Means 聚类，k={num_clusters}...")
    kmeans = KMeans(
        n_clusters=num_clusters,
        n_init=10,
        max_iter=300,
        random_state=random_state
    )
    labels = kmeans.fit_predict(embeddings)
    return labels


def run_hierarchical(
    embeddings: np.ndarray,
    num_clusters: int,
    linkage_method: str = 'ward',
    metric: str = 'euclidean',
    plot_dendrogram: bool = False,
    output_dir: Optional[str] = None
) -> np.ndarray:
    """运行层次聚类并返回聚类标签

    Args:
        embeddings: 嵌入矩阵
        num_clusters: 聚类数量
        linkage_method: 链接方法 ('ward', 'complete', 'average', 'single')
        metric: 距离度量（当 linkage 不是 'ward' 时使用）
        plot_dendrogram: 是否绘制树状图
        output_dir: 输出目录（用于保存树状图）
    """
    print(f"使用层次聚类，k={num_clusters}，linkage={linkage_method}...")

    # 对于大数据，使用 AgglomerativeClustering（更快）
    # 对于小数据且需要树状图，使用 linkage + fcluster

    n_samples = embeddings.shape[0]

    if plot_dendrogram and n_samples <= 1000:
        # 小数据集：计算完整链接矩阵并绘制树状图
        print("计算链接矩阵以绘制树状图...")
        if linkage_method == 'ward':
            Z = linkage(embeddings, method=linkage_method, metric='euclidean')
        else:
            Z = linkage(embeddings, method=linkage_method, metric=metric)

        # 绘制树状图
        if output_dir:
            plt.figure(figsize=(15, 8))
            dendrogram(Z, truncate_mode='level', p=min(10, num_clusters * 2))
            plt.title(f'层次聚类树状图 (linkage={linkage_method})')
            plt.xlabel('样本索引')
            plt.ylabel('距离')
            plt.tight_layout()
            dendrogram_path = os.path.join(output_dir, 'dendrogram.png')
            plt.savefig(dendrogram_path, dpi=300)
            plt.close()
            print(f"树状图已保存至: {dendrogram_path}")

        # 使用 AgglomerativeClustering 进行聚类
        if linkage_method == 'ward':
            clustering = AgglomerativeClustering(
                n_clusters=num_clusters,
                linkage=linkage_method
            )
        else:
            clustering = AgglomerativeClustering(
                n_clusters=num_clusters,
                linkage=linkage_method,
                metric=metric
            )
        labels = clustering.fit_predict(embeddings)
    else:
        # 大数据集：直接使用 AgglomerativeClustering（不计算完整链接矩阵）
        if plot_dendrogram and n_samples > 1000:
            print(f"[警告] 样本数({n_samples})过多，跳过树状图绘制（计算量过大）")

        if linkage_method == 'ward':
            clustering = AgglomerativeClustering(
                n_clusters=num_clusters,
                linkage=linkage_method
            )
        else:
            clustering = AgglomerativeClustering(
                n_clusters=num_clusters,
                linkage=linkage_method,
                metric=metric
            )
        labels = clustering.fit_predict(embeddings)

    return labels


def run_clustering(
    embeddings: np.ndarray,
    method: str,
    num_clusters: int,
    random_state: int = 42,
    linkage_method: str = 'ward',
    metric: str = 'euclidean',
    plot_dendrogram: bool = False,
    output_dir: Optional[str] = None
) -> Tuple[np.ndarray, str]:
    """根据指定方法运行聚类

    Returns:
        labels: 聚类标签
        method_name: 聚类方法名称（用于显示）
    """
    method = method.lower()

    if method == 'kmeans':
        labels = run_kmeans(embeddings, num_clusters, random_state)
        method_name = 'K-Means'
    elif method in ['hierarchical', 'hier', 'agg']:
        labels = run_hierarchical(
            embeddings, num_clusters, linkage_method, metric,
            plot_dendrogram, output_dir
        )
        method_name = f'层次聚类 (linkage={linkage_method})'
    else:
        raise ValueError(f"未知的聚类方法: {method}。支持的方法: 'kmeans', 'hierarchical'")

    return labels, method_name


def project_embeddings(
    embeddings: np.ndarray,
    method: str,
    random_state: int,
    tsne_perplexity: float,
    umap_neighbors: int,
    umap_min_dist: float
) -> Tuple[np.ndarray, str]:
    """根据指定降维方法将嵌入映射到二维"""
    method = method.lower()
    n_samples = embeddings.shape[0]

    if method == 'pca':
        reducer = PCA(n_components=2, random_state=random_state)
        proj = reducer.fit_transform(embeddings)
        title = 'PCA'
    elif method == 'tsne':
        effective_perplexity = min(tsne_perplexity, max(5, n_samples - 1))
        # 新版本 scikit-learn 使用 max_iter 而不是 n_iter
        reducer = TSNE(
            n_components=2,
            perplexity=effective_perplexity,
            init='pca',
            learning_rate='auto',
            max_iter=1000,
            random_state=random_state
        )
        proj = reducer.fit_transform(embeddings)
        title = f"t-SNE (perplexity={effective_perplexity:.1f})"
    elif method == 'umap':
        if umap is None:
            raise ImportError(
                "未安装 umap-learn，请运行 `pip install umap-learn` 后重试。"
            )
        # 抑制 UMAP 关于 random_state 和 n_jobs 的警告
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', message='.*n_jobs.*random_state.*')
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=umap_neighbors,
                min_dist=umap_min_dist,
                random_state=random_state,
                n_jobs=1  # 设置 random_state 时会强制单线程
            )
            proj = reducer.fit_transform(embeddings)
        title = f"UMAP (n_neighbors={umap_neighbors}, min_dist={umap_min_dist})"
    else:
        raise ValueError(f"未知的可视化方法: {method}")

    return proj, title


def visualize_clusters(
    proj: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_path: str,
    clustering_method_name: str = 'Clustering'
):
    """根据二维投影绘制聚类散点图"""
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        proj[:, 0],
        proj[:, 1],
        c=labels,
        cmap='tab20',
        s=20,
        alpha=0.85,
        linewidths=0
    )
    plt.colorbar(scatter, label='Cluster ID')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title(f'{clustering_method_name} Clusters ({title})')
    plt.grid(alpha=0.2, linestyle='--')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"聚类可视化已保存至: {output_path}")


def summarize_clusters(node_ids: np.ndarray, labels: np.ndarray):
    """打印每个聚类的节点数量"""
    unique, counts = np.unique(labels, return_counts=True)
    print("聚类结果统计：")
    for cid, count in zip(unique, counts):
        print(f"  - Cluster {cid}: {count} nodes")


def save_cluster_csv(
    node_ids: np.ndarray,
    labels: np.ndarray,
    output_dir: str,
    filename: str = 'cluster_assignments.csv'
):
    """保存 node_id 与 cluster_id 的对应关系"""
    df = pd.DataFrame({
        'node_id': node_ids,
        'cluster_id': labels
    })
    csv_path = os.path.join(output_dir, filename)
    df.to_csv(csv_path, index=False)
    print(f"聚类结果 CSV 已保存至: {csv_path}")
    return csv_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="对融合嵌入执行 KMeans 聚类并生成多种可视化")
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
        '--clustering_method',
        type=str,
        default=DEFAULT_CONFIG['clustering_method'],
        choices=['kmeans', 'hierarchical', 'hier', 'agg'],
        help=f"聚类方法（默认：{DEFAULT_CONFIG['clustering_method']}）。可选: 'kmeans', 'hierarchical'"
    )
    parser.add_argument(
        '--num_clusters',
        type=int,
        default=DEFAULT_CONFIG['num_clusters'],
        help=f'聚类数量（默认：{DEFAULT_CONFIG["num_clusters"]}）'
    )
    parser.add_argument(
        '--linkage',
        type=str,
        default=DEFAULT_CONFIG['linkage'],
        choices=['ward', 'complete', 'average', 'single'],
        help=f"层次聚类的链接方法（默认：{DEFAULT_CONFIG['linkage']}）。可选: 'ward', 'complete', 'average', 'single'"
    )
    parser.add_argument(
        '--metric',
        type=str,
        default=DEFAULT_CONFIG['metric'],
        help=f"层次聚类的距离度量（默认：{DEFAULT_CONFIG['metric']}）。当 linkage 不是 'ward' 时使用"
    )
    parser.add_argument(
        '--plot_dendrogram',
        action='store_true',
        default=DEFAULT_CONFIG['plot_dendrogram'],
        help=f'是否绘制层次聚类的树状图（默认：{DEFAULT_CONFIG["plot_dendrogram"]}，大数据时可能很慢）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=DEFAULT_CONFIG['output_dir'],
        help=f'输出目录（保存 CSV 及所有可视化图片）（默认：{DEFAULT_CONFIG["output_dir"]}）'
    )
    parser.add_argument(
        '--viz_methods',
        type=str,
        nargs='+',
        default=DEFAULT_CONFIG['viz_methods'],
        help=f'可视化方法列表，可选: pca, tsne, umap（默认：{" ".join(DEFAULT_CONFIG["viz_methods"])}）'
    )
    parser.add_argument(
        '--random_state',
        type=int,
        default=DEFAULT_CONFIG['random_state'],
        help=f'随机种子（默认：{DEFAULT_CONFIG["random_state"]}）'
    )
    parser.add_argument(
        '--tsne_perplexity',
        type=float,
        default=DEFAULT_CONFIG['tsne_perplexity'],
        help=f't-SNE 的 perplexity（将自动限制在样本数范围内）（默认：{DEFAULT_CONFIG["tsne_perplexity"]}）'
    )
    parser.add_argument(
        '--umap_neighbors',
        type=int,
        default=DEFAULT_CONFIG['umap_neighbors'],
        help=f'UMAP 的 n_neighbors（默认：{DEFAULT_CONFIG["umap_neighbors"]}）'
    )
    parser.add_argument(
        '--umap_min_dist',
        type=float,
        default=DEFAULT_CONFIG['umap_min_dist'],
        help=f'UMAP 的 min_dist（默认：{DEFAULT_CONFIG["umap_min_dist"]}）'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    node_ids, embeddings = load_embeddings(
        args.embeddings, args.embedding_key, args.concat_keys)

    # 运行聚类
    labels, clustering_method_name = run_clustering(
        embeddings=embeddings,
        method=args.clustering_method,
        num_clusters=args.num_clusters,
        random_state=args.random_state,
        linkage_method=args.linkage,
        metric=args.metric,
        plot_dendrogram=args.plot_dendrogram,
        output_dir=args.output_dir
    )

    summarize_clusters(node_ids, labels)

    os.makedirs(args.output_dir, exist_ok=True)
    save_cluster_csv(node_ids, labels, args.output_dir)

    methods: List[str] = args.viz_methods
    for method in methods:
        method = method.lower()
        try:
            proj, title = project_embeddings(
                embeddings=embeddings,
                method=method,
                random_state=args.random_state,
                tsne_perplexity=args.tsne_perplexity,
                umap_neighbors=args.umap_neighbors,
                umap_min_dist=args.umap_min_dist
            )
        except Exception as exc:
            print(f"[警告] {method} 可视化失败: {exc}")
            continue

        filename = f'clusters_{method}.png'
        output_path = os.path.join(args.output_dir, filename)
        visualize_clusters(proj, labels, title, output_path,
                           clustering_method_name)


if __name__ == '__main__':
    main()
