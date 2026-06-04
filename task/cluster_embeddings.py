#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cluster fused node embeddings with K-Means (or hierarchical clustering) and
visualize with PCA / t-SNE / UMAP. Outputs a CSV mapping node_id to cluster_id.

Example:
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
except ImportError:  # pragma: no cover - only when dependency is missing
    umap = None

# ========== Configuration: edit defaults here ==========
# If you prefer not to use the CLI, change the values below directly.
DEFAULT_CONFIG = {
    'embeddings': 'checkpoints/train_20251222_220544/best_embeddings.npz',  # Path to embeddings file
    # Embedding key: 'h_spatial', 'h_od', 'z', or 'h'/'concat' (concatenation mode)
    'embedding_key': 'z',
    'concat_keys': ['h_spatial', 'h_od'],  # Keys to concatenate when embedding_key='h'
    'clustering_method': 'hierarchical',  # 'kmeans' or 'hierarchical'
    'num_clusters': 3,  # Number of clusters
    'output_dir': 'checkpoints/train_20251222_220544/cluster_results',  # Output directory
    'viz_methods': ['pca', 'tsne', 'umap'],  # Visualization methods
    'random_state': 42,  # Random seed
    'tsne_perplexity': 30.0,  # t-SNE perplexity
    'umap_neighbors': 15,  # UMAP n_neighbors
    'umap_min_dist': 0.1,  # UMAP min_dist
    # Hierarchical clustering parameters
    'linkage': 'ward',  # Linkage: 'ward', 'complete', 'average', 'single'
    'metric': 'euclidean',  # Distance metric (when linkage is not 'ward')
    'plot_dendrogram': False,  # Plot dendrogram (can be slow on large data)
}
# ====================================================


def _pick_embedding_key(data: "np.lib.npyio.NpzFile") -> str:
    """Pick the key in an npz file that most likely holds the embedding matrix.

    Selection order (highest priority first):
    - Use 'embeddings' if present
    - Else use 'z' (many models use z for fused representations)
    - Else pick the 2D array with the largest embedding dimension (shape[1])
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
            f"Could not find a usable 2D embedding matrix in {keys}. "
            f"Ensure the npz contains an (N, D) array, or pass --embedding_key explicitly."
        )

    candidates.sort(reverse=True)
    return candidates[0][1]


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

    requested = (embedding_key or "embeddings").strip()

    # Special mode: concatenate to form h (default: h_spatial + h_od)
    if requested.lower() in {"h", "concat"}:
        concat_keys = concat_keys or ["h_spatial", "h_od"]
        embeddings = _concat_embeddings(data, concat_keys)
        key_to_use = f"concat({'+'.join(concat_keys)})"
        print(f"[Info] Using concatenated embeddings: {key_to_use}, shape={embeddings.shape}")
    else:
        if requested in keys:
            key_to_use = requested
        else:
            key_to_use = _pick_embedding_key(data)
            print(
                f"[Info] key='{requested}' not found in npz; using key='{key_to_use}'. "
                f"Available keys: {keys}"
            )

        embeddings = data[key_to_use]
        if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
            raise ValueError(
                f"Data for key='{key_to_use}' is not a 2D ndarray. "
                f"type={type(embeddings)}, ndim={getattr(embeddings, 'ndim', None)}, shape={getattr(embeddings, 'shape', None)}"
            )

    node_ids = data.get('node_ids', np.arange(len(embeddings)))

    # Convert byte strings to str if needed
    if node_ids.dtype.kind in {'S', 'O'}:
        node_ids = np.array([str(x) for x in node_ids])

    # Align node_ids with embeddings when possible
    if len(node_ids) != len(embeddings):
        print(
            f"[Warning] node_ids length ({len(node_ids)}) != embeddings rows ({len(embeddings)}); "
            f"using 0..N-1 as node_ids."
        )
        node_ids = np.arange(len(embeddings))

    return node_ids, embeddings


def run_kmeans(embeddings: np.ndarray, num_clusters: int, random_state: int) -> np.ndarray:
    """Run KMeans and return cluster labels."""
    print(f"Running K-Means clustering, k={num_clusters}...")
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
    """Run hierarchical clustering and return cluster labels.

    Args:
        embeddings: Embedding matrix
        num_clusters: Number of clusters
        linkage_method: Linkage method ('ward', 'complete', 'average', 'single')
        metric: Distance metric (when linkage is not 'ward')
        plot_dendrogram: Whether to plot a dendrogram
        output_dir: Output directory (for saving the dendrogram)
    """
    print(f"Running hierarchical clustering, k={num_clusters}, linkage={linkage_method}...")

    # For large data, use AgglomerativeClustering (faster)
    # For small data with dendrogram, use linkage + fcluster

    n_samples = embeddings.shape[0]

    if plot_dendrogram and n_samples <= 1000:
        # Small dataset: full linkage matrix and dendrogram
        print("Computing linkage matrix for dendrogram...")
        if linkage_method == 'ward':
            Z = linkage(embeddings, method=linkage_method, metric='euclidean')
        else:
            Z = linkage(embeddings, method=linkage_method, metric=metric)

        # Plot dendrogram
        if output_dir:
            plt.figure(figsize=(15, 8))
            dendrogram(Z, truncate_mode='level', p=min(10, num_clusters * 2))
            plt.title(f'Hierarchical clustering dendrogram (linkage={linkage_method})')
            plt.xlabel('Sample index')
            plt.ylabel('Distance')
            plt.tight_layout()
            dendrogram_path = os.path.join(output_dir, 'dendrogram.png')
            plt.savefig(dendrogram_path, dpi=300)
            plt.close()
            print(f"Dendrogram saved to: {dendrogram_path}")

        # Cluster with AgglomerativeClustering
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
        # Large dataset: AgglomerativeClustering without full linkage matrix
        if plot_dendrogram and n_samples > 1000:
            print(f"[Warning] Too many samples ({n_samples}); skipping dendrogram (expensive).")

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
    """Run clustering with the specified method.

    Returns:
        labels: Cluster labels
        method_name: Method name for display
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
        method_name = f'Hierarchical clustering (linkage={linkage_method})'
    else:
        raise ValueError(f"Unknown clustering method: {method}. Supported: 'kmeans', 'hierarchical'")

    return labels, method_name


def project_embeddings(
    embeddings: np.ndarray,
    method: str,
    random_state: int,
    tsne_perplexity: float,
    umap_neighbors: int,
    umap_min_dist: float
) -> Tuple[np.ndarray, str]:
    """Project embeddings to 2D with the given dimensionality reduction method."""
    method = method.lower()
    n_samples = embeddings.shape[0]

    if method == 'pca':
        reducer = PCA(n_components=2, random_state=random_state)
        proj = reducer.fit_transform(embeddings)
        title = 'PCA'
    elif method == 'tsne':
        effective_perplexity = min(tsne_perplexity, max(5, n_samples - 1))
        # Newer scikit-learn uses max_iter instead of n_iter
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
                "umap-learn is not installed. Run `pip install umap-learn` and retry."
            )
        # Suppress UMAP warnings about random_state and n_jobs
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', message='.*n_jobs.*random_state.*')
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=umap_neighbors,
                min_dist=umap_min_dist,
                random_state=random_state,
                n_jobs=1  # single-threaded when random_state is set
            )
            proj = reducer.fit_transform(embeddings)
        title = f"UMAP (n_neighbors={umap_neighbors}, min_dist={umap_min_dist})"
    else:
        raise ValueError(f"Unknown visualization method: {method}")

    return proj, title


def visualize_clusters(
    proj: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_path: str,
    clustering_method_name: str = 'Clustering'
):
    """Plot cluster scatter from a 2D projection."""
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
    print(f"Cluster visualization saved to: {output_path}")


def summarize_clusters(node_ids: np.ndarray, labels: np.ndarray):
    """Print node count per cluster."""
    unique, counts = np.unique(labels, return_counts=True)
    print("Cluster summary:")
    for cid, count in zip(unique, counts):
        print(f"  - Cluster {cid}: {count} nodes")


def save_cluster_csv(
    node_ids: np.ndarray,
    labels: np.ndarray,
    output_dir: str,
    filename: str = 'cluster_assignments.csv'
):
    """Save node_id to cluster_id mapping."""
    df = pd.DataFrame({
        'node_id': node_ids,
        'cluster_id': labels
    })
    csv_path = os.path.join(output_dir, filename)
    df.to_csv(csv_path, index=False)
    print(f"Cluster assignments CSV saved to: {csv_path}")
    return csv_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="KMeans (or hierarchical) clustering on fused embeddings with visualizations")
    parser.add_argument(
        '--embeddings',
        type=str,
        default=DEFAULT_CONFIG['embeddings'],
        help=f'Path to fused_embeddings.npz (default: {DEFAULT_CONFIG["embeddings"]})'
    )
    parser.add_argument(
        '--embedding_key',
        type=str,
        default=DEFAULT_CONFIG['embedding_key'],
        help=(
            f"Embedding key in npz for clustering (default: {DEFAULT_CONFIG['embedding_key']}). "
            f"Use h_spatial / h_od / z, or 'h'/'concat' to concatenate per --concat_keys. "
            f"If missing, auto-selects a 2D embedding (e.g. embeddings)."
        )
    )
    parser.add_argument(
        '--concat_keys',
        type=str,
        nargs='+',
        default=DEFAULT_CONFIG['concat_keys'],
        help=(
            f"Keys to concatenate when --embedding_key is 'h'/'concat' "
            f"(default: {' '.join(DEFAULT_CONFIG['concat_keys'])})."
        )
    )
    parser.add_argument(
        '--clustering_method',
        type=str,
        default=DEFAULT_CONFIG['clustering_method'],
        choices=['kmeans', 'hierarchical', 'hier', 'agg'],
        help=f"Clustering method (default: {DEFAULT_CONFIG['clustering_method']}). Options: kmeans, hierarchical"
    )
    parser.add_argument(
        '--num_clusters',
        type=int,
        default=DEFAULT_CONFIG['num_clusters'],
        help=f'Number of clusters (default: {DEFAULT_CONFIG["num_clusters"]})'
    )
    parser.add_argument(
        '--linkage',
        type=str,
        default=DEFAULT_CONFIG['linkage'],
        choices=['ward', 'complete', 'average', 'single'],
        help=f"Hierarchical linkage (default: {DEFAULT_CONFIG['linkage']}). Options: ward, complete, average, single"
    )
    parser.add_argument(
        '--metric',
        type=str,
        default=DEFAULT_CONFIG['metric'],
        help=f"Distance metric for hierarchical clustering (default: {DEFAULT_CONFIG['metric']}); used when linkage != ward"
    )
    parser.add_argument(
        '--plot_dendrogram',
        action='store_true',
        default=DEFAULT_CONFIG['plot_dendrogram'],
        help=f'Plot hierarchical dendrogram (default: {DEFAULT_CONFIG["plot_dendrogram"]}; slow on large data)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=DEFAULT_CONFIG['output_dir'],
        help=f'Output directory for CSV and plots (default: {DEFAULT_CONFIG["output_dir"]})'
    )
    parser.add_argument(
        '--viz_methods',
        type=str,
        nargs='+',
        default=DEFAULT_CONFIG['viz_methods'],
        help=f'Visualization methods: pca, tsne, umap (default: {" ".join(DEFAULT_CONFIG["viz_methods"])})'
    )
    parser.add_argument(
        '--random_state',
        type=int,
        default=DEFAULT_CONFIG['random_state'],
        help=f'Random seed (default: {DEFAULT_CONFIG["random_state"]})'
    )
    parser.add_argument(
        '--tsne_perplexity',
        type=float,
        default=DEFAULT_CONFIG['tsne_perplexity'],
        help=f't-SNE perplexity (clamped to sample size) (default: {DEFAULT_CONFIG["tsne_perplexity"]})'
    )
    parser.add_argument(
        '--umap_neighbors',
        type=int,
        default=DEFAULT_CONFIG['umap_neighbors'],
        help=f'UMAP n_neighbors (default: {DEFAULT_CONFIG["umap_neighbors"]})'
    )
    parser.add_argument(
        '--umap_min_dist',
        type=float,
        default=DEFAULT_CONFIG['umap_min_dist'],
        help=f'UMAP min_dist (default: {DEFAULT_CONFIG["umap_min_dist"]})'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    node_ids, embeddings = load_embeddings(
        args.embeddings, args.embedding_key, args.concat_keys)

    # Run clustering
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
            print(f"[Warning] {method} visualization failed: {exc}")
            continue

        filename = f'clusters_{method}.png'
        output_path = os.path.join(args.output_dir, filename)
        visualize_clusters(proj, labels, title, output_path,
                           clustering_method_name)


if __name__ == '__main__':
    main()
