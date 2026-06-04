#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find the best k for K-Means using elbow method, silhouette score,
Davies-Bouldin Index, and Calinski-Harabasz Index.

Example:
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

# ========== Configuration: edit defaults here ==========
# If you prefer not to use the CLI, change the values below directly.
DEFAULT_CONFIG = {
    'embeddings': 'checkpoints/train_20251222_220544/best_embeddings.npz',  # Path to embeddings file
    'embedding_key': 'h',  # Key: h_spatial, h_od, z, or h/concat (concatenation mode)
    'concat_keys': ['h_spatial', 'h_od'],  # Keys to concatenate when embedding_key='h'
    'k_min': 2,  # Minimum k
    'k_max': 8,  # Maximum k
    'output_dir': 'checkpoints/train_20251222_220544/k_selection',  # Output directory
    'random_state': 42,  # Random seed
    'standardize': False,  # Standardize features (default False; embeddings often normalized)
    'use_pca': True,  # Apply PCA dimensionality reduction
    'pca_components': 128,  # PCA target dimension
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


def _concat_embeddings(data: "np.lib.npyio.NpzFile", keys: list) -> np.ndarray:
    """Concatenate multiple 2D embedding matrices along the feature dimension."""
    arrays = []
    n_rows = None
    available = list(getattr(data, "files", []))
    for k in keys:
        if k not in available:
            raise KeyError(f"key='{k}' not found in npz; available keys: {available}")
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
    concat_keys: Optional[list] = None
) -> np.ndarray:
    """Load embedding matrix (2D ndarray) from an npz file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embeddings file not found: {path}")

    data = np.load(path, allow_pickle=True)
    keys = list(getattr(data, "files", []))

    requested = (embedding_key or "embeddings").strip()

    # Special mode: concatenate to form h (default: h_spatial + h_od)
    if requested.lower() in {"h", "concat"}:
        concat_keys = concat_keys or ["h_spatial", "h_od"]
        embeddings = _concat_embeddings(data, concat_keys)
        print(
            f"[Info] Using concatenated embeddings: concat({'+'.join(concat_keys)}), shape={embeddings.shape}")
        return embeddings

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
    return embeddings


def compute_metrics(embeddings: np.ndarray, k: int, random_state: int) -> Tuple[float, float, float, float]:
    """Compute four evaluation metrics for a given k.

    Returns:
        inertia: Within-cluster sum of squares (WCSS) for elbow method; lower is better
        silhouette: Silhouette score (higher is better, range [-1, 1])
        dbi: Davies-Bouldin Index (lower is better, >= 0)
        ch: Calinski-Harabasz Index (higher is better, >= 0)
    """
    kmeans = KMeans(
        n_clusters=k,
        n_init=10,
        max_iter=300,
        random_state=random_state
    )
    labels = kmeans.fit_predict(embeddings)
    inertia = kmeans.inertia_  # WCSS

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
    """Sweep k range and compute all metrics.

    Returns:
        k_values: List of k values
        inertia_scores: Inertia values (elbow method)
        silhouette_scores: Silhouette scores
        dbi_scores: DBI scores
        ch_scores: CH scores
    """
    k_values = list(range(k_min, k_max + 1))
    inertia_scores = []
    silhouette_scores = []
    dbi_scores = []
    ch_scores = []

    print(f"Evaluating k from {k_min} to {k_max}...")
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
    """Find best k via elbow method.

    Elbow: point where inertia decrease rate slows sharply.
    Uses percent change in inertia between adjacent k values.

    Returns:
        Recommended k (elbow point)
    """
    if len(inertia_scores) < 3:
        return k_values[len(k_values) // 2]

    decreases = []
    for i in range(1, len(inertia_scores)):
        if inertia_scores[i-1] > 0:
            decrease_rate = (
                inertia_scores[i-1] - inertia_scores[i]) / inertia_scores[i-1]
            decreases.append(decrease_rate)
        else:
            decreases.append(0)

    if len(decreases) < 2:
        return k_values[1] if len(k_values) > 1 else k_values[0]

    accelerations = []
    for i in range(1, len(decreases)):
        acceleration = decreases[i-1] - decreases[i]
        accelerations.append(acceleration)

    if accelerations:
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
    """Suggest best k from all four metrics.

    Returns:
        Dict with per-metric recommendations and combined suggestion
    """
    best_k_elbow = find_elbow_point(inertia_scores, k_values)
    best_inertia_at_elbow = inertia_scores[k_values.index(best_k_elbow)]

    best_k_silhouette = k_values[np.argmax(silhouette_scores)]
    best_silhouette_score = max(silhouette_scores)

    best_k_dbi = k_values[np.argmin(dbi_scores)]
    best_dbi_score = min(dbi_scores)

    best_k_ch = k_values[np.argmax(ch_scores)]
    best_ch_score = max(ch_scores)

    recommendations = [best_k_elbow, best_k_silhouette, best_k_dbi, best_k_ch]
    k_range = max(recommendations) - min(recommendations)

    if k_range <= 3:
        suggested_k = min(recommendations)
        reason = "Multiple metrics agree on similar k"
    else:
        if abs(best_k_elbow - best_k_silhouette) <= 2:
            suggested_k = int(np.mean([best_k_elbow, best_k_silhouette]))
            reason = "Elbow method and silhouette score suggest similar k"
        else:
            suggested_k = best_k_silhouette
            reason = "Metrics disagree; prefer silhouette score (more stable)"

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
    """Plot curves for all four metrics."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('K-Means optimal k: elbow + silhouette + other metrics',
                 fontsize=16, fontweight='bold')

    # 1. Elbow method
    ax0 = axes[0, 0]
    ax0.plot(k_values, inertia_scores, 'o-',
             color='orange', linewidth=2, markersize=6)
    best_k_elbow = recommendations['elbow']['k']
    best_inertia_elbow = recommendations['elbow']['score']
    ax0.axvline(best_k_elbow, color='red', linestyle='--',
                linewidth=1.5, alpha=0.7)
    ax0.plot(best_k_elbow, best_inertia_elbow, 'ro', markersize=10)
    ax0.set_xlabel('k (number of clusters)', fontsize=11)
    ax0.set_ylabel('Inertia (WCSS)', fontsize=11)
    ax0.set_title(
        f'Elbow method (best k={best_k_elbow})', fontsize=12, fontweight='bold')
    ax0.grid(True, alpha=0.3)
    ax0.legend(['Inertia', f'Elbow k={best_k_elbow}'], fontsize=9)

    # 2. Silhouette score
    ax1 = axes[0, 1]
    ax1.plot(k_values, silhouette_scores, 'o-',
             color='blue', linewidth=2, markersize=6)
    best_k_sil = recommendations['silhouette']['k']
    best_score_sil = recommendations['silhouette']['score']
    ax1.axvline(best_k_sil, color='red', linestyle='--',
                linewidth=1.5, alpha=0.7)
    ax1.plot(best_k_sil, best_score_sil, 'ro', markersize=10)
    ax1.set_xlabel('k (number of clusters)', fontsize=11)
    ax1.set_ylabel('Silhouette Score', fontsize=11)
    ax1.set_title(
        f'Silhouette score (best k={best_k_sil})', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(['Score', f'Best k={best_k_sil}'], fontsize=9)

    # 3. Davies-Bouldin Index
    ax2 = axes[0, 2]
    ax2.plot(k_values, dbi_scores, 'o-',
             color='green', linewidth=2, markersize=6)
    best_k_dbi = recommendations['dbi']['k']
    best_score_dbi = recommendations['dbi']['score']
    ax2.axvline(best_k_dbi, color='red', linestyle='--',
                linewidth=1.5, alpha=0.7)
    ax2.plot(best_k_dbi, best_score_dbi, 'ro', markersize=10)
    ax2.set_xlabel('k (number of clusters)', fontsize=11)
    ax2.set_ylabel('Davies-Bouldin Index', fontsize=11)
    ax2.set_title(
        f'Davies-Bouldin Index (best k={best_k_dbi}, lower is better)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(['Score', f'Best k={best_k_dbi}'], fontsize=9)

    # 4. Calinski-Harabasz Index
    ax3 = axes[1, 0]
    ax3.plot(k_values, ch_scores, 'o-',
             color='purple', linewidth=2, markersize=6)
    best_k_ch = recommendations['ch']['k']
    best_score_ch = recommendations['ch']['score']
    ax3.axvline(best_k_ch, color='red', linestyle='--',
                linewidth=1.5, alpha=0.7)
    ax3.plot(best_k_ch, best_score_ch, 'ro', markersize=10)
    ax3.set_xlabel('k (number of clusters)', fontsize=11)
    ax3.set_ylabel('Calinski-Harabasz Index', fontsize=11)
    ax3.set_title(
        f'Calinski-Harabasz Index (best k={best_k_ch})', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(['Score', f'Best k={best_k_ch}'], fontsize=9)

    # 5. Elbow vs silhouette
    ax4 = axes[1, 1]
    inertia_norm = 1 - (np.array(inertia_scores) - min(inertia_scores)) / \
        (max(inertia_scores) - min(inertia_scores) + 1e-8)
    sil_norm = (np.array(silhouette_scores) - min(silhouette_scores)) / \
        (max(silhouette_scores) - min(silhouette_scores) + 1e-8)

    ax4.plot(k_values, inertia_norm, 'o-', label='Elbow (normalized, inverted)',
             color='orange', linewidth=2, markersize=6)
    ax4.plot(k_values, sil_norm, 's-', label='Silhouette (normalized)',
             color='blue', linewidth=2, markersize=6)

    ax4.axvline(best_k_elbow, color='orange', linestyle='--',
                linewidth=1.5, alpha=0.7, label=f'Elbow k={best_k_elbow}')
    ax4.axvline(best_k_sil, color='blue', linestyle='--',
                linewidth=1.5, alpha=0.7, label=f'Silhouette k={best_k_sil}')

    ax4.set_xlabel('k (number of clusters)', fontsize=11)
    ax4.set_ylabel('Normalized score', fontsize=11)
    ax4.set_title('Elbow vs silhouette', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9)

    # 6. Combined comparison (all metrics normalized)
    ax5 = axes[1, 2]
    dbi_norm = 1 - (np.array(dbi_scores) - min(dbi_scores)) / \
        (max(dbi_scores) - min(dbi_scores) + 1e-8)
    ch_norm = (np.array(ch_scores) - min(ch_scores)) / \
        (max(ch_scores) - min(ch_scores) + 1e-8)

    ax5.plot(k_values, inertia_norm, 'o-', label='Elbow (normalized)',
             color='orange', linewidth=2, markersize=5)
    ax5.plot(k_values, sil_norm, 's-', label='Silhouette (normalized)',
             color='blue', linewidth=2, markersize=5)
    ax5.plot(k_values, dbi_norm, '^-', label='DBI (normalized, inverted)',
             color='green', linewidth=2, markersize=5)
    ax5.plot(k_values, ch_norm, 'd-', label='CH (normalized)',
             color='purple', linewidth=2, markersize=5)

    suggested_k = recommendations['suggested']['k']
    ax5.axvline(suggested_k, color='red',
                linestyle='--', linewidth=2, alpha=0.8, label=f'Suggested k={suggested_k}')

    ax5.set_xlabel('k (number of clusters)', fontsize=11)
    ax5.set_ylabel('Normalized score', fontsize=11)
    ax5.set_title(f'Combined (suggested k={suggested_k})', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=8)

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'k_selection_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nVisualization saved to: {output_path}")


def save_results(
    k_values: list,
    inertia_scores: list,
    silhouette_scores: list,
    dbi_scores: list,
    ch_scores: list,
    recommendations: dict,
    output_dir: str
):
    """Save evaluation results to CSV and summary text."""
    os.makedirs(output_dir, exist_ok=True)

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
    print(f"Evaluation results saved to: {csv_path}")

    summary_path = os.path.join(output_dir, 'k_recommendations.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("K-Means optimal k selection results\n")
        f.write("=" * 60 + "\n\n")

        f.write("Best k per metric:\n")
        f.write(
            f"  - Elbow method: k={recommendations['elbow']['k']} (inertia={recommendations['elbow']['score']:.2f})\n")
        f.write(
            f"  - Silhouette score: k={recommendations['silhouette']['k']} (score={recommendations['silhouette']['score']:.4f})\n")
        f.write(
            f"  - Davies-Bouldin Index: k={recommendations['dbi']['k']} (score={recommendations['dbi']['score']:.4f}, lower is better)\n")
        f.write(
            f"  - Calinski-Harabasz Index: k={recommendations['ch']['k']} (score={recommendations['ch']['score']:.4f})\n")
        f.write("\n")

        f.write("Combined suggestion:\n")
        f.write(f"  Recommended k = {recommendations['suggested']['k']}\n")
        f.write(f"  Reason: {recommendations['suggested']['reason']}\n")
        f.write("\n")

        f.write("Notes:\n")
        f.write("  - Elbow method: find k where WCSS decrease rate slows sharply\n")
        f.write("  - Silhouette score: higher is better [-1, 1]; measures cohesion vs separation\n")
        f.write("  - Davies-Bouldin Index: lower is better; cluster compactness vs separation\n")
        f.write("  - Calinski-Harabasz Index: higher is better; ratio of between/within variance\n")

    print(f"Recommendations saved to: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Find optimal k for K-Means clustering")
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
            f"Embedding key in npz (default: {DEFAULT_CONFIG['embedding_key']}). "
            f"Use h_spatial / h_od / z, or 'h'/'concat' with --concat_keys. "
            f"If missing, auto-selects a 2D embedding."
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
        '--k_min',
        type=int,
        default=DEFAULT_CONFIG['k_min'],
        help=f'Minimum k (default: {DEFAULT_CONFIG["k_min"]})'
    )
    parser.add_argument(
        '--k_max',
        type=int,
        default=DEFAULT_CONFIG['k_max'],
        help=f'Maximum k (default: {DEFAULT_CONFIG["k_max"]})'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=DEFAULT_CONFIG['output_dir'],
        help=f'Output directory (default: {DEFAULT_CONFIG["output_dir"]})'
    )
    parser.add_argument(
        '--random_state',
        type=int,
        default=DEFAULT_CONFIG['random_state'],
        help=f'Random seed (default: {DEFAULT_CONFIG["random_state"]})'
    )
    parser.add_argument(
        '--standardize',
        action='store_true',
        help=f'Standardize features (default: {DEFAULT_CONFIG["standardize"]}; embeddings often already normalized)'
    )
    parser.add_argument(
        '--use_pca',
        action='store_true',
        help=f'Apply PCA (default: {DEFAULT_CONFIG["use_pca"]})'
    )
    parser.add_argument(
        '--pca_components',
        type=int,
        default=DEFAULT_CONFIG['pca_components'],
        help=f'PCA target dimension (default: {DEFAULT_CONFIG["pca_components"]}; only with --use_pca)'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    use_standardize = DEFAULT_CONFIG['standardize'] if not args.standardize else args.standardize
    use_pca = DEFAULT_CONFIG['use_pca'] if not args.use_pca else args.use_pca

    print("Loading embeddings...")
    embeddings = load_embeddings(
        args.embeddings, args.embedding_key, args.concat_keys)
    print(f"Embedding shape: {embeddings.shape}")

    if use_standardize:
        print("Standardizing features...")
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)

    if use_pca:
        original_dim = embeddings.shape[1]
        target_dim = args.pca_components

        if original_dim <= target_dim:
            print(f"[Warning] Original dim ({original_dim}) <= target dim ({target_dim}); skipping PCA")
        else:
            print(f"Applying PCA: {original_dim} -> {target_dim}...")
            pca = PCA(n_components=target_dim, random_state=args.random_state)
            embeddings = pca.fit_transform(embeddings)
            explained_var = sum(pca.explained_variance_ratio_)
            print(
                f"PCA done; variance retained: {explained_var:.4f} ({explained_var*100:.2f}%)")
            print(f"Embedding shape after PCA: {embeddings.shape}")

    k_values, inertia_scores, silhouette_scores, dbi_scores, ch_scores = find_optimal_k(
        embeddings,
        args.k_min,
        args.k_max,
        args.random_state
    )

    print("\n" + "=" * 60)
    print("Optimal k recommendations:")
    print("=" * 60)
    recommendations = suggest_optimal_k(
        k_values, inertia_scores, silhouette_scores, dbi_scores, ch_scores)

    print(f"\nPer-metric recommendations:")
    print(
        f"  - Elbow method: k={recommendations['elbow']['k']} (inertia={recommendations['elbow']['score']:.2f})")
    print(
        f"  - Silhouette score: k={recommendations['silhouette']['k']} (score={recommendations['silhouette']['score']:.4f})")
    print(
        f"  - Davies-Bouldin Index: k={recommendations['dbi']['k']} (score={recommendations['dbi']['score']:.4f}, lower is better)")
    print(
        f"  - Calinski-Harabasz Index: k={recommendations['ch']['k']} (score={recommendations['ch']['score']:.4f})")
    print(f"\nCombined suggestion: k={recommendations['suggested']['k']}")
    print(f"Reason: {recommendations['suggested']['reason']}")

    print("\nGenerating plots...")
    visualize_k_selection(k_values, inertia_scores, silhouette_scores,
                          dbi_scores, ch_scores, recommendations, args.output_dir)

    save_results(k_values, inertia_scores, silhouette_scores, dbi_scores,
                 ch_scores, recommendations, args.output_dir)

    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
