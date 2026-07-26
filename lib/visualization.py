"""
Reusable visualization functions for projection and clustering analysis.

Provides utilities for plotting projections, metrics, and clustering results
with consistent styling.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lib import config


# ============================================================================
# PROJECTION VISUALIZATION
# ============================================================================

def plot_projection_scatter(data, ax=None, title="Projection", alpha=None, size=None, **scatter_kw):
    """
    Create a basic scatter plot of 2D projection data.

    Args:
        data: DataFrame with 'X' and 'Y' columns
        ax: Matplotlib axis (creates new if None)
        title: Plot title
        alpha: Point transparency
        size: Point size
        **scatter_kw: Additional kwargs for scatter()

    Returns:
        Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    if alpha is None:
        alpha = config.SCATTER_ALPHA
    if size is None:
        size = config.SCATTER_SIZE

    ax.scatter(data['X'], data['Y'], alpha=alpha, s=size, **scatter_kw)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return ax


def plot_projection_comparison(projections_dict, figsize=None, suptitle="Projection Methods"):
    """
    Create a grid comparing multiple projection methods side-by-side.

    Args:
        projections_dict: Dict mapping method_name -> DataFrame with X,Y columns
        figsize: Figure size (default from config)
        suptitle: Overall figure title

    Returns:
        Matplotlib figure and axes
    """
    if figsize is None:
        figsize = config.PROJECTION_COMPARISON_SIZE

    n_methods = len(projections_dict)
    n_cols = min(3, n_methods)
    n_rows = (n_methods + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_methods == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (method_name, df) in enumerate(projections_dict.items()):
        plot_projection_scatter(df, ax=axes[idx], title=method_name)

    # Hide unused subplots
    for idx in range(n_methods, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(suptitle, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    return fig, axes


# ============================================================================
# METRICS VISUALIZATION
# ============================================================================

def plot_metric_comparison(df_metrics, metrics=None, figsize=None, suptitle="Quality Metrics"):
    """
    Create bar charts comparing metrics across projection methods.

    Args:
        df_metrics: DataFrame with 'Method' column and metric columns
        metrics: List of metric column names to plot (default: all except 'Method')
        figsize: Figure size
        suptitle: Overall title

    Returns:
        Matplotlib figure and axes
    """
    if figsize is None:
        figsize = config.QUALITY_METRICS_SIZE

    if metrics is None:
        metrics = [col for col in df_metrics.columns if col != 'Method']

    n_metrics = len(metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Sort by metric value
        df_sorted = df_metrics.sort_values(metric, ascending=False)

        color = config.METRIC_COLORS.get(metric, 'steelblue')
        bars = ax.barh(df_sorted['Method'], df_sorted[metric], color=color, alpha=0.7)

        ax.set_xlabel(metric, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric} by Method', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # Add value labels and highlight best
        for i, (bar, val) in enumerate(zip(bars, df_sorted[metric])):
            if not np.isnan(val):
                ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9)

        # Highlight best method with red border
        if not df_sorted[metric].isna().all():
            best_idx = df_sorted[metric].idxmax()
            bars[df_sorted.index.get_loc(best_idx)].set_edgecolor('red')
            bars[df_sorted.index.get_loc(best_idx)].set_linewidth(3)

    # Hide unused subplots
    for idx in range(len(metrics), len(axes)):
        axes[idx].axis('off')

    plt.suptitle(suptitle, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    return fig, axes


def plot_metrics_heatmap(df_metrics, metrics=None, figsize=(14, 8)):
    """
    Create a normalized heatmap of metrics for easy comparison.

    Args:
        df_metrics: DataFrame with metrics
        metrics: List of metric columns to include
        figsize: Figure size

    Returns:
        Matplotlib figure and axis
    """
    if metrics is None:
        metrics = [col for col in df_metrics.columns if col != 'Method']

    # Prepare data
    data_for_heatmap = df_metrics.set_index('Method')[metrics].copy()

    # Normalize each column to 0-1
    for col in data_for_heatmap.columns:
        col_min = data_for_heatmap[col].min()
        col_max = data_for_heatmap[col].max()
        if col_max > col_min:
            data_for_heatmap[col] = (data_for_heatmap[col] - col_min) / (col_max - col_min)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(data_for_heatmap, annot=True, fmt='.3f', cmap='YlOrRd',
                cbar_kws={'label': 'Normalized Score (0-1)'}, ax=ax,
                linewidths=0.5, linecolor='gray')

    ax.set_title('Normalized Quality Metrics Heatmap', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Quality Metric', fontsize=12, fontweight='bold')
    ax.set_ylabel('Projection Method', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    return fig, ax


# ============================================================================
# CLUSTERING VISUALIZATION
# ============================================================================

def plot_clustering_results(proj_coords, labels, ax=None, title="Clustering Results",
                            show_noise=True):
    """
    Create a scatter plot of clustering results.

    Args:
        proj_coords: 2D projection coordinates (n_samples, 2)
        labels: Cluster labels (n_samples,). Use -1 for noise points.
        ax: Matplotlib axis
        title: Plot title
        show_noise: If True, highlight noise points with X markers

    Returns:
        Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    # Plot clusters
    mask_valid = labels != -1
    scatter = ax.scatter(proj_coords[mask_valid, 0], proj_coords[mask_valid, 1],
                        c=labels[mask_valid], cmap=config.SCATTER_CMAP,
                        alpha=0.6, s=10, edgecolors='none')

    # Plot noise points if present
    if show_noise and (labels == -1).sum() > 0:
        noise_mask = labels == -1
        ax.scatter(proj_coords[noise_mask, 0], proj_coords[noise_mask, 1],
                  c='black', marker='x', s=20, alpha=0.5, label='Noise')

    ax.set_xlabel('Dimension 1', fontsize=10)
    ax.set_ylabel('Dimension 2', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    if show_noise and (labels == -1).sum() > 0:
        ax.legend()

    return ax


def plot_kmeans_grid(kmeans_results, proj_coords, k_values, figsize=None):
    """
    Create a 2x2 grid of k-Means clustering results.

    Args:
        kmeans_results: Dict mapping k -> {'labels': labels, 'model': model, ...}
        proj_coords: 2D projection coordinates
        k_values: List of k values
        figsize: Figure size

    Returns:
        Matplotlib figure and axes
    """
    if figsize is None:
        figsize = config.KMEANS_COMPARISON_SIZE
    
    n_plots = len(k_values)
    print(n_plots)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    print(n_rows, n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for idx, k in enumerate(k_values):
        ax = axes[idx]
        labels = kmeans_results[k]['labels']

        plot_clustering_results(proj_coords, labels, ax=ax,
                               title=f"k-Means (k={k})\nSilhouette: {kmeans_results[k]['silhouette']:.3f}",
                               show_noise=False)

        # Plot cluster centers
        centers = kmeans_results[k]['model'].cluster_centers_
        ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='x',
                  s=200, linewidths=3, label='Centroids')
        ax.legend()

    plt.suptitle('k-Means Clustering Results', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    return fig, axes


# ============================================================================
# METRIC TRENDS
# ============================================================================

def plot_kmeans_validation_metrics(kmeans_df, figsize=None):
    """
    Plot k-Means validation metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz).

    Args:
        kmeans_df: DataFrame with k and metric columns
        figsize: Figure size

    Returns:
        Matplotlib figure and axes
    """
    if figsize is None:
        figsize = config.KMEANS_METRICS_SIZE

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Find best k values
    best_k_silhouette = kmeans_df.loc[kmeans_df['silhouette'].idxmax(), 'k']
    best_k_davies = kmeans_df.loc[kmeans_df['davies_bouldin'].idxmin(), 'k']

    # Silhouette Score
    axes[0].plot(kmeans_df['k'], kmeans_df['silhouette'], 'o-', linewidth=2, markersize=8)
    axes[0].axvline(best_k_silhouette, color='r', linestyle='--', alpha=0.7,
                   label=f'Best k={int(best_k_silhouette)}')
    axes[0].set_xlabel('Number of Clusters (k)', fontsize=11)
    axes[0].set_ylabel('Silhouette Score', fontsize=11)
    axes[0].set_title('Silhouette Score (Higher is Better)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Davies-Bouldin Index
    axes[1].plot(kmeans_df['k'], kmeans_df['davies_bouldin'], 'o-',
                linewidth=2, markersize=8, color='orange')
    axes[1].axvline(best_k_davies, color='r', linestyle='--', alpha=0.7,
                   label=f'Best k={int(best_k_davies)}')
    axes[1].set_xlabel('Number of Clusters (k)', fontsize=11)
    axes[1].set_ylabel('Davies-Bouldin Index', fontsize=11)
    axes[1].set_title('Davies-Bouldin Index (Lower is Better)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Calinski-Harabasz Index
    axes[2].plot(kmeans_df['k'], kmeans_df['calinski_harabasz'], 'o-',
                linewidth=2, markersize=8, color='green')
    axes[2].set_xlabel('Number of Clusters (k)', fontsize=11)
    axes[2].set_ylabel('Calinski-Harabasz Index', fontsize=11)
    axes[2].set_title('Calinski-Harabasz Index (Higher is Better)', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('k-Means Validation Metrics', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig, axes


def plot_kdistance_graph(distances, k=15, figsize=None):
    """
    Plot k-distance graph for DBSCAN eps selection.

    Args:
        distances: k-th nearest neighbor distances, sorted
        k: k value used (for labeling)
        figsize: Figure size

    Returns:
        Matplotlib figure and axis
    """
    if figsize is None:
        figsize = config.DBSCAN_K_DISTANCE_SIZE

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(np.sort(distances))
    ax.set_xlabel('Points sorted by distance', fontsize=11)
    ax.set_ylabel(f'{k}-NN distance', fontsize=11)
    ax.set_title('k-Distance Graph for DBSCAN eps Selection', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add percentile lines
    ax.axhline(y=np.percentile(distances, 90), color='r', linestyle='--',
              label=f'90th percentile: {np.percentile(distances, 90):.3f}')
    ax.axhline(y=np.percentile(distances, 95), color='orange', linestyle='--',
              label=f'95th percentile: {np.percentile(distances, 95):.3f}')

    ax.legend()
    plt.tight_layout()

    return fig, ax


def plot_dbscan_grid(dbscan_results_list, proj_coords, figsize=None, suptitle="Top DBSCAN Results"):
    """
    Create a 2x2 grid of top DBSCAN clustering results.

    Args:
        dbscan_results_list: List of top DBSCAN result dicts (4 or fewer)
        proj_coords: 2D projection coordinates
        figsize: Figure size
        suptitle: Overall title

    Returns:
        Matplotlib figure and axes
    """
    if figsize is None:
        figsize = config.DBSCAN_COMPARISON_SIZE

    n_results = min(len(dbscan_results_list), 4)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for idx, result in enumerate(dbscan_results_list[:4]):
        ax = axes[idx]
        labels = result['labels']

        title = f"DBSCAN (eps={result['eps']:.3f}, min_samples={result['min_samples']})\n"
        title += f"Clusters: {result['n_clusters']}, Noise: {result['n_noise']}"
        if not np.isnan(result['silhouette']):
            title += f", Silhouette: {result['silhouette']:.3f}"

        plot_clustering_results(proj_coords, labels, ax=ax, title=title, show_noise=True)

    # Hide unused subplots
    for idx in range(n_results, 4):
        axes[idx].axis('off')

    plt.suptitle(suptitle, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    return fig, axes


# ============================================================================
# COMPREHENSIVE CLUSTERING ANALYSIS
# ============================================================================

def plot_clustering_comprehensive(results_df, method='kmeans', best_indices=None,
                                  figsize=(20, 12)):
    """
    Create comprehensive clustering analysis grid (3x3 or 3x2).

    Shows unified visualization for both k-Means and DBSCAN with:
    - Row 1: Core metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz)
    - Row 2: Interpretability metrics (n_clusters, noise/balance, avg_cluster_size)
    - Row 3: Composite score

    Args:
        results_df: DataFrame with clustering results
        method: 'kmeans' or 'dbscan'
        best_indices: Dict with 'by_silhouette', 'by_davies', 'by_composite', 'interpretable'
        figsize: Figure size

    Returns:
        Matplotlib figure and axes
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    if method == 'kmeans':
        param_col = 'k'
        x_label = 'Number of Clusters (k)'
        x_vals = results_df['k'].values
    else:
        param_col = 'config_idx'
        x_label = 'Configuration Index'
        x_vals = range(len(results_df))

    # ========== ROW 1: CORE METRICS ==========

    # Silhouette Score
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x_vals, results_df['silhouette'], 'o-', linewidth=2, markersize=6, color='steelblue')
    if best_indices and 'by_silhouette' in best_indices:
        ax1.axvline(x_vals[best_indices['by_silhouette']], color='red', linestyle='--',
                   alpha=0.7, linewidth=2, label='Best')
    ax1.set_ylabel('Silhouette Score', fontsize=10, fontweight='bold')
    ax1.set_title('Silhouette Score (Higher is Better)', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    if best_indices:
        ax1.legend(fontsize=9)

    # Davies-Bouldin Index
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(x_vals, results_df['davies_bouldin'], 'o-', linewidth=2, markersize=6, color='coral')
    if best_indices and 'by_davies' in best_indices:
        ax2.axvline(x_vals[best_indices['by_davies']], color='red', linestyle='--',
                   alpha=0.7, linewidth=2, label='Best')
    ax2.set_ylabel('Davies-Bouldin Index', fontsize=10, fontweight='bold')
    ax2.set_title('Davies-Bouldin Index (Lower is Better)', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    if best_indices:
        ax2.legend(fontsize=9)

    # Calinski-Harabasz Index
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(x_vals, results_df['calinski_harabasz'], 'o-', linewidth=2, markersize=6,
            color='mediumseagreen')
    ax3.set_ylabel('Calinski-Harabasz Index', fontsize=10, fontweight='bold')
    ax3.set_title('Calinski-Harabasz Index (Higher is Better)', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # ========== ROW 2: INTERPRETABILITY METRICS ==========

    # Number of Clusters
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(x_vals, results_df['n_clusters'], 'o-', linewidth=2, markersize=6, color='purple')
    ax4.axhline(y=config.DBSCAN_INTERPRETABLE_K_MIN, color='orange', linestyle=':', alpha=0.7,
               label=f'Target: {config.DBSCAN_INTERPRETABLE_K_MIN}-{config.DBSCAN_INTERPRETABLE_K_MAX}')
    ax4.axhline(y=config.DBSCAN_INTERPRETABLE_K_MAX, color='orange', linestyle=':', alpha=0.7)
    ax4.set_ylabel('Number of Clusters', fontsize=10, fontweight='bold')
    ax4.set_title('Number of Clusters', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8)

    # Cluster Balance / Noise Percentage
    ax5 = fig.add_subplot(gs[1, 1])
    if method == 'kmeans' and 'cluster_balance' in results_df.columns:
        ax5.plot(x_vals, results_df['cluster_balance'], 'o-', linewidth=2, markersize=6,
                color='crimson')
        ax5.set_ylabel('Cluster Balance', fontsize=10, fontweight='bold')
        ax5.set_title('Cluster Balance (Higher = More Even Distribution)', fontsize=11, fontweight='bold')
        ax5.set_ylim([0, 1])
        ax5.axhline(y=0.7, color='orange', linestyle='--', alpha=0.7, label='Min: 0.7')
        ax5.legend(fontsize=8)
    else:
        ax5.plot(x_vals, results_df['noise_pct'], 'o-', linewidth=2, markersize=6, color='crimson')
        ax5.set_ylabel('Noise Percentage (%)', fontsize=10, fontweight='bold')
        ax5.set_title('Noise Percentage (Lower is Better)', fontsize=11, fontweight='bold')
        ax5.axhline(y=config.DBSCAN_MAX_NOISE_PERCENTAGE, color='orange', linestyle='--',
                   alpha=0.7, label=f'Max: {config.DBSCAN_MAX_NOISE_PERCENTAGE}%')
        ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # Average Cluster Size
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(x_vals, results_df['avg_cluster_size'], 'o-', linewidth=2, markersize=6, color='teal')
    ax6.set_ylabel('Avg Cluster Size', fontsize=10, fontweight='bold')
    ax6.set_title('Average Cluster Size', fontsize=11, fontweight='bold')
    ax6.axhline(y=500, color='orange', linestyle='--', alpha=0.7, label='Min: 500')
    ax6.grid(True, alpha=0.3)
    ax6.legend(fontsize=8)

    # ========== ROW 3: COMPOSITE SCORE ==========

    ax7 = fig.add_subplot(gs[2, :])
    if 'composite_score' in results_df.columns:
        bars = ax7.bar(range(len(results_df)), results_df['composite_score'],
                      color='steelblue', alpha=0.7, edgecolor='black')

        if best_indices:
            if 'by_composite' in best_indices:
                ax7.axvline(best_indices['by_composite'], color='red', linestyle='--',
                           linewidth=2, label=f"Best composite: idx {best_indices['by_composite']}")

            if 'interpretable' in best_indices:
                ax7.axvline(best_indices['interpretable'], color='green', linestyle='--',
                           linewidth=2, label=f"Best interpretable: idx {best_indices['interpretable']}")

        ax7.set_xticks(range(len(results_df)))
        if method == 'kmeans':
            ax7.set_xticklabels([f"k={int(k)}" for k in results_df['k']], fontsize=9)
        else:
            ax7.set_xticklabels([f"#{i}" for i in range(len(results_df))], fontsize=9)

        ax7.set_ylabel('Composite Score', fontsize=11, fontweight='bold')
        ax7.set_xlabel(x_label, fontsize=11, fontweight='bold')
        ax7.set_title('Composite Quality Score', fontsize=12, fontweight='bold')
        ax7.set_ylim([0, 1])
        ax7.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, results_df['composite_score'])):
            ax7.text(i, score + 0.02, f'{score:.2f}', ha='center', va='bottom',
                    fontsize=8)

        if best_indices:
            ax7.legend(fontsize=9, loc='upper right')

    plt.suptitle(f'{method.upper()} Clustering - Comprehensive Analysis',
                fontsize=14, fontweight='bold', y=0.995)
    # plt.tight_layout()

    return fig, ax7
