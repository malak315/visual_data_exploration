"""
Unified clustering evaluation and visualization for k-Means and DBSCAN.

Provides consistent interfaces for both clustering methods with unified
composite scoring and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from lib import config


# ============================================================================
# CORE CLUSTERING EVALUATION
# ============================================================================

def evaluate_clustering(coords, labels, model=None, method='kmeans'):
    """
    Compute validation metrics for any clustering method.

    Args:
        coords: Data coordinates (n_samples, n_features)
        labels: Cluster labels (n_samples,)
        model: Optional fitted clustering model (for centroids, etc.)
        method: 'kmeans' or 'dbscan'

    Returns:
        Dictionary with clustering metrics
    """
    results = {}

    # Core metrics - only compute on non-noise points
    mask = labels != -1
    if mask.sum() < 2:
        return {
            'silhouette': np.nan,
            'davies_bouldin': np.nan,
            'calinski_harabasz': np.nan,
            'n_clusters': len(set(labels)) - (1 if -1 in labels else 0)
        }

    results['silhouette'] = silhouette_score(coords[mask], labels[mask])
    results['davies_bouldin'] = davies_bouldin_score(coords[mask], labels[mask])
    results['calinski_harabasz'] = calinski_harabasz_score(coords[mask], labels[mask])

    # Basic clustering info
    results['n_clusters'] = len(set(labels)) - (1 if -1 in labels else 0)

    # DBSCAN-specific metrics
    if method == 'dbscan':
        n_noise = list(labels).count(-1)
        results['n_noise'] = n_noise
        results['noise_pct'] = 100 * n_noise / len(labels)
        n_clustered = len(labels) - n_noise
        if results['n_clusters'] > 0:
            results['avg_cluster_size'] = n_clustered / results['n_clusters']
        else:
            results['avg_cluster_size'] = 0

    # k-Means-specific metrics
    elif method == 'kmeans':
        if len(labels) > 0:
            cluster_sizes = np.bincount(labels)
            results['avg_cluster_size'] = np.mean(cluster_sizes)
            results['cluster_size_std'] = np.std(cluster_sizes)
            results['cluster_balance'] = 1.0 - (results['cluster_size_std'] / results['avg_cluster_size'])

    return results


# ============================================================================
# k-MEANS PARAMETER TESTING
# ============================================================================

def test_kmeans_parameters(coords, k_values=None, subsample_size=None, progress=True):
    """
    Test k-Means with multiple k values and compute metrics.

    Args:
        coords: Data coordinates (n_samples, n_features)
        k_values: List of k values to test (default: config.KMEANS_K_VALUES)
        subsample_size: Optional subsample size for efficiency
        progress: Show progress bar

    Returns:
        Tuple of (results_dict, results_dataframe)
    """
    if k_values is None:
        k_values = config.KMEANS_K_VALUES

    # create k random initializations
    np.random.seed(config.RANDOM_STATE)
    random_seeds = np.random.randint(0, 10000, size=len(k_values))

    # Sample if needed
    if subsample_size and coords.shape[0] > subsample_size:
        np.random.seed(config.RANDOM_SEED)
        sample_idx = np.random.choice(coords.shape[0], subsample_size, replace=False)
        coords_sample = coords[sample_idx]
    else:
        coords_sample = coords
        subsample_size = coords.shape[0]

    results_dict = {}
    results_list = []

    iterator = tqdm(k_values, desc="k-Means") if progress else k_values

    for i, k in enumerate(iterator):

        kmeans = KMeans(
            n_clusters=k,
            random_state=random_seeds[i],
            n_init=config.KMEANS_N_INIT,
            init='k-means++',
            max_iter=config.KMEANS_MAX_ITER
        )
        labels = kmeans.fit_predict(coords_sample)
        metrics = evaluate_clustering(coords_sample, labels, model=kmeans, method='kmeans')

        results_dict[k] = {
            'labels': labels,
            'model': kmeans,
            **metrics
        }

        row = {'k': k, **metrics}
        results_list.append(row)

    df_results = pd.DataFrame(results_list)

    return results_dict, df_results


# ============================================================================
# DBSCAN PARAMETER TESTING
# ============================================================================

def estimate_dbscan_eps(coords, k=15, subsample_size=50000, plot=True):
    """
    Estimate DBSCAN eps using k-distance graph.

    Args:
        coords: Data coordinates
        k: k for k-nearest neighbors
        subsample_size: Optional subsample for efficiency
        plot: Whether to plot k-distance graph

    Returns:
        Dictionary with eps suggestions and optionally the plot
    """
    # Sample if needed
    if subsample_size and coords.shape[0] > subsample_size:
        np.random.seed(config.RANDOM_SEED)
        sample_idx = np.random.choice(coords.shape[0], subsample_size, replace=False)
        coords_sample = coords[sample_idx]
    else:
        coords_sample = coords

    # Compute k-distances
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(coords_sample)
    distances, indices = neighbors.kneighbors(coords_sample)
    distances = np.sort(distances[:, k - 1])

    result = {
        'distances': distances,
        'eps_98': np.percentile(distances, 98),
        'eps_99': np.percentile(distances, 99),
        'eps_99_5': np.percentile(distances, 99.5),
    }

    if plot:
        from . import visualization
        fig, ax = visualization.plot_kdistance_graph(distances, k=k)
        result['figure'] = fig

    return result


def test_dbscan_parameters(coords, eps_values=None, min_samples_values=None,
                           subsample_size=None, progress=True):
    """
    Test DBSCAN with multiple parameter combinations.

    Args:
        coords: Data coordinates
        eps_values: List of eps values to test
        min_samples_values: List of min_samples values to test
        subsample_size: Optional subsample size for efficiency
        progress: Show progress bar

    Returns:
        Tuple of (results_dict, results_dataframe)
    """
    if eps_values is None:
        # Estimate eps if not provided
        eps_dict = estimate_dbscan_eps(coords, k=config.DBSCAN_K_NEIGHBORS,
                                       subsample_size=subsample_size, plot=False)
        eps_values = [eps_dict[f'eps_{p}'] for p in config.DBSCAN_EPS_PERCENTILES]

    if min_samples_values is None:
        min_samples_values = config.DBSCAN_MIN_SAMPLES_VALUES

    # Sample if needed
    if subsample_size and coords.shape[0] > subsample_size:
        np.random.seed(config.RANDOM_SEED)
        sample_idx = np.random.choice(coords.shape[0], subsample_size, replace=False)
        coords_sample = coords[sample_idx]
    else:
        coords_sample = coords
        subsample_size = coords.shape[0]

    results_dict = {}
    results_list = []
    test_idx = 0

    total_tests = len(eps_values) * len(min_samples_values)
    iterator = tqdm(total=total_tests, desc="DBSCAN") if progress else None

    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(coords_sample)

            metrics = evaluate_clustering(coords_sample, labels, method='dbscan')

            results_dict[test_idx] = {
                'eps': eps,
                'min_samples': min_samples,
                'labels': labels,
                **metrics
            }

            row = {'eps': eps, 'min_samples': min_samples, **metrics}
            results_list.append(row)

            test_idx += 1
            if iterator:
                iterator.update(1)

    if iterator:
        iterator.close()

    df_results = pd.DataFrame(results_list)
    return results_dict, df_results


# ============================================================================
# COMPOSITE SCORING
# ============================================================================

def calculate_composite_score(results_df, method='kmeans'):
    """
    Calculate unified composite quality score.

    Args:
        results_df: DataFrame with clustering results
        method: 'kmeans' or 'dbscan'

    Returns:
        Series with composite scores
    """
    # Normalize core metrics to 0-1
    norm_silhouette = (results_df['silhouette'] - results_df['silhouette'].min()) / \
                      (results_df['silhouette'].max() - results_df['silhouette'].min() + 1e-10)

    norm_davies = 1 - ((results_df['davies_bouldin'] - results_df['davies_bouldin'].min()) / \
                       (results_df['davies_bouldin'].max() - results_df['davies_bouldin'].min() + 1e-10))

    norm_calinski = (results_df['calinski_harabasz'] - results_df['calinski_harabasz'].min()) / \
                    (results_df['calinski_harabasz'].max() - results_df['calinski_harabasz'].min() + 1e-10)

    # Base score from core metrics
    composite = (
        0.30 * norm_silhouette +
        0.20 * norm_davies +
        0.20 * norm_calinski
    )

    # Add method-specific components
    if method == 'kmeans':
        # For k-Means: add cluster balance and k-range penalty
        if 'cluster_balance' in results_df.columns:
            composite += 0.20 * results_df['cluster_balance']
        else:
            composite += 0.20

        # Penalty: prefer 5-15 clusters
        k_penalty = np.where(
            (results_df['k'] >= config.KMEANS_INTERPRETABLE_K_MIN) &
            (results_df['k'] <= config.KMEANS_INTERPRETABLE_K_MAX),
            1.0, 0.5
        )
        composite += 0.10 * k_penalty

    elif method == 'dbscan':
        # For DBSCAN: add outlier handling and cluster count penalty
        if 'noise_pct' in results_df.columns:
            norm_noise = 1 - (results_df['noise_pct'] / 100)
            composite += 0.25 * norm_noise
        else:
            composite += 0.25

        # Penalty: prefer 5-20 clusters
        cluster_penalty = np.where(
            (results_df['n_clusters'] >= config.DBSCAN_INTERPRETABLE_K_MIN) &
            (results_df['n_clusters'] <= config.DBSCAN_INTERPRETABLE_K_MAX),
            1.0, 0.5
        )
        composite += 0.05 * cluster_penalty

    return composite


# ============================================================================
# RESULTS MANAGEMENT
# ============================================================================

def display_clustering_summary(results_df, method='kmeans', top_n=10):
    """
    Display formatted summary of clustering results.

    Args:
        results_df: Results DataFrame
        method: 'kmeans' or 'dbscan'
        top_n: Number of top results to display
    """
    if method == 'kmeans':
        display_cols = ['k', 'silhouette', 'davies_bouldin', 'calinski_harabasz',
                       'avg_cluster_size', 'cluster_balance', 'composite_score']
    else:
        display_cols = ['eps', 'min_samples', 'n_clusters', 'noise_pct',
                       'silhouette', 'davies_bouldin', 'calinski_harabasz', 'composite_score']

    display_cols = [c for c in display_cols if c in results_df.columns]
    return results_df[display_cols].head(top_n)


def find_best_clustering(results_df, method='kmeans', criteria=None):
    """
    Find best clustering configuration by various criteria.

    Args:
        results_df: Results DataFrame
        method: 'kmeans' or 'dbscan'
        criteria: Dict of criteria to apply (optional)

    Returns:
        Dictionary with best indices and configurations
    """
    best = {
        'by_silhouette': results_df['silhouette'].idxmax(),
        'by_davies': results_df['davies_bouldin'].idxmin(),
        'by_calinski': results_df['calinski_harabasz'].idxmax(),
    }

    if 'composite_score' in results_df.columns:
        best['by_composite'] = results_df['composite_score'].idxmax()

    # Method-specific best interpretable configuration
    if method == 'kmeans':
        mask = (
            (results_df['k'] >= config.KMEANS_INTERPRETABLE_K_MIN) &
            (results_df['k'] <= config.KMEANS_INTERPRETABLE_K_MAX)
        )
    else:
        mask = (
            (results_df['n_clusters'] >= config.DBSCAN_INTERPRETABLE_K_MIN) &
            (results_df['n_clusters'] <= config.DBSCAN_INTERPRETABLE_K_MAX) &
            (results_df['noise_pct'] < config.DBSCAN_MAX_NOISE_PERCENTAGE) &
            (results_df['avg_cluster_size'] > config.DBSCAN_MIN_CLUSTER_SIZE)
        )

    if mask.sum() > 0:
        best_interpretable_idx = results_df[mask]['composite_score'].idxmax()
        best['interpretable'] = best_interpretable_idx

    return best
