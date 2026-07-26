"""
Quality metrics for projection and clustering evaluation.

Provides metrics for evaluating projection quality and clustering performance,
with utilities for batch computation and comparison.
"""

import numpy as np
import pandas as pd
from sklearn.manifold import trustworthiness as sklearn_trustworthiness
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import accuracy_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from tqdm import tqdm

from lib import config
from lib import logger_utils

logger = logger_utils.get_logger("projection_metrics")


# ============================================================================
# PROJECTION QUALITY METRICS
# ============================================================================

def compute_trustworthiness(X_high, X_low, n_neighbors=10):
    """
    Compute trustworthiness score - measures local neighborhood preservation.

    Higher values (0-1) indicate better preservation of local structure.

    Args:
        X_high: High-dimensional data (n_samples, n_features_high)
        X_low: Low-dimensional projection (n_samples, n_features_low)
        n_neighbors: Number of neighbors to consider

    Returns:
        Trustworthiness score (float)
    """
    return sklearn_trustworthiness(X_high, X_low, n_neighbors=n_neighbors)


def compute_continuity(X_high, X_low, n_neighbors=10):
    """
    Compute continuity score - measures global structure preservation.

    Similar to trustworthiness but in reverse direction.
    Higher values (0-1) indicate better preservation.

    Args:
        X_high: High-dimensional data
        X_low: Low-dimensional projection
        n_neighbors: Number of neighbors to consider

    Returns:
        Continuity score (float)
    """
    return sklearn_trustworthiness(X_low, X_high, n_neighbors=n_neighbors)


def compute_knn_overlap(X_high, X_low, k=10):
    """
    Compute kNN overlap - percentage of neighbors preserved in projection.

    Measures how many of the k nearest neighbors in high-D space
    remain neighbors in low-D space.

    Args:
        X_high: High-dimensional data
        X_low: Low-dimensional projection
        k: Number of neighbors to consider

    Returns:
        kNN overlap score (0-1, higher is better)
    """
    n_samples = X_high.shape[0]
    neighbors_high = NearestNeighbors(n_neighbors=k + 1).fit(X_high)
    neighbors_low = NearestNeighbors(n_neighbors=k + 1).fit(X_low)

    overlaps = []
    for i in range(n_samples):
        # Get k nearest neighbors (excluding self)
        nn_high = neighbors_high.kneighbors([X_high[i]], return_distance=False)[0][1:]
        nn_low = neighbors_low.kneighbors([X_low[i]], return_distance=False)[0][1:]

        # Compute overlap
        overlap = len(set(nn_high) & set(nn_low)) / k
        overlaps.append(overlap)

    return np.mean(overlaps)


def compute_distance_correlation(X_high, X_low, n_sample=5000):
    """
    Compute Spearman correlation between distance matrices.

    Measures how well pairwise distances are preserved.

    Args:
        X_high: High-dimensional data
        X_low: Low-dimensional projection
        n_sample: Sample size for efficiency (smaller = faster)

    Returns:
        Correlation coefficient (float, -1 to 1)
    """
    # Sample subset for efficiency
    n_samples = min(n_sample, X_high.shape[0])
    sample_idx = np.random.choice(X_high.shape[0], n_samples, replace=False)

    dist_high = pdist(X_high[sample_idx])
    dist_low = pdist(X_low[sample_idx])

    corr, _ = spearmanr(dist_high, dist_low)
    return corr


def compute_knn_classification_accuracy(X_high, X_low, k=5, test_size=0.2):
    """
    Compute kNN classification accuracy on projected data.

    Trains kNN on high-D space and tests on low-D projection.

    Args:
        X_high: High-dimensional data
        X_low: Low-dimensional projection
        k: Number of neighbors for kNN
        test_size: Fraction of data to use as test set

    Returns:
        Accuracy score (0-1)
    """
    from sklearn.model_selection import train_test_split

    n_samples = X_high.shape[0]
    indices = np.arange(n_samples)

    train_idx, test_idx = train_test_split(indices, test_size=test_size,
                                           random_state=config.RANDOM_SEED)

    # Train kNN on high-D, predict on high-D test set
    knn_high = KNeighborsClassifier(n_neighbors=k)
    knn_high.fit(X_high[train_idx], train_idx)
    y_true = knn_high.predict(X_high[test_idx])

    # Train kNN on low-D, predict on low-D test set
    knn_low = KNeighborsClassifier(n_neighbors=k)
    knn_low.fit(X_low[train_idx], train_idx)
    y_pred = knn_low.predict(X_low[test_idx])

    return accuracy_score(y_true, y_pred)


def compute_projection_quality_metrics(X_high, X_low, method_name="Method"):
    """
    Compute all quality metrics for a single projection.

    Args:
        X_high: High-dimensional data
        X_low: Low-dimensional projection
        method_name: Name of the projection method (for reporting)

    Returns:
        Dictionary with all computed metrics
    """
    logger.info(f"\n{method_name}:")
    results = {'Method': method_name}

    # Trustworthiness
    logger.info("  Computing Trustworthiness...")
    results['Trustworthiness'] = compute_trustworthiness(X_high, X_low, n_neighbors=10)

    # Continuity
    logger.info("  Computing Continuity...")
    results['Continuity'] = compute_continuity(X_high, X_low, n_neighbors=10)

    # kNN Overlap
    logger.info("  Computing kNN Overlap...")
    results['kNN Overlap'] = compute_knn_overlap(X_high, X_low, k=10)

    # Distance Correlation
    logger.info("  Computing Distance Correlation...")
    results['Distance Correlation'] = compute_distance_correlation(X_high, X_low)

    # kNN Classification Accuracy
    logger.info("  Computing kNN Classification Accuracy...")
    try:
        results['kNN Accuracy'] = compute_knn_classification_accuracy(X_high, X_low, k=5)
    except Exception as e:
        logger.info(f"  Warning: kNN classification accuracy failed: {e}")
        results['kNN Accuracy'] = np.nan

    # Print summary
    for metric, value in results.items():
        if metric != 'Method':
            logger.info(f"   {metric}: {value:.4f}" if not np.isnan(value) else f"   {metric}: NaN")

    return results


def compute_all_projection_metrics(X_high, projections_dict, sample_size=None):
    """
    Compute quality metrics for multiple projections.

    Args:
        X_high: High-dimensional data
        projections_dict: Dictionary mapping method_name -> low-D coordinates
        sample_size: Use a sample for faster computation (None = use all)

    Returns:
        DataFrame with all metrics computed
    """
    if sample_size is None:
        sample_size = config.QUALITY_METRICS_SAMPLE_SIZE

    # Sample if needed
    if X_high.shape[0] > sample_size:
        np.random.seed(config.RANDOM_SEED)
        sample_idx = np.random.choice(X_high.shape[0], sample_size, replace=False)
        X_high_sample = X_high[sample_idx]
        logger.info(f"Using {sample_size:,} samples (out of {X_high.shape[0]:,}) for faster computation")
    else:
        X_high_sample = X_high
        sample_idx = np.arange(X_high.shape[0])
        logger.info(f"Using all {X_high.shape[0]:,} samples for metrics")

    results = []
    for method_name, X_low in tqdm(projections_dict.items(), desc="Computing metrics"):
        # Handle both DataFrames and numpy arrays
        if isinstance(X_low, pd.DataFrame):
            # Extract X, Y coordinates from DataFrame
            X_low_array = X_low[['X', 'Y']].values
        else:
            X_low_array = X_low

        # Use sampled data
        X_low_sample = X_low_array[sample_idx] if X_low_array.shape[0] == X_high.shape[0] else X_low_array

        metrics = compute_projection_quality_metrics(X_high_sample, X_low_sample, method_name)
        results.append(metrics)

    return pd.DataFrame(results)


# ============================================================================
# CLUSTERING METRICS (wrappers for sklearn)
# ============================================================================

def compute_clustering_metrics(data, labels):
    """
    Compute standard clustering evaluation metrics.

    Args:
        data: Feature data (n_samples, n_features)
        labels: Cluster labels (n_samples,)

    Returns:
        Dictionary with silhouette, davies_bouldin, calinski_harabasz scores
    """
    # Filter out noise points (label -1 in DBSCAN)
    mask = labels != -1
    if mask.sum() < 2:
        return {
            'silhouette': np.nan,
            'davies_bouldin': np.nan,
            'calinski_harabasz': np.nan
        }

    silhouette = silhouette_score(data[mask], labels[mask])
    davies_bouldin = davies_bouldin_score(data[mask], labels[mask])
    calinski_harabasz = calinski_harabasz_score(data[mask], labels[mask])

    return {
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin,
        'calinski_harabasz': calinski_harabasz
    }


# ============================================================================
# COMPOSITE SCORING
# ============================================================================

def compute_kmeans_composite_score(df_analysis):
    """
    Compute composite quality score for k-Means results.

    Combines multiple metrics with configured weights to create a single
    comparison score, with preference for interpretable k values.

    Args:
        df_analysis: DataFrame with k-Means results including metrics

    Returns:
        numpy array of composite scores (same length as df_analysis)
    """
    # Normalize metrics to 0-1 range
    normalized_silhouette = (df_analysis['Silhouette'] - df_analysis['Silhouette'].min()) / \
                            (df_analysis['Silhouette'].max() - df_analysis['Silhouette'].min())

    normalized_davies = 1 - (df_analysis['Davies-Bouldin'] - df_analysis['Davies-Bouldin'].min()) / \
                            (df_analysis['Davies-Bouldin'].max() - df_analysis['Davies-Bouldin'].min())

    # k-range preference
    k_penalty = np.where(
        (df_analysis['k'] >= config.KMEANS_INTERPRETABLE_K_MIN) &
        (df_analysis['k'] <= config.KMEANS_INTERPRETABLE_K_MAX),
        1.0, 0.7
    )

    # Combine with configured weights
    composite_score = (
        config.KMEANS_SCORE_WEIGHTS['silhouette'] * normalized_silhouette +
        config.KMEANS_SCORE_WEIGHTS['davies_bouldin'] * normalized_davies +
        config.KMEANS_SCORE_WEIGHTS['cluster_balance'] * df_analysis['cluster_balance'] +
        config.KMEANS_SCORE_WEIGHTS['k_range_penalty'] * k_penalty
    )

    return composite_score


def compute_dbscan_composite_score(df_analysis):
    """
    Compute composite quality score for DBSCAN results.

    Args:
        df_analysis: DataFrame with DBSCAN results including metrics

    Returns:
        numpy array of composite scores
    """
    # Normalize metrics
    normalized_silhouette = (df_analysis['silhouette'] - df_analysis['silhouette'].min()) / \
                            (df_analysis['silhouette'].max() - df_analysis['silhouette'].min())

    normalized_davies = 1 - (df_analysis['davies_bouldin'] - df_analysis['davies_bouldin'].min()) / \
                            (df_analysis['davies_bouldin'].max() - df_analysis['davies_bouldin'].min())

    normalized_noise = 1 - df_analysis['noise_pct'] / 100

    # Cluster count preference
    cluster_penalty = np.where(
        (df_analysis['n_clusters'] >= config.DBSCAN_INTERPRETABLE_K_MIN) &
        (df_analysis['n_clusters'] <= config.DBSCAN_INTERPRETABLE_K_MAX),
        1.0, 0.5
    )

    # Combine with configured weights
    composite_score = (
        config.DBSCAN_SCORE_WEIGHTS['silhouette'] * normalized_silhouette +
        config.DBSCAN_SCORE_WEIGHTS['davies_bouldin'] * normalized_davies +
        config.DBSCAN_SCORE_WEIGHTS['noise_percentage'] * normalized_noise +
        config.DBSCAN_SCORE_WEIGHTS['cluster_count_penalty'] * cluster_penalty
    )

    return composite_score
