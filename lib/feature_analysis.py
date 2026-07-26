"""
Feature importance analysis utilities for cluster characterization.
Computes statistical measures of feature differentiation between clusters and overall distribution.
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency


def prepare_analysis_dataset(
    df_tracks,
    df_tracks_sample,
    cluster_labels,
    df_artists,
    add_categorical_fn,
):
    """
    Prepare analysis dataset with cluster labels mapped from sample to full dataset.

    Args:
        df_tracks: Full tracks dataset
        df_tracks_sample: Sampled tracks (used for clustering)
        cluster_labels: Cluster assignments for the sample
        df_artists: Artists dataset
        add_categorical_fn: Function to add categorical features

    Returns:
        tuple: (df_clustered, top_clusters, cluster_sizes)
    """
    # Use full dataset
    df_analysis = df_tracks.copy()

    # Map cluster labels from sample to full dataset using sample indices
    sample_indices = df_tracks_sample.index.values
    full_labels = np.full(len(df_analysis), -1, dtype=int)
    full_labels[sample_indices] = cluster_labels
    df_analysis['cluster'] = full_labels

    # Add categorical features
    df_analysis = add_categorical_fn(df_analysis, df_artists)

    # Filter out noise points (-1) for analysis
    df_clustered = df_analysis[df_analysis['cluster'] != -1].copy()

    # Get top clusters by size
    top_clusters = df_clustered['cluster'].value_counts().head(10).index.tolist()
    cluster_sizes = df_clustered['cluster'].value_counts()

    return df_clustered, top_clusters, cluster_sizes


def compute_cluster_feature_importance(
    cluster_id, df_cluster, df_all, audio_features, categorical_features
):
    """
    Compute feature importance scores for a single cluster using statistical tests.

    Uses:
    - Kolmogorov-Smirnov test for continuous features
    - Chi-square test for categorical features

    Args:
        cluster_id: Cluster identifier
        df_cluster: DataFrame with cluster data
        df_all: Full dataset for comparison
        audio_features: List of continuous feature column names
        categorical_features: List of categorical feature column names

    Returns:
        dict: Feature importance scores with test statistics
    """
    importance = {}

    # Continuous features: Kolmogorov-Smirnov test
    for feature in audio_features:
        if feature not in df_cluster.columns or feature not in df_all.columns:
            continue

        cluster_vals = df_cluster[feature].dropna().values
        all_vals = df_all[feature].dropna().values

        if len(cluster_vals) == 0 or len(all_vals) == 0:
            continue

        ks_stat, p_value = ks_2samp(cluster_vals, all_vals)

        importance[feature] = {
            'statistic': ks_stat,
            'p_value': p_value,
            'type': 'continuous',
            'cluster_mean': cluster_vals.mean(),
            'overall_mean': all_vals.mean(),
        }

    # Categorical features: Chi-square test
    for feature in categorical_features:
        if feature not in df_cluster.columns or feature not in df_all.columns:
            continue

        # Build contingency table
        cluster_counts = df_cluster[feature].value_counts()
        all_counts = df_all[feature].value_counts()

        categories = set(cluster_counts.index) | set(all_counts.index)

        contingency_table = pd.DataFrame({
            'cluster': [cluster_counts.get(cat, 0) for cat in categories],
            'other': [all_counts.get(cat, 0) - cluster_counts.get(cat, 0) for cat in categories],
        }, index=categories).T

        chi2, p_value, dof, expected = chi2_contingency(contingency_table)

        # Normalize by degrees of freedom for comparability
        normalized_chi2 = chi2 / (dof + 1) if dof > 0 else 0

        importance[feature] = {
            'statistic': normalized_chi2,
            'p_value': p_value,
            'type': 'categorical',
        }

    return importance


def compute_all_cluster_importance(
    top_clusters, df_clustered, df_analysis, audio_features, categorical_features
):
    """
    Compute feature importance for all top clusters.

    Args:
        top_clusters: List of cluster IDs to analyze
        df_clustered: DataFrame with cluster assignments
        df_analysis: Full dataset for comparison
        audio_features: List of continuous feature names
        categorical_features: List of categorical feature names

    Returns:
        dict: Nested dict mapping cluster_id → feature_name → importance scores
    """
    cluster_importance = {}

    for cluster_id in top_clusters:
        df_cluster = df_clustered[df_clustered['cluster'] == cluster_id]
        cluster_importance[cluster_id] = compute_cluster_feature_importance(
            cluster_id, df_cluster, df_analysis, audio_features, categorical_features
        )

    return cluster_importance
