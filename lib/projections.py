"""
Unified projection methods for dimensionality reduction.

Provides a consistent interface for various projection methods (PCA, t-SNE, UMAP, ICA)
and utilities for hyperparameter testing.
"""

import time
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.decomposition import FastICA
from umap import UMAP
from tqdm import tqdm

from lib import config
from lib import logger_utils

logger = logger_utils.get_logger("projections")

def normalize_features(data):
    """
    Normalize features using StandardScaler.

    Args:
        data: numpy array of shape (n_samples, n_features)

    Returns:
        Normalized data as numpy array
    """
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    return scaler.fit_transform(data)


def _create_output_dataframe(coords, metadata, feature_names=None):
    """
    Create standardized output DataFrame from projection coordinates.

    Args:
        coords: Projection coordinates (n_samples, 2)
        metadata: DataFrame with metadata
        feature_names: Optional list of feature names for output columns

    Returns:
        DataFrame with metadata + projection coordinates
    """
    if feature_names is None:
        feature_names = ['X', 'Y']

    df = pd.DataFrame(coords, columns=feature_names)
    return pd.concat([metadata.reset_index(drop=True), df], axis=1)


# ============================================================================
# PCA Projection
# ============================================================================

def project_pca(proj_data, metadata, n_components=2, timer=False):
    """
    Project data using PCA (fast baseline method).

    Args:
        proj_data: Standardized projection data (n_samples, n_features)
        metadata: DataFrame with metadata to preserve
        n_components: Number of output dimensions
        timer: Print execution time if True

    Returns:
        Tuple of (DataFrame with results, PCA model)
    """
    if timer:
        start_time = time.time()

    pca = PCA(n_components=n_components)
    pca_coords = pca.fit_transform(proj_data)

    if timer:
        elapsed = time.time() - start_time
        logger.info(f"PCA completed in {elapsed:.2f} seconds")

    logger.info(f"Explained variance: {pca.explained_variance_ratio_}")
    logger.info(f"Total explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    df = _create_output_dataframe(pca_coords, metadata)
    return df, pca


# ============================================================================
# t-SNE Projection
# ============================================================================

def project_tsne(proj_data, metadata, perplexity=30, n_components=2, timer=False):
    """
    Project data using t-SNE.

    Args:
        proj_data: Standardized projection data
        metadata: DataFrame with metadata
        perplexity: t-SNE perplexity parameter
        n_components: Number of output dimensions
        timer: Print execution time if True

    Returns:
        DataFrame with results
    """
    if timer:
        start_time = time.time()

    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate='auto',
        n_jobs=-1,
        random_state=config.RANDOM_SEED
    )
    tsne_coords = tsne.fit_transform(proj_data)

    if timer:
        elapsed = time.time() - start_time
        logger.info(f"t-SNE (perplexity={perplexity}) completed in {elapsed:.2f} seconds")

    return _create_output_dataframe(tsne_coords, metadata)


def test_tsne_perplexity(proj_data, metadata, perplexities=None, show_plots=True):
    """
    Test multiple t-SNE perplexity values and visualize results.

    Args:
        proj_data: Standardized projection data
        metadata: DataFrame with metadata
        perplexities: List of perplexity values to test
        show_plots: If True, create visualization

    Returns:
        Dictionary mapping perplexity -> DataFrame of results
    """
    if perplexities is None:
        perplexities = config.TSNE_PERPLEXITIES

    results = {}

    if show_plots:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(perplexities), figsize=(18, 6))
        if len(perplexities) == 1:
            axes = [axes]

    for i, perplexity in enumerate(perplexities):
        logger.info(f"\nRunning t-SNE with perplexity={perplexity}...")
        df = project_tsne(proj_data, metadata, perplexity=perplexity, timer=True)
        results[perplexity] = df

        if show_plots:
            axes[i].scatter(df['X'], df['Y'], alpha=0.3, s=1)
            axes[i].set_xlabel('t-SNE X')
            axes[i].set_ylabel('t-SNE Y')
            axes[i].set_title(f't-SNE (perplexity={perplexity})')
            axes[i].grid(True, alpha=0.3)

    if show_plots:
        plt.suptitle('t-SNE Perplexity Effects', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    return results


# ============================================================================
# UMAP Projection
# ============================================================================

def project_umap(proj_data, metadata, n_neighbors=15, min_dist=0.1, n_components=2, timer=False):
    """
    Project data using UMAP.

    Args:
        proj_data: Standardized projection data
        metadata: DataFrame with metadata
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        n_components: Number of output dimensions
        timer: Print execution time if True

    Returns:
        DataFrame with results
    """
    if timer:
        start_time = time.time()

    umap = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric='euclidean',
        random_state=config.RANDOM_SEED,
        n_jobs=1 # is overriden when random state to 1 anyways
    )
    umap_coords = umap.fit_transform(proj_data)

    if timer:
        elapsed = time.time() - start_time
        logger.info(f"UMAP (n_neighbors={n_neighbors}) completed in {elapsed:.2f} seconds")

    return _create_output_dataframe(umap_coords, metadata)


def test_umap_n_neighbors(proj_data, metadata, n_neighbors_values=None,
                          min_dist=None, show_plots=True):
    """
    Test multiple UMAP n_neighbors values and visualize results.

    Args:
        proj_data: Standardized projection data
        metadata: DataFrame with metadata
        n_neighbors_values: List of n_neighbors values to test
        min_dist: UMAP min_dist parameter
        show_plots: If True, create visualization

    Returns:
        Dictionary mapping n_neighbors -> DataFrame of results
    """
    if n_neighbors_values is None:
        n_neighbors_values = config.UMAP_N_NEIGHBORS_VALUES
    if min_dist is None:
        min_dist = config.UMAP_MIN_DIST

    results = {}

    if show_plots:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(n_neighbors_values), figsize=(18, 6))
        if len(n_neighbors_values) == 1:
            axes = [axes]

    for i, n_neighbors in enumerate(n_neighbors_values):
        logger.info(f"\nRunning UMAP with n_neighbors={n_neighbors}...")
        df = project_umap(proj_data, metadata, n_neighbors=n_neighbors,
                         min_dist=min_dist, timer=True)
        results[n_neighbors] = df

        if show_plots:
            axes[i].scatter(df['X'], df['Y'], alpha=0.3, s=1)
            axes[i].set_xlabel('UMAP X')
            axes[i].set_ylabel('UMAP Y')
            axes[i].set_title(f'UMAP (n_neighbors={n_neighbors})')
            axes[i].grid(True, alpha=0.3)

    if show_plots:
        plt.suptitle('UMAP n_neighbors Effects', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    return results


# ============================================================================
# ICA Projection
# ============================================================================

def project_ica(proj_data, metadata, n_components=2, timer=False):
    """
    Project data using Independent Component Analysis (ICA).

    Args:
        proj_data: Standardized projection data
        metadata: DataFrame with metadata
        n_components: Number of output dimensions
        timer: Print execution time if True

    Returns:
        DataFrame with results
    """
    if timer:
        start_time = time.time()

    ica = FastICA(n_components=n_components, random_state=config.RANDOM_SEED)
    ica_coords = ica.fit_transform(proj_data)

    if timer:
        elapsed = time.time() - start_time
        logger.info(f"ICA completed in {elapsed:.2f} seconds")

    return _create_output_dataframe(ica_coords, metadata)


# ============================================================================
# Batch Projection (compute multiple methods at once)
# ============================================================================

def run_all_projections(proj_data, metadata, methods=None):
    """
    Run multiple projection methods on the same data.

    Args:
        proj_data: Standardized projection data
        metadata: DataFrame with metadata
        methods: List of methods to run. Options: 'pca', 'tsne', 'umap', 'ica'
                 Default: all methods

    Returns:
        Dictionary mapping method names -> DataFrames of results
    """
    if methods is None:
        methods = ['pca', 'tsne', 'umap', 'ica']

    results = {}

    logger.info("Running projection methods...")
    for method in methods:
        if method.lower() == 'pca':
            logger.info("\n[PCA]")
            results['PCA'], _ = project_pca(proj_data, metadata, timer=True)

        elif method.lower() == 'tsne':
            logger.info("\n[t-SNE (perplexity=50)]")
            results['t-SNE (perp=50)'] = project_tsne(proj_data, metadata,
                                                       perplexity=50, timer=True)

        elif method.lower() == 'umap':
            logger.info("\n[UMAP (n_neighbors=50)]")
            results['UMAP (n=50)'] = project_umap(proj_data, metadata,
                                                   n_neighbors=50, timer=True)

        elif method.lower() == 'ica':
            logger.info("\n[ICA]")
            results['ICA'] = project_ica(proj_data, metadata, timer=True)

    return results
