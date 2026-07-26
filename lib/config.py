"""
Centralized configuration for the XAI Project Space analysis.

This module contains all settings, parameters, and paths used throughout the analysis.
This ensures consistency and avoids mismatches in sample sizes and other parameters.
"""

import os
from pathlib import Path

# ============================================================================
# Data Paths
# ============================================================================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
SPOTIFY_DIR = DATA_DIR / "spotify"

FILE_PATHS = {
    "artists": str(SPOTIFY_DIR / "artists.csv"),
    "tracks": str(SPOTIFY_DIR / "tracks.csv"),
    "top20": str(SPOTIFY_DIR / "dict_artists.json"),
}

# ============================================================================
# Sample Sizes - Centralized to avoid mismatches
# ============================================================================
# Main dataset sampling
SAMPLE_SIZE = 30000  # Larger sample sizes only recommended if you have 48-64GB RAM
USE_FULL_DATASET = False  # Set to True to use full dataset

# Metrics computation sampling
METRICS_SAMPLE_SIZE = 5000  # Sample size for all metrics (can be increased for better accuracy)
USE_FULL_FOR_METRICS = False  # Set to True to use full dataset for metrics

# Clustering sampling
CLUSTERING_SAMPLE_SIZE = 10000  # Adjust this for speed vs accuracy trade-off
USE_FULL_FOR_CLUSTERING = False  # Set to True to use full dataset for clustering

# Random seed for reproducibility
RANDOM_STATE = 42

# ============================================================================
# Feature Configuration
# ============================================================================
# Acoustic features for clustering
ACOUSTIC_FEATURES = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence',
    'tempo', 'duration_ms'
]

# Continuous features for outlier detection
CONTINUOUS_FEATURES = [
    'energy', 'danceability', 'acousticness', 'valence',
    'loudness', 'instrumentalness', 'liveness', 'speechiness', 'tempo'
]

# Categorical features for analysis
CATEGORICAL_FEATURES = ['era', 'mood_quadrant', 'danceability_cat']

# ============================================================================
# Projection Parameters
# ============================================================================
# t-SNE parameters
TSNE_PARAMS = {
    'n_components': 2,
    'perplexity': 30,
    'random_state': RANDOM_STATE,
    'n_iter': 1000,
}

# UMAP parameters
UMAP_PARAMS = {
    'n_components': 2,
    'n_neighbors': 15,
    'min_dist': 0.1,
    'random_state': RANDOM_STATE,
}

# PCA parameters
PCA_PARAMS = {
    'n_components': 2,
    'random_state': RANDOM_STATE,
}

# ============================================================================
# Clustering Parameters
# ============================================================================
# DBSCAN parameters
DBSCAN_EPS_RANGE = [0.1, 0.2, 0.3, 0.4, 0.5]
DBSCAN_MIN_SAMPLES_RANGE = [5, 10, 15, 20]

# ============================================================================
# Outlier Detection Parameters
# ============================================================================
OUTLIER_CONTAMINATION = 0.05  # 5% contamination for LOF and Isolation Forest
OUTLIER_TOP_PERCENT = 0.05  # Top 5% for distance-based outlier detection

# ============================================================================
# Output Directories
# ============================================================================
OUTPUT_DIR = BASE_DIR / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"
LOGS_DIR = OUTPUT_DIR / "logs"

# Create output directories if they don't exist
OUTPUT_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ============================================================================
# Logging Configuration
# ============================================================================
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_TO_FILE = True
LOG_TO_CONSOLE = False

# ============================================================================
# Plotting Configuration
# ============================================================================
SAVE_PLOTS = True  # Whether to save plots to files
SHOW_PLOTS = True  # Whether to display plots (set False for headless execution)
PLOT_FORMAT = "png"  # Format for saved plots: png, pdf, svg
PLOT_DPI = 300  # Resolution for saved plots

# ============================================================================
# Analysis Flags
# ============================================================================
DEMONSTRATE_DATASET_PROCESSING = True  # Disable to skip dataset processing demonstration


# Audio features for projection
AUDIO_FEATURES = [
    'acousticness', 'danceability', 'energy', 'instrumentalness',
    'liveness', 'loudness', 'speechiness', 'valence', 'tempo',
    'duration_ms', 'key', 'mode', 'time_signature'
]

# Metadata features to preserve in output
META_DATA_FEATURES = [
    'id', 'name', 'num_artists', 'release_days', 'popularity',
    'explicit', 'name_length', 'artist_popularity_mean', 'artist_related_pop_mean'
]

# ============================================================================
# PROJECTION CONFIGURATION
# ============================================================================

# Projection output settings
PROJECTION_N_COMPONENTS = 2  # Always 2D for visualization

# t-SNE hyperparameters to test
TSNE_PERPLEXITIES = [5, 30, 50]
TSNE_DEFAULT_PERPLEXITY = 30

# UMAP hyperparameters to test
UMAP_N_NEIGHBORS_VALUES = [15, 30, 50]
UMAP_DEFAULT_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1

# ============================================================================
# QUALITY METRICS CONFIGURATION
# ============================================================================

# Quality metric computation settings
QUALITY_METRICS_SAMPLE_SIZE = 10000  # Sample size for metric computation
QUALITY_METRICS_USE_FULL_DATASET = False  # Set to True for full dataset (slow)
KNN_NEIGHBORS = 10  # k for kNN-based metrics

# ============================================================================
# CLUSTERING CONFIGURATION
# ============================================================================

# k-Means configuration
KMEANS_K_VALUES = [3, 5, 8, 10, 15, 20, 25, 30]
KMEANS_N_INIT = 7  # Number of initializations (balance speed vs accuracy)
KMEANS_MAX_ITER = 400

# k-Means composite score weights
KMEANS_SCORE_WEIGHTS = {
    'silhouette': 0.3,
    'davies_bouldin': 0.2,
    'calinski_harabasz': 0.2,
    'cluster_balance': 0.2,
    'k_range_penalty': 0.1
}

# k-Means interpretability thresholds
KMEANS_INTERPRETABLE_K_MIN = 5
KMEANS_INTERPRETABLE_K_MAX = 15
KMEANS_MIN_CLUSTER_SIZE = 1000

# DBSCAN configuration
DBSCAN_SAMPLE_SIZE = 10000  # Sample size for DBSCAN testing
DBSCAN_USE_FULL_DATASET = False  # Set to True for full dataset (much slower)
DBSCAN_K_NEIGHBORS = 15  # k for k-distance graph
DBSCAN_EPS_PERCENTILES = [98, 99, 99.5]  # Percentiles for eps estimation
DBSCAN_MIN_SAMPLES_VALUES = [2, 5, 10, 15]

# DBSCAN composite score weights
DBSCAN_SCORE_WEIGHTS = {
    'silhouette': 0.3,
    'davies_bouldin': 0.2,
    'noise_percentage': 0.3,
    'cluster_count_penalty': 0.2
}

# DBSCAN interpretability thresholds
DBSCAN_INTERPRETABLE_K_MIN = 5
DBSCAN_INTERPRETABLE_K_MAX = 20
DBSCAN_MAX_NOISE_PERCENTAGE = 30
DBSCAN_MIN_CLUSTER_SIZE = 500

# ============================================================================
# CLUSTERING SAMPLE SETTINGS
# ============================================================================

# Global clustering sample size (for both k-Means and DBSCAN visualization)
CLUSTERING_SAMPLE_SIZE = 10000
CLUSTERING_USE_FULL_DATASET = False

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

# Scatter plot settings
SCATTER_ALPHA = 0.3
SCATTER_SIZE = 1  # Point size for full dataset scatter plots
SCATTER_CMAP = 'tab10'

# Figure sizes
PROJECTION_COMPARISON_SIZE = (20, 12)
QUALITY_METRICS_SIZE = (18, 12)
KMEANS_COMPARISON_SIZE = (16, 14)
KMEANS_METRICS_SIZE = (18, 5)
DBSCAN_K_DISTANCE_SIZE = (10, 6)
DBSCAN_COMPARISON_SIZE = (16, 14)

# Metric visualization settings
METRIC_COLORS = {
    'Silhouette': 'steelblue',
    'Davies-Bouldin': 'coral',
    'Calinski-Harabasz': 'mediumseagreen',
    'Trustworthiness': 'steelblue',
    'Continuity': 'coral',
    'kNN Overlap': 'mediumseagreen',
    'Distance Correlation': 'darkviolet',
    'kNN Accuracy': 'gold'
}

# ============================================================================
# RANDOM SEEDS (for reproducibility)
# ============================================================================

RANDOM_SEED = 42
