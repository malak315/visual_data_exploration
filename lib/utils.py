"""
Data processing and utility functions for the XAI Project Space analysis.

This module contains reusable functions for data processing, feature engineering,
and analysis that can be used across the notebook and other scripts.
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
from tqdm import tqdm
from typing import Optional, List, Dict, Any, Tuple
from scipy.stats import ks_2samp, chi2_contingency

import lib.logger_utils as logger_utils
from lib.config import FILE_PATHS, RANDOM_STATE

logger = logger_utils.get_logger("utils")


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_datasets(
    artists_path: Optional[str] = None,
    tracks_path: Optional[str] = None,
    top20_path: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the Spotify datasets.
    
    Args:
        artists_path: Path to artists CSV file (defaults to config)
        tracks_path: Path to tracks CSV file (defaults to config)
        top20_path: Path to top20 JSON file (defaults to config)
        
    Returns:
        Tuple of (df_artists, df_tracks)
    """
    artists_path = artists_path or FILE_PATHS["artists"]
    tracks_path = tracks_path or FILE_PATHS["tracks"]
    
    logger.info(f"Loading datasets from {artists_path} and {tracks_path}")
    
    df_artists = pd.read_csv(artists_path)
    df_tracks = pd.read_csv(tracks_path)
    
    logger.info(f"Loaded {len(df_artists):,} artists and {len(df_tracks):,} tracks")
    
    return df_artists, df_tracks


def sample_dataframe(
    df: pd.DataFrame,
    sample_size: int,
    random_state: int = RANDOM_STATE,
    use_full: bool = False
) -> pd.DataFrame:
    """
    Sample a dataframe with consistent parameters.
    
    Args:
        df: DataFrame to sample
        sample_size: Size of sample
        random_state: Random seed for reproducibility
        use_full: If True, return full dataframe regardless of sample_size
        
    Returns:
        Sampled DataFrame
    """
    if use_full or len(df) <= sample_size:
        logger.info(f"Using full dataset: {len(df):,} rows")
        return df.copy()
    
    sampled = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    logger.info(f"Sampled {len(sampled):,} rows from {len(df):,} total rows")
    return sampled


# ============================================================================
# Statistical Analysis Functions
# ============================================================================

def compute_feature_importance(
    cluster_id: int,
    df_cluster: pd.DataFrame,
    df_all: pd.DataFrame,
    audio_features: List[str],
    categorical_features: List[str]
) -> Optional[Dict[str, Any]]:
    """
    Compute statistical importance of features for a cluster.
    
    Uses Kolmogorov-Smirnov test for continuous features and
    Chi-square test for categorical features.
    """
    cluster_data = df_cluster[df_cluster['cluster'] == cluster_id]
    
    if len(cluster_data) < 10:  # Too small for meaningful statistics
        return None
    
    importance_scores = {}
    
    # Continuous features: Kolmogorov-Smirnov test
    for feature in audio_features:
        if feature in df_cluster.columns:
            cluster_values = cluster_data[feature].dropna()
            all_values = df_all[feature].dropna()
            
            if len(cluster_values) > 1 and len(all_values) > 1:
                ks_stat, p_value = ks_2samp(cluster_values, all_values)
                importance_scores[feature] = {
                    'statistic': ks_stat,
                    'p_value': p_value,
                    'type': 'continuous',
                    'cluster_mean': cluster_values.mean(),
                    'overall_mean': all_values.mean(),
                    'difference': cluster_values.mean() - all_values.mean()
                }
    
    # Categorical features: Chi-square test
    for feature in categorical_features:
        if feature in df_cluster.columns:
            cluster_counts = cluster_data[feature].value_counts()
            all_counts = df_all[feature].value_counts()
            
            all_cats = set(cluster_counts.index) | set(all_counts.index)
            
            if len(all_cats) > 1:
                cluster_total = len(cluster_data)
                all_total = len(df_all)
                
                contingency = []
                for cat in all_cats:
                    cluster_val = cluster_counts.get(cat, 0)
                    all_val = all_counts.get(cat, 0)
                    contingency.append([cluster_val, all_val])
                
                contingency = np.array(contingency)
                
                if contingency.sum() > 0 and contingency.shape[0] > 1:
                    try:
                        chi2, p_value, dof, expected = chi2_contingency(contingency)
                        normalized_chi2 = chi2 / dof if dof > 0 else 0
                        importance_scores[feature] = {
                            'statistic': normalized_chi2,
                            'p_value': p_value,
                            'type': 'categorical',
                            'cluster_dist': cluster_counts.to_dict(),
                            'overall_dist': all_counts.to_dict()
                        }
                    except Exception as e:
                        logger.warning(f"Error computing chi-square for {feature}: {e}")
    
    return importance_scores

