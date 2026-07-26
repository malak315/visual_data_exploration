from scipy.stats import ks_2samp, chi2_contingency
from scipy import stats
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


# Function to compute feature importance for a cluster
def compute_feature_importance(cluster_id, df_cluster, df_all, audio_features, categorical_features):
    """Compute statistical importance of features for a cluster."""
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
                # KS test statistic (higher = more different)
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
            # Create contingency table
            cluster_counts = cluster_data[feature].value_counts()
            all_counts = df_all[feature].value_counts()
            
            # Get all categories
            all_cats = set(cluster_counts.index) | set(all_counts.index)
            
            if len(all_cats) > 1:
                # Build contingency table
                cluster_total = len(cluster_data)
                all_total = len(df_all)
                
                contingency = []
                for cat in all_cats:
                    cluster_val = cluster_counts.get(cat, 0)
                    all_val = all_counts.get(cat, 0)
                    contingency.append([cluster_val, all_val])
                
                contingency = np.array(contingency)
                
                # Chi-square test
                if contingency.sum() > 0 and contingency.shape[0] > 1:
                    try:
                        chi2, p_value, dof, expected = chi2_contingency(contingency)
                        # Normalize chi2 by degrees of freedom for comparison
                        normalized_chi2 = chi2 / dof if dof > 0 else 0
                        importance_scores[feature] = {
                            'statistic': normalized_chi2,
                            'p_value': p_value,
                            'type': 'categorical',
                            'cluster_dist': cluster_counts.to_dict(),
                            'overall_dist': all_counts.to_dict()
                        }
                    except:
                        pass
    
    return importance_scores

def demonstrate_discriminative_features_per_cluster(df_clustered, df_analysis, cluster_sizes):
    """Compute and visualize discriminative features for each cluster.

    Returns:
        cluster_importance: dict mapping cluster_id to feature importance scores
        audio_features: list of continuous features analyzed
        categorical_features: list of categorical features analyzed
    """
    # Define audio features (continuous) and categorical features
    audio_features = [
        'acousticness', 'danceability', 'energy', 'instrumentalness',
        'liveness', 'loudness', 'speechiness', 'valence', 'tempo',
        'duration_ms', 'key', 'mode', 'time_signature'
    ]

    categorical_features = [
        'era', 'mood_quadrant', 'genre_binned', 'acousticness_cat',
        'danceability_cat', 'popularity_tier', 'decade_str'
    ]

    # Compute feature importance for all clusters
    cluster_importance = {}
    for cluster_id in tqdm(cluster_sizes.index, desc="Analyzing clusters"):
        importance = compute_feature_importance(cluster_id, df_clustered, df_analysis,
                                            audio_features, categorical_features)
        if importance:
            cluster_importance[cluster_id] = importance

    # Visualize top discriminative features for each cluster
    n_top_features = 5
    n_cols = 5
    n_rows = len(cluster_sizes) // n_cols + (len(cluster_sizes) % n_cols > 0)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
    axes = axes.flatten()

    for idx, cluster_id in enumerate(cluster_sizes.index):
        ax = axes[idx]

        if cluster_id not in cluster_importance:
            ax.axis('off')
            continue

        importance = cluster_importance[cluster_id]
        sorted_features = sorted(importance.items(),
                                key=lambda x: x[1]['statistic'],
                                reverse=True)[:n_top_features]

        feature_names = [f[0] for f in sorted_features]
        statistics = [f[1]['statistic'] for f in sorted_features]

        bars = ax.barh(range(len(feature_names)), statistics,
                      color=plt.cm.viridis(np.linspace(0, 1, len(feature_names))))
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names, fontsize=9)
        ax.set_xlabel('Importance', fontsize=9)
        ax.set_title(f'Cluster {cluster_id} (n={cluster_sizes[cluster_id]:,})',
                    fontsize=10, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        for i, (bar, stat) in enumerate(zip(bars, statistics)):
            ax.text(stat + 0.01*max(statistics), i, f'{stat:.3f}',
                va='center', fontsize=8)

    # Hide unused subplots
    for idx in range(len(cluster_sizes), len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Top Discriminative Features per Cluster (KS for continuous, Chi²/dof for categorical)',
                fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return cluster_importance, audio_features, categorical_features


def visualize_continuous_features(df_clustered, df_analysis, cluster_sizes, cluster_importance, top_n=5):
    """Visualize continuous feature distributions for top N clusters."""
    key_features = ['energy', 'valence', 'danceability', 'acousticness',
                    'tempo', 'loudness', 'speechiness', 'instrumentalness']

    top_clusters = cluster_sizes.head(top_n).index

    for cluster_id in top_clusters:
        if cluster_id not in cluster_importance:
            continue

        cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]

        n_cols = 4
        n_rows = (len(key_features) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        axes = axes.flatten()

        for idx, feature in enumerate(key_features):
            ax = axes[idx]

            cluster_values = cluster_data[feature].dropna()
            all_values = df_analysis[feature].dropna()

            data_to_plot = [all_values, cluster_values]
            parts = ax.violinplot(data_to_plot, positions=[0, 1], showmeans=True, showmedians=True)

            for pc, color in zip(parts['bodies'], ['lightblue', 'orange']):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)

            bp = ax.boxplot(data_to_plot, positions=[0, 1], widths=0.3,
                           patch_artist=True, showfliers=False)
            for patch, color in zip(bp['boxes'], ['lightblue', 'orange']):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)

            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Overall', f'C{cluster_id}'], fontsize=10)
            ax.set_ylabel(feature, fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

            diff_pct = 100 * (cluster_values.mean() - all_values.mean()) / all_values.mean()
            ax.text(0.5, 0.95, f'Δ={diff_pct:+.1f}%',
                   transform=ax.transAxes, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=9)

        for idx in range(len(key_features), len(axes)):
            axes[idx].axis('off')

        plt.suptitle(f'Cluster {cluster_id} Feature Distributions (n={len(cluster_data):,})',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


def visualize_categorical_features(df_clustered, df_analysis, cluster_sizes, top_n=5):
    """Visualize categorical feature distributions for top N clusters."""
    cat_features = ['era', 'mood_quadrant', 'genre_binned',
                    'acousticness_cat', 'danceability_cat', 'popularity_tier']
    available_cat_features = [f for f in cat_features if f in df_clustered.columns]

    if len(available_cat_features) == 0:
        print("No categorical features found in dataframe. Skipping categorical visualization.")
        return

    top_clusters = cluster_sizes.head(top_n).index

    for cluster_id in top_clusters:
        cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]

        n_cols = 3
        n_rows = (len(available_cat_features) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]

        for idx, cat_feat in enumerate(available_cat_features):
            ax = axes[idx]

            overall_dist = df_analysis[cat_feat].dropna().value_counts(normalize=True) * 100
            cluster_dist = cluster_data[cat_feat].dropna().value_counts(normalize=True) * 100

            all_cats = sorted(set(overall_dist.index) | set(cluster_dist.index))

            if len(all_cats) > 12:
                top_cats = overall_dist.head(12).index.tolist()
                all_cats = [c for c in all_cats if c in top_cats]

            x = np.arange(len(all_cats))
            width = 0.35

            overall_values = [overall_dist.get(cat, 0) for cat in all_cats]
            cluster_values = [cluster_dist.get(cat, 0) for cat in all_cats]

            ax.bar(x - width/2, overall_values, width, label='Overall',
                   color='lightblue', alpha=0.8, edgecolor='black', linewidth=0.5)
            ax.bar(x + width/2, cluster_values, width, label=f'Cluster {cluster_id}',
                   color='orange', alpha=0.8, edgecolor='black', linewidth=0.5)

            ax.set_xticks(x)
            ax.set_xticklabels(all_cats, rotation=45, ha='right', fontsize=9)
            ax.set_ylabel('Percentage (%)', fontsize=10)
            ax.set_title(cat_feat.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(axis='y', alpha=0.3)

        for idx in range(len(available_cat_features), len(axes)):
            axes[idx].axis('off')

        plt.suptitle(f'Cluster {cluster_id} Categorical Features (n={len(cluster_data):,})',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


def create_cluster_comparison_table(df_clustered, cluster_sizes, top_n=10):
    """Create comparison table for top N clusters."""
    comparison_features = ['energy', 'valence', 'danceability', 'acousticness',
                           'tempo', 'loudness', 'speechiness', 'instrumentalness']

    top_clusters = cluster_sizes.head(top_n).index

    comparison_data = []
    for cluster_id in top_clusters:
        cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]

        row = {'Cluster': cluster_id, 'Size': len(cluster_data)}
        for feat in comparison_features:
            row[feat] = cluster_data[feat].mean()

        if 'genre_binned' in cluster_data.columns:
            row['Top Genre'] = cluster_data['genre_binned'].mode()[0] if len(cluster_data['genre_binned'].mode()) > 0 else 'Unknown'
        if 'era' in cluster_data.columns:
            row['Top Era'] = cluster_data['era'].mode()[0] if len(cluster_data['era'].mode()) > 0 else 'Unknown'

        comparison_data.append(row)

    df_comparison = pd.DataFrame(comparison_data)
    return df_comparison, comparison_features


def visualize_cluster_comparison(df_clustered, df_analysis, cluster_sizes, comparison_features, top_n=10):
    """Create side-by-side violin plots comparing top N clusters."""
    top_clusters = cluster_sizes.head(top_n).index

    n_cols = 4
    n_rows = (len(comparison_features) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten()

    for idx, feature in enumerate(comparison_features):
        ax = axes[idx]

        data_to_plot = []
        labels = []
        for cluster_id in top_clusters:
            cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
            values = cluster_data[feature].dropna()
            if len(values) > 0:
                data_to_plot.append(values)
                labels.append(f'C{cluster_id}')

        parts = ax.violinplot(data_to_plot, positions=range(len(data_to_plot)),
                             showmeans=True, showmedians=True)

        colors = plt.cm.Set3(np.linspace(0, 1, len(data_to_plot)))
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

        bp = ax.boxplot(data_to_plot, positions=range(len(data_to_plot)),
                       widths=0.3, patch_artist=True, showfliers=False)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9, rotation=45)
        ax.set_ylabel(feature, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        overall_mean = df_analysis[feature].mean()
        ax.axhline(y=overall_mean, color='red', linestyle='--', linewidth=2,
                  alpha=0.7, label='Overall Mean')
        ax.legend(fontsize=8)

    for idx in range(len(comparison_features), len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'Feature Distribution Comparison: Top {top_n} Clusters',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def create_radar_charts(df_clustered, df_analysis, cluster_sizes, top_n=5):
    """Create radar charts for top N clusters."""
    radar_features = ['energy', 'valence', 'danceability', 'acousticness',
                     'speechiness', 'instrumentalness']

    # Normalize features to 0-1 scale
    normalized_data = {}
    for feature in radar_features:
        overall_min = df_analysis[feature].min()
        overall_max = df_analysis[feature].max()
        normalized_data[feature] = {
            'min': overall_min,
            'range': overall_max - overall_min if overall_max != overall_min else 1
        }

    top_clusters = cluster_sizes.head(top_n).index.tolist()
    N = len(radar_features)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    n_cols = 5
    n_rows = (top_n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), subplot_kw=dict(projection='polar'))
    axes = axes.flatten()
    if top_n == 1:
        axes = [axes]

    for idx, cluster_id in enumerate(top_clusters):
        ax = axes[idx]
        cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]

        values = []
        for feature in radar_features:
            cluster_mean = cluster_data[feature].mean()
            norm_info = normalized_data[feature]
            normalized_value = (cluster_mean - norm_info['min']) / norm_info['range']
            values.append(normalized_value)

        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_features, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(f'Cluster {cluster_id} (n={len(cluster_data):,})',
                    fontsize=11, fontweight='bold', pad=20)
        ax.grid(True)

    plt.suptitle(f'Cluster Profiles: Top {top_n} Clusters by Size',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def create_similarity_matrix(df_clustered, cluster_sizes, comparison_features, top_n=10):
    """Create cluster similarity matrix for top N clusters."""
    top_clusters = cluster_sizes.head(top_n).index

    # Get feature means for each cluster
    cluster_means = {}
    for cluster_id in top_clusters:
        cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
        cluster_means[cluster_id] = {feat: cluster_data[feat].mean() for feat in comparison_features}

    df_cluster_means = pd.DataFrame(cluster_means).T

    # Normalize and compute cosine similarity
    scaler = StandardScaler()
    cluster_means_normalized = scaler.fit_transform(df_cluster_means)
    similarity_matrix = cosine_similarity(cluster_means_normalized)

    df_similarity = pd.DataFrame(similarity_matrix,
                                index=top_clusters,
                                columns=top_clusters)

    # Visualize
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_similarity, annot=True, fmt='.2f', cmap='coolwarm',
               center=0, vmin=-1, vmax=1, square=True,
               cbar_kws={'label': 'Cosine Similarity'}, linewidths=0.5)
    plt.title(f'Cluster Similarity Matrix: Top {top_n} Clusters (Based on Normalized Feature Means)',
             fontsize=13, fontweight='bold')
    plt.xlabel('Cluster ID', fontsize=11, fontweight='bold')
    plt.ylabel('Cluster ID', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.show()
