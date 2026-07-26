import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def get_colormap(n_categories):
    """Get appropriate colormap based on number of categories."""
    if n_categories <= 10:
        return plt.cm.tab10
    elif n_categories <= 20:
        return plt.cm.tab20
    else:
        return plt.cm.viridis


def plot_projection_with_encoding(proj_coords, df, encoding_col, ax=None,
                                   title=None, show_legend=True):
    """Plot projection colored by a single categorical encoding.

    Args:
        proj_coords: (N, 2) array of 2D projection coordinates
        df: DataFrame containing encoding columns
        encoding_col: Name of column to use for coloring
        ax: Matplotlib axis (if None, creates new figure)
        title: Plot title
        show_legend: Whether to show legend
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Handle categorical data properly
    data = df[encoding_col]
    if pd.api.types.is_categorical_dtype(data):
        data = data.astype(str).replace('nan', 'Unknown')
    else:
        data = data.fillna('Unknown')

    unique_vals = sorted(data.unique())
    cmap = get_colormap(len(unique_vals))
    colors = {val: cmap(i/len(unique_vals)) for i, val in enumerate(unique_vals)}

    for val in unique_vals:
        mask = data == val
        ax.scatter(proj_coords[mask, 0], proj_coords[mask, 1],
                  c=[colors[val]], label=val, alpha=0.4, s=4, edgecolors='none')

    if show_legend and len(unique_vals) <= 15:
        ax.legend(fontsize=8, markerscale=2, loc='best')

    if title:
        ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('Dim 1', fontsize=9)
    ax.set_ylabel('Dim 2', fontsize=9)

    return ax


def plot_projection_grid(proj_coords, df, encodings, proj_name='Projection',
                         ncols=3, figsize=(18, 12), force_legend=False):
    """Create grid of projections with different categorical encodings.

    Args:
        proj_coords: (N, 2) array of 2D projection coordinates
        df: DataFrame containing encoding columns
        encodings: List of encoding column names to visualize
        proj_name: Name of projection method
        ncols: Number of columns in grid
        figsize: Figure size tuple
        force_legend: If True, always show legend regardless of category count
    """
    nrows = (len(encodings) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array([axes]) if nrows == 1 and ncols == 1 else axes.flatten()

    for idx, enc in enumerate(encodings):
        # Show legend if forced or if categories <= 8
        show_legend = force_legend or (df[enc].nunique() <= 8)
        plot_projection_with_encoding(proj_coords, df, enc, ax=axes[idx],
                                      title=enc,
                                      show_legend=show_legend)

    for idx in range(len(encodings), len(axes)):
        axes[idx].axis('off')

    fig.suptitle(f'{proj_name} - Categorical Encodings', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_continuous_feature(proj_coords, df, feature, proj_name='Projection',
                           vmin=None, vmax=None, figsize=(12, 9)):
    """Plot projection colored by continuous feature.

    Args:
        proj_coords: (N, 2) array of 2D projection coordinates
        df: DataFrame containing feature columns
        feature: Name of continuous feature to visualize
        proj_name: Name of projection method
        vmin, vmax: Value range for colormap
        figsize: Figure size tuple
    """
    fig, ax = plt.subplots(figsize=figsize)
    values = df[feature].values

    scatter = ax.scatter(proj_coords[:, 0], proj_coords[:, 1],
                        c=values, cmap='viridis', alpha=0.7, s=8,
                        vmin=vmin, vmax=vmax, edgecolors='none')

    plt.colorbar(scatter, ax=ax, label=feature)
    ax.set_title(f'{proj_name} - {feature}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Dim 1', fontsize=10)
    ax.set_ylabel('Dim 2', fontsize=10)
    plt.tight_layout()
    return fig


def plot_all_encoding_groups(projections_dict, df, encoding_groups,
                             show_each_projection=True):
    """Plot all encoding groups for all projection methods.

    Args:
        projections_dict: Dict of {proj_name: coords_array}
        df: DataFrame with encoding columns
        encoding_groups: Dict of {group_name: [encoding_cols]}
        show_each_projection: If True, show separate plots per projection method
    """
    # Define standard groups and their display parameters
    group_params = {
        'temporal': {'ncols': 2, 'figsize': (14, 12), 'force_legend': True},
        'mood': {'ncols': 2, 'figsize': (14, 8)},
        'musical': {'ncols': 3, 'figsize': (18, 12)},
        'genre': {'ncols': 1, 'figsize': (14, 10), 'special': True},
        'technical': {'ncols': 3, 'figsize': (18, 8)},
        'popularity': {'ncols': 2, 'figsize': (14, 8)},
        'combined': {'ncols': 3, 'figsize': (18, 8), 'single_proj': 't-SNE'},
        'key_insights': {'ncols': 3, 'figsize': (18, 12)}
    }

    for group_name, encodings in encoding_groups.items():
        if group_name not in group_params:
            continue

        params = group_params[group_name]

        # Special handling for genre (single large plot)
        if params.get('special'):
            for proj_name, coords in projections_dict.items():
                fig, ax = plt.subplots(figsize=params['figsize'])
                plot_projection_with_encoding(coords, df, encodings[0],
                                             ax=ax, show_legend=True)
                ax.set_title(f'{proj_name} - Genre Distribution',
                           fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.show()
        # Single projection only
        elif 'single_proj' in params:
            proj_name = params['single_proj']
            if proj_name in projections_dict:
                plot_projection_grid(projections_dict[proj_name], df, encodings,
                                   proj_name, ncols=params['ncols'],
                                   figsize=params['figsize'])
                plt.show()
        # All projections
        else:
            for proj_name, coords in projections_dict.items():
                plot_projection_grid(coords, df, encodings, proj_name,
                                   ncols=params['ncols'], figsize=params['figsize'],
                                   force_legend=params.get('force_legend', False))
                plt.show()


def plot_continuous_features(projections_dict, df, features, proj_name='t-SNE'):
    """Plot multiple continuous features for a specific projection in a grid layout.

    Args:
        projections_dict: Dict of {proj_name: coords_array}
        df: DataFrame containing feature columns
        features: List of continuous feature names to visualize
        proj_name: Which projection to use (default: t-SNE)
    """
    if proj_name not in projections_dict:
        print(f"Projection '{proj_name}' not found. Available: {list(projections_dict.keys())}")
        return

    coords = projections_dict[proj_name]

    # Filter to only existing features
    existing_features = [f for f in features if f in df.columns]
    if len(existing_features) == 0:
        print("No valid features found in dataframe")
        return

    # Create subplot grid with 3 columns
    ncols = 3
    nrows = (len(existing_features) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 6*nrows))
    axes = axes.flatten() if nrows > 1 else [axes] if len(existing_features) == 1 else axes

    for idx, feature in enumerate(existing_features):
        ax = axes[idx]
        values = df[feature].values

        scatter = ax.scatter(coords[:, 0], coords[:, 1],
                            c=values, cmap='viridis', alpha=0.7, s=8,
                            edgecolors='none')

        plt.colorbar(scatter, ax=ax, label=feature)
        ax.set_title(f'{feature}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Dim 1', fontsize=10)
        ax.set_ylabel('Dim 2', fontsize=10)

    # Hide unused subplots
    for idx in range(len(existing_features), len(axes)):
        axes[idx].axis('off')

    fig.suptitle(f'{proj_name} - Continuous Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def export_projections_with_metadata(projections_dict, df, output_dir="./data/spotify/exports/"):
    """Export projection coordinates with full metadata to CSV files.

    Args:
        projections_dict: Dict of {proj_name: coords_array}
        df: DataFrame with all metadata columns
        output_dir: Directory to save CSV files
    """
    os.makedirs(output_dir, exist_ok=True)

    for proj_name, proj_coords in projections_dict.items():
        df_export = df.copy()
        df_export['x'] = proj_coords[:, 0]
        df_export['y'] = proj_coords[:, 1]

        filename = f"{output_dir}{proj_name.lower().replace('-', '_')}_full.csv"
        df_export.to_csv(filename, index=False)
        print(f"Exported {proj_name} to {filename}")
