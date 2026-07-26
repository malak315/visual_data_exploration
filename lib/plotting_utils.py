"""
Plotting utilities with file-saving capabilities and headless support.

This module provides plotting functions that automatically save plots to files
and can run in headless mode (without displaying plots).
"""

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from datetime import datetime

from lib.config import (
    PLOTS_DIR, SAVE_PLOTS, SHOW_PLOTS, PLOT_FORMAT, PLOT_DPI
)
import lib.logger_utils as logger_utils

# Set matplotlib backend for headless execution if needed
if not SHOW_PLOTS:
    matplotlib.use('Agg')

# Get logger
logger = logger_utils.get_logger("plotting")


class PlotManager:
    """
    Manages plot creation, saving, and display.
    
    This class handles the complexity of saving plots to files while
    optionally displaying them, making it easy to run analysis headless.
    """
    
    def __init__(
        self,
        save_dir: Optional[Path] = None,
        save_plots: bool = True,
        show_plots: bool = False,
        format: str = "png",
        dpi: int = 300
    ):
        """
        Initialize the plot manager.
        
        Args:
            save_dir: Directory to save plots (defaults to config.PLOTS_DIR)
            save_plots: Whether to save plots to files
            show_plots: Whether to display plots
            format: Format for saved plots (png, pdf, svg)
            dpi: Resolution for saved plots
        """
        self.save_dir = save_dir or PLOTS_DIR
        self.save_plots = save_plots
        self.show_plots = show_plots
        self.format = format
        self.dpi = dpi
        
        if self.save_plots:
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def save_figure(
        self,
        fig: plt.Figure,
        filename: str,
        subdirectory: Optional[str] = None
    ) -> Path:
        """
        Save a figure to file.
        
        Args:
            fig: Matplotlib figure to save
            filename: Name of the file (without extension)
            subdirectory: Optional subdirectory within plots directory
            
        Returns:
            Path to saved file
        """
        if not self.save_plots:
            return None
        
        # Create subdirectory if specified
        save_path = self.save_dir
        if subdirectory:
            save_path = save_path / subdirectory
            save_path.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp to filename to avoid overwrites
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename_with_ext = f"{filename}_{timestamp}.{self.format}"
        filepath = save_path / filename_with_ext
        
        # Save figure
        fig.savefig(
            filepath,
            format=self.format,
            dpi=self.dpi,
            bbox_inches='tight',
            facecolor='white'
        )
        
        logger.info(f"Saved plot: {filepath}")
        return filepath
    
    def show_or_save(
        self,
        fig: plt.Figure,
        filename: str,
        subdirectory: Optional[str] = None,
        close: bool = True
    ) -> Optional[Path]:
        """
        Display and/or save a figure.
        
        Args:
            fig: Matplotlib figure
            filename: Name for saved file (if saving)
            subdirectory: Optional subdirectory for saved file
            close: Whether to close the figure after saving/displaying
            
        Returns:
            Path to saved file if saved, None otherwise
        """
        saved_path = None
        
        # Save if requested
        if self.save_plots:
            saved_path = self.save_figure(fig, filename, subdirectory)
        
        # Show if requested
        if self.show_plots:
            plt.show()
        elif close:
            plt.close(fig)
        
        return saved_path


# Global plot manager instance
_plot_manager: Optional[PlotManager] = None


def get_plot_manager() -> PlotManager:
    """
    Get or create the global plot manager instance.
    
    Returns:
        PlotManager instance
    """
    global _plot_manager
    if _plot_manager is None:
        _plot_manager = PlotManager(
            save_plots=SAVE_PLOTS,
            show_plots=SHOW_PLOTS,
            format=PLOT_FORMAT,
            dpi=PLOT_DPI
        )
    return _plot_manager


def plot_and_save(
    fig: plt.Figure,
    filename: str,
    subdirectory: Optional[str] = None,
    close: bool = True
) -> Optional[Path]:
    """
    Convenience function to plot and save a figure.
    
    Args:
        fig: Matplotlib figure
        filename: Name for saved file
        subdirectory: Optional subdirectory for saved file
        close: Whether to close the figure after saving
        
    Returns:
        Path to saved file if saved, None otherwise
    """
    return get_plot_manager().show_or_save(fig, filename, subdirectory, close)


def create_figure(figsize: Tuple[int, int] = (10, 6), **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a figure and axes with consistent styling.
    
    Args:
        figsize: Figure size (width, height)
        **kwargs: Additional arguments to pass to plt.subplots
        
    Returns:
        Tuple of (figure, axes)
    """
    fig, ax = plt.subplots(figsize=figsize, **kwargs)
    return fig, ax


def create_subplots(
    nrows: int = 1,
    ncols: int = 1,
    figsize: Tuple[int, int] = (10, 6),
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create subplots with consistent styling.
    
    Args:
        nrows: Number of rows
        ncols: Number of columns
        figsize: Figure size (width, height)
        **kwargs: Additional arguments to pass to plt.subplots
        
    Returns:
        Tuple of (figure, axes)
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    return fig, axes


def style_plot(ax: plt.Axes, title: str = "", xlabel: str = "", ylabel: str = ""):
    """
    Apply consistent styling to a plot.
    
    Args:
        ax: Matplotlib axes
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def save_plot_data(data: Dict[str, Any], filename: str, subdirectory: Optional[str] = None):
    """
    Save plot data (e.g., statistics, metrics) to a JSON file for later analysis.
    
    Args:
        data: Dictionary of data to save
        filename: Name of the file (without extension)
        subdirectory: Optional subdirectory within plots directory
    """
    import json
    
    save_path = PLOTS_DIR
    if subdirectory:
        save_path = save_path / subdirectory
        save_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = save_path / f"{filename}_{timestamp}.json"
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    logger.info(f"Saved plot data: {filepath}")
    return filepath

