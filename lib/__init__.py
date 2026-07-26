"""
Projection Analysis Library

A reusable library for projection methods, quality metrics, clustering evaluation,
and visualization for dimensionality reduction and clustering analysis.
"""

__version__ = "0.2.0"

from . import config
from . import projections
from . import metrics
from . import visualization
from . import clustering
from . import logger_utils
from . import plotting_utils
from . import utils
from . import dataset_processing
from . import data_exploration

__all__ = ['config', 'projections', 'metrics', 'visualization', 'clustering', 
           'logger_utils', 'plotting_utils', 'utils', 'dataset_processing',
           'data_exploration']
