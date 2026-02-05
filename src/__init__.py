"""
Superconductor DeepSet Package

This package provides a complete pipeline for superconductor critical temperature
prediction using DeepSet neural networks.
"""

__version__ = '1.0.0'
__author__ = 'Research Team'

from . import data
from . import features
from . import models
from . import training
from . import evaluation
from . import prediction
from . import analysis
from . import utils

__all__ = [
    'data',
    'features',
    'models',
    'training',
    'evaluation',
    'prediction',
    'analysis',
    'utils'
]
