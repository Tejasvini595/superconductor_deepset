"""
Features Package

This package provides functionality for feature extraction and input creation
for the superconductor DeepSet model.
"""

from .composition_parser import (
    parse_composition,
    validate_composition,
    normalize_composition,
    get_element_count
)
from .mendeleev_features import MendeleevFeatureExtractor
from .deepset_input import DeepSetInputCreator

__all__ = [
    'parse_composition',
    'validate_composition',
    'normalize_composition',
    'get_element_count',
    'MendeleevFeatureExtractor',
    'DeepSetInputCreator'
]
