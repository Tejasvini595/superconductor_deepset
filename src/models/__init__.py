"""
Models Package

This package provides the DeepSet model architecture for superconductor prediction.
"""

from .deepset import (
    DeepSetSuperconductor,
    create_deepset_model,
    compile_model
)

__all__ = [
    'DeepSetSuperconductor',
    'create_deepset_model',
    'compile_model'
]
