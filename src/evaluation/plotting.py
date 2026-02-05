"""
Plotting Module

This module provides visualization functions for model evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


def plot_predictions_vs_actual(
    predicted: np.ndarray,
    actual: np.ndarray,
    errors: Optional[np.ndarray] = None,
    title: str = "Predicted vs Measured Tc",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot predicted vs actual values with error bars.
    
    Args:
        predicted: Predicted values
        actual: Actual values
        errors: Optional error bars (std dev)
        title: Plot title
        save_path: Path to save figure
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    if errors is not None:
        ax.errorbar(
            actual, predicted, yerr=errors,
            fmt='o', alpha=0.6, capsize=3, label='Predictions'
        )
    else:
        ax.scatter(actual, predicted, alpha=0.6, label='Predictions')
    
    # Perfect prediction line
    min_val, max_val = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
            'k--', alpha=0.5, label='Perfect prediction')
    
    ax.set_xlabel('Measured Tc (K)', fontsize=14)
    ax.set_ylabel('Predicted Tc (K)', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_prediction_distribution(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    save_dir: Optional[str] = None,
    show: bool = True
):
    """
    Plot distribution of predictions and uncertainties.
    
    Args:
        predictions: Array of predicted values
        uncertainties: Array of prediction uncertainties
        save_dir: Directory to save figures
        show: Whether to display plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of predictions
    axes[0].hist(predictions, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Predicted Tc (K)', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Distribution of Predicted Tc', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # Histogram of uncertainties
    axes[1].hist(uncertainties, bins=50, color='coral', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Prediction Uncertainty (Std, K)', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Distribution of Prediction Uncertainties', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'prediction_distributions.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Distribution plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_top_predictions(
    predictions_df: pd.DataFrame,
    top_n: int = 20,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot top N predictions with error bars.
    
    Args:
        predictions_df: DataFrame with columns ['Composition', 'Predicted_Tc_Mean', 'Predicted_Tc_Std']
        top_n: Number of top predictions to show
        save_path: Path to save figure
        show: Whether to display plot
    """
    top = predictions_df.nlargest(top_n, 'Predicted_Tc_Mean')
    
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    
    y_pos = np.arange(len(top))
    
    ax.barh(y_pos, top['Predicted_Tc_Mean'],
            xerr=top['Predicted_Tc_Std'],
            alpha=0.7, color='purple', capsize=3)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top['Composition'], fontsize=8)
    ax.set_xlabel('Predicted Tc (K)', fontsize=12)
    ax.set_title(f'Top {top_n} Predicted Superconductors', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Top predictions plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
