"""
Metrics Module

This module provides evaluation metrics for superconductor prediction.
"""

import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary with R², MAE, RMSE, MSE
    """
    metrics = {
        'r2': r2_score(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mse': mean_squared_error(y_true, y_pred)
    }
    
    return metrics


def calculate_ensemble_metrics(
    all_test_predictions: Dict[int, List[float]],
    y: np.ndarray,
    min_tests: int = 10
) -> Dict:
    """
    Calculate metrics for ensemble predictions.
    
    Args:
        all_test_predictions: Dict mapping sample indices to list of predictions
        y: True target values
        min_tests: Minimum number of times a sample must be tested
        
    Returns:
        Dictionary with metrics and filtered data
    """
    # Filter materials tested at least min_tests times
    qualified_materials = {
        idx: preds for idx, preds in all_test_predictions.items()
        if len(preds) >= min_tests
    }
    
    logger.info(f"Materials tested at least {min_tests} times: {len(qualified_materials)}")
    
    # Calculate mean predictions and uncertainties
    predicted_temps = []
    measured_temps = []
    error_bars = []
    
    for idx, predictions in qualified_materials.items():
        predicted_temps.append(np.mean(predictions))
        measured_temps.append(y[idx])
        error_bars.append(np.std(predictions))
    
    predicted_temps = np.array(predicted_temps)
    measured_temps = np.array(measured_temps)
    error_bars = np.array(error_bars)
    
    # Calculate metrics
    metrics = calculate_regression_metrics(measured_temps, predicted_temps)
    
    results = {
        'metrics': metrics,
        'num_materials': len(qualified_materials),
        'predicted_temps': predicted_temps,
        'measured_temps': measured_temps,
        'error_bars': error_bars,
        'min_tests': min_tests
    }
    
    logger.info(f"Ensemble metrics (materials tested ≥{min_tests} times):")
    logger.info(f"  N = {results['num_materials']}")
    logger.info(f"  R² = {metrics['r2']:.4f}")
    logger.info(f"  MAE = {metrics['mae']:.2f}K")
    logger.info(f"  RMSE = {metrics['rmse']:.2f}K")
    
    return results
