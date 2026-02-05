"""
SHAP Analysis Module

This module provides SHAP-based interpretability analysis for DeepSet models.
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging
from tensorflow import keras

logger = logging.getLogger(__name__)


class SHAPAnalyzer:
    """SHAP interpretability analysis for DeepSet models."""
    
    def __init__(self, config: dict, models_dir: str):
        """
        Initialize SHAP analyzer.
        
        Args:
            config: Configuration dictionary
            models_dir: Directory containing trained models
        """
        self.config = config
        self.models_dir = Path(models_dir)
        self.shap_config = config.get('shap', {})
        self.features_config = config.get('features', {})
        
        self.max_elements = self.features_config.get('max_elements', 10)
        self.feature_names = self.features_config.get('feature_names', [])
        
        self.num_top_models = self.shap_config.get('num_top_models', 5)
        self.num_background = self.shap_config.get('num_background_samples', 50)
        self.num_explain = self.shap_config.get('num_explain_samples', 100)
        self.kernel_nsamples = self.shap_config.get('kernel_nsamples', 50)
        self.clip_percentile = self.shap_config.get('clip_percentile', 99.9)
    
    def load_top_models(self, metrics_path: Optional[str] = None) -> Tuple[List, List, List[int]]:
        """
        Load top N models based on test R².
        
        Args:
            metrics_path: Path to metrics CSV
            
        Returns:
            Tuple of (models, scalers, model_indices)
        """
        if metrics_path is None:
            metrics_path = self.models_dir / 'all_runs_metrics.csv'
        
        # Load metrics
        metrics_df = pd.read_csv(metrics_path)
        logger.info(f"Loaded metrics from: {metrics_path}")
        
        # Get top models
        top_models_df = metrics_df.nlargest(self.num_top_models, 'test_r2')
        top_indices = (top_models_df['run'].values - 1).astype(int)
        
        logger.info(f"Top {self.num_top_models} models (by R²):")
        for idx, row in top_models_df.iterrows():
            logger.info(f"  Run {int(row['run'])}: R²={row['test_r2']:.4f}, "
                       f"MAE={row['test_mae']:.2f}K, RMSE={row['test_rmse']:.2f}K")
        
        # Load models and scalers
        models = []
        scalers = []
        
        for idx in top_indices:
            run_num = idx + 1
            
            model_path = self.models_dir / f'deepset_model_run_{run_num}.h5'
            scaler_path = self.models_dir / f'scaler_run_{run_num}.pkl'
            
            models.append(keras.models.load_model(str(model_path)))
            scalers.append(joblib.load(scaler_path))
        
        logger.info(f"Loaded {len(models)} models for SHAP analysis")
        
        return models, scalers, top_indices
    
    def compute_shap_values(
        self,
        models: List,
        scalers: List,
        X_data: np.ndarray
    ) -> np.ndarray:
        """
        Compute SHAP values for top models.
        
        Args:
            models: List of trained models
            scalers: List of fitted scalers
            X_data: Data to explain (n_samples, max_elements, 23)
            
        Returns:
            SHAP values array of shape (n_models, n_samples, max_elements, 23)
        """
        all_shap_values = []
        
        # Prepare data
        background_data = X_data[:self.num_background]
        explain_data = X_data[self.num_background:self.num_background+self.num_explain]
        
        logger.info(f"Computing SHAP values:")
        logger.info(f"  Background samples: {self.num_background}")
        logger.info(f"  Samples to explain: {self.num_explain}")
        
        for i, (model, scaler) in enumerate(zip(models, scalers)):
            logger.info(f"Processing model {i+1}/{len(models)}...")
            
            try:
                # Normalize data
                bg_norm = scaler.transform(background_data.reshape(-1, 23)).reshape(-1, self.max_elements, 23)
                ex_norm = scaler.transform(explain_data.reshape(-1, 23)).reshape(-1, self.max_elements, 23)
                
                # Flatten for SHAP
                bg_flat = bg_norm.reshape(self.num_background, -1)
                ex_flat = ex_norm.reshape(self.num_explain, -1)
                
                # Model wrapper
                def model_predict(x):
                    if len(x.shape) == 2:
                        x = x.reshape(-1, self.max_elements, 23)
                    return model.predict(x, verbose=0)
                
                # Compute SHAP values
                explainer = shap.KernelExplainer(model_predict, bg_flat)
                shap_values = explainer.shap_values(ex_flat, nsamples=self.kernel_nsamples)
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                
                # Clip extreme values
                shap_abs = np.abs(shap_values)
                threshold = np.percentile(shap_abs, self.clip_percentile)
                shap_values = np.clip(shap_values, -threshold, threshold)
                
                # Reshape to (num_samples, max_elements, 23)
                shap_values_reshaped = shap_values.reshape(self.num_explain, self.max_elements, 23)
                all_shap_values.append(shap_values_reshaped)
                
                logger.info(f"  ✓ Model {i+1} complete")
                
            except Exception as e:
                logger.error(f"  ✗ Error with model {i+1}: {e}")
                continue
        
        return np.array(all_shap_values)
    
    def aggregate_shap_values(self, shap_values: np.ndarray) -> Dict:
        """
        Aggregate SHAP values across models.
        
        Args:
            shap_values: Array of shape (n_models, n_samples, max_elements, 23)
            
        Returns:
            Dictionary with aggregated results
        """
        # Take absolute values
        shap_abs = np.abs(shap_values)
        
        # Average across models, samples, and elements
        ensemble_shap = np.mean(shap_abs, axis=0)  # (n_samples, max_elements, 23)
        ensemble_shap_flat = ensemble_shap.reshape(-1, 23)
        
        # Feature importance
        mean_importance = np.mean(ensemble_shap_flat, axis=0)
        median_importance = np.median(ensemble_shap_flat, axis=0)
        std_importance = np.std(ensemble_shap_flat, axis=0)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Mean_Abs_SHAP': mean_importance,
            'Median_Abs_SHAP': median_importance,
            'Std_Dev': std_importance
        })
        importance_df['Rank'] = importance_df['Median_Abs_SHAP'].rank(ascending=False)
        importance_df = importance_df.sort_values('Median_Abs_SHAP', ascending=False)
        
        logger.info("\nFeature Importance (Top 10):")
        for _, row in importance_df.head(10).iterrows():
            logger.info(f"  {int(row['Rank']):2d}. {row['Feature']:<30} "
                       f"Median: {row['Median_Abs_SHAP']:.6f}")
        
        return {
            'importance_df': importance_df,
            'mean_importance': mean_importance,
            'median_importance': median_importance,
            'ensemble_shap': ensemble_shap
        }
    
    def save_results(self, results: Dict, output_dir: str):
        """
        Save SHAP analysis results.
        
        Args:
            results: Dictionary with analysis results
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save importance DataFrame
        importance_path = output_dir / 'shap_importance.csv'
        results['importance_df'].to_csv(importance_path, index=False)
        logger.info(f"Saved importance to: {importance_path}")
        
        # Save SHAP values
        shap_path = output_dir / 'shap_values.npy'
        np.save(shap_path, results['ensemble_shap'])
        logger.info(f"Saved SHAP values to: {shap_path}")
    
    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        output_dir: str,
        top_n: int = 23
    ):
        """
        Plot feature importance bar chart.
        
        Args:
            importance_df: DataFrame with feature importance
            output_dir: Directory to save plots
            top_n: Number of top features to show
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sort by median importance
        sorted_df = importance_df.sort_values('Median_Abs_SHAP', ascending=True).tail(top_n)
        
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
        
        y_pos = np.arange(len(sorted_df))
        ax.barh(y_pos, sorted_df['Median_Abs_SHAP'], alpha=0.8, color='steelblue')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_df['Feature'], fontsize=10)
        ax.set_xlabel('Median |SHAP Value|', fontsize=12)
        ax.set_title(f'Top {top_n} Most Important Features\n(SHAP Analysis)', 
                     fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        save_path = output_dir / 'feature_importance.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved feature importance plot to: {save_path}")
        plt.close()
