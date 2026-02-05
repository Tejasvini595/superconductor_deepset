"""
Training Module

This module handles the training of DeepSet models including
callbacks, model saving, and experiment tracking.
"""

import tensorflow as tf
import mlflow
import mlflow.tensorflow
import numpy as np
import joblib
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train DeepSet models with MLflow tracking and model saving."""
    
    def __init__(self, config: dict):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.training_config = config.get('training', {})
        self.paths_config = config.get('paths', {})
        self.mlflow_config = config.get('mlflow', {})
        
        self.batch_size = self.training_config.get('batch_size', 64)
        self.epochs = self.training_config.get('epochs', 400)
        
        # Set up MLflow
        experiment_name = self.mlflow_config.get('experiment_name', 'deepset_superconductor')
        mlflow.set_experiment(experiment_name)
    
    def create_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """
        Create training callbacks.
        
        Returns:
            List of Keras callbacks
        """
        callbacks = []
        
        # Early stopping
        early_stopping_config = self.training_config.get('early_stopping', {})
        if early_stopping_config:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor=early_stopping_config.get('monitor', 'val_loss'),
                patience=early_stopping_config.get('patience', 40),
                restore_best_weights=early_stopping_config.get('restore_best_weights', True)
            )
            callbacks.append(early_stopping)
            logger.info(f"Added EarlyStopping callback with patience={early_stopping_config.get('patience')}")
        
        # Reduce learning rate on plateau
        reduce_lr_config = self.training_config.get('reduce_lr', {})
        if reduce_lr_config:
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor=reduce_lr_config.get('monitor', 'val_loss'),
                patience=reduce_lr_config.get('patience', 15),
                factor=reduce_lr_config.get('factor', 0.5)
            )
            callbacks.append(reduce_lr)
            logger.info(f"Added ReduceLROnPlateau callback with patience={reduce_lr_config.get('patience')}")
        
        return callbacks
    
    def train_single_run(
        self,
        model: tf.keras.Model,
        scaler,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        run_number: int,
        test_indices: np.ndarray,
        verbose: int = 1
    ) -> Tuple[tf.keras.Model, Dict]:
        """
        Train a single model run.
        
        Args:
            model: Compiled Keras model
            scaler: Fitted StandardScaler
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data
            run_number: Run number (for logging and saving)
            test_indices: Indices of test samples
            verbose: Verbosity level for training
            
        Returns:
            Tuple of (trained_model, metrics_dict)
        """
        logger.info(f"Starting Run {run_number}")
        
        # Create callbacks
        callbacks = self.create_callbacks()
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"run_{run_number}"):
            # Log parameters
            mlflow.log_param("run_number", run_number)
            mlflow.log_param("random_state", run_number - 1)
            mlflow.log_param("batch_size", self.batch_size)
            mlflow.log_param("epochs", self.epochs)
            
            # Train model
            start_time = time.time()
            
            history = model.fit(
                X_train, y_train,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=verbose
            )
            
            training_time = time.time() - start_time
            
            # Make predictions on test set
            test_pred = model.predict(X_test, verbose=0)
            
            # Calculate metrics
            test_r2 = r2_score(y_test, test_pred)
            test_mae = mean_absolute_error(y_test, test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            
            # Log metrics
            mlflow.log_metric("test_r2", test_r2)
            mlflow.log_metric("test_mae", test_mae)
            mlflow.log_metric("test_rmse", test_rmse)
            mlflow.log_metric("training_time", training_time)
            mlflow.log_metric("epochs_trained", len(history.history['loss']))
            
            # Save model and scaler
            models_dir = Path(self.paths_config.get('saved_models_dir', 'saved_models'))
            models_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = models_dir / f'deepset_model_run_{run_number}.h5'
            scaler_path = models_dir / f'scaler_run_{run_number}.pkl'
            
            model.save(str(model_path))
            joblib.dump(scaler, str(scaler_path))
            
            logger.info(f"Run {run_number} completed:")
            logger.info(f"  Training time: {training_time:.2f}s")
            logger.info(f"  Test R²: {test_r2:.4f}")
            logger.info(f"  Test MAE: {test_mae:.4f}K")
            logger.info(f"  Test RMSE: {test_rmse:.4f}K")
            logger.info(f"  Epochs trained: {len(history.history['loss'])}")
            logger.info(f"  Model saved: {model_path}")
            logger.info(f"  Scaler saved: {scaler_path}")
            
            # Prepare metrics dictionary
            metrics = {
                'run': run_number,
                'test_r2': test_r2,
                'test_mae': test_mae,
                'test_rmse': test_rmse,
                'training_time': training_time,
                'epochs_trained': len(history.history['loss']),
                'final_train_loss': history.history['loss'][-1],
                'final_val_loss': history.history['val_loss'][-1]
            }
            
            # Return model, predictions, and metrics
            return model, {
                'metrics': metrics,
                'test_predictions': test_pred.flatten(),
                'test_indices': test_indices
            }
    
    def train_multiple_runs(
        self,
        X: np.ndarray,
        y: np.ndarray,
        element_features_dict: dict,
        num_runs: Optional[int] = None,
        verbose: int = 1
    ) -> Tuple[List, List, Dict]:
        """
        Train multiple independent models with different random seeds.
        
        Args:
            X: Feature array (n_samples, max_elements, feature_dim)
            y: Target array (n_samples,)
            element_features_dict: Dictionary of element features
            num_runs: Number of independent runs (default from config)
            verbose: Verbosity level
            
        Returns:
            Tuple of (models_list, scalers_list, results_dict)
        """
        from ..data import SuperconductorPreprocessor
        from ..models import create_deepset_model, compile_model
        
        if num_runs is None:
            num_runs = self.training_config.get('num_runs', 50)
        
        logger.info(f"Starting training of {num_runs} independent models")
        
        preprocessor = SuperconductorPreprocessor(self.config)
        
        all_models = []
        all_scalers = []
        all_metrics = []
        all_test_predictions = {}
        
        for run_num in range(1, num_runs + 1):
            logger.info("="*80)
            
            # Create train/val/test split with different random seed
            random_state = run_num - 1
            (indices_train, indices_val, indices_test,
             X_train, X_val, X_test, 
             y_train, y_val, y_test) = preprocessor.create_train_val_test_split_with_indices(
                X, y, random_state=random_state
            )
            
            # Normalize features
            scaler, X_train_norm, X_val_norm, X_test_norm = preprocessor.normalize_features(
                X_train, X_val, X_test
            )
            
            # Create and compile model
            model = create_deepset_model(self.config)
            model = compile_model(model, self.config)
            
            # Train model
            model, results = self.train_single_run(
                model, scaler,
                X_train_norm, y_train,
                X_val_norm, y_val,
                X_test_norm, y_test,
                run_number=run_num,
                test_indices=indices_test,
                verbose=verbose
            )
            
            # Store results
            all_models.append(model)
            all_scalers.append(scaler)
            all_metrics.append(results['metrics'])
            
            # Store test predictions
            for idx, pred in zip(results['test_indices'], results['test_predictions']):
                if idx not in all_test_predictions:
                    all_test_predictions[idx] = []
                all_test_predictions[idx].append(pred)
        
        logger.info("="*80)
        logger.info(f"All {num_runs} runs completed!")
        
        # Save aggregated results
        self._save_aggregated_results(
            all_metrics, 
            all_test_predictions,
            element_features_dict
        )
        
        return all_models, all_scalers, {
            'metrics': all_metrics,
            'test_predictions': all_test_predictions
        }
    
    def _save_aggregated_results(
        self,
        all_metrics: List[Dict],
        all_test_predictions: Dict,
        element_features_dict: dict
    ):
        """Save aggregated results from all runs."""
        import pandas as pd
        
        models_dir = Path(self.paths_config.get('saved_models_dir', 'saved_models'))
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_df = pd.DataFrame(all_metrics)
        metrics_path = models_dir / 'all_runs_metrics.csv'
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Metrics saved to: {metrics_path}")
        
        # Save test predictions
        predictions_path = models_dir / 'all_test_predictions.npy'
        np.save(predictions_path, all_test_predictions)
        logger.info(f"Test predictions saved to: {predictions_path}")
        
        # Save element features
        features_path = models_dir / 'element_features_dict.pkl'
        joblib.dump(element_features_dict, features_path)
        logger.info(f"Element features saved to: {features_path}")
        
        # Print summary statistics
        logger.info(f"\nSummary across all runs:")
        logger.info(f"  Mean Test R²: {metrics_df['test_r2'].mean():.4f} ± {metrics_df['test_r2'].std():.4f}")
        logger.info(f"  Mean Test MAE: {metrics_df['test_mae'].mean():.2f} ± {metrics_df['test_mae'].std():.2f}K")
        logger.info(f"  Mean Test RMSE: {metrics_df['test_rmse'].mean():.2f} ± {metrics_df['test_rmse'].std():.2f}K")
        logger.info(f"  Mean Training Time: {metrics_df['training_time'].mean():.2f} ± {metrics_df['training_time'].std():.2f}s")
