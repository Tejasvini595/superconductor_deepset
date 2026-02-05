"""
Data Preprocessor Module

This module handles data preprocessing including train/test splits,
feature normalization, and DeepSet input creation.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SuperconductorPreprocessor:
    """Preprocess data for DeepSet model training."""
    
    def __init__(self, config: dict):
        """
        Initialize the preprocessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.training_config = config.get('training', {})
        self.features_config = config.get('features', {})
        self.max_elements = self.features_config.get('max_elements', 10)
        self.feature_dim = self.features_config.get('feature_dim', 23)
    
    def create_train_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: Optional[float] = None,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create train/test split.
        
        Args:
            X: Feature array of shape (n_samples, max_elements, feature_dim)
            y: Target array of shape (n_samples,)
            test_size: Fraction for test set (default from config)
            random_state: Random seed
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        if test_size is None:
            test_size = self.training_config.get('test_size', 0.20)
        
        logger.info(f"Creating train/test split with test_size={test_size}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
        
        return X_train, X_test, y_train, y_test
    
    def create_train_val_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: Optional[float] = None,
        val_size: Optional[float] = None,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create train/validation/test split.
        
        Args:
            X: Feature array
            y: Target array
            test_size: Fraction for test set (default from config)
            val_size: Fraction of remaining data for validation (default from config)
            random_state: Random seed
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        if test_size is None:
            test_size = self.training_config.get('test_size', 0.20)
        
        if val_size is None:
            val_size = self.training_config.get('val_size', 0.176)
        
        logger.info(
            f"Creating train/val/test split with test_size={test_size}, val_size={val_size}"
        )
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Second split: separate validation from training
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state
        )
        
        logger.info(
            f"Train size: {X_train.shape[0]}, "
            f"Val size: {X_val.shape[0]}, "
            f"Test size: {X_test.shape[0]}"
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_train_val_test_split_with_indices(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: Optional[float] = None,
        val_size: Optional[float] = None,
        random_state: int = 42
    ) -> Tuple:
        """
        Create train/validation/test split and return indices as well.
        
        This is useful for tracking which samples were used in which set.
        
        Returns:
            Tuple of (indices_train, indices_val, indices_test, 
                     X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if test_size is None:
            test_size = self.training_config.get('test_size', 0.20)
        
        if val_size is None:
            val_size = self.training_config.get('val_size', 0.176)
        
        # Create indices array
        indices = np.arange(len(X))
        
        # First split
        indices_temp, indices_test, X_temp, X_test, y_temp, y_test = train_test_split(
            indices, X, y, test_size=test_size, random_state=random_state
        )
        
        # Second split
        indices_train, indices_val, X_train, X_val, y_train, y_val = train_test_split(
            indices_temp, X_temp, y_temp, test_size=val_size, random_state=random_state
        )
        
        return (indices_train, indices_val, indices_test,
                X_train, X_val, X_test, y_train, y_val, y_test)
    
    def normalize_features(
        self,
        X_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None
    ) -> Tuple:
        """
        Normalize features using StandardScaler fitted on training data.
        
        Args:
            X_train: Training features (n_train, max_elements, feature_dim)
            X_val: Validation features (optional)
            X_test: Test features (optional)
            
        Returns:
            Tuple of (scaler, X_train_normalized, X_val_normalized, X_test_normalized)
            If X_val or X_test is None, returns None for those
        """
        logger.info("Normalizing features with StandardScaler")
        
        # Reshape for fitting (flatten element dimension)
        X_train_reshaped = X_train.reshape(-1, self.feature_dim)
        
        # Fit scaler on training data
        scaler = StandardScaler()
        X_train_normalized = scaler.fit_transform(X_train_reshaped)
        X_train_normalized = X_train_normalized.reshape(-1, self.max_elements, self.feature_dim)
        
        logger.info(
            f"Scaler fitted - mean: {scaler.mean_[:3]}..., std: {scaler.scale_[:3]}..."
        )
        
        # Transform validation data if provided
        X_val_normalized = None
        if X_val is not None:
            X_val_reshaped = X_val.reshape(-1, self.feature_dim)
            X_val_normalized = scaler.transform(X_val_reshaped)
            X_val_normalized = X_val_normalized.reshape(-1, self.max_elements, self.feature_dim)
        
        # Transform test data if provided
        X_test_normalized = None
        if X_test is not None:
            X_test_reshaped = X_test.reshape(-1, self.feature_dim)
            X_test_normalized = scaler.transform(X_test_reshaped)
            X_test_normalized = X_test_normalized.reshape(-1, self.max_elements, self.feature_dim)
        
        return scaler, X_train_normalized, X_val_normalized, X_test_normalized
