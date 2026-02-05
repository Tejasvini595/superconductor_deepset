"""
Data Loader Module

This module handles loading and initial validation of the superconductor dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SuperconductorDataLoader:
    """Load and validate superconductor dataset."""
    
    def __init__(self, config: dict):
        """
        Initialize the data loader.
        
        Args:
            config: Configuration dictionary containing data paths and column names
        """
        self.config = config
        self.data_config = config.get('data', {})
    
    def load_dataset(self, dataset_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load the superconductor dataset from CSV.
        
        Args:
            dataset_path: Path to CSV file. If None, uses path from config.
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If dataset file doesn't exist
            ValueError: If dataset is empty or invalid
        """
        if dataset_path is None:
            dataset_path = self.data_config.get('dataset_path')
        
        if not dataset_path:
            raise ValueError("Dataset path not specified in config or arguments")
        
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        logger.info(f"Loading dataset from: {dataset_path}")
        
        # Load with automatic delimiter detection
        df = pd.read_csv(dataset_path, sep=None, engine='python')
        
        logger.info(f"Loaded dataset with shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        
        if df.empty:
            raise ValueError("Dataset is empty")
        
        return df
    
    def rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename columns to standard names.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with renamed columns
        """
        composition_col = self.data_config.get('composition_column', 'element')
        target_col = self.data_config.get('target_column', 'critical_temp_K')
        renamed_composition = self.data_config.get('renamed_composition', 'composition')
        renamed_target = self.data_config.get('renamed_target', 'Tc')
        
        rename_dict = {
            composition_col: renamed_composition,
            target_col: renamed_target
        }
        
        df = df.rename(columns=rename_dict)
        logger.info(f"Renamed columns: {rename_dict}")
        
        return df
    
    def clean_target(self, df: pd.DataFrame, target_col: str = 'Tc') -> pd.DataFrame:
        """
        Clean the target variable (Tc) by removing non-numeric values.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            
        Returns:
            Cleaned DataFrame
        """
        initial_size = len(df)
        
        # Convert to numeric, coercing errors to NaN
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        
        # Remove NaN values
        df = df[df[target_col].notna()].copy()
        
        final_size = len(df)
        removed = initial_size - final_size
        
        if removed > 0:
            logger.info(f"Removed {removed} rows with invalid Tc values")
        
        logger.info(f"Final dataset shape after cleaning: {df.shape}")
        
        return df
    
    def get_data_summary(self, df: pd.DataFrame, target_col: str = 'Tc') -> dict:
        """
        Get summary statistics of the dataset.
        
        Args:
            df: DataFrame
            target_col: Name of target column
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'num_samples': len(df),
            'num_features': len(df.columns),
            'target_min': df[target_col].min(),
            'target_max': df[target_col].max(),
            'target_mean': df[target_col].mean(),
            'target_std': df[target_col].std(),
            'target_median': df[target_col].median()
        }
        
        return summary
    
    def load_test_compositions(
        self, 
        filepath: Optional[str] = None
    ) -> list:
        """
        Load test compositions from a text file (one per line).
        
        Args:
            filepath: Path to text file. If None, uses path from config.
            
        Returns:
            List of composition strings
        """
        if filepath is None:
            filepath = self.data_config.get('test_compositions_path')
        
        if not filepath:
            raise ValueError("Test compositions path not specified")
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Test compositions file not found: {filepath}")
        
        logger.info(f"Loading test compositions from: {filepath}")
        
        with open(filepath, 'r') as f:
            compositions = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Loaded {len(compositions)} test compositions")
        
        return compositions
    
    def load_and_prepare(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Load and prepare the complete dataset.
        
        Returns:
            Tuple of (DataFrame, target values as numpy array)
        """
        # Load dataset
        df = self.load_dataset()
        
        # Rename columns
        df = self.rename_columns(df)
        
        # Clean target
        df = self.clean_target(df)
        
        # Extract target values
        target_col = self.data_config.get('renamed_target', 'Tc')
        y = df[target_col].values
        
        # Log summary
        summary = self.get_data_summary(df, target_col)
        logger.info(f"Dataset summary: {summary}")
        
        return df, y
