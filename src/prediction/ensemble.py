"""
Ensemble Prediction Module

This module provides ensemble prediction functionality using multiple trained models.
"""

import numpy as np
import joblib
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging
from tensorflow import keras

from ..features import DeepSetInputCreator

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """Ensemble predictor using multiple trained DeepSet models."""
    
    def __init__(
        self,
        models_dir: str,
        config: dict,
        num_models: Optional[int] = None
    ):
        """
        Initialize ensemble predictor.
        
        Args:
            models_dir: Directory containing trained models
            config: Configuration dictionary
            num_models: Number of models to load (default: all models in directory)
        """
        self.models_dir = Path(models_dir)
        self.config = config
        self.features_config = config.get('features', {})
        self.max_elements = self.features_config.get('max_elements', 10)
        
        # Load element features
        features_path = self.models_dir / 'element_features_dict.pkl'
        if not features_path.exists():
            raise FileNotFoundError(f"Element features not found: {features_path}")
        
        self.element_features_dict = joblib.load(features_path)
        logger.info(f"Loaded element features from: {features_path}")
        
        # Load models and scalers
        self.models, self.scalers = self._load_models_and_scalers(num_models)
        logger.info(f"Loaded {len(self.models)} models for ensemble prediction")
        
        # Create input creator
        self.input_creator = DeepSetInputCreator(
            self.element_features_dict,
            max_elements=self.max_elements
        )
    
    def _load_models_and_scalers(self, num_models: Optional[int] = None) -> Tuple[List, List]:
        """
        Load trained models and scalers from directory.
        
        Args:
            num_models: Number of models to load (if None, load all)
            
        Returns:
            Tuple of (models_list, scalers_list)
        """
        models = []
        scalers = []
        
        # Find all model files
        model_files = sorted(self.models_dir.glob('deepset_model_run_*.h5'))
        
        if num_models is not None:
            model_files = model_files[:num_models]
        
        logger.info(f"Loading {len(model_files)} models...")
        
        for model_file in model_files:
            # Extract run number
            run_num = int(model_file.stem.split('_')[-1])
            
            # Load model
            model = keras.models.load_model(str(model_file))
            models.append(model)
            
            # Load corresponding scaler
            scaler_file = self.models_dir / f'scaler_run_{run_num}.pkl'
            if not scaler_file.exists():
                raise FileNotFoundError(f"Scaler not found: {scaler_file}")
            
            scaler = joblib.load(scaler_file)
            scalers.append(scaler)
        
        return models, scalers
    
    def predict_single(
        self,
        composition: str,
        debug: bool = False
    ) -> Tuple[float, float]:
        """
        Predict Tc for a single composition using ensemble.
        
        Args:
            composition: Chemical formula (e.g., "YBa2Cu3O7")
            debug: If True, print detailed information
            
        Returns:
            Tuple of (mean_prediction, std_prediction)
        """
        predictions = []
        
        if debug:
            logger.info(f"Predicting Tc for: {composition}")
            logger.info(f"Using {len(self.models)} models in ensemble")
        
        for i, (model, scaler) in enumerate(zip(self.models, self.scalers)):
            # Create input
            deepset_input = self.input_creator.create_input(composition)
            deepset_input = deepset_input.reshape(1, self.max_elements, 23)
            
            # Normalize
            normalized_input = scaler.transform(
                deepset_input.reshape(-1, 23)
            ).reshape(1, self.max_elements, 23)
            
            # Predict
            pred = model.predict(normalized_input, verbose=0)[0][0]
            predictions.append(pred)
            
            if debug and i < 3:
                logger.info(f"  Model {i+1} prediction: {pred:.2f}K")
        
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        if debug:
            logger.info(f"  ...")
            logger.info(f"\nEnsemble Results:")
            logger.info(f"  Mean: {mean_pred:.2f}K")
            logger.info(f"  Std:  {std_pred:.2f}K")
            logger.info(f"  Range: [{np.min(predictions):.2f}, {np.max(predictions):.2f}]K")
            logger.info(f"  95% CI: [{mean_pred - 1.96*std_pred:.2f}, {mean_pred + 1.96*std_pred:.2f}]K")
        
        return mean_pred, std_pred
    
    def predict_batch(
        self,
        compositions: List[str],
        show_progress: bool = True
    ) -> Dict:
        """
        Predict Tc for multiple compositions.
        
        Args:
            compositions: List of chemical formulas
            show_progress: If True, show progress bar
            
        Returns:
            Dictionary with predictions and uncertainties
        """
        import pandas as pd
        
        results = []
        failed = []
        
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(compositions, desc="Predicting Tc")
        else:
            iterator = compositions
        
        for comp in iterator:
            try:
                mean_tc, std_tc = self.predict_single(comp, debug=False)
                
                results.append({
                    'Composition': comp,
                    'Predicted_Tc_Mean': mean_tc,
                    'Predicted_Tc_Std': std_tc,
                    'Predicted_Tc_Min': mean_tc - std_tc,
                    'Predicted_Tc_Max': mean_tc + std_tc,
                    'CI_95_Lower': mean_tc - 1.96 * std_tc,
                    'CI_95_Upper': mean_tc + 1.96 * std_tc
                })
                
            except Exception as e:
                logger.warning(f"Error predicting {comp}: {e}")
                failed.append({'Composition': comp, 'Error': str(e)})
                results.append({
                    'Composition': comp,
                    'Predicted_Tc_Mean': None,
                    'Predicted_Tc_Std': None,
                    'Predicted_Tc_Min': None,
                    'Predicted_Tc_Max': None,
                    'CI_95_Lower': None,
                    'CI_95_Upper': None
                })
        
        results_df = pd.DataFrame(results)
        
        logger.info(f"Predictions complete:")
        logger.info(f"  Total: {len(compositions)}")
        logger.info(f"  Successful: {len(results) - len(failed)}")
        logger.info(f"  Failed: {len(failed)}")
        
        return {
            'predictions': results_df,
            'failed': failed,
            'num_successful': len(results) - len(failed),
            'num_failed': len(failed)
        }
    
    def save_predictions(
        self,
        predictions_df,
        output_path: str
    ):
        """
        Save predictions to CSV file.
        
        Args:
            predictions_df: DataFrame with predictions
            output_path: Path to save CSV
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        predictions_df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to: {output_path}")
