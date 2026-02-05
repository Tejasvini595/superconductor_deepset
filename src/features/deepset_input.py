"""
DeepSet Input Creation Module

This module creates the input tensors for the DeepSet neural network by combining
elemental features with stoichiometric coefficients.
"""

import numpy as np
from typing import Dict, List, Optional
import logging
from .composition_parser import parse_composition

logger = logging.getLogger(__name__)


class DeepSetInputCreator:
    """
    Creates DeepSet-compatible input tensors from chemical compositions.
    
    Each composition is converted to a (max_elements, 23) tensor where:
    - Each row represents one element in the composition
    - First 22 columns are Mendeleev features
    - Last column is the stoichiometric coefficient
    - Padded with zeros if fewer than max_elements
    """
    
    def __init__(
        self, 
        element_features_dict: Dict[str, List[float]], 
        max_elements: int = 10
    ):
        """
        Initialize the input creator.
        
        Args:
            element_features_dict: Dictionary mapping element symbols to 22D feature vectors
            max_elements: Maximum number of elements to consider (padding/truncation)
        """
        self.element_features_dict = element_features_dict
        self.max_elements = max_elements
        self.feature_dim = 23  # 22 elemental features + 1 stoichiometry
    
    def create_input(
        self, 
        composition_str: str, 
        debug: bool = False
    ) -> np.ndarray:
        """
        Create DeepSet input tensor for a single composition.
        
        Args:
            composition_str: Chemical formula (e.g., "YBa2Cu3O7")
            debug: If True, print detailed information
            
        Returns:
            Numpy array of shape (max_elements, 23)
        """
        # Parse composition
        element_composition = parse_composition(composition_str)
        
        if not element_composition:
            if debug:
                logger.warning(f"Empty composition: {composition_str}")
            return np.zeros((self.max_elements, self.feature_dim))
        
        element_vectors = []
        
        for element_symbol, stoichiometry in element_composition.items():
            # Check if we've reached max_elements
            if len(element_vectors) >= self.max_elements:
                if debug:
                    logger.warning(
                        f"Composition {composition_str} has more than {self.max_elements} elements. "
                        f"Truncating."
                    )
                break
            
            # Get element features
            if element_symbol in self.element_features_dict:
                elem_features = self.element_features_dict[element_symbol].copy()
                elem_features.append(float(stoichiometry))
                element_vectors.append(elem_features)
                
                if debug:
                    logger.debug(
                        f"Element {element_symbol}: stoichiometry={stoichiometry}, "
                        f"features shape={len(elem_features)}"
                    )
            else:
                if debug:
                    logger.warning(
                        f"Element {element_symbol} not found in feature dictionary. Skipping."
                    )
        
        # Pad with zeros if necessary
        original_length = len(element_vectors)
        while len(element_vectors) < self.max_elements:
            element_vectors.append([0.0] * self.feature_dim)
        
        if debug:
            logger.debug(
                f"Composition {composition_str}: {original_length} elements, "
                f"padded to {self.max_elements}"
            )
        
        return np.array(element_vectors[:self.max_elements])
    
    def create_batch_inputs(
        self, 
        compositions: List[str], 
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Create DeepSet input tensors for multiple compositions.
        
        Args:
            compositions: List of chemical formulas
            show_progress: If True, show progress during processing
            
        Returns:
            Numpy array of shape (num_compositions, max_elements, 23)
        """
        inputs = []
        
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(compositions, desc="Creating DeepSet inputs")
        else:
            iterator = compositions
        
        for composition in iterator:
            deepset_input = self.create_input(composition)
            inputs.append(deepset_input)
        
        return np.array(inputs)
    
    def get_input_shape(self) -> tuple:
        """Get the expected input shape for the model."""
        return (self.max_elements, self.feature_dim)
