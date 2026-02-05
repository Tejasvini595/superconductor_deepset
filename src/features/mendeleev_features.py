"""
Mendeleev Features Extraction Module

This module extracts 22 elemental features from the Mendeleev periodic table
for use in the DeepSet superconductor prediction model.
"""

import numpy as np
from mendeleev import element
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class MendeleevFeatureExtractor:
    """
    Extract 22 elemental features from the periodic table using the Mendeleev library.
    
    Features extracted (in order):
    1. atomic_number
    2. atomic_volume
    3. block_encoded (s=1, p=2, d=3, f=4)
    4. density
    5. dipole_polarizability
    6. electron_affinity
    7. evaporation_heat
    8. fusion_heat
    9. group_id
    10. lattice_constant
    11. lattice_structure (placeholder: 1)
    12. melting_point
    13. period
    14. specific_heat
    15. thermal_conductivity
    16. vdw_radius
    17. covalent_radius_pyykko
    18. en_pauling
    19. atomic_weight
    20. atomic_radius_rahm
    21. first_ionization_energy
    22. valence_electrons
    """
    
    def __init__(self, max_atomic_number: int = 96):
        """
        Initialize the feature extractor.
        
        Args:
            max_atomic_number: Maximum atomic number to process (default: 96)
        """
        self.max_atomic_number = max_atomic_number
        self.feature_names = [
            'atomic_number', 'atomic_volume', 'block_encoded', 'density', 
            'dipole_polarizability', 'electron_affinity', 'evaporation_heat', 
            'fusion_heat', 'group_id', 'lattice_constant', 'lattice_structure', 
            'melting_point', 'period', 'specific_heat', 'thermal_conductivity',
            'vdw_radius', 'covalent_radius_pyykko', 'en_pauling', 'atomic_weight', 
            'atomic_radius_rahm', 'first_ionization_energy', 'valence_electrons'
        ]
        self.num_features = 22
        self._element_features = None
    
    def _encode_block(self, block: str) -> int:
        """
        Encode electron block as integer.
        
        Args:
            block: Electron block (s, p, d, f)
            
        Returns:
            Encoded value: s=1, p=2, d=3, f=4
        """
        block_map = {'s': 1, 'p': 2, 'd': 3, 'f': 4}
        return block_map.get(block, 0)
    
    def extract_features(self) -> Dict[str, List[float]]:
        """
        Extract features for all elements up to max_atomic_number.
        
        Returns:
            Dictionary mapping element symbols to feature vectors (22D)
        """
        if self._element_features is not None:
            return self._element_features
        
        element_features = {}
        
        logger.info(f"Extracting Mendeleev features for elements 1-{self.max_atomic_number}")
        
        for atomic_num in range(1, self.max_atomic_number + 1):
            try:
                elem = element(atomic_num)
                symbol = elem.symbol
                
                features = [
                    elem.atomic_number,
                    elem.atomic_volume if elem.atomic_volume is not None else 0,
                    self._encode_block(elem.block),
                    elem.density if elem.density is not None else 0,
                    elem.dipole_polarizability if elem.dipole_polarizability is not None else 0,
                    elem.electron_affinity if elem.electron_affinity is not None else 0,
                    elem.evaporation_heat if elem.evaporation_heat is not None else 0,
                    elem.fusion_heat if elem.fusion_heat is not None else 0,
                    elem.group_id if elem.group_id is not None else 0,
                    elem.lattice_constant if elem.lattice_constant is not None else 0,
                    1,  # lattice_structure placeholder
                    elem.melting_point if elem.melting_point is not None else 0,
                    elem.period,
                    elem.specific_heat if hasattr(elem, 'specific_heat') and elem.specific_heat is not None else 0,
                    elem.thermal_conductivity if elem.thermal_conductivity is not None else 0,
                    elem.vdw_radius if elem.vdw_radius is not None else 0,
                    elem.covalent_radius_pyykko if elem.covalent_radius_pyykko is not None else 0,
                    elem.en_pauling if elem.en_pauling is not None else 0,
                    elem.atomic_weight,
                    elem.atomic_radius_rahm if elem.atomic_radius_rahm is not None else 0,
                    elem.ionenergies[1] if len(elem.ionenergies) > 1 else 0,
                    elem.nvalence() if hasattr(elem, 'nvalence') else 0
                ]
                
                # Handle missing values and convert to float
                element_features[symbol] = [
                    0.0 if (f is None or (isinstance(f, float) and np.isnan(f))) 
                    else float(f) 
                    for f in features
                ]
                
            except Exception as e:
                logger.warning(f"Error processing element {atomic_num}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(element_features)} elements")
        self._element_features = element_features
        
        return element_features
    
    def get_element_features(self, element_symbol: str) -> Optional[List[float]]:
        """
        Get feature vector for a specific element.
        
        Args:
            element_symbol: Chemical symbol (e.g., 'Cu', 'O', 'Y')
            
        Returns:
            22D feature vector, or None if element not found
        """
        if self._element_features is None:
            self.extract_features()
        
        return self._element_features.get(element_symbol)
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names.copy()
    
    @property
    def element_features(self) -> Dict[str, List[float]]:
        """Get all element features (cached)."""
        if self._element_features is None:
            self.extract_features()
        return self._element_features
