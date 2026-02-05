"""
Composition Parser Module

This module provides utilities for parsing chemical composition strings
and extracting element-stoichiometry pairs.
"""

import re
import pandas as pd
from typing import Dict


def parse_composition(composition_str: str) -> Dict[str, float]:
    """
    Parse composition string to extract elements and their stoichiometry.
    
    Args:
        composition_str: Chemical formula string (e.g., "YBa2Cu3O7", "MgB2")
        
    Returns:
        Dictionary mapping element symbols to their stoichiometric coefficients
        
    Examples:
        >>> parse_composition("YBa2Cu3O7")
        {'Y': 1.0, 'Ba': 2.0, 'Cu': 3.0, 'O': 7.0}
        
        >>> parse_composition("MgB2")
        {'Mg': 1.0, 'B': 2.0}
        
        >>> parse_composition("La0.0039SiV2.9961")
        {'La': 0.0039, 'Si': 1.0, 'V': 2.9961}
    """
    if pd.isna(composition_str):
        return {}
    
    # Pattern to match element symbols followed by optional numeric coefficients
    # Matches: Element (capital letter + optional lowercase) + optional number (int or float)
    pattern = r'([A-Z][a-z]?)(\d*\.?\d*)'
    matches = re.findall(pattern, composition_str)
    
    element_dict = {}
    for element, count in matches:
        if count == '':
            count = 1.0
        else:
            count = float(count)
        element_dict[element] = count
    
    return element_dict


def validate_composition(composition_str: str) -> bool:
    """
    Validate if a composition string is properly formatted.
    
    Args:
        composition_str: Chemical formula string
        
    Returns:
        True if valid, False otherwise
    """
    if pd.isna(composition_str) or composition_str.strip() == '':
        return False
    
    try:
        element_dict = parse_composition(composition_str)
        return len(element_dict) > 0
    except Exception:
        return False


def normalize_composition(element_dict: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize stoichiometric coefficients to sum to 1.
    
    Args:
        element_dict: Dictionary of element-stoichiometry pairs
        
    Returns:
        Normalized dictionary where values sum to 1
        
    Example:
        >>> normalize_composition({'Y': 1, 'Ba': 2, 'Cu': 3, 'O': 7})
        {'Y': 0.0769, 'Ba': 0.1538, 'Cu': 0.2308, 'O': 0.5385}
    """
    if not element_dict:
        return {}
    
    total = sum(element_dict.values())
    if total == 0:
        return element_dict
    
    return {elem: count / total for elem, count in element_dict.items()}


def get_element_count(composition_str: str) -> int:
    """
    Get the number of unique elements in a composition.
    
    Args:
        composition_str: Chemical formula string
        
    Returns:
        Number of unique elements
    """
    element_dict = parse_composition(composition_str)
    return len(element_dict)
