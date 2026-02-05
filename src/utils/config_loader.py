"""
Configuration Loader Module

This module provides utilities for loading and validating YAML configuration files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    logger.info(f"Loading configuration from: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("Configuration loaded successfully")
    
    return config


def save_config(config: Dict[str, Any], output_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save YAML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Configuration saved to: {output_path}")


def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration with new values (deep update).
    
    Args:
        config: Original configuration dictionary
        updates: Dictionary with updates
        
    Returns:
        Updated configuration
    """
    import copy
    
    def deep_update(base_dict, update_dict):
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                base_dict[key] = deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
        return base_dict
    
    updated_config = copy.deepcopy(config)
    updated_config = deep_update(updated_config, updates)
    
    return updated_config


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration structure.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    required_sections = ['data', 'features', 'model', 'training', 'paths']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate data section
    data_config = config['data']
    required_data_fields = ['dataset_path', 'composition_column', 'target_column']
    for field in required_data_fields:
        if field not in data_config:
            raise ValueError(f"Missing required data config field: {field}")
    
    # Validate features section
    features_config = config['features']
    if 'max_elements' not in features_config:
        raise ValueError("Missing 'max_elements' in features config")
    
    # Validate model section
    model_config = config['model']
    if 'latent_dim' not in model_config:
        raise ValueError("Missing 'latent_dim' in model config")
    
    # Validate training section
    training_config = config['training']
    required_training_fields = ['batch_size', 'epochs', 'learning_rate']
    for field in required_training_fields:
        if field not in training_config:
            raise ValueError(f"Missing required training config field: {field}")
    
    logger.info("Configuration validation passed")
    return True
