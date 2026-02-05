#!/usr/bin/env python
"""
Training Script

Train multiple DeepSet models for superconductor Tc prediction.

Usage:
    python scripts/train.py --config config/config.yaml --num-runs 50
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, validate_config, setup_logger_from_config
from src.data import SuperconductorDataLoader
from src.features import MendeleevFeatureExtractor, DeepSetInputCreator
from src.training import ModelTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train DeepSet models for superconductor prediction'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--num-runs',
        type=int,
        default=None,
        help='Number of independent training runs (overrides config)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for models (overrides config)'
    )
    
    parser.add_argument(
        '--verbose',
        type=int,
        default=1,
        help='Verbosity level for training (0=silent, 1=progress bar, 2=one line per epoch)'
    )
    
    return parser.parse_args()


def main():
    """Main training pipeline."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    validate_config(config)
    
    # Override config with command line arguments
    if args.num_runs is not None:
        config['training']['num_runs'] = args.num_runs
    
    if args.output_dir is not None:
        config['paths']['saved_models_dir'] = args.output_dir
    
    # Setup logging
    logger = setup_logger_from_config(config)
    logger.info("="*80)
    logger.info("SUPERCONDUCTOR DeepSet TRAINING")
    logger.info("="*80)
    
    # Load data
    logger.info("\n1. LOADING DATA")
    data_loader = SuperconductorDataLoader(config)
    df, y = data_loader.load_and_prepare()
    
    # Extract features
    logger.info("\n2. EXTRACTING MENDELEEV FEATURES")
    feature_extractor = MendeleevFeatureExtractor(
        max_atomic_number=config['features']['max_atomic_number']
    )
    element_features_dict = feature_extractor.extract_features()
    
    # Create DeepSet inputs
    logger.info("\n3. CREATING DEEPSET INPUTS")
    input_creator = DeepSetInputCreator(
        element_features_dict,
        max_elements=config['features']['max_elements']
    )
    
    compositions = df[config['data']['renamed_composition']].values
    X = input_creator.create_batch_inputs(compositions, show_progress=True)
    
    logger.info(f"Created DeepSet inputs with shape: {X.shape}")
    
    # Train models
    logger.info(f"\n4. TRAINING {config['training']['num_runs']} MODELS")
    trainer = ModelTrainer(config)
    
    models, scalers, results = trainer.train_multiple_runs(
        X, y,
        element_features_dict=element_features_dict,
        num_runs=config['training']['num_runs'],
        verbose=args.verbose
    )
    
    logger.info("="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info(f"Models saved to: {config['paths']['saved_models_dir']}")
    logger.info(f"Total models trained: {len(models)}")


if __name__ == '__main__':
    main()
