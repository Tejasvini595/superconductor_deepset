#!/usr/bin/env python
"""
SHAP Analysis Script

Run SHAP interpretability analysis on trained models.

Usage:
    python scripts/analyze_shap.py --config config/config.yaml
"""

import argparse
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, setup_logger_from_config
from src.data import SuperconductorDataLoader
from src.features import MendeleevFeatureExtractor, DeepSetInputCreator
from src.analysis import SHAPAnalyzer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run SHAP analysis on DeepSet models'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--models-dir',
        type=str,
        default=None,
        help='Directory with trained models (overrides config)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results (default: results/shap)'
    )
    
    parser.add_argument(
        '--top-models',
        type=int,
        default=None,
        help='Number of top models to use (overrides config)'
    )
    
    return parser.parse_args()


def main():
    """Main SHAP analysis pipeline."""
    args = parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Override config
    if args.top_models is not None:
        config['shap']['num_top_models'] = args.top_models
    
    # Setup logging
    logger = setup_logger_from_config(config)
    logger.info("="*80)
    logger.info("SUPERCONDUCTOR DeepSet SHAP ANALYSIS")
    logger.info("="*80)
    
    # Determine directories
    models_dir = args.models_dir or config['paths']['saved_models_dir']
    output_dir = args.output_dir or Path(config['paths']['results_dir']) / 'shap'
    
    # Load data for SHAP analysis
    logger.info("\n1. LOADING DATA")
    data_loader = SuperconductorDataLoader(config)
    df, y = data_loader.load_and_prepare()
    
    # Extract features
    logger.info("\n2. EXTRACTING FEATURES")
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
    
    # Use subset for SHAP (full dataset would be too slow)
    max_samples = 1000
    if len(compositions) > max_samples:
        logger.info(f"Using {max_samples} samples for SHAP analysis (subset of {len(compositions)})")
        indices = np.random.choice(len(compositions), max_samples, replace=False)
        compositions_subset = compositions[indices]
    else:
        compositions_subset = compositions
    
    X = input_creator.create_batch_inputs(compositions_subset, show_progress=True)
    logger.info(f"Created DeepSet inputs with shape: {X.shape}")
    
    # Initialize SHAP analyzer
    logger.info(f"\n4. RUNNING SHAP ANALYSIS")
    analyzer = SHAPAnalyzer(config, models_dir)
    
    # Load top models
    models, scalers, model_indices = analyzer.load_top_models()
    
    # Compute SHAP values
    logger.info("\nComputing SHAP values (this may take a while)...")
    shap_values = analyzer.compute_shap_values(models, scalers, X)
    
    if len(shap_values) == 0:
        logger.error("No SHAP values computed. Exiting.")
        sys.exit(1)
    
    # Aggregate results
    logger.info("\nAggregating SHAP values...")
    results = analyzer.aggregate_shap_values(shap_values)
    
    # Save results
    logger.info(f"\n5. SAVING RESULTS")
    analyzer.save_results(results, output_dir)
    
    # Plot feature importance
    logger.info("\n6. GENERATING PLOTS")
    analyzer.plot_feature_importance(
        results['importance_df'],
        output_dir,
        top_n=23
    )
    
    logger.info("="*80)
    logger.info("SHAP ANALYSIS COMPLETE!")
    logger.info("="*80)
    logger.info(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
