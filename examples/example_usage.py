#!/usr/bin/env python
"""
Example: Using the package programmatically

This shows how to use the superconductor-deepset package in your own Python code.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config
from src.prediction import EnsemblePredictor


def example_single_prediction():
    """Example: Predict Tc for a single composition."""
    print("="*60)
    print("Example 1: Single Prediction")
    print("="*60)
    
    # Load configuration
    config = load_config('config/config.yaml')
    
    # Initialize predictor
    predictor = EnsemblePredictor(
        models_dir='saved_models',
        config=config
    )
    
    # Predict
    composition = "YBa2Cu3O7"
    mean_tc, std_tc = predictor.predict_single(composition, debug=True)
    
    print(f"\nResult: {composition}")
    print(f"  Predicted Tc: {mean_tc:.2f} ± {std_tc:.2f} K")
    print(f"  95% CI: [{mean_tc - 1.96*std_tc:.2f}, {mean_tc + 1.96*std_tc:.2f}] K")


def example_batch_prediction():
    """Example: Predict Tc for multiple compositions."""
    print("\n" + "="*60)
    print("Example 2: Batch Prediction")
    print("="*60)
    
    # Load configuration
    config = load_config('config/config.yaml')
    
    # Initialize predictor
    predictor = EnsemblePredictor(
        models_dir='saved_models',
        config=config,
        num_models=10  # Use only 10 models for faster prediction
    )
    
    # Compositions to predict
    compositions = [
        "YBa2Cu3O7",
        "MgB2",
        "NbTi",
        "Nb3Sn",
        "La3Ni2O7"
    ]
    
    # Predict
    results = predictor.predict_batch(compositions, show_progress=True)
    
    # Display results
    predictions_df = results['predictions']
    print(f"\nPredictions:")
    print(predictions_df[['Composition', 'Predicted_Tc_Mean', 'Predicted_Tc_Std']].to_string(index=False))
    
    # Save to CSV
    predictor.save_predictions(predictions_df, 'results/example_predictions.csv')


def example_custom_workflow():
    """Example: Custom workflow with individual components."""
    print("\n" + "="*60)
    print("Example 3: Custom Workflow")
    print("="*60)
    
    from src.utils import load_config, setup_logger
    from src.data import SuperconductorDataLoader
    from src.features import MendeleevFeatureExtractor, DeepSetInputCreator
    
    # Setup
    config = load_config('config/config.yaml')
    logger = setup_logger(level='INFO')
    
    logger.info("Loading data...")
    data_loader = SuperconductorDataLoader(config)
    df, y = data_loader.load_and_prepare()
    
    logger.info(f"Dataset: {len(df)} samples")
    logger.info(f"Tc range: {y.min():.2f} - {y.max():.2f} K")
    
    logger.info("Extracting features...")
    feature_extractor = MendeleevFeatureExtractor()
    element_features = feature_extractor.extract_features()
    logger.info(f"Extracted features for {len(element_features)} elements")
    
    logger.info("Creating DeepSet inputs for first 5 compositions...")
    input_creator = DeepSetInputCreator(element_features, max_elements=10)
    
    for i in range(5):
        composition = df.iloc[i]['composition']
        deepset_input = input_creator.create_input(composition)
        logger.info(f"  {composition}: shape {deepset_input.shape}")
    
    print("\n✓ Custom workflow completed successfully!")


if __name__ == '__main__':
    # Check if models exist
    models_dir = Path('saved_models')
    if not models_dir.exists() or not list(models_dir.glob('deepset_model_run_*.h5')):
        print("ERROR: No trained models found!")
        print("Please run training first: python scripts/train.py")
        sys.exit(1)
    
    # Run examples
    try:
        example_single_prediction()
        example_batch_prediction()
        example_custom_workflow()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
