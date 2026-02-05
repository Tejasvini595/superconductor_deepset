#!/usr/bin/env python
"""
Prediction Script

Make ensemble predictions for new superconductor compositions.

Usage:
    # Predict from command line
    python scripts/predict.py --config config/config.yaml --compositions "YBa2Cu3O7" "MgB2"
    
    # Predict from file
    python scripts/predict.py --config config/config.yaml --input data/test_compositions.txt --output results/predictions.csv
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, setup_logger_from_config
from src.prediction import EnsemblePredictor
from src.evaluation import plot_prediction_distribution, plot_top_predictions


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Predict Tc for superconductor compositions'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--compositions',
        type=str,
        nargs='+',
        default=None,
        help='Compositions to predict (space-separated)'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Path to text file with compositions (one per line)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file for predictions'
    )
    
    parser.add_argument(
        '--models-dir',
        type=str,
        default=None,
        help='Directory with trained models (overrides config)'
    )
    
    parser.add_argument(
        '--num-models',
        type=int,
        default=None,
        help='Number of models to use in ensemble'
    )
    
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate plots'
    )
    
    return parser.parse_args()


def main():
    """Main prediction pipeline."""
    args = parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logger_from_config(config)
    logger.info("="*80)
    logger.info("SUPERCONDUCTOR DeepSet PREDICTION")
    logger.info("="*80)
    
    # Determine models directory
    models_dir = args.models_dir or config['paths']['saved_models_dir']
    
    # Determine compositions to predict
    compositions = None
    
    if args.compositions:
        compositions = args.compositions
        logger.info(f"Predicting for {len(compositions)} compositions from command line")
    
    elif args.input:
        logger.info(f"Loading compositions from: {args.input}")
        with open(args.input, 'r') as f:
            compositions = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(compositions)} compositions")
    
    else:
        # Use test compositions from config
        input_path = config['data'].get('test_compositions_path')
        if input_path:
            logger.info(f"Loading compositions from: {input_path}")
            with open(input_path, 'r') as f:
                compositions = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(compositions)} compositions")
        else:
            logger.error("No compositions specified. Use --compositions or --input")
            sys.exit(1)
    
    # Initialize predictor
    logger.info(f"\nLoading ensemble predictor from: {models_dir}")
    predictor = EnsemblePredictor(
        models_dir=models_dir,
        config=config,
        num_models=args.num_models
    )
    
    # Make predictions
    logger.info(f"\nMaking predictions...")
    results = predictor.predict_batch(compositions, show_progress=True)
    
    predictions_df = results['predictions']
    
    # Print summary
    logger.info(f"\nPrediction Summary:")
    logger.info(f"  Total: {len(compositions)}")
    logger.info(f"  Successful: {results['num_successful']}")
    logger.info(f"  Failed: {results['num_failed']}")
    
    # Print top predictions
    successful_df = predictions_df[predictions_df['Predicted_Tc_Mean'].notna()]
    if len(successful_df) > 0:
        logger.info(f"\nTop 10 Predictions:")
        top_10 = successful_df.nlargest(10, 'Predicted_Tc_Mean')
        for i, row in enumerate(top_10.itertuples(), 1):
            logger.info(
                f"{i:2d}. {row.Composition:<30} "
                f"{row.Predicted_Tc_Mean:7.2f} ± {row.Predicted_Tc_Std:5.2f} K"
            )
    
    # Save predictions
    if args.output:
        output_path = args.output
    else:
        results_dir = Path(config['paths']['results_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / 'predictions.csv'
    
    predictor.save_predictions(predictions_df, output_path)
    
    # Generate plots
    if args.plot and len(successful_df) > 0:
        logger.info("\nGenerating plots...")
        
        figures_dir = Path(config['paths']['figures_dir'])
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Distribution plots
        plot_prediction_distribution(
            predictions=successful_df['Predicted_Tc_Mean'].values,
            uncertainties=successful_df['Predicted_Tc_Std'].values,
            save_dir=str(figures_dir),
            show=False
        )
        
        # Top predictions plot
        plot_top_predictions(
            predictions_df=successful_df,
            top_n=20,
            save_path=str(figures_dir / 'top_20_predictions.png'),
            show=False
        )
        
        logger.info(f"Plots saved to: {figures_dir}")
    
    logger.info("="*80)
    logger.info("PREDICTION COMPLETE!")
    logger.info("="*80)


if __name__ == '__main__':
    main()
