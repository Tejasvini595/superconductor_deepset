# Superconductor Critical Temperature Prediction using DeepSet Neural Networks

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the implementation of a **DeepSet neural network** for predicting the critical temperature (Tc) of superconducting materials based on their chemical composition. The model achieves state-of-the-art performance by leveraging permutation-invariant set representations of elemental features.

### Key Features

- **Composition-based prediction**: Predicts Tc from chemical formulas (e.g., `YBa2Cu3O7`)
- **DeepSet architecture**: Permutation-invariant neural network respecting set structure
- **Ensemble learning**: 50 independent models for robust predictions with uncertainty quantification
- **SHAP interpretability**: Identify the most important elemental features
- **MLflow tracking**: Comprehensive experiment tracking and model versioning
- **22 Elemental features**: Extracted from Mendeleev's periodic table

## Project Structure

```
superconductor-deepset/
├── config/
│   └── config.yaml              # Configuration for model, training, and paths
├── data/
│   ├── test_compositions.txt    # Hosono dataset compositions for testing
│   └── superconductors_kaggle_ready_v2.csv  # Training dataset (add your data here)
├── notebooks/
│   └── original_notebook.ipynb  # Original research notebook
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py            # Data loading utilities
│   │   └── preprocessor.py      # Data preprocessing
│   ├── features/
│   │   ├── __init__.py
│   │   ├── composition_parser.py # Parse chemical formulas
│   │   ├── mendeleev_features.py # Extract elemental features
│   │   └── deepset_input.py      # Create DeepSet input tensors
│   ├── models/
│   │   ├── __init__.py
│   │   └── deepset.py           # DeepSet model architecture
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py           # Training loop and callbacks
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py           # Evaluation metrics
│   │   └── plotting.py          # Visualization functions
│   ├── prediction/
│   │   ├── __init__.py
│   │   └── ensemble.py          # Ensemble prediction
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── shap_analysis.py     # SHAP interpretability
│   └── utils/
│       ├── __init__.py
│       ├── config_loader.py     # Load YAML configuration
│       └── logger.py            # Logging setup
├── scripts/
│   ├── train.py                 # Train 50 models
│   ├── predict.py               # Make predictions on new compositions
│   └── analyze_shap.py          # Run SHAP analysis
├── saved_models/                # Trained models (created during training)
├── results/                     # Evaluation results and figures
├── logs/                        # Training logs
├── requirements.txt             # Python dependencies
├── .gitignore
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/superconductor-deepset.git
cd superconductor-deepset
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### 1. Training

Train 50 independent models with different random seeds:

```bash
python scripts/train.py --config config/config.yaml
```

**Arguments:**
- `--config`: Path to configuration file (default: `config/config.yaml`)
- `--num-runs`: Number of independent runs (default: 50)
- `--output-dir`: Directory to save models (default: `saved_models`)

**Output:**
- Trained models: `saved_models/deepset_model_run_*.h5`
- Scalers: `saved_models/scaler_run_*.pkl`
- Element features: `saved_models/element_features_dict.pkl`
- Metrics: `saved_models/all_runs_metrics.csv`

### 2. Prediction

Predict Tc for new compositions using ensemble:

```bash
python scripts/predict.py --config config/config.yaml --compositions "YBa2Cu3O7" "MgB2"
```

Or predict from a file:

```bash
python scripts/predict.py --config config/config.yaml --input data/test_compositions.txt --output results/predictions.csv
```

**Arguments:**
- `--config`: Path to configuration file
- `--compositions`: List of chemical formulas (space-separated)
- `--input`: Path to text file with compositions (one per line)
- `--output`: Output CSV file for results
- `--models-dir`: Directory with trained models (default: `saved_models`)

**Output:**
- Predictions with mean, std, and 95% confidence intervals
- CSV file: `Composition, Predicted_Tc_Mean, Predicted_Tc_Std, CI_95_Lower, CI_95_Upper`

### 3. SHAP Analysis

Run interpretability analysis to identify important features:

```bash
python scripts/analyze_shap.py --config config/config.yaml
```

**Arguments:**
- `--config`: Path to configuration file
- `--top-models`: Number of best models to use (default: 5)
- `--output-dir`: Directory to save results (default: `results/shap`)

**Output:**
- Feature importance CSV: `results/shap/shap_importance.csv`
- Visualizations: `results/shap/*.png`
- SHAP values: `results/shap/shap_values.npy`

## Model Architecture

### DeepSet Neural Network

The model uses a permutation-invariant architecture suitable for set-structured data:

1. **φ-network** (per-element encoder):
   - Input: 23D vector per element (22 features + stoichiometry)
   - Layers: [992, 768, 512, 384, 256, 128, 300]
   - Activation: ReLU

2. **Pooling**: Sum pooling across elements (permutation-invariant)

3. **ρ-network** (aggregator):
   - Input: 300D pooled representation
   - Layers: [960, 832, 768, 640, 512, 384, 256, 192, 160, 128, 96, 64, 1]
   - Output: Predicted Tc (K)

### Elemental Features (22 features from Mendeleev)

1. atomic_number
2. atomic_volume
3. block_encoded
4. density
5. dipole_polarizability
6. electron_affinity
7. evaporation_heat
8. fusion_heat
9. group_id
10. lattice_constant
11. lattice_structure
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

## Results

### Performance Metrics

- **Test RMSE**: ~X.XX K (averaged over 50 runs)
- **Test R²**: ~0.XXX (averaged over 50 runs)
- **Test MAE**: ~X.XX K (averaged over 50 runs)

### Example Predictions

| Composition | Predicted Tc (K) | Uncertainty (±K) |
|-------------|------------------|------------------|
| YBa₂Cu₃O₇   | XX.XX           | ±X.XX            |
| MgB₂        | XX.XX           | ±X.XX            |
| Nb₃Sn       | XX.XX           | ±X.XX            |

## Configuration

All hyperparameters can be adjusted in `config/config.yaml`:

- **Model architecture**: Layer sizes, latent dimensions
- **Training**: Batch size, learning rate, epochs, callbacks
- **Data**: Paths, column names, feature settings
- **Ensemble**: Number of models, confidence intervals
- **SHAP**: Background samples, aggregation method

## Citation

If you use this code in your research, please cite:

```bibtex
@article{your_paper_2026,
  title={Predicting Superconductor Critical Temperature using DeepSet Neural Networks},
  author={Your Name},
  journal={Journal Name},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- SuperCon dataset: [Kaggle](https://www.kaggle.com/)
- Mendeleev library for elemental properties
- DeepSet architecture: [Zaheer et al., 2017](https://arxiv.org/abs/1703.06114)

## Contact

For questions or collaborations, please contact:
- **Email**: your.email@institution.edu
- **GitHub**: [@yourusername](https://github.com/yourusername)

## Reproducibility

To reproduce the results:

1. Place your dataset in `data/superconductors_kaggle_ready_v2.csv`
2. Run training: `python scripts/train.py`
3. Run prediction: `python scripts/predict.py --input data/test_compositions.txt`
4. Run SHAP analysis: `python scripts/analyze_shap.py`

All random seeds are fixed in the configuration for reproducibility.
