# Notebook Usage Guide

## Overview

This notebook (`superconductor-deepset.ipynb`) contains the complete implementation from the research paper. It has been adapted from the original Kaggle environment to work in VS Code or any local Jupyter environment.

## Setup Instructions

### 1. Dataset Preparation

The notebook requires the SuperCon dataset. Place your dataset file in the parent `data/` directory:

```
superconductor-deepset/
├── data/
│   └── superconductors_kaggle_ready_v2.csv  ← Place your dataset here
├── notebooks/
│   └── superconductor-deepset.ipynb         ← This notebook
└── ...
```

**Expected dataset format:**
- Columns: `element` (chemical composition), `critical_temp_K` (target Tc)
- Example rows: `YBa2Cu3O7, 92.0` or `MgB2, 39.0`

### 2. Installation

The notebook has been updated to work in local environments. Install dependencies using:

**Option 1: From requirements file** (recommended)
```bash
pip install -r ../requirements.txt
```

**Option 2: Direct installation** (minimal packages)
```bash
pip install mendeleev shap mlflow "numpy>=1.26.0" "protobuf>=5.28.0"
```

## Notebook Structure

### Cell 1: Package Installation
- Installs required packages (mendeleev, shap, mlflow, etc.)
- Can skip if already installed via requirements.txt

### Cell 2: Data Loading & Model Training
- **Updated for local environment**: Dataset path changed from `/kaggle/input/...` to `../data/superconductors_kaggle_ready_v2.csv`
- Loads and preprocesses data
- Extracts 22 Mendeleev features + stoichiometry
- Trains 50 independent DeepSet models
- Saves models to `../saved_models/`
- Training takes ~2-4 hours depending on hardware

### Cell 3: Ensemble Predictions
- Loads all 50 trained models
- Makes predictions on Hosono dataset test compositions (909 compounds)
- Provides mean ± std predictions with confidence intervals
- Generates visualization plots
- Saves results to `ensemble_predictions.csv`

**Note**: Test compositions are hardcoded in this cell. To use your own compositions:
```python
# Option 1: Load from file
with open('../data/test_compositions.txt', 'r') as f:
    new_compositions = [line.strip() for line in f if line.strip()]

# Option 2: Define your own list
new_compositions = ["La3Ni2O7", "YBa2Cu3O7", "MgB2", ...]
```

### Cell 4: SHAP Analysis
- Performs interpretability analysis on top 5 models
- Identifies most important features
- Generates SHAP plots
- Saves feature importance results

## Key Differences from Kaggle Version

| Aspect | Kaggle Version | Local Version |
|--------|---------------|---------------|
| Dataset path | `/kaggle/input/dataset1/...` | `../data/...` |
| Model saving | `/kaggle/working/saved_models/` | `../saved_models/` |
| Results saving | `/kaggle/working/` | `../results/` |
| Package installation | Kaggle pre-installed | Manual via pip |

## Expected Outputs

After running the notebook, you'll have:

```
superconductor-deepset/
├── saved_models/
│   ├── model_run_1.keras
│   ├── scaler_run_1.pkl
│   ├── model_run_2.keras
│   ├── scaler_run_2.pkl
│   └── ... (100 files total: 50 models + 50 scalers)
├── results/
│   ├── training_results.csv
│   ├── ensemble_predictions.csv
│   ├── ensemble_predictions_summary.png
│   └── shap_feature_importance.csv
└── mlruns/
    └── (MLflow experiment tracking data)
```

## Performance Expectations

**Training (Cell 2):**
- Time: ~2-4 hours for 50 models (depends on CPU/GPU)
- Expected results: R² ≈ 0.92, RMSE ≈ 9-10 K

**Prediction (Cell 3):**
- Time: ~5-10 minutes for 909 compositions
- Output: Mean Tc ± uncertainty for each composition

**SHAP Analysis (Cell 4):**
- Time: ~10-20 minutes
- Top features: Stoichiometry, thermal conductivity, electron affinity

## Troubleshooting

### Dataset Not Found Error
```
FileNotFoundError: Dataset not found at ../data/superconductors_kaggle_ready_v2.csv
```
**Solution**: Place your dataset file in the `data/` directory

### Memory Error During Training
**Solution**: Reduce number of runs:
```python
num_runs = 10  # Instead of 50
```

### SHAP Numerical Instability
The notebook includes numerical clipping at 99.9th percentile to prevent extreme SHAP values. This is already implemented.

### GPU Not Detected
TensorFlow will automatically use CPU if GPU is not available. Training will be slower but results will be identical.

## Running Individual Sections

You can run cells independently:
- **Just training**: Run cells 1-2
- **Just prediction** (if models exist): Run cells 1, 3
- **Just SHAP** (if models exist): Run cells 1, 4

## Alternative: Use Command-Line Scripts

Instead of the notebook, you can use the modular scripts in `../scripts/`:

```bash
# Training
python scripts/train.py --config config/config.yaml

# Prediction
python scripts/predict.py --input data/test_compositions.txt

# SHAP analysis
python scripts/analyze_shap.py --config config/config.yaml
```

## Citation

If you use this code, please cite the paper:
> "From Individual Elements to Macroscopic Materials: In Search of New Superconductors via Machine Learning"  
> Centre for Computational Natural Sciences and Bioinformatics, IIIT Hyderabad, 2026

## Support

For issues or questions:
- GitHub: https://github.com/Tejasvini595/superconductor_deepset
- Check the main `README.md` for detailed documentation
