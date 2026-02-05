# Quick Start Guide

## Installation

1. **Clone the repository** (or navigate to the project directory)
```bash
cd superconductor-deepset
```

2. **Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Data Setup

Place your dataset in the `data/` directory:
```
data/superconductors_kaggle_ready_v2.csv
```

The dataset should have columns:
- `element` (or your composition column name)
- `critical_temp_K` (or your target column name)

## Training Models

Train 50 independent models:
```bash
python scripts/train.py --config config/config.yaml
```

Options:
- `--num-runs N`: Train N models instead of 50
- `--verbose 0`: Silent mode (no progress bars)
- `--verbose 2`: Detailed output (one line per epoch)

**Expected time**: ~2-4 hours for 50 models (depends on dataset size and hardware)

## Making Predictions

### Predict from command line:
```bash
python scripts/predict.py --compositions "YBa2Cu3O7" "MgB2" "NbTi"
```

### Predict from file:
```bash
python scripts/predict.py --input data/test_compositions.txt --output results/predictions.csv
```

### With plots:
```bash
python scripts/predict.py --input data/test_compositions.txt --plot
```

## SHAP Analysis

Run interpretability analysis:
```bash
python scripts/analyze_shap.py --config config/config.yaml
```

Results will be saved to `results/shap/`

## Directory Structure After Training

```
superconductor-deepset/
├── saved_models/
│   ├── deepset_model_run_1.h5
│   ├── scaler_run_1.pkl
│   ├── ...
│   ├── element_features_dict.pkl
│   └── all_runs_metrics.csv
├── results/
│   ├── predictions.csv
│   ├── figures/
│   └── shap/
└── logs/
    └── superconductor_deepset.log
```

## Customization

Edit `config/config.yaml` to customize:
- Model architecture (layer sizes, latent dimension)
- Training hyperparameters (batch size, learning rate, epochs)
- Data paths and column names
- SHAP analysis parameters

## Troubleshooting

**Q: Training is very slow**
- Reduce `epochs` in config (try 100 instead of 400)
- Reduce `num_runs` (try 10 instead of 50)
- Use a smaller dataset

**Q: Out of memory errors**
- Reduce `batch_size` in config (try 32 instead of 64)
- Use fewer models for ensemble prediction

**Q: SHAP analysis takes too long**
- Reduce `num_background_samples` and `num_explain_samples` in config
- Use fewer top models (`--top-models 3`)

## Example Workflow

```bash
# 1. Train models (takes a while)
python scripts/train.py --num-runs 10  # Start with 10 for testing

# 2. Check metrics
cat saved_models/all_runs_metrics.csv

# 3. Make predictions
python scripts/predict.py --input data/test_compositions.txt --plot

# 4. Run SHAP analysis
python scripts/analyze_shap.py --top-models 3

# 5. Review results
ls results/
```

## Citation

If you use this code, please cite:
```
[Your paper citation here]
```
