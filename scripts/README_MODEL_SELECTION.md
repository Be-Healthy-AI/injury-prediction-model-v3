# Model Selection and Hyperparameter Tuning System V5

## Overview

This system provides a comprehensive pipeline for model selection and hyperparameter optimization using multiple feature selection strategies and multiple algorithms.

## Execution Order

1. **Feature Selection** (`feature_selection_v5.py`)
   - Generates multiple feature sets using different strategies
   - Evaluates each feature set with baseline Random Forest
   - Outputs: `feature_sets/*.json` and `analysis/feature_selection_results.json`

2. **Model Training** (`run_model_selection_parallel.py`)
   - Trains all model/feature set combinations in parallel
   - Uses Optuna for hyperparameter optimization
   - Outputs: `models/optimized/*.joblib`, `*_params.json`, `*_metrics.json`

3. **Comparison Analysis** (`compare_optimized_models.py`)
   - Analyzes and ranks all trained models
   - Generates comprehensive report
   - Outputs: `analysis/model_comparison_results.csv` and `analysis/model_selection_report_v5.md`

4. **Ensemble Creation** (Optional) (`create_ensemble_v5.py`)
   - Combines top N models into ensemble
   - Tests Voting and Stacking classifiers
   - Outputs: `models/optimized/ensemble_*.joblib`

## Quick Start

```bash
# Step 1: Feature Selection
python scripts/feature_selection_v5.py

# Step 2: Train Models (parallel)
python scripts/run_model_selection_parallel.py

# Step 3: Compare Results
python scripts/compare_optimized_models.py

# Step 4: Create Ensemble (optional)
python scripts/create_ensemble_v5.py
```

## Configuration

Edit `config/model_selection_config.json` to customize:
- Feature selection parameters
- Optuna optimization settings
- Model hyperparameter search spaces
- Parallel execution settings

## Output Files

- **Feature Sets**: `feature_sets/*.json`
- **Trained Models**: `models/optimized/*.joblib`
- **Metrics**: `models/optimized/*_metrics.json`
- **Comparison Report**: `analysis/model_selection_report_v5.md`
- **CSV Results**: `analysis/model_comparison_results.csv`

## Notes

- The system automatically detects available models (XGBoost, LightGBM, CatBoost)
- Models are trained with focus on out-of-sample F1-Score
- All results are saved for resumability
- Parallel execution uses all available CPU cores by default


