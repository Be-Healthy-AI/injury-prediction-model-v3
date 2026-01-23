# V3 Model Reproducibility Guide

This document provides step-by-step instructions to reproduce the V3_natural_filtered_excl_2023_2024 model.

## Model Information

- **Version**: V3_natural_filtered_excl_2023_2024
- **Training Date**: 2026-01-06 13:53:30
- **Configuration**: PL-only timelines, seasons 2018-2026 excluding 2021-2022, 2022-2023, and 2023-2024

## Prerequisites

### Software Dependencies

- Python 3.8+
- Required packages (see requirements.txt or install individually):
  - lightgbm >= 3.0.0
  - pandas >= 1.5.0
  - numpy >= 1.20.0
  - scikit-learn >= 1.0.0
  - joblib >= 1.0.0

### Data Dependencies

1. **V1 Timelines**: Original timelines from `models_production/lgbm_muscular_v1/data/timelines/`
2. **Raw Match Data**: For PL club identification (from V1 raw data)
3. **Career Data**: Player career transfers (from V1 raw data)

## Step-by-Step Reproduction

### Step 1: Filter V1 Timelines to PL-Only

Run the filtering script to generate PL-only timelines:

```bash
python models_production/lgbm_muscular_v3/code/timelines/filter_timelines_pl_only.py
```

This script:
- Reads V1 timelines from `models_production/lgbm_muscular_v1/data/timelines/`
- Identifies PL clubs per season from raw match data
- Determines PL membership periods for each player from career data
- Filters timelines to only include days when players were at PL clubs
- Saves filtered timelines to `models_production/lgbm_muscular_v3/data/timelines/`

**Expected Output**: Filtered timeline files in train/ and test/ directories

### Step 2: Train the Model

Run the training script:

```bash
python models_production/lgbm_muscular_v3/code/modeling/train_v3_natural_filtered_excl_2023_2024.py
```

This script:
- Loads natural ratio timeline files for seasons 2018-2019, 2019-2020, 2020-2021, 2024-2025, 2025-2026
- Excludes seasons 2021-2022, 2022-2023, and 2023-2024
- Prepares data (categorical encoding, feature engineering)
- Applies correlation filtering (threshold=0.8)
- Trains the LightGBM model
- Saves the trained model, feature columns, and training metrics to `models_production/lgbm_muscular_v3/model_natural_filtered_excl_2023_2024/`

**Expected Output**: Trained model artifacts in `model_natural_filtered_excl_2023_2024/` folder.

### Step 3: Evaluate the Model

Run the evaluation script:

```bash
python models_production/lgbm_muscular_v3/code/modeling/evaluate_and_compare_filtered_models.py
```

**Expected Output**: Performance metrics on the test set.

## Model Configuration Summary

- **Training Seasons**: 2018-2019, 2019-2020, 2020-2021, 2024-2025, 2025-2026
- **Excluded Seasons**: 2021-2022, 2022-2023, 2023-2024
- **Target Ratio**: Natural (unbalanced)
- **Filter Type**: PL-only timelines
- **Training Records**: 357,281
- **Positives**: 2,219 (0.62% injury rate)

## Expected Performance

Based on the training and evaluation:

- **Training Metrics**:
  - Accuracy: 99.00%
  - Precision: 38.40%
  - Recall: 100.00%
  - F1-Score: 55.49%
  - ROC AUC: 99.99%
  - Gini: 99.98%

- **Test Metrics (2025-2026 PL-only)**:
  - Accuracy: 99.17%
  - Precision: 48.09%
  - Recall: 100.00%
  - F1-Score: 64.95%
  - ROC AUC: 99.98%
  - Gini: 99.96%

## Verification

To verify the model matches the production version:

1. Check model hash in MODEL_METADATA.json
2. Compare training metrics with expected values above
3. Compare test metrics with expected values above
4. Verify feature count matches (check columns.json)

## Notes

- The model uses a fixed random_state (42) for reproducibility
- Correlation filtering threshold is 0.8
- All hyperparameters are documented in MODEL_METADATA.json
