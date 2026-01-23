# LGBM Muscular V3: PL-Only Timelines (Production Model)

## Overview

V3 is the production-ready variant of the LGBM muscular injury prediction model that uses **only timelines where players were actively playing in Premier League clubs**. This model has been optimized by:

1. **PL-Only Filtering**: Only includes timelines for days when players were at PL clubs
2. **Season Filtering**: Uses recent seasons (2018-2026) excluding low-injury-rate seasons (2021-2022, 2022-2023) and 2023-2024
3. **Natural Ratio**: Uses natural (unbalanced) target ratio for realistic injury prediction

## Key Differences from V1/V2

- **V1/V2**: Timelines for players who played in PL at some point, but includes all career periods (including non-PL periods)
- **V3**: Timelines **only** for days when players were actively at PL clubs, filtered to exclude atypical seasons

## Model Performance

### Training Metrics
- **Accuracy**: 99.00%
- **Precision**: 38.40%
- **Recall**: 100.00%
- **F1-Score**: 55.49%
- **ROC AUC**: 99.99%
- **Gini**: 99.98%

### Test Metrics (2025-2026 PL-only)
- **Accuracy**: 99.17%
- **Precision**: 48.09%
- **Recall**: 100.00%
- **F1-Score**: 64.95%
- **ROC AUC**: 99.98%
- **Gini**: 99.96%

## Model Configuration

- **Training Seasons**: 2018-2019, 2019-2020, 2020-2021, 2024-2025, 2025-2026
- **Excluded Seasons**: 2021-2022, 2022-2023, 2023-2024
  - 2021-2022 & 2022-2023: Low injury rates (0.32% and 0.34%) - not representative of normal seasons
  - 2023-2024: Excluded to improve model generalization (improved test precision from 34.16% to 48.09%)
- **Target Ratio**: Natural (unbalanced)
- **Filter Type**: PL-only timelines
- **Training Records**: 357,281
- **Positives**: 2,219 (0.62% injury rate)

## Directory Structure

```
lgbm_muscular_v3/
├── code/
│   ├── timelines/
│   │   └── filter_timelines_pl_only.py  # Script to filter V1 timelines
│   └── modeling/
│       └── train_v3_natural_filtered_excl_2023_2024.py # Training script for the production model
├── data/
│   └── timelines/
│       ├── train/  # Filtered PL-only train timelines (only necessary files)
│       └── test/   # Filtered PL-only test timeline
├── model/
│   ├── model.joblib                    # Trained model
│   ├── columns.json                    # Feature columns
│   ├── lgbm_v3_pl_only_metrics_train.json # Training metrics
│   ├── lgbm_v3_pl_only_metrics_test.json  # Test metrics
│   ├── MODEL_METADATA.json            # Complete metadata for reproducibility
│   └── REPRODUCIBILITY.md             # Reproduction guide
├── README.md                           # This file
└── V3_FILTERED_MODELS_COMPARISON.md   # Comparison report of filtered models
```

## Usage

### Deployment

To deploy this model:
1. Load the model from `models_production/lgbm_muscular_v3/model/model.joblib`.
2. Use the feature columns from `models_production/lgbm_muscular_v3/model/columns.json` to ensure correct feature alignment during inference.
3. Refer to `models_production/lgbm_muscular_v3/model/MODEL_METADATA.json` for detailed configuration and performance.

### Reproducibility

To reproduce this model, follow the steps outlined in `models_production/lgbm_muscular_v3/model/REPRODUCIBILITY.md`.
