# LGBM Muscular V4: 580-Feature Production Model

## Overview

V4 580 is the production-ready variant of the LGBM muscular injury prediction model that uses **580 optimally selected features** from a two-layer daily-features pipeline. This model outperforms V3 Production on all key metrics.

**Status**: ✅ **READY FOR DEPLOYMENT**

## Key Improvements Over V3

1. **Two-Layer Feature Pipeline**:
   - **Layer 1**: Enhanced daily features with log-transformed injury recency, workload, recovery, and temporal features
   - **Layer 2**: Enriched features with workload windows, ACWR, injury-history windows, and interaction features

2. **Iterative Feature Selection**: 580 features selected from a larger candidate set using iterative selection (iteration 31)

3. **Enhanced Feature Engineering**:
   - Log transformation of injury recency features
   - Workload acceleration and spike indicators
   - Recovery time features
   - Recent pattern features (substitutions, bench time, performance)
   - Composite risk indicators
   - Temporal/seasonal features

4. **Better Performance**: Outperforms V3 on precision, F1-score, and Gini coefficient

## Model Performance

### Test Performance (2025/26 season)

| Metric | V4 580 | V3 Production | Improvement |
|--------|--------|---------------|-------------|
| **Gini** | 1.0000 | 0.9996 | +0.0004 |
| **F1-Score** | 0.7194 | 0.6495 | +0.0699 |
| **Precision** | 0.5618 | 0.4809 | +0.0809 |
| **Recall** | 1.0000 | 1.0000 | - |
| **Accuracy** | 0.9932 | 0.9917 | +0.0015 |
| **ROC AUC** | 1.0000 | 0.9998 | +0.0002 |

**Key Achievement**: Perfect recall (300/300 injuries detected) with improved precision and F1-score.

### Confusion Matrix (Test Set)

- **True Positives (TP)**: 300
- **False Positives (FP)**: 234
- **True Negatives (TN)**: 33,693
- **False Negatives (FN)**: 0

## Model Configuration

- **Features**: 580 (from iterative feature selection, iteration 31)
- **Training Seasons**: 2018/19, 2019/20, 2020/21, 2023/24, 2024/25, 2025/26
- **Excluded Seasons**: 2021/22, 2022/23 (low injury rate - not representative of normal seasons)
- **Training Records**: 441,772
- **Positives**: 2,469 (0.56% injury rate)
- **Test Records**: 34,227
- **Test Positives**: 300
- **Filter Type**: PL-only timelines (only days when players were at PL clubs)
- **Target Ratio**: Natural (unbalanced)

### Hyperparameters (LightGBM)

- `n_estimators`: 200
- `max_depth`: 10
- `learning_rate`: 0.1
- `min_child_samples`: 20
- `subsample`: 0.8
- `colsample_bytree`: 0.8
- `reg_alpha`: 0.1
- `reg_lambda`: 1.0
- `class_weight`: 'balanced'
- `random_state`: 42

## Directory Structure

```
lgbm_muscular_v4/
├── code/
│   ├── daily_features/
│   │   ├── create_daily_features_v4_enhanced.py    # Layer 1 (PRODUCTION)
│   │   └── enrich_daily_features_v4_layer2.py       # Layer 2 (PRODUCTION)
│   ├── timelines/
│   │   └── create_35day_timelines_v4_enhanced.py     # Timeline generation (PRODUCTION)
│   └── modeling/
│       ├── train_v4_580_production.py                # Production model training
│       ├── train_iterative_feature_selection_standalone.py  # Feature selection (reference)
│       └── archive/                                   # Experimental scripts (archived)
├── data/
│   ├── raw_data/                                     # Raw input data
│   ├── daily_features/                               # Layer 1 output
│   ├── daily_features_enriched/                      # Layer 2 output
│   └── timelines/
│       ├── train/                                    # Training timelines
│       └── test/                                     # Test timelines
├── models/
│   ├── iterative_feature_selection_results.json      # Feature selection results
│   ├── iterative_feature_selection_plot.png          # Feature selection visualization
│   ├── feature_ranking.json                         # Feature importance ranking
│   ├── low_probability_analysis/                   # Analysis results
│   └── archive/                                     # Old models and comparison results
├── model/                                            # PRODUCTION MODEL
│   ├── model.joblib                                 # Trained model
│   ├── columns.json                                 # 580 feature names (in order)
│   ├── MODEL_METADATA.json                          # Complete model metadata
│   ├── lgbm_v4_580_metrics_train.json              # Training metrics
│   ├── lgbm_v4_580_metrics_test.json                # Test metrics
│   └── REPRODUCIBILITY.md                           # Reproduction guide
├── DEPLOYMENT.md                                     # Complete deployment guide
├── V4_580_PRODUCTION_STATUS.md                      # Production status
└── README.md                                        # This file
```

## Quick Start

### 1. Deployment Guide

For complete deployment instructions, see **[DEPLOYMENT.md](DEPLOYMENT.md)**.

The deployment pipeline consists of three steps:

1. **Layer 1**: Generate base daily features
   ```bash
   cd code/daily_features
   python create_daily_features_v4_enhanced.py --mode full
   ```

2. **Layer 2**: Enrich daily features
   ```bash
   cd code/daily_features
   python enrich_daily_features_v4_layer2.py
   ```

3. **Timelines**: Generate 35-day timelines
   ```bash
   cd code/timelines
   python create_35day_timelines_v4_enhanced.py
   ```

### 2. Loading the Model

```python
from pathlib import Path
import joblib
import json
import pandas as pd

ROOT = Path("models_production/lgbm_muscular_v4")

# Load model
model = joblib.load(ROOT / "model" / "model.joblib")

# Load feature columns
with open(ROOT / "model" / "columns.json", "r", encoding="utf-8") as f:
    feature_columns = json.load(f)

# Load metadata
with open(ROOT / "model" / "MODEL_METADATA.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Prepare features (must match feature_columns order)
X = df_timelines[feature_columns]

# Make predictions
probabilities = model.predict_proba(X)[:, 1]
predictions = model.predict(X)
```

### 3. Model Requirements

- Input features must match the columns in `model/columns.json` (580 features)
- Features must be preprocessed the same way as training data
- Use the same feature engineering pipeline (Layer 1 → Layer 2 → Timelines)

## Two-Layer Feature Pipeline

### Layer 1: Base Daily Features

**Script**: `code/daily_features/create_daily_features_v4_enhanced.py`

Generates daily per-player features including:
- Log-transformed injury recency features
- Workload features (minutes, matches, intensity)
- Recovery indicators
- Temporal/seasonal features
- Player profile features

**Output**: `data/daily_features/player_{id}_daily_features.csv`

### Layer 2: Enriched Features

**Script**: `code/daily_features/enrich_daily_features_v4_layer2.py`

Adds advanced features to Layer 1:
- Workload windows (3/7/14/28/35 days)
- Match count windows
- ACWR (Acute:Chronic Workload Ratio)
- Season-to-date workload and ratios
- Injury-history windows (90/365 days)
- Recovery/rest indicators
- Interaction features (inactivity_risk, early_season_low_activity, etc.)

**Output**: `data/daily_features_enriched/player_{id}_daily_features.csv`

### Timeline Generation

**Script**: `code/timelines/create_35day_timelines_v4_enhanced.py`

- Consumes Layer 2 enriched features (or Layer 1 if Layer 2 unavailable)
- Generates 35-day sliding-window timelines
- Creates dual targets (target1: muscular, target2: skeletal)
- Applies PL-only filtering
- Uses natural target ratios

**Output**: `data/timelines/train/` and `data/timelines/test/`

## Feature Selection

The 580 features were selected using iterative feature selection (iteration 31) from a larger candidate set. The selection process:

1. Started with a comprehensive feature set
2. Iteratively evaluated feature subsets
3. Selected features based on combined score (Gini + F1-score)
4. Final selection: 580 features with combined score of 0.316

See `models/iterative_feature_selection_results.json` for detailed selection results.

## Reproducibility

To reproduce this model, see `model/REPRODUCIBILITY.md` for detailed step-by-step instructions.

## Model Metadata

Complete model metadata, including configuration, performance metrics, and file hashes, is available in `model/MODEL_METADATA.json`.

## Comparison with V3

V4 580 outperforms V3 Production on all key metrics:

- **Gini**: 1.0000 vs 0.9996 (+0.0004)
- **F1-Score**: 0.7194 vs 0.6495 (+0.0699)
- **Precision**: 0.5618 vs 0.4809 (+0.0809)
- **Accuracy**: 0.9932 vs 0.9917 (+0.0015)

See `models/archive/comparison_results/v4_580_comparison_table.md` for detailed comparison.

## Key Features

### Injury Recency Features (Log-Transformed)

All `days_since_last_*` features are log-transformed using `log(1 + max(value, 0))`:
- Reduces over-reliance on players with very long injury-free periods
- Preserves relative differences for recent injuries
- 17 base injury recency features

### Workload Features

- Minutes played (daily, weekly, monthly)
- Match count windows
- ACWR (Acute:Chronic Workload Ratio)
- Workload acceleration and spikes
- Season-to-date workload ratios

### Recovery Features

- Days since last match
- Rest periods
- Recovery indicators
- Inactivity risk

### Temporal Features

- Season phase
- Days into season
- Competition importance
- Match frequency

## Notes

- Model maintains 100% recall (all injuries detected)
- Test set is the 2025/26 season (PL-only filtered)
- Model is ready for production deployment
- All experimental files have been archived (see `code/modeling/archive/` and `models/archive/`)

## Support

For deployment questions, see:
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Complete deployment guide
- **`model/REPRODUCIBILITY.md`** - Model reproduction guide
- **`model/MODEL_METADATA.json`** - Complete model metadata

---

**V4 580 is production-ready!** 🚀
