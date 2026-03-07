# V4 580 Model Reproducibility Guide

This document provides step-by-step instructions to reproduce the V4 580 (With Test, Excl 2021/22-2022/23) production model.

## Model Information

- **Version**: V4_580_with_test_excl_2021_2022_2022_2023
- **Training Date**: 2026-01-23 09:28:48
- **Configuration**: PL-only timelines, 580 features, seasons 2018-2026 excluding 2021-2022 and 2022-2023
- **Feature Selection**: Iterative feature selection, iteration 31, 580 features

## Prerequisites

### Software Dependencies

- Python 3.8+
- Required packages:
  - `lightgbm >= 3.0.0`
  - `pandas >= 1.5.0`
  - `numpy >= 1.20.0`
  - `scikit-learn >= 1.0.0`
  - `joblib >= 1.0.0`
  - `tqdm >= 4.65.0`

### Data Dependencies

1. **Raw Data** (in `data/raw_data/`):
   - `players_profile.csv`
   - `players_career.csv`
   - `injuries_data.csv`
   - `teams_data.csv`
   - `match_data/match_*.csv`

2. **Feature Selection Results**:
   - `models/iterative_feature_selection_results.json` (contains the 580 optimal features)

## Step-by-Step Reproduction

### Step 1: Generate Layer 1 Daily Features

Run the Layer 1 daily features generation script:

```bash
cd models_production/lgbm_muscular_v4/code/daily_features
python create_daily_features_v4_enhanced.py --mode full --log-level INFO
```

**What it does**:
- Reads raw data from `data/raw_data/`
- Generates daily per-player feature series with V4 enhanced feature set
- Applies log-transformed injury recency, workload, recovery, and temporal features
- Writes one CSV per player to `data/daily_features/`

**Expected Output**: 
- Files: `data/daily_features/player_{player_id}_daily_features.csv`
- One file per player (excluding goalkeepers)

**Important**: Use the same configuration (modes/flags) used during the successful V4 580 run to ensure feature parity.

### Step 2: Enrich Daily Features (Layer 2)

Run the Layer 2 enrichment script:

```bash
cd models_production/lgbm_muscular_v4/code/daily_features
python enrich_daily_features_v4_layer2.py
```

**What it does**:
- Reads Layer 1 daily features from `data/daily_features/`
- Adds Layer 2 features:
  - Workload windows (3/7/14/28/35 days)
  - Match count windows
  - ACWR (`acwr_min_7_28`)
  - Season-to-date workload, ratios to season totals
  - Injury-history windows (90/365 days)
  - Recovery/rest indicators
  - Interaction features (inactivity_risk, early_season_low_activity, etc.)
- Writes enriched CSVs to `data/daily_features_enriched/`

**Expected Output**:
- Files: `data/daily_features_enriched/player_{player_id}_daily_features.csv`
- Same file names as Layer 1, but with additional features

**Note**: The script is restart-friendly and skips files that are already enriched by default.

### Step 3: Generate 35-Day Timelines

Run the timeline generation script:

```bash
cd models_production/lgbm_muscular_v4/code/timelines
python create_35day_timelines_v4_enhanced.py
```

**What it does**:
- Consumes V4 daily features (prefers Layer 2 enriched if available)
- Generates 35-day sliding-window timelines for each player/season
- Builds dual targets (target1: muscular, target2: skeletal)
- Applies PL-only filtering using career and match data
- Uses natural target ratios (no downsampling)
- Creates 5 injury-timelines per injury (D-1 to D-5)
- Splits by season:
  - **Train**: seasons ≤ 2024/25
  - **Test**: 2025/26

**Expected Output**:
- `data/timelines/train/timelines_35day_season_YYYY_YYYY+1_v4_muscular_train.csv`
- `data/timelines/test/timelines_35day_season_2025_2026_v4_muscular_test.csv`

**Path Selection**:
- Prefers `data/daily_features_enriched/` when present
- Falls back to `data/daily_features/` if Layer 2 is missing

### Step 4: Train the Production Model

Run the production training script:

```bash
cd models_production/lgbm_muscular_v4/code/modeling
python train_v4_580_production.py
```

**What it does**:
- Loads 580 optimal features from `models/iterative_feature_selection_results.json`
- Loads training data (seasons 2018/19-2025/26, excluding 2021/22-2022/23)
- Loads test dataset (2025/26)
- Prepares data (categorical encoding, feature alignment, filtering)
- Trains LightGBM model with production hyperparameters
- Saves model, columns, metadata, and metrics to `model/` directory

**Expected Output**:
- `model/model.joblib` - Trained model
- `model/columns.json` - 580 feature names (in order)
- `model/MODEL_METADATA.json` - Complete model metadata
- `model/lgbm_v4_580_metrics_train.json` - Training metrics
- `model/lgbm_v4_580_metrics_test.json` - Test metrics
- `model/train_v4_580_production.log` - Training log

## Model Configuration Summary

### Training Configuration

- **Training Seasons**: 2018/19, 2019/20, 2020/21, 2023/24, 2024/25, 2025/26
- **Excluded Seasons**: 2021/22, 2022/23
  - Reason: Low injury rates (not representative of normal seasons)
- **Test Season**: 2025/26
- **Filter Type**: PL-only (only days when players were at PL clubs)
- **Target Ratio**: Natural (unbalanced)
- **Training Records**: 441,772
- **Training Positives**: 2,469 (0.56% injury rate)
- **Test Records**: 34,227
- **Test Positives**: 300

### Feature Selection

- **Method**: Iterative feature selection
- **Iteration**: 31
- **Number of Features**: 580
- **Combined Score**: 0.316
- **Source**: `models/iterative_feature_selection_results.json`

### Hyperparameters

```python
LGBMClassifier(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
```

## Expected Performance

### Training Metrics
- **Accuracy**: ~99.24%
- **Precision**: ~40.07%
- **Recall**: 100.00%
- **F1-Score**: ~57.22%
- **ROC AUC**: ~99.99%
- **Gini**: ~99.99%

### Test Metrics (2025/26)
- **Accuracy**: 99.32%
- **Precision**: 56.18%
- **Recall**: 100.00%
- **F1-Score**: 71.94%
- **ROC AUC**: 100.00%
- **Gini**: 100.00%

## Verification

After reproduction, verify:

1. **Model Files**: Check that all files in `model/` directory are present
2. **Feature Count**: Verify `columns.json` contains exactly 580 features
3. **Performance**: Compare metrics in `lgbm_v4_580_metrics_test.json` with expected values above
4. **Metadata**: Check `MODEL_METADATA.json` matches the configuration above

## Troubleshooting

### Issue: Feature Mismatch

**Symptom**: Error about missing features or feature count mismatch

**Solution**:
- Ensure you're using the exact same feature selection results (`iterative_feature_selection_results.json`)
- Verify Layer 1 and Layer 2 scripts match the production versions
- Check that timeline generation uses enriched features

### Issue: Different Performance Metrics

**Symptom**: Metrics differ from expected values

**Possible Causes**:
- Different random seed (should be 42)
- Different data preprocessing
- Different feature selection results
- Missing or incorrect data files

**Solution**:
- Verify `random_state=42` in training script
- Check that all raw data files are present and correct
- Ensure feature selection results match production

### Issue: Missing Data Files

**Symptom**: FileNotFoundError for raw data files

**Solution**:
- Ensure all required files are in `data/raw_data/`
- Check file names match expected patterns
- Verify file paths in scripts

## Notes

- The model uses caching for preprocessed data (see `train_v4_580_production.py` for cache locations)
- Training may take 10-30 minutes depending on hardware
- The script automatically aligns features between train and test sets
- Feature selection results are loaded from `models/iterative_feature_selection_results.json`

## Additional Resources

- **DEPLOYMENT.md**: Complete deployment guide with detailed script usage
- **MODEL_METADATA.json**: Complete model metadata and configuration
- **V4_580_PRODUCTION_STATUS.md**: Production status and summary

---

**For questions or issues, refer to the main README.md or DEPLOYMENT.md.**
