# LGBM V3 Training Summary

## Training Date
2026-01-04

## Dataset Configuration

- **Data Source**: PL-only filtered timelines from V1
- **Target Ratio**: 10% (balanced)
- **Seasons Included**: All seasons including 2025-2026
- **Filter Applied**: Only timelines where players were at PL clubs
- **Total Records**: 32,008
- **Injury Ratio**: 9.70% (3,104 positives, 28,904 negatives)

## Data Loading

### Seasons Loaded (10% ratio files)
- 2011-2012: 36 records (5 positives)
- 2012-2013: 122 records (10 positives)
- 2013-2014: 187 records (25 positives)
- 2014-2015: 358 records (60 positives)
- 2015-2016: 638 records (90 positives)
- 2016-2017: 961 records (110 positives)
- 2017-2018: 1,382 records (110 positives)
- 2018-2019: 2,767 records (345 positives)
- 2019-2020: 3,293 records (463 positives)
- 2020-2021: 4,387 records (448 positives)
- 2021-2022: 3,912 records (310 positives)
- 2022-2023: 4,081 records (390 positives)
- 2023-2024: 4,531 records (340 positives)
- 2024-2025: 4,476 records (318 positives)
- 2025-2026: 877 records (80 positives)

**Note**: Some early seasons (2008-2010) had 10% files but with 0 positives, so were skipped.

## Feature Engineering

- **Initial Features**: 1,629 (after one-hot encoding)
- **After Correlation Filtering (0.8 threshold)**: 1,324 features
- **Features Removed**: 305 (18.7% reduction)

## Model Configuration

- **Algorithm**: LightGBM
- **Hyperparameters** (same as v1/v2):
  - n_estimators: 200
  - max_depth: 10
  - learning_rate: 0.1
  - min_child_samples: 20
  - subsample: 0.8
  - colsample_bytree: 0.8
  - reg_alpha: 0.1
  - reg_lambda: 1.0
  - class_weight: "balanced"
  - random_state: 42

## Training Results (on Full Training Data)

### Classic Metrics
- **Accuracy**: 98.54%
- **Precision**: 86.90%
- **Recall**: 100.00%
- **F1-Score**: 92.99%
- **ROC AUC**: 99.99%
- **Gini**: 99.98%

### Confusion Matrix
- **True Positives (TP)**: 3,104
- **False Positives (FP)**: 468
- **True Negatives (TN)**: 28,436
- **False Negatives (FN)**: 0

## Training Time

- **Data Preparation**: ~15.5 seconds
- **Correlation Filtering**: ~2.4 minutes
- **Model Training**: ~2.6 seconds
- **Total Time**: ~2.7 minutes

## Key Observations

1. **Excellent Performance**: 99.98% Gini and 100% Recall on training data
2. **High Precision**: 86.90% precision indicates good specificity
3. **Perfect Recall**: 100% recall means all injuries are detected (on training data)
4. **Smaller Dataset**: 32K records vs ~4M in V1/V2 (99.2% reduction)
5. **Focused Context**: PL-only data provides more homogeneous training context

## Model Artifacts

Saved to `models_production/lgbm_muscular_v3/model/`:
- `model.joblib` - Trained LightGBM model
- `columns.json` - Feature column names
- `lgbm_v3_pl_only_metrics_train.json` - Training metrics

## Next Steps

1. ✅ Filter timelines (COMPLETE)
2. ✅ Train V3 model (COMPLETE)
3. ⏳ Evaluate on test set (2025-2026 PL-only timeline)
4. ⏳ Compare with V1/V2 metrics
5. ⏳ Create V3 bundle (metadata, docs, code snapshots)

## Notes

- Training metrics are on the full training dataset (no hold-out)
- Test evaluation should be performed on the filtered 2025-2026 PL-only timeline
- Compare with V1/V2 to assess impact of PL-only filtering on model performance




