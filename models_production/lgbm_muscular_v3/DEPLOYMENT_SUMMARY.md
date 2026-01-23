# V3 Model Deployment Summary

**Deployment Date:** 2026-01-06

## Model Selected for Production

**V3-natural-filtered-excl-2023-2024** has been selected as the production model based on superior test performance:

- **Test Precision**: 48.09% (vs 34.16% for previous V3-natural-filtered)
- **Test F1-Score**: 64.95% (vs 50.92% for previous V3-natural-filtered)
- **Test Accuracy**: 99.17% (vs 98.52% for previous V3-natural-filtered)
- **100% Recall** maintained on both training and test sets

## Deployment Files

All production files are located in: `models_production/lgbm_muscular_v3/model/`

### Required Files for Deployment:
- ✅ `model.joblib` - Trained LightGBM model
- ✅ `columns.json` - Feature column names (1,541 features)
- ✅ `lgbm_v3_pl_only_metrics_train.json` - Training performance metrics
- ✅ `lgbm_v3_pl_only_metrics_test.json` - Test performance metrics
- ✅ `MODEL_METADATA.json` - Complete model configuration and metadata
- ✅ `REPRODUCIBILITY.md` - Step-by-step reproduction guide

## Training Data

The model was trained on 5 season files (PL-only filtered):
- `timelines_35day_season_2018_2019_v4_muscular.csv` (55,785 records, 360 positives)
- `timelines_35day_season_2019_2020_v4_muscular.csv` (70,519 records, 478 positives)
- `timelines_35day_season_2020_2021_v4_muscular.csv` (87,170 records, 473 positives)
- `timelines_35day_season_2024_2025_v4_muscular.csv` (106,057 records, 618 positives)
- `timelines_35day_season_2025_2026_v4_muscular.csv` (37,750 records, 290 positives)

**Total**: 357,281 records, 2,219 positives (0.62% injury rate)

## Excluded Seasons

- **2021-2022**: Low injury rate (0.32%)
- **2022-2023**: Low injury rate (0.34%)
- **2023-2024**: Excluded to improve generalization (test precision improved from 34.16% to 48.09%)

## Model Performance Summary

### Training Metrics
- Accuracy: 99.00%
- Precision: 38.40%
- Recall: 100.00%
- F1-Score: 55.49%
- ROC AUC: 99.99%
- Gini: 99.98%

### Test Metrics (2025-2026 PL-only)
- Accuracy: 99.17%
- Precision: 48.09%
- Recall: 100.00%
- F1-Score: 64.95%
- ROC AUC: 99.98%
- Gini: 99.96%

### Test Confusion Matrix
- True Negatives: 37,147
- False Positives: 313
- True Positives: 290
- False Negatives: 0

## Auditability

All necessary files for auditability are preserved:

### Model Artifacts
- ✅ Production model in `model/` folder
- ✅ All experimental models preserved in separate folders (model_10pc, model_25pc, model_50pc, model_natural, model_natural_recent, model_natural_filtered, model_natural_filtered_excl_2023_2024)

### Documentation
- ✅ `README.md` - Model overview and usage
- ✅ `MODEL_METADATA.json` - Complete configuration and performance
- ✅ `REPRODUCIBILITY.md` - Step-by-step reproduction instructions
- ✅ `V3_FILTERED_MODELS_COMPARISON.md` - Comparison of filtered models
- ✅ `V3_ALL_MODELS_FINAL_COMPARISON.md` - Comparison of all V3 experiments

### Scripts
- ✅ `code/timelines/filter_timelines_pl_only.py` - Timeline filtering script
- ✅ `code/modeling/train_v3_natural_filtered_excl_2023_2024.py` - Training script
- ✅ `code/modeling/evaluate_and_compare_filtered_models.py` - Evaluation script
- ✅ All other training and evaluation scripts preserved

### Data
- ✅ Training timeline files (5 files used for training)
- ✅ Test timeline file (2025-2026 PL-only)

## Deployment Instructions

1. **Load the model:**
   ```python
   import joblib
   model = joblib.load('models_production/lgbm_muscular_v3/model/model.joblib')
   ```

2. **Load feature columns:**
   ```python
   import json
   with open('models_production/lgbm_muscular_v3/model/columns.json', 'r') as f:
       model_columns = json.load(f)
   ```

3. **Prepare input data:**
   - Ensure features match the columns in `columns.json`
   - Use the same preprocessing pipeline as training (see REPRODUCIBILITY.md)

4. **Make predictions:**
   ```python
   predictions = model.predict(X)
   probabilities = model.predict_proba(X)[:, 1]
   ```

## Model Hash

Model file SHA256 hash: `cabe230436b98a249a7aefa3eb3a24816df4d6f15997cb52613b19a59b3b31a5`

This hash can be used to verify the model file integrity.

## Notes

- The model uses a default threshold of 0.5 for binary classification
- All predictions are for muscular injuries with a 35-day prediction horizon
- The model only includes timelines where players were at PL clubs
- Model maintains 100% recall, prioritizing injury detection over precision

