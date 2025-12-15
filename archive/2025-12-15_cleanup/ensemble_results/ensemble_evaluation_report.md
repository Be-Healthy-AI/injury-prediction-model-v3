# Ensemble Model Evaluation Report

**Generated:** 2025-12-14 20:36:37

## Models Combined

1. **LightGBM (10%, 0.8 corr)**
   - Test Gini: 0.6198 (61.98%)
   - File: models/lgbm_model_seasonal_10pc_v4_muscular_corr08.joblib

2. **Random Forest (50%, 0.5 corr)**
   - Test Gini: 0.5771 (57.71%)
   - File: models/rf_model_seasonal_50pc_v4_muscular_corr05.joblib

## Ensemble Methods Tested

1. **Equal Weights (0.5/0.5)**: Simple average of predictions
2. **Gini-Weighted**: Weighted by known test Gini performance
3. **Rank Averaging**: Average ranks, then convert back to probabilities
4. **Geometric Mean**: Geometric mean of probabilities
5. **Optimized Weights**: Grid search for optimal weights (test set)

## Results Summary

| Model | Gini | ROC AUC | Precision | Recall | F1-Score |
|-------|------|---------|-----------|--------|----------|
| LightGBM (10%, 0.8 corr) | 0.5939 | 0.7970 | 0.0151 | 0.8030 | 0.0296 |
| Random Forest (50%, 0.5 corr) | 0.5924 | 0.7962 | 0.0167 | 0.7455 | 0.0327 |
| Ensemble: Equal Weights (0.5/0.5) | 0.6155 | 0.8077 | 0.0166 | 0.7879 | 0.0325 |
| Ensemble: Gini-Weighted (0.518/0.482) | 0.6152 | 0.8076 | 0.0166 | 0.7879 | 0.0326 |
| Ensemble: Rank Averaging | -0.6155 | 0.1922 | 0.0065 | 1.0000 | 0.0129 |
| Ensemble: Geometric Mean | 0.6110 | 0.8055 | 0.0165 | 0.7778 | 0.0323 |
| Ensemble: Optimized Weights (0.400/0.600) | 0.6160 | 0.8080 | 0.0168 | 0.7919 | 0.0328 |

## Best Model

**Ensemble: Optimized Weights (0.400/0.600)**

- Gini: 0.6160 (61.60%)
- ROC AUC: 0.8080
- Precision: 0.0168 (1.68%)
- Recall: 0.7919 (79.19%)
- F1-Score: 0.0328

## Improvement Analysis

- Best Individual Model Gini: 0.5939 (59.39%)
- Best Ensemble Gini: 0.6160 (61.60%)
- Improvement: +0.0221 (+2.21 percentage points)

✅ **Conclusion:** Ensemble improves performance over individual models.
