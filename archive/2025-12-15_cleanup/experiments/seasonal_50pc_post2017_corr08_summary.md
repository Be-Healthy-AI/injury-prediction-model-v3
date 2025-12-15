# Seasonal Combined Datasets - 10% Target Ratio - Model Performance (with Correlation Filtering, threshold=0.8)

**Date:** 2025-12-14 09:51:08

## Dataset Split

- **Training:** All seasons (2000-2025) with 50% target ratio, combined (24,848 records, 50.0% injury ratio)
- **Validation:** None (trained on all data)
- **Test:** Season 2025-2026 with natural target ratio (153,006 records, 0.6% injury ratio)

## Target

- **Injury Type:** Muscular injuries only

## Approach

- **Correlation filtering:** Threshold = 0.8 (removed one feature from each highly correlated pair)
- **Features:** 1463 features (after correlation filtering, down from 1733 initial features)
- **Models:** Random Forest, Gradient Boosting, XGBoost, and LightGBM

## Performance Metrics

| Model | Dataset | Precision | Recall | F1-Score | ROC AUC | Gini |
|-------|---------|-----------|--------|----------|---------|------|
| **RF** | Training | 0.8443 | 0.9560 | 0.8966 | 0.9689 | 0.9378 |
| | Test | 0.0168 | 0.6875 | 0.0328 | 0.7936 | 0.5872 |
| | | | | | | |
| **GB** | Training | 0.9830 | 0.9991 | 0.9910 | 0.9998 | 0.9997 |
| | Test | 0.0224 | 0.5563 | 0.0432 | 0.7818 | 0.5636 |
| | | | | | | |
| **XGB** | Training | 0.9288 | 1.0000 | 0.9631 | 1.0000 | 0.9999 |
| | Test | 0.0159 | 0.7063 | 0.0311 | 0.7903 | 0.5806 |
| | | | | | | |
| **LGBM** | Training | 0.9163 | 0.9907 | 0.9520 | 0.9924 | 0.9849 |
| | Test | 0.0203 | 0.6312 | 0.0393 | 0.8045 | 0.6090 |
| | | | | | | |

## Performance Gaps

### RF
- **F1 Gap (Train → Test):** 0.8639 (96.3% relative)

### GB
- **F1 Gap (Train → Test):** 0.9478 (95.6% relative)

### XGB
- **F1 Gap (Train → Test):** 0.9320 (96.8% relative)

### LGBM
- **F1 Gap (Train → Test):** 0.9128 (95.9% relative)
