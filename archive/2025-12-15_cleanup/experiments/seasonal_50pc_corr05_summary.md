# Seasonal Combined Datasets - 50% Target Ratio - Model Performance (with Correlation Filtering, threshold=0.5)

**Date:** 2025-12-13 23:59:38

## Dataset Split

- **Training:** All seasons (2000-2025) with 50% target ratio, combined (27,618 records, 50.0% injury ratio)
- **Validation:** None (trained on all data)
- **Test:** Season 2025-2026 with natural target ratio (153,006 records, 0.6% injury ratio)

## Target

- **Injury Type:** Muscular injuries only

## Approach

- **Correlation filtering:** Threshold = 0.5 (removed one feature from each highly correlated pair)
- **Features:** 1243 features (after correlation filtering, down from 1746 initial features)
- **Models:** Random Forest, Gradient Boosting, XGBoost, and LightGBM

## Performance Metrics

| Model | Dataset | Precision | Recall | F1-Score | ROC AUC | Gini |
|-------|---------|-----------|--------|----------|---------|------|
| **RF** | Training | 0.7553 | 0.8697 | 0.8085 | 0.8998 | 0.7996 |
| | Test | 0.0144 | 0.7562 | 0.0283 | 0.7885 | 0.5771 |
| | | | | | | |
| **GB** | Training | 0.8926 | 0.9682 | 0.9289 | 0.9816 | 0.9632 |
| | Test | 0.0168 | 0.6062 | 0.0328 | 0.7834 | 0.5668 |
| | | | | | | |
| **XGB** | Training | 0.8269 | 1.0000 | 0.9053 | 0.9991 | 0.9982 |
| | Test | 0.0128 | 0.8063 | 0.0253 | 0.7741 | 0.5482 |
| | | | | | | |
| **LGBM** | Training | 0.8746 | 0.9678 | 0.9188 | 0.9752 | 0.9505 |
| | Test | 0.0173 | 0.6687 | 0.0338 | 0.7921 | 0.5842 |
| | | | | | | |

## Performance Gaps

### RF
- **F1 Gap (Train → Test):** 0.7802 (96.5% relative)

### GB
- **F1 Gap (Train → Test):** 0.8961 (96.5% relative)

### XGB
- **F1 Gap (Train → Test):** 0.8800 (97.2% relative)

### LGBM
- **F1 Gap (Train → Test):** 0.8851 (96.3% relative)
