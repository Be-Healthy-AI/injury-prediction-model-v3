# Seasonal Combined Datasets - 10% Target Ratio - Model Performance (with Correlation Filtering, threshold=0.8)

**Date:** 2025-12-15 09:38:31

## Dataset Split

- **Training:** All seasons (2000-2025) with 50% target ratio, combined (138,090 records, 10.0% injury ratio)
- **Validation:** None (trained on all data)
- **Test:** Season 2025-2026 with natural target ratio (153,006 records, 0.6% injury ratio)

## Target

- **Injury Type:** Muscular injuries only

## Approach

- **Correlation filtering:** Threshold = 0.8 (removed one feature from each highly correlated pair)
- **Features:** 1589 features (after correlation filtering, down from 1852 initial features)
- **Models:** Random Forest, Gradient Boosting, XGBoost, and LightGBM

## Performance Metrics

| Model | Dataset | Precision | Recall | F1-Score | ROC AUC | Gini |
|-------|---------|-----------|--------|----------|---------|------|
| **RF** | Training | 0.4238 | 0.9798 | 0.5917 | 0.9773 | 0.9545 |
| | Test | 0.0225 | 0.6062 | 0.0434 | 0.7942 | 0.5885 |
| | | | | | | |
| **GB** | Training | 0.9935 | 0.7807 | 0.8743 | 0.9979 | 0.9958 |
| | Test | 0.0536 | 0.0375 | 0.0441 | 0.7690 | 0.5380 |
| | | | | | | |
| **XGB** | Training | 0.7927 | 1.0000 | 0.8844 | 0.9998 | 0.9997 |
| | Test | 0.0339 | 0.2375 | 0.0593 | 0.7983 | 0.5965 |
| | | | | | | |
| **LGBM** | Training | 0.4334 | 0.9883 | 0.6025 | 0.9751 | 0.9502 |
| | Test | 0.0206 | 0.5375 | 0.0396 | 0.8099 | 0.6198 |
| | | | | | | |

## Performance Gaps

### RF
- **F1 Gap (Train → Test):** 0.5483 (92.7% relative)

### GB
- **F1 Gap (Train → Test):** 0.8302 (95.0% relative)

### XGB
- **F1 Gap (Train → Test):** 0.8250 (93.3% relative)

### LGBM
- **F1 Gap (Train → Test):** 0.5629 (93.4% relative)
