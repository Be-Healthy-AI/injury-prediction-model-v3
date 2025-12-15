# V4 Timeline Datasets - Muscular Injuries Only - Combined Train+Val Model Performance (with Correlation Filtering)

**Date:** 2025-12-10 22:56:00

## Dataset Split

- **Training:** 2022-07-01 to 2025-06-30 (combined train + validation: 92,374 records, 8.0% injury ratio)
- **Test:** >= 2025-07-01 (153,006 records, 0.6% injury ratio)

## Target

- **Injury Type:** Muscular injuries only

## Approach

- **Correlation filtering:** Threshold = 0.8 (removed one feature from each highly correlated pair)
- **Features:** 1568 features (after correlation filtering, down from 1839 initial features)
- **Models:** Random Forest, Gradient Boosting, XGBoost, and LightGBM

## Performance Metrics

| Model | Dataset | Precision | Recall | F1-Score | ROC AUC | Gini |
|-------|---------|-----------|--------|----------|---------|------|
| **RF** | Training (Train+Val Combined) | 0.4951 | 0.9984 | 0.6619 | 0.9950 | 0.9900 |
| | Test (>= 2025-07-01) | 0.0318 | 0.5061 | 0.0599 | 0.8000 | 0.5999 |
| | | | | | | |
| **GB** | Training (Train+Val Combined) | 0.9975 | 0.9168 | 0.9554 | 0.9999 | 0.9997 |
| | Test (>= 2025-07-01) | 0.0942 | 0.0364 | 0.0525 | 0.7869 | 0.5739 |
| | | | | | | |
| **XGB** | Training (Train+Val Combined) | 0.9556 | 1.0000 | 0.9773 | 1.0000 | 1.0000 |
| | Test (>= 2025-07-01) | 0.0466 | 0.0980 | 0.0632 | 0.7835 | 0.5671 |
| | | | | | | |
| **LGBM** | Training (Train+Val Combined) | 0.5209 | 0.9996 | 0.6849 | 0.9941 | 0.9883 |
| | Test (>= 2025-07-01) | 0.0260 | 0.4111 | 0.0489 | 0.7920 | 0.5840 |
| | | | | | | |

## Performance Gaps

### RF
- **F1 Gap (Train → Test):** 0.6020 (91.0% relative)

### GB
- **F1 Gap (Train → Test):** 0.9030 (94.5% relative)

### XGB
- **F1 Gap (Train → Test):** 0.9142 (93.5% relative)

### LGBM
- **F1 Gap (Train → Test):** 0.6360 (92.9% relative)
