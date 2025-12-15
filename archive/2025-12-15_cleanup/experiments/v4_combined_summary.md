# V4 Timeline Datasets - Combined Train+Val Model Performance

**Date:** 2025-12-09 21:58:00

## Dataset Split

- **Training:** 2022-07-01 to 2025-06-30 (combined train + validation: 144,125 records, 8.0% injury ratio)
- **Test:** >= 2025-07-01 (109,022 records, 1.6% injury ratio)

## Approach

- **Baseline approach:** No calibration, no feature selection, no correlation filtering
- **Features:** 1729 features (after one-hot encoding)
- **Models:** Random Forest and Gradient Boosting only

## Performance Metrics

| Model | Dataset | Precision | Recall | F1-Score | ROC AUC | Gini |
|-------|---------|-----------|--------|----------|---------|------|
| **RF** | Training (Train+Val Combined) | 0.4674 | 1.0000 | 0.6370 | 0.9976 | 0.9951 |
| | Test (>= 2025-07-01) | 0.0440 | 0.4546 | 0.0802 | 0.7649 | 0.5298 |
| | | | | | | |
| **GB** | Training (Train+Val Combined) | 1.0000 | 0.9375 | 0.9677 | 1.0000 | 0.9999 |
| | Test (>= 2025-07-01) | 0.1740 | 0.0362 | 0.0599 | 0.7323 | 0.4647 |
| | | | | | | |

## Performance Gaps

### RF
- **F1 Gap (Train → Test):** 0.5568 (87.4% relative)

### GB
- **F1 Gap (Train → Test):** 0.9078 (93.8% relative)
