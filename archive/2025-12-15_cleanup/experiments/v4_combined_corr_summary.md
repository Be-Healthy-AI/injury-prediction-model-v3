# V4 Timeline Datasets - Combined Train+Val Model Performance (with Correlation Filtering)

**Date:** 2025-12-09 22:17:57

## Dataset Split

- **Training:** 2022-07-01 to 2025-06-30 (combined train + validation: 144,125 records, 8.0% injury ratio)
- **Test:** >= 2025-07-01 (109,022 records, 1.6% injury ratio)

## Approach

- **Correlation filtering:** Threshold = 0.8 (removed one feature from each highly correlated pair)
- **Features:** 1482 features (after correlation filtering, down from 1729 initial features)
- **Models:** Random Forest and Gradient Boosting only

## Performance Metrics

| Model | Dataset | Precision | Recall | F1-Score | ROC AUC | Gini |
|-------|---------|-----------|--------|----------|---------|------|
| **RF** | Training (Train+Val Combined) | 0.3882 | 0.9877 | 0.5573 | 0.9889 | 0.9778 |
| | Test (>= 2025-07-01) | 0.0410 | 0.5264 | 0.0761 | 0.7610 | 0.5220 |
| | | | | | | |
| **GB** | Training (Train+Val Combined) | 0.9973 | 0.7448 | 0.8527 | 0.9984 | 0.9968 |
| | Test (>= 2025-07-01) | 0.0751 | 0.0109 | 0.0191 | 0.7296 | 0.4592 |
| | | | | | | |

## Performance Gaps

### RF
- **F1 Gap (Train → Test):** 0.4812 (86.3% relative)

### GB
- **F1 Gap (Train → Test):** 0.8337 (97.8% relative)
