# V4 Timeline Datasets - Muscular Injuries Only - 50% TARGET RATIO Model Performance (with Correlation Filtering, threshold=0.8)

**Date:** 2025-12-11 23:52:10

## Dataset Split

- **Training:** 2021-07-01 to 2024-06-30 (9,890 records, 50.0% injury ratio)
- **Validation:** 2024-07-01 to 2025-06-30 (4,890 records, 50.0% injury ratio)
- **Test:** >= 2025-07-01 (153,006 records, 0.6% injury ratio)

## Target

- **Injury Type:** Muscular injuries only

## Approach

- **Correlation filtering:** Threshold = 0.8 (removed one feature from each highly correlated pair)
- **Features:** 1089 features (after correlation filtering, down from 1351 initial features)
- **Models:** Random Forest, Gradient Boosting, XGBoost, and LightGBM

## Performance Metrics

| Model | Dataset | Precision | Recall | F1-Score | ROC AUC | Gini |
|-------|---------|-----------|--------|----------|---------|------|
| **RF** | Training | 0.8823 | 0.9765 | 0.9270 | 0.9872 | 0.9745 |
| | Validation | 0.7423 | 0.6892 | 0.7147 | 0.7979 | 0.5958 |
| | Test | 0.0184 | 0.6778 | 0.0358 | 0.7940 | 0.5880 |
| | | | | | | |
| **GB** | Training | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| | Validation | 0.7872 | 0.4070 | 0.5365 | 0.7857 | 0.5713 |
| | Test | 0.0283 | 0.4273 | 0.0532 | 0.7766 | 0.5533 |
| | | | | | | |
| **XGB** | Training | 0.9880 | 1.0000 | 0.9940 | 1.0000 | 1.0000 |
| | Validation | 0.7290 | 0.6119 | 0.6653 | 0.7723 | 0.5446 |
| | Test | 0.0167 | 0.6091 | 0.0325 | 0.7615 | 0.5230 |
| | | | | | | |
| **LGBM** | Training | 0.9962 | 1.0000 | 0.9981 | 1.0000 | 1.0000 |
| | Validation | 0.7642 | 0.4785 | 0.5885 | 0.7780 | 0.5560 |
| | Test | 0.0215 | 0.4747 | 0.0412 | 0.7681 | 0.5361 |
| | | | | | | |

## Performance Gaps

### RF
- **F1 Gap (Train → Val):** 0.2123 (22.9% relative)
- **F1 Gap (Train → Test):** 0.8912 (96.1% relative)
- **F1 Gap (Val → Test):** 0.6789 (95.0% relative)

### GB
- **F1 Gap (Train → Val):** 0.4635 (46.3% relative)
- **F1 Gap (Train → Test):** 0.9468 (94.7% relative)
- **F1 Gap (Val → Test):** 0.4834 (90.1% relative)

### XGB
- **F1 Gap (Train → Val):** 0.3286 (33.1% relative)
- **F1 Gap (Train → Test):** 0.9615 (96.7% relative)
- **F1 Gap (Val → Test):** 0.6329 (95.1% relative)

### LGBM
- **F1 Gap (Train → Val):** 0.4096 (41.0% relative)
- **F1 Gap (Train → Test):** 0.9569 (95.9% relative)
- **F1 Gap (Val → Test):** 0.5473 (93.0% relative)
