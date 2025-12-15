# V4 Timeline Datasets - Muscular Injuries Only - 10% TARGET RATIO Model Performance (with Correlation Filtering, threshold=0.8)

**Date:** 2025-12-11 23:29:44

## Dataset Split

- **Training:** 2021-07-01 to 2024-06-30 (49,450 records, 10.0% injury ratio)
- **Validation:** 2024-07-01 to 2025-06-30 (24,450 records, 10.0% injury ratio)
- **Test:** >= 2025-07-01 (153,006 records, 0.6% injury ratio)

## Target

- **Injury Type:** Muscular injuries only

## Approach

- **Correlation filtering:** Threshold = 0.8 (removed one feature from each highly correlated pair)
- **Features:** 1317 features (after correlation filtering, down from 1581 initial features)
- **Models:** Random Forest, Gradient Boosting, XGBoost, and LightGBM

## Performance Metrics

| Model | Dataset | Precision | Recall | F1-Score | ROC AUC | Gini |
|-------|---------|-----------|--------|----------|---------|------|
| **RF** | Training | 0.5815 | 0.9990 | 0.7351 | 0.9967 | 0.9933 |
| | Validation | 0.3131 | 0.4270 | 0.3613 | 0.7931 | 0.5863 |
| | Test | 0.0304 | 0.4566 | 0.0571 | 0.7933 | 0.5866 |
| | | | | | | |
| **GB** | Training | 0.9984 | 0.9947 | 0.9966 | 1.0000 | 1.0000 |
| | Validation | 0.4870 | 0.0229 | 0.0437 | 0.7733 | 0.5467 |
| | Test | 0.0418 | 0.0101 | 0.0163 | 0.7602 | 0.5204 |
| | | | | | | |
| **XGB** | Training | 0.9773 | 1.0000 | 0.9885 | 1.0000 | 1.0000 |
| | Validation | 0.4003 | 0.0978 | 0.1571 | 0.7663 | 0.5326 |
| | Test | 0.0348 | 0.0646 | 0.0452 | 0.7533 | 0.5067 |
| | | | | | | |
| **LGBM** | Training | 0.7261 | 1.0000 | 0.8413 | 0.9991 | 0.9982 |
| | Validation | 0.3440 | 0.2994 | 0.3201 | 0.7655 | 0.5310 |
| | Test | 0.0336 | 0.3051 | 0.0606 | 0.7781 | 0.5562 |
| | | | | | | |

## Performance Gaps

### RF
- **F1 Gap (Train → Val):** 0.3738 (50.8% relative)
- **F1 Gap (Train → Test):** 0.6780 (92.2% relative)
- **F1 Gap (Val → Test):** 0.3042 (84.2% relative)

### GB
- **F1 Gap (Train → Val):** 0.9528 (95.6% relative)
- **F1 Gap (Train → Test):** 0.9803 (98.4% relative)
- **F1 Gap (Val → Test):** 0.0275 (62.8% relative)

### XGB
- **F1 Gap (Train → Val):** 0.8314 (84.1% relative)
- **F1 Gap (Train → Test):** 0.9433 (95.4% relative)
- **F1 Gap (Val → Test):** 0.1119 (71.2% relative)

### LGBM
- **F1 Gap (Train → Val):** 0.5212 (61.9% relative)
- **F1 Gap (Train → Test):** 0.7808 (92.8% relative)
- **F1 Gap (Val → Test):** 0.2596 (81.1% relative)
