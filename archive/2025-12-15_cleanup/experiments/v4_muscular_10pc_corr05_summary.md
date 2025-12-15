# V4 Timeline Datasets - Muscular Injuries Only - 10% Target Ratio Model Performance (with Correlation Filtering, threshold=0.5)

**Date:** 2025-12-11 19:36:27

## Dataset Split

- **Training:** 2021-07-01 to 2024-06-30 (49,450 records, 10.0% injury ratio)
- **Validation:** 2024-07-01 to 2025-06-30 (24,450 records, 10.0% injury ratio)
- **Test:** >= 2025-07-01 (153,006 records, 0.6% injury ratio)

## Target

- **Injury Type:** Muscular injuries only

## Approach

- **Correlation filtering:** Threshold = 0.5 (removed one feature from each highly correlated pair)
- **Features:** 1102 features (after correlation filtering, down from 1581 initial features)
- **Models:** Random Forest, Gradient Boosting, XGBoost, and LightGBM

## Performance Metrics

| Model | Dataset | Precision | Recall | F1-Score | ROC AUC | Gini |
|-------|---------|-----------|--------|----------|---------|------|
| **RF** | Training | 0.3672 | 0.9614 | 0.5314 | 0.9657 | 0.9313 |
| | Validation | 0.2226 | 0.6172 | 0.3272 | 0.7789 | 0.5578 |
| | Test | 0.0192 | 0.6131 | 0.0372 | 0.7826 | 0.5652 |
| | | | | | | |
| **GB** | Training | 0.9851 | 0.7347 | 0.8417 | 0.9975 | 0.9950 |
| | Validation | 0.4700 | 0.0544 | 0.0975 | 0.7799 | 0.5599 |
| | Test | 0.0282 | 0.0263 | 0.0272 | 0.7663 | 0.5326 |
| | | | | | | |
| **XGB** | Training | 0.9132 | 1.0000 | 0.9546 | 1.0000 | 0.9999 |
| | Validation | 0.3721 | 0.1742 | 0.2373 | 0.7648 | 0.5297 |
| | Test | 0.0329 | 0.1323 | 0.0527 | 0.7526 | 0.5051 |
| | | | | | | |
| **LGBM** | Training | 0.5686 | 0.9998 | 0.7249 | 0.9945 | 0.9889 |
| | Validation | 0.2783 | 0.4233 | 0.3358 | 0.7660 | 0.5319 |
| | Test | 0.0220 | 0.4111 | 0.0418 | 0.7658 | 0.5317 |
| | | | | | | |

## Performance Gaps

### RF
- **F1 Gap (Train → Val):** 0.2042 (38.4% relative)
- **F1 Gap (Train → Test):** 0.4942 (93.0% relative)
- **F1 Gap (Val → Test):** 0.2900 (88.6% relative)

### GB
- **F1 Gap (Train → Val):** 0.7441 (88.4% relative)
- **F1 Gap (Train → Test):** 0.8144 (96.8% relative)
- **F1 Gap (Val → Test):** 0.0703 (72.1% relative)

### XGB
- **F1 Gap (Train → Val):** 0.7173 (75.1% relative)
- **F1 Gap (Train → Test):** 0.9019 (94.5% relative)
- **F1 Gap (Val → Test):** 0.1846 (77.8% relative)

### LGBM
- **F1 Gap (Train → Val):** 0.3891 (53.7% relative)
- **F1 Gap (Train → Test):** 0.6831 (94.2% relative)
- **F1 Gap (Val → Test):** 0.2940 (87.5% relative)
