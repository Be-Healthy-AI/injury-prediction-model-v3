# V4 Timeline Datasets - Muscular Injuries Only - 5% TARGET RATIO Model Performance (with Correlation Filtering, threshold=0.8)

**Date:** 2025-12-11 23:22:52

## Dataset Split

- **Training:** 2021-07-01 to 2024-06-30 (98,900 records, 5.0% injury ratio)
- **Validation:** 2024-07-01 to 2025-06-30 (48,900 records, 5.0% injury ratio)
- **Test:** >= 2025-07-01 (153,006 records, 0.6% injury ratio)

## Target

- **Injury Type:** Muscular injuries only

## Approach

- **Correlation filtering:** Threshold = 0.8 (removed one feature from each highly correlated pair)
- **Features:** 1368 features (after correlation filtering, down from 1631 initial features)
- **Models:** Random Forest, Gradient Boosting, XGBoost, and LightGBM

## Performance Metrics

| Model | Dataset | Precision | Recall | F1-Score | ROC AUC | Gini |
|-------|---------|-----------|--------|----------|---------|------|
| **RF** | Training | 0.4374 | 0.9996 | 0.6085 | 0.9984 | 0.9967 |
| | Validation | 0.1863 | 0.3697 | 0.2478 | 0.7933 | 0.5866 |
| | Test | 0.0331 | 0.3949 | 0.0610 | 0.7975 | 0.5949 |
| | | | | | | |
| **GB** | Training | 0.9985 | 0.9664 | 0.9822 | 1.0000 | 1.0000 |
| | Validation | 0.2432 | 0.0074 | 0.0143 | 0.7646 | 0.5292 |
| | Test | 0.0160 | 0.0020 | 0.0036 | 0.7334 | 0.4668 |
| | | | | | | |
| **XGB** | Training | 0.9688 | 1.0000 | 0.9842 | 1.0000 | 1.0000 |
| | Validation | 0.2725 | 0.0372 | 0.0655 | 0.7640 | 0.5281 |
| | Test | 0.0351 | 0.0162 | 0.0221 | 0.7530 | 0.5060 |
| | | | | | | |
| **LGBM** | Training | 0.5213 | 1.0000 | 0.6854 | 0.9986 | 0.9973 |
| | Validation | 0.1863 | 0.2761 | 0.2225 | 0.7551 | 0.5103 |
| | Test | 0.0289 | 0.2697 | 0.0522 | 0.7639 | 0.5278 |
| | | | | | | |

## Performance Gaps

### RF
- **F1 Gap (Train → Val):** 0.3607 (59.3% relative)
- **F1 Gap (Train → Test):** 0.5475 (90.0% relative)
- **F1 Gap (Val → Test):** 0.1868 (75.4% relative)

### GB
- **F1 Gap (Train → Val):** 0.9679 (98.5% relative)
- **F1 Gap (Train → Test):** 0.9786 (99.6% relative)
- **F1 Gap (Val → Test):** 0.0107 (74.9% relative)

### XGB
- **F1 Gap (Train → Val):** 0.9187 (93.3% relative)
- **F1 Gap (Train → Test):** 0.9620 (97.8% relative)
- **F1 Gap (Val → Test):** 0.0434 (66.2% relative)

### LGBM
- **F1 Gap (Train → Val):** 0.4629 (67.5% relative)
- **F1 Gap (Train → Test):** 0.6331 (92.4% relative)
- **F1 Gap (Val → Test):** 0.1702 (76.5% relative)
