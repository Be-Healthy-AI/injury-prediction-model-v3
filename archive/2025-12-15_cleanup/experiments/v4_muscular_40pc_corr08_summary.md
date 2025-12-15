# V4 Timeline Datasets - Muscular Injuries Only - 40% TARGET RATIO Model Performance (with Correlation Filtering, threshold=0.8)

**Date:** 2025-12-11 23:48:52

## Dataset Split

- **Training:** 2021-07-01 to 2024-06-30 (12,362 records, 40.0% injury ratio)
- **Validation:** 2024-07-01 to 2025-06-30 (6,112 records, 40.0% injury ratio)
- **Test:** >= 2025-07-01 (153,006 records, 0.6% injury ratio)

## Target

- **Injury Type:** Muscular injuries only

## Approach

- **Correlation filtering:** Threshold = 0.8 (removed one feature from each highly correlated pair)
- **Features:** 1147 features (after correlation filtering, down from 1409 initial features)
- **Models:** Random Forest, Gradient Boosting, XGBoost, and LightGBM

## Performance Metrics

| Model | Dataset | Precision | Recall | F1-Score | ROC AUC | Gini |
|-------|---------|-----------|--------|----------|---------|------|
| **RF** | Training | 0.8406 | 0.9856 | 0.9074 | 0.9882 | 0.9763 |
| | Validation | 0.6729 | 0.6176 | 0.6441 | 0.7989 | 0.5978 |
| | Test | 0.0216 | 0.6556 | 0.0418 | 0.7954 | 0.5908 |
| | | | | | | |
| **GB** | Training | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| | Validation | 0.7658 | 0.2982 | 0.4292 | 0.7806 | 0.5613 |
| | Test | 0.0274 | 0.2242 | 0.0489 | 0.7691 | 0.5382 |
| | | | | | | |
| **XGB** | Training | 0.9866 | 1.0000 | 0.9933 | 1.0000 | 1.0000 |
| | Validation | 0.6669 | 0.4904 | 0.5652 | 0.7670 | 0.5340 |
| | Test | 0.0198 | 0.4646 | 0.0379 | 0.7556 | 0.5111 |
| | | | | | | |
| **LGBM** | Training | 0.9804 | 1.0000 | 0.9901 | 1.0000 | 1.0000 |
| | Validation | 0.7300 | 0.4213 | 0.5342 | 0.7828 | 0.5656 |
| | Test | 0.0260 | 0.4202 | 0.0489 | 0.7677 | 0.5355 |
| | | | | | | |

## Performance Gaps

### RF
- **F1 Gap (Train → Val):** 0.2633 (29.0% relative)
- **F1 Gap (Train → Test):** 0.8655 (95.4% relative)
- **F1 Gap (Val → Test):** 0.6022 (93.5% relative)

### GB
- **F1 Gap (Train → Val):** 0.5708 (57.1% relative)
- **F1 Gap (Train → Test):** 0.9511 (95.1% relative)
- **F1 Gap (Val → Test):** 0.3803 (88.6% relative)

### XGB
- **F1 Gap (Train → Val):** 0.4281 (43.1% relative)
- **F1 Gap (Train → Test):** 0.9554 (96.2% relative)
- **F1 Gap (Val → Test):** 0.5273 (93.3% relative)

### LGBM
- **F1 Gap (Train → Val):** 0.4559 (46.0% relative)
- **F1 Gap (Train → Test):** 0.9412 (95.1% relative)
- **F1 Gap (Val → Test):** 0.4853 (90.8% relative)
