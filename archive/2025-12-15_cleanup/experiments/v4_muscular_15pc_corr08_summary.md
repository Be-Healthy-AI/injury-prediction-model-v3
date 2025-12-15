# V4 Timeline Datasets - Muscular Injuries Only - 15% TARGET RATIO Model Performance (with Correlation Filtering, threshold=0.8)

**Date:** 2025-12-11 23:34:53

## Dataset Split

- **Training:** 2021-07-01 to 2024-06-30 (32,966 records, 15.0% injury ratio)
- **Validation:** 2024-07-01 to 2025-06-30 (16,300 records, 15.0% injury ratio)
- **Test:** >= 2025-07-01 (153,006 records, 0.6% injury ratio)

## Target

- **Injury Type:** Muscular injuries only

## Approach

- **Correlation filtering:** Threshold = 0.8 (removed one feature from each highly correlated pair)
- **Features:** 1270 features (after correlation filtering, down from 1539 initial features)
- **Models:** Random Forest, Gradient Boosting, XGBoost, and LightGBM

## Performance Metrics

| Model | Dataset | Precision | Recall | F1-Score | ROC AUC | Gini |
|-------|---------|-----------|--------|----------|---------|------|
| **RF** | Training | 0.6778 | 0.9976 | 0.8072 | 0.9959 | 0.9918 |
| | Validation | 0.4043 | 0.4597 | 0.4302 | 0.7955 | 0.5910 |
| | Test | 0.0295 | 0.5081 | 0.0558 | 0.7919 | 0.5837 |
| | | | | | | |
| **GB** | Training | 0.9988 | 0.9976 | 0.9982 | 1.0000 | 1.0000 |
| | Validation | 0.5370 | 0.0474 | 0.0872 | 0.7824 | 0.5647 |
| | Test | 0.0429 | 0.0313 | 0.0362 | 0.7544 | 0.5089 |
| | | | | | | |
| **XGB** | Training | 0.9794 | 1.0000 | 0.9896 | 1.0000 | 1.0000 |
| | Validation | 0.5042 | 0.1734 | 0.2581 | 0.7735 | 0.5470 |
| | Test | 0.0293 | 0.1162 | 0.0468 | 0.7477 | 0.4954 |
| | | | | | | |
| **LGBM** | Training | 0.8235 | 1.0000 | 0.9032 | 0.9993 | 0.9986 |
| | Validation | 0.4400 | 0.3137 | 0.3663 | 0.7761 | 0.5522 |
| | Test | 0.0285 | 0.2788 | 0.0517 | 0.7715 | 0.5431 |
| | | | | | | |

## Performance Gaps

### RF
- **F1 Gap (Train → Val):** 0.3769 (46.7% relative)
- **F1 Gap (Train → Test):** 0.7513 (93.1% relative)
- **F1 Gap (Val → Test):** 0.3744 (87.0% relative)

### GB
- **F1 Gap (Train → Val):** 0.9110 (91.3% relative)
- **F1 Gap (Train → Test):** 0.9620 (96.4% relative)
- **F1 Gap (Val → Test):** 0.0510 (58.5% relative)

### XGB
- **F1 Gap (Train → Val):** 0.7315 (73.9% relative)
- **F1 Gap (Train → Test):** 0.9428 (95.3% relative)
- **F1 Gap (Val → Test):** 0.2113 (81.9% relative)

### LGBM
- **F1 Gap (Train → Val):** 0.5369 (59.4% relative)
- **F1 Gap (Train → Test):** 0.8515 (94.3% relative)
- **F1 Gap (Val → Test):** 0.3146 (85.9% relative)
