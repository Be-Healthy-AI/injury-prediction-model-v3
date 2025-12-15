# V4 Timeline Datasets - Muscular Injuries Only - 30% TARGET RATIO Model Performance (with Correlation Filtering, threshold=0.8)

**Date:** 2025-12-11 23:46:07

## Dataset Split

- **Training:** 2021-07-01 to 2024-06-30 (16,483 records, 30.0% injury ratio)
- **Validation:** 2024-07-01 to 2025-06-30 (8,150 records, 30.0% injury ratio)
- **Test:** >= 2025-07-01 (153,006 records, 0.6% injury ratio)

## Target

- **Injury Type:** Muscular injuries only

## Approach

- **Correlation filtering:** Threshold = 0.8 (removed one feature from each highly correlated pair)
- **Features:** 1192 features (after correlation filtering, down from 1456 initial features)
- **Models:** Random Forest, Gradient Boosting, XGBoost, and LightGBM

## Performance Metrics

| Model | Dataset | Precision | Recall | F1-Score | ROC AUC | Gini |
|-------|---------|-----------|--------|----------|---------|------|
| **RF** | Training | 0.7961 | 0.9901 | 0.8826 | 0.9916 | 0.9831 |
| | Validation | 0.5837 | 0.5636 | 0.5734 | 0.7954 | 0.5907 |
| | Test | 0.0245 | 0.6081 | 0.0471 | 0.7946 | 0.5892 |
| | | | | | | |
| **GB** | Training | 0.9998 | 1.0000 | 0.9999 | 1.0000 | 1.0000 |
| | Validation | 0.7085 | 0.1571 | 0.2571 | 0.7709 | 0.5418 |
| | Test | 0.0329 | 0.1222 | 0.0518 | 0.7744 | 0.5488 |
| | | | | | | |
| **XGB** | Training | 0.9825 | 1.0000 | 0.9912 | 1.0000 | 1.0000 |
| | Validation | 0.6379 | 0.4049 | 0.4954 | 0.7748 | 0.5495 |
| | Test | 0.0269 | 0.3909 | 0.0504 | 0.7635 | 0.5271 |
| | | | | | | |
| **LGBM** | Training | 0.9426 | 1.0000 | 0.9705 | 0.9999 | 0.9997 |
| | Validation | 0.6148 | 0.3669 | 0.4595 | 0.7699 | 0.5398 |
| | Test | 0.0256 | 0.3394 | 0.0475 | 0.7605 | 0.5210 |
| | | | | | | |

## Performance Gaps

### RF
- **F1 Gap (Train → Val):** 0.3091 (35.0% relative)
- **F1 Gap (Train → Test):** 0.8354 (94.7% relative)
- **F1 Gap (Val → Test):** 0.5263 (91.8% relative)

### GB
- **F1 Gap (Train → Val):** 0.7428 (74.3% relative)
- **F1 Gap (Train → Test):** 0.9481 (94.8% relative)
- **F1 Gap (Val → Test):** 0.2053 (79.9% relative)

### XGB
- **F1 Gap (Train → Val):** 0.4958 (50.0% relative)
- **F1 Gap (Train → Test):** 0.9408 (94.9% relative)
- **F1 Gap (Val → Test):** 0.4450 (89.8% relative)

### LGBM
- **F1 Gap (Train → Val):** 0.5109 (52.6% relative)
- **F1 Gap (Train → Test):** 0.9229 (95.1% relative)
- **F1 Gap (Val → Test):** 0.4120 (89.7% relative)
