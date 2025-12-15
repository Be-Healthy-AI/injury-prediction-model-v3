# V4 Timeline Datasets - Muscular Injuries Only - 20% TARGET RATIO Model Performance (with Correlation Filtering, threshold=0.8)

**Date:** 2025-12-11 23:38:58

## Dataset Split

- **Training:** 2021-07-01 to 2024-06-30 (24,725 records, 20.0% injury ratio)
- **Validation:** 2024-07-01 to 2025-06-30 (12,225 records, 20.0% injury ratio)
- **Test:** >= 2025-07-01 (153,006 records, 0.6% injury ratio)

## Target

- **Injury Type:** Muscular injuries only

## Approach

- **Correlation filtering:** Threshold = 0.8 (removed one feature from each highly correlated pair)
- **Features:** 1241 features (after correlation filtering, down from 1509 initial features)
- **Models:** Random Forest, Gradient Boosting, XGBoost, and LightGBM

## Performance Metrics

| Model | Dataset | Precision | Recall | F1-Score | ROC AUC | Gini |
|-------|---------|-----------|--------|----------|---------|------|
| **RF** | Training | 0.7299 | 0.9966 | 0.8426 | 0.9947 | 0.9895 |
| | Validation | 0.4698 | 0.4892 | 0.4793 | 0.7900 | 0.5800 |
| | Test | 0.0281 | 0.5404 | 0.0535 | 0.7944 | 0.5888 |
| | | | | | | |
| **GB** | Training | 0.9986 | 0.9994 | 0.9990 | 1.0000 | 1.0000 |
| | Validation | 0.6577 | 0.0802 | 0.1429 | 0.7864 | 0.5727 |
| | Test | 0.0401 | 0.0566 | 0.0469 | 0.7462 | 0.4925 |
| | | | | | | |
| **XGB** | Training | 0.9794 | 1.0000 | 0.9896 | 1.0000 | 1.0000 |
| | Validation | 0.5294 | 0.2397 | 0.3300 | 0.7647 | 0.5293 |
| | Test | 0.0319 | 0.2111 | 0.0554 | 0.7511 | 0.5023 |
| | | | | | | |
| **LGBM** | Training | 0.8846 | 1.0000 | 0.9388 | 0.9996 | 0.9992 |
| | Validation | 0.5250 | 0.3350 | 0.4090 | 0.7667 | 0.5334 |
| | Test | 0.0258 | 0.2838 | 0.0474 | 0.7524 | 0.5048 |
| | | | | | | |

## Performance Gaps

### RF
- **F1 Gap (Train → Val):** 0.3633 (43.1% relative)
- **F1 Gap (Train → Test):** 0.7891 (93.7% relative)
- **F1 Gap (Val → Test):** 0.4258 (88.8% relative)

### GB
- **F1 Gap (Train → Val):** 0.8561 (85.7% relative)
- **F1 Gap (Train → Test):** 0.9520 (95.3% relative)
- **F1 Gap (Val → Test):** 0.0960 (67.2% relative)

### XGB
- **F1 Gap (Train → Val):** 0.6596 (66.7% relative)
- **F1 Gap (Train → Test):** 0.9342 (94.4% relative)
- **F1 Gap (Val → Test):** 0.2745 (83.2% relative)

### LGBM
- **F1 Gap (Train → Val):** 0.5298 (56.4% relative)
- **F1 Gap (Train → Test):** 0.8914 (95.0% relative)
- **F1 Gap (Val → Test):** 0.3616 (88.4% relative)
