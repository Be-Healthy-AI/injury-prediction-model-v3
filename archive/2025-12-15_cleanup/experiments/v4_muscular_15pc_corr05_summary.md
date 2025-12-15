# V4 Timeline Datasets - Muscular Injuries Only - 15% Target Ratio Model Performance (with Correlation Filtering, threshold=0.5)

**Date:** 2025-12-11 19:53:02

## Dataset Split

- **Training:** 2021-07-01 to 2024-06-30 (32,966 records, 15.0% injury ratio)
- **Validation:** 2024-07-01 to 2025-06-30 (16,300 records, 15.0% injury ratio)
- **Test:** >= 2025-07-01 (153,006 records, 0.6% injury ratio)

## Target

- **Injury Type:** Muscular injuries only

## Approach

- **Correlation filtering:** Threshold = 0.5 (removed one feature from each highly correlated pair)
- **Features:** 1055 features (after correlation filtering, down from 1539 initial features)
- **Models:** Random Forest, Gradient Boosting, XGBoost, and LightGBM

## Performance Metrics

| Model | Dataset | Precision | Recall | F1-Score | ROC AUC | Gini |
|-------|---------|-----------|--------|----------|---------|------|
| **RF** | Training | 0.4620 | 0.9527 | 0.6222 | 0.9582 | 0.9165 |
| | Validation | 0.3109 | 0.6528 | 0.4212 | 0.7814 | 0.5627 |
| | Test | 0.0181 | 0.6202 | 0.0352 | 0.7809 | 0.5618 |
| | | | | | | |
| **GB** | Training | 0.9775 | 0.8340 | 0.9000 | 0.9973 | 0.9947 |
| | Validation | 0.5556 | 0.1104 | 0.1842 | 0.7835 | 0.5670 |
| | Test | 0.0296 | 0.0707 | 0.0417 | 0.7655 | 0.5311 |
| | | | | | | |
| **XGB** | Training | 0.9181 | 1.0000 | 0.9573 | 1.0000 | 0.9999 |
| | Validation | 0.4564 | 0.2781 | 0.3456 | 0.7680 | 0.5360 |
| | Test | 0.0240 | 0.1909 | 0.0427 | 0.7570 | 0.5140 |
| | | | | | | |
| **LGBM** | Training | 0.6904 | 1.0000 | 0.8169 | 0.9957 | 0.9914 |
| | Validation | 0.3778 | 0.4389 | 0.4061 | 0.7715 | 0.5430 |
| | Test | 0.0204 | 0.3939 | 0.0389 | 0.7497 | 0.4994 |
| | | | | | | |

## Performance Gaps

### RF
- **F1 Gap (Train → Val):** 0.2010 (32.3% relative)
- **F1 Gap (Train → Test):** 0.5870 (94.3% relative)
- **F1 Gap (Val → Test):** 0.3860 (91.6% relative)

### GB
- **F1 Gap (Train → Val):** 0.7158 (79.5% relative)
- **F1 Gap (Train → Test):** 0.8583 (95.4% relative)
- **F1 Gap (Val → Test):** 0.1425 (77.3% relative)

### XGB
- **F1 Gap (Train → Val):** 0.6117 (63.9% relative)
- **F1 Gap (Train → Test):** 0.9146 (95.5% relative)
- **F1 Gap (Val → Test):** 0.3029 (87.7% relative)

### LGBM
- **F1 Gap (Train → Val):** 0.4108 (50.3% relative)
- **F1 Gap (Train → Test):** 0.7780 (95.2% relative)
- **F1 Gap (Val → Test):** 0.3672 (90.4% relative)
