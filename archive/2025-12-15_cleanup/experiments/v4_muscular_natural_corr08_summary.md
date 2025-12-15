# V4 Timeline Datasets - Muscular Injuries Only - Natural Target Ratio Model Performance (with Correlation Filtering, threshold=0.8)

**Date:** 2025-12-11 22:52:40

## Dataset Split

- **Training:** 2021-07-01 to 2024-06-30 (1,302,810 records, 0.4% injury ratio)
- **Validation:** 2024-07-01 to 2025-06-30 (447,293 records, 0.5% injury ratio)
- **Test:** >= 2025-07-01 (153,006 records, 0.6% injury ratio)

## Target

- **Injury Type:** Muscular injuries only

## Approach

- **Correlation filtering:** Threshold = 0.8 (removed one feature from each highly correlated pair)
- **Features:** 1434 features (after correlation filtering, down from 1695 initial features)
- **Models:** Random Forest, Gradient Boosting, XGBoost, and LightGBM

## Performance Metrics

| Model | Dataset | Precision | Recall | F1-Score | ROC AUC | Gini |
|-------|---------|-----------|--------|----------|---------|------|
| **RF** | Training | 0.0579 | 0.9998 | 0.1095 | 0.9989 | 0.9978 |
| | Validation | 0.0246 | 0.3256 | 0.0457 | 0.7934 | 0.5867 |
| | Test | 0.0343 | 0.3545 | 0.0625 | 0.7917 | 0.5834 |
| | | | | | | |
| **GB** | Training | 0.8636 | 0.7630 | 0.8102 | 0.9840 | 0.9679 |
| | Validation | 0.0133 | 0.0474 | 0.0208 | 0.7277 | 0.4555 |
| | Test | 0.0162 | 0.0576 | 0.0253 | 0.7148 | 0.4296 |
| | | | | | | |
| **XGB** | Training | 0.9546 | 1.0000 | 0.9768 | 1.0000 | 1.0000 |
| | Validation | 0.0000 | 0.0000 | 0.0000 | 0.7572 | 0.5144 |
| | Test | 0.0000 | 0.0000 | 0.0000 | 0.7375 | 0.4751 |
| | | | | | | |
| **LGBM** | Training | 0.0672 | 1.0000 | 0.1259 | 0.9984 | 0.9968 |
| | Validation | 0.0257 | 0.2879 | 0.0473 | 0.7652 | 0.5304 |
| | Test | 0.0320 | 0.2626 | 0.0571 | 0.7570 | 0.5139 |
| | | | | | | |

## Performance Gaps

### RF
- **F1 Gap (Train → Val):** 0.0638 (58.2% relative)
- **F1 Gap (Train → Test):** 0.0470 (42.9% relative)
- **F1 Gap (Val → Test):** -0.0168 (-36.7% relative)

### GB
- **F1 Gap (Train → Val):** 0.7894 (97.4% relative)
- **F1 Gap (Train → Test):** 0.7849 (96.9% relative)
- **F1 Gap (Val → Test):** -0.0045 (-21.4% relative)

### XGB
- **F1 Gap (Train → Val):** 0.9768 (100.0% relative)
- **F1 Gap (Train → Test):** 0.9768 (100.0% relative)
- **F1 Gap (Val → Test):** N/A

### LGBM
- **F1 Gap (Train → Val):** 0.0787 (62.5% relative)
- **F1 Gap (Train → Test):** 0.0688 (54.7% relative)
- **F1 Gap (Val → Test):** -0.0098 (-20.8% relative)
