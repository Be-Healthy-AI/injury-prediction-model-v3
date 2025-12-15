# V4 Timeline Datasets - Muscular Injuries Only - Train/Val/Test Split Model Performance (with Correlation Filtering, threshold=0.5)

**Date:** 2025-12-11 18:52:14

## Dataset Split

- **Training:** 2021-07-01 to 2024-06-30 (1,302,810 records, 0.4% injury ratio)
- **Validation:** 2024-07-01 to 2025-06-30 (447,293 records, 0.5% injury ratio)
- **Test:** >= 2025-07-01 (153,006 records, 0.6% injury ratio)

## Target

- **Injury Type:** Muscular injuries only

## Approach

- **Correlation filtering:** Threshold = 0.5 (removed one feature from each highly correlated pair)
- **Features:** 1240 features (after correlation filtering, down from 1695 initial features)
- **Models:** Random Forest, Gradient Boosting, XGBoost, and LightGBM

## Performance Metrics

| Model | Dataset | Precision | Recall | F1-Score | ROC AUC | Gini |
|-------|---------|-----------|--------|----------|---------|------|
| **RF** | Training | 0.0250 | 0.9881 | 0.0487 | 0.9835 | 0.9671 |
| | Validation | 0.0152 | 0.5133 | 0.0296 | 0.7686 | 0.5372 |
| | Test | 0.0205 | 0.5242 | 0.0395 | 0.7800 | 0.5601 |
| | | | | | | |
| **GB** | Training | 0.7541 | 0.4762 | 0.5838 | 0.9856 | 0.9712 |
| | Validation | 0.0098 | 0.0188 | 0.0129 | 0.7340 | 0.4679 |
| | Test | 0.0076 | 0.0172 | 0.0106 | 0.6933 | 0.3865 |
| | | | | | | |
| **XGB** | Training | 0.8993 | 0.9951 | 0.9448 | 1.0000 | 1.0000 |
| | Validation | 0.0000 | 0.0000 | 0.0000 | 0.7767 | 0.5533 |
| | Test | 0.2000 | 0.0010 | 0.0020 | 0.7602 | 0.5203 |
| | | | | | | |
| **LGBM** | Training | 0.0398 | 0.9992 | 0.0765 | 0.9931 | 0.9863 |
| | Validation | 0.0187 | 0.4123 | 0.0358 | 0.7688 | 0.5375 |
| | Test | 0.0193 | 0.3323 | 0.0365 | 0.7542 | 0.5084 |
| | | | | | | |

## Performance Gaps

### RF
- **F1 Gap (Train → Val):** 0.0191 (39.3% relative)
- **F1 Gap (Train → Test):** 0.0092 (18.9% relative)
- **F1 Gap (Val → Test):** -0.0100 (-33.7% relative)

### GB
- **F1 Gap (Train → Val):** 0.5709 (97.8% relative)
- **F1 Gap (Train → Test):** 0.5732 (98.2% relative)
- **F1 Gap (Val → Test):** 0.0023 (17.7% relative)

### XGB
- **F1 Gap (Train → Val):** 0.9448 (100.0% relative)
- **F1 Gap (Train → Test):** 0.9428 (99.8% relative)
- **F1 Gap (Val → Test):** N/A

### LGBM
- **F1 Gap (Train → Val):** 0.0407 (53.2% relative)
- **F1 Gap (Train → Test):** 0.0401 (52.4% relative)
- **F1 Gap (Val → Test):** -0.0007 (-1.8% relative)
