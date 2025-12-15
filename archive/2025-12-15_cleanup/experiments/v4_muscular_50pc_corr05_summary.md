# V4 Timeline Datasets - Muscular Injuries Only - 50% Target Ratio Model Performance (with Correlation Filtering, threshold=0.5)

**Date:** 2025-12-11 20:49:52

## Dataset Split

- **Training:** 2021-07-01 to 2024-06-30 (9,890 records, 50.0% injury ratio)
- **Validation:** 2024-07-01 to 2025-06-30 (4,890 records, 50.0% injury ratio)
- **Test:** >= 2025-07-01 (153,006 records, 0.6% injury ratio)

## Target

- **Injury Type:** Muscular injuries only

## Approach

- **Correlation filtering:** Threshold = 0.5 (removed one feature from each highly correlated pair)
- **Features:** 879 features (after correlation filtering, down from 1351 initial features)
- **Models:** Random Forest, Gradient Boosting, XGBoost, and LightGBM

## Performance Metrics

| Model | Dataset | Precision | Recall | F1-Score | ROC AUC | Gini |
|-------|---------|-----------|--------|----------|---------|------|
| **RF** | Training | 0.8100 | 0.9242 | 0.8633 | 0.9489 | 0.8977 |
| | Validation | 0.7120 | 0.7693 | 0.7395 | 0.7942 | 0.5884 |
| | Test | 0.0164 | 0.7323 | 0.0321 | 0.7955 | 0.5909 |
| | | | | | | |
| **GB** | Training | 0.9898 | 1.0000 | 0.9949 | 1.0000 | 1.0000 |
| | Validation | 0.7602 | 0.4732 | 0.5833 | 0.7721 | 0.5441 |
| | Test | 0.0210 | 0.4626 | 0.0402 | 0.7614 | 0.5228 |
| | | | | | | |
| **XGB** | Training | 0.9700 | 1.0000 | 0.9848 | 1.0000 | 1.0000 |
| | Validation | 0.7157 | 0.6732 | 0.6938 | 0.7798 | 0.5597 |
| | Test | 0.0154 | 0.6566 | 0.0301 | 0.7547 | 0.5093 |
| | | | | | | |
| **LGBM** | Training | 0.9825 | 1.0000 | 0.9912 | 0.9999 | 0.9999 |
| | Validation | 0.7604 | 0.5178 | 0.6161 | 0.7804 | 0.5608 |
| | Test | 0.0198 | 0.5141 | 0.0382 | 0.7620 | 0.5240 |
| | | | | | | |

## Performance Gaps

### RF
- **F1 Gap (Train → Val):** 0.1238 (14.3% relative)
- **F1 Gap (Train → Test):** 0.8312 (96.3% relative)
- **F1 Gap (Val → Test):** 0.7075 (95.7% relative)

### GB
- **F1 Gap (Train → Val):** 0.4116 (41.4% relative)
- **F1 Gap (Train → Test):** 0.9546 (96.0% relative)
- **F1 Gap (Val → Test):** 0.5431 (93.1% relative)

### XGB
- **F1 Gap (Train → Val):** 0.2910 (29.5% relative)
- **F1 Gap (Train → Test):** 0.9547 (96.9% relative)
- **F1 Gap (Val → Test):** 0.6637 (95.7% relative)

### LGBM
- **F1 Gap (Train → Val):** 0.3751 (37.8% relative)
- **F1 Gap (Train → Test):** 0.9530 (96.2% relative)
- **F1 Gap (Val → Test):** 0.5779 (93.8% relative)
