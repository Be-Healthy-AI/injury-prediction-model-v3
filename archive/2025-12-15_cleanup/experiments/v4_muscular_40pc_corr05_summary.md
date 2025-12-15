# V4 Timeline Datasets - Muscular Injuries Only - 40% Target Ratio Model Performance (with Correlation Filtering, threshold=0.5)

**Date:** 2025-12-11 20:47:45

## Dataset Split

- **Training:** 2021-07-01 to 2024-06-30 (12,362 records, 40.0% injury ratio)
- **Validation:** 2024-07-01 to 2025-06-30 (6,112 records, 40.0% injury ratio)
- **Test:** >= 2025-07-01 (153,006 records, 0.6% injury ratio)

## Target

- **Injury Type:** Muscular injuries only

## Approach

- **Correlation filtering:** Threshold = 0.5 (removed one feature from each highly correlated pair)
- **Features:** 925 features (after correlation filtering, down from 1409 initial features)
- **Models:** Random Forest, Gradient Boosting, XGBoost, and LightGBM

## Performance Metrics

| Model | Dataset | Precision | Recall | F1-Score | ROC AUC | Gini |
|-------|---------|-----------|--------|----------|---------|------|
| **RF** | Training | 0.7342 | 0.9294 | 0.8203 | 0.9471 | 0.8943 |
| | Validation | 0.6269 | 0.7407 | 0.6790 | 0.7920 | 0.5840 |
| | Test | 0.0177 | 0.7192 | 0.0345 | 0.7922 | 0.5844 |
| | | | | | | |
| **GB** | Training | 0.9821 | 0.9996 | 0.9908 | 0.9998 | 0.9996 |
| | Validation | 0.7405 | 0.4143 | 0.5313 | 0.7853 | 0.5707 |
| | Test | 0.0227 | 0.3020 | 0.0422 | 0.7626 | 0.5253 |
| | | | | | | |
| **XGB** | Training | 0.9506 | 1.0000 | 0.9747 | 1.0000 | 1.0000 |
| | Validation | 0.6343 | 0.5207 | 0.5719 | 0.7687 | 0.5373 |
| | Test | 0.0184 | 0.5081 | 0.0354 | 0.7457 | 0.4914 |
| | | | | | | |
| **LGBM** | Training | 0.9455 | 1.0000 | 0.9720 | 0.9995 | 0.9990 |
| | Validation | 0.6820 | 0.4728 | 0.5585 | 0.7735 | 0.5470 |
| | Test | 0.0198 | 0.4162 | 0.0379 | 0.7602 | 0.5205 |
| | | | | | | |

## Performance Gaps

### RF
- **F1 Gap (Train → Val):** 0.1413 (17.2% relative)
- **F1 Gap (Train → Test):** 0.7859 (95.8% relative)
- **F1 Gap (Val → Test):** 0.6446 (94.9% relative)

### GB
- **F1 Gap (Train → Val):** 0.4594 (46.4% relative)
- **F1 Gap (Train → Test):** 0.9486 (95.7% relative)
- **F1 Gap (Val → Test):** 0.4892 (92.1% relative)

### XGB
- **F1 Gap (Train → Val):** 0.4028 (41.3% relative)
- **F1 Gap (Train → Test):** 0.9392 (96.4% relative)
- **F1 Gap (Val → Test):** 0.5365 (93.8% relative)

### LGBM
- **F1 Gap (Train → Val):** 0.4135 (42.5% relative)
- **F1 Gap (Train → Test):** 0.9341 (96.1% relative)
- **F1 Gap (Val → Test):** 0.5206 (93.2% relative)
