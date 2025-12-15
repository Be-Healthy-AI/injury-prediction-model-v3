# V4 Timeline Datasets - Muscular Injuries Only - 25% Target Ratio Model Performance (with Correlation Filtering, threshold=0.5)

**Date:** 2025-12-11 20:10:18

## Dataset Split

- **Training:** 2021-07-01 to 2024-06-30 (19,780 records, 25.0% injury ratio)
- **Validation:** 2024-07-01 to 2025-06-30 (9,780 records, 25.0% injury ratio)
- **Test:** >= 2025-07-01 (153,006 records, 0.6% injury ratio)

## Target

- **Injury Type:** Muscular injuries only

## Approach

- **Correlation filtering:** Threshold = 0.5 (removed one feature from each highly correlated pair)
- **Features:** 998 features (after correlation filtering, down from 1484 initial features)
- **Models:** Random Forest, Gradient Boosting, XGBoost, and LightGBM

## Performance Metrics

| Model | Dataset | Precision | Recall | F1-Score | ROC AUC | Gini |
|-------|---------|-----------|--------|----------|---------|------|
| **RF** | Training | 0.6136 | 0.9509 | 0.7459 | 0.9552 | 0.9105 |
| | Validation | 0.4666 | 0.7035 | 0.5611 | 0.7851 | 0.5702 |
| | Test | 0.0184 | 0.6737 | 0.0359 | 0.7926 | 0.5851 |
| | | | | | | |
| **GB** | Training | 0.9748 | 0.9701 | 0.9724 | 0.9989 | 0.9978 |
| | Validation | 0.6700 | 0.2176 | 0.3285 | 0.7754 | 0.5507 |
| | Test | 0.0309 | 0.1485 | 0.0512 | 0.7695 | 0.5391 |
| | | | | | | |
| **XGB** | Training | 0.9432 | 1.0000 | 0.9707 | 1.0000 | 1.0000 |
| | Validation | 0.5456 | 0.3820 | 0.4494 | 0.7645 | 0.5291 |
| | Test | 0.0228 | 0.3364 | 0.0426 | 0.7528 | 0.5056 |
| | | | | | | |
| **LGBM** | Training | 0.8507 | 1.0000 | 0.9193 | 0.9984 | 0.9968 |
| | Validation | 0.5543 | 0.4282 | 0.4832 | 0.7691 | 0.5381 |
| | Test | 0.0229 | 0.3758 | 0.0431 | 0.7671 | 0.5341 |
| | | | | | | |

## Performance Gaps

### RF
- **F1 Gap (Train → Val):** 0.1848 (24.8% relative)
- **F1 Gap (Train → Test):** 0.7100 (95.2% relative)
- **F1 Gap (Val → Test):** 0.5252 (93.6% relative)

### GB
- **F1 Gap (Train → Val):** 0.6439 (66.2% relative)
- **F1 Gap (Train → Test):** 0.9212 (94.7% relative)
- **F1 Gap (Val → Test):** 0.2773 (84.4% relative)

### XGB
- **F1 Gap (Train → Val):** 0.5214 (53.7% relative)
- **F1 Gap (Train → Test):** 0.9281 (95.6% relative)
- **F1 Gap (Val → Test):** 0.4067 (90.5% relative)

### LGBM
- **F1 Gap (Train → Val):** 0.4362 (47.4% relative)
- **F1 Gap (Train → Test):** 0.8762 (95.3% relative)
- **F1 Gap (Val → Test):** 0.4400 (91.1% relative)
