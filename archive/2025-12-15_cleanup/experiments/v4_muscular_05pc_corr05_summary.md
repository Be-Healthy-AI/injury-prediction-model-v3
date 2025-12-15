# V4 Timeline Datasets - Muscular Injuries Only - 5% Target Ratio Model Performance (with Correlation Filtering, threshold=0.5)

**Date:** 2025-12-11 19:27:30

## Dataset Split

- **Training:** 2021-07-01 to 2024-06-30 (98,900 records, 5.0% injury ratio)
- **Validation:** 2024-07-01 to 2025-06-30 (48,900 records, 5.0% injury ratio)
- **Test:** >= 2025-07-01 (153,006 records, 0.6% injury ratio)

## Target

- **Injury Type:** Muscular injuries only

## Approach

- **Correlation filtering:** Threshold = 0.5 (removed one feature from each highly correlated pair)
- **Features:** 1151 features (after correlation filtering, down from 1631 initial features)
- **Models:** Random Forest, Gradient Boosting, XGBoost, and LightGBM

## Performance Metrics

| Model | Dataset | Precision | Recall | F1-Score | ROC AUC | Gini |
|-------|---------|-----------|--------|----------|---------|------|
| **RF** | Training | 0.2297 | 0.9753 | 0.3718 | 0.9728 | 0.9457 |
| | Validation | 0.1234 | 0.5791 | 0.2035 | 0.7758 | 0.5515 |
| | Test | 0.0196 | 0.5778 | 0.0379 | 0.7809 | 0.5618 |
| | | | | | | |
| **GB** | Training | 0.9925 | 0.6154 | 0.7597 | 0.9982 | 0.9964 |
| | Validation | 0.2463 | 0.0135 | 0.0256 | 0.7714 | 0.5429 |
| | Test | 0.0075 | 0.0020 | 0.0032 | 0.7545 | 0.5090 |
| | | | | | | |
| **XGB** | Training | 0.9108 | 1.0000 | 0.9533 | 1.0000 | 0.9999 |
| | Validation | 0.2667 | 0.0851 | 0.1290 | 0.7657 | 0.5314 |
| | Test | 0.0437 | 0.0485 | 0.0460 | 0.7520 | 0.5040 |
| | | | | | | |
| **LGBM** | Training | 0.3699 | 1.0000 | 0.5400 | 0.9936 | 0.9873 |
| | Validation | 0.1559 | 0.4204 | 0.2275 | 0.7664 | 0.5329 |
| | Test | 0.0206 | 0.3657 | 0.0390 | 0.7603 | 0.5206 |
| | | | | | | |

## Performance Gaps

### RF
- **F1 Gap (Train → Val):** 0.1683 (45.3% relative)
- **F1 Gap (Train → Test):** 0.3340 (89.8% relative)
- **F1 Gap (Val → Test):** 0.1656 (81.4% relative)

### GB
- **F1 Gap (Train → Val):** 0.7341 (96.6% relative)
- **F1 Gap (Train → Test):** 0.7565 (99.6% relative)
- **F1 Gap (Val → Test):** 0.0224 (87.5% relative)

### XGB
- **F1 Gap (Train → Val):** 0.8244 (86.5% relative)
- **F1 Gap (Train → Test):** 0.9074 (95.2% relative)
- **F1 Gap (Val → Test):** 0.0830 (64.4% relative)

### LGBM
- **F1 Gap (Train → Val):** 0.3125 (57.9% relative)
- **F1 Gap (Train → Test):** 0.5010 (92.8% relative)
- **F1 Gap (Val → Test):** 0.1885 (82.9% relative)
