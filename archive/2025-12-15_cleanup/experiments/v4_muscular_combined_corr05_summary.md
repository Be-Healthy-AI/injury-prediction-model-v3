# V4 Timeline Datasets - Muscular Injuries Only - Combined Train+Val Model Performance (with Correlation Filtering, threshold=0.5)

**Date:** 2025-12-10 23:29:35

## Dataset Split

- **Training:** 2022-07-01 to 2025-06-30 (combined train + validation: 92,374 records, 8.0% injury ratio)
- **Test:** >= 2025-07-01 (153,006 records, 0.6% injury ratio)

## Target

- **Injury Type:** Muscular injuries only

## Approach

- **Correlation filtering:** Threshold = 0.5 (removed one feature from each highly correlated pair)
- **Features:** 1342 features (after correlation filtering, down from 1839 initial features)
- **Models:** Random Forest, Gradient Boosting, XGBoost, and LightGBM

## Performance Metrics

| Model | Dataset | Precision | Recall | F1-Score | ROC AUC | Gini |
|-------|---------|-----------|--------|----------|---------|------|
| **RF** | Training (Train+Val Combined) | 0.2701 | 0.9382 | 0.4194 | 0.9489 | 0.8977 |
| | Test (>= 2025-07-01) | 0.0180 | 0.6293 | 0.0349 | 0.7883 | 0.5767 |
| | | | | | | |
| **GB** | Training (Train+Val Combined) | 0.9813 | 0.5189 | 0.6789 | 0.9930 | 0.9860 |
| | Test (>= 2025-07-01) | 0.0356 | 0.0263 | 0.0302 | 0.7868 | 0.5737 |
| | | | | | | |
| **XGB** | Training (Train+Val Combined) | 0.8164 | 1.0000 | 0.8989 | 0.9999 | 0.9997 |
| | Test (>= 2025-07-01) | 0.0300 | 0.1606 | 0.0505 | 0.7657 | 0.5313 |
| | | | | | | |
| **LGBM** | Training (Train+Val Combined) | 0.4148 | 0.9926 | 0.5850 | 0.9842 | 0.9684 |
| | Test (>= 2025-07-01) | 0.0200 | 0.4798 | 0.0383 | 0.7805 | 0.5609 |
| | | | | | | |

## Performance Gaps

### RF
- **F1 Gap (Train → Test):** 0.3845 (91.7% relative)

### GB
- **F1 Gap (Train → Test):** 0.6487 (95.5% relative)

### XGB
- **F1 Gap (Train → Test):** 0.8484 (94.4% relative)

### LGBM
- **F1 Gap (Train → Test):** 0.5467 (93.4% relative)
