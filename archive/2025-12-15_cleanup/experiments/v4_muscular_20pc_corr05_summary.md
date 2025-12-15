# V4 Timeline Datasets - Muscular Injuries Only - 20% Target Ratio Model Performance (with Correlation Filtering, threshold=0.5)

**Date:** 2025-12-11 20:02:50

## Dataset Split

- **Training:** 2021-07-01 to 2024-06-30 (24,725 records, 20.0% injury ratio)
- **Validation:** 2024-07-01 to 2025-06-30 (12,225 records, 20.0% injury ratio)
- **Test:** >= 2025-07-01 (153,006 records, 0.6% injury ratio)

## Target

- **Injury Type:** Muscular injuries only

## Approach

- **Correlation filtering:** Threshold = 0.5 (removed one feature from each highly correlated pair)
- **Features:** 1025 features (after correlation filtering, down from 1509 initial features)
- **Models:** Random Forest, Gradient Boosting, XGBoost, and LightGBM

## Performance Metrics

| Model | Dataset | Precision | Recall | F1-Score | ROC AUC | Gini |
|-------|---------|-----------|--------|----------|---------|------|
| **RF** | Training | 0.5500 | 0.9592 | 0.6991 | 0.9591 | 0.9183 |
| | Validation | 0.3954 | 0.6798 | 0.5000 | 0.7815 | 0.5631 |
| | Test | 0.0188 | 0.6636 | 0.0366 | 0.7895 | 0.5791 |
| | | | | | | |
| **GB** | Training | 0.9811 | 0.9252 | 0.9523 | 0.9983 | 0.9966 |
| | Validation | 0.6085 | 0.1583 | 0.2512 | 0.7822 | 0.5643 |
| | Test | 0.0308 | 0.0970 | 0.0467 | 0.7638 | 0.5276 |
| | | | | | | |
| **XGB** | Training | 0.9373 | 1.0000 | 0.9676 | 1.0000 | 1.0000 |
| | Validation | 0.4893 | 0.3002 | 0.3721 | 0.7653 | 0.5306 |
| | Test | 0.0271 | 0.2919 | 0.0495 | 0.7590 | 0.5181 |
| | | | | | | |
| **LGBM** | Training | 0.7926 | 0.9998 | 0.8842 | 0.9979 | 0.9958 |
| | Validation | 0.4719 | 0.4368 | 0.4537 | 0.7672 | 0.5345 |
| | Test | 0.0208 | 0.3545 | 0.0393 | 0.7693 | 0.5385 |
| | | | | | | |

## Performance Gaps

### RF
- **F1 Gap (Train → Val):** 0.1991 (28.5% relative)
- **F1 Gap (Train → Test):** 0.6625 (94.8% relative)
- **F1 Gap (Val → Test):** 0.4634 (92.7% relative)

### GB
- **F1 Gap (Train → Val):** 0.7011 (73.6% relative)
- **F1 Gap (Train → Test):** 0.9056 (95.1% relative)
- **F1 Gap (Val → Test):** 0.2045 (81.4% relative)

### XGB
- **F1 Gap (Train → Val):** 0.5955 (61.5% relative)
- **F1 Gap (Train → Test):** 0.9181 (94.9% relative)
- **F1 Gap (Val → Test):** 0.3226 (86.7% relative)

### LGBM
- **F1 Gap (Train → Val):** 0.4305 (48.7% relative)
- **F1 Gap (Train → Test):** 0.8449 (95.6% relative)
- **F1 Gap (Val → Test):** 0.4144 (91.3% relative)
