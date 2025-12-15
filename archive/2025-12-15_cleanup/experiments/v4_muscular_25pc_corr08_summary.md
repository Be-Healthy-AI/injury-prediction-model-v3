# V4 Timeline Datasets - Muscular Injuries Only - 25% TARGET RATIO Model Performance (with Correlation Filtering, threshold=0.8)

**Date:** 2025-12-11 23:42:38

## Dataset Split

- **Training:** 2021-07-01 to 2024-06-30 (19,780 records, 25.0% injury ratio)
- **Validation:** 2024-07-01 to 2025-06-30 (9,780 records, 25.0% injury ratio)
- **Test:** >= 2025-07-01 (153,006 records, 0.6% injury ratio)

## Target

- **Injury Type:** Muscular injuries only

## Approach

- **Correlation filtering:** Threshold = 0.8 (removed one feature from each highly correlated pair)
- **Features:** 1219 features (after correlation filtering, down from 1484 initial features)
- **Models:** Random Forest, Gradient Boosting, XGBoost, and LightGBM

## Performance Metrics

| Model | Dataset | Precision | Recall | F1-Score | ROC AUC | Gini |
|-------|---------|-----------|--------|----------|---------|------|
| **RF** | Training | 0.7619 | 0.9941 | 0.8627 | 0.9932 | 0.9864 |
| | Validation | 0.5313 | 0.5382 | 0.5347 | 0.7903 | 0.5805 |
| | Test | 0.0256 | 0.5636 | 0.0489 | 0.7918 | 0.5835 |
| | | | | | | |
| **GB** | Training | 0.9994 | 1.0000 | 0.9997 | 1.0000 | 1.0000 |
| | Validation | 0.6562 | 0.1117 | 0.1908 | 0.7752 | 0.5505 |
| | Test | 0.0451 | 0.0939 | 0.0610 | 0.7641 | 0.5282 |
| | | | | | | |
| **XGB** | Training | 0.9845 | 1.0000 | 0.9922 | 1.0000 | 1.0000 |
| | Validation | 0.5883 | 0.3174 | 0.4123 | 0.7770 | 0.5541 |
| | Test | 0.0227 | 0.2293 | 0.0413 | 0.7594 | 0.5188 |
| | | | | | | |
| **LGBM** | Training | 0.9159 | 1.0000 | 0.9561 | 0.9998 | 0.9995 |
| | Validation | 0.5684 | 0.3431 | 0.4280 | 0.7679 | 0.5357 |
| | Test | 0.0271 | 0.3242 | 0.0501 | 0.7682 | 0.5363 |
| | | | | | | |

## Performance Gaps

### RF
- **F1 Gap (Train → Val):** 0.3279 (38.0% relative)
- **F1 Gap (Train → Test):** 0.8138 (94.3% relative)
- **F1 Gap (Val → Test):** 0.4858 (90.9% relative)

### GB
- **F1 Gap (Train → Val):** 0.8089 (80.9% relative)
- **F1 Gap (Train → Test):** 0.9387 (93.9% relative)
- **F1 Gap (Val → Test):** 0.1299 (68.0% relative)

### XGB
- **F1 Gap (Train → Val):** 0.5798 (58.4% relative)
- **F1 Gap (Train → Test):** 0.9509 (95.8% relative)
- **F1 Gap (Val → Test):** 0.3711 (90.0% relative)

### LGBM
- **F1 Gap (Train → Val):** 0.5282 (55.2% relative)
- **F1 Gap (Train → Test):** 0.9060 (94.8% relative)
- **F1 Gap (Val → Test):** 0.3779 (88.3% relative)
