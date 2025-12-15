# V4 Timeline Datasets - Muscular Injuries Only - 30% Target Ratio Model Performance (with Correlation Filtering, threshold=0.5)

**Date:** 2025-12-11 20:45:41

## Dataset Split

- **Training:** 2021-07-01 to 2024-06-30 (16,483 records, 30.0% injury ratio)
- **Validation:** 2024-07-01 to 2025-06-30 (8,150 records, 30.0% injury ratio)
- **Test:** >= 2025-07-01 (153,006 records, 0.6% injury ratio)

## Target

- **Injury Type:** Muscular injuries only

## Approach

- **Correlation filtering:** Threshold = 0.5 (removed one feature from each highly correlated pair)
- **Features:** 980 features (after correlation filtering, down from 1456 initial features)
- **Models:** Random Forest, Gradient Boosting, XGBoost, and LightGBM

## Performance Metrics

| Model | Dataset | Precision | Recall | F1-Score | ROC AUC | Gini |
|-------|---------|-----------|--------|----------|---------|------|
| **RF** | Training | 0.6631 | 0.9428 | 0.7786 | 0.9533 | 0.9066 |
| | Validation | 0.5297 | 0.7031 | 0.6042 | 0.7875 | 0.5750 |
| | Test | 0.0184 | 0.6970 | 0.0358 | 0.7926 | 0.5851 |
| | | | | | | |
| **GB** | Training | 0.9820 | 0.9905 | 0.9862 | 0.9996 | 0.9993 |
| | Validation | 0.6758 | 0.2712 | 0.3870 | 0.7761 | 0.5523 |
| | Test | 0.0294 | 0.2111 | 0.0516 | 0.7634 | 0.5269 |
| | | | | | | |
| **XGB** | Training | 0.9480 | 1.0000 | 0.9733 | 1.0000 | 1.0000 |
| | Validation | 0.5956 | 0.4331 | 0.5015 | 0.7702 | 0.5403 |
| | Test | 0.0209 | 0.4061 | 0.0398 | 0.7421 | 0.4842 |
| | | | | | | |
| **LGBM** | Training | 0.8912 | 1.0000 | 0.9424 | 0.9990 | 0.9980 |
| | Validation | 0.5903 | 0.4172 | 0.4889 | 0.7695 | 0.5390 |
| | Test | 0.0189 | 0.3475 | 0.0358 | 0.7527 | 0.5054 |
| | | | | | | |

## Performance Gaps

### RF
- **F1 Gap (Train → Val):** 0.1743 (22.4% relative)
- **F1 Gap (Train → Test):** 0.7428 (95.4% relative)
- **F1 Gap (Val → Test):** 0.5684 (94.1% relative)

### GB
- **F1 Gap (Train → Val):** 0.5992 (60.8% relative)
- **F1 Gap (Train → Test):** 0.9346 (94.8% relative)
- **F1 Gap (Val → Test):** 0.3354 (86.7% relative)

### XGB
- **F1 Gap (Train → Val):** 0.4718 (48.5% relative)
- **F1 Gap (Train → Test):** 0.9335 (95.9% relative)
- **F1 Gap (Val → Test):** 0.4618 (92.1% relative)

### LGBM
- **F1 Gap (Train → Val):** 0.4536 (48.1% relative)
- **F1 Gap (Train → Test):** 0.9067 (96.2% relative)
- **F1 Gap (Val → Test):** 0.4531 (92.7% relative)
