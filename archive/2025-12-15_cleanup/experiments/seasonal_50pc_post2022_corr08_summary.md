# Seasonal Combined Datasets - 10% Target Ratio - Model Performance (with Correlation Filtering, threshold=0.8)

**Date:** 2025-12-14 14:15:53

## Dataset Split

- **Training:** All seasons (2000-2025) with 50% target ratio, combined (11,690 records, 50.0% injury ratio)
- **Validation:** None (trained on all data)
- **Test:** Season 2025-2026 with natural target ratio (153,006 records, 0.6% injury ratio)

## Target

- **Injury Type:** Muscular injuries only

## Approach

- **Correlation filtering:** Threshold = 0.8 (removed one feature from each highly correlated pair)
- **Features:** 1340 features (after correlation filtering, down from 1636 initial features)
- **Models:** Random Forest, Gradient Boosting, XGBoost, and LightGBM

## Performance Metrics

| Model | Dataset | Precision | Recall | F1-Score | ROC AUC | Gini |
|-------|---------|-----------|--------|----------|---------|------|
| **RF** | Training | 0.8746 | 0.9644 | 0.9173 | 0.9819 | 0.9638 |
| | Test | 0.0202 | 0.6438 | 0.0392 | 0.7904 | 0.5809 |
| | | | | | | |
| **GB** | Training | 0.9998 | 1.0000 | 0.9999 | 1.0000 | 1.0000 |
| | Test | 0.0272 | 0.4375 | 0.0513 | 0.7844 | 0.5688 |
| | | | | | | |
| **XGB** | Training | 0.9804 | 1.0000 | 0.9901 | 1.0000 | 1.0000 |
| | Test | 0.0176 | 0.5938 | 0.0342 | 0.7871 | 0.5742 |
| | | | | | | |
| **LGBM** | Training | 0.9868 | 1.0000 | 0.9934 | 1.0000 | 0.9999 |
| | Test | 0.0233 | 0.4875 | 0.0445 | 0.7739 | 0.5478 |
| | | | | | | |

## Performance Gaps

### RF
- **F1 Gap (Train → Test):** 0.8781 (95.7% relative)

### GB
- **F1 Gap (Train → Test):** 0.9486 (94.9% relative)

### XGB
- **F1 Gap (Train → Test):** 0.9559 (96.5% relative)

### LGBM
- **F1 Gap (Train → Test):** 0.9489 (95.5% relative)
