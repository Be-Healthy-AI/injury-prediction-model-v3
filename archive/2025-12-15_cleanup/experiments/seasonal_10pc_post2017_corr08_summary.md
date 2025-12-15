# Seasonal Combined Datasets - 10% Target Ratio - Model Performance (with Correlation Filtering, threshold=0.8)

**Date:** 2025-12-14 13:01:47

## Dataset Split

- **Training:** All seasons (2000-2025) with 50% target ratio, combined (124,240 records, 10.0% injury ratio)
- **Validation:** None (trained on all data)
- **Test:** Season 2025-2026 with natural target ratio (153,006 records, 0.6% injury ratio)

## Target

- **Injury Type:** Muscular injuries only

## Approach

- **Correlation filtering:** Threshold = 0.8 (removed one feature from each highly correlated pair)
- **Features:** 1578 features (after correlation filtering, down from 1840 initial features)
- **Models:** Random Forest, Gradient Boosting, XGBoost, and LightGBM

## Performance Metrics

| Model | Dataset | Precision | Recall | F1-Score | ROC AUC | Gini |
|-------|---------|-----------|--------|----------|---------|------|
| **RF** | Training | 0.4424 | 0.9872 | 0.6110 | 0.9816 | 0.9631 |
| | Test | 0.0239 | 0.5813 | 0.0459 | 0.7915 | 0.5829 |
| | | | | | | |
| **GB** | Training | 0.9950 | 0.8199 | 0.8990 | 0.9985 | 0.9970 |
| | Test | 0.0750 | 0.0375 | 0.0500 | 0.7793 | 0.5586 |
| | | | | | | |
| **XGB** | Training | 0.8414 | 1.0000 | 0.9139 | 0.9999 | 0.9999 |
| | Test | 0.0376 | 0.2062 | 0.0636 | 0.7865 | 0.5730 |
| | | | | | | |
| **LGBM** | Training | 0.4475 | 0.9891 | 0.6162 | 0.9782 | 0.9564 |
| | Test | 0.0212 | 0.5188 | 0.0408 | 0.7904 | 0.5807 |
| | | | | | | |

## Performance Gaps

### RF
- **F1 Gap (Train → Test):** 0.5652 (92.5% relative)

### GB
- **F1 Gap (Train → Test):** 0.8490 (94.4% relative)

### XGB
- **F1 Gap (Train → Test):** 0.8503 (93.0% relative)

### LGBM
- **F1 Gap (Train → Test):** 0.5755 (93.4% relative)
