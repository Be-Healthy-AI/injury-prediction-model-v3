# Seasonal Combined Datasets - 50% Target Ratio - Model Performance (with Correlation Filtering, threshold=0.5)

**Date:** 2025-12-15 10:38:42

## Dataset Split

- **Training:** All seasons (2000-2025) with 50% target ratio, combined (27,618 records, 50.0% injury ratio)
- **Validation:** None (trained on all data)
- **Test:** Season 2025-2026 with natural target ratio (153,006 records, 0.6% injury ratio)

## Target

- **Injury Type:** Muscular injuries only

## Approach

- **Correlation filtering:** Threshold = 0.5 (removed one feature from each highly correlated pair)
- **Features:** 1243 features (after correlation filtering, down from 1746 initial features)
- **Models:** Random Forest, Gradient Boosting, XGBoost, and LightGBM

## Performance Metrics

| Model | Dataset | Precision | Recall | F1-Score | ROC AUC | Gini |
|-------|---------|-----------|--------|----------|---------|------|
| **RF** | Training | 0.7553 | 0.8697 | 0.8085 | 0.8998 | 0.7996 |
| | Test | 0.0144 | 0.7562 | 0.0283 | 0.7885 | 0.5771 |
| | | | | | | |

## Performance Gaps

### RF
- **F1 Gap (Train → Test):** 0.7802 (96.5% relative)
