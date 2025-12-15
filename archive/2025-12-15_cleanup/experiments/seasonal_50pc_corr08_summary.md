# Seasonal Combined Datasets - 50% Target Ratio - Model Performance (with Correlation Filtering, threshold=0.8)

**Date:** 2025-12-13 23:47:38

## Dataset Split

- **Training:** All seasons (2000-2025) with 50% target ratio, combined (27,618 records, 50.0% injury ratio)
- **Validation:** None (trained on all data)
- **Test:** Season 2025-2026 with natural target ratio (153,006 records, 0.6% injury ratio)

## Target

- **Injury Type:** Muscular injuries only

## Approach

- **Correlation filtering:** Threshold = 0.8 (removed one feature from each highly correlated pair)
- **Features:** 1474 features (after correlation filtering, down from 1746 initial features)
- **Models:** Random Forest, Gradient Boosting, XGBoost, and LightGBM

## Performance Metrics

| Model | Dataset | Precision | Recall | F1-Score | ROC AUC | Gini |
|-------|---------|-----------|--------|----------|---------|------|
| **RF** | Training | 0.8305 | 0.9439 | 0.8835 | 0.9618 | 0.9236 |
| | Test | 0.0165 | 0.7125 | 0.0323 | 0.7871 | 0.5741 |
| | | | | | | |
| **GB** | Training | 0.9754 | 0.9978 | 0.9865 | 0.9996 | 0.9992 |
| | Test | 0.0222 | 0.5750 | 0.0427 | 0.7857 | 0.5713 |
| | | | | | | |
| **XGB** | Training | 0.9058 | 1.0000 | 0.9506 | 1.0000 | 1.0000 |
| | Test | 0.0150 | 0.7188 | 0.0293 | 0.7953 | 0.5905 |
| | | | | | | |
| **LGBM** | Training | 0.9046 | 0.9843 | 0.9428 | 0.9881 | 0.9761 |
| | Test | 0.0197 | 0.6438 | 0.0382 | 0.7969 | 0.5939 |
| | | | | | | |

## Performance Gaps

### RF
- **F1 Gap (Train → Test):** 0.8513 (96.3% relative)

### GB
- **F1 Gap (Train → Test):** 0.9437 (95.7% relative)

### XGB
- **F1 Gap (Train → Test):** 0.9212 (96.9% relative)

### LGBM
- **F1 Gap (Train → Test):** 0.9045 (95.9% relative)
