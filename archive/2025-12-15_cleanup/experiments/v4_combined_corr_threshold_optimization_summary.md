# Combined Train+Val Models (with Correlation Filtering) - Threshold Optimization Results

**Date:** 2025-12-09 22:41:55

## Test Dataset

- **Records:** 109,022
- **Injury ratio:** 1.6%

## Best Operating Points

### RF - Best F1-Score
- **Threshold:** 0.600
- **Precision:** 0.0555
- **Recall:** 0.2678
- **F1-Score:** 0.0919
- **Accuracy:** 0.9155
- **ROC AUC:** 0.7610
- **Gini:** 0.5220
- **TP:** 466, **FP:** 7936, **TN:** 99346, **FN:** 1274

### GB - Best F1-Score
- **Threshold:** 0.160
- **Precision:** 0.0491
- **Recall:** 0.2667
- **F1-Score:** 0.0829
- **Accuracy:** 0.9058
- **ROC AUC:** 0.7296
- **Gini:** 0.4592
- **TP:** 464, **FP:** 8994, **TN:** 98288, **FN:** 1276

## Performance Comparison Table

| Model | Threshold | Precision | Recall | F1-Score | Accuracy | ROC AUC | Gini | TP | FP | TN | FN |
|-------|-----------|-----------|--------|----------|----------|---------|------|----|----|----|----|
| **RF** | 0.600 | 0.0555 | 0.2678 | 0.0919 | 0.9155 | 0.7610 | 0.5220 | 466 | 7936 | 99346 | 1274 |
| **GB** | 0.160 | 0.0491 | 0.2667 | 0.0829 | 0.9058 | 0.7296 | 0.4592 | 464 | 8994 | 98288 | 1276 |