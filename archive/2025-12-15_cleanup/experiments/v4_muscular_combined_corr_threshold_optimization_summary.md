# Muscular Injuries Only - Combined Train+Val Models (with Correlation Filtering) - Threshold Optimization Results

**Date:** 2025-12-10 08:59:08

## Test Dataset

- **Records:** 108,272
- **Injury ratio:** 0.9%
- **Target:** Muscular injuries only

## Best Operating Points

### RF - Best F1-Score
- **Threshold:** 0.600
- **Precision:** 0.0429
- **Recall:** 0.2606
- **F1-Score:** 0.0736
- **Accuracy:** 0.9400
- **ROC AUC:** 0.7946
- **Gini:** 0.5892
- **TP:** 258, **FP:** 5759, **TN:** 101523, **FN:** 732

### GB - Best F1-Score
- **Threshold:** 0.140
- **Precision:** 0.0404
- **Recall:** 0.2293
- **F1-Score:** 0.0687
- **Accuracy:** 0.9431
- **ROC AUC:** 0.7760
- **Gini:** 0.5520
- **TP:** 227, **FP:** 5394, **TN:** 101888, **FN:** 763

## Performance Comparison Table

| Model | Threshold | Precision | Recall | F1-Score | Accuracy | ROC AUC | Gini | TP | FP | TN | FN |
|-------|-----------|-----------|--------|----------|----------|---------|------|----|----|----|----|
| **RF** | 0.600 | 0.0429 | 0.2606 | 0.0736 | 0.9400 | 0.7946 | 0.5892 | 258 | 5759 | 101523 | 732 |
| **GB** | 0.140 | 0.0404 | 0.2293 | 0.0687 | 0.9431 | 0.7760 | 0.5520 | 227 | 5394 | 101888 | 763 |