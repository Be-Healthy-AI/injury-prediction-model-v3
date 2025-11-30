# GB Model - Combined Training + Validation - Performance Summary

**Date:** 2025-11-30 15:09:03

## Configuration

- **Training Dataset:** Combined train + validation (78,375 records, 8.0% injury ratio)
- **Test Dataset:** Test (2025/26 season) (2,750 records, 8.0% injury ratio)
- **Threshold:** 0.5

## Performance Metrics

| Dataset | Accuracy | Precision | Recall | F1-Score | ROC AUC | Gini | TP | FP | TN | FN |
|---------|----------|-----------|--------|----------|---------|------|----|----|----|----|
| **Train+Val** | 0.9978 | 0.9942 | 0.9780 | 0.9860 | 0.9998 | 0.9996 | 6132 | 36 | 72069 | 138 |
| **Test (2025/26)** | 0.9225 | 0.5778 | 0.1182 | 0.1962 | 0.7402 | 0.4804 | 26 | 19 | 2511 | 194 |

## Comparison with Previous Model (Trained on Training Only)

| Model | Precision | Recall | F1-Score | Improvement |
|-------|-----------|--------|----------|-------------|
| **Previous (Train only)** | 0.4156 | 0.1455 | 0.2155 | Baseline |
| **Current (Train+Val)** | 0.5778 | 0.1182 | 0.1962 | F1: -8.9% |