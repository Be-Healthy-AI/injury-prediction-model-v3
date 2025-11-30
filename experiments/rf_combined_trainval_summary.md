# RF Model - Combined Training + Validation - Performance Summary

**Date:** 2025-11-30 15:17:58

## Configuration

- **Training Dataset:** Combined train + validation (78,375 records, 8.0% injury ratio)
- **Test Dataset:** Test (2025/26 season) (2,750 records, 8.0% injury ratio)
- **Threshold:** 0.5

## Performance Metrics

| Dataset | Accuracy | Precision | Recall | F1-Score | ROC AUC | Gini | TP | FP | TN | FN |
|---------|----------|-----------|--------|----------|---------|------|----|----|----|----|
| **Train+Val** | 0.9774 | 0.8911 | 0.8180 | 0.8530 | 0.9930 | 0.9859 | 5129 | 627 | 71478 | 1141 |
| **Test (2025/26)** | 0.9215 | 0.8333 | 0.0227 | 0.0442 | 0.7628 | 0.5255 | 5 | 1 | 2529 | 215 |