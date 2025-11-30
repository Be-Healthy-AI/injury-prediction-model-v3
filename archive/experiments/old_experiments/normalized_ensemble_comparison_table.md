# Ensemble Optimization - Comparison Table

**Date:** 2025-11-27

## Best Ensemble Configurations

| Ensemble | Threshold | Precision | Recall | F1-Score | ROC AUC | TP | FP | TN | FN |
|----------|-----------|-----------|--------|----------|---------|----|----|----|----|
| **GB_only** | **0.060** | **0.1679** | **0.2000** | **0.1826** | 0.7500 | 44 | 218 | 11780 | 176 |
| GB_only (Balanced) | 0.060 | 0.1679 | 0.2000 | 0.1826 | 0.7500 | 44 | 218 | 11780 | 176 |
| RF_70_GB_30 (Precision) | 0.450 | 0.2295 | 0.0636 | 0.0996 | 0.7863 | 14 | 47 | 11951 | 206 |
| RF_50_GB_50 (Recall) | 0.120 | 0.0501 | 0.6727 | 0.0932 | 0.7891 | 148 | 2809 | 9189 | 72 |

## Comparison with Individual Models (Best Thresholds)

| Model/Ensemble | Threshold | Precision | Recall | F1-Score |
|----------------|-----------|-----------|--------|----------|
| RF (best) | 0.500 | 0.1377 | 0.0864 | 0.1061 |
| GB (best) | 0.060 | 0.1679 | 0.2000 | 0.1826 |
| LR (best) | 0.500 | 0.0244 | 0.8591 | 0.0474 |
| **GB_only (best)** | **0.060** | **0.1679** | **0.2000** | **0.1826** |
