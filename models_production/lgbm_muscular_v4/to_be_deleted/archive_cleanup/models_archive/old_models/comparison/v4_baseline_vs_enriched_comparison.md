# V4 Baseline vs Enriched Models Comparison

**Generated:** 2026-01-21 18:14:35

## Overview

This report compares the performance of V4 baseline models (trained on original features) vs V4 enriched models (trained on Layer 2 enriched features).

## Configuration

| Setting | Baseline | Enriched |
|---------|----------|----------|
| Training Date | 2026-01-19T15:47:24.071996 | 2026-01-21T18:08:16.600013 |
| Min Season | 2018_2019 | 2018_2019 |
| Features | Original | Layer 2 enriched (workload, recovery, injury history) |
| Model 1 Features | N/A | N/A |
| Model 2 Features | N/A | N/A |

## Performance Metrics Comparison

### Model 1 (Muscular)

#### Train Dataset

| Metric | Baseline | Enriched | Change | % Change |
|--------|----------|----------|--------|----------|
| ACCURACY | 0.9901 | 0.9905 | 0.0004 | 0.04% |
| PRECISION | 0.3219 | 0.3320 | 0.0101 | 3.13% |
| RECALL | 1.0000 | 1.0000 | 0.0000 | 0.00% |
| F1 | 0.4871 | 0.4985 | 0.0114 | 2.35% |
| ROC_AUC | 1.0000 | 1.0000 | 0.0000 | 0.00% |
| GINI | 1.0000 | 1.0000 | 0.0000 | 0.00% |

#### Test Dataset

| Metric | Baseline | Enriched | Change | % Change |
|--------|----------|----------|--------|----------|
| ACCURACY | 0.9678 | 0.9708 | 0.0031 | 0.32% |
| PRECISION | 0.0422 | 0.0526 | 0.0104 | 24.59% |
| RECALL | 0.1233 | 0.1367 | 0.0133 | 10.81% |
| F1 | 0.0629 | 0.0759 | 0.0131 | 20.76% |
| ROC_AUC | 0.7740 | 0.7561 | -0.0179 | -2.31% |
| GINI | 0.5480 | 0.5122 | -0.0358 | -6.54% |

### Model 2 (Skeletal)

#### Train Dataset

| Metric | Baseline | Enriched | Change | % Change |
|--------|----------|----------|--------|----------|
| ACCURACY | 0.9964 | 0.9969 | 0.0005 | 0.05% |
| PRECISION | 0.4944 | 0.5338 | 0.0394 | 7.97% |
| RECALL | 1.0000 | 1.0000 | 0.0000 | 0.00% |
| F1 | 0.6617 | 0.6961 | 0.0344 | 5.20% |
| ROC_AUC | 1.0000 | 1.0000 | -0.0000 | -0.00% |
| GINI | 1.0000 | 1.0000 | -0.0000 | -0.00% |

#### Test Dataset

| Metric | Baseline | Enriched | Change | % Change |
|--------|----------|----------|--------|----------|
| ACCURACY | 0.9935 | 0.9927 | -0.0009 | -0.09% |
| PRECISION | 0.0488 | 0.0270 | -0.0218 | -44.59% |
| RECALL | 0.0545 | 0.0364 | -0.0182 | -33.33% |
| F1 | 0.0515 | 0.0310 | -0.0205 | -39.79% |
| ROC_AUC | 0.6802 | 0.6898 | 0.0095 | 1.40% |
| GINI | 0.3605 | 0.3795 | 0.0191 | 5.29% |

## Confusion Matrix Comparison

### Model 1 (Muscular)

#### Train Dataset

| Metric | Baseline | Enriched | Change |
|--------|----------|----------|--------|
| TP | 2,874 | 2,874 | +0 |
| FP | 6,053 | 5,782 | -271 |
| TN | 602,346 | 602,617 | +271 |
| FN | 0 | 0 | +0 |

#### Test Dataset

| Metric | Baseline | Enriched | Change |
|--------|----------|----------|--------|
| TP | 37 | 41 | +4 |
| FP | 840 | 739 | -101 |
| TN | 33,087 | 33,188 | +101 |
| FN | 263 | 259 | -4 |

### Model 2 (Skeletal)

#### Train Dataset

| Metric | Baseline | Enriched | Change |
|--------|----------|----------|--------|
| TP | 2,169 | 2,169 | +0 |
| FP | 2,218 | 1,894 | -324 |
| TN | 606,181 | 606,505 | +324 |
| FN | 0 | 0 | +0 |

#### Test Dataset

| Metric | Baseline | Enriched | Change |
|--------|----------|----------|--------|
| TP | 6 | 4 | -2 |
| FP | 117 | 144 | +27 |
| TN | 33,810 | 33,783 | -27 |
| FN | 104 | 106 | +2 |

## Summary

- **Average PRECISION improvement (Test):** -10.00%
- **Average F1 improvement (Test):** -9.52%
- **Average ROC_AUC improvement (Test):** -0.46%
- **Average GINI improvement (Test):** -0.62%

## Key Findings

1. **Feature Count:** Enriched models use Layer 2 features (workload, recovery, injury history)
2. **Training:** Both models use same hyperparameters and training configuration
3. **Test Set:** Both evaluated on 2025/26 season test dataset
