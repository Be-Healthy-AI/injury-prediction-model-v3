# Cascading Ensemble Approaches - Results

**Date:** 2025-11-30 15:34:13

## Best Cascading Configurations

### RF → GB (RF Screening, GB Filtering)

- **RF Threshold:** 0.25 (screening for high recall)
- **GB Threshold:** 0.70 (filtering for high precision)
- **Precision:** 0.6000
- **Recall:** 0.0136
- **F1-Score:** 0.0267
- **TP:** 3, **FP:** 2, **TN:** 2528, **FN:** 217

### GB → RF (GB Screening, RF Filtering)

- **GB Threshold:** 0.10 (screening for high recall)
- **RF Threshold:** 0.60 (filtering for high precision)
- **Precision:** 0.0000
- **Recall:** 0.0000
- **F1-Score:** 0.0000
- **TP:** 0, **FP:** 0, **TN:** 2530, **FN:** 220

## Comparison with Existing Ensemble Methods

| Method | Precision | Recall | F1-Score | TP | FP | TN | FN |
|--------|-----------|--------|----------|----|----|----|----|
| **Weighted Avg 70Gb 30Rf (Weighted Avg)** | 0.3676 | 0.3091 | 0.3358 | 68 | 117 | 2413 | 152 |
| **RF → GB (Cascading)** | 0.6000 | 0.0136 | 0.0267 | 3 | 2 | 2528 | 217 |
| **GB → RF (Cascading)** | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 2530 | 220 |