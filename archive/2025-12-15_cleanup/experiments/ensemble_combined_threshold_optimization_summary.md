# GB + RF Ensemble - Combined Train+Val - Threshold Optimization Results

**Date:** 2025-11-30 15:19:39

## Best Operating Points by Ensemble Method (Test Dataset)

| Ensemble Method | Config | Precision | Recall | F1-Score | TP | FP | TN | FN |
|----------------|--------|-----------|--------|----------|----|----|----|----|
| **Weighted Avg 70Gb 30Rf** | Threshold: 0.15 | 0.3676 | 0.3091 | 0.3358 | 68 | 117 | 2413 | 152 |
| **Weighted Avg 60Gb 40Rf** | Threshold: 0.20 | 0.4150 | 0.2773 | 0.3324 | 61 | 86 | 2444 | 159 |
| **Geometric Mean** | Threshold: 0.15 | 0.3568 | 0.3000 | 0.3259 | 66 | 119 | 2411 | 154 |
| **Or Gate** | GB: 0.30, RF: 0.40 | 0.5200 | 0.2364 | 0.3250 | 52 | 48 | 2482 | 168 |
| **Weighted Avg 50Gb 50Rf** | Threshold: 0.15 | 0.3333 | 0.3136 | 0.3232 | 69 | 138 | 2392 | 151 |
| **Weighted Avg 40Gb 60Rf** | Threshold: 0.15 | 0.3209 | 0.3136 | 0.3172 | 69 | 146 | 2384 | 151 |
| **Weighted Avg 30Gb 70Rf** | Threshold: 0.15 | 0.3108 | 0.3136 | 0.3122 | 69 | 153 | 2377 | 151 |
| **And Gate** | GB: 0.30, RF: 0.30 | 0.4130 | 0.0864 | 0.1429 | 19 | 27 | 2503 | 201 |