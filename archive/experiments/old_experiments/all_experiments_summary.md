# Out-of-Sample Validation Performance Summary

All experiments use:
- 10% target ratio for training
- Natural injury ratio (~1.8%) for validation
- Week-5 features included
- Correlation filter (threshold=0.80)
- Threshold: 0.5

## Performance Metrics (Out-of-Sample)

| Window | Model | Precision | Recall | F1-Score | ROC AUC | Gini | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|---|
| 24 months | RF | 0.2424 | 0.0364 | 0.0632 | 0.7405 | 0.4811 | 8 | 25 | 11973 | 212 |
| 24 months | GB | 0.1351 | 0.0227 | 0.0389 | 0.7447 | 0.4894 | 5 | 32 | 11966 | 215 |
| 24 months | LR | 0.0250 | 0.8955 | 0.0486 | 0.6712 | 0.3425 | 197 | 7692 | 4306 | 23 |
| 36 months | RF | 1.0000 | 0.0182 | 0.0357 | 0.7011 | 0.4022 | 4 | 0 | 11998 | 216 |
| 36 months | GB | 0.3333 | 0.0045 | 0.0090 | 0.7340 | 0.4681 | 1 | 2 | 11996 | 219 |
| 36 months | LR | 0.0222 | 0.6364 | 0.0429 | 0.5815 | 0.1630 | 140 | 6168 | 5830 | 80 |
| 48 months | RF | 1.0000 | 0.0045 | 0.0090 | 0.7076 | 0.4152 | 1 | 0 | 11998 | 219 |
| 48 months | GB | 0.0000 | 0.0000 | 0.0000 | 0.6965 | 0.3930 | 0 | 5 | 11993 | 220 |
| 48 months | LR | 0.0247 | 0.7500 | 0.0477 | 0.6100 | 0.2200 | 165 | 6528 | 5470 | 55 |
| 60 months | RF | 1.0000 | 0.0045 | 0.0090 | 0.7516 | 0.5032 | 1 | 0 | 11998 | 219 |
| 60 months | GB | 0.0833 | 0.0091 | 0.0164 | 0.7163 | 0.4327 | 2 | 22 | 11976 | 218 |
| 60 months | LR | 0.0243 | 0.7273 | 0.0471 | 0.6175 | 0.2351 | 160 | 6413 | 5585 | 60 |
| No limit | RF | 0.2424 | 0.0364 | 0.0632 | 0.7405 | 0.4811 | 8 | 25 | 11973 | 212 |
| No limit | GB | 0.1351 | 0.0227 | 0.0389 | 0.7447 | 0.4894 | 5 | 32 | 11966 | 215 |
| No limit | LR | 0.0250 | 0.8955 | 0.0486 | 0.6712 | 0.3425 | 197 | 7692 | 4306 | 23 |


## Detailed Confusion Matrices

### 24 months

**RF:**
- TP: 8, FP: 25, TN: 11973, FN: 212
**GB:**
- TP: 5, FP: 32, TN: 11966, FN: 215
**LR:**
- TP: 197, FP: 7692, TN: 4306, FN: 23

### 36 months

**RF:**
- TP: 4, FP: 0, TN: 11998, FN: 216
**GB:**
- TP: 1, FP: 2, TN: 11996, FN: 219
**LR:**
- TP: 140, FP: 6168, TN: 5830, FN: 80

### 48 months

**RF:**
- TP: 1, FP: 0, TN: 11998, FN: 219
**GB:**
- TP: 0, FP: 5, TN: 11993, FN: 220
**LR:**
- TP: 165, FP: 6528, TN: 5470, FN: 55

### 60 months

**RF:**
- TP: 1, FP: 0, TN: 11998, FN: 219
**GB:**
- TP: 2, FP: 22, TN: 11976, FN: 218
**LR:**
- TP: 160, FP: 6413, TN: 5585, FN: 60

### No limit

**RF:**
- TP: 8, FP: 25, TN: 11973, FN: 212
**GB:**
- TP: 5, FP: 32, TN: 11966, FN: 215
**LR:**
- TP: 197, FP: 7692, TN: 4306, FN: 23

