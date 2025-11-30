# Ensemble Optimization - Normalized Cumulative Features

**Date:** 2025-11-27
**Models:** RF, GB, LR with normalized cumulative features
**Validation Dataset:** Out-of-sample (12,218 records, 1.8% injury ratio)

## Best Operating Points

### Best F1-Score

- **Ensemble:** GB_only
- **Threshold:** 0.060
- **Precision:** 0.1679
- **Recall:** 0.2000
- **F1-Score:** 0.1826
- **ROC AUC:** 0.7500
- **Confusion Matrix:** TP=44, FP=218, TN=11780, FN=176

### Best Balanced (Precision > 0.1, Recall > 0.1)

- **Ensemble:** GB_only
- **Threshold:** 0.060
- **Precision:** 0.1679
- **Recall:** 0.2000
- **F1-Score:** 0.1826

### Best Precision (Recall > 0.05)

- **Ensemble:** RF_70_GB_30
- **Threshold:** 0.450
- **Precision:** 0.2295
- **Recall:** 0.0636
- **F1-Score:** 0.0996

### Best Recall (Precision > 0.05)

- **Ensemble:** RF_50_GB_50
- **Threshold:** 0.120
- **Precision:** 0.0501
- **Recall:** 0.6727
- **F1-Score:** 0.0932

## Top 10 Ensembles by F1-Score

| Rank | Ensemble | Threshold | Precision | Recall | F1-Score | ROC AUC |
|------|----------|-----------|-----------|--------|----------|----------|
| 1 | GB_only | 0.060 | 0.1679 | 0.2000 | 0.1826 | 0.7500 |
| 2 | RF_AND_GB | 0.060 | 0.1679 | 0.2000 | 0.1826 | 0.7500 |
| 3 | GB_AND_LR | 0.060 | 0.1679 | 0.2000 | 0.1826 | 0.7500 |
| 4 | GB_only | 0.050 | 0.1523 | 0.2091 | 0.1762 | 0.7500 |
| 5 | RF_AND_GB | 0.050 | 0.1523 | 0.2091 | 0.1762 | 0.7500 |
| 6 | GB_AND_LR | 0.050 | 0.1523 | 0.2091 | 0.1762 | 0.7500 |
| 7 | GB_only | 0.080 | 0.1749 | 0.1773 | 0.1761 | 0.7500 |
| 8 | RF_AND_GB | 0.080 | 0.1749 | 0.1773 | 0.1761 | 0.7500 |
| 9 | GB_AND_LR | 0.080 | 0.1749 | 0.1773 | 0.1761 | 0.7500 |
| 10 | GB_only | 0.070 | 0.1667 | 0.1818 | 0.1739 | 0.7500 |
