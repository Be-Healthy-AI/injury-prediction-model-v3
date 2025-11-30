# Threshold Optimization - Phase 2 Models

**Date:** 2025-11-27
**Models:** RF, GB, LR (Phase 2)
**Validation Dataset:** Out-of-sample (12,218 records, 1.8% injury ratio)

## RF Model

### Best F1-Score

- **Threshold:** 0.400
- **Precision:** 0.0872
- **Recall:** 0.1364
- **F1-Score:** 0.1064
- **ROC AUC:** 0.7503
- **Confusion Matrix:** TP=30, FP=314, TN=11684, FN=190

### Best Precision (Recall > 0.05)

- **Threshold:** 0.600
- **Precision:** 0.1128
- **Recall:** 0.0682
- **F1-Score:** 0.0850

### Best Recall (Precision > 0.05)

- **Threshold:** 0.100
- **Precision:** 0.0523
- **Recall:** 0.5727
- **F1-Score:** 0.0958

## GB Model

### Best F1-Score

- **Threshold:** 0.700
- **Precision:** 0.3625
- **Recall:** 0.1318
- **F1-Score:** 0.1933
- **ROC AUC:** 0.7255
- **Confusion Matrix:** TP=29, FP=51, TN=11947, FN=191

### Best Balanced (Precision > 0.1, Recall > 0.1)

- **Threshold:** 0.700
- **Precision:** 0.3625
- **Recall:** 0.1318
- **F1-Score:** 0.1933

### Best Precision (Recall > 0.05)

- **Threshold:** 0.900
- **Precision:** 0.7222
- **Recall:** 0.0591
- **F1-Score:** 0.1092

### Best Recall (Precision > 0.05)

- **Threshold:** 0.060
- **Precision:** 0.0512
- **Recall:** 0.4318
- **F1-Score:** 0.0915

## LR Model

### Best F1-Score

- **Threshold:** 0.300
- **Precision:** 0.0310
- **Recall:** 0.3591
- **F1-Score:** 0.0570
- **ROC AUC:** 0.6530
- **Confusion Matrix:** TP=79, FP=2472, TN=9526, FN=141

### Best Precision (Recall > 0.05)

- **Threshold:** 0.300
- **Precision:** 0.0310
- **Recall:** 0.3591
- **F1-Score:** 0.0570

