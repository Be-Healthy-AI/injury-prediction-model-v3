# Gradient Boosting 12-Month Threshold Optimization

**Date:** 2025-11-27 20:11:20  
**Model:** Gradient Boosting trained on 12-month recent negative timelines  
**Validation Set:** 1,467 samples (Injury rate: 15.00%)

---

## Current Performance (Threshold = 0.5)

| Metric | Value |
|--------|-------|
| Precision | 0.3333 |
| Recall | 0.0227 |
| F1-Score | 0.0426 |
| ROC AUC | 0.6883 |

**Analysis:** Model is extremely conservative at threshold 0.5, predicting only 5 true positives out of 220 injuries.

---

## Optimal Thresholds

### 🎯 Best F1-Score

| Metric | Value |
|--------|-------|
| **Threshold** | 0.010 |
| **Precision** | 0.244 |
| **Recall** | 0.132 |
| **F1-Score** | 0.171 |
| **Accuracy** | 0.808 |
| **TP / FP / FN / TN** | 29.0 / 90.0 / 191.0 / 1157.0 |

### ⚖️ Best Balanced (Precision & Recall >= 0.15)

| Metric | Value |
|--------|-------|
| **Threshold** | 0.010 |
| **Precision** | 0.244 |
| **Recall** | 0.132 |
| **F1-Score** | 0.171 |
| **Accuracy** | 0.808 |
| **TP / FP / FN / TN** | 29.0 / 90.0 / 191.0 / 1157.0 |

### 📊 Best in Precision Range (0.20-0.30)

| Metric | Value |
|--------|-------|
| **Threshold** | 0.010 |
| **Precision** | 0.244 |
| **Recall** | 0.132 |
| **F1-Score** | 0.171 |
| **Accuracy** | 0.808 |
| **TP / FP / FN / TN** | 29.0 / 90.0 / 191.0 / 1157.0 |

### 📈 Best in Recall Range (0.50-0.70)

| Metric | Value |
|--------|-------|
| **Threshold** | 0.010 |
| **Precision** | 0.244 |
| **Recall** | 0.132 |
| **F1-Score** | 0.171 |
| **Accuracy** | 0.808 |
| **TP / FP / FN / TN** | 29.0 / 90.0 / 191.0 / 1157.0 |

---

## Threshold Sweep Summary

### Top 10 Thresholds by F1-Score


| Threshold | Precision | Recall | F1-Score | TP | FP | FN |
|-----------|-----------|--------|----------|----|----|----|
| 0.010 | 0.244 | 0.132 | 0.171 | 29 | 90 | 191 |
| 0.015 | 0.163 | 0.073 | 0.101 | 16 | 82 | 204 |
| 0.020 | 0.169 | 0.064 | 0.092 | 14 | 69 | 206 |
| 0.025 | 0.123 | 0.041 | 0.061 | 9 | 64 | 211 |
| 0.030 | 0.118 | 0.036 | 0.056 | 8 | 60 | 212 |
| 0.880 | 1.000 | 0.023 | 0.044 | 5 | 0 | 215 |
| 0.890 | 1.000 | 0.023 | 0.044 | 5 | 0 | 215 |
| 0.900 | 1.000 | 0.023 | 0.044 | 5 | 0 | 215 |
| 0.910 | 1.000 | 0.023 | 0.044 | 5 | 0 | 215 |
| 0.920 | 1.000 | 0.023 | 0.044 | 5 | 0 | 215 |


---

## Recommendations

Based on the threshold sweep:

1. **For Maximum F1-Score:** Use threshold **0.010**
   - Precision: 0.244, Recall: 0.132

2. **For Balanced Performance:** Use threshold **0.010**
   - Precision: 0.244, Recall: 0.132

3. **For Moderate Precision (20-30%):** Use threshold **0.010**
   - Precision: 0.244, Recall: 0.132

4. **For Good Recall (50-70%):** Use threshold **0.010**
   - Precision: 0.244, Recall: 0.132

---

**Note:** All metrics computed on out-of-sample validation set (temporal split, dates >= 2025-07-01).
