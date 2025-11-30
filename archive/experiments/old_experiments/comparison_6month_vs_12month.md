# Comparison: 6-Month vs 12-Month Recent Negative Sampling

**Date:** 2025-11-27  
**Comparison:** Out-of-sample validation performance using different recent negative sampling windows

---

## Out-of-Sample Validation Results (Threshold = 0.5)

### Random Forest

| Metric | 6-Month Window | 12-Month Window (Fixed) | Difference |
|--------|----------------|-------------------------|------------|
| **Precision** | 0.1578 | 0.0000 | -0.1578 ❌ |
| **Recall** | 1.0000 | 0.0000 | -1.0000 ❌ |
| **F1-Score** | 0.2726 | 0.0000 | -0.2726 ❌ |
| **ROC AUC** | 0.6023 | 0.6777 | +0.0754 ✅ |
| **Accuracy** | 0.1997 | 0.8425 | +0.6428 ✅ |
| **TP** | 220 | 0 | -220 ❌ |
| **FP** | 1246 | 11 | -1235 ✅ |
| **FN** | 0 | 220 | +220 ❌ |
| **TN** | 1 | 1236 | +1235 ✅ |

**Analysis:**
- The 12-month window made the model extremely conservative - it predicted **zero positives** (TP=0)
- While ROC AUC improved slightly (0.68 vs 0.60), the model is unusable at threshold 0.5
- The 6-month window had very high recall (100%) but very low precision (15.8%)

---

### Gradient Boosting

| Metric | 6-Month Window | 12-Month Window (Fixed) | Difference |
|--------|----------------|-------------------------|------------|
| **Precision** | 0.1568 | 0.3333 | +0.1765 ✅ |
| **Recall** | 0.9909 | 0.0227 | -0.9682 ❌ |
| **F1-Score** | 0.2708 | 0.0426 | -0.2282 ❌ |
| **ROC AUC** | 0.5955 | 0.6883 | +0.0928 ✅ |
| **Accuracy** | 0.1997 | 0.8466 | +0.6469 ✅ |
| **TP** | 218 | 5 | -213 ❌ |
| **FP** | 1247 | 10 | -1237 ✅ |
| **FN** | 2 | 215 | +213 ❌ |
| **TN** | 0 | 1237 | +1237 ✅ |

**Analysis:**
- The 12-month window also made GB very conservative, but it did predict 5 true positives
- Precision improved significantly (33.3% vs 15.7%), but recall dropped dramatically (2.3% vs 99.1%)
- ROC AUC improved (0.69 vs 0.60), indicating better discrimination ability
- At threshold 0.5, the model is too conservative to be useful

---

## Key Observations

### 1. **Model Behavior Shift**
- **6-Month:** Models predict almost everything as positive (very high recall, very low precision)
- **12-Month:** Models predict almost nothing as positive (very low recall, better precision when they do predict)

### 2. **ROC AUC Improvement**
- Both models show improved ROC AUC with 12-month window (RF: +0.08, GB: +0.09)
- This suggests better **discrimination ability**, but the threshold needs adjustment

### 3. **Threshold Issue**
- At threshold 0.5, both windows show poor balance
- **6-Month:** Too many false positives (overly sensitive)
- **12-Month:** Too many false negatives (overly conservative)
- **Solution:** Threshold optimization needed for 12-month window

### 4. **Training Data Distribution**
- **6-Month:** Non-injury timelines from last 6 months (2025-01-01 to 2025-06-30)
- **12-Month:** Non-injury timelines from last 12 months (2024-07-01 to 2025-06-30)
- The wider window provides more diverse negative examples, making the model more conservative

---

## Recommendations

1. **For 12-Month Window:**
   - Lower the threshold significantly (e.g., 0.2-0.3) to recover recall
   - The improved ROC AUC suggests the model can achieve better precision/recall balance with threshold tuning

2. **For 6-Month Window:**
   - Raise the threshold (e.g., 0.6-0.7) to reduce false positives
   - Or use ensemble methods to improve precision

3. **Best Approach:**
   - Use 12-month window with optimized thresholds
   - The better ROC AUC indicates superior model quality
   - Threshold optimization should recover recall while maintaining better precision than 6-month window

---

**Note:** The 12-month experiment was rerun after fixing the script bug where `RECENT_NEGATIVE_MONTHS` was hardcoded to 6, overriding the environment variable.

