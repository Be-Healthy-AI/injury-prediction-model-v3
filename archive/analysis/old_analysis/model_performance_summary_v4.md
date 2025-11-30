# Model Performance Summary - V4 Feature Engineering

**Date:** 2025-11-26  
**Changes:** Week 5 fix + Normalized cumulative features

---

## Quick Comparison: Out-of-Sample Performance

### Random Forest

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **Precision** | 0.5000 | **0.6333** | **+26.7%** ✅ |
| **Recall** | 0.1318 | **0.0864** | **-34.4%** ❌ |
| **F1-Score** | 0.2086 | **0.1520** | **-27.1%** ❌ |
| **ROC AUC** | 0.7633 | **0.7703** | **+0.7%** ✅ |

**Verdict:** Precision improved significantly, but recall decreased. Model is more conservative.

---

### Gradient Boosting

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **Precision** | 0.6190 | **0.7500** | **+21.2%** ✅ |
| **Recall** | 0.1773 | **0.0545** | **-69.3%** ❌❌ |
| **F1-Score** | 0.2756 | **0.1017** | **-63.1%** ❌❌ |
| **ROC AUC** | 0.7346 | **0.7382** | **+0.4%** ✅ |

**Verdict:** Precision improved, but recall dropped dramatically. Model is very conservative.

---

### Logistic Regression

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **Precision** | 0.1922 | **0.1975** | **+2.8%** ✅ |
| **Recall** | 0.8500 | **0.8591** | **+0.9%** ✅ |
| **F1-Score** | 0.3135 | **0.3212** | **+0.8%** ✅ |
| **ROC AUC** | 0.6505 | **0.6642** | **+1.4%** ✅ |

**Verdict:** Best overall performance - maintains high recall while improving precision slightly.

---

## Summary

### ✅ What Improved:
1. **Precision increased** for all models (especially RF and GB)
2. **ROC AUC maintained/improved** - ranking ability preserved
3. **In-sample performance improved** across all models

### ❌ What Worsened:
1. **Recall decreased significantly** for RF and GB (models too conservative)
2. **F1-Score decreased** for RF and GB due to recall drop

### 🎯 Best Model for Out-of-Sample:
**Logistic Regression** - Best balance:
- **Recall: 85.9%** (catches most injuries)
- **F1-Score: 0.3212** (best among all models)
- **Precision: 19.8%** (acceptable given high recall)

---

## Recommendation

**The feature engineering changes made models more conservative.** To improve recall:

1. **Optimize threshold** - Lower from 0.5 to 0.3-0.4 for RF/GB
2. **Use Logistic Regression** for production (best recall)
3. **Consider ensemble** - Combine LR (high recall) with GB (high precision)

## Threshold Optimization & Ensembles (2025-11-27)

We executed the threshold/ensemble plan and documented the sweep in `analysis/threshold_optimization_v4.md`. Highlights (out-of-sample metrics):

- **Random Forest @ threshold 0.25:** Precision 0.337 • Recall 0.605 • F1 0.433 (up from 0.152) – balanced operating point without retraining.
- **Random Forest high-recall mode @ 0.20:** Precision 0.280 • Recall 0.777 – captures ~78% of injuries with tolerable precision.
- **Logistic Regression @ 0.60:** Precision 0.212 • Recall 0.759 • F1 0.331 – improves precision over the 0.50 default without losing much recall.
- **Ensemble LR (60%) + RF (40%) @ 0.50:** Precision 0.249 • Recall 0.627 • F1 0.357 – best blended F1, marrying LR recall with RF precision.
- **Same ensemble @ 0.30:** Precision 0.185 • Recall 0.923 – extreme recall mode (captures >92% of injuries) for monitoring use cases.

These operating points implement the second half of the plan: we now have concrete thresholds per model plus a practical LR+RF weighted ensemble configuration to restore recall while keeping precision serviceable.


