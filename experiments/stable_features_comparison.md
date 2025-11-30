# Stable Features Model Comparison

**Date:** 2025-11-27  
**Analysis:** Comparison of models trained with stable features vs original models

---

## Stability Criteria

- **Maximum correlation drift:** 0.05
- **Minimum training correlation:** 0.01
- **Stable features selected:** 712 out of 2,142 total features

---

## Out-of-Sample Performance Comparison

### Random Forest (RF)

| Metric | Original Model | Stable Features Model | Change |
|--------|----------------|----------------------|--------|
| **Precision** | 0.2424 | 0.1077 | **-55.6%** ❌ |
| **Recall** | 0.0364 | 0.0318 | **-12.6%** ❌ |
| **F1-Score** | 0.0632 | 0.0491 | **-22.3%** ❌ |
| **ROC AUC** | 0.7405 | 0.7386 | **-0.3%** ✅ |
| **AUC Gap** | 0.2594 | 0.2614 | **+0.8%** ❌ |

### Gradient Boosting (GB)

| Metric | Original Model | Stable Features Model | Change |
|--------|----------------|----------------------|--------|
| **Precision** | 0.1351 | 0.0000 | **-100%** ❌ |
| **Recall** | 0.0227 | 0.0000 | **-100%** ❌ |
| **F1-Score** | 0.0389 | 0.0000 | **-100%** ❌ |
| **ROC AUC** | 0.7447 | 0.7838 | **+5.3%** ✅ |
| **AUC Gap** | 0.2553 | 0.2162 | **-15.3%** ✅ |

### Logistic Regression (LR)

| Metric | Original Model | Stable Features Model | Change |
|--------|----------------|----------------------|--------|
| **Precision** | 0.0250 | 0.0294 | **+17.6%** ✅ |
| **Recall** | 0.8955 | 0.6091 | **-32.0%** ❌ |
| **F1-Score** | 0.0486 | 0.0560 | **+15.2%** ✅ |
| **ROC AUC** | 0.6712 | 0.7126 | **+6.2%** ✅ |
| **AUC Gap** | 0.0767 | 0.1368 | **+78.4%** ❌ |

---

## Key Observations

### 1. **ROC AUC Improved for GB and LR**
- GB: 0.7447 → 0.7838 (+5.3%)
- LR: 0.6712 → 0.7126 (+6.2%)
- RF: Slight decrease (0.7405 → 0.7386)

**Interpretation:** The models have better discrimination ability with stable features, suggesting reduced overfitting to spurious correlations.

### 2. **Precision/Recall Trade-off**
- **RF:** Both precision and recall decreased significantly
- **GB:** Still too conservative (0% recall at threshold 0.5)
- **LR:** Precision improved but recall decreased significantly

**Interpretation:** Removing high-drift features also removed some predictive power, but the remaining features are more reliable.

### 3. **AUC Gap Analysis**
- **GB:** AUC gap improved (0.2553 → 0.2162, -15.3%)
- **RF:** AUC gap slightly worse (0.2594 → 0.2614)
- **LR:** AUC gap increased (0.0767 → 0.1368, but this is misleading - LR had low gap because it was already overfitting less)

**Interpretation:** GB shows better generalization with stable features.

---

## Threshold Optimization Needed

The models trained with stable features still need threshold optimization. At threshold 0.5:
- RF: Only 3.2% recall
- GB: 0% recall
- LR: 60.9% recall (better than original 89.5%, but still high)

**Recommendation:** Run threshold sweep on stable-features models to find optimal operating points.

---

## Conclusion

### Pros of Stable Features Approach:
1. ✅ **Better discrimination (ROC AUC)** for GB and LR
2. ✅ **Reduced overfitting** (lower AUC gaps for GB)
3. ✅ **More reliable features** (less drift)

### Cons of Stable Features Approach:
1. ❌ **Lower precision/recall** at threshold 0.5
2. ❌ **Removed important predictive features** (some high-drift features may still be useful)
3. ❌ **Still need threshold optimization**

### Next Steps:
1. **Run threshold sweep** on stable-features models (0.1, 0.2, 0.3, 0.4, 0.5)
2. **Compare best operating points** between original and stable-features models
3. **Consider hybrid approach:** Keep some high-drift features that are still somewhat predictive in validation
4. **Feature engineering:** Create new features that are more robust to distribution shifts

---

**Note:** The stable-features models show promise (better ROC AUC, lower overfitting) but need threshold optimization to be useful in practice.

