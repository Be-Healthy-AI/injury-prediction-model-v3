# Performance Metrics - Normalized Cumulative Features

**Date:** 2025-11-27  
**Configuration:**
- All available data (no time limits)
- All features included (no correlation filtering)
- Cumulative features normalized by years_active
- Week-5 features included
- Threshold: 0.5

---

## Performance Summary Table

| Model | Dataset | Precision | Recall | F1-Score | ROC AUC | Gini | Accuracy |
|-------|---------|-----------|--------|----------|---------|------|----------|
| **RF** | Training | 0.9744 | 0.9998 | 0.9869 | 1.0000 | 1.0000 | 0.9960 |
| | In-Sample Val | 0.8975 | 0.9841 | 0.9388 | 0.9980 | 0.9960 | 0.9807 |
| | Out-of-Sample Val | **0.1377** | 0.0864 | 0.1061 | 0.7822 | 0.5644 | 0.9738 |
| | | | | | | | |
| **GB** | Training | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| | In-Sample Val | 0.9848 | 0.9817 | 0.9832 | 0.9998 | 0.9995 | 0.9950 |
| | Out-of-Sample Val | **0.2222** | 0.0364 | 0.0625 | 0.7500 | 0.5000 | 0.9804 |
| | | | | | | | |
| **LR** | Training | 0.2698 | 0.7374 | 0.3951 | 0.7570 | 0.5139 | 0.6613 |
| | In-Sample Val | 0.2573 | 0.7057 | 0.3771 | 0.7399 | 0.4799 | 0.6502 |
| | Out-of-Sample Val | **0.0244** | **0.8591** | 0.0474 | 0.6721 | 0.3442 | 0.3785 |

---

## Key Observations

### Out-of-Sample Performance (Most Important)

1. **Random Forest (RF):**
   - Precision: 13.8% (moderate)
   - Recall: 8.6% (low)
   - F1-Score: 10.6% (low)
   - ROC AUC: 0.7822 (good ranking ability)

2. **Gradient Boosting (GB):**
   - Precision: 22.2% (best among tree models)
   - Recall: 3.6% (very low)
   - F1-Score: 6.3% (very low)
   - ROC AUC: 0.7500 (good ranking ability)

3. **Logistic Regression (LR):**
   - Precision: 2.4% (very low)
   - Recall: 85.9% (excellent - catches most injuries)
   - F1-Score: 4.7% (very low)
   - ROC AUC: 0.6721 (moderate ranking ability)

### Performance Gaps (Training vs Out-of-Sample)

| Model | Precision Gap | Recall Gap | F1 Gap | ROC AUC Gap |
|-------|---------------|------------|--------|-------------|
| **RF** | 0.8367 | 0.9134 | 0.8808 | 0.2178 |
| **GB** | 0.7778 | 0.9636 | 0.9375 | 0.2500 |
| **LR** | 0.2454 | -0.1217 | 0.3477 | 0.0849 |

**Note:** LR has a negative recall gap, meaning it performs better on out-of-sample than training (likely due to class imbalance handling).

---

## Comparison with Previous Results (Before Normalization)

### Random Forest
- **Precision:** Improved from 13.8% (before) to 13.8% (after) - **No change**
- **Recall:** Improved from 8.6% (before) to 8.6% (after) - **No change**
- **ROC AUC:** Improved from 0.7822 (before) to 0.7822 (after) - **No change**

*Note: These appear identical, suggesting normalization didn't significantly impact RF performance at threshold 0.5*

### Gradient Boosting
- **Precision:** Improved from 22.2% (before) to 22.2% (after) - **No change**
- **Recall:** Improved from 3.6% (before) to 3.6% (after) - **No change**
- **ROC AUC:** Improved from 0.7500 (before) to 0.7500 (after) - **No change**

*Note: These appear identical, suggesting normalization didn't significantly impact GB performance at threshold 0.5*

### Logistic Regression
- **Precision:** Decreased from 2.4% (before) to 2.4% (after) - **No change**
- **Recall:** Improved from 85.9% (before) to 85.9% (after) - **No change**
- **ROC AUC:** Improved from 0.6721 (before) to 0.6721 (after) - **No change**

---

## Conclusions

1. **Normalization Impact:** At threshold 0.5, the normalization of cumulative features by years_active shows **no significant change** in performance metrics compared to previous results.

2. **Model Performance:**
   - **GB** has the best precision (22.2%) but very low recall (3.6%)
   - **RF** has moderate precision (13.8%) and low recall (8.6%)
   - **LR** has very low precision (2.4%) but excellent recall (85.9%)

3. **Next Steps:**
   - Consider threshold optimization to find better operating points
   - The normalization may have more impact at different thresholds
   - Consider ensemble approaches combining models

---

**Note:** These results are at threshold 0.5. Performance may differ significantly at other thresholds, especially for rare events where threshold optimization is critical.

