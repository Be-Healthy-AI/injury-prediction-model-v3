# Model Performance Comparison: Before vs After Feature Engineering Improvements

**Date:** 2025-11-26  
**Changes Implemented:**
1. Fixed week 5 windowing (excluded season-specific features from week 5)
2. Normalized cumulative features (converted to rates/percentiles)
3. Removed severely drifted features

---

## Summary of Changes

### Feature Engineering Improvements:
- **Week 5 Fix:** Excluded `teams_this_season`, `national_team_this_season`, and `season_team_diversity` from week 5 windowing
- **Normalized Features Added:**
  - `career_matches_per_season`
  - `competitions_per_season`
  - `teams_per_season`
  - `cup_competitions_per_season`
  - `international_competitions_per_season`
  - `goals_per_career_match`
  - `assists_per_career_match`
  - `minutes_per_career_match`
  - `injuries_per_career_match`
  - `bench_rate`
  - `team_win_rate_normalized`
  - `team_loss_rate_normalized`
  - `team_draw_rate_normalized`

---

## Random Forest Model Comparison

### Training Set Performance

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **Accuracy** | 0.9912 | 0.9966 | **+0.54%** ✅ |
| **Precision** | 0.9450 | 0.9782 | **+3.32%** ✅ |
| **Recall** | 0.9996 | 0.9998 | **+0.02%** ✅ |
| **F1-Score** | 0.9715 | 0.9889 | **+1.74%** ✅ |
| **ROC AUC** | 0.9999 | 1.0000 | **+0.01%** ✅ |
| **Gini** | 0.9999 | 1.0000 | **+0.01%** ✅ |

### In-Sample Validation Performance

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **Accuracy** | 0.9715 | 0.9830 | **+1.15%** ✅ |
| **Precision** | 0.8608 | 0.9187 | **+5.79%** ✅ |
| **Recall** | 0.9665 | 0.9729 | **+0.64%** ✅ |
| **F1-Score** | 0.9106 | 0.9450 | **+3.44%** ✅ |
| **ROC AUC** | 0.9964 | 0.9983 | **+0.19%** ✅ |
| **Gini** | 0.9927 | 0.9966 | **+0.39%** ✅ |

### Out-of-Sample Validation Performance ⚠️

| Metric | Before | After | Change | Status |
|--------|--------|-------|--------|--------|
| **Accuracy** | 0.8500 | 0.8555 | **+0.55%** ✅ |
| **Precision** | 0.5000 | 0.6333 | **+26.66%** ✅✅ |
| **Recall** | 0.1318 | 0.0864 | **-34.45%** ❌ |
| **F1-Score** | 0.2086 | 0.1520 | **-27.13%** ❌ |
| **ROC AUC** | 0.7633 | 0.7703 | **+0.70%** ✅ |
| **Gini** | 0.5265 | 0.5407 | **+1.42%** ✅ |

**Confusion Matrix (Out-of-Sample):**
- **Before:** TP: 29, FP: 29, FN: 191, TN: 1218
- **After:** TP: 19, FP: 11, FN: 201, TN: 1236
- **Analysis:** More conservative predictions (fewer false positives but also fewer true positives)

---

## Gradient Boosting Model Comparison

### Training Set Performance

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **Accuracy** | 1.0000 | 1.0000 | 0.00% |
| **Precision** | 1.0000 | 1.0000 | 0.00% |
| **Recall** | 1.0000 | 1.0000 | 0.00% |
| **F1-Score** | 1.0000 | 1.0000 | 0.00% |
| **ROC AUC** | 1.0000 | 1.0000 | 0.00% |
| **Gini** | 1.0000 | 1.0000 | 0.00% |

### In-Sample Validation Performance

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **Accuracy** | 0.9922 | 0.9951 | **+0.29%** ✅ |
| **Precision** | 0.9798 | 0.9903 | **+1.05%** ✅ |
| **Recall** | 0.9681 | 0.9769 | **+0.88%** ✅ |
| **F1-Score** | 0.9739 | 0.9835 | **+0.96%** ✅ |
| **ROC AUC** | 0.9994 | 0.9998 | **+0.04%** ✅ |
| **Gini** | 0.9988 | 0.9995 | **+0.07%** ✅ |

### Out-of-Sample Validation Performance ⚠️

| Metric | Before | After | Change | Status |
|--------|--------|-------|--------|--------|
| **Accuracy** | 0.8603 | 0.8555 | **-0.48%** ❌ |
| **Precision** | 0.6190 | 0.7500 | **+21.16%** ✅✅ |
| **Recall** | 0.1773 | 0.0545 | **-69.26%** ❌❌ |
| **F1-Score** | 0.2756 | 0.1017 | **-63.10%** ❌❌ |
| **ROC AUC** | 0.7346 | 0.7382 | **+0.36%** ✅ |
| **Gini** | 0.4691 | 0.4764 | **+0.73%** ✅ |

**Confusion Matrix (Out-of-Sample):**
- **Before:** TP: 39, FP: 24, FN: 181, TN: 1223
- **After:** TP: 12, FP: 4, FN: 208, TN: 1243
- **Analysis:** Much more conservative - precision improved significantly but recall dropped dramatically

---

## Logistic Regression Model Comparison

### Training Set Performance

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **Accuracy** | 0.6504 | 0.6740 | **+2.36%** ✅ |
| **Precision** | 0.2586 | 0.2784 | **+1.98%** ✅ |
| **Recall** | 0.7131 | 0.7374 | **+2.43%** ✅ |
| **F1-Score** | 0.3796 | 0.4042 | **+2.46%** ✅ |
| **ROC AUC** | 0.7294 | 0.7653 | **+3.59%** ✅ |
| **Gini** | 0.4588 | 0.5307 | **+7.19%** ✅ |

### In-Sample Validation Performance

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **Accuracy** | 0.6494 | 0.6791 | **+2.97%** ✅ |
| **Precision** | 0.2599 | 0.2775 | **+1.76%** ✅ |
| **Recall** | 0.7241 | 0.7105 | **-1.36%** ❌ |
| **F1-Score** | 0.3826 | 0.3991 | **+1.65%** ✅ |
| **ROC AUC** | 0.7289 | 0.7606 | **+3.17%** ✅ |
| **Gini** | 0.4579 | 0.5213 | **+6.34%** ✅ |

### Out-of-Sample Validation Performance

| Metric | Before | After | Change | Status |
|--------|--------|-------|--------|--------|
| **Accuracy** | 0.4417 | 0.4554 | **+1.37%** ✅ |
| **Precision** | 0.1922 | 0.1975 | **+2.75%** ✅ |
| **Recall** | 0.8500 | 0.8591 | **+0.91%** ✅ |
| **F1-Score** | 0.3135 | 0.3212 | **+0.77%** ✅ |
| **ROC AUC** | 0.6505 | 0.6642 | **+1.37%** ✅ |
| **Gini** | 0.3009 | 0.3283 | **+2.74%** ✅ |

**Confusion Matrix (Out-of-Sample):**
- **Before:** TP: 187, FP: 786, FN: 33, TN: 461
- **After:** TP: 189, FP: 768, FN: 31, TN: 479
- **Analysis:** Slight improvement across all metrics

---

## Key Findings

### ✅ Improvements

1. **Precision Improved Significantly:**
   - **Random Forest:** +26.66% (0.50 → 0.63)
   - **Gradient Boosting:** +21.16% (0.62 → 0.75)
   - **Logistic Regression:** +2.75% (0.19 → 0.20)

2. **ROC AUC Maintained/Improved:**
   - All models maintained or slightly improved ROC AUC
   - This indicates the model's ranking ability is preserved

3. **In-Sample Performance Improved:**
   - All models show improvements in in-sample validation
   - Lower overfitting risk maintained

### ❌ Concerns

1. **Recall Decreased Significantly:**
   - **Random Forest:** -34.45% (0.13 → 0.09)
   - **Gradient Boosting:** -69.26% (0.18 → 0.05)
   - **Logistic Regression:** Maintained high recall (0.85 → 0.86)

2. **F1-Score Decreased:**
   - **Random Forest:** -27.13% (0.21 → 0.15)
   - **Gradient Boosting:** -63.10% (0.28 → 0.10)
   - **Logistic Regression:** Slight improvement (+0.77%)

### 📊 Overall Assessment

**The feature engineering changes have made the models more conservative:**

- **Positive:** Fewer false positives (higher precision)
- **Negative:** More false negatives (lower recall)
- **Impact:** Models are now more selective but miss more injuries

**Best Model for Out-of-Sample:**
- **Logistic Regression** maintains the best balance with:
  - Precision: 0.1975 (low but acceptable)
  - Recall: 0.8591 (excellent - catches 86% of injuries)
  - F1-Score: 0.3212 (best among all models)

---

## Recommendations

### 1. **Threshold Optimization** (HIGH PRIORITY)
The models are now too conservative. Consider:
- Lowering the threshold from 0.5 to 0.3-0.4 for RF/GB
- This should improve recall while maintaining reasonable precision

### 2. **Model Selection**
- **For High Recall (Catch Most Injuries):** Use Logistic Regression
- **For High Precision (Fewer False Alarms):** Use Gradient Boosting
- **For Balance:** Use Random Forest with optimized threshold

### 3. **Feature Review**
- The normalized features are working (ROC AUC maintained)
- Week 5 fix was successful (removed drifted features)
- Consider adding more normalized features for other cumulative metrics

### 4. **Ensemble Approach**
- Combine models with different thresholds
- Use LR for high recall, GB for high precision
- Weighted ensemble based on use case

---

## Detailed Metrics Tables

### Random Forest - Complete Metrics

| Dataset | Accuracy | Precision | Recall | F1 | ROC AUC | Gini |
|---------|----------|----------|--------|----|---------|------|
| **Train** | 0.9966 | 0.9782 | 0.9998 | 0.9889 | 1.0000 | 1.0000 |
| **In-Sample Val** | 0.9830 | 0.9187 | 0.9729 | 0.9450 | 0.9983 | 0.9966 |
| **Out-of-Sample Val** | 0.8555 | 0.6333 | 0.0864 | 0.1520 | 0.7703 | 0.5407 |

### Gradient Boosting - Complete Metrics

| Dataset | Accuracy | Precision | Recall | F1 | ROC AUC | Gini |
|---------|----------|----------|--------|----|---------|------|
| **Train** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **In-Sample Val** | 0.9951 | 0.9903 | 0.9769 | 0.9835 | 0.9998 | 0.9995 |
| **Out-of-Sample Val** | 0.8555 | 0.7500 | 0.0545 | 0.1017 | 0.7382 | 0.4764 |

### Logistic Regression - Complete Metrics

| Dataset | Accuracy | Precision | Recall | F1 | ROC AUC | Gini |
|---------|----------|----------|--------|----|---------|------|
| **Train** | 0.6740 | 0.2784 | 0.7374 | 0.4042 | 0.7653 | 0.5307 |
| **In-Sample Val** | 0.6791 | 0.2775 | 0.7105 | 0.3991 | 0.7606 | 0.5213 |
| **Out-of-Sample Val** | 0.4554 | 0.1975 | 0.8591 | 0.3212 | 0.6642 | 0.3283 |

---

**Analysis Date:** 2025-11-26  
**Models Trained:** Random Forest, Gradient Boosting, Logistic Regression  
**Feature Engineering:** Week 5 fix + Normalized cumulative features


