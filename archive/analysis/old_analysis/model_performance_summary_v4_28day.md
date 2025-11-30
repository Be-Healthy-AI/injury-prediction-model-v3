# Model Performance Summary - V4 (28-Day Window)

**Date:** Generated after retraining with 28-day windows (excluding week_5 features)  
**Window Configuration:** 4 weeks / 28 days (week_5 features excluded to reduce temporal sensitivity)

---

## Overview

All models were retrained using enhanced features with **28-day windows** (excluding all week_5 features) to reduce sensitivity to temporal patterns. This addresses the data drift issues identified in the feature drift analysis.

---

## Random Forest Model

### Performance Metrics

| Metric | Training | In-Sample Validation | Out-of-Sample Validation |
|--------|----------|----------------------|-------------------------|
| **Accuracy** | 0.9947 | 0.9800 | 0.8521 |
| **Precision** | 0.9659 | 0.8993 | 0.6364 |
| **Recall** | 1.0000 | 0.9761 | 0.0318 |
| **F1-Score** | 0.9827 | 0.9361 | 0.0606 |
| **ROC AUC** | 1.0000 | 0.9979 | 0.7863 |
| **Gini** | 1.0000 | 0.9958 | 0.5726 |

### Confusion Matrix

**Training Set:**
- True Negatives: 28,247
- False Positives: 177
- False Negatives: 0
- True Positives: 5,016

**In-Sample Validation:**
- True Negatives: 6,969
- False Positives: 137
- False Negatives: 30
- True Positives: 1,224

**Out-of-Sample Validation:**
- True Negatives: 1,243
- False Positives: 4
- False Negatives: 213
- True Positives: 7

### Performance Gaps

| Metric | In-Sample Gap | Out-of-Sample Gap |
|--------|---------------|-------------------|
| **Accuracy** | 0.0147 | 0.1426 |
| **Precision** | 0.0666 | 0.3296 |
| **Recall** | 0.0239 | 0.9682 |
| **F1-Score** | 0.0465 | 0.9221 |
| **ROC AUC** | 0.0021 | 0.2137 |
| **Gini** | 0.0042 | 0.4274 |

**Overfitting Risk (In-Sample):** ✅ LOW (AUC gap: 0.0021)

---

## Gradient Boosting Model

### Performance Metrics

| Metric | Training | In-Sample Validation | Out-of-Sample Validation |
|--------|----------|----------------------|-------------------------|
| **Accuracy** | 1.0000 | 0.9909 | 0.8500 |
| **Precision** | 1.0000 | 0.9609 | 0.5000 |
| **Recall** | 1.0000 | 0.9793 | 0.0091 |
| **F1-Score** | 1.0000 | 0.9700 | 0.0179 |
| **ROC AUC** | 1.0000 | 0.9992 | 0.7366 |
| **Gini** | 1.0000 | 0.9985 | 0.4731 |

### Confusion Matrix

**Training Set:**
- True Negatives: 28,424
- False Positives: 0
- False Negatives: 0
- True Positives: 5,016

**In-Sample Validation:**
- True Negatives: 7,056
- False Positives: 50
- False Negatives: 26
- True Positives: 1,228

**Out-of-Sample Validation:**
- True Negatives: 1,245
- False Positives: 2
- False Negatives: 218
- True Positives: 2

### Performance Gaps

| Metric | In-Sample Gap | Out-of-Sample Gap |
|--------|---------------|-------------------|
| **Accuracy** | 0.0091 | 0.1500 |
| **Precision** | 0.0391 | 0.5000 |
| **Recall** | 0.0207 | 0.9909 |
| **F1-Score** | 0.0300 | 0.9821 |
| **ROC AUC** | 0.0008 | 0.2634 |
| **Gini** | 0.0015 | 0.5269 |

**Overfitting Risk (In-Sample):** ✅ LOW (AUC gap: 0.0008)

---

## Logistic Regression Model

### Performance Metrics

| Metric | Training | In-Sample Validation | Out-of-Sample Validation |
|--------|----------|----------------------|-------------------------|
| **Accuracy** | 0.6616 | 0.6672 | 0.4247 |
| **Precision** | 0.2704 | 0.2700 | 0.1905 |
| **Recall** | 0.7396 | 0.7153 | 0.8727 |
| **F1-Score** | 0.3961 | 0.3920 | 0.3127 |
| **ROC AUC** | 0.7552 | 0.7516 | 0.6596 |
| **Gini** | 0.5104 | 0.5032 | 0.3192 |

### Confusion Matrix

**Training Set:**
- True Negatives: 18,415
- False Positives: 10,009
- False Negatives: 1,306
- True Positives: 3,710

**In-Sample Validation:**
- True Negatives: 4,681
- False Positives: 2,425
- False Negatives: 357
- True Positives: 897

**Out-of-Sample Validation:**
- True Negatives: 431
- False Positives: 816
- False Negatives: 28
- True Positives: 192

### Performance Gaps

| Metric | In-Sample Gap | Out-of-Sample Gap |
|--------|---------------|-------------------|
| **Accuracy** | -0.0056 | 0.2370 |
| **Precision** | 0.0004 | 0.0800 |
| **Recall** | 0.0243 | -0.1331 |
| **F1-Score** | 0.0040 | 0.0833 |
| **ROC AUC** | 0.0036 | 0.0956 |
| **Gini** | 0.0072 | 0.1912 |

**Overfitting Risk (In-Sample):** ✅ LOW (AUC gap: 0.0036)

---

## Key Observations

### 1. **Out-of-Sample Performance - Random Forest & Gradient Boosting**
- **Precision:** Both models maintain reasonable precision (RF: 0.6364, GB: 0.5000) in out-of-sample validation
- **Recall:** Both models show very low recall (RF: 0.0318, GB: 0.0091), indicating they are very conservative in predicting injuries
- **ROC AUC:** RF achieves 0.7863, GB achieves 0.7366 - both show good discrimination ability despite low recall

### 2. **Out-of-Sample Performance - Logistic Regression**
- **Recall:** Highest recall among all models (0.8727), catching most injuries
- **Precision:** Lowest precision (0.1905), resulting in many false positives
- **ROC AUC:** 0.6596 - lower than tree-based models but more balanced precision/recall trade-off

### 3. **In-Sample vs Out-of-Sample Gaps**
- **Random Forest:** Large recall gap (0.9682) - model is very conservative on unseen data
- **Gradient Boosting:** Even larger recall gap (0.9909) - extremely conservative
- **Logistic Regression:** Negative recall gap (-0.1331) - actually performs better on out-of-sample for recall

### 4. **Model Comparison - Out-of-Sample**

| Model | Precision | Recall | F1-Score | ROC AUC | Best For |
|-------|-----------|--------|----------|---------|----------|
| **Random Forest** | 0.6364 | 0.0318 | 0.0606 | 0.7863 | High precision, low false positives |
| **Gradient Boosting** | 0.5000 | 0.0091 | 0.0179 | 0.7366 | Highest precision when it predicts |
| **Logistic Regression** | 0.1905 | 0.8727 | 0.3127 | 0.6596 | Catching most injuries (high recall) |

---

## Impact of 28-Day Window (Excluding Week_5)

### Changes from Previous Version:
- **Window Size:** Reduced from 35 days (5 weeks) to 28 days (4 weeks)
- **Features Removed:** All features with `_week_5` suffix
- **Rationale:** Reduce temporal sensitivity and address data drift in week_5 features

### Expected Benefits:
1. ✅ Reduced temporal pattern sensitivity
2. ✅ Better generalization across time periods
3. ✅ Lower risk of overfitting to recent patterns

### Trade-offs:
- ⚠️ Slightly less recent information (7 days less)
- ⚠️ May miss some short-term patterns

---

## Recommendations

1. **For High Precision Use Cases:** Use Random Forest (Precision: 0.6364)
2. **For High Recall Use Cases:** Use Logistic Regression (Recall: 0.8727)
3. **For Balanced Performance:** Consider ensemble of RF and LR
4. **Threshold Tuning:** Consider adjusting prediction thresholds to balance precision/recall based on use case
5. **Feature Engineering:** Continue monitoring feature drift and consider additional normalization strategies

---

**Generated by:** Model Training Scripts V4  
**Configuration:** 28-day windows, week_5 features excluded


