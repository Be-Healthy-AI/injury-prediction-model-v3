# Seasonal Split vs Phase 2 Comparison

**Date:** 2025-11-29

## Overview

This comparison analyzes model performance using a **seasonal temporal split** versus the **Phase 2** approach (80/20 random split with low-shift features, covariate shift correction, and calibration).

## Dataset Configurations

### Seasonal Split
- **Training:** <= 2024-06-30 (36,753 records, 14.3% injury ratio)
- **Validation:** 2024/25 season (5,047 records, 19.8% injury ratio)
- **Test:** 2025/26 season (12,218 records, 1.8% injury ratio)

### Phase 2 (80/20 Random Split)
- **Training:** 80% random split (stratified)
- **In-Sample Validation:** 20% random split (stratified)
- **Out-of-Sample Validation:** >= 2025-07-01 (12,218 records, 1.8% injury ratio)

## Performance Comparison - Test/Out-of-Sample (2025/26 Season)

| Model | Metric | Seasonal Split | Phase 2 | Difference |
|-------|--------|----------------|---------|------------|
| **RF** | Precision | 0.0385 | 0.0905 | -0.0520 (-57.5%) |
| | Recall | 0.0591 | 0.0864 | -0.0273 (-31.6%) |
| | F1-Score | 0.0466 | 0.0884 | -0.0418 (-47.3%) |
| | ROC AUC | 0.7500 | 0.7503 | -0.0003 (-0.04%) |
| | | | | |
| **GB** | Precision | 0.1304 | 0.1538 | -0.0234 (-15.2%) |
| | Recall | 0.1364 | 0.1364 | 0.0000 (0.0%) |
| | F1-Score | 0.1333 | 0.1446 | -0.0113 (-7.8%) |
| | ROC AUC | 0.7627 | 0.7255 | +0.0372 (+5.1%) |
| | | | | |
| **LR** | Precision | 0.0000 | 0.0239 | -0.0239 (-100%) |
| | Recall | 0.0000 | 0.0864 | -0.0864 (-100%) |
| | F1-Score | 0.0000 | 0.0375 | -0.0375 (-100%) |
| | ROC AUC | 0.6809 | 0.6530 | +0.0279 (+4.3%) |

## Key Findings

### 1. **Gradient Boosting (GB) - Best Overall**

**Seasonal Split:**
- **Test (2025/26):** F1=0.1333, Precision=0.1304, Recall=0.1364
- **Validation (2024/25):** F1=0.1599, Precision=0.5053, Recall=0.0950
- **Gap (Validation → Test):** F1 drops by 0.0266 (16.6% relative)

**Phase 2:**
- **Out-of-Sample:** F1=0.1446, Precision=0.1538, Recall=0.1364

**Insight:** 
- GB shows **minimal degradation** between validation (2024/25) and test (2025/26) seasons
- The gap is only 16.6% relative, suggesting performance stabilizes after the first year
- Phase 2 performs slightly better on test, but both approaches show similar recall (13.6%)

### 2. **Random Forest (RF) - Significant Degradation**

**Seasonal Split:**
- **Test (2025/26):** F1=0.0466, Precision=0.0385, Recall=0.0591
- **Validation (2024/25):** F1=0.1660, Precision=0.4274, Recall=0.1030
- **Gap (Validation → Test):** F1 drops by 0.1194 (71.9% relative)

**Phase 2:**
- **Out-of-Sample:** F1=0.0884, Precision=0.0905, Recall=0.0864

**Insight:**
- RF shows **severe degradation** between validation and test
- Performance drops by 72% from validation to test
- Phase 2 performs **much better** (F1=0.0884 vs 0.0466), suggesting the 80/20 split was more optimistic

### 3. **Logistic Regression (LR) - Complete Failure**

**Seasonal Split:**
- **Test (2025/26):** F1=0.0000 (no predictions)
- **Validation (2024/25):** F1=0.0714, Precision=0.5846, Recall=0.0380

**Phase 2:**
- **Out-of-Sample:** F1=0.0375, Precision=0.0239, Recall=0.0864

**Insight:**
- LR completely fails on the test set (2025/26 season)
- This suggests the model is too conservative and the threshold (0.5) is too high for the natural 1.8% injury ratio

## Temporal Degradation Analysis

### Gradient Boosting (GB)

| Period | F1-Score | Precision | Recall | ROC AUC |
|--------|----------|-----------|--------|---------|
| Training (<= 2024-06-30) | 0.9788 | 0.9852 | 0.9725 | 0.9993 |
| Validation (2024/25) | 0.1599 | 0.5053 | 0.0950 | 0.6691 |
| Test (2025/26) | 0.1333 | 0.1304 | 0.1364 | 0.7627 |

**Degradation Pattern:**
- **Train → Validation:** F1 drops by 0.8189 (83.7% relative) - **Major drop**
- **Validation → Test:** F1 drops by 0.0266 (16.6% relative) - **Small drop**

**Conclusion:** Most degradation happens in the first year (2024/25 season). Performance then stabilizes somewhat, suggesting the model adapts to the new distribution after initial shock.

### Random Forest (RF)

| Period | F1-Score | Precision | Recall | ROC AUC |
|--------|----------|-----------|--------|---------|
| Training (<= 2024-06-30) | 0.8053 | 0.8350 | 0.7776 | 0.9780 |
| Validation (2024/25) | 0.1660 | 0.4274 | 0.1030 | 0.6660 |
| Test (2025/26) | 0.0466 | 0.0385 | 0.0591 | 0.7500 |

**Degradation Pattern:**
- **Train → Validation:** F1 drops by 0.6393 (79.4% relative) - **Major drop**
- **Validation → Test:** F1 drops by 0.1194 (71.9% relative) - **Continued major drop**

**Conclusion:** RF shows **continuous degradation** across both periods, suggesting it's more sensitive to temporal drift than GB.

## Key Insights

### 1. **The 80/20 Random Split Was Optimistic**

The Phase 2 approach (80/20 random split) shows better performance than the seasonal split because:
- Random splits maintain similar feature distributions
- Temporal splits expose real-world distribution shift
- The seasonal split is a **more realistic** evaluation

### 2. **Most Degradation Happens in First Year**

For GB (the best model):
- **83.7%** of the F1 drop happens between training and validation (2024/25)
- Only **16.6%** additional drop between validation and test (2025/26)
- This suggests the model experiences a "shock" in the first new season, then stabilizes

### 3. **ROC AUC Remains Stable**

Despite F1-score degradation:
- GB ROC AUC: 0.6691 (validation) → 0.7627 (test) - **Actually improves!**
- RF ROC AUC: 0.6660 (validation) → 0.7500 (test) - **Also improves!**

**Interpretation:** The models can still **rank** players by injury risk, but the **threshold** (0.5) is inappropriate for the natural 1.8% injury ratio in the test set.

### 4. **Threshold Optimization Needed**

The test set has a **1.8% natural injury ratio**, but models are using a **0.5 threshold** (designed for 15% ratio in training). This explains:
- Low precision (many false positives)
- Low recall (threshold too high)
- Good ROC AUC (ranking ability is preserved)

## Recommendations

1. **Use Seasonal Split for Evaluation:** More realistic than random splits
2. **Focus on GB Model:** Shows best stability across temporal periods
3. **Optimize Threshold:** Test set needs much lower threshold (e.g., 0.05-0.10) for 1.8% injury ratio
4. **Consider Rolling Retraining:** Retrain annually with latest season to adapt to distribution changes
5. **Monitor Validation Performance:** Use 2024/25 season as ongoing validation to catch degradation early

## Conclusion

The seasonal split reveals that:
- **Most degradation happens in the first year** (training → validation)
- **Performance stabilizes somewhat** after the initial shock (validation → test)
- **GB is the most stable** model across temporal periods
- **Threshold optimization is critical** for the natural 1.8% injury ratio

The 80/20 random split was indeed optimistic, and the seasonal split provides a more realistic view of production performance.

