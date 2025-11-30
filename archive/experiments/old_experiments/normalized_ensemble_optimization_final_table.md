# Ensemble Optimization - Final Summary Table

**Date:** 2025-11-27  
**Models:** RF, GB, LR with normalized cumulative features  
**Validation Dataset:** Out-of-sample (12,218 records, 1.8% injury ratio)

---

## Best Ensemble Configurations

| Ensemble | Threshold | Precision | Recall | F1-Score | ROC AUC | TP | FP | TN | FN | Notes |
|----------|-----------|-----------|--------|----------|---------|----|----|----|----|-------|
| **GB_only** | **0.060** | **0.1679** | **0.2000** | **0.1826** | 0.7500 | 44 | 218 | 11,780 | 176 | **Best Overall** |
| RF_AND_GB | 0.060 | 0.1679 | 0.2000 | 0.1826 | - | 44 | 218 | 11,780 | 176 | Same as GB_only |
| GB_AND_LR | 0.060 | 0.1679 | 0.2000 | 0.1826 | - | 44 | 218 | 11,780 | 176 | Same as GB_only |
| | | | | | | | | | | |
| **RF_70_GB_30** | **0.450** | **0.2295** | 0.0636 | 0.0996 | - | 14 | 47 | 11,951 | 206 | **Best Precision** |
| | | | | | | | | | | |
| **RF_50_GB_50** | **0.120** | 0.0501 | **0.6727** | 0.0932 | - | 148 | 2,809 | 9,189 | 72 | **Best Recall** |
| | | | | | | | | | | |
| GB_only | 0.050 | 0.1523 | 0.2091 | 0.1762 | 0.7500 | 46 | 256 | 11,742 | 174 | Previous best |

---

## Comparison: Individual Models vs Best Ensemble

| Model/Ensemble | Threshold | Precision | Recall | F1-Score | Improvement |
|----------------|-----------|-----------|--------|----------|-------------|
| **RF (best)** | 0.25 | 0.0534 | 0.6409 | 0.0986 | Baseline |
| **GB (best)** | 0.05 | 0.1523 | 0.2091 | 0.1762 | Baseline |
| **LR (best)** | 0.01 | 0.0180 | 1.0000 | 0.0354 | Baseline |
| | | | | | |
| **GB_only (optimized)** | **0.060** | **0.1679** | **0.2000** | **0.1826** | **+3.6% vs GB@0.05** |
| **RF_70_GB_30** | **0.450** | **0.2295** | 0.0636 | 0.0996 | **+50.6% precision vs GB** |
| **RF_50_GB_50** | **0.120** | 0.0501 | **0.6727** | 0.0932 | **+5.0% recall vs RF** |

---

## Key Findings

### 1. Best Overall Performance: GB @ Threshold 0.060

**Performance:**
- **Precision:** 16.8%
- **Recall:** 20.0%
- **F1-Score:** 0.1826 (best overall)
- **Detects:** 44 out of 220 injuries

**Improvement over GB @ 0.05:**
- Precision: +1.6 percentage points
- Recall: -0.9 percentage points
- F1-Score: +3.6% improvement

### 2. Best Precision: RF_70_GB_30 @ Threshold 0.450

**Performance:**
- **Precision:** 22.95% (highest among all configurations)
- **Recall:** 6.4%
- **F1-Score:** 0.0996
- **Detects:** 14 out of 220 injuries

**Use case:** When precision is critical and false positives are costly.

### 3. Best Recall: RF_50_GB_50 @ Threshold 0.120

**Performance:**
- **Precision:** 5.0%
- **Recall:** 67.3% (catches 2/3 of injuries)
- **F1-Score:** 0.0932
- **Detects:** 148 out of 220 injuries

**Use case:** When catching most injuries is critical, even with more false positives.

### 4. Ensemble Insights

**Interesting observations:**
- **RF_AND_GB** and **GB_AND_LR** at threshold 0.060 produce identical results to **GB_only**
  - This suggests GB is the dominant model in these ensembles
  - The AND gate effectively filters to GB's predictions
  
- **RF_50_GB_50** (weighted average) provides the best recall
  - Combines RF's recall strength with GB's precision
  - At threshold 0.120, achieves 67.3% recall

- **RF_70_GB_30** provides the best precision
  - Higher weight on RF (which has better precision at higher thresholds)
  - At threshold 0.450, achieves 22.95% precision

---

## Top 10 Ensembles by F1-Score

| Rank | Ensemble | Threshold | Precision | Recall | F1-Score |
|------|----------|-----------|-----------|--------|----------|
| 1 | **GB_only** | **0.060** | **0.1679** | **0.2000** | **0.1826** |
| 2 | RF_AND_GB | 0.060 | 0.1679 | 0.2000 | 0.1826 |
| 3 | GB_AND_LR | 0.060 | 0.1679 | 0.2000 | 0.1826 |
| 4 | GB_only | 0.050 | 0.1523 | 0.2091 | 0.1762 |
| 5 | RF_AND_GB | 0.050 | 0.1523 | 0.2091 | 0.1762 |
| 6 | GB_AND_LR | 0.050 | 0.1523 | 0.2091 | 0.1762 |
| 7 | GB_only | 0.080 | 0.1749 | 0.1773 | 0.1761 |
| 8 | RF_AND_GB | 0.080 | 0.1749 | 0.1773 | 0.1761 |
| 9 | GB_AND_LR | 0.080 | 0.1749 | 0.1773 | 0.1761 |
| 10 | GB_only | 0.070 | 0.1667 | 0.1818 | 0.1739 |

**Note:** RF_AND_GB and GB_AND_LR produce identical results to GB_only, suggesting GB dominates these AND gate ensembles.

---

## Recommendations

### For Balanced Performance (Recommended)
**Gradient Boosting @ Threshold 0.060**
- **Precision:** 16.8%
- **Recall:** 20.0%
- **F1-Score:** 0.1826 (best overall)
- **Best general-purpose configuration**

### For High Precision
**RF_70_GB_30 @ Threshold 0.450**
- **Precision:** 22.95%
- **Recall:** 6.4%
- **Use when false positives are very costly**

### For High Recall
**RF_50_GB_50 @ Threshold 0.120**
- **Precision:** 5.0%
- **Recall:** 67.3%
- **Use when catching most injuries is critical**

---

## Conclusion

1. **Best overall:** GB @ 0.060 achieves the highest F1-Score (0.1826)
2. **Ensemble benefit:** RF_50_GB_50 provides better recall (67.3%) than individual models
3. **Precision optimization:** RF_70_GB_30 achieves highest precision (22.95%)
4. **AND gate ensembles:** RF_AND_GB and GB_AND_LR produce identical results to GB_only, suggesting GB is the limiting factor

**Key insight:** While ensembles can improve recall (RF_50_GB_50) or precision (RF_70_GB_30), the best overall F1-Score is achieved by a single optimized model (GB @ 0.060).

