# Seasonal Split - Performance Metrics Across All Datasets

**Date:** 2025-11-29

## Dataset Information

| Dataset | Date Range | Records | Injury Ratio | Description |
|---------|------------|----------|--------------|-------------|
| **Training** | <= 2024-06-30 | 36,753 | 14.3% | All seasons before 2024/25 |
| **Validation** | 2024/25 season | 5,047 | 19.8% | 2024-07-01 to 2025-06-30 |
| **Test** | 2025/26 season | 12,218 | 1.8% | >= 2025-07-01 |

---

## Random Forest (RF) Performance

| Dataset | Accuracy | Precision | Recall | F1-Score | ROC AUC | Gini | TP | FP | TN | FN |
|---------|----------|-----------|--------|-----------|---------|------|----|----|----|----|
| **Training** | 0.9461 | 0.8350 | 0.7776 | 0.8053 | 0.9780 | 0.9559 | 4,098 | 810 | 30,673 | 1,172 |
| **Validation (2024/25)** | 0.7949 | 0.4274 | 0.1030 | 0.1660 | 0.6660 | 0.3321 | 103 | 138 | 3,909 | 897 |
| **Test (2025/26)** | 0.9565 | 0.0385 | 0.0591 | 0.0466 | 0.7500 | 0.5000 | 13 | 325 | 11,673 | 207 |

### RF Performance Gaps

| Gap | Precision | Recall | F1-Score | ROC AUC |
|-----|-----------|--------|----------|---------|
| **Train → Validation** | -0.4076 (-48.8%) | -0.6746 (-86.7%) | -0.6393 (-79.4%) | -0.3120 (-31.9%) |
| **Validation → Test** | -0.3889 (-91.0%) | -0.0439 (-42.6%) | -0.1194 (-71.9%) | +0.0840 (+12.6%) |
| **Train → Test** | -0.7965 (-95.4%) | -0.7185 (-92.4%) | -0.7587 (-94.2%) | -0.2280 (-23.3%) |

---

## Gradient Boosting (GB) Performance

| Dataset | Accuracy | Precision | Recall | F1-Score | ROC AUC | Gini | TP | FP | TN | FN |
|---------|----------|-----------|--------|-----------|---------|------|----|----|----|----|
| **Training** | 0.9940 | 0.9852 | 0.9725 | 0.9788 | 0.9993 | 0.9987 | 5,125 | 77 | 31,406 | 145 |
| **Validation (2024/25)** | 0.8023 | 0.5053 | 0.0950 | 0.1599 | 0.6691 | 0.3383 | 95 | 93 | 3,954 | 905 |
| **Test (2025/26)** | 0.9681 | 0.1304 | 0.1364 | 0.1333 | 0.7627 | 0.5253 | 30 | 200 | 11,798 | 190 |

### GB Performance Gaps

| Gap | Precision | Recall | F1-Score | ROC AUC |
|-----|-----------|--------|----------|---------|
| **Train → Validation** | -0.4799 (-48.7%) | -0.8775 (-90.2%) | -0.8189 (-83.7%) | -0.3302 (-33.1%) |
| **Validation → Test** | -0.3749 (-74.2%) | +0.0414 (+43.6%) | -0.0266 (-16.6%) | +0.0936 (+14.0%) |
| **Train → Test** | -0.8548 (-86.8%) | -0.8361 (-86.0%) | -0.8455 (-86.4%) | -0.2366 (-23.7%) |

---

## Logistic Regression (LR) Performance

| Dataset | Accuracy | Precision | Recall | F1-Score | ROC AUC | Gini | TP | FP | TN | FN |
|---------|----------|-----------|--------|-----------|---------|------|----|----|----|----|
| **Training** | 0.8563 | 0.4615 | 0.0125 | 0.0244 | 0.7667 | 0.5335 | 66 | 77 | 31,406 | 5,204 |
| **Validation (2024/25)** | 0.8040 | 0.5846 | 0.0380 | 0.0714 | 0.6654 | 0.3307 | 38 | 27 | 4,020 | 962 |
| **Test (2025/26)** | 0.9664 | 0.0000 | 0.0000 | 0.0000 | 0.6809 | 0.3617 | 0 | 190 | 11,808 | 220 |

### LR Performance Gaps

| Gap | Precision | Recall | F1-Score | ROC AUC |
|-----|-----------|--------|----------|---------|
| **Train → Validation** | +0.1231 (+26.7%) | +0.0255 (+204.0%) | +0.0470 (+192.6%) | -0.1013 (-13.2%) |
| **Validation → Test** | -0.5846 (-100.0%) | -0.0380 (-100.0%) | -0.0714 (-100.0%) | +0.0155 (+2.3%) |
| **Train → Test** | -0.4615 (-100.0%) | -0.0125 (-100.0%) | -0.0244 (-100.0%) | -0.0858 (-11.2%) |

---

## Summary Comparison

### Best Model by Dataset

| Dataset | Best F1-Score | Model | Precision | Recall |
|---------|---------------|-------|-----------|--------|
| **Training** | 0.9788 | GB | 0.9852 | 0.9725 |
| **Validation (2024/25)** | 0.1660 | RF | 0.4274 | 0.1030 |
| **Test (2025/26)** | 0.1333 | GB | 0.1304 | 0.1364 |

### Key Observations

1. **Training Performance:**
   - GB achieves near-perfect performance (F1=0.9788)
   - RF also performs well (F1=0.8053)
   - LR is very conservative (F1=0.0244, Recall=1.25%)

2. **Validation (2024/25) Performance:**
   - All models show significant degradation from training
   - RF has best F1 (0.1660) but low recall (10.3%)
   - GB has better balance (F1=0.1599, Precision=50.5%)
   - LR shows improvement from training (F1=0.0714)

3. **Test (2025/26) Performance:**
   - GB is the best model (F1=0.1333, Recall=13.6%)
   - RF shows further degradation (F1=0.0466)
   - LR completely fails (F1=0.0000, no predictions)

4. **Temporal Stability:**
   - **GB:** Small gap between validation and test (16.6% relative F1 drop)
   - **RF:** Large gap between validation and test (71.9% relative F1 drop)
   - **LR:** Complete failure on test set

5. **ROC AUC Stability:**
   - All models maintain reasonable ROC AUC (0.66-0.76) across datasets
   - GB ROC AUC actually **improves** from validation to test (0.6691 → 0.7627)
   - This suggests models can still rank players, but threshold is inappropriate

---

## Confusion Matrix Summary

### Random Forest (RF)

| Dataset | True Positives | False Positives | True Negatives | False Negatives |
|---------|----------------|-----------------|----------------|-----------------|
| Training | 4,098 | 810 | 30,673 | 1,172 |
| Validation | 103 | 138 | 3,909 | 897 |
| Test | 13 | 325 | 11,673 | 207 |

### Gradient Boosting (GB)

| Dataset | True Positives | False Positives | True Negatives | False Negatives |
|---------|----------------|-----------------|----------------|-----------------|
| Training | 5,125 | 77 | 31,406 | 145 |
| Validation | 95 | 93 | 3,954 | 905 |
| Test | 30 | 200 | 11,798 | 190 |

### Logistic Regression (LR)

| Dataset | True Positives | False Positives | True Negatives | False Negatives |
|---------|----------------|-----------------|----------------|-----------------|
| Training | 66 | 77 | 31,406 | 5,204 |
| Validation | 38 | 27 | 4,020 | 962 |
| Test | 0 | 190 | 11,808 | 220 |

---

## Recommendations

1. **Use GB Model:** Best overall performance and temporal stability
2. **Optimize Threshold:** Test set has 1.8% natural ratio, needs lower threshold (0.05-0.10)
3. **Monitor Validation:** Use 2024/25 season as ongoing validation
4. **Consider Retraining:** Retrain annually with latest season to adapt to distribution changes



