# Comparison: 14-day vs 21-day (both optimizing on validation)

Both experiments use **optimize_on = validation** (best iteration and early stopping by validation combined score: 0.6×Gini + 0.4×F1).

---

## 1. Setup

| | **14-day (optimize validation)** | **21-day (optimize validation)** |
|--|----------------------------------|----------------------------------|
| **Best iteration** | 10 | 14 |
| **n_features** | 200 | 280 |
| **Label window** | D-14..D-1 (muscular) | 21 days to label positives |
| **Selection** | Optimize on validation | Optimize on validation |

---

## 2. Train

| Metric | 14-day (200 feat) | 21-day (280 feat) | Better |
|--------|--------------------|--------------------|--------|
| Accuracy | 0.9546 | 0.9478 | 14-day |
| Precision | 0.2804 | **0.3371** | 21-day |
| Recall | **0.9993** | 0.9985 | 14-day |
| F1 | 0.4379 | **0.5041** | 21-day |
| ROC AUC | **0.9980** | 0.9972 | 14-day |
| Gini | **0.9961** | 0.9944 | 14-day |

---

## 3. Validation

| Metric | 14-day (200 feat) | 21-day (280 feat) | Better |
|--------|--------------------|--------------------|--------|
| Accuracy | 0.9532 | 0.9472 | 14-day |
| Precision | 0.2824 | **0.3396** | 21-day |
| Recall | **0.9966** | 0.9964 | 14-day |
| F1 | 0.4400 | **0.5066** | 21-day |
| ROC AUC | **0.9970** | 0.9961 | 14-day |
| Gini | **0.9940** | 0.9922 | 14-day |
| **Combined (0.6×Gini + 0.4×F1)** | 0.7724 | **0.7979** | **21-day** |

---

## 4. Test (2025/26)

| Metric | 14-day (200 feat) | 21-day (280 feat) | Better |
|--------|--------------------|--------------------|--------|
| Accuracy | **0.8578** | 0.5369 | 14-day |
| Precision | **0.0847** | 0.0218 | 14-day |
| Recall | 0.4771 | **0.5019** | 21-day |
| F1 | **0.1439** | 0.0417 | 14-day |
| ROC AUC | **0.7514** | 0.5346 | 14-day |
| Gini | **0.5027** | 0.0692 | 14-day |

---

## 5. Confusion matrices (test)

**14-day (200 feat):** TN 48,765 · FP 7,444 · FN 755 · TP 689  

**21-day (280 feat):** TN 56,183 · FP 48,325 · FN 1,068 · TP 1,076  

---

## 6. Summary

- **Validation:** 21-day has higher combined score (0.7979 vs 0.7724) and better precision/F1.
- **Test:** 14-day is better on all metrics; 21-day has many more test FPs (different test set or labeling likely).

---

*Source: 14-day from `iterative_feature_selection_results_muscular.json` (best iteration 10). 21-day from user-provided best_iteration_metrics (iteration 14).*
