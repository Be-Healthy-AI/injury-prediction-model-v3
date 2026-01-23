# V3 Model Comparison: Three Natural Ratio Models

## Overview

This document compares three V3_natural models trained with different season ranges:

1. **V3_natural (all)**: Trained on all seasons (2011-2026), natural ratio, PL-only
2. **V3_natural_recent**: Trained on recent seasons only (2018-2026), natural ratio, PL-only
3. **V3_natural_filtered**: Trained on recent seasons (2018-2026) excluding 2021-2022 and 2022-2023, natural ratio, PL-only

All models are evaluated on the same test set: 2025-2026 PL-only timeline (natural ratio).

---

## Training Metrics Comparison

| Metric | V3_natural (all seasons) | V3_natural_recent (2018-2026) | V3_natural_filtered (2018-2026 excl. 2021-2022 & 2022-2023) |
|--------|-------------------------|-------------------------------|----------------------------------------------------------------|
| **Accuracy** | 0.9781 | 0.9844 | 0.9924 | 
| **Precision** | 0.1544 | 0.2227 | 0.4007 | 
| **Recall** | 1.0000 | 1.0000 | 1.0000 | 
| **F1-Score** | 0.2675 | 0.3643 | 0.5722 | 
| **ROC AUC** | 0.9997 | 0.9998 | 0.9999 | 
| **Gini** | 0.9994 | 0.9996 | 0.9999 | 

### Training Confusion Matrix

| Model | TP | FP | TN | FN | False Positive Rate |
|-------|----|----|----|----|---------------------|
| V3_natural (all) | 3104 | 17001 | 757263 | 0 | 0.0220 |
| V3_natural_recent | 2694 | 9404 | 591925 | 0 | 0.0156 |
| V3_natural_filtered | 1994 | 2982 | 387259 | 0 | 0.0076 |

---

## Test Metrics Comparison (2025-2026 PL-only)

| Metric | V3_natural (all seasons) | V3_natural_recent (2018-2026) | V3_natural_filtered (2018-2026 excl. 2021-2022 & 2022-2023) |
|--------|-------------------------|-------------------------------|----------------------------------------------------------------|
| **Accuracy** | 0.9827 | 0.9884 | 0.9934 | 
| **Precision** | 0.2564 | 0.3404 | 0.4762 | 
| **Recall** | 1.0000 | 1.0000 | 1.0000 | 
| **F1-Score** | 0.4082 | 0.5079 | 0.6452 | 
| **ROC AUC** | 0.9997 | 0.9999 | 1.0000 | 
| **Gini** | 0.9995 | 0.9999 | 1.0000 | 

### Test Confusion Matrix

| Model | TP | FP | TN | FN | False Positive Rate |
|-------|----|----|----|----|---------------------|
| V3_natural (all) | 80 | 232 | 13074 | 0 | 0.0174 |
| V3_natural_recent | 80 | 155 | 13151 | 0 | 0.0116 |
| V3_natural_filtered | 80 | 88 | 13218 | 0 | 0.0066 |

---

## Key Observations

### Training Performance

- **V3_natural (all)**: Precision = 15.44%, Recall = 100.00%, Gini = 99.94%
- **V3_natural_recent**: Precision = 22.27%, Recall = 100.00%, Gini = 99.96%
- **V3_natural_filtered**: Precision = 40.07%, Recall = 100.00%, Gini = 99.99%

- **Precision vs All**: Recent = +6.83%, Filtered = +24.63%
- **Precision Filtered vs Recent**: +17.80%

### Test Performance

- **V3_natural (all)**: Precision = 25.64%, Recall = 100.00%, Gini = 99.95%
- **V3_natural_recent**: Precision = 34.04%, Recall = 100.00%, Gini = 99.99%
- **V3_natural_filtered**: Precision = 47.62%, Recall = 100.00%, Gini = 100.00%

- **Precision vs All**: Recent = +8.40%, Filtered = +21.98%
- **Precision Filtered vs Recent**: +13.58%
- **Gini vs All**: Recent = +0.0004, Filtered = +0.0005
- **Gini Filtered vs Recent**: +0.0001

### Recommendations

Based on the comparison:

- **Best Precision**: V3_natural_filtered (47.62%)
- **Best Gini**: V3_natural_filtered (1.0000)
- **Best Overall**: Consider the model with best balance of precision and Gini
- **Trade-offs**: Filtered model has fewer training samples but higher precision
