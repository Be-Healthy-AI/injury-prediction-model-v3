# V3 Model Comparison: Recent Seasons vs All Seasons

## Overview

This document compares V3_natural models trained with different season ranges:

- **V3_natural (all)**: Trained on all seasons (2011-2026), natural ratio, PL-only
- **V3_natural_recent**: Trained on recent seasons only (2018-2026), natural ratio, PL-only

Both models are evaluated on the same test set: 2025-2026 PL-only timeline (natural ratio).

---

## Training Metrics Comparison

| Metric | V3_natural (all seasons) | V3_natural_recent (2018-2026) |
|--------|-------------------------|-------------------------------|
| **Accuracy** | 0.9781 | 0.9844 | 
| **Precision** | 0.1544 | 0.2227 | 
| **Recall** | 1.0000 | 1.0000 | 
| **F1-Score** | 0.2675 | 0.3643 | 
| **ROC AUC** | 0.9997 | 0.9998 | 
| **Gini** | 0.9994 | 0.9996 | 

---

## Test Metrics Comparison (2025-2026 PL-only)

| Metric | V3_natural (all seasons) | V3_natural_recent (2018-2026) |
|--------|-------------------------|-------------------------------|
| **Accuracy** | 0.9827 | 0.9884 | 
| **Precision** | 0.2564 | 0.3404 | 
| **Recall** | 1.0000 | 1.0000 | 
| **F1-Score** | 0.4082 | 0.5079 | 
| **ROC AUC** | 0.9997 | 0.9999 | 
| **Gini** | 0.9995 | 0.9999 | 

---

## Confusion Matrix Details (Test Set)

| Model | TP | FP | TN | FN |
|-------|----|----|----|----|
| V3_natural (all) | 80 | 232 | 13074 | 0 |
| V3_natural_recent | 80 | 155 | 13151 | 0 |

---

## Key Observations

### Training Performance
- **V3_natural (all)**: Precision = 15.44%, Recall = 100.00%
- **V3_natural_recent**: Precision = 22.27%, Recall = 100.00%
- **Precision Improvement**: 6.83% higher with recent seasons only

### Test Performance
- **V3_natural (all)**: Precision = 25.64%, Recall = 100.00%, Gini = 99.95%
- **V3_natural_recent**: Precision = 34.04%, Recall = 100.00%, Gini = 99.99%
- **Precision Improvement**: 8.40% higher with recent seasons only
- **Gini Improvement**: 0.04% higher with recent seasons only

### Recommendations
- If recent seasons model shows higher precision with similar Gini/Recall: Consider using recent seasons only
- If recent seasons model shows lower performance: Keep all seasons for better generalization
- Consider the trade-off between precision and dataset size
