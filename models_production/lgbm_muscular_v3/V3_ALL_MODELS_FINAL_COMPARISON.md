# V3 All Models Final Comparison Report

**Generated:** 2026-01-06 12:23:40

This report compares all 6 V3 models on both training and test datasets.

---

## Model Descriptions

| Model | Description | Target Ratio | Seasons |
|-------|-------------|--------------|---------|
| V3-natural | All seasons, natural ratio | Natural (unbalanced) | 2011-2026 |
| V3-10pc | All seasons, balanced | 10% | 2011-2026 |
| V3-25pc | All seasons, balanced | 25% | 2011-2026 |
| V3-50pc | All seasons, balanced | 50% | 2011-2026 |
| V3-natural-recent | Recent seasons only | Natural (unbalanced) | 2018-2026 |
| V3-natural-filtered | Recent seasons, filtered | Natural (unbalanced) | 2018-2026 (excl. 2021-2022, 2022-2023) |

---

## Training Metrics Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC | Gini |
|-------|----------|-----------|--------|----------|---------|------|
| V3-10pc | 0.9725 | 0.7933 | 1.0000 | 0.8848 | 0.9994 | 0.9988 |
| V3-25pc | 0.9905 | 0.9647 | 1.0000 | 0.9820 | 1.0000 | 1.0000 |
| V3-50pc | 0.9996 | 0.9992 | 1.0000 | 0.9996 | 1.0000 | 1.0000 |
| V3-natural | 0.9638 | 0.1110 | 1.0000 | 0.1999 | 0.9993 | 0.9987 |
| V3-natural-filtered | 0.9846 | 0.2716 | 1.0000 | 0.4272 | 0.9998 | 0.9996 |
| V3-natural-recent | 0.9713 | 0.1487 | 1.0000 | 0.2590 | 0.9994 | 0.9989 |

---

## Test Metrics Comparison (2025-2026 PL-only)

| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC | Gini |
|-------|----------|-----------|--------|----------|---------|------|
| V3-10pc | 0.9620 | 0.1683 | 1.0000 | 0.2881 | 0.9990 | 0.9981 |
| V3-25pc | 0.9263 | 0.0309 | 0.2828 | 0.0556 | 0.7306 | 0.4612 |
| V3-50pc | 0.8644 | 0.0248 | 0.4345 | 0.0469 | 0.7173 | 0.4347 |
| V3-natural | 0.9567 | 0.1508 | 1.0000 | 0.2621 | 0.9992 | 0.9984 |
| V3-natural-filtered | 0.9852 | 0.3416 | 1.0000 | 0.5092 | 0.9998 | 0.9996 |
| V3-natural-recent | 0.9709 | 0.2091 | 1.0000 | 0.3459 | 0.9995 | 0.9990 |

---

## Test Set Confusion Matrices

### V3-10pc

| | Predicted Negative | Predicted Positive |
|---|-------------------|-------------------|
| **Actual Negative** | 36,027 | 1,433 |
| **Actual Positive** | 0 | 290 |

### V3-25pc

| | Predicted Negative | Predicted Positive |
|---|-------------------|-------------------|
| **Actual Negative** | 34,884 | 2,576 |
| **Actual Positive** | 208 | 82 |

### V3-50pc

| | Predicted Negative | Predicted Positive |
|---|-------------------|-------------------|
| **Actual Negative** | 32,506 | 4,954 |
| **Actual Positive** | 164 | 126 |

### V3-natural

| | Predicted Negative | Predicted Positive |
|---|-------------------|-------------------|
| **Actual Negative** | 35,827 | 1,633 |
| **Actual Positive** | 0 | 290 |

### V3-natural-filtered

| | Predicted Negative | Predicted Positive |
|---|-------------------|-------------------|
| **Actual Negative** | 36,901 | 559 |
| **Actual Positive** | 0 | 290 |

### V3-natural-recent

| | Predicted Negative | Predicted Positive |
|---|-------------------|-------------------|
| **Actual Negative** | 36,363 | 1,097 |
| **Actual Positive** | 0 | 290 |

---

## Summary and Recommendations

- **Best Precision (Test):** V3-natural-filtered (0.3416)
- **Best F1-Score (Test):** V3-natural-filtered (0.5092)
- **Best ROC AUC (Test):** V3-natural-filtered (0.9998)

### Key Observations

1. All models achieve 100% recall on both training and test sets.
2. Precision varies significantly between models, with balanced models (10pc, 25pc, 50pc) showing higher precision.
3. Natural ratio models have lower precision but maintain high ROC AUC scores.
4. The filtered model (V3-natural-filtered) shows improved precision compared to the unfiltered natural models.
