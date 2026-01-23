# V3 Filtered Models Comparison Report

**Generated:** 2026-01-06 13:22:55

This report compares two V3 filtered models:

1. **V3-natural-filtered**: Excludes 2021-2022 and 2022-2023 seasons
2. **V3-natural-filtered-excl-2023-2024**: Excludes 2021-2022, 2022-2023, and 2023-2024 seasons

---

## Training Metrics Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC | Gini |
|-------|----------|-----------|--------|----------|---------|------|
| V3-natural-filtered | 0.9846 | 0.2716 | 1.0000 | 0.4272 | 0.9998 | 0.9996 |
| V3-natural-filtered-excl-2023-2024 | 0.9900 | 0.3840 | 1.0000 | 0.5549 | 0.9999 | 0.9998 |

---

## Test Metrics Comparison (2025-2026 PL-only)

| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC | Gini |
|-------|----------|-----------|--------|----------|---------|------|
| V3-natural-filtered | 0.9852 | 0.3416 | 1.0000 | 0.5092 | 0.9998 | 0.9996 |
| V3-natural-filtered-excl-2023-2024 | 0.9917 | 0.4809 | 1.0000 | 0.6495 | 0.9998 | 0.9996 |

---

## Test Set Confusion Matrices

### V3-natural-filtered

| | Predicted Negative | Predicted Positive |
|---|-------------------|-------------------|
| **Actual Negative** | 36,901 | 559 |
| **Actual Positive** | 0 | 290 |

### V3-natural-filtered-excl-2023-2024

| | Predicted Negative | Predicted Positive |
|---|-------------------|-------------------|
| **Actual Negative** | 37,147 | 313 |
| **Actual Positive** | 0 | 290 |

---

## Summary

### Key Differences

- **Training Records**: V3-natural-filtered: 476,921 | V3-natural-filtered-excl-2023-2024: 357,281
- **Training Positives**: V3-natural-filtered: 2,739 | V3-natural-filtered-excl-2023-2024: 2,219

### Performance Comparison

- **Precision (Test)**: V3-natural-filtered-excl-2023-2024 is +13.94% vs V3-natural-filtered
- **F1-Score (Test)**: V3-natural-filtered-excl-2023-2024 is +14.03% vs V3-natural-filtered
- **ROC AUC (Test)**: V3-natural-filtered-excl-2023-2024 is -0.0000 vs V3-natural-filtered
