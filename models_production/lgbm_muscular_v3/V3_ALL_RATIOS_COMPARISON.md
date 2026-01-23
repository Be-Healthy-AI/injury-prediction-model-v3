# V3 Model Comparison: All Target Ratios vs V2

## Overview

This document compares the performance of V3 models trained with different target ratios against V2:

- **V2**: Trained on all seasons (2008-2026), 10% target ratio, tested on 2025-2026 natural timeline (in-sample)
- **V3_10pc**: PL-only timelines, 10% target ratio, tested on 2025-2026 PL-only
- **V3_25pc**: PL-only timelines, 25% target ratio, tested on 2025-2026 PL-only
- **V3_50pc**: PL-only timelines, 50% target ratio, tested on 2025-2026 PL-only
- **V3_natural**: PL-only timelines, natural/unbalanced ratio, tested on 2025-2026 PL-only

All V3 models are evaluated on the same test set: 2025-2026 PL-only timeline (natural ratio).

---

## Training Metrics Comparison

| Metric | V2 | V3_10pc | V3_25pc | V3_50pc | V3_natural |
|--------|----|---------|---------|---------|------------|
| **Accuracy** | 86.25% | 98.54% | 99.61% | 100.00% | 97.81% |
| **Precision** | 41.97% | 86.90% | 98.44% | 100.00% | 15.44% |
| **Recall** | 98.03% | 100.00% | 100.00% | 100.00% | 100.00% |
| **F1-Score** | 58.77% | 92.99% | 99.21% | 100.00% | 26.75% |
| **ROC AUC** | 96.88% | 99.99% | 100.00% | 100.00% | 99.97% |
| **Gini** | 93.77% | 99.98% | 100.00% | 100.00% | 99.94% |

### Training Dataset Sizes

- **V2**: ~4,000,000 records (all seasons, 10% ratio)
- **V3_10pc**: 32,008 records (PL-only, 10% ratio)
- **V3_25pc**: 12,364 records (PL-only, 25% ratio)
- **V3_50pc**: 6,170 records (PL-only, 50% ratio)
- **V3_natural**: 777,368 records (PL-only, natural ratio)

---

## Test Metrics Comparison (2025-2026)

| Metric | V2 (natural) | V3_10pc (PL-only) | V3_25pc (PL-only) | V3_50pc (PL-only) | V3_natural (PL-only) |
|--------|--------------|-------------------|-------------------|-------------------|----------------------|
| **Accuracy** | 84.29% | 97.91% | 93.31% | 85.01% | 98.27% |
| **Precision** | 3.84% | 22.22% | 1.89% | 1.99% | 25.64% |
| **Recall** | 96.77% | 100.00% | 20.00% | 50.00% | 100.00% |
| **F1-Score** | 7.38% | 36.36% | 3.45% | 3.84% | 40.82% |
| **ROC AUC** | 96.74% | 99.97% | 68.45% | 70.58% | 99.97% |
| **Gini** | 93.47% | 99.93% | 36.90% | 41.15% | 99.95% |

### Test Dataset Sizes

- **V2**: 153,006 records (2025-2026 natural timeline)
- **V3_10pc, V3_25pc, V3_50pc, V3_natural**: 13,386 records (2025-2026 PL-only timeline)

---

## Confusion Matrix Details (Test Set)

| Model | TP | FP | TN | FN |
|-------|----|----|----|----|
| **V2** | 958 | 24,011 | 128,005 | 32 |
| **V3_10pc** | 80 | 280 | 13,026 | 0 |
| **V3_25pc** | 16 | 831 | 12,475 | 64 |
| **V3_50pc** | 40 | 1,966 | 11,340 | 40 |
| **V3_natural** | 80 | 232 | 13,074 | 0 |

---

## Key Findings

### 1. Training Performance

- **All V3 models** show superior training metrics compared to V2, with Gini coefficients > 99.9%
- **V3_50pc** achieves perfect training metrics (100% across all metrics), indicating potential overfitting
- **V3_25pc** also shows near-perfect training performance (99.61% accuracy, 98.44% precision)
- **V3_natural** has lower precision (15.44%) due to class imbalance but maintains high recall (100%) and Gini (99.94%)

### 2. Test Performance - Critical Insights

#### **V3_natural: Best Overall Performance** 🏆
- **Highest Gini**: 99.95% (vs 93.47% V2, 99.93% V3_10pc)
- **Highest Precision**: 25.64% (vs 3.84% V2, 22.22% V3_10pc)
- **Perfect Recall**: 100% (0 false negatives)
- **Best F1-Score**: 40.82% (vs 7.38% V2, 36.36% V3_10pc)
- **Lowest False Positives**: 232 (vs 280 V3_10pc, 831 V3_25pc, 1,966 V3_50pc)

#### **V3_10pc: Strong Performance**
- **Excellent Gini**: 99.93% (nearly identical to V3_natural)
- **Good Precision**: 22.22% (significantly better than V2's 3.84%)
- **Perfect Recall**: 100% (0 false negatives)
- **Good F1-Score**: 36.36%

#### **V3_25pc and V3_50pc: Overfitting Detected** ⚠️
- **Severe Overfitting**: Perfect training metrics but poor test performance
- **V3_25pc**: Gini drops from 100% (train) to 36.90% (test) - **63.1% drop**
- **V3_50pc**: Gini drops from 100% (train) to 41.15% (test) - **58.85% drop**
- **Low Recall**: V3_25pc only captures 20% of injuries, V3_50pc captures 50%
- **Very Low Precision**: Both models have <2% precision on test set

### 3. Comparison with V2

| Aspect | V2 | Best V3 (natural) | Improvement |
|--------|----|-------------------|-------------|
| **Gini** | 93.47% | 99.95% | +6.48% |
| **Precision** | 3.84% | 25.64% | +21.80% (6.7x better) |
| **Recall** | 96.77% | 100.00% | +3.23% |
| **F1-Score** | 7.38% | 40.82% | +33.44% (5.5x better) |
| **False Positives** | 24,011 | 232 | 99.0% reduction |

---

## Recommendations

### Best Model: **V3_natural** ✅

**Reasons:**
1. **Best test performance** across all metrics (Gini, Precision, F1-Score)
2. **Perfect recall** (100%) - captures all injuries
3. **Highest precision** (25.64%) - best balance between precision and recall
4. **Lowest false positives** (232) - reduces unnecessary alerts
5. **No overfitting** - excellent generalization from training to test
6. **Realistic training distribution** - trained on natural class imbalance, which matches real-world conditions

### Alternative: **V3_10pc**

**Use when:**
- You need a balanced training set (10% ratio)
- Slightly lower precision (22.22% vs 25.64%) is acceptable
- Similar performance to V3_natural but with balanced training data

### Avoid: **V3_25pc and V3_50pc** ❌

**Reasons:**
1. **Severe overfitting** - perfect training metrics but poor test performance
2. **Low recall** - miss many injuries (V3_25pc: 80% missed, V3_50pc: 50% missed)
3. **Very low precision** - high false positive rate
4. **Poor Gini** - weak ranking ability (36.90% and 41.15%)

---

## Insights and Analysis

### Why V3_natural Performs Best

1. **Realistic Class Distribution**: Training on natural ratio (0.40% injury rate) matches the test set distribution, leading to better generalization
2. **Larger Training Set**: 777,368 records provide more diverse examples
3. **No Overfitting**: The model learns realistic patterns rather than memorizing the balanced training distribution
4. **Better Feature Learning**: The natural imbalance forces the model to learn more discriminative features

### Why V3_25pc and V3_50pc Overfit

1. **Small Training Sets**: 12,364 and 6,170 records respectively - too small for the high target ratios
2. **Artificial Balance**: 25% and 50% ratios are far from the natural 0.40% rate, causing the model to learn patterns that don't generalize
3. **Perfect Training Metrics**: 100% accuracy on training suggests memorization rather than learning

### PL-Only Filtering Impact

- **Reduced Dataset Size**: PL-only filtering reduces test set from 153K (V2) to 13K (V3)
- **Improved Performance**: Despite smaller size, V3 models show superior performance
- **Better Context**: Focusing on PL context improves model relevance and precision

---

## Summary

**V3_natural is the clear winner**, demonstrating:
- **6.7x better precision** than V2 (25.64% vs 3.84%)
- **5.5x better F1-Score** than V2 (40.82% vs 7.38%)
- **99.0% reduction in false positives** (232 vs 24,011)
- **Perfect recall** (100%) with excellent precision (25.64%)
- **No overfitting** - excellent generalization

The natural ratio approach proves that training on realistic class distributions, even with severe imbalance, produces better models than artificially balanced datasets when the test set also has natural imbalance.

---

## Next Steps

1. **Deploy V3_natural** as the primary model for injury prediction
2. **Monitor performance** on new data to ensure continued generalization
3. **Consider ensemble** approaches combining V3_natural and V3_10pc if needed
4. **Investigate feature importance** to understand what drives the superior performance

