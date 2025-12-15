# Why F1-Score Drops: 80/20 Random Split vs Temporal Split

## Executive Summary

The dramatic F1-score drop (from ~97% to ~4%) when using temporal split instead of 80/20 random split is **primarily due to temporal feature drift** - features have significantly different distributions in different time periods, and the model cannot generalize from past to future periods.

## Key Findings

### 1. Feature Distribution Drift

| Metric | 80/20 Random Split | Temporal Split | Difference |
|--------|-------------------|----------------|------------|
| **Mean KS Statistic** | 0.0058 | 0.0656 | **11.3x higher** |
| **Mean % Difference** | 3.43% | 25.71% | **7.5x higher** |
| **High Drift Features** | 0 | 21 | **21 features** |
| **Medium Drift Features** | 8 | 44 | **5.5x more** |
| **Low Drift Features** | 92 | 35 | **2.6x fewer** |

**Conclusion:** Temporal split has **11x more feature drift** than random split.

### 2. Top Drifted Features (Temporal Split)

The most drifted features are **cumulative metrics** that naturally increase over time:

| Feature | Train Mean | Val Mean | Mean Diff % | KS Stat | Drift Severity |
|---------|------------|----------|-------------|---------|----------------|
| `cum_competitions` | 15.16 | 21.65 | -42.84% | 0.3241 | High |
| `cum_matches_bench` | 26.40 | 42.50 | -61.00% | 0.2939 | High |
| `cum_matches_not_selected` | 95.67 | 128.26 | -34.06% | 0.2760 | High |
| `cum_minutes_played_numeric` | 12,602 | 18,550 | -47.20% | 0.2279 | High |
| `cum_inj_starts` | 2.98 | 5.24 | -76.02% | 0.2479 | High |
| `other_injuries` | 0.35 | 0.88 | -154.27% | 0.2195 | High |
| `covid_count` | 0.16 | 0.38 | -131.13% | 0.1773 | High |

**Key Insight:** Players in the validation period (2024-07-01 to 2025-06-30) have:
- More cumulative experience (more matches, minutes, competitions)
- More injury history (more cumulative injuries)
- Different patterns (e.g., COVID-related features)

### 3. Dataset Characteristics

| Split Type | Training Size | Training Ratio | Validation Size | Validation Ratio | Player Overlap |
|------------|---------------|----------------|-----------------|------------------|----------------|
| **80/20 Random** | 50,160 | 10.0% | 12,540 | 10.0% | 100% (225/225) |
| **Temporal** | 55,087 | 9.6% | 7,613 | 13.1% | 99.5% (218/219) |

**Key Observations:**
- Player overlap is similar (100% vs 99.5%), so this is NOT the issue
- Injury ratio differs in temporal split (9.6% vs 13.1%), but this alone doesn't explain the drop
- The main issue is **feature distribution shift**, not player composition

### 4. Injury Pattern Analysis

| Split Type | Training Injuries | Training Date Range | Validation Injuries | Validation Date Range |
|------------|-------------------|---------------------|---------------------|----------------------|
| **80/20 Random** | 5,016 | 2007-02-10 to 2025-06-17 | 1,254 | 2007-02-12 to 2025-05-31 |
| **Temporal** | 5,270 | 2007-02-10 to 2024-06-30 | 1,000 | 2024-07-01 to 2025-06-17 |

**Key Insight:** In the 80/20 random split, both training and validation contain injuries from **all time periods** (2007-2025), allowing the model to learn patterns across time. In the temporal split, validation injuries are from a **completely different time period** (2024-07-01 to 2025-06-17), which the model has never seen during training.

## Root Cause Explanation

### Why 80/20 Random Split Works Well

1. **No Temporal Drift:** Features have similar distributions because both sets contain data from all time periods
2. **Pattern Recognition:** Model learns patterns that work across all time periods
3. **Data Leakage (Beneficial):** Model can see future patterns during training, making validation easier
4. **Consistent Distributions:** Training and validation have identical feature distributions

### Why Temporal Split Fails

1. **Severe Temporal Drift:** Features have different distributions in different time periods
   - Cumulative features are much higher in validation period
   - Players have more history, different injury patterns
   - COVID-related features differ significantly

2. **Distribution Mismatch:** Model learns on one distribution (2001-2024) but must predict on another (2024-2025)
   - Features that were predictive in training period are not predictive in validation period
   - Model's learned thresholds don't apply to new distribution

3. **No Future Information:** Model cannot use patterns from validation period during training
   - This is more realistic but much harder
   - Model must truly generalize, not just memorize

4. **Overfitting to Past:** Tree-based models (RF, GB) memorize patterns specific to training period
   - Perfect training performance (100% recall) but fails on new distribution
   - Logistic Regression is more robust (57% recall vs 2-3% for trees)

## Performance Comparison

| Model | Split Type | Training F1 | Validation F1 | Gap |
|-------|------------|-------------|---------------|-----|
| **RF** | 80/20 Random | ~0.97 | ~0.91 | 0.06 |
| **RF** | Temporal | 0.98 | 0.04 | **0.94** |
| **GB** | 80/20 Random | ~1.00 | ~0.97 | 0.03 |
| **GB** | Temporal | 1.00 | 0.05 | **0.95** |
| **LR** | 80/20 Random | ~0.39 | ~0.38 | 0.01 |
| **LR** | Temporal | 0.39 | 0.28 | 0.11 |

**Key Insight:** Logistic Regression is **much more robust** to temporal drift (F1 gap: 0.11 vs 0.94-0.95 for tree models).

## Recommendations

1. **Use Temporal Split for Realistic Evaluation:** The temporal split is more realistic and reveals true model generalization ability

2. **Address Feature Drift:**
   - Normalize cumulative features by player age/career length
   - Use relative features instead of absolute cumulative values
   - Apply feature selection to remove highly drifted features

3. **Consider More Robust Models:**
   - Logistic Regression shows better generalization
   - Regularize tree-based models more aggressively
   - Use ensemble methods that are more robust to drift

4. **Feature Engineering:**
   - Convert cumulative features to rates (per season, per year)
   - Use rolling windows instead of cumulative sums
   - Normalize features by time period

5. **Model Calibration:**
   - Recalibrate models periodically as new data arrives
   - Use online learning or incremental updates
   - Monitor feature drift and retrain when drift exceeds threshold

## Conclusion

The poor F1-score in temporal split is **not a bug** - it's the model's true generalization ability. The 80/20 random split gives **overly optimistic results** because it allows the model to see patterns from all time periods during training. The temporal split reveals that the model cannot generalize from past to future periods due to:

1. **Severe feature drift** (11x higher than random split)
2. **Distribution mismatch** between training and validation periods
3. **Overfitting** to past patterns that don't apply to future

This is a fundamental challenge in time-series prediction and requires different approaches than standard cross-validation.



