# False Negative Analysis Summary
## Model: lgbm_muscular_v3 | Date: 2026-01-02

## Executive Summary

**82 false negative cases** (0.58% of 14,185 predictions) were identified where the model predicted low probability (< 0.3) but players got injured within 5 days. This analysis identifies key patterns and provides actionable recommendations for model improvement.

---

## Key Statistics

### Overall Performance
- **False Negatives**: 82 cases (0.58%)
- **True Positives**: 14 cases (0.10%)
- **True Negatives**: 11,825 cases (83.36%)
- **False Positives**: 2,264 cases (15.96%)

### False Negative Characteristics
- **Average predicted probability**: 0.1115 (very low)
- **Median predicted probability**: 0.0823
- **Range**: 0.0062 to 0.2959
- **Average days to injury**: 2.94 days
- **Injury timing**: Evenly distributed across 1-5 days (17, 17, 17, 16, 15 cases)

### True Positive Comparison
- **Average predicted probability**: 0.6162 (much higher)
- **Median predicted probability**: 0.5489

---

## Critical Findings

### 1. **Over-Reliance on "Time Since Last Injury" Features**

**Problem**: The model is being too conservative when players have been injury-free for extended periods, even when other risk factors suggest elevated risk.

**Evidence**:
- **FN cases**: `days_since_last_upper_body_week_1` = **3,561 days** (avg) with **negative SHAP** (-1.075)
- **FN cases**: `days_since_last_lower_leg_week_1` = **1,469 days** (avg) with **negative SHAP** (-0.012)
- **TP cases**: `days_since_last_lower_leg_week_1` = **404 days** (avg) with **positive SHAP** (+0.299)

**Impact**: Players who haven't had upper body injuries in ~10 years are getting very low risk scores, even when workload and other factors suggest risk.

### 2. **Injury History × Workload Feature Not Strong Enough**

**Finding**: False negatives actually have **higher** `injury_history_x_recent_workload` values than true positives:
- **FN average**: 1,468.9
- **TP average**: 1,293.1

However, the SHAP values show:
- **FN**: +0.593 (moderate positive)
- **TP**: +0.864 (strong positive)

**Issue**: The feature is present but not strong enough to overcome the negative impact of long "time since injury" features.

### 3. **Feature Differences: FN vs TP**

**Common features** (appear in both):
- `injury_history_x_recent_workload` (stronger in TP)
- `cum_matches_bench` (stronger in TP)
- `early_substitution_off_count_week_2` (appears in both)

**FN-only features** (unique to false negatives):
- `days_since_last_upper_body_week_1` (very long, negative SHAP)
- `days_since_last_skeletal_week_1` (very long, negative SHAP)
- `days_since_last_lower_leg_week_1` (very long, negative SHAP)
- `lower_leg_injuries` (negative SHAP)
- `teams_last_season` (negative SHAP)

**TP-only features** (unique to true positives):
- `matches_this_season_to_last_ratio` (positive SHAP)
- `season_team_diversity_week_1` (positive SHAP)
- `minutes_per_match_week_1` (negative SHAP - lower minutes)
- `international_competitions` (positive SHAP)

### 4. **Player and Club Patterns**

**Top players with multiple FN cases**:
- Estêvão (Chelsea FC): 10 cases
- Marc Cucurella (Chelsea FC): 10 cases
- Ben White (Arsenal FC): 5 cases
- Yankuba Minteh (Brighton): 5 cases
- Riccardo Calafiori (Arsenal FC): 5 cases

**Club distribution**:
- Chelsea FC: 20 cases (24%)
- Liverpool FC: 12 cases (15%)
- Arsenal FC: 10 cases (12%)
- Manchester City: 9 cases (11%)
- Manchester United: 9 cases (11%)

**Temporal pattern**: 79 cases in December 2025, 3 in January 2026

---

## Root Cause Analysis

### Primary Issue: Diminishing Returns on "Time Since Injury"

The model appears to have learned that "long time since injury = low risk" too strongly. However, this relationship may have **diminishing returns**:
- A player who hasn't been injured in 1 year vs 2 years: significant difference
- A player who hasn't been injured in 9 years vs 10 years: minimal difference

But the model treats both extremes similarly, and the very long "time since injury" values are masking other risk factors.

### Secondary Issue: Workload Signals Not Strong Enough

While `injury_history_x_recent_workload` is present in FN cases, its positive signal is being overwhelmed by the negative signals from long "time since injury" features. The model needs to better balance these competing signals.

---

## Recommendations for Model Improvement

### 1. **Cap or Transform "Time Since Injury" Features**

**Action**: Apply a non-linear transformation to `days_since_last_*` features to reduce their impact beyond a certain threshold (e.g., 2-3 years).

**Options**:
- **Log transformation**: `log(1 + days_since_last_injury)`
- **Sigmoid transformation**: `1 / (1 + exp(-(days - threshold) / scale))`
- **Capping**: `min(days_since_last_injury, 730)` (cap at 2 years)

**Rationale**: After 2-3 years, additional time since injury provides minimal additional information about current risk.

### 2. **Strengthen Workload-Based Features**

**Action**: Enhance the `injury_history_x_recent_workload` feature or create additional workload interaction features.

**Options**:
- **Recent workload acceleration**: Rate of change in minutes over the last 2-3 weeks
- **Workload × position**: Different positions have different injury risk profiles
- **Workload × age**: Older players may be more sensitive to workload spikes
- **Workload × recovery time**: Recent matches with short recovery periods

**Rationale**: Workload patterns are strong predictors but need more weight relative to historical injury-free periods.

### 3. **Add "Recent Pattern" Features**

**Action**: Create features that capture recent changes in player status, even if historical injury data is sparse.

**Options**:
- **Recent substitution patterns**: Early substitutions may indicate fatigue/risk
- **Recent match intensity**: High-intensity matches in short succession
- **Recent training load**: If available, training load data
- **Recent performance decline**: Drop in performance metrics may indicate fatigue

**Rationale**: These features can signal risk even when historical injury data suggests low risk.

### 4. **Feature Engineering: "Risk Context"**

**Action**: Create composite features that combine multiple risk signals.

**Options**:
- **Workload spike indicator**: Recent increase in minutes > X%
- **Fatigue indicator**: High minutes + short recovery + early substitutions
- **Risk convergence**: Multiple risk factors present simultaneously

**Rationale**: Single features may not be strong enough, but combinations can be more predictive.

### 5. **Model Architecture Adjustments**

**Action**: Consider model adjustments to better handle feature interactions.

**Options**:
- **Feature importance reweighting**: Manually adjust feature importance for "time since injury" features
- **Ensemble approach**: Combine models with different feature weightings
- **Threshold optimization**: Adjust decision thresholds based on false negative cost

**Rationale**: The model may need architectural changes to better balance competing signals.

### 6. **Player-Specific Calibration**

**Action**: Identify players with recurring false negatives and investigate player-specific factors.

**Options**:
- **Player-level analysis**: Deep dive into Estêvão, Marc Cucurella, Ben White cases
- **Position-specific models**: Different models for different positions
- **Club-specific factors**: Some clubs may have different injury patterns

**Rationale**: Some players may have unique risk profiles not captured by general features.

---

## Immediate Actions

### Short-Term (Next Model Training Cycle)

1. **Apply log transformation** to all `days_since_last_*` features
2. **Create workload acceleration features** (rate of change in minutes)
3. **Add recent substitution pattern features** (early substitutions, substitution frequency)
4. **Retrain model** with these changes

### Medium-Term (Next 2-3 Training Cycles)

1. **Implement feature capping** for "time since injury" features (cap at 2-3 years)
2. **Create composite risk indicators** (workload spike, fatigue indicator)
3. **Add position-specific features** (position × workload interactions)
4. **Conduct player-specific analysis** for recurring false negative cases

### Long-Term (Ongoing)

1. **Collect additional data**:
   - Training load data
   - GPS tracking data
   - Sleep/recovery data
   - Player-reported fatigue
2. **Develop ensemble models** with different feature weightings
3. **Implement active learning** to focus training on edge cases

---

## Validation Plan

After implementing changes, validate improvements by:

1. **Re-running false negative analysis** on new model predictions
2. **Comparing calibration charts** (before vs after)
3. **Tracking false negative rate** over time
4. **Monitoring player-specific cases** (Estêvão, Cucurella, etc.)

---

## Conclusion

The primary issue is the model's over-reliance on "time since last injury" features, which masks other risk factors when players have been injury-free for extended periods. By applying non-linear transformations to these features and strengthening workload-based signals, we can improve the model's ability to identify at-risk players even when they have clean injury histories.

The 82 false negative cases represent a small percentage (0.58%) but are critical for player safety. Addressing these cases will improve the model's practical utility and trustworthiness.

---

## Files Generated

1. **Detailed false negatives CSV**: `false_negative_analysis_20260102_false_negatives.csv`
2. **FN feature analysis CSV**: `false_negative_analysis_20260102_fn_features.csv`
3. **TP feature analysis CSV**: `false_negative_analysis_20260102_tp_features.csv`
4. **Full analysis report**: `false_negative_analysis_20260102.txt`

