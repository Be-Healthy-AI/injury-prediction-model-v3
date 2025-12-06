# Feature Drift Analysis - Summary

**Date:** 2025-11-27  
**Analysis:** Feature importance and correlation comparison between training and out-of-sample validation

---

## Key Findings

### 🚨 **CRITICAL ISSUE: Massive Feature Drift Detected**

The analysis reveals **significant correlation drift** between training and validation datasets, explaining the performance gap.

### Statistics Summary

| Metric | Training | Validation | Difference |
|--------|----------|------------|------------|
| **Mean absolute correlation** | 0.0185 | 0.0080 | **-0.0105** |
| **Max absolute correlation** | 0.2420 | 0.1941 | **-0.0479** |
| **Features with \|corr\| > 0.1** | 82 | 8 | **-74 features** |
| **Features with \|corr\| > 0.2** | 18 | 0 | **-18 features** |
| **Features with drift > 0.05** | - | - | **127 features** |
| **Features with drift > 0.1** | - | - | **62 features** |

---

## Top 10 Features with Highest Correlation Drift

These features show strong correlation with injuries in training but **much weaker** correlation in validation:

| Rank | Feature | Train Corr | Val Corr | Drift | Model Importance |
|------|---------|------------|----------|-------|------------------|
| 1 | **cum_inj_starts** | 0.225 | 0.050 | **0.176** | High (RF, GB, LR) |
| 2 | **national_team_appearances** | 0.161 | 0.001 | **0.160** | Medium (RF, GB) |
| 3 | **days_since_last_injury_week_1** | 0.216 | 0.060 | **0.156** | Very High (RF, GB) |
| 4 | **hip_injuries** | 0.149 | 0.001 | **0.148** | Low |
| 5 | **cum_minutes_played_numeric** | 0.163 | 0.019 | **0.145** | High (RF, GB) |
| 6 | **cum_competitions** | 0.188 | 0.051 | **0.137** | High (RF, GB, LR) |
| 7 | **cum_disciplinary_actions** | 0.118 | 0.005 | **0.113** | Medium |
| 8 | **upper_body_injuries** | 0.125 | 0.014 | **0.111** | Low |
| 9 | **cup_competitions** | 0.139 | 0.032 | **0.107** | Medium |
| 10 | **cum_red_cards_numeric** | 0.113 | 0.007 | **0.105** | Low |

---

## Most Problematic Features

### 1. **cum_inj_starts** (Cumulative Injury Starts)
- **Training correlation:** 0.225 (very strong)
- **Validation correlation:** 0.050 (weak)
- **Drift:** 0.176
- **Impact:** This is a **career-level cumulative feature** that captures historical injury frequency. The high correlation in training suggests players with more past injuries are more likely to get injured in the training period, but this pattern doesn't hold in the validation period.

**Possible reasons:**
- Training data may have players with longer injury histories
- Validation period may have different player composition
- This feature may be capturing a spurious correlation from the training distribution

### 2. **national_team_appearances**
- **Training correlation:** 0.161 (strong)
- **Validation correlation:** 0.001 (essentially zero)
- **Drift:** 0.160
- **Impact:** This feature has **almost no predictive power** in validation despite being moderately correlated in training.

**Possible reasons:**
- Training data may have different distribution of national team players
- National team activity patterns may have changed over time
- This could be a spurious correlation

### 3. **days_since_last_injury_week_1**
- **Training correlation:** 0.216 (very strong)
- **Validation correlation:** 0.060 (weak)
- **Drift:** 0.156
- **Impact:** This is a **recent injury recency feature** that should be predictive, but shows much weaker correlation in validation.

**Possible reasons:**
- Training data may have different injury patterns
- Recent injury patterns may have changed
- This could indicate data quality issues or temporal shifts

### 4. **cum_minutes_played_numeric**
- **Training correlation:** 0.163 (strong)
- **Validation correlation:** 0.019 (very weak)
- **Drift:** 0.145
- **Impact:** Career minutes played shows strong correlation in training but almost none in validation.

**Possible reasons:**
- Training data may have different player activity levels
- This could indicate distribution shift in player populations

### 5. **cum_competitions**
- **Training correlation:** 0.188 (strong)
- **Validation correlation:** 0.051 (weak)
- **Drift:** 0.137
- **Impact:** Number of competitions played shows strong correlation in training but weak in validation.

---

## Model-Specific Top Features

### Random Forest (RF)
**Top 5 by Importance:**
1. `days_since_last_injury_week_1` (importance: 0.032, drift: 0.156)
2. `cum_inj_starts` (importance: 0.025, drift: 0.176)
3. `cum_minutes_played_numeric` (importance: 0.024, drift: 0.145)
4. `cum_competitions` (importance: 0.020, drift: 0.137)
5. `competition_frequency` (importance: 0.016, drift: 0.094)

### Gradient Boosting (GB)
**Top 5 by Importance:**
1. `days_since_last_injury_week_1` (importance: 0.026, drift: 0.156)
2. `competition_frequency` (importance: 0.019, drift: 0.094)
3. `cum_minutes_played_numeric` (importance: 0.018, drift: 0.145)
4. `cum_competitions` (importance: 0.018, drift: 0.137)
5. `days_since_last_match_week_5` (importance: 0.018, drift: 0.035)

### Logistic Regression (LR)
**Top 5 by Importance:**
1. `cum_competitions` (importance: 0.034, drift: 0.137)
2. `cum_inj_starts` (importance: 0.028, drift: 0.176)
3. `minutes_per_career_match` (importance: 0.018, drift: 0.039)
4. `national_team_last_season` (importance: 0.017, drift: 0.043)
5. `teams_this_season_week_1` (importance: 0.014, drift: 0.028)

---

## Interpretation

### Why This Explains the Performance Gap

1. **Feature Drift:** 127 features show correlation drift > 0.05, meaning they're predictive in training but not in validation.

2. **Cumulative Features Are Problematic:** Many of the top features are **cumulative/career-level** features (`cum_inj_starts`, `cum_minutes_played_numeric`, `cum_competitions`). These may capture spurious correlations from the training distribution.

3. **Distribution Shift:** The validation period may have:
   - Different player composition
   - Different injury patterns
   - Different activity levels
   - Different competition structures

4. **Temporal Effects:** Features related to recent activity (`days_since_last_injury_week_1`) show high drift, suggesting temporal patterns have changed.

### Recommendations

1. **Investigate Data Quality:**
   - Check if validation period has different player populations
   - Verify if injury reporting patterns changed
   - Check for data collection issues

2. **Feature Engineering:**
   - Consider removing or downweighting high-drift features
   - Focus on features with consistent correlation across periods
   - Create features that are more robust to distribution shifts

3. **Model Retraining:**
   - Retrain with features that show consistent correlation
   - Consider using only recent/activity-based features
   - Remove career-level cumulative features that show high drift

4. **Validation Strategy:**
   - Use time-based cross-validation to detect drift earlier
   - Monitor feature distributions over time
   - Implement drift detection in production

---

## Files Generated

- `experiments/feature_correlation_comparison.csv` - Full correlation comparison
- `experiments/feature_importance_and_correlation.csv` - Combined importance and correlation
- `experiments/suspicious_features_summary.txt` - Detailed suspicious features list

---

**Conclusion:** The performance gap is primarily explained by **feature drift** - features that are highly predictive in training but lose predictive power in validation. This is especially true for cumulative/career-level features and suggests a distribution shift between training and validation periods.



