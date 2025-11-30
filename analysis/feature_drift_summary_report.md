# Feature Distribution Drift Analysis Summary

**Analysis Date:** 2025-11-26  
**Training Period:** ≤ 2025-06-30  
**Validation Period:** ≥ 2025-07-01

---

## Executive Summary

### Overall Findings
- **Total Features Analyzed:** 1,690
- **Features with Detected Drift:** 85 (5.0%)
- **Features without Drift:** 1,605 (95.0%)

### Critical Insight
While only 5% of all features show drift, **70.8% of numeric features (85 out of 120) exhibit distribution drift** between training and validation periods. This is a significant concern as numeric features are typically the most important for model predictions.

---

## Drift Severity Breakdown

| Severity | Count | Percentage | Description |
|----------|-------|------------|-------------|
| **Severe** | 2 | 0.1% | Large effect size (Cohen's d > 0.8) |
| **Moderate** | 14 | 0.8% | Medium effect size (Cohen's d 0.5-0.8) |
| **Mild** | 27 | 1.6% | Small effect size (Cohen's d 0.2-0.5) |
| **Minimal** | 31 | 1.8% | Very small effect size (Cohen's d < 0.2) |
| **None** | 1,616 | 95.6% | No statistically significant drift |

---

## Feature Type Analysis

| Type | Total Count | Drift Count | Drift % |
|------|-------------|-------------|---------|
| **Numeric** | 120 | 85 | **70.8%** ⚠️ |
| **Categorical** | 1,570 | 0 | 0.0% |

**Key Finding:** All drift is concentrated in numeric features. Categorical features (one-hot encoded) show no drift, which is expected as they represent binary presence/absence indicators.

---

## Top 20 Most Drifted Features

### Severe Drift (Cohen's d > 0.8)

1. **`teams_this_season_week_5`** ⚠️⚠️⚠️
   - **Severity:** Severe
   - **Cohen's d:** -1.69
   - **Mean Difference:** -70.1%
   - **Issue:** Validation set has 70% fewer teams in week 5 compared to training
   - **Impact:** This feature is likely unreliable for predictions in validation period

2. **`national_team_this_season_week_5`** ⚠️⚠️⚠️
   - **Severity:** Severe
   - **Cohen's d:** -1.01
   - **Mean Difference:** -89.3%
   - **Zero % Change:** 43.9% → 83.5% (39.6% increase in zeros)
   - **Issue:** National team activity drops dramatically in validation period
   - **Impact:** Feature becomes mostly zero in validation, losing predictive power

### Moderate Drift (Cohen's d 0.5-0.8)

3. **`month_week_5`** - Mean diff: +31.9%
4. **`cum_competitions`** - Mean diff: +38.7%
5. **`season_team_diversity_week_5`** - Mean diff: +30.4%
6. **`seasons_count`** - Mean diff: +40.5%
7. **`cum_teams`** - Mean diff: +31.7%
8. **`cum_matches_bench`** - Mean diff: +56.2%
9. **`teams_last_season`** - Mean diff: +24.1%
10. **`cum_team_losses_week_5`** - Mean diff: +38.4%
11. **`career_matches`** - Mean diff: +37.6%
12. **`competition_experience`** - Mean diff: +37.6%
13. **`cum_team_draws_week_5`** - Mean diff: +34.8%
14. **`cum_team_points_week_5`** - Mean diff: +38.0%
15. **`cum_team_wins_week_5`** - Mean diff: +38.5%
16. **`cup_competitions`** - Mean diff: +39.7%

---

## Key Patterns Identified

### 1. **Week 5 Features Show Severe Drift** 🔴
- `teams_this_season_week_5`: -70.1% drift
- `national_team_this_season_week_5`: -89.3% drift
- `season_team_diversity_week_5`: +30.4% drift

**Implication:** The 5-week rolling window may be capturing temporal patterns that don't generalize across the train/validation split.

### 2. **Cumulative Career Features Show Systematic Increase** 📈
All cumulative career features show positive drift (higher values in validation):
- `career_matches`: +37.6%
- `cum_competitions`: +38.7%
- `cum_teams`: +31.7%
- `seasons_count`: +40.5%

**Implication:** Validation period contains players with more career experience on average. This is expected as time progresses, but may affect model predictions if it wasn't accounted for during training.

### 3. **Injury-Related Features Show Mixed Patterns** 🩹
- `other_injuries`: +122.7% (large increase)
- `cum_inj_starts`: +54.2% (more injuries in validation)
- `cum_inj_days`: +45.7% (more injury days)
- `cum_matches_injured`: +41.9% (more matches missed due to injury)

**Zero-value changes:**
- `cum_inj_starts`: 39.0% → 18.1% zeros (-20.9%)
- `cum_inj_days`: 38.9% → 18.1% zeros (-20.7%)
- `cum_matches_injured`: 41.2% → 19.5% zeros (-21.7%)

**Implication:** Validation period has more players with injury history, which could affect model performance if the model was trained on a different distribution.

### 4. **Team Performance Features Show Consistent Increase** ⚽
All team performance features (wins, losses, draws, points) show ~35-40% increase:
- `cum_team_wins_week_5`: +38.5%
- `cum_team_losses_week_5`: +38.4%
- `cum_team_draws_week_5`: +34.8%
- `cum_team_points_week_5`: +38.0%

**Implication:** These features may be capturing cumulative effects that grow over time, making them less stable across temporal splits.

---

## Features with Significant Zero-Value Changes

Features where the percentage of zero values changed significantly:

| Feature | Train Zero % | Val Zero % | Difference |
|---------|--------------|------------|------------|
| `national_team_this_season_week_5` | 43.9% | 83.5% | **+39.6%** ⚠️ |
| `other_injuries` | 70.5% | 48.0% | **-22.5%** |
| `cum_matches_injured` | 41.2% | 19.5% | **-21.7%** |
| `cum_inj_starts` | 39.0% | 18.1% | **-20.9%** |
| `cum_inj_days` | 38.9% | 18.1% | **-20.7%** |
| `injury_frequency_week_5` | 38.9% | 18.1% | **-20.7%** |

**Implication:** Features that become mostly zero (like `national_team_this_season_week_5`) lose their predictive power. Features that become less sparse (like injury features) may have different distributions.

---

## Recommendations

### 1. **Review Week 5 Feature Windowing** 🔴 HIGH PRIORITY
The severe drift in `teams_this_season_week_5` and `national_team_this_season_week_5` suggests:
- The 5-week window may be too sensitive to temporal patterns
- Consider using shorter windows (e.g., 2-3 weeks) for season-specific features
- Or use relative/percentage-based features instead of absolute counts

### 2. **Normalize Cumulative Features** 📊 MEDIUM PRIORITY
Cumulative features show systematic increases. Consider:
- Normalizing by time period (e.g., per season, per year)
- Using rates instead of absolute counts
- Adding time-based decay factors

### 3. **Feature Engineering Adjustments** 🔧 MEDIUM PRIORITY
- **Season-specific features:** Use relative measures (e.g., "teams_this_season / teams_last_season") instead of absolute
- **Career features:** Normalize by career duration or use percentiles
- **Team performance:** Use win rate instead of cumulative wins

### 4. **Monitor Feature Stability** 📈 ONGOING
- Implement feature drift monitoring in production
- Set up alerts for features with drift > 20%
- Regularly retrain models when significant drift is detected

### 5. **Consider Feature Selection** 🎯 MEDIUM PRIORITY
- Remove or down-weight severely drifted features (`teams_this_season_week_5`, `national_team_this_season_week_5`)
- Focus on features with stable distributions across time periods
- Test model performance with/without drifted features

### 6. **Temporal Validation Strategy** ⏰ HIGH PRIORITY
- Use rolling window validation instead of single split
- Train on multiple time periods to ensure robustness
- Consider ensemble models trained on different time windows

---

## Next Steps

1. ✅ **Completed:** Feature drift analysis
2. 🔄 **In Progress:** Review windowing approach for week 5 features
3. ⏳ **Pending:** Implement feature normalization for cumulative features
4. ⏳ **Pending:** Test model performance with adjusted features
5. ⏳ **Pending:** Implement feature drift monitoring

---

## Files Generated

- `analysis/feature_drift_analysis.csv` - Detailed drift metrics for all features
- `analysis/feature_drift_summary.json` - Summary statistics in JSON format
- `analysis/feature_drift_summary_report.md` - This report

---

**Analysis performed by:** Feature Drift Analysis Script  
**Script:** `scripts/analyze_feature_drift.py`

