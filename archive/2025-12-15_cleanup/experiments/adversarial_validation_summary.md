# Adversarial Validation Results

**Goal:** Identify features that distinguish training from validation datasets

## Summary

- **Adversarial ROC AUC:** 0.9987
- **Interpretation:** HIGH distribution shift - model can easily distinguish datasets

## Feature Shift Categories

- **High-shift features (top 5%):** 42
- **Medium-shift features (top 10%):** 41
- **Low-shift features:** 741

## Top 20 Most Problematic Features

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | teams_this_season_week_4 | 0.067364 |
| 2 | teams_this_season_week_2 | 0.058542 |
| 3 | teams_this_season_week_3 | 0.055395 |
| 4 | month_week_4 | 0.043686 |
| 5 | month_week_3 | 0.040006 |
| 6 | month_week_2 | 0.037522 |
| 7 | month_week_5 | 0.033439 |
| 8 | teams_this_season_week_1 | 0.029707 |
| 9 | month_week_1 | 0.018920 |
| 10 | teams_last_season | 0.015316 |
| 11 | national_team_this_season_week_3 | 0.014511 |
| 12 | season_team_diversity_week_4 | 0.014292 |
| 13 | cum_competitions | 0.013078 |
| 14 | avg_injury_duration_week_3 | 0.012840 |
| 15 | days_since_last_fragility | 0.012769 |
| 16 | current_club_FC Alverca | 0.011105 |
| 17 | avg_injury_duration_week_1 | 0.010357 |
| 18 | national_team_this_season_week_4 | 0.010342 |
| 19 | avg_injury_duration_week_5 | 0.010186 |
| 20 | national_team_this_season_week_2 | 0.009830 |

## Recommendations

1. **Remove high-shift features** from training
2. **Consider removing medium-shift features** if performance doesn't improve
3. **Investigate why these features shift** - may indicate data quality issues
