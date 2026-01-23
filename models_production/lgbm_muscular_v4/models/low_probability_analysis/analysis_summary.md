# Low-Probability Injury Analysis - Muscular Injuries

**Generated:** 2026-01-21 18:50:30

## Overview

- **Total injuries in test set:** 300
- **Low-probability injuries (0.0-0.3):** 227 (75.7%)
- **High-probability injuries (0.7-1.0):** 21 (7.0%)

## Key Findings

### Comparison 1: Low vs High Probability Injuries

- **Significant differences found:** 199 features

**Top 10 Most Different Features:**

1. **muscular_injuries_last_2_years**: Low=1.3304, High=4.1905, Diff=+2.8601 (p=0.0000)
2. **yellow_cards_numeric_week_4**: Low=0.0396, High=0.4286, Diff=+0.3889 (p=0.0000)
3. **workload_spike_14d_week_3**: Low=0.1013, High=0.6803, Diff=+0.5790 (p=0.0000)
4. **upper_leg_injuries**: Low=1.6035, High=4.6190, Diff=+3.0155 (p=0.0000)
5. **workload_acceleration_14d_week_3**: Low=0.0427, High=10.3021, Diff=+10.2593 (p=0.0000)
6. **injuries_last_2_years**: Low=2.6256, High=5.1905, Diff=+2.5649 (p=0.0000)
7. **muscular_to_total_ratio**: Low=0.3541, High=0.6618, Diff=+0.3077 (p=0.0000)
8. **impact_substitution_count_week_4**: Low=0.0000, High=0.1429, Diff=+0.1429 (p=0.0000)
9. **workload_acceleration_14d_week_4**: Low=0.0569, High=1.0428, Diff=+0.9859 (p=0.0000)
10. **workload_spike_14d_week_4**: Low=0.1674, High=0.6122, Diff=+0.4448 (p=0.0000)

### Comparison 2: Low Probability Test vs All Training Injuries

- **Significant differences found:** 290 features

**Top 10 Most Different Features:**

1. **season_year**: Test=2025.0000, Training=2021.7797, Diff=-3.2203 (p=0.0000)
2. **days_into_season**: Test=80.4934, Training=171.4509, Diff=+90.9575 (p=0.0000)
3. **teams_season_today_week_5**: Test=10.2247, Training=22.3344, Diff=+12.1097 (p=0.0000)
4. **teams_season_today_week_4**: Test=9.8018, Training=21.6124, Diff=+11.8106 (p=0.0000)
5. **teams_this_season_week_4**: Test=9.8018, Training=21.6124, Diff=+11.8106 (p=0.0000)
6. **teams_season_today_week_3**: Test=10.7313, Training=20.9572, Diff=+10.2259 (p=0.0000)
7. **teams_this_season_week_3**: Test=10.7313, Training=20.9572, Diff=+10.2259 (p=0.0000)
8. **teams_season_today_week_2**: Test=10.1189, Training=20.2105, Diff=+10.0916 (p=0.0000)
9. **teams_this_season_week_2**: Test=10.1189, Training=20.2105, Diff=+10.0916 (p=0.0000)
10. **teams_season_today_week_1**: Test=12.1454, Training=19.5894, Diff=+7.4440 (p=0.0000)

## Recommendations

1. **Review top different features** to understand what signals are missing
2. **Investigate low-probability injury cases** in detail
3. **Consider adding features** that capture the differences identified
4. **Review feature engineering** for features with large differences

## Files Generated

- `comparison_low_vs_high_probability.csv`: Detailed feature comparison
- `comparison_low_probability_vs_all_training.csv`: Baseline comparison
- `low_probability_injuries_detailed.csv`: All low-probability injury cases
- `high_probability_injuries_detailed.csv`: All high-probability injury cases