#!/usr/bin/env python3
"""Check timeline generation results"""
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd

df_train = pd.read_csv('timelines_35day_enhanced_balanced_v4_muscular_train.csv')
df_val = pd.read_csv('timelines_35day_enhanced_balanced_v4_muscular_val.csv')
df_test = pd.read_csv('timelines_35day_enhanced_balanced_v4_muscular_test.csv')

print("="*80)
print("TIMELINE GENERATION RESULTS - TEST RUN (5 PLAYERS)")
print("="*80)

print("\n📊 DATASET SUMMARY:")
print(f"  Train: {len(df_train):,} records, {df_train['target'].mean():.1%} injury ratio")
print(f"  Val: {len(df_val):,} records, {df_val['target'].mean():.1%} injury ratio")
print(f"  Test: {len(df_test):,} records, {df_test['target'].mean():.1%} injury ratio")
print(f"  Total features: {len(df_train.columns)}")

print("\n🆕 NEW ENHANCED FEATURES VERIFICATION:")
new_features = [col for col in df_train.columns if any(x in col for x in [
    'matches_this_season_to_last_ratio', 'matches_to_avg_season_ratio',
    'goals_per_match_to_career_ratio', 'assists_per_match_to_career_ratio',
    'minutes_per_match_to_career_ratio', 'recent_to_career_injury_frequency_ratio',
    'minutes_trend_slope', 'matches_trend_slope', 'goals_trend_slope',
    'minutes_3week_avg', 'minutes_5week_avg', 'minutes_acceleration',
    'minutes_volatility', 'matches_volatility',
    'acute_chronic_minutes_ratio', 'acute_chronic_matches_ratio',
    'workload_spike_indicator', 'total_5week_minutes', 'total_5week_matches',
    'avg_weekly_minutes', 'avg_weekly_matches',
    'recovery_ratio', 'recovery_status', 'days_since_match_to_avg_ratio',
    'goals_per_match_ratio', 'assists_per_match_ratio', 'minutes_per_match_ratio',
    'age_x_5week_minutes', 'career_matches_x_recent_goals',
    'injury_history_x_recent_workload', 'position_x_workload',
    'club_goals_per_season', 'club_assists_per_season', 'club_minutes_per_season',
    'national_team_apps_per_year', 'national_team_minutes_per_year'
])]

print(f"  Found {len(new_features)} new enhanced features:")
for f in sorted(new_features):
    print(f"    ✓ {f}")

print("\n✅ Timeline generation test completed successfully!")
print("="*80)

