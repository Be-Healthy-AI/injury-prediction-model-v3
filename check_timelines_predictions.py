#!/usr/bin/env python3
"""Quick check of timelines and predictions status for Arsenal FC (V4 challenger)."""
import pandas as pd
from pathlib import Path

base_path = Path(r"C:\Users\joao.henriques\IPM V3\production\deployments\England\challenger\Arsenal FC")

# Check timelines
timelines_file = base_path / "timelines" / "timelines_35day_season_2025_2026_v4_muscular.csv"
if timelines_file.exists():
    df = pd.read_csv(timelines_file, usecols=['reference_date', 'player_id'], low_memory=False)
    print("=== TIMELINES ===")
    print(f"  File: {timelines_file.name}")
    print(f"  Total rows: {len(df)}")
    print(f"  Unique players: {df['player_id'].nunique()}")
    print(f"  Min date: {df['reference_date'].min()}")
    print(f"  Max date: {df['reference_date'].max()}")
else:
    print(f"Timelines file not found: {timelines_file}")

print()

# Check predictions (find any predictions file)
predictions_dir = base_path / "predictions"
if predictions_dir.exists():
    predictions_files = list(predictions_dir.glob("predictions_lgbm_v4_*.csv"))
    if predictions_files:
        latest_file = sorted(predictions_files)[-1]
        df = pd.read_csv(latest_file, low_memory=False)
        print("=== PREDICTIONS ===")
        print(f"  File: {latest_file.name}")
        print(f"  Total rows: {len(df)}")
        print(f"  Unique players: {df['player_id'].nunique()}")
        print(f"  Min date: {df['reference_date'].min()}")
        print(f"  Max date: {df['reference_date'].max()}")
    else:
        print("No predictions files found in:", predictions_dir)
else:
    print(f"Predictions directory not found: {predictions_dir}")

print()

# Check daily features
daily_features_dir = base_path / "daily_features"
if daily_features_dir.exists():
    df_files = list(daily_features_dir.glob("player_*_daily_features.csv"))
    print("=== DAILY FEATURES ===")
    print(f"  Directory: {daily_features_dir}")
    print(f"  Number of player files: {len(df_files)}")
    
    # Check one file for date range
    if df_files:
        sample_file = df_files[0]
        df = pd.read_csv(sample_file, usecols=['date'], low_memory=False)
        print(f"  Sample file: {sample_file.name}")
        print(f"  Min date: {df['date'].min()}")
        print(f"  Max date: {df['date'].max()}")
else:
    print(f"Daily features directory not found: {daily_features_dir}")
