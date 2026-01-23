#!/usr/bin/env python3
"""Show summary of fetched FBRef data."""

import pandas as pd
from pathlib import Path

file_path = Path("data_exports/fbref/test_direct/match_stats/player_dc7f8a28_matches.csv")

if not file_path.exists():
    print(f"File not found: {file_path}")
    exit(1)

df = pd.read_csv(file_path, parse_dates=['match_date'])

print("=" * 70)
print("FBRef Data Summary - Cole Palmer")
print("=" * 70)
print(f"Total matches: {len(df)}")
print(f"Date range: {df['match_date'].min().date()} to {df['match_date'].max().date()}")
print(f"\nSeasons covered: {sorted(df['season'].dropna().unique())}")
print(f"\nTop Competitions:")
print(df['competition'].value_counts().head(10))
print(f"\nSample match (most recent):")
recent = df.sort_values('match_date', ascending=False).iloc[0]
print(f"  Date: {recent['match_date'].date()}")
print(f"  Team: {recent['team']} vs {recent['opponent']}")
print(f"  Goals: {recent.get('goals', 'N/A')}")
print(f"  Minutes: {recent.get('minutes', 'N/A')}")
print(f"  Passes: {recent.get('passes_attempted', 'N/A')}")
print(f"  Shots: {recent.get('shots', 'N/A')}")
print(f"\nColumns with data ({len([c for c in df.columns if df[c].notna().sum() > 0])}):")
cols_with_data = [c for c in df.columns if df[c].notna().sum() > 0]
for i, col in enumerate(cols_with_data[:20], 1):
    non_null = df[col].notna().sum()
    print(f"  {i:2d}. {col:30s} ({non_null:3d} non-null)")
if len(cols_with_data) > 20:
    print(f"  ... and {len(cols_with_data) - 20} more")
print("=" * 70)









