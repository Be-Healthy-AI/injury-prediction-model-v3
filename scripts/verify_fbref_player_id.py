#!/usr/bin/env python3
"""Verify fbref_player_id is populated correctly."""

import pandas as pd

df = pd.read_csv('data_exports/fbref/test_direct/match_stats/player_dc7f8a28_matches.csv')

print("=" * 70)
print("fbref_player_id Verification")
print("=" * 70)
print(f"Total rows: {len(df)}")
print(f"Unique fbref_player_id values: {df['fbref_player_id'].unique()}")
print(f"Non-null count: {df['fbref_player_id'].notna().sum()} out of {len(df)}")
print(f"All rows have 'dc7f8a28': {(df['fbref_player_id'] == 'dc7f8a28').all()}")
print(f"\nFirst 5 rows:")
print(df[['fbref_player_id', 'match_date', 'team', 'opponent']].head(5).to_string())
print(f"\nLast 5 rows:")
print(df[['fbref_player_id', 'match_date', 'team', 'opponent']].tail(5).to_string())
print("=" * 70)









