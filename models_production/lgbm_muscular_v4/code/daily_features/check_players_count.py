#!/usr/bin/env python3
"""Check player counts in players_profile.csv"""

import pandas as pd
from pathlib import Path

# Find the players_profile.csv file
script_dir = Path(__file__).resolve().parent
data_dir = script_dir.parent.parent / 'data' / 'raw_data'
players_path = data_dir / 'players_profile.csv'

print(f"Loading players from: {players_path}")
df = pd.read_csv(players_path, sep=';', encoding='utf-8')

print(f"\nTotal players in players_profile.csv: {len(df)}")
print(f"Columns: {list(df.columns)}")

if 'position' in df.columns:
    # Check for goalkeepers
    gk_mask = df['position'].astype(str).str.lower().str.contains('goalkeeper|gk|keeper', na=False)
    gk_count = gk_mask.sum()
    non_gk_count = (~gk_mask).sum()
    
    print(f"\nGoalkeepers: {gk_count}")
    print(f"Non-goalkeepers: {non_gk_count}")
    print(f"\nPosition distribution (top 15):")
    print(df['position'].value_counts().head(15))
    
    # Check for missing positions
    missing_pos = df['position'].isna().sum()
    if missing_pos > 0:
        print(f"\n⚠️  Players with missing position: {missing_pos}")
else:
    print("\n⚠️  'position' column not found!")
