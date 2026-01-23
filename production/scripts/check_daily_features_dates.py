#!/usr/bin/env python3
"""
Check the latest date in daily features files for a sample club.
"""

import pandas as pd
from pathlib import Path
import json

SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent

# Check Arsenal FC
club = "Arsenal FC"
config_path = PRODUCTION_ROOT / "deployments" / "England" / club / "config.json"
daily_features_dir = PRODUCTION_ROOT / "deployments" / "England" / club / "daily_features"

with open(config_path, 'r') as f:
    config = json.load(f)
    player_ids = config['player_ids']

print(f"Checking {club}...")
print(f"Total players: {len(player_ids)}")

# Check first 5 players
sample_players = player_ids[:5]
for player_id in sample_players:
    file_path = daily_features_dir / f"player_{player_id}_daily_features.csv"
    if file_path.exists():
        df = pd.read_csv(file_path, usecols=['date'], low_memory=False)
        df['date'] = pd.to_datetime(df['date'])
        max_date = df['date'].max()
        print(f"  Player {player_id}: {len(df)} rows, latest date: {max_date.date()}")
    else:
        print(f"  Player {player_id}: File not found")







