#!/usr/bin/env python3
"""Quick verification script to check log transformation was applied."""

import sys
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pandas as pd
from pathlib import Path

# Add script directory to path
SCRIPT_DIR = Path(__file__).resolve().parent
V4_ROOT = SCRIPT_DIR.parent.parent
DAILY_FEATURES_DIR = V4_ROOT / "data" / "daily_features"

# Find all generated files
files = list(DAILY_FEATURES_DIR.glob("player_*_daily_features.csv"))

if not files:
    print("No daily features files found!")
    sys.exit(1)

print(f"Found {len(files)} daily features files\n")

# Check a few files
for file_path in files[:3]:
    player_id = file_path.stem.replace("player_", "").replace("_daily_features", "")
    df = pd.read_csv(file_path, index_col=0)
    
    print(f"Player {player_id}:")
    print(f"  Total rows: {len(df)}")
    
    # Check injury recency features
    injury_features = [
        'days_since_last_injury',
        'days_since_last_muscular',
        'days_since_last_upper_body',
        'days_since_last_critical'
    ]
    
    for feat in injury_features:
        if feat in df.columns:
            max_val = df[feat].max()
            non_zero_count = (df[feat] > 0).sum()
            print(f"  {feat}: max={max_val:.2f}, non-zero={non_zero_count}")
            
            # If we have non-zero values, show a sample
            if non_zero_count > 0:
                sample = df[df[feat] > 0][feat].head(3).values
                print(f"    Sample values: {sample}")
    
    # Check club country features
    if 'current_club_country' in df.columns:
        non_null_count = df['current_club_country'].notna().sum()
        if non_null_count > 0:
            sample_countries = df[df['current_club_country'].notna()]['current_club_country'].unique()[:3]
            print(f"  current_club_country: {non_null_count} non-null, samples: {list(sample_countries)}")
    
    print()

print("\n✅ Verification complete!")
print("\nExpected behavior:")
print("  - Injury recency features should have max values around 0-7 (log-transformed)")
print("  - Club country features should be populated (not all empty)")
