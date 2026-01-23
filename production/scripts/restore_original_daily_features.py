#!/usr/bin/env python3
"""
Restore original daily features files (up to 2025-12-05) and prepare for regeneration.
This ensures aggregate injury features (avg_injury_severity, max_injury_severity) are calculated
correctly using the original severity values for historical data.
"""

import json
import pandas as pd
from pathlib import Path
import sys

# Calculate paths
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent
ROOT_DIR = PRODUCTION_ROOT.parent

def main():
    # Paths
    ORIGINAL_DIR = ROOT_DIR / "daily_features_output"
    TARGET_DIR = PRODUCTION_ROOT / "deployments" / "England" / "Chelsea FC" / "daily_features"
    CONFIG_PATH = PRODUCTION_ROOT / "deployments" / "England" / "Chelsea FC" / "config.json"
    CUTOFF_DATE = "2025-12-05"
    
    # Load player IDs
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    player_ids = config['player_ids']
    
    cutoff_ts = pd.to_datetime(CUTOFF_DATE).normalize()
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("RESTORING ORIGINAL DAILY FEATURES FILES")
    print("=" * 80)
    print(f"Original directory: {ORIGINAL_DIR}")
    print(f"Target directory: {TARGET_DIR}")
    print(f"Cutoff date: {CUTOFF_DATE}")
    print(f"Player IDs: {len(player_ids)} players")
    print("=" * 80)
    print()
    
    restored_count = 0
    skipped_count = 0
    error_count = 0
    
    for player_id in player_ids:
        try:
            orig_file = ORIGINAL_DIR / f"player_{player_id}_daily_features.csv"
            if not orig_file.exists():
                print(f"  [{player_id}] SKIP: File not found in original directory")
                skipped_count += 1
                continue
            
            # Read original file
            df = pd.read_csv(orig_file, parse_dates=['date'])
            
            # Filter to keep only rows up to cutoff_date
            df_filtered = df[df['date'] <= cutoff_ts].copy()
            
            if df_filtered.empty:
                print(f"  [{player_id}] SKIP: No rows before cutoff date")
                skipped_count += 1
                continue
            
            # Save to target directory
            target_file = TARGET_DIR / f"player_{player_id}_daily_features.csv"
            df_filtered.to_csv(target_file, index=False)
            
            min_date = df_filtered['date'].min().date()
            max_date = df_filtered['date'].max().date()
            print(f"  [{player_id}] OK: Restored {len(df_filtered)} rows ({min_date} to {max_date})")
            restored_count += 1
            
        except Exception as e:
            print(f"  [{player_id}] ERROR: {e}")
            error_count += 1
    
    print()
    print("=" * 80)
    print("RESTORATION COMPLETE")
    print("=" * 80)
    print(f"  Restored: {restored_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Errors: {error_count}")
    print("=" * 80)

if __name__ == "__main__":
    main()








