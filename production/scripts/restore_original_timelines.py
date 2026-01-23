#!/usr/bin/env python3
"""
Restore original timelines from test dataset (up to 2025-12-05) for Chelsea players
and prepare for regeneration from 2025-12-06 onwards.
"""

import json
import pandas as pd
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent
ROOT_DIR = PRODUCTION_ROOT.parent

# Input: Original test timelines
ORIGINAL_TIMELINES = ROOT_DIR / "models_production" / "lgbm_muscular_v1" / "data" / "timelines" / "test" / "timelines_35day_season_2025_2026_v4_muscular.csv"

# Output: Production timelines folder
OUTPUT_DIR = PRODUCTION_ROOT / "deployments" / "England" / "Chelsea FC" / "timelines"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "timelines_35day_season_2025_2026_v4_muscular.csv"

# Load Chelsea player IDs from config
CONFIG_PATH = PRODUCTION_ROOT / "deployments" / "England" / "Chelsea FC" / "config.json"
CUTOFF_DATE = "2025-12-05"

def main():
    print("=" * 80)
    print("RESTORING ORIGINAL TIMELINES FOR CHELSEA PLAYERS")
    print("=" * 80)
    print(f"Original file: {ORIGINAL_TIMELINES}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Cutoff date: {CUTOFF_DATE}")
    print("=" * 80)
    print()
    
    # Load Chelsea player IDs
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
    chelsea_player_ids = set(config['player_ids'])
    print(f"Loaded {len(chelsea_player_ids)} Chelsea player IDs")
    print()
    
    # Load original timelines
    if not ORIGINAL_TIMELINES.exists():
        raise FileNotFoundError(f"Original timelines file not found: {ORIGINAL_TIMELINES}")
    
    print(f"Loading original timelines from: {ORIGINAL_TIMELINES}")
    df = pd.read_csv(ORIGINAL_TIMELINES, low_memory=False)
    print(f"Total timelines in original file: {len(df):,}")
    print()
    
    # Check for required columns
    if 'player_id' not in df.columns:
        raise ValueError("'player_id' column not found in timelines file")
    
    # Determine the reference_date column name
    date_col = None
    for col in ['reference_date', 'date', 'Date', 'Reference_Date']:
        if col in df.columns:
            date_col = col
            break
    
    if date_col is None:
        raise ValueError("Could not find reference_date column in timelines file")
    
    print(f"Using date column: {date_col}")
    
    # Convert date column to datetime
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    cutoff_ts = pd.to_datetime(CUTOFF_DATE).normalize()
    
    # Filter to Chelsea players
    print(f"Filtering to Chelsea players...")
    chelsea_df = df[df['player_id'].isin(chelsea_player_ids)].copy()
    print(f"Chelsea timelines found: {len(chelsea_df):,}")
    print()
    
    # Filter to keep only rows up to cutoff_date
    print(f"Filtering to keep rows with {date_col} <= {CUTOFF_DATE}...")
    before_filter = len(chelsea_df)
    chelsea_df = chelsea_df[chelsea_df[date_col] <= cutoff_ts].copy()
    after_filter = len(chelsea_df)
    print(f"Timelines after date filter: {after_filter:,} (removed {before_filter - after_filter:,} rows)")
    print()
    
    # Check date range
    if len(chelsea_df) > 0:
        min_date = chelsea_df[date_col].min().date()
        max_date = chelsea_df[date_col].max().date()
        print(f"Date range in filtered data: {min_date} to {max_date}")
        print(f"Timelines on {CUTOFF_DATE}: {len(chelsea_df[chelsea_df[date_col] == cutoff_ts]):,}")
    else:
        print("WARNING: No timelines found after filtering!")
        return
    
    # Ensure reference_date is in date-only format (YYYY-MM-DD) before saving
    if date_col in chelsea_df.columns:
        chelsea_df[date_col] = pd.to_datetime(chelsea_df[date_col], errors='coerce').dt.date.astype(str)
    
    # Save to output folder
    print()
    print(f"Saving to: {OUTPUT_FILE}")
    chelsea_df.to_csv(OUTPUT_FILE, index=False)
    print(f"[OK] Saved {len(chelsea_df):,} Chelsea timelines to {OUTPUT_FILE}")
    print()
    print("=" * 80)
    print("RESTORATION COMPLETE")
    print("=" * 80)
    print(f"Next step: Run update_timelines.py with --regenerate-from-date 2025-12-05")
    print("=" * 80)

if __name__ == "__main__":
    main()

