#!/usr/bin/env python3
"""
Complete missing timeline rows in the period 2025-07-01 to 2025-11-08
by generating timelines for dates that exist in daily features but are missing in timelines.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Set, List, Tuple
import sys
import os

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent
ROOT_DIR = PRODUCTION_ROOT.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# Import timeline generation functions
from production.scripts.update_timelines import (
    get_all_valid_dates_for_timelines,
    generate_timelines_for_dates,
    build_timeline,
    create_windowed_features_vectorized,
    get_season_date_range,
    load_player_names_mapping
)

# Paths
TIMELINES_FILE = PRODUCTION_ROOT / "deployments" / "England" / "Chelsea FC" / "timelines" / "timelines_35day_season_2025_2026_v4_muscular.csv"
DAILY_FEATURES_DIR = PRODUCTION_ROOT / "deployments" / "England" / "Chelsea FC" / "daily_features"
CONFIG_PATH = PRODUCTION_ROOT / "deployments" / "England" / "Chelsea FC" / "config.json"

# Date range to complete
START_DATE = "2025-07-01"
END_DATE = "2025-11-08"

def get_existing_timeline_dates(timelines_df: pd.DataFrame, player_id: int) -> Set[pd.Timestamp]:
    """Get set of reference dates that already have timelines for a player."""
    player_timelines = timelines_df[timelines_df['player_id'] == player_id]
    if len(player_timelines) == 0:
        return set()
    
    dates = pd.to_datetime(player_timelines['reference_date'], errors='coerce').dropna()
    return set(dates.dt.normalize())

def get_available_daily_feature_dates(daily_features_file: Path, 
                                     start_date: pd.Timestamp, 
                                     end_date: pd.Timestamp) -> Set[pd.Timestamp]:
    """Get set of dates available in daily features file within the date range."""
    df = pd.read_csv(daily_features_file, parse_dates=['date'])
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    return set(df['date'].dt.normalize())

def main():
    print("=" * 80)
    print("COMPLETING MISSING TIMELINES (2025-07-01 to 2025-11-08)")
    print("=" * 80)
    print()
    
    # Load config
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    player_ids = config['player_ids']
    
    # Load existing timelines
    print(f"Loading existing timelines from: {TIMELINES_FILE}")
    timelines_df = pd.read_csv(TIMELINES_FILE, low_memory=False)
    timelines_df['reference_date'] = pd.to_datetime(timelines_df['reference_date'], errors='coerce')
    print(f"Total existing timelines: {len(timelines_df):,}")
    print()
    
    # Date range
    start_ts = pd.to_datetime(START_DATE).normalize()
    end_ts = pd.to_datetime(END_DATE).normalize()
    season_start, season_end = get_season_date_range(2025)
    
    # Load player names
    player_names_map = load_player_names_mapping()
    
    # Collect missing timelines
    all_new_timelines = []
    total_missing = 0
    
    print(f"Processing {len(player_ids)} players...")
    print()
    
    for player_id in player_ids:
        try:
            # Get existing timeline dates for this player
            existing_dates = get_existing_timeline_dates(timelines_df, player_id)
            
            # Load daily features
            daily_features_file = DAILY_FEATURES_DIR / f"player_{player_id}_daily_features.csv"
            if not daily_features_file.exists():
                print(f"  [{player_id}] SKIP: Daily features file not found")
                continue
            
            # Get available dates in daily features
            available_dates = get_available_daily_feature_dates(
                daily_features_file, start_ts, end_ts
            )
            
            # Find missing dates
            missing_dates = available_dates - existing_dates
            
            if len(missing_dates) == 0:
                print(f"  [{player_id}] OK: No missing dates")
                continue
            
            print(f"  [{player_id}] Found {len(missing_dates)} missing dates")
            total_missing += len(missing_dates)
            
            # Load daily features for timeline generation
            df = pd.read_csv(daily_features_file, parse_dates=['date'])
            df = df[(df['date'] >= start_ts - timedelta(days=34)) & (df['date'] <= end_ts)]
            
            if len(df) == 0:
                print(f"  [{player_id}] SKIP: No daily features in date range")
                continue
            
            # Get player name
            player_name = player_names_map.get(player_id, f"Player_{player_id}")
            
            # Generate timelines for missing dates
            # First, determine targets (check for injuries in next 35 days)
            missing_dates_with_targets = []
            for ref_date in sorted(missing_dates):
                # Check if there's an injury in the next 35 days
                future_end = ref_date + timedelta(days=34)
                future_mask = (df['date'] >= ref_date) & (df['date'] <= future_end)
                future_df = df[future_mask]
                
                # Check if cum_inj_starts increases in this window
                if len(future_df) > 0 and 'cum_inj_starts' in future_df.columns:
                    start_cum = future_df.iloc[0]['cum_inj_starts'] if pd.notna(future_df.iloc[0]['cum_inj_starts']) else 0
                    end_cum = future_df.iloc[-1]['cum_inj_starts'] if pd.notna(future_df.iloc[-1]['cum_inj_starts']) else 0
                    target = 1 if end_cum > start_cum else 0
                else:
                    target = 0
                
                missing_dates_with_targets.append((ref_date, target))
            
            # Generate timelines
            new_timelines = generate_timelines_for_dates(
                player_id, player_name, df, missing_dates_with_targets
            )
            
            all_new_timelines.extend(new_timelines)
            print(f"  [{player_id}] Generated {len(new_timelines)} timelines")
            
        except Exception as e:
            print(f"  [{player_id}] ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print()
    print(f"Total missing dates found: {total_missing}")
    print(f"Total new timelines generated: {len(all_new_timelines)}")
    print()
    
    if len(all_new_timelines) > 0:
        # Convert to DataFrame and merge with existing
        print("Merging new timelines with existing...")
        new_timelines_df = pd.DataFrame(all_new_timelines)
        
        # Combine with existing
        combined = pd.concat([timelines_df, new_timelines_df], ignore_index=True)
        
        # Deduplicate (keep existing if duplicate)
        if 'player_id' in combined.columns and 'reference_date' in combined.columns:
            combined = combined.sort_values(['player_id', 'reference_date'])
            before_dedup = len(combined)
            # Keep first (existing) if duplicate
            combined = combined.drop_duplicates(
                subset=['player_id', 'reference_date'], 
                keep='first'
            )
            after_dedup = len(combined)
            if before_dedup != after_dedup:
                print(f"Removed {before_dedup - after_dedup} duplicate rows (kept existing)")
        
        # Save
        combined.to_csv(TIMELINES_FILE, index=False)
        print(f"Saved {len(combined):,} total timelines to {TIMELINES_FILE}")
        
        # Verify
        date_range = pd.to_datetime(combined['reference_date'], errors='coerce').dropna()
        if len(date_range) > 0:
            print(f"Date range: {date_range.min().date()} to {date_range.max().date()}")
    else:
        print("No new timelines to add.")
    
    print()
    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()








