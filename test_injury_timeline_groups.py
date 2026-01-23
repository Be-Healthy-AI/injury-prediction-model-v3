#!/usr/bin/env python3
"""
Test script to verify that injury timelines come in groups of 5
and all reference dates are within PL periods.

Updated for V4 with dual targets (target1 = muscular, target2 = skeletal).
"""

import sys
import io

if sys.platform == "win32":
    # Only wrap if not already wrapped and if buffer exists
    if not isinstance(sys.stdout, io.TextIOWrapper) and hasattr(sys.stdout, 'buffer'):
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    if not isinstance(sys.stderr, io.TextIOWrapper) and hasattr(sys.stderr, 'buffer'):
        try:
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import PL filtering functions from V4
from models_production.lgbm_muscular_v4.code.timelines.create_35day_timelines_v4_enhanced import (
    build_pl_clubs_per_season,
    build_player_pl_membership_periods,
    is_player_in_pl_on_date,
)

def find_injury_timeline_group(df, player_id, reference_date, target_column='target1'):
    """
    Find all 5 timelines for the same injury.
    
    For a given reference_date R, the injury date could be:
    - R + 1 (if this is D-1 timeline)
    - R + 2 (if this is D-2 timeline)
    - R + 3 (if this is D-3 timeline)
    - R + 4 (if this is D-4 timeline)
    - R + 5 (if this is D-5 timeline)
    
    We need to find which injury date this belongs to, then find all 5 timelines.
    
    Args:
        df: DataFrame with timelines
        player_id: Player ID
        reference_date: Reference date of the timeline
        target_column: 'target1' (muscular) or 'target2' (skeletal)
    """
    # Try each possible injury date (R+1 through R+5)
    for days_after in range(1, 6):
        injury_date = reference_date + timedelta(days=days_after)
        
        # Calculate what the 5 reference dates should be
        expected_ref_dates = [
            injury_date - timedelta(days=d) for d in range(1, 6)
        ]
        
        # Find timelines for this player with these reference dates
        matching_timelines = df[
            (df['player_id'] == player_id) &
            (df['reference_date'].isin(expected_ref_dates)) &
            (df[target_column] == 1)
        ].copy()
        
        if len(matching_timelines) == 5:
            # Found the complete group!
            matching_timelines = matching_timelines.sort_values('reference_date')
            matching_timelines['days_before_injury'] = [
                (injury_date - ref_date).days for ref_date in matching_timelines['reference_date']
            ]
            return matching_timelines, injury_date
    
    return None, None

def test_random_injury_timelines(num_tests_per_target=3):
    """
    Test random injury timelines to verify they come in groups of 5 and are within PL periods.
    Tests both target1 (muscular) and target2 (skeletal) injuries.
    """
    
    print("=" * 80)
    print("TESTING INJURY TIMELINE GROUPS AND PL MEMBERSHIP (V4 - Dual Targets)")
    print("=" * 80)
    
    # Paths - V4
    v4_root = PROJECT_ROOT / "models_production" / "lgbm_muscular_v4"
    timelines_train_dir = v4_root / "data" / "timelines" / "train"
    timelines_test_dir = v4_root / "data" / "timelines" / "test"
    
    # Find all timeline files (train and test)
    train_files = list(timelines_train_dir.glob("timelines_35day_season_*_v4_muscular_train.csv"))
    test_files = list(timelines_test_dir.glob("timelines_35day_season_*_v4_muscular_test.csv"))
    all_files = train_files + test_files
    
    if not all_files:
        print(f"ERROR: No timeline files found in {timelines_train_dir} or {timelines_test_dir}")
        return
    
    print(f"\n📂 Found {len(all_files)} timeline files ({len(train_files)} train, {len(test_files)} test)")
    
    # Load PL membership periods from V4 raw data
    print("\n📂 Loading PL membership periods from V4 raw data...")
    raw_match_dir = v4_root / "data" / "raw_data" / "match_data"
    career_file = v4_root / "data" / "raw_data" / "players_career.csv"
    
    if not raw_match_dir.exists():
        print(f"ERROR: Raw match data directory not found: {raw_match_dir}")
        return
    
    if not career_file.exists():
        print(f"ERROR: Career file not found: {career_file}")
        return
    
    pl_clubs_by_season = build_pl_clubs_per_season(str(raw_match_dir))
    player_pl_periods = build_player_pl_membership_periods(str(career_file), pl_clubs_by_season)
    print(f"✅ Loaded PL periods for {len(player_pl_periods)} players")
    
    # Load all positive timelines from all files (both target1 and target2)
    print("\n📂 Loading positive timelines from all files...")
    all_positives_target1 = []
    all_positives_target2 = []
    
    for file_path in all_files:
        print(f"   Loading {file_path.name}...")
        try:
            df = pd.read_csv(file_path, encoding='utf-8-sig', low_memory=False, nrows=None)
            df['reference_date'] = pd.to_datetime(df['reference_date'], errors='coerce')
            
            # Check for target1, target2, or legacy target column
            has_target1 = 'target1' in df.columns
            has_target2 = 'target2' in df.columns
            has_legacy_target = 'target' in df.columns
            
            if not has_target1 and not has_target2 and not has_legacy_target:
                print(f"      ⚠️  Warning: No target columns found in {file_path.name}")
                continue
            
            # Get positive timelines for each target
            if has_target1:
                positives_t1 = df[df['target1'] == 1].copy()
                positives_t1['source_file'] = file_path.name
                positives_t1['target_type'] = 'target1 (muscular)'
                all_positives_target1.append(positives_t1)
                print(f"      Found {len(positives_t1):,} target1 (muscular) positive timelines")
            elif has_legacy_target:
                # Legacy format: treat 'target' as target1 (muscular) for backward compatibility
                positives_t1 = df[df['target'] == 1].copy()
                positives_t1['source_file'] = file_path.name
                positives_t1['target_type'] = 'target (legacy - treated as muscular)'
                all_positives_target1.append(positives_t1)
                print(f"      Found {len(positives_t1):,} target (legacy) positive timelines (treated as muscular)")
            
            if has_target2:
                positives_t2 = df[df['target2'] == 1].copy()
                positives_t2['source_file'] = file_path.name
                positives_t2['target_type'] = 'target2 (skeletal)'
                all_positives_target2.append(positives_t2)
                print(f"      Found {len(positives_t2):,} target2 (skeletal) positive timelines")
        except Exception as e:
            print(f"      ❌ Error loading {file_path.name}: {e}")
            continue
    
    # Combine timelines for each target
    combined_target1 = pd.concat(all_positives_target1, ignore_index=True) if all_positives_target1 else pd.DataFrame()
    combined_target2 = pd.concat(all_positives_target2, ignore_index=True) if all_positives_target2 else pd.DataFrame()
    
    print(f"\n✅ Total positive timelines loaded:")
    print(f"   Target1 (muscular): {len(combined_target1):,}")
    print(f"   Target2 (skeletal): {len(combined_target2):,}")
    
    if len(combined_target1) == 0 and len(combined_target2) == 0:
        print("ERROR: No positive timelines found for either target")
        return
    
    # Test target1 (muscular) injuries
    if len(combined_target1) > 0:
        print("\n" + "=" * 80)
        print("TESTING TARGET1 (MUSCULAR) INJURIES")
        print("=" * 80)
        test_target_timelines(combined_target1, 'target1', 'muscular', num_tests_per_target, player_pl_periods)
    
    # Test target2 (skeletal) injuries
    if len(combined_target2) > 0:
        print("\n" + "=" * 80)
        print("TESTING TARGET2 (SKELETAL) INJURIES")
        print("=" * 80)
        test_target_timelines(combined_target2, 'target2', 'skeletal', num_tests_per_target, player_pl_periods)

def test_target_timelines(combined_df, target_column, target_name, num_tests, player_pl_periods):
    """Test timelines for a specific target type."""
    
    # Pick random positive timelines to test
    print(f"\n🎲 Picking {num_tests} random positive timelines to test...")
    np.random.seed(42)
    test_indices = np.random.choice(len(combined_df), size=min(num_tests, len(combined_df)), replace=False)
    
    for test_num, idx in enumerate(test_indices, 1):
        print("\n" + "=" * 80)
        print(f"TEST {test_num} - {target_name.upper()} INJURY")
        print("=" * 80)
        
        test_timeline = combined_df.iloc[idx]
        player_id = test_timeline['player_id']
        reference_date = test_timeline['reference_date']
        
        print(f"\n📋 Selected Timeline:")
        print(f"   Player ID: {player_id}")
        print(f"   Player Name: {test_timeline.get('player_name', 'N/A')}")
        print(f"   Reference Date: {reference_date.strftime('%Y-%m-%d')}")
        print(f"   Source File: {test_timeline['source_file']}")
        print(f"   Target Type: {target_name} ({target_column})")
        # Use 'target' if target_column doesn't exist (legacy format)
        actual_target_col = target_column if target_column in test_timeline.index else 'target'
        print(f"   {actual_target_col}: {test_timeline[actual_target_col]}")
        
        # Find the injury timeline group
        print(f"\n🔍 Finding injury timeline group...")
        # Use 'target' if target_column doesn't exist (legacy format)
        actual_target_col = target_column if target_column in combined_df.columns else 'target'
        timeline_group, injury_date = find_injury_timeline_group(combined_df, player_id, reference_date, actual_target_col)
        
        if timeline_group is None:
            print(f"   ❌ ERROR: Could not find complete group of 5 timelines!")
            print(f"   This timeline might be missing its siblings.")
            continue
        
        print(f"   ✅ Found complete group of {len(timeline_group)} timelines")
        print(f"   Injury Date: {injury_date.strftime('%Y-%m-%d')}")
        
        # Display all 5 timelines
        print(f"\n📊 All 5 Timelines for This {target_name.upper()} Injury:")
        print(f"{'Days Before':<12} {'Reference Date':<15} {'In PL?':<8} {'Season':<12} {'Source File':<40}")
        print("-" * 100)
        
        all_in_pl = True
        for _, timeline in timeline_group.iterrows():
            ref_date = timeline['reference_date']
            days_before = timeline['days_before_injury']
            source_file = timeline['source_file']
            
            # Determine season from reference date
            season_year = ref_date.year if ref_date.month >= 7 else ref_date.year - 1
            season_str = f"{season_year}/{season_year+1}"
            
            # Check if reference date is in PL period
            in_pl = is_player_in_pl_on_date(player_id, ref_date, player_pl_periods)
            if not in_pl:
                all_in_pl = False
            
            status = "✅ YES" if in_pl else "❌ NO"
            print(f"D-{days_before:<10} {ref_date.strftime('%Y-%m-%d'):<15} {status:<8} {season_str:<12} {source_file}")
        
        # Summary
        print(f"\n📋 Summary:")
        if all_in_pl:
            print(f"   ✅ ALL 5 reference dates are within PL periods")
        else:
            print(f"   ❌ WARNING: Some reference dates are NOT within PL periods!")
            print(f"   This should not happen in V4 filtered data!")
        
        # Show PL periods for this player
        if player_id in player_pl_periods:
            periods = player_pl_periods[player_id]
            print(f"\n📅 Player PL Membership Periods:")
            for i, (start, end) in enumerate(periods, 1):
                # Determine season for each period
                start_season = start.year if start.month >= 7 else start.year - 1
                end_season = end.year if end.month >= 7 else end.year - 1
                print(f"   Period {i}: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')} (seasons {start_season}/{start_season+1} to {end_season}/{end_season+1})")
        else:
            print(f"\n   ⚠️  No PL periods found for this player!")
            print(f"   This should not happen if the timeline passed PL filtering!")

if __name__ == "__main__":
    test_random_injury_timelines(num_tests_per_target=3)

