#!/usr/bin/env python3
"""
Incremental Daily Features Update Script for Production Predictions

This script processes all players in the provided raw data files and:
- For existing players: Appends new days to existing daily features files
- For new players: Generates daily features from scratch

All outputs are saved to production_predictions/daily_features/
"""

from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.create_daily_features_v3 import (
    load_data_with_cache,
    preprocess_data_optimized,
    preprocess_career_data,
    initialize_team_country_map,
    initialize_competition_type_map,
    generate_daily_features_for_player_enhanced,
    generate_incremental_features_for_player,
)

# Configuration
PRODUCTION_DATA_DIR = ROOT_DIR / "production_predictions" / "raw_data"
PRODUCTION_FEATURES_DIR = ROOT_DIR / "production_predictions" / "daily_features"
CONFIG = {
    'DATA_DIR': str(PRODUCTION_DATA_DIR),
    'CACHE_FILE': 'data_cache_production.pkl',
    'CACHE_DURATION': 3600,
}


def load_production_data() -> Dict[str, pd.DataFrame]:
    """
    Load data files from production_predictions/raw_data/ directory.
    Uses the same caching mechanism as the main pipeline.
    """
    # Temporarily override CONFIG in create_daily_features_v3
    import scripts.create_daily_features_v3 as features_module
    original_data_dir = features_module.CONFIG['DATA_DIR']
    original_cache = features_module.CONFIG['CACHE_FILE']
    
    try:
        features_module.CONFIG['DATA_DIR'] = CONFIG['DATA_DIR']
        features_module.CONFIG['CACHE_FILE'] = CONFIG['CACHE_FILE']
        data = load_data_with_cache()
        return data
    finally:
        # Restore original config
        features_module.CONFIG['DATA_DIR'] = original_data_dir
        features_module.CONFIG['CACHE_FILE'] = original_cache


def get_existing_last_date(player_id: int) -> Optional[pd.Timestamp]:
    """Get the last date from existing daily features file."""
    features_file = PRODUCTION_FEATURES_DIR / f"player_{player_id}_daily_features.csv"
    
    if not features_file.exists():
        return None
    
    try:
        df = pd.read_csv(features_file, parse_dates=['date'], nrows=1)
        if 'date' not in df.columns:
            return None
        
        # Read last row to get last date
        df_full = pd.read_csv(features_file, parse_dates=['date'])
        if df_full.empty:
            return None
        
        last_date = pd.to_datetime(df_full['date'].max())
        return last_date
    except Exception as e:
        print(f"   ⚠️  Error reading existing file for player {player_id}: {e}")
        return None


def get_last_cumulative_state(player_id: int) -> Optional[Dict]:
    """Extract cumulative feature values from the last row of existing file."""
    features_file = PRODUCTION_FEATURES_DIR / f"player_{player_id}_daily_features.csv"
    
    if not features_file.exists():
        return None
    
    try:
        df = pd.read_csv(features_file, parse_dates=['date'])
        if df.empty:
            return None
        
        last_row = df.iloc[-1]
        
        # Extract all cumulative features
        cumulative_features = {}
        cum_prefixes = ['cum_', 'career_', 'club_cum_']
        
        for col in df.columns:
            if any(col.startswith(prefix) for prefix in cum_prefixes):
                cumulative_features[col] = last_row[col]
        
        # Also get non-cumulative state that affects future calculations
        state_features = [
            'current_club', 'current_club_country', 'previous_club', 'previous_club_country',
            'teams_today', 'cum_teams', 'seasons_count'
        ]
        for feat in state_features:
            if feat in df.columns:
                cumulative_features[feat] = last_row[feat]
        
        return cumulative_features
    except Exception as e:
        print(f"   ⚠️  Error extracting cumulative state for player {player_id}: {e}")
        return None


def backup_existing_file(player_id: int) -> bool:
    """Create backup of existing daily features file."""
    features_file = PRODUCTION_FEATURES_DIR / f"player_{player_id}_daily_features.csv"
    
    if not features_file.exists():
        return True
    
    try:
        backup_file = PRODUCTION_FEATURES_DIR / f"player_{player_id}_daily_features.backup.csv"
        shutil.copy2(features_file, backup_file)
        return True
    except Exception as e:
        print(f"   ⚠️  Failed to backup file for player {player_id}: {e}")
        return False


def restore_backup(player_id: int) -> bool:
    """Restore from backup if update failed."""
    backup_file = PRODUCTION_FEATURES_DIR / f"player_{player_id}_daily_features.backup.csv"
    features_file = PRODUCTION_FEATURES_DIR / f"player_{player_id}_daily_features.csv"
    
    if not backup_file.exists():
        return False
    
    try:
        shutil.copy2(backup_file, features_file)
        backup_file.unlink()  # Remove backup after restore
        return True
    except Exception as e:
        print(f"   ⚠️  Failed to restore backup for player {player_id}: {e}")
        return False


def validate_existing_file(player_id: int) -> bool:
    """Validate structure of existing daily features file."""
    features_file = PRODUCTION_FEATURES_DIR / f"player_{player_id}_daily_features.csv"
    
    if not features_file.exists():
        return True  # New file, no validation needed
    
    try:
        df = pd.read_csv(features_file, parse_dates=['date'])
        
        # Check required columns
        required_cols = ['player_id', 'date']
        if not all(col in df.columns for col in required_cols):
            print(f"   ⚠️  Missing required columns in existing file for player {player_id}")
            return False
        
        # Check date continuity (allow small gaps but warn)
        if len(df) > 1:
            df_sorted = df.sort_values('date')
            date_diffs = df_sorted['date'].diff().dt.days
            large_gaps = date_diffs[date_diffs > 2]
            if len(large_gaps) > 0:
                print(f"   ⚠️  Large date gaps detected in existing file for player {player_id}")
        
        # Check for duplicate dates
        if df['date'].duplicated().any():
            print(f"   ⚠️  Duplicate dates found in existing file for player {player_id}")
            return False
        
        return True
    except Exception as e:
        print(f"   ⚠️  Error validating file for player {player_id}: {e}")
        return False


def update_player_features(
    player_id: int,
    player_row: pd.Series,
    player_matches: pd.DataFrame,
    player_physio_injuries: pd.DataFrame,
    player_non_physio_injuries: pd.DataFrame,
    player_career: Optional[pd.DataFrame],
    data: Dict[str, pd.DataFrame],
    force_rebuild: bool = False,
) -> tuple[bool, int, str]:
    """
    Update daily features for a single player.
    
    Returns:
        (success, days_added, status_message)
    """
    features_file = PRODUCTION_FEATURES_DIR / f"player_{player_id}_daily_features.csv"
    is_new_player = not features_file.exists()
    
    # Validate existing file if it exists
    if not is_new_player:
        if not validate_existing_file(player_id):
            return False, 0, "validation_failed"
        
        # Create backup
        if not backup_existing_file(player_id):
            return False, 0, "backup_failed"
    
    try:
        # Determine end date from raw data - cap at Nov 9, 2025 as requested
        target_end_date = pd.Timestamp('2025-11-09').normalize()
        global_end_date_cap = target_end_date
        
        if is_new_player or force_rebuild:
            label = "🆕 New player" if is_new_player else "♻️  Force rebuild"
            print(f"   {label} - generating full history")
            daily_features = generate_daily_features_for_player_enhanced(
                player_id=player_id,
                player_row=player_row,
                player_matches=player_matches,
                player_physio_injuries=player_physio_injuries,
                player_non_physio_injuries=player_non_physio_injuries,
                player_career=player_career,
                global_end_date_cap=global_end_date_cap
            )
            
            if daily_features is None or daily_features.empty:
                return False, 0, "generation_failed"
            
            # Save new file
            daily_features.to_csv(features_file, index=False, encoding='utf-8-sig')
            days_added = len(daily_features)
            status = "new_player" if is_new_player else "rebuilt"
            return True, days_added, status
        
        else:
            # Incremental update
            last_date = get_existing_last_date(player_id)
            if last_date is None:
                return False, 0, "cannot_read_last_date"
            
            # Check if we already have data up to today's date
            target_end_date = pd.Timestamp.today().normalize()
            if last_date >= target_end_date:
                return True, 0, "no_new_data"
            
            print(f"   🔄 Updating from {last_date.date()} onwards")
            
            new_features = generate_incremental_features_for_player(
                player_id=player_id,
                player_row=player_row,
                player_matches=player_matches,
                player_physio_injuries=player_physio_injuries,
                player_non_physio_injuries=player_non_physio_injuries,
                player_career=player_career,
                existing_file_path=features_file,
                global_end_date_cap=global_end_date_cap
            )
            
            if new_features is None or new_features.empty:
                return True, 0, "no_new_days"
            
            # Append to existing file
            existing_df = pd.read_csv(features_file, parse_dates=['date'])
            combined_df = pd.concat([existing_df, new_features], ignore_index=True)
            combined_df = combined_df.sort_values('date').drop_duplicates(subset=['date'], keep='last')
            combined_df.to_csv(features_file, index=False, encoding='utf-8-sig')
            
            days_added = len(new_features)
            
            # Remove backup on success
            backup_file = PRODUCTION_FEATURES_DIR / f"player_{player_id}_daily_features.backup.csv"
            if backup_file.exists():
                backup_file.unlink()
            
            return True, days_added, "updated"
    
    except Exception as e:
        print(f"   ❌ Error updating player {player_id}: {e}")
        # Restore backup on failure
        if not is_new_player:
            restore_backup(player_id)
        return False, 0, f"error: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Directory containing raw data Excel files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory for output daily features files'
    )
    parser.add_argument(
        '--players',
        type=int,
        nargs='*',
        help='Specific player IDs to process (if not provided, processes all players)'
    )
    parser.add_argument(
        '--filter-club',
        type=str,
        default=None,
        help='Filter players by current_club (e.g., "Benfica" or "SL Benfica")'
    )
    parser.add_argument(
        '--force-rebuild',
        action='store_true',
        help='Regenerate daily features from scratch for all processed players'
    )
    args = parser.parse_args()
    
    # Update paths if provided (update globals for use in helper functions)
    global PRODUCTION_DATA_DIR, PRODUCTION_FEATURES_DIR
    if args.data_dir:
        PRODUCTION_DATA_DIR = Path(args.data_dir)
    if args.output_dir:
        PRODUCTION_FEATURES_DIR = Path(args.output_dir)
    PRODUCTION_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Update CONFIG
    CONFIG['DATA_DIR'] = str(PRODUCTION_DATA_DIR)
    
    print("=" * 70)
    print("INCREMENTAL DAILY FEATURES UPDATE FOR PRODUCTION PREDICTIONS")
    print("=" * 70)
    print(f"📂 Data directory: {PRODUCTION_DATA_DIR}")
    print(f"📂 Output directory: {PRODUCTION_FEATURES_DIR}")
    print()
    
    # Load data
    print("📥 Loading raw data files...")
    try:
        data = load_production_data()
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return 1
    
    players = data['players']
    injuries = data['injuries']
    matches = data['matches']
    teams = data.get('teams')
    competitions = data.get('competitions')
    career = data.get('career')
    
    # Initialize mappings
    if teams is not None:
        initialize_team_country_map(teams)
    if competitions is not None:
        initialize_competition_type_map(competitions)
    if career is not None:
        career = preprocess_career_data(career)
    
    # Filter by current_club BEFORE preprocessing (to preserve the column)
    if args.filter_club and not args.players:
        if 'current_club' not in players.columns:
            print(f"⚠️  Warning: 'current_club' column not found in players_profile.xlsx")
            print(f"   Processing all non-goalkeeper players instead")
        else:
            # Case-insensitive partial match
            club_mask = players['current_club'].astype(str).str.contains(
                args.filter_club, case=False, na=False
            )
            players = players[club_mask].copy()
            club_count = len(players)
            print(f"🎯 Filtering by current_club containing '{args.filter_club}'")
            print(f"   Found {club_count} players")
    
    # Preprocess data
    print("🔧 Preprocessing data...")
    players, physio_injuries, non_physio_injuries, matches = preprocess_data_optimized(players, injuries, matches)
    
    # Get player IDs to process
    if args.players:
        player_ids = args.players
        print(f"🎯 Processing {len(player_ids)} specified players")
    else:
        # Get all non-goalkeeper players
        player_ids = players[players['position'] != 'Goalkeeper']['id'].tolist()
        print(f"🎯 Processing {len(player_ids)} non-goalkeeper players")
    
    # Process each player
    print()
    print("🔄 Processing players...")
    print("-" * 70)
    
    results = {
        'new_players': [],
        'rebuilt_players': [],
        'updated_players': [],
        'skipped_players': [],
        'failed_players': []
    }
    
    for i, player_id in enumerate(player_ids, 1):
        print(f"\n[{i}/{len(player_ids)}] Player {player_id}")
        
        # Get player data
        player_filter = players[players['id'] == player_id]
        if player_filter.empty:
            print(f"   ⚠️  Player {player_id} not found in players data")
            results['skipped_players'].append((player_id, "not_found"))
            continue
        
        player_row = player_filter.iloc[0]
        player_matches = matches[matches['player_id'] == player_id].copy()
        player_physio_injuries = physio_injuries[physio_injuries['player_id'] == player_id].copy()
        player_non_physio_injuries = non_physio_injuries[non_physio_injuries['player_id'] == player_id].copy()
        player_career = None
        if career is not None:
            id_col = 'player_id' if 'player_id' in career.columns else ('id' if 'id' in career.columns else None)
            if id_col is not None:
                player_career = career[career[id_col] == player_id].copy()
        
        # Update features
        success, days_added, status = update_player_features(
            player_id=player_id,
            player_row=player_row,
            player_matches=player_matches,
            player_physio_injuries=player_physio_injuries,
            player_non_physio_injuries=player_non_physio_injuries,
            player_career=player_career,
            data=data,
            force_rebuild=args.force_rebuild,
        )
        
        if success:
            if status == "new_player":
                results['new_players'].append((player_id, days_added))
                print(f"   ✅ New player created: {days_added} days")
            elif status == "rebuilt":
                results['rebuilt_players'].append((player_id, days_added))
                print(f"   ✅ Rebuilt from scratch: {days_added} days")
            elif status == "updated":
                results['updated_players'].append((player_id, days_added))
                print(f"   ✅ Updated: +{days_added} days")
            elif status == "no_new_data":
                results['skipped_players'].append((player_id, "no_new_data"))
                print(f"   ⏭️  No new data available")
            elif status == "no_new_days":
                results['skipped_players'].append((player_id, "no_new_days"))
                print(f"   ⏭️  No new days generated")
            else:
                results['skipped_players'].append((player_id, status))
                print(f"   ⏭️  {status}")
        else:
            results['failed_players'].append((player_id, status))
            print(f"   ❌ Failed: {status}")
    
    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✅ New players created: {len(results['new_players'])}")
    print(f"♻️  Players rebuilt: {len(results['rebuilt_players'])}")
    print(f"✅ Existing players updated: {len(results['updated_players'])}")
    print(f"⏭️  Skipped: {len(results['skipped_players'])}")
    print(f"❌ Failed: {len(results['failed_players'])}")
    print()
    
    if results['new_players']:
        total_days_new = sum(days for _, days in results['new_players'])
        print(f"📊 Total days generated for new players: {total_days_new}")
    
    if results['rebuilt_players']:
        total_days_rebuilt = sum(days for _, days in results['rebuilt_players'])
        print(f"📊 Total days generated for rebuilt players: {total_days_rebuilt}")
    
    if results['updated_players']:
        total_days_updated = sum(days for _, days in results['updated_players'])
        print(f"📊 Total days added for updated players: {total_days_updated}")
    
    if results['failed_players']:
        print("\n❌ Failed players:")
        for player_id, reason in results['failed_players']:
            print(f"   - Player {player_id}: {reason}")
    
    return 0 if len(results['failed_players']) == 0 else 1


if __name__ == "__main__":
    exit(main())

