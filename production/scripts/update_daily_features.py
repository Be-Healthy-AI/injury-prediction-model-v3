#!/usr/bin/env python3
"""
Incremental Daily Features Update Script for Production Predictions

This script processes players and:
- For existing players: Appends new days to existing daily features files
- For new players: Generates daily features from scratch

Adapted for production structure with country/club organization.
Reads from Transfermarkt CSV format in production/raw_data/{country}/{date}/
"""

from __future__ import annotations

import argparse
import os
import shutil
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import json
import sys
import io

import pandas as pd
import numpy as np

# Calculate paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent
ROOT_DIR = PRODUCTION_ROOT.parent

# Add both root and scripts directory to path (scripts first for benfica_parity_config)
if str(ROOT_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "scripts"))
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# Ensure stdout is in original state before imports that might modify it
if sys.platform == 'win32':
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

# Temporarily patch io.TextIOWrapper to handle closed file errors gracefully
_original_textiowrapper = io.TextIOWrapper
def _safe_textiowrapper(*args, **kwargs):
    try:
        return _original_textiowrapper(*args, **kwargs)
    except (ValueError, AttributeError, OSError):
        # If wrapping fails, return the original stdout/stderr
        return sys.__stdout__ if 'stdout' in str(kwargs.get('file', '')) else sys.__stderr__

# Import from root scripts (this may modify stdout/stderr)
# We'll handle any errors and restore stdout/stderr after
try:
    from scripts.create_daily_features_v3 import (
        preprocess_data_optimized,
        preprocess_career_data,
        initialize_team_country_map,
        initialize_competition_type_map,
        generate_daily_features_for_player_enhanced,
        generate_incremental_features_for_player,
    )
except (ValueError, OSError):
    # If import failed due to stdout issues, restore and try again
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    from scripts.create_daily_features_v3 import (
        preprocess_data_optimized,
        preprocess_career_data,
        initialize_team_country_map,
        initialize_competition_type_map,
        generate_daily_features_for_player_enhanced,
        generate_incremental_features_for_player,
    )

# Import Transfermarkt data loading functions
try:
    from create_daily_features_transfermarkt import (
        load_data_with_cache as load_transfermarkt_data,
        load_match_data_from_folder,
    )
    HAS_TRANSFERMARKT_LOADER = True
except ImportError:
    HAS_TRANSFERMARKT_LOADER = False

# After all imports, ensure stdout/stderr are valid
# If they were closed by any import, restore from __stdout__/__stderr__
try:
    sys.stdout.write('')
    sys.stdout.flush()
except (ValueError, AttributeError, OSError):
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


def get_club_path(country: str, club: str) -> Path:
    """Get the base path for a specific club deployment."""
    return PRODUCTION_ROOT / "deployments" / country / club


def get_latest_data_folder(country: str) -> Optional[Path]:
    """Find the most recent date folder in production/raw_data/{country}/."""
    raw_data_dir = PRODUCTION_ROOT / "raw_data" / country.lower()
    if not raw_data_dir.exists():
        return None
    
    # Find all date folders (YYYYMMDD format)
    date_folders = []
    for item in raw_data_dir.iterdir():
        if item.is_dir() and item.name.isdigit() and len(item.name) == 8:
            try:
                # Validate it's a valid date
                datetime.strptime(item.name, "%Y%m%d")
                date_folders.append((item.name, item))
            except ValueError:
                continue
    
    if not date_folders:
        return None
    
    # Sort by date (most recent first)
    date_folders.sort(reverse=True)
    return date_folders[0][1]


def load_production_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load data files from Transfermarkt CSV format.
    """
    if not HAS_TRANSFERMARKT_LOADER:
        # Fallback: manual loading
        return load_transfermarkt_data_manual(data_dir)
    
    # Temporarily override CONFIG in create_daily_features_transfermarkt
    import scripts.create_daily_features_transfermarkt as tm_module
    original_data_dir = tm_module.original_module.CONFIG['DATA_DIR']
    original_cache = tm_module.original_module.CONFIG['CACHE_FILE']
    
    try:
        tm_module.original_module.CONFIG['DATA_DIR'] = str(data_dir)
        tm_module.original_module.CONFIG['CACHE_FILE'] = f'data_cache_production_{data_dir.name}.pkl'
        data = load_transfermarkt_data()
        return data
    finally:
        # Restore original config
        tm_module.original_module.CONFIG['DATA_DIR'] = original_data_dir
        tm_module.original_module.CONFIG['CACHE_FILE'] = original_cache


def load_transfermarkt_data_manual(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """Manual loading of Transfermarkt CSV data."""
    players_path = data_dir / "players_profile.csv"
    injuries_path = data_dir / "injuries_data.csv"
    career_path = data_dir / "players_career.csv"
    match_data_dir = data_dir / "match_data"
    
    # Also load from previous_seasons folder
    country_name = data_dir.parent.name  # e.g., "england"
    previous_seasons_dir = data_dir.parent / "previous_seasons"
    
    # Load players (semicolon-separated)
    players = pd.read_csv(players_path, sep=';', encoding='utf-8')
    if 'date_of_birth' in players.columns:
        players['date_of_birth'] = pd.to_datetime(players['date_of_birth'], format='%d/%m/%Y', errors='coerce')
    if 'foot' in players.columns and 'dominant_foot' not in players.columns:
        players['dominant_foot'] = players['foot']
    if 'height' in players.columns and 'height_cm' not in players.columns:
        players['height_cm'] = pd.to_numeric(players['height'], errors='coerce')
    
    # Load injuries (semicolon-separated)
    injuries = pd.read_csv(injuries_path, sep=';', encoding='utf-8')
    
    # Load match data from BOTH current date folder AND previous_seasons folder
    matches = []
    
    # Load from current date folder (current season)
    if match_data_dir.exists():
        match_files = list(match_data_dir.glob("match_*.csv"))
        print(f"   📂 Loading {len(match_files)} match files from current season folder...")
        for match_file in match_files:
            try:
                df = pd.read_csv(match_file, encoding='utf-8-sig')
                matches.append(df)
            except Exception as e:
                print(f"   ⚠️  Error loading {match_file.name}: {e}")
    
    # Load from previous_seasons folder (old seasons)
    if previous_seasons_dir.exists():
        previous_match_files = list(previous_seasons_dir.glob("match_*.csv"))
        print(f"   📂 Loading {len(previous_match_files)} match files from previous_seasons folder...")
        for match_file in previous_match_files:
            try:
                df = pd.read_csv(match_file, encoding='utf-8-sig')
                matches.append(df)
            except Exception as e:
                print(f"   ⚠️  Error loading {match_file.name}: {e}")
    
    matches = pd.concat(matches, ignore_index=True) if matches else pd.DataFrame()
    print(f"   ✅ Loaded {len(matches)} total match records from {len(matches) if matches else 0} files")
    
    # Load career (semicolon-separated)
    career = None
    if career_path.exists():
        career = pd.read_csv(career_path, sep=';', encoding='utf-8')
    
    return {
        'players': players,
        'injuries': injuries,
        'matches': matches,
        'teams': None,
        'competitions': None,
        'career': career,
    }


def get_existing_last_date(player_id: int, features_dir: Path) -> Optional[pd.Timestamp]:
    """Get the last date from existing daily features file."""
    features_file = features_dir / f"player_{player_id}_daily_features.csv"
    
    if not features_file.exists():
        return None
    
    try:
        df = pd.read_csv(features_file, parse_dates=['date'], nrows=1)
        if 'date' not in df.columns:
            return None
        
        df_full = pd.read_csv(features_file, parse_dates=['date'])
        if df_full.empty:
            return None
        
        last_date = pd.to_datetime(df_full['date'].max())
        return last_date
    except Exception as e:
        print(f"   ⚠️  Error reading existing file for player {player_id}: {e}")
        return None


def backup_existing_file(player_id: int, features_dir: Path) -> bool:
    """Create backup of existing daily features file."""
    features_file = features_dir / f"player_{player_id}_daily_features.csv"
    
    if not features_file.exists():
        return True
    
    try:
        backup_file = features_dir / f"player_{player_id}_daily_features.backup.csv"
        shutil.copy2(features_file, backup_file)
        return True
    except Exception as e:
        print(f"   ⚠️  Failed to backup file for player {player_id}: {e}")
        return False


def restore_backup(player_id: int, features_dir: Path) -> bool:
    """Restore from backup if update failed."""
    backup_file = features_dir / f"player_{player_id}_daily_features.backup.csv"
    features_file = features_dir / f"player_{player_id}_daily_features.csv"
    
    if not backup_file.exists():
        return False
    
    try:
        shutil.copy2(backup_file, features_file)
        backup_file.unlink()
        return True
    except Exception as e:
        print(f"   ⚠️  Failed to restore backup for player {player_id}: {e}")
        return False


def update_player_features(
    player_id: int,
    player_row: pd.Series,
    player_matches: pd.DataFrame,
    player_physio_injuries: pd.DataFrame,
    player_non_physio_injuries: pd.DataFrame,
    player_career: Optional[pd.DataFrame],
    features_dir: Path,
    force_rebuild: bool = False,
) -> tuple[bool, int, str]:
    """
    Update daily features for a single player.
    
    Returns:
        (success, days_added, status_message)
    """
    features_file = features_dir / f"player_{player_id}_daily_features.csv"
    is_new_player = not features_file.exists()
    
    if not is_new_player:
        if not backup_existing_file(player_id, features_dir):
            return False, 0, "backup_failed"
    
    try:
        # Use today as end date (no hard cap)
        target_end_date = pd.Timestamp.today().normalize()
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
            
            daily_features.to_csv(features_file, index=False, encoding='utf-8-sig')
            days_added = len(daily_features)
            status = "new_player" if is_new_player else "rebuilt"
            return True, days_added, status
        
        else:
            # Incremental update
            last_date = get_existing_last_date(player_id, features_dir)
            if last_date is None:
                return False, 0, "cannot_read_last_date"
            
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
            backup_file = features_dir / f"player_{player_id}_daily_features.backup.csv"
            if backup_file.exists():
                backup_file.unlink()
            
            return True, days_added, "updated"
    
    except Exception as e:
        print(f"   ❌ Error updating player {player_id}: {e}")
        if not is_new_player:
            restore_backup(player_id, features_dir)
        return False, 0, f"error: {str(e)}"


def main():
    # Ensure stdout is properly set up and not closed
    try:
        # Test if stdout is writable
        sys.stdout.write('')
        sys.stdout.flush()
    except (ValueError, AttributeError, OSError) as e:
        # stdout is closed or invalid, restore from __stdout__
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--country',
        type=str,
        required=True,
        help='Country name (e.g., "England")'
    )
    parser.add_argument(
        '--club',
        type=str,
        required=True,
        help='Club name (e.g., "Chelsea FC")'
    )
    parser.add_argument(
        '--data-date',
        type=str,
        default=None,
        help='Date folder to use for raw data (YYYYMMDD format). Defaults to latest available.'
    )
    parser.add_argument(
        '--players',
        type=int,
        nargs='*',
        help='Specific player IDs to process (if not provided, uses club config or processes all)'
    )
    parser.add_argument(
        '--force-rebuild',
        action='store_true',
        help='Regenerate daily features from scratch for all processed players'
    )
    args = parser.parse_args()
    
    # Get club path
    club_path = get_club_path(args.country, args.club)
    features_dir = club_path / "daily_features"
    features_dir.mkdir(parents=True, exist_ok=True)
    
    # Get data directory
    if args.data_date:
        data_dir = PRODUCTION_ROOT / "raw_data" / args.country.lower() / args.data_date
    else:
        data_dir = get_latest_data_folder(args.country)
        if data_dir is None:
            print(f"[ERROR] No data folder found for country '{args.country}'")
            print(f"   Expected location: {PRODUCTION_ROOT / 'raw_data' / args.country.lower()}")
            return 1
    
    if not data_dir.exists():
        print(f"[ERROR] Data directory not found: {data_dir}")
        return 1
    
    print("=" * 70)
    print("INCREMENTAL DAILY FEATURES UPDATE FOR PRODUCTION PREDICTIONS")
    print("=" * 70)
    print(f"🌍 Country: {args.country}")
    print(f" club: {args.club}")
    print(f"📂 Data directory: {data_dir}")
    print(f"📂 Output directory: {features_dir}")
    print()
    
    # Load data
    print("📥 Loading raw data files...")
    try:
        data = load_production_data(data_dir)
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        import traceback
        traceback.print_exc()
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
    
    # Preprocess data
    print("🔧 Preprocessing data...")
    players, physio_injuries, non_physio_injuries, matches = preprocess_data_optimized(players, injuries, matches)
    
    # Get player IDs to process
    if args.players:
        player_ids = args.players
        print(f"🎯 Processing {len(player_ids)} specified players")
    else:
        # Try to load from club config
        config_file = club_path / "config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
            player_ids = config.get('player_ids', [])
            print(f"🎯 Processing {len(player_ids)} players from club config")
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
            features_dir=features_dir,
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

