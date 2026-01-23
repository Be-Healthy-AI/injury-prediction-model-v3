#!/usr/bin/env python3
"""
Multi-club migration script for Premier League.

Migrates daily features and timelines from centralized locations to 
club-specific deployment folders. This script does NOT touch Chelsea FC
as it's already set up and running in production.

This script should only be run once during initial setup.
"""

from __future__ import annotations

import argparse
import json
import pandas as pd
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set

# Calculate paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent
ROOT_DIR = PRODUCTION_ROOT.parent

# Paths
DAILY_FEATURES_SOURCE = ROOT_DIR / "daily_features_output"
TIMELINES_SOURCE = ROOT_DIR / "models_production" / "lgbm_muscular_v1" / "data" / "timelines" / "test" / "timelines_35day_season_2025_2026_v4_muscular.csv"
DEPLOYMENTS_DIR = PRODUCTION_ROOT / "deployments" / "England"
CHELSEA_CLUB = "Chelsea FC"  # Skip this club as it's already set up

# Premier League clubs (2025-2026 season)
PREMIER_LEAGUE_CLUBS = {
    "Arsenal FC", "Aston Villa", "AFC Bournemouth", "Brentford FC", 
    "Brighton & Hove Albion", "Burnley FC", "Chelsea FC", "Crystal Palace",
    "Everton FC", "Fulham FC", "Ipswich Town", "Leicester City",
    "Liverpool FC", "Luton Town", "Manchester City", "Manchester United",
    "Newcastle United", "Nottingham Forest", "Sheffield United", 
    "Southampton FC", "Tottenham Hotspur", "West Ham United",
    "Wolverhampton Wanderers"
}


def get_player_club_mapping(timelines_file: Path) -> Dict[int, str]:
    """
    Extract player_id -> club_name mapping from timelines file.
    Uses the most recent entry for each player to determine current club.
    """
    print(f"Loading timelines to extract club mappings from {timelines_file}...")
    
    if not timelines_file.exists():
        print(f"ERROR: Timelines file not found: {timelines_file}")
        return {}
    
    # Read only necessary columns to save memory
    print("Reading timelines file (this may take a moment)...")
    df = pd.read_csv(timelines_file, usecols=['player_id', 'current_club', 'reference_date'], low_memory=False)
    
    # Get most recent club for each player (assuming timelines are sorted by date)
    player_clubs = {}
    for player_id in df['player_id'].unique():
        player_data = df[df['player_id'] == player_id].copy()
        
        # Sort by reference_date to get most recent
        if 'reference_date' in player_data.columns:
            player_data['reference_date'] = pd.to_datetime(player_data['reference_date'], errors='coerce')
            player_data = player_data.sort_values('reference_date', na_position='last')
        
        # Get the most recent entry
        most_recent = player_data.iloc[-1]
        club = most_recent['current_club']
        
        if pd.notna(club) and str(club).strip():
            player_clubs[player_id] = str(club).strip()
    
    print(f"Found {len(player_clubs)} players with club information")
    return player_clubs


def create_club_config(club_name: str, player_ids: List[int], country: str = "England") -> dict:
    """Create a config.json structure for a club."""
    return {
        "club_name": club_name,
        "country": country,
        "player_ids": sorted(player_ids),
        "models": {
            "lgbm_muscular_v2": "../../../models/lgbm_muscular_v2/model.joblib",
            "lgbm_columns": "../../../models/lgbm_muscular_v2/columns.json"
        },
        "default_date_range_days": 7
    }


def migrate_daily_features(
    source_dir: Path,
    player_clubs: Dict[int, str],
    exclude_clubs: Set[str] = None,
    dry_run: bool = False
) -> Dict[str, Dict[str, int]]:
    """
    Copy daily features files to club-specific folders.
    Skips clubs in exclude_clubs set.
    """
    if exclude_clubs is None:
        exclude_clubs = set()
    
    club_stats = defaultdict(lambda: {"copied": 0, "skipped": 0, "errors": 0, "missing": 0})
    
    print(f"\nMigrating daily features from {source_dir}...")
    print(f"Excluding clubs: {', '.join(sorted(exclude_clubs))}")
    
    if not source_dir.exists():
        print(f"WARNING: Daily features source directory not found: {source_dir}")
        return dict(club_stats)
    
    # Get all available daily features files
    available_files = {f.stem.replace('_daily_features', '').replace('player_', '') for f in source_dir.glob('player_*_daily_features.csv')}
    print(f"Found {len(available_files)} daily features files in source directory")
    
    for player_id, club_name in player_clubs.items():
        # Skip excluded clubs
        if club_name in exclude_clubs:
            continue
        
        player_id_str = str(player_id)
        source_file = source_dir / f"player_{player_id}_daily_features.csv"
        
        if not source_file.exists():
            club_stats[club_name]["missing"] += 1
            continue
        
        # Create club folder structure
        club_dir = DEPLOYMENTS_DIR / club_name / "daily_features"
        club_dir.mkdir(parents=True, exist_ok=True)
        
        dest_file = club_dir / f"player_{player_id}_daily_features.csv"
        
        if dest_file.exists():
            club_stats[club_name]["skipped"] += 1
            continue
        
        if dry_run:
            print(f"[DRY RUN] Would copy: player_{player_id}_daily_features.csv -> {club_name}/")
            club_stats[club_name]["copied"] += 1
        else:
            try:
                shutil.copy2(source_file, dest_file)
                file_size = dest_file.stat().st_size / (1024 * 1024)  # MB
                club_stats[club_name]["copied"] += 1
            except Exception as e:
                print(f"[ERROR] Failed to copy player_{player_id}_daily_features.csv for {club_name}: {e}")
                club_stats[club_name]["errors"] += 1
    
    return dict(club_stats)


def migrate_timelines(
    timelines_file: Path,
    player_clubs: Dict[int, str],
    exclude_clubs: Set[str] = None,
    dry_run: bool = False
) -> Dict[str, int]:
    """
    Split timelines file by club and save to club-specific folders.
    Skips clubs in exclude_clubs set.
    """
    if exclude_clubs is None:
        exclude_clubs = set()
    
    print(f"\nMigrating timelines from {timelines_file}...")
    print(f"Excluding clubs: {', '.join(sorted(exclude_clubs))}")
    
    if not timelines_file.exists():
        print(f"ERROR: Timelines file not found: {timelines_file}")
        return {}
    
    # Load timelines
    print("Loading timelines file (this may take a moment)...")
    df = pd.read_csv(timelines_file, low_memory=False)
    print(f"Loaded {len(df)} timeline rows")
    
    # Filter out excluded clubs
    valid_players = {pid: club for pid, club in player_clubs.items() if club not in exclude_clubs}
    valid_player_ids = set(valid_players.keys())
    
    # Filter timelines to only include valid players
    df_filtered = df[df['player_id'].isin(valid_player_ids)].copy()
    print(f"Filtered to {len(df_filtered)} timeline rows for non-excluded clubs")
    
    # Group by club
    club_timelines = defaultdict(list)
    for player_id, club_name in valid_players.items():
        player_timelines = df_filtered[df_filtered['player_id'] == player_id]
        if not player_timelines.empty:
            club_timelines[club_name].append(player_timelines)
    
    # Save timelines for each club
    club_stats = {}
    for club_name, timeline_dfs in club_timelines.items():
        if not timeline_dfs:
            continue
        
        club_timelines_df = pd.concat(timeline_dfs, ignore_index=True)
        
        # Sort by reference_date
        if 'reference_date' in club_timelines_df.columns:
            club_timelines_df['reference_date'] = pd.to_datetime(club_timelines_df['reference_date'], errors='coerce')
            club_timelines_df = club_timelines_df.sort_values('reference_date', na_position='last')
        
        club_dir = DEPLOYMENTS_DIR / club_name / "timelines"
        club_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = club_dir / "timelines_35day_season_2025_2026_v4_muscular.csv"
        
        if dry_run:
            print(f"[DRY RUN] Would save {len(club_timelines_df)} timelines to {club_name}/timelines/")
        else:
            club_timelines_df.to_csv(output_file, index=False)
            print(f"[OK] Saved {len(club_timelines_df)} timelines for {club_name}")
        
        club_stats[club_name] = len(club_timelines_df)
    
    return club_stats


def create_club_configs(
    player_clubs: Dict[int, str],
    exclude_clubs: Set[str] = None,
    dry_run: bool = False
) -> Dict[str, int]:
    """
    Create config.json files for each club.
    Skips clubs in exclude_clubs set.
    """
    if exclude_clubs is None:
        exclude_clubs = set()
    
    print(f"\nCreating club config files...")
    print(f"Excluding clubs: {', '.join(sorted(exclude_clubs))}")
    
    # Group players by club
    club_players = defaultdict(list)
    for player_id, club_name in player_clubs.items():
        if club_name not in exclude_clubs:
            club_players[club_name].append(player_id)
    
    created = {}
    for club_name, player_ids in club_players.items():
        if not player_ids:
            continue
        
        # Convert numpy int64 to Python int for JSON serialization
        player_ids_int = [int(pid) for pid in player_ids]
        config = create_club_config(club_name, player_ids_int)
        config_file = DEPLOYMENTS_DIR / club_name / "config.json"
        
        if dry_run:
            print(f"[DRY RUN] Would create config for {club_name} with {len(player_ids_int)} players")
        else:
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            print(f"[OK] Created config for {club_name} with {len(player_ids_int)} players")
        
        created[club_name] = len(player_ids_int)
    
    return created


def main():
    parser = argparse.ArgumentParser(
        description='Migrate Premier League data to club-specific folders (excludes Chelsea FC)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually doing it'
    )
    parser.add_argument(
        '--skip-daily-features',
        action='store_true',
        help='Skip daily features migration'
    )
    parser.add_argument(
        '--skip-timelines',
        action='store_true',
        help='Skip timelines migration'
    )
    parser.add_argument(
        '--skip-configs',
        action='store_true',
        help='Skip config file creation'
    )
    parser.add_argument(
        '--exclude-clubs',
        type=str,
        default=CHELSEA_CLUB,
        help=f'Comma-separated list of clubs to exclude (default: {CHELSEA_CLUB})'
    )
    parser.add_argument(
        '--daily-features-source',
        type=Path,
        default=DAILY_FEATURES_SOURCE,
        help='Source directory for daily features'
    )
    parser.add_argument(
        '--timelines-source',
        type=Path,
        default=TIMELINES_SOURCE,
        help='Source file for timelines'
    )
    parser.add_argument(
        '--premier-league-only',
        action='store_true',
        help='Filter to only Premier League clubs (excludes U21, U18, retired, etc.)'
    )
    
    args = parser.parse_args()
    
    # Parse exclude clubs
    exclude_clubs = {c.strip() for c in args.exclude_clubs.split(',') if c.strip()}
    print("=" * 80)
    print("PREMIER LEAGUE MULTI-CLUB MIGRATION")
    print("=" * 80)
    print(f"Excluding clubs: {', '.join(sorted(exclude_clubs))}")
    print(f"Dry run: {args.dry_run}")
    print()
    
    # Step 1: Extract player-club mapping from timelines
    if not args.timelines_source.exists():
        print(f"ERROR: Timelines file not found: {args.timelines_source}")
        return 1
    
    player_clubs = get_player_club_mapping(args.timelines_source)
    
    if not player_clubs:
        print("ERROR: No player-club mappings found")
        return 1
    
    # Filter out excluded clubs
    valid_clubs = {club for club in player_clubs.values() if club not in exclude_clubs}
    
    # Optionally filter to only Premier League clubs
    if args.premier_league_only:
        original_count = len(valid_clubs)
        valid_clubs = {club for club in valid_clubs if club in PREMIER_LEAGUE_CLUBS}
        print(f"\nFiltered to Premier League clubs only: {len(valid_clubs)} clubs (from {original_count} total)")
        
        # Filter player_clubs to only include players from valid clubs
        player_clubs = {pid: club for pid, club in player_clubs.items() if club in valid_clubs}
    
    print(f"\nFound {len(valid_clubs)} clubs to migrate (excluding {len(exclude_clubs)} excluded clubs)")
    if len(valid_clubs) <= 30:
        print(f"Clubs: {', '.join(sorted(valid_clubs))}")
    else:
        print(f"Clubs (first 30): {', '.join(sorted(valid_clubs)[:30])}...")
        print(f"(and {len(valid_clubs) - 30} more)")
    
    # Step 2: Create config files
    if not args.skip_configs:
        print("\n" + "=" * 80)
        print("STEP 1: Creating club config files")
        print("=" * 80)
        create_club_configs(player_clubs, exclude_clubs, args.dry_run)
    
    # Step 3: Migrate daily features
    if not args.skip_daily_features:
        print("\n" + "=" * 80)
        print("STEP 2: Migrating daily features")
        print("=" * 80)
        daily_features_stats = migrate_daily_features(
            args.daily_features_source,
            player_clubs,
            exclude_clubs,
            args.dry_run
        )
        
        # Print summary
        print("\nDaily Features Migration Summary:")
        for club, stats in sorted(daily_features_stats.items()):
            print(f"  {club}: {stats['copied']} copied, {stats['skipped']} skipped, "
                  f"{stats['errors']} errors, {stats['missing']} missing")
    
    # Step 4: Migrate timelines
    if not args.skip_timelines:
        print("\n" + "=" * 80)
        print("STEP 3: Migrating timelines")
        print("=" * 80)
        timelines_stats = migrate_timelines(
            args.timelines_source,
            player_clubs,
            exclude_clubs,
            args.dry_run
        )
        
        # Print summary
        print("\nTimelines Migration Summary:")
        for club, count in sorted(timelines_stats.items()):
            print(f"  {club}: {count} timelines")
    
    print("\n" + "=" * 80)
    print("MIGRATION COMPLETE!")
    print("=" * 80)
    print(f"[OK] Processed {len(valid_clubs)} clubs")
    print(f"[SKIP] Excluded {len(exclude_clubs)} clubs: {', '.join(sorted(exclude_clubs))}")
    print("\nNext steps:")
    print("1. Verify migrated data in production/deployments/England/{Club Name}/")
    print("2. Run incremental updates for all clubs from 2025-12-06 onwards")
    print("3. Generate predictions and dashboards for all clubs")
    
    return 0


if __name__ == '__main__':
    exit(main())

