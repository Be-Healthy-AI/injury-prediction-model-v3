#!/usr/bin/env python3
"""
Diagnostic script to investigate why some players don't have predictions.

This script checks:
1. Which players are in timeline file vs config
2. PL membership data for missing players
3. Activity requirements for missing players
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Set

import pandas as pd

# Fix encoding issues on Windows
if sys.platform == 'win32':
    import io
    if not isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Calculate paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent
ROOT_DIR = PRODUCTION_ROOT.parent

# Add paths for V4 imports
V4_CODE_DIR = ROOT_DIR / "models_production" / "lgbm_muscular_v4" / "code" / "timelines"
if str(V4_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(V4_CODE_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import V4 timeline functions
try:
    import importlib.util
    v4_timeline_module_path = V4_CODE_DIR / "create_35day_timelines_v4_enhanced.py"
    spec = importlib.util.spec_from_file_location("create_35day_timelines_v4_enhanced", v4_timeline_module_path)
    v4_timeline_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v4_timeline_module)
    
    # Get functions we need
    build_player_pl_membership_periods_v4 = v4_timeline_module.build_player_pl_membership_periods
    build_pl_clubs_per_season_v4 = v4_timeline_module.build_pl_clubs_per_season
    get_season_date_range_v4 = v4_timeline_module.get_season_date_range
    is_club_pl_club_v4 = v4_timeline_module.is_club_pl_club
    
    # Get constants
    MIN_ACTIVITY_MINUTES = getattr(v4_timeline_module, 'MIN_ACTIVITY_MINUTES', 90)
    
except Exception as e:
    print(f"[ERROR] Failed to import V4 timeline module: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


def get_challenger_path(country: str, club: str) -> Path:
    """Get base path for challenger club."""
    return PRODUCTION_ROOT / "deployments" / country / "challenger" / club


def get_raw_data_path(country: str, data_date: str) -> Path:
    """Get raw data path (shared with V3)."""
    country_lower = country.lower().replace(" ", "_")
    return PRODUCTION_ROOT / "raw_data" / country_lower / data_date


def load_player_ids_from_config(config_path: Path) -> List[int]:
    """Load player IDs from deployment config.json."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        player_ids = config.get('player_ids', [])
        return player_ids
    except Exception as e:
        print(f"[ERROR] Error loading config from {config_path}: {e}")
        return []


def check_pl_membership(player_id: int, season: str, raw_data_path: Path, pl_clubs_per_season: dict, player_pl_periods: dict) -> bool:
    """Check if player has PL membership for the given season."""
    try:
        # Check if player has any PL membership periods
        if player_id not in player_pl_periods:
            return False
        
        # Get season date range - extract season start year from string
        if '-' in season:
            season_start_year = int(season.split('-')[0])
        elif '_' in season:
            season_start_year = int(season.split('_')[0])
        else:
            season_start_year = int(season)
        season_start, season_end = get_season_date_range_v4(season_start_year)
        
        # Check if any period overlaps with the season
        for period in player_pl_periods[player_id]:
            period_start, period_end = period
            if period_start <= season_end and period_end >= season_start:
                return True
        
        return False
    except Exception as e:
        print(f"  [ERROR] Error checking PL membership for player {player_id}: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_activity(player_id: int, challenger_path: Path) -> dict:
    """Check activity requirements for a player."""
    result = {
        'has_daily_features': False,
        'daily_features_file': None,
        'date_range': None,
        'min_minutes': None,
        'meets_activity': False
    }
    
    # Check for daily features files
    daily_features_dir = challenger_path / "daily_features"
    if not daily_features_dir.exists():
        return result
    
    # Look for player's daily features file
    pattern = f"daily_features_{player_id}_*.csv"
    feature_files = list(daily_features_dir.glob(pattern))
    
    if len(feature_files) == 0:
        return result
    
    result['has_daily_features'] = True
    result['daily_features_file'] = str(feature_files[0])
    
    # Load and check date range and activity
    try:
        df = pd.read_csv(feature_files[0])
        if 'reference_date' in df.columns:
            df['reference_date'] = pd.to_datetime(df['reference_date'])
            result['date_range'] = (df['reference_date'].min(), df['reference_date'].max())
        
        # Check for minutes_last_35d or similar activity columns
        activity_cols = ['minutes_last_35d', 'minutes_last_28d', 'minutes_last_7d']
        for col in activity_cols:
            if col in df.columns:
                max_minutes = df[col].max()
                if pd.notna(max_minutes):
                    result['min_minutes'] = max(result['min_minutes'] or 0, max_minutes)
        
        result['meets_activity'] = (result['min_minutes'] or 0) >= MIN_ACTIVITY_MINUTES
    except Exception as e:
        print(f"  [ERROR] Error reading daily features for player {player_id}: {e}")
    
    return result


def main():
    """Main diagnostic function."""
    parser = argparse.ArgumentParser(description='Diagnose missing players (V4)')
    parser.add_argument('--country', type=str, default='England', help='Country name (default: England)')
    parser.add_argument('--club', type=str, default='Arsenal FC', help='Club name (default: Arsenal FC)')
    parser.add_argument('--season', type=str, default='2025_2026', help='Season (default: 2025_2026)')
    parser.add_argument('--data-date', type=str, help='Raw data date (e.g., 20251205). If not provided, will try to find latest.')
    
    args = parser.parse_args()
    
    challenger_path = get_challenger_path(args.country, args.club)
    config_path = challenger_path / "config.json"
    timeline_path = challenger_path / "timelines" / f"timelines_35day_season_{args.season}_v4_muscular.csv"
    
    print("=" * 80)
    print("DIAGNOSTIC: Missing Players Investigation")
    print("=" * 80)
    print(f"Country: {args.country}")
    print(f"Club: {args.club}")
    print(f"Season: {args.season}")
    print()
    
    # 1. Load config and timeline
    print("[1] Loading config and timeline...")
    expected_players = set(load_player_ids_from_config(config_path))
    print(f"  Expected players in config: {len(expected_players)}")
    
    if not timeline_path.exists():
        print(f"  [ERROR] Timeline file not found: {timeline_path}")
        return 1
    
    timeline_df = pd.read_csv(timeline_path)
    actual_players = set(timeline_df['player_id'].dropna().unique().astype(int))
    print(f"  Players in timeline: {len(actual_players)}")
    
    # Find missing players
    missing_players = sorted(expected_players - actual_players)
    print(f"  Missing players: {len(missing_players)}")
    print()
    
    if len(missing_players) == 0:
        print("✅ All players have timelines!")
        return 0
    
    # 2. Find raw data path and build PL membership data
    print("[2] Finding raw data and building PL membership...")
    if args.data_date:
        raw_data_path = get_raw_data_path(args.country, args.data_date)
    else:
        # Try to find latest raw data
        country_lower = args.country.lower().replace(" ", "_")
        raw_data_base = PRODUCTION_ROOT / "raw_data" / country_lower
        if raw_data_base.exists():
            data_dates = sorted([d.name for d in raw_data_base.iterdir() if d.is_dir() and d.name != "previous_seasons"], reverse=True)
            if data_dates:
                raw_data_path = raw_data_base / data_dates[0]
                print(f"  Using latest raw data: {data_dates[0]}")
            else:
                print(f"  [WARN] No raw data directories found in {raw_data_base}")
                raw_data_path = None
        else:
            print(f"  [WARN] Raw data base path not found: {raw_data_base}")
            raw_data_path = None
    
    # Build PL membership periods if raw data is available
    pl_clubs_per_season = {}
    player_pl_periods = {}
    if raw_data_path and raw_data_path.exists():
        try:
            match_data_dir = raw_data_path / "match_data"
            career_file = raw_data_path / "players_career.csv"
            
            if match_data_dir.exists() and career_file.exists():
                print(f"  Building PL clubs per season...")
                pl_clubs_per_season = build_pl_clubs_per_season_v4(str(match_data_dir))
                print(f"  Found PL clubs for {len(pl_clubs_per_season)} seasons")
                
                # Diagnostic: Check if Arsenal is detected as PL club
                if pl_clubs_per_season:
                    for season_year, clubs in sorted(pl_clubs_per_season.items()):
                        arsenal_variations = [c for c in clubs if 'arsenal' in c.lower()]
                        print(f"    Season {season_year}: {len(clubs)} clubs")
                        if arsenal_variations:
                            print(f"      ✅ Arsenal found: {arsenal_variations}")
                        else:
                            print(f"      ❌ Arsenal NOT found in PL clubs for season {season_year}")
                            # Show sample clubs
                            sample_clubs = sorted(list(clubs))[:10]
                            print(f"      Sample clubs: {sample_clubs}")
                
                print(f"  Building player PL membership periods...")
                player_pl_periods = build_player_pl_membership_periods_v4(str(career_file), pl_clubs_per_season)
                print(f"  Found PL membership for {len(player_pl_periods)} players")
                
                # Diagnostic: Check player 144028's career data
                if (raw_data_path / "players_career.csv").exists():
                    career_df = pd.read_csv(raw_data_path / "players_career.csv", sep=';', encoding='utf-8-sig', low_memory=False)
                    player_144028 = career_df[career_df['id'] == 144028].copy()
                    if len(player_144028) > 0:
                        print(f"\n  [DEBUG] Player 144028 (Leandro Trossard) career data:")
                        player_144028['Date'] = pd.to_datetime(player_144028['Date'], errors='coerce')
                        player_144028 = player_144028.sort_values('Date')
                        for idx, row in player_144028.iterrows():
                            date = row['Date']
                            to_club = row['To']
                            if pd.notna(date):
                                season_year = date.year if date.month >= 7 else date.year - 1
                                is_pl = is_club_pl_club_v4(str(to_club), season_year, pl_clubs_per_season) if pl_clubs_per_season else False
                                print(f"    {date.date()}: {to_club} (season {season_year}, PL club: {'✅' if is_pl else '❌'})")
                        if 144028 in player_pl_periods:
                            print(f"    PL periods created: {len(player_pl_periods[144028])}")
                            for i, (start, end) in enumerate(player_pl_periods[144028]):
                                print(f"      Period {i+1}: {start.date()} to {end.date()}")
                        else:
                            print(f"    ❌ No PL periods created for player 144028")
            else:
                print(f"  [WARN] Match data or career file not found")
        except Exception as e:
            print(f"  [ERROR] Failed to build PL membership: {e}")
            import traceback
            traceback.print_exc()
    
    print()
    
    # 3. Check each missing player
    print(f"[3] Investigating {len(missing_players)} missing players...")
    print()
    
    season_full = f"{args.season.split('_')[0]}-{args.season.split('_')[1]}"
    
    for player_id in missing_players:
        print(f"Player ID: {player_id}")
        print("-" * 60)
        
        # Check PL membership
        if player_pl_periods:
            has_pl_membership = check_pl_membership(player_id, season_full, raw_data_path, pl_clubs_per_season, player_pl_periods)
            print(f"  PL Membership (season {season_full}): {'✅ YES' if has_pl_membership else '❌ NO'}")
            if player_id in player_pl_periods:
                periods = player_pl_periods[player_id]
                print(f"    PL periods found: {len(periods)}")
                for i, (start, end) in enumerate(periods[:3]):  # Show first 3
                    print(f"      Period {i+1}: {start.date()} to {end.date()}")
                if len(periods) > 3:
                    print(f"      ... and {len(periods) - 3} more")
        else:
            print(f"  PL Membership: ⚠️  Cannot check (PL membership data not available)")
            has_pl_membership = None
        
        # Check activity
        activity_info = check_activity(player_id, challenger_path)
        print(f"  Daily Features File: {'✅ YES' if activity_info['has_daily_features'] else '❌ NO'}")
        if activity_info['has_daily_features']:
            if activity_info['date_range']:
                print(f"    Date range: {activity_info['date_range'][0].date()} to {activity_info['date_range'][1].date()}")
            if activity_info['min_minutes'] is not None:
                meets = activity_info['min_minutes'] >= MIN_ACTIVITY_MINUTES
                status = '✅' if meets else '❌'
                print(f"    Max minutes (35d window): {activity_info['min_minutes']:.0f} (required: {MIN_ACTIVITY_MINUTES}, {status})")
                print(f"    Meets activity requirement: {'✅ YES' if meets else '❌ NO'}")
            else:
                print(f"    ⚠️  Could not determine activity from daily features")
        
        print()
    
    print("=" * 80)
    print("Summary:")
    print(f"  Total expected players: {len(expected_players)}")
    print(f"  Players with timelines: {len(actual_players)}")
    print(f"  Missing players: {len(missing_players)}")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
