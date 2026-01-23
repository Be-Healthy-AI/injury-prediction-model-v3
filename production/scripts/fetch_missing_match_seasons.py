#!/usr/bin/env python3
"""
Fetch missing match data seasons for players.
Reads the verification report and fetches missing seasons from Transfermarkt.
"""

import argparse
import sys
import io
# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional
import sys

# Add scripts directory to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent
ROOT_DIR = PRODUCTION_ROOT.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.data_collection.transfermarkt_scraper import (
    ScraperConfig,
    TransfermarktScraper,
)
from scripts.data_collection.transformers import transform_matches

def _season_label(season_int: int) -> str:
    """Return season label like '2024_2025' for season key 2024."""
    return f"{season_int}_{season_int + 1}"

def get_player_slug_from_profile(profiles_df: pd.DataFrame, player_id: int) -> Optional[str]:
    """Get player slug from profiles DataFrame."""
    player_row = profiles_df[profiles_df['id'] == player_id]
    if not player_row.empty:
        return player_row.iloc[0].get('slug', None)
    return None

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch missing match data seasons from Transfermarkt"
    )
    parser.add_argument(
        "--country",
        required=True,
        help="Country name (e.g., 'Spain', 'Saudi Arabia') - used to locate data folder"
    )
    parser.add_argument(
        "--as-of-date",
        required=True,
        help="Data extraction date in YYYYMMDD format (e.g., '20260106')"
    )
    parser.add_argument(
        "--league",
        default="",
        help="League name for display purposes (optional)"
    )
    return parser.parse_args()

def _get_current_season_key(as_of_date: datetime) -> int:
    """Determine current season key from date."""
    # Season starts in summer, so if month >= 7, we're in the new season
    if as_of_date.month >= 7:
        return as_of_date.year
    else:
        return as_of_date.year - 1

def main():
    args = parse_args()
    
    # Parse as-of-date
    try:
        if len(args.as_of_date) == 8 and args.as_of_date.isdigit():
            as_of = datetime.strptime(args.as_of_date, "%Y%m%d")
        else:
            as_of = datetime.fromisoformat(args.as_of_date)
    except ValueError:
        print(f"ERROR: Invalid date format: {args.as_of_date}. Use YYYYMMDD format.")
        return 1
    
    # Setup paths
    country_folder = args.country.lower().replace(" ", "_")
    DATA_DIR = PRODUCTION_ROOT / "raw_data" / country_folder / args.as_of_date
    CURRENT_SEASON_DIR = DATA_DIR / "match_data"
    PREVIOUS_SEASONS_DIR = PRODUCTION_ROOT / "raw_data" / country_folder / "previous_seasons"
    VERIFICATION_REPORT = DATA_DIR / "match_data_verification_report.csv"
    
    # Determine current season
    CURRENT_SEASON_KEY = _get_current_season_key(as_of)
    AS_OF_DATE = as_of
    
    # League name for display
    league_name = args.league or args.country
    
    print("=" * 80)
    print(f"FETCHING MISSING MATCH DATA SEASONS FROM TRANSFERMARKT")
    print(f"League: {league_name}")
    print("=" * 80)
    print(f"Country: {args.country}")
    print(f"Data folder: {DATA_DIR}")
    print(f"Current season: {CURRENT_SEASON_KEY}/{CURRENT_SEASON_KEY + 1}")
    print()
    
    # Load verification report
    if not VERIFICATION_REPORT.exists():
        print(f"ERROR: Verification report not found: {VERIFICATION_REPORT}")
        return 1
    
    print(f"Loading verification report: {VERIFICATION_REPORT}")
    report_df = pd.read_csv(VERIFICATION_REPORT, sep=';', encoding='utf-8-sig')
    print(f"  Found {len(report_df)} players with missing seasons")
    print()
    
    # Load profiles to get player slugs
    profiles_path = DATA_DIR / "players_profile.csv"
    if profiles_path.exists():
        print(f"Loading player profiles: {profiles_path}")
        profiles_df = pd.read_csv(profiles_path, sep=';', encoding='utf-8-sig')
        print(f"  Loaded {len(profiles_df)} profiles")
    else:
        print(f"WARNING: Profiles file not found: {profiles_path}")
        profiles_df = pd.DataFrame()
    print()
    
    # Initialize scraper
    scraper = TransfermarktScraper(ScraperConfig())
    
    try:
        total_missing_seasons = 0
        total_fetched = 0
        total_created = 0
        total_no_matches = 0
        total_errors = 0
        total_already_exists = 0
        
        print("Fetching missing seasons for each player...")
        print()
        
        for idx, row in report_df.iterrows():
            player_id = int(row['player_id'])
            player_name = row['player_name']
            missing_seasons_str = row['missing_seasons']
            
            # Parse missing seasons
            if pd.isna(missing_seasons_str) or not missing_seasons_str:
                continue
            
            missing_seasons = [int(s.strip()) for s in str(missing_seasons_str).split(',') if s.strip().isdigit()]
            
            if not missing_seasons:
                continue
            
            total_missing_seasons += len(missing_seasons)
            
            # Get player slug
            player_slug = None
            if not profiles_df.empty:
                player_slug = get_player_slug_from_profile(profiles_df, player_id)
            
            print(f"[{idx+1}/{len(report_df)}] {player_name} (ID: {player_id})")
            print(f"  Missing seasons: {missing_seasons}")
            
            for season in missing_seasons:
                season_label = _season_label(season)
                filename = f"match_{player_id}_{season_label}.csv"
                
                # Determine destination
                if season == CURRENT_SEASON_KEY:
                    match_file_path = CURRENT_SEASON_DIR / filename
                else:
                    match_file_path = PREVIOUS_SEASONS_DIR / filename
                
                # Skip if already exists
                if match_file_path.exists():
                    print(f"    Season {season_label}: File already exists, skipping")
                    total_already_exists += 1
                    total_fetched += 1
                    continue
                
                print(f"    Season {season_label}...", end=" ", flush=True)
                
                try:
                    # Fetch match log
                    raw_matches = scraper.fetch_player_match_log(
                        player_slug, player_id, season=season
                    )
                    
                    if raw_matches.empty:
                        print("No matches")
                        total_no_matches += 1
                        total_fetched += 1
                        continue
                    
                    # Transform matches
                    transformed = transform_matches(player_id, player_name, raw_matches)
                    
                    # Filter by date
                    filtered = transformed[
                        transformed["date"].isna()
                        | (transformed["date"] <= pd.Timestamp(AS_OF_DATE))
                    ]
                    
                    if filtered.empty:
                        print("No matches after date filter")
                        total_no_matches += 1
                        total_fetched += 1
                        continue
                    
                    # Save file
                    match_file_path.parent.mkdir(parents=True, exist_ok=True)
                    filtered.to_csv(match_file_path, index=False, encoding="utf-8-sig", na_rep="")
                    
                    # Stats summary
                    stats_cols = ["position", "goals", "assists", "yellow_cards", "red_cards", "minutes_played"]
                    stats_summary = []
                    for col in stats_cols:
                        if col in filtered.columns:
                            populated = filtered[col].notna().sum()
                            if populated > 0:
                                stats_summary.append(f"{col}:{populated}")
                    
                    stats_str = ", ".join(stats_summary) if stats_summary else "no stats"
                    dest_type = "current season" if season == CURRENT_SEASON_KEY else "previous seasons"
                    print(f"[OK] {len(filtered)} matches ({stats_str}) → {dest_type}")
                    
                    total_created += 1
                    total_fetched += 1
                    
                except Exception as e:
                    print(f"Error: {e}")
                    total_errors += 1
                    total_fetched += 1
                    continue
            
            print()
        
        print()
        print("=" * 80)
        print("FETCHING SUMMARY")
        print("=" * 80)
        print(f"Total missing seasons to fetch: {total_missing_seasons}")
        print(f"Total seasons processed: {total_fetched}")
        print(f"  - Files created: {total_created}")
        print(f"  - No matches found: {total_no_matches}")
        print(f"  - Errors: {total_errors}")
        print(f"  - Already existed: {total_already_exists}")
        print("=" * 80)
        
    finally:
        scraper.close()
    
    return 0

if __name__ == "__main__":
    exit(main())

