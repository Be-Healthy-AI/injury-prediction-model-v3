#!/usr/bin/env python3
"""
Fetch missing match data files for players that don't have them yet.

This script:
1. Loads players_profile.csv and players_career.csv
2. Identifies players missing match data files for current season
3. Fetches only match data for those missing players
4. Skips profile/career/injury fetching to save time
"""
from __future__ import annotations

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Calculate paths relative to this script
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


def _get_current_season_key() -> int:
    """Get current season year (e.g., 2025 for 2025/2026 season)."""
    today = datetime.today()
    # Season starts in July/August, so if we're before August, use previous year
    if today.month < 8:
        return today.year - 1
    return today.year


def _season_label(season: int) -> str:
    """Convert season year to label (e.g., 2025 -> 2025_2026)."""
    return f"{season}_{season+1}"


def _get_all_available_seasons(
    career_df: pd.DataFrame,
    profile_dict: Dict[str, any],
    current_year: int,
) -> List[int]:
    """
    Determine all available seasons for a player to fetch match data.
    """
    seasons_set = set()
    
    if not career_df.empty and "Season" in career_df.columns:
        for season_str in career_df["Season"].dropna().unique():
            if isinstance(season_str, str):
                if "/" in season_str:
                    try:
                        parts = season_str.split("/")
                        if len(parts) == 2:
                            year1 = int(parts[0])
                            year2 = int(parts[1])
                            if year1 < 100:
                                year1 = 2000 + year1 if year1 < 50 else 1900 + year1
                            if year2 < 100:
                                year2 = 2000 + year2 if year2 < 50 else 1900 + year2
                            seasons_set.add(year1)
                            seasons_set.add(year2)
                    except (ValueError, IndexError):
                        pass
                elif "-" in season_str:
                    try:
                        year = int(season_str.split("-")[0])
                        seasons_set.add(year)
                        seasons_set.add(year + 1)
                    except (ValueError, IndexError):
                        pass
    
    date_of_birth = profile_dict.get("date_of_birth") if profile_dict else None
    if date_of_birth:
        try:
            if isinstance(date_of_birth, str):
                birth_year = pd.to_datetime(date_of_birth).year
            else:
                birth_year = date_of_birth.year if hasattr(date_of_birth, 'year') else None
            if birth_year:
                debut_year = birth_year + 16
                for year in range(debut_year, current_year + 1):
                    seasons_set.add(year)
        except (ValueError, AttributeError, TypeError):
            pass
    
    if not seasons_set:
        for year in range(current_year, current_year - 30, -1):
            seasons_set.add(year)
    
    return sorted(seasons_set, reverse=True)


def main():
    parser = argparse.ArgumentParser(description="Fetch missing match data files")
    parser.add_argument(
        "--country",
        type=str,
        default="england",
        help="Country name (default: england)"
    )
    parser.add_argument(
        "--data-date",
        type=str,
        required=True,
        help="Data date in YYYYMMDD format (e.g., 20260122)"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = PRODUCTION_ROOT / "raw_data" / args.country.lower().replace(" ", "_") / args.data_date
    match_data_dir = data_dir / "match_data"
    profile_path = data_dir / "players_profile.csv"
    career_path = data_dir / "players_career.csv"
    
    if not profile_path.exists():
        print(f"[ERROR] Profile file not found: {profile_path}")
        return 1
    
    if not career_path.exists():
        print(f"[ERROR] Career file not found: {career_path}")
        return 1
    
    # Load players
    print(f"Loading players from {profile_path.name}...")
    profiles_df = pd.read_csv(profile_path, sep=';', encoding='utf-8-sig', low_memory=False)
    careers_df = pd.read_csv(career_path, sep=';', encoding='utf-8-sig', low_memory=False)
    
    player_id_col = 'player_id' if 'player_id' in profiles_df.columns else 'id'
    print(f"Found {len(profiles_df)} players")
    
    # Determine current season
    current_season_key = _get_current_season_key()
    as_of = datetime.strptime(args.data_date, "%Y%m%d")
    
    # Find players missing match data
    print(f"\nChecking for missing match data files (current season: {current_season_key})...")
    missing_players = []
    
    for idx, row in profiles_df.iterrows():
        player_id = int(row[player_id_col])
        player_name = row.get('name', f'Player {player_id}')
        
        # Check if match file exists
        season_label = _season_label(current_season_key)
        filename = f"match_{player_id}_{season_label}.csv"
        match_file_path = match_data_dir / filename
        
        if not match_file_path.exists():
            # Get player slug from career data if available
            player_slug = None
            if 'id' in careers_df.columns:
                player_career = careers_df[careers_df['id'] == player_id]
                if not player_career.empty and 'player_slug' in player_career.columns:
                    player_slug = player_career.iloc[0].get('player_slug')
            
            missing_players.append({
                'player_id': player_id,
                'player_name': player_name,
                'player_slug': player_slug
            })
    
    print(f"Found {len(missing_players)} players missing match data files")
    
    if len(missing_players) == 0:
        print("[OK] All players have match data files!")
        return 0
    
    # Initialize scraper
    print(f"\nInitializing Transfermarkt scraper...")
    scraper = TransfermarktScraper(ScraperConfig())
    
    # Fetch match data for missing players
    print(f"\nFetching match data for {len(missing_players)} players...")
    print(f"Current season: {current_season_key} ({_season_label(current_season_key)})")
    
    success_count = 0
    error_count = 0
    
    for idx, player_info in enumerate(missing_players, 1):
        player_id = player_info['player_id']
        player_name = player_info['player_name']
        player_slug = player_info.get('player_slug')
        
        print(f"\n[{idx}/{len(missing_players)}] {player_name} (ID: {player_id})")
        
        season_label = _season_label(current_season_key)
        filename = f"match_{player_id}_{season_label}.csv"
        match_file_path = match_data_dir / filename
        
        try:
            print(f"  Fetching match data for season {season_label}...", end=" ", flush=True)
            raw_matches = scraper.fetch_player_match_log(
                player_slug, player_id, season=current_season_key
            )
            
            if raw_matches.empty:
                print("[OK] No matches found")
                # Create empty file to mark as processed
                pd.DataFrame().to_csv(match_file_path, index=False, encoding="utf-8-sig", na_rep="")
                success_count += 1
                continue
            
            transformed = transform_matches(player_id, player_name, raw_matches)
            filtered = transformed[
                transformed["date"].isna()
                | (transformed["date"] <= pd.Timestamp(as_of))
            ]
            
            if filtered.empty:
                print("[OK] No matches after date filter")
                # Create empty file to mark as processed
                pd.DataFrame().to_csv(match_file_path, index=False, encoding="utf-8-sig", na_rep="")
                success_count += 1
                continue
            
            filtered.to_csv(match_file_path, index=False, encoding="utf-8-sig", na_rep="")
            
            stats_cols = ["position", "goals", "assists", "yellow_cards", "red_cards", "minutes_played"]
            stats_summary = []
            for col in stats_cols:
                if col in filtered.columns:
                    populated = filtered[col].notna().sum()
                    if populated > 0:
                        stats_summary.append(f"{col}:{populated}")
            
            stats_str = ", ".join(stats_summary) if stats_summary else "no stats"
            print(f"[OK] {len(filtered)} matches ({stats_str})")
            success_count += 1
            
        except Exception as e:
            print(f"[ERROR] {str(e)}")
            import traceback
            traceback.print_exc()
            error_count += 1
            continue
    
    scraper.close()
    
    print(f"\n{'='*80}")
    print(f"Match data fetch completed!")
    print(f"  Players processed: {len(missing_players)}")
    print(f"  Success: {success_count}")
    print(f"  Errors: {error_count}")
    print(f"{'='*80}\n")
    
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
