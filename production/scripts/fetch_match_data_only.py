#!/usr/bin/env python3
"""
Extract ONLY match data for players, skipping profile/career/injury fetching.
This script loads existing profile/career data and goes directly to match data extraction.
"""

import argparse
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent
ROOT_DIR = PRODUCTION_ROOT.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.data_collection.transfermarkt_scraper import ScraperConfig, TransfermarktScraper
from scripts.data_collection.transformers import transform_matches


def _season_label(season_int: int) -> str:
    """Return season label like '2024_2025' for season key 2024."""
    return f"{season_int}_{season_int + 1}"


def _get_current_season_key() -> int:
    """Determine current season key (e.g., 2025 for 2025/26 season)."""
    now = datetime.now()
    if now.month >= 7:
        return now.year
    else:
        return now.year - 1


def _get_all_available_seasons(
    career_df: pd.DataFrame,
    profile_dict: Dict,
    current_year: int,
) -> List[int]:
    """Determine all available seasons for a player to fetch match data."""
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
    parser = argparse.ArgumentParser(description="Extract match data only (skip profile/career/injury fetching)")
    parser.add_argument("--country", required=True, help="Country name (e.g., 'Portugal')")
    parser.add_argument("--as-of-date", required=True, help="Data extraction date (YYYYMMDD format)")
    parser.add_argument("--resume", action="store_true", help="Skip existing match files")
    args = parser.parse_args()
    
    # Parse as-of-date
    if len(args.as_of_date) == 8 and args.as_of_date.isdigit():
        as_of = datetime.strptime(args.as_of_date, "%Y%m%d")
    else:
        as_of = datetime.fromisoformat(args.as_of_date)
    
    # Setup paths
    country_folder = args.country.lower().replace(" ", "_")
    output_dir = PRODUCTION_ROOT / "raw_data" / country_folder / args.as_of_date
    match_data_dir = output_dir / "match_data"
    previous_seasons_dir = PRODUCTION_ROOT / "raw_data" / country_folder / "previous_seasons"
    
    # Create directories if they don't exist
    match_data_dir.mkdir(parents=True, exist_ok=True)
    previous_seasons_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Match data directory: {match_data_dir}")
    print(f"  Previous seasons directory: {previous_seasons_dir}")
    
    profile_path = output_dir / "players_profile.csv"
    career_path = output_dir / "players_career.csv"
    
    if not profile_path.exists():
        print(f"ERROR: Profile file not found: {profile_path}")
        print("Please run the full extraction first to generate profiles and careers.")
        return 1
    
    if not career_path.exists():
        print(f"ERROR: Career file not found: {career_path}")
        print("Please run the full extraction first to generate profiles and careers.")
        return 1
    
    # Load existing data
    print(f"Loading existing profiles and careers...")
    profiles_df = pd.read_csv(profile_path, encoding="utf-8-sig", sep=';')
    careers_df = pd.read_csv(career_path, encoding="utf-8-sig", sep=';')
    print(f"  Loaded {len(profiles_df)} profiles and {len(careers_df)} career entries")
    
    # Create master_players dict from profiles
    master_players: Dict[int, Dict] = {}
    for _, row in profiles_df.iterrows():
        player_id = int(row['id'])
        master_players[player_id] = {
            "player_id": player_id,
            "player_slug": None,  # We'll try to get from profile if available
            "player_name": row.get('name', f"Player {player_id}"),
            "seasons": set(),  # Will be expanded from career data
        }
    
    print(f"  Found {len(master_players)} players")
    
    # Expand seasons for each player using existing career data
    print(f"\nExpanding seasons for each player from career data...")
    current_season_key = _get_current_season_key()
    
    for player_id, player in master_players.items():
        player_name = player["player_name"]
        career_rows = careers_df[careers_df['id'] == player_id]
        
        profile_row = profiles_df[profiles_df['id'] == player_id]
        profile_dict = {}
        if not profile_row.empty:
            profile_dict = {
                "date_of_birth": profile_row.iloc[0].get("date_of_birth"),
            }
        
        try:
            all_seasons = _get_all_available_seasons(career_rows, profile_dict, as_of.year)
            player["seasons"] = sorted(list(set(all_seasons)), reverse=True)
        except Exception as e:
            print(f"  Warning: Could not expand seasons for {player_name} (ID: {player_id}): {e}")
            # Use a default range if expansion fails
            player["seasons"] = list(range(current_season_key, current_season_key - 10, -1))
    
    print(f"  [OK] Expanded seasons for all players")
    
    # Initialize scraper
    scraper = TransfermarktScraper(ScraperConfig())
    
    # Step 5: Extract match data
    print(f"\n{'='*80}")
    print(f"Extracting match data only (skipping profile/career/injury fetching)")
    print(f"{'='*80}")
    print(f"Country: {args.country}")
    print(f"As of date: {as_of:%Y-%m-%d}")
    print(f"Output: {output_dir}")
    print(f"Current season: {current_season_key} ({_season_label(current_season_key)})")
    print(f"Players: {len(master_players)}")
    print(f"{'='*80}\n")
    
    total_match_files = sum(len(p["seasons"]) for p in master_players.values())
    processed_files = 0
    current_season_files = 0
    old_season_files = 0
    skipped_old_seasons = 0
    skipped_current_season = 0
    
    for player_idx, player in enumerate(master_players.values(), 1):
        player_id = player["player_id"]
        player_slug = player.get("player_slug")
        player_name = player["player_name"]
        player_seasons = player["seasons"]
        
        print(f"\n[{player_idx}/{len(master_players)}] {player_name} (ID: {player_id})")
        print(f"  Seasons: {len(player_seasons)}")
        
        for season_idx, season in enumerate(player_seasons, 1):
            season_label = _season_label(season)
            filename = f"match_{player_id}_{season_label}.csv"
            
            if season == current_season_key:
                match_file_path = match_data_dir / filename
                is_current_season = True
            else:
                match_file_path = previous_seasons_dir / filename
                is_current_season = False
            
            # Skip if file exists (previous seasons always skip, current season only with --resume)
            if not is_current_season and match_file_path.exists():
                print(f"    Season {season_label} ({season_idx}/{len(player_seasons)}): Skipping (previous season file exists)")
                processed_files += 1
                old_season_files += 1
                skipped_old_seasons += 1
                continue
            
            if is_current_season and match_file_path.exists():
                if args.resume:
                    print(f"    Season {season_label} ({season_idx}/{len(player_seasons)}): Skipping (file exists, --resume)")
                    processed_files += 1
                    current_season_files += 1
                    skipped_current_season += 1
                    continue
                else:
                    # Without --resume, we still skip existing current season files to avoid overwriting
                    print(f"    Season {season_label} ({season_idx}/{len(player_seasons)}): Skipping (file exists)")
                    processed_files += 1
                    current_season_files += 1
                    skipped_current_season += 1
                    continue
            
            print(f"    Season {season_label} ({season_idx}/{len(player_seasons)})...", end=" ", flush=True)
            
            try:
                # Try to get player_slug from profile if not available
                if not player_slug:
                    profile_row = profiles_df[profiles_df['id'] == player_id]
                    if not profile_row.empty:
                        # Try to construct slug from name (approximation)
                        player_slug = None  # Will use player_id directly
                
                raw_matches = scraper.fetch_player_match_log(player_slug, player_id, season=season)
                
                if raw_matches.empty:
                    print("No matches")
                    processed_files += 1
                    continue
                
                transformed = transform_matches(player_id, player_name, raw_matches)
                filtered = transformed[
                    transformed["date"].isna()
                    | (transformed["date"] <= pd.Timestamp(as_of))
                ]
                
                if filtered.empty:
                    print("No matches after date filter")
                    processed_files += 1
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
                dest_type = "current season" if is_current_season else "previous seasons"
                print(f"[OK] {len(filtered)} matches ({stats_str}) → {dest_type}")
                processed_files += 1
                if is_current_season:
                    current_season_files += 1
                else:
                    old_season_files += 1
                
            except Exception as e:
                print(f"Error for {player_name} (ID: {player_id}), season {season_label}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\n{'='*80}")
    print(f"Match data extraction completed!")
    print(f"  Players processed: {len(master_players)}")
    print(f"  Match files processed: {processed_files}")
    print(f"    - Current season ({current_season_key}): {current_season_files} (skipped {skipped_current_season})")
    print(f"    - Previous seasons: {old_season_files} (skipped {skipped_old_seasons} existing files)")
    print(f"{'='*80}\n")
    
    scraper.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n[WARNING] Interrupted by user. Exiting...", flush=True)
        sys.exit(130)
    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
