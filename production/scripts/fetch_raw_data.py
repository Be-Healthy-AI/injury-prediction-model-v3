#!/usr/bin/env python3
"""
Scrape Transfermarkt data for an entire league and export tables following the
reference schema. 

By default, focuses on current season players only:
1. Fetches current season clubs in the specified league
2. Identifies all players of all clubs
3. Compares with players from last update folder
4. Re-fetches all data (profiles, careers, injuries) for all current season players
5. Updates match data for current season

Output structure:
- production/raw_data/{country}/{as_of_date}/players_profile.csv (consolidated)
- production/raw_data/{country}/{as_of_date}/players_career.csv (consolidated)
- production/raw_data/{country}/{as_of_date}/injuries_data.csv (consolidated)
- production/raw_data/{country}/{as_of_date}/match_data/match_<player_id>_<season-1>_<season>.csv (current season only)
- production/raw_data/{country}/previous_seasons/match_<player_id>_<season-1>_<season>.csv (old seasons, shared across dates)

Adapted for production structure.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd

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
from scripts.data_collection.transformers import (
    transform_career,
    transform_injuries,
    transform_matches,
    transform_profile,
    calculate_severity_from_days,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--country",
        default="england",
        help="Country name (e.g., 'england') - used for output folder. Default: england",
    )
    parser.add_argument(
        "--league",
        required=True,
        help="League display name (e.g., 'Premier League') - for logging",
    )
    parser.add_argument(
        "--competition-id",
        required=True,
        help="Transfermarkt competition ID (e.g., 'GB1' for Premier League, or numeric ID)",
    )
    parser.add_argument(
        "--competition-slug",
        required=True,
        help="Transfermarkt URL slug (e.g., 'premier-league')",
    )
    parser.add_argument(
        "--seasons",
        default=None,
        help="Comma-separated list of season years (e.g., '2022,2023,2024,2025'). Defaults to current season only.",
    )
    parser.add_argument(
        "--as-of-date",
        default=None,
        help="ISO date – rows beyond this date are discarded. Defaults to today (YYYYMMDD format).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-processed players/files",
    )
    parser.add_argument(
        "--max-clubs",
        type=int,
        help="Limit number of clubs for testing (optional)",
    )
    parser.add_argument(
        "--max-players-per-club",
        type=int,
        help="Limit number of players per club for testing (optional)",
    )
    return parser.parse_args(argv)


def _season_label(season_int: int) -> str:
    """Return season label like '2024_2025' for season key 2024 (which represents 2024/25 season)."""
    return f"{season_int}_{season_int + 1}"


def _get_current_season_key() -> int:
    """Determine current season key (e.g., 2025 for 2025/26 season)."""
    now = datetime.now()
    # Season key is the starting year (e.g., 2025 for 2025/26 season)
    # If we're past June, we're in the new season
    if now.month >= 7:
        return now.year
    else:
        return now.year - 1


def get_latest_data_folder(country: str) -> Optional[Path]:
    """Find the most recent date folder in production/raw_data/{country}/."""
    raw_data_dir = PRODUCTION_ROOT / "raw_data" / country.lower().replace(" ", "_")
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


def check_existing_daily_features(player_id: int, country: str) -> Optional[Path]:
    """
    Check if player has existing daily features file in any club deployment.
    Returns path to the file if found, None otherwise.
    """
    deployments_dir = PRODUCTION_ROOT / "deployments" / country
    if not deployments_dir.exists():
        return None
    
    # Search all club folders
    for club_dir in deployments_dir.iterdir():
        if club_dir.is_dir():
            features_file = club_dir / "daily_features" / f"player_{player_id}_daily_features.csv"
            if features_file.exists():
                return features_file
    
    return None


def _enrich_injuries_with_clubs(injuries_df: pd.DataFrame, career_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich injuries DataFrame with club information from career history.
    
    For each injury, finds the club the player was at during the injury period
    based on the career transfer history.
    """
    if injuries_df.empty or career_df.empty:
        return injuries_df
    
    from_col = None
    for col_name in ["from", "From", "fromDate"]:
        if col_name in injuries_df.columns:
            from_col = col_name
            break
    
    if not from_col:
        return injuries_df
    
    career_with_dates = career_df.copy()
    if "Date" in career_with_dates.columns:
        career_with_dates["Date"] = pd.to_datetime(career_with_dates["Date"], errors="coerce")
        career_with_dates = career_with_dates.sort_values("Date", ascending=False)
    
    injuries_with_dates = injuries_df.copy()
    injuries_with_dates["_fromDate"] = pd.to_datetime(
        injuries_with_dates[from_col], errors="coerce", dayfirst=True
    )
    
    clubs_list = []
    for idx, row in injuries_with_dates.iterrows():
        injury_date = row["_fromDate"]
        if pd.isna(injury_date):
            clubs_list.append(None)
            continue
        
        club_name = None
        if "Date" in career_with_dates.columns and "To" in career_with_dates.columns:
            valid_transfers = career_with_dates[
                career_with_dates["Date"].notna() & 
                (career_with_dates["Date"] <= injury_date)
            ]
            if not valid_transfers.empty:
                club_name = valid_transfers.iloc[0].get("To")
        
        clubs_list.append(club_name)
    
    if "Club" in injuries_df.columns:
        injuries_df["Club"] = clubs_list
    elif "clubs" in injuries_df.columns:
        injuries_df["clubs"] = clubs_list
    else:
        injuries_df["Club"] = clubs_list
    
    return injuries_df


def _get_all_available_seasons(
    career_df: pd.DataFrame,
    profile_dict: Dict[str, Any],
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


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    
    # Determine as_of_date
    if args.as_of_date:
        # If provided as YYYYMMDD, convert to ISO format
        if len(args.as_of_date) == 8 and args.as_of_date.isdigit():
            as_of = datetime.strptime(args.as_of_date, "%Y%m%d")
        else:
            as_of = datetime.fromisoformat(args.as_of_date)
    else:
        as_of = datetime.today()
    
    # Setup output directory - use production structure
    output_dir = (
        PRODUCTION_ROOT / "raw_data" / args.country.lower().replace(" ", "_") / as_of.strftime("%Y%m%d")
    )
    match_data_dir = output_dir / "match_data"
    previous_seasons_dir = PRODUCTION_ROOT / "raw_data" / args.country.lower().replace(" ", "_") / "previous_seasons"
    output_dir.mkdir(parents=True, exist_ok=True)
    match_data_dir.mkdir(parents=True, exist_ok=True)
    previous_seasons_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine current season
    current_season_key = _get_current_season_key()
    
    # Parse seasons - default to current season if not provided
    if args.seasons:
        seasons = [int(s.strip()) for s in args.seasons.split(",")]
    else:
        # Default to current season only
        seasons = [current_season_key]
        print(f"  No --seasons provided, defaulting to current season: {current_season_key}")
    
    # State tracking file
    state_file = output_dir / ".pipeline_state.json"
    state: Dict[str, Any] = {}
    if args.resume and state_file.exists():
        with open(state_file, "r", encoding="utf-8") as f:
            state = json.load(f)
    
    scraper = TransfermarktScraper(ScraperConfig())
    
    # Load injury mappings
    injury_mappings = None
    mapping_file = PRODUCTION_ROOT / "config" / "injury_mappings.json"
    if mapping_file.exists():
        with open(mapping_file, 'r', encoding='utf-8') as f:
            injury_mappings = json.load(f)
        print(f"Loaded {len(injury_mappings)} injury type mappings")
    else:
        print(f"Warning: Injury mappings file not found: {mapping_file}")
        print(f"  Injury classification will use defaults")
    
    print(f"\n{'='*80}")
    print(f"League-Wide Transfermarkt Pipeline")
    print(f"{'='*80}")
    print(f"Country: {args.country}")
    print(f"League: {args.league}")
    print(f"Competition ID: {args.competition_id}")
    print(f"Competition Slug: {args.competition_slug}")
    print(f"Seasons: {seasons}")
    print(f"As of date: {as_of:%Y-%m-%d}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")
    
    # Step 1: Get all clubs per season
    print("Step 1: Fetching clubs from league standings...")
    all_clubs: Dict[int, Dict[str, Any]] = {}
    
    for season in seasons:
        print(f"  Fetching clubs for season {season-1}/{str(season)[-2:]}...", end=" ", flush=True)
        try:
            clubs = scraper.fetch_league_clubs(
                args.competition_slug, args.competition_id, season
            )
            print(f"Found {len(clubs)} clubs")
            
            for club in clubs:
                club_id = club["club_id"]
                if club_id not in all_clubs:
                    all_clubs[club_id] = {
                        "club_id": club_id,
                        "club_slug": club["club_slug"],
                        "club_name": club["club_name"],
                        "seasons": [],
                    }
                all_clubs[club_id]["seasons"].append(season)
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    unique_clubs = list(all_clubs.values())
    print(f"\nTotal unique clubs: {len(unique_clubs)}")
    
    if args.max_clubs:
        unique_clubs = unique_clubs[: args.max_clubs]
        print(f"Limited to {len(unique_clubs)} clubs for testing")
    
    # Step 2: Get players per club per season
    print(f"\nStep 2: Fetching players from squad pages...")
    master_players: Dict[int, Dict[str, Any]] = {}
    
    for club_idx, club in enumerate(unique_clubs, 1):
        club_id = club["club_id"]
        club_slug = club["club_slug"]
        club_name = club["club_name"]
        club_seasons = club["seasons"]
        
        print(f"\n[{club_idx}/{len(unique_clubs)}] {club_name} (ID: {club_id})")
        print(f"  Club {club_idx}/{len(unique_clubs)}")
        
        for season in club_seasons:
            print(f"  Season {season-1}/{str(season)[-2:]}...", end=" ", flush=True)
            try:
                squad_players = scraper.get_squad_players(
                    club_slug, club_id, season_id=season
                )
                print(f"Found {len(squad_players)} players")
                
                if args.max_players_per_club:
                    squad_players = squad_players[: args.max_players_per_club]
                
                for player in squad_players:
                    player_id = player["player_id"]
                    if player_id not in master_players:
                        master_players[player_id] = {
                            "player_id": player_id,
                            "player_slug": player.get("player_slug"),
                            "player_name": player["player_name"],
                            "clubs": [],
                            "seasons": set(),
                        }
                    master_players[player_id]["clubs"].append(
                        {"club_id": club_id, "club_name": club_name, "season": season}
                    )
                    master_players[player_id]["seasons"].add(season)
            except Exception as e:
                print(f"Error: {e}")
                continue
    
    # Convert seasons sets to sorted lists
    for player_id in master_players:
        master_players[player_id]["seasons"] = sorted(
            master_players[player_id]["seasons"], reverse=True
        )
    
    print(f"\nTotal unique players: {len(master_players)}")
    
    # Step 2.5: Compare with last update
    print(f"\nStep 2.5: Comparing with last update...")
    last_update_dir = get_latest_data_folder(args.country)
    existing_player_ids_from_last_update = set()

    if last_update_dir:
        last_profile_path = last_update_dir / "players_profile.csv"
        if last_profile_path.exists():
            try:
                last_profiles = pd.read_csv(last_profile_path, encoding="utf-8-sig", sep=';')
                if not last_profiles.empty and "id" in last_profiles.columns:
                    existing_player_ids_from_last_update = set(last_profiles["id"].unique())
                    print(f"  Found {len(existing_player_ids_from_last_update)} players in last update ({last_update_dir.name})")
                else:
                    print(f"  Last update folder exists but profile file is empty")
            except Exception as e:
                print(f"  Warning: Could not load last update profile file: {e}")
        else:
            print(f"  No profile file found in last update folder")
    else:
        print(f"  No previous update folder found - treating all players as new")

    # Identify new vs existing players
    current_season_player_ids = set(p["player_id"] for p in master_players.values())
    new_player_ids = current_season_player_ids - existing_player_ids_from_last_update
    removed_player_ids = existing_player_ids_from_last_update - current_season_player_ids

    print(f"  Current season players: {len(current_season_player_ids)}")
    print(f"  New players: {len(new_player_ids)}")
    print(f"  Removed players (not in current season): {len(removed_player_ids)}")
    
    # Step 3: Load existing CSVs from last update (if exists)
    print(f"\nStep 3: Loading existing data from last update...")
    profile_path = output_dir / "players_profile.csv"
    career_path = output_dir / "players_career.csv"
    injuries_path = output_dir / "injuries_data.csv"
    
    existing_profiles = pd.DataFrame()
    existing_careers = pd.DataFrame()
    existing_injuries = pd.DataFrame()

    if last_update_dir:
        last_profile_path = last_update_dir / "players_profile.csv"
        last_career_path = last_update_dir / "players_career.csv"
        last_injuries_path = last_update_dir / "injuries_data.csv"
        
        if last_profile_path.exists():
            try:
                existing_profiles = pd.read_csv(last_profile_path, encoding="utf-8-sig", sep=';')
                print(f"  Loaded {len(existing_profiles)} profiles from last update")
            except Exception as e:
                print(f"  Warning: Could not load profiles from last update: {e}")
        if last_career_path.exists():
            try:
                existing_careers = pd.read_csv(last_career_path, encoding="utf-8-sig", sep=';')
                print(f"  Loaded {len(existing_careers)} careers from last update")
            except Exception as e:
                print(f"  Warning: Could not load careers from last update: {e}")
        if last_injuries_path.exists():
            try:
                existing_injuries = pd.read_csv(last_injuries_path, encoding="utf-8-sig", sep=';')
                print(f"  Loaded {len(existing_injuries)} injuries from last update")
            except Exception as e:
                print(f"  Warning: Could not load injuries from last update: {e}")
    else:
        print(f"  No last update folder - starting fresh")
    
    # Step 4: Extract profile/career/injury data for ALL current season players
    print(f"\nStep 4: Extracting profile/career/injury data for all current season players...")
    print(f"  Re-fetching all data to capture latest changes (loans, transfers, new injuries)")

    updated_profiles = []
    updated_careers = []
    updated_injuries = []

    # Process all current season players
    all_players_to_process = list(master_players.values())
    
    # Also add removed players (not in current season but in last update)
    removed_players_to_process = []
    if removed_player_ids and not existing_profiles.empty and "id" in existing_profiles.columns:
        for player_id in removed_player_ids:
            player_row = existing_profiles[existing_profiles["id"] == player_id]
            if not player_row.empty:
                player_name = player_row.iloc[0].get("name", f"Player {player_id}")
                removed_player_info = {
                    "player_id": player_id,
                    "player_slug": None,  # We don't have slug for removed players
                    "player_name": player_name,
                    "clubs": [],
                    "seasons": set(),
                    "is_removed": True,
                }
                removed_players_to_process.append(removed_player_info)
                # Also add to master_players for season expansion and match data fetching
                master_players[player_id] = removed_player_info

    print(f"  Processing {len(all_players_to_process)} current season players...")
    if removed_players_to_process:
        print(f"  Also processing {len(removed_players_to_process)} removed players (not in current season)")
        all_players_to_process.extend(removed_players_to_process)

    for player_idx, player in enumerate(all_players_to_process, 1):
        player_id = player["player_id"]
        player_slug = player.get("player_slug")
        player_name = player["player_name"]
        is_new = player_id in new_player_ids
        is_removed = player.get("is_removed", False)
        
        status_label = "[NEW]" if is_new else ("[REMOVED]" if is_removed else "[EXISTING]")
        print(f"\n  [{player_idx}/{len(all_players_to_process)}] {player_name} (ID: {player_id}) {status_label}")
        
        # Check for existing daily features (edge case)
        existing_features = check_existing_daily_features(player_id, args.country)
        if existing_features and is_new:
            print(f"    ⚠️  Found existing daily features file: {existing_features}")
            print(f"    → Will update existing file instead of creating from scratch")
        
        try:
            # Always re-fetch profile (may have updated info)
            print(f"    Fetching profile...", end=" ", flush=True)
            profile_dict = scraper.fetch_player_profile(player_slug, player_id)
            profile_dict.setdefault("name", player_name)
            
            # Enrich with transfer data
            latest_transfer = scraper.get_latest_completed_transfer(player_id)
            if latest_transfer:
                transfer_details = latest_transfer.get("details", {})
                dest_id = latest_transfer.get("transferDestination", {}).get("clubId")
                dest_name = scraper.get_club_name(dest_id)
                if dest_name:
                    profile_dict["current_club"] = dest_name
                joined_date = transfer_details.get("date")
                if joined_date and not profile_dict.get("joined_on"):
                    profile_dict["joined_on"] = joined_date.split("T", 1)[0]
            
            updated_profiles.append(transform_profile(player_id, profile_dict))
            print("[OK]")
            
            # Always re-fetch career (loans, transfers may have changed)
            print(f"    Fetching career (this may take 20-60 seconds due to rate limiting)...", end=" ", flush=True)
            try:
                career_df = scraper.fetch_player_career(player_slug, player_id)
                updated_careers.append(transform_career(player_id, player_name, career_df))
                print(f"[OK] ({len(career_df)} transfers)")
            except KeyboardInterrupt:
                print(f"\n[WARNING] Interrupted during career fetch for {player_name}")
                raise
            except Exception as e:
                print(f"[ERROR] Error: {e}")
                import traceback
                traceback.print_exc()
                career_df = pd.DataFrame()
                updated_careers.append(transform_career(player_id, player_name, career_df))
            
            # Always re-fetch injuries (new injuries may have been added)
            print(f"    Fetching injuries...", end=" ", flush=True)
            injuries_df = scraper.fetch_player_injuries(player_slug, player_id)
            if not injuries_df.empty and not career_df.empty:
                injuries_df = _enrich_injuries_with_clubs(injuries_df, career_df)
            updated_injuries.append(transform_injuries(player_id, injuries_df, injury_mappings))
            print(f"[OK] ({len(injuries_df)} injuries)")
            
        except KeyboardInterrupt:
            print(f"\n[WARNING] Pipeline interrupted by user")
            raise
        except Exception as e:
            print(f"[ERROR] Error processing {player_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Merge with existing data and write
    print(f"\n  Writing consolidated files...")

    # Remove old entries for updated players, then add new ones
    if not existing_profiles.empty and "id" in existing_profiles.columns:
        updated_player_ids = {p["player_id"] for p in all_players_to_process}
        existing_profiles = existing_profiles[~existing_profiles["id"].isin(updated_player_ids)]
        
    if not existing_careers.empty and "id" in existing_careers.columns:
        updated_player_ids = {p["player_id"] for p in all_players_to_process}
        existing_careers = existing_careers[~existing_careers["id"].isin(updated_player_ids)]
        
    if not existing_injuries.empty and "id" in existing_injuries.columns:
        updated_player_ids = {p["player_id"] for p in all_players_to_process}
        existing_injuries = existing_injuries[~existing_injuries["id"].isin(updated_player_ids)]

    # Convert updated data to DataFrames
    if updated_profiles:
        updated_profiles_df = pd.concat(updated_profiles, ignore_index=True)
        all_profiles = pd.concat([existing_profiles, updated_profiles_df], ignore_index=True)
        all_profiles.to_csv(profile_path, index=False, encoding="utf-8-sig", sep=';')
        print(f"    [OK] Wrote {len(all_profiles)} profiles (updated {len(updated_profiles)} players, kept {len(existing_profiles)} from last update)")

    if updated_careers:
        updated_careers_df = pd.concat(updated_careers, ignore_index=True)
        all_careers = pd.concat([existing_careers, updated_careers_df], ignore_index=True)
        all_careers.to_csv(career_path, index=False, encoding="utf-8-sig", sep=';')
        print(f"    [OK] Wrote {len(all_careers)} careers (updated {len(updated_careers)} players, kept {len(existing_careers)} from last update)")

    if updated_injuries:
        updated_injuries_df = pd.concat(updated_injuries, ignore_index=True)
        
        # Ensure existing_injuries has the new columns (injury_class, body_part, severity)
        # If not, add them with default values
        new_columns = ['injury_class', 'body_part', 'severity']
        for col in new_columns:
            if col not in existing_injuries.columns:
                if col == 'severity':
                    # Calculate severity for existing injuries based on days
                    existing_injuries[col] = existing_injuries['days'].apply(calculate_severity_from_days)
                elif col == 'injury_class':
                    # Try to map from injury_type if mappings are available
                    existing_injuries[col] = existing_injuries['injury_type'].apply(
                        lambda x: injury_mappings.get(str(x), {}).get('injury_class', 'unknown') 
                        if pd.notna(x) and injury_mappings else 'unknown'
                    )
                elif col == 'body_part':
                    existing_injuries[col] = existing_injuries['injury_type'].apply(
                        lambda x: injury_mappings.get(str(x), {}).get('body_part', '') 
                        if pd.notna(x) and injury_mappings else ''
                    )
        
        all_injuries = pd.concat([existing_injuries, updated_injuries_df], ignore_index=True)
        all_injuries.to_csv(injuries_path, index=False, encoding="utf-8-sig", sep=';')
        print(f"    [OK] Wrote {len(all_injuries)} injuries (updated {len(updated_injuries)} players, kept {len(existing_injuries)} from last update)")
    
    # Step 4.5: Expand player seasons
    print(f"\nStep 4.5: Determining all available seasons for each player...")
    
    all_profiles_combined = pd.DataFrame()
    all_careers_combined = pd.DataFrame()
    
    if profile_path.exists():
        all_profiles_combined = pd.read_csv(profile_path, encoding="utf-8-sig", sep=';')
    if career_path.exists():
        all_careers_combined = pd.read_csv(career_path, encoding="utf-8-sig", sep=';')
    
    if all_profiles_combined.empty or all_careers_combined.empty:
        print(f"  Warning: Profile or career CSV not found. Using original seasons from squad pages.")
    else:
        print(f"  Loaded {len(all_profiles_combined)} profiles and {len(all_careers_combined)} careers")
    
    for player_id, player in master_players.items():
        player_name = player["player_name"]
        original_seasons = set(player["seasons"])
        
        profile_row = None
        career_rows = pd.DataFrame()
        
        if not all_profiles_combined.empty and "id" in all_profiles_combined.columns:
            profile_rows = all_profiles_combined[all_profiles_combined["id"] == player_id]
            if not profile_rows.empty:
                profile_row = profile_rows.iloc[0].to_dict()
        
        if not all_careers_combined.empty and "id" in all_careers_combined.columns:
            career_rows = all_careers_combined[all_careers_combined["id"] == player_id]
        
        profile_dict = None
        if profile_row:
            profile_dict = {
                "date_of_birth": profile_row.get("date_of_birth"),
            }
        
        try:
            all_seasons = _get_all_available_seasons(
                career_rows, profile_dict, as_of.year
            )
            all_seasons_set = set(all_seasons)
            player["seasons"] = sorted(list(all_seasons_set), reverse=True)
            
            if len(all_seasons_set) > len(original_seasons):
                added_seasons = sorted(list(all_seasons_set - original_seasons))
                print(f"  {player_name} (ID: {player_id}): Expanded from {len(original_seasons)} to {len(all_seasons_set)} seasons (added: {added_seasons})")
        except Exception as e:
            print(f"  Warning: Could not expand seasons for {player_name} (ID: {player_id}): {e}")
            continue
    
    print(f"  [OK] Updated seasons for all players")
    
    # Step 5: Extract match data (per player, per season)
    print(f"\nStep 5: Extracting match data (per player, per season)...")
    print(f"   Current season: {current_season_key} ({_season_label(current_season_key)})")
    print(f"   Current season files → {match_data_dir}")
    print(f"   Old season files → {previous_seasons_dir}")
    
    total_match_files = sum(len(p["seasons"]) for p in master_players.values())
    processed_files = 0
    current_season_files = 0
    old_season_files = 0
    
    for player_idx, player in enumerate(master_players.values(), 1):
        player_id = player["player_id"]
        player_slug = player.get("player_slug")
        player_name = player["player_name"]
        player_seasons = player["seasons"]
        
        print(
            f"\n[{player_idx}/{len(master_players)}] {player_name} (ID: {player_id})"
        )
        print(f"  Player {player_idx}/{len(master_players)}")
        print(f"  Seasons: {len(player_seasons)}")
        
        for season_idx, season in enumerate(player_seasons, 1):
            season_label = _season_label(season)
            filename = f"match_{player_id}_{season_label}.csv"
            
            # Determine destination: current season → date folder, old seasons → previous_seasons
            if season == current_season_key:
                match_file_path = match_data_dir / filename
                is_current_season = True
            else:
                match_file_path = previous_seasons_dir / filename
                is_current_season = False
            
            if args.resume and match_file_path.exists():
                dest_type = "current season" if is_current_season else "previous seasons"
                print(f"    Season {season_label} ({season_idx}/{len(player_seasons)}): Skipping (file exists in {dest_type})")
                processed_files += 1
                if is_current_season:
                    current_season_files += 1
                else:
                    old_season_files += 1
                continue
            
            print(f"    Season {season_label} ({season_idx}/{len(player_seasons)})...", end=" ", flush=True)
            
            try:
                raw_matches = scraper.fetch_player_match_log(
                    player_slug, player_id, season=season
                )
                
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
                
                stats_cols = [
                    "position",
                    "goals",
                    "assists",
                    "yellow_cards",
                    "red_cards",
                    "minutes_played",
                ]
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
                print(f"Error: {e}")
                continue
    
    print(f"\n{'='*80}")
    print(f"Pipeline completed!")
    print(f"  Clubs processed: {len(unique_clubs)}")
    print(f"  Players processed: {len(master_players)}")
    print(f"  Match files created: {processed_files}")
    print(f"    - Current season ({current_season_key}): {current_season_files}")
    print(f"    - Previous seasons: {old_season_files}")
    print(f"{'='*80}\n")
    
    scraper.close()


if __name__ == "__main__":
    import sys
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    else:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    
    print("Pipeline script starting...", flush=True)
    try:
        main()
        sys.exit(0)
    except KeyboardInterrupt:
        print(f"\n[WARNING] Pipeline interrupted by user. Exiting...", flush=True)
        sys.exit(130)
    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

