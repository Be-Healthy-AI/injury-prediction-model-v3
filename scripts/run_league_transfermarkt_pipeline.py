#!/usr/bin/env python3
"""
Scrape Transfermarkt data for an entire league and export tables following the
reference schema. Processes all clubs in a league across multiple seasons.

Output structure:
- {country}/{as_of_date}/players_profile.csv (consolidated)
- {country}/{as_of_date}/players_career.csv (consolidated)
- {country}/{as_of_date}/injuries_data.csv (consolidated)
- {country}/{as_of_date}/match_data/match_<player_id>_<season-1>_<season>.csv (one per player/season)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.data_collection.transfermarkt_scraper import (
    ScraperConfig,
    TransfermarktScraper,
)
from scripts.data_collection.transformers import (
    transform_career,
    transform_injuries,
    transform_matches,
    transform_profile,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--country",
        required=True,
        help="Country name (e.g., 'England') - used for output folder",
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
        required=True,
        help="Comma-separated list of season years (e.g., '2022,2023,2024,2025')",
    )
    parser.add_argument(
        "--as-of-date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="ISO date – rows beyond this date are discarded. Defaults to today.",
    )
    parser.add_argument(
        "--output-root",
        default="data_exports/transfermarkt",
        help="Root directory for output files.",
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


def _enrich_injuries_with_clubs(injuries_df: pd.DataFrame, career_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich injuries DataFrame with club information from career history.
    
    For each injury, finds the club the player was at during the injury period
    based on the career transfer history.
    """
    if injuries_df.empty or career_df.empty:
        return injuries_df
    
    # Ensure we have date columns - check for both raw and transformed column names
    from_col = None
    for col_name in ["from", "From", "fromDate"]:
        if col_name in injuries_df.columns:
            from_col = col_name
            break
    
    if not from_col:
        return injuries_df
    
    # Parse dates in career DataFrame
    career_with_dates = career_df.copy()
    if "Date" in career_with_dates.columns:
        career_with_dates["Date"] = pd.to_datetime(career_with_dates["Date"], errors="coerce")
        # Sort by date descending (most recent first)
        career_with_dates = career_with_dates.sort_values("Date", ascending=False)
    
    # Parse injury dates
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
        
        # Find the most recent transfer before or on the injury date
        club_name = None
        if "Date" in career_with_dates.columns and "To" in career_with_dates.columns:
            # Find transfers where Date <= injury_date
            valid_transfers = career_with_dates[
                career_with_dates["Date"].notna() & 
                (career_with_dates["Date"] <= injury_date)
            ]
            if not valid_transfers.empty:
                # Get the most recent transfer (first row after sorting)
                club_name = valid_transfers.iloc[0].get("To")
        
        clubs_list.append(club_name)
    
    # Update the Club column (for raw data) or clubs column (for transformed data)
    if "Club" in injuries_df.columns:
        injuries_df["Club"] = clubs_list
    elif "clubs" in injuries_df.columns:
        injuries_df["clubs"] = clubs_list
    else:
        # Add Club column for raw data (will be renamed to clubs in transformer)
        injuries_df["Club"] = clubs_list
    
    return injuries_df


def _get_seasons_from_squad_pages(
    scraper: TransfermarktScraper,
    player_id: int,
    player_slug: Optional[str],
    clubs_seasons: List[Dict[str, Any]],
) -> Set[int]:
    """
    Determine which seasons a player participated in based on squad pages.
    This is the primary source for league participation.
    """
    seasons_set = set()
    
    for club_info in clubs_seasons:
        club_id = club_info["club_id"]
        club_slug = club_info["club_slug"]
        season = club_info["season"]
        
        # Check if player was in this club's squad for this season
        try:
            squad_players = scraper.get_squad_players(
                club_slug, club_id, season_id=season
            )
            player_ids = {p["player_id"] for p in squad_players}
            if player_id in player_ids:
                seasons_set.add(season)
        except Exception as e:
            # Log but continue - squad page might not be accessible
            print(f"      Warning: Could not check squad for club {club_id}, season {season}: {e}")
    
    return seasons_set


def _get_all_available_seasons(
    career_df: pd.DataFrame,
    profile_dict: Dict[str, Any],
    current_year: int,
) -> List[int]:
    """
    Determine all available seasons for a player to fetch match data.
    
    Uses career data to find seasons, and also includes a range from
    player's potential debut year to current year to ensure we don't miss any.
    """
    seasons_set = set()
    
    # Extract seasons from career data
    if not career_df.empty and "Season" in career_df.columns:
        for season_str in career_df["Season"].dropna().unique():
            # Parse season strings like "23/24" or "2023-2024"
            if isinstance(season_str, str):
                # Handle "23/24" format
                if "/" in season_str:
                    try:
                        parts = season_str.split("/")
                        if len(parts) == 2:
                            year1 = int(parts[0])
                            year2 = int(parts[1])
                            # Convert 2-digit years (e.g., "23" -> 2023)
                            if year1 < 100:
                                year1 = 2000 + year1 if year1 < 50 else 1900 + year1
                            if year2 < 100:
                                year2 = 2000 + year2 if year2 < 50 else 1900 + year2
                            # Add both years (season spans two calendar years)
                            seasons_set.add(year1)
                            seasons_set.add(year2)
                    except (ValueError, IndexError):
                        pass
                # Handle "2023-2024" format
                elif "-" in season_str:
                    try:
                        year = int(season_str.split("-")[0])
                        seasons_set.add(year)
                        seasons_set.add(year + 1)
                    except (ValueError, IndexError):
                        pass
    
    # Also add a range from player's potential debut to current year
    # Use date_of_birth to estimate debut year (typically 16-18 years old)
    date_of_birth = profile_dict.get("date_of_birth") if profile_dict else None
    if date_of_birth:
        try:
            if isinstance(date_of_birth, str):
                birth_year = pd.to_datetime(date_of_birth).year
            else:
                birth_year = date_of_birth.year if hasattr(date_of_birth, 'year') else None
            if birth_year:
                # Assume player could start playing at age 16
                debut_year = birth_year + 16
                # Add all seasons from debut to current year
                for year in range(debut_year, current_year + 1):
                    seasons_set.add(year)
        except (ValueError, AttributeError, TypeError):
            pass
    
    # If we still don't have seasons, use a wide range (last 30 years)
    if not seasons_set:
        for year in range(current_year, current_year - 30, -1):
            seasons_set.add(year)
    
    # Return sorted list (most recent first)
    return sorted(seasons_set, reverse=True)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    as_of = datetime.fromisoformat(args.as_of_date)
    
    # Parse seasons
    seasons = [int(s.strip()) for s in args.seasons.split(",")]
    
    # Setup output directory
    output_dir = (
        Path(args.output_root)
        / args.country.lower().replace(" ", "_")
        / as_of.strftime("%Y%m%d")
    )
    match_data_dir = output_dir / "match_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    match_data_dir.mkdir(parents=True, exist_ok=True)
    
    # State tracking file
    state_file = output_dir / ".pipeline_state.json"
    state: Dict[str, Any] = {}
    if args.resume and state_file.exists():
        with open(state_file, "r", encoding="utf-8") as f:
            state = json.load(f)
    
    scraper = TransfermarktScraper(ScraperConfig())
    
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
    all_clubs: Dict[int, Dict[str, Any]] = {}  # club_id -> club_info
    
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
    master_players: Dict[int, Dict[str, Any]] = {}  # player_id -> player_info
    
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
    
    # Step 3: Load existing CSVs if they exist
    print(f"\nStep 3: Loading existing data files...")
    profile_path = output_dir / "players_profile.csv"
    career_path = output_dir / "players_career.csv"
    injuries_path = output_dir / "injuries_data.csv"
    
    existing_profiles = pd.DataFrame()
    existing_careers = pd.DataFrame()
    existing_injuries = pd.DataFrame()
    
    if profile_path.exists():
        existing_profiles = pd.read_csv(profile_path, encoding="utf-8-sig")
        print(f"  Loaded {len(existing_profiles)} existing profiles")
    if career_path.exists():
        existing_careers = pd.read_csv(career_path, encoding="utf-8-sig")
        print(f"  Loaded {len(existing_careers)} existing careers")
    if injuries_path.exists():
        existing_injuries = pd.read_csv(injuries_path, encoding="utf-8-sig")
        print(f"  Loaded {len(existing_injuries)} existing injuries")
    
    existing_player_ids = set()
    if not existing_profiles.empty and "id" in existing_profiles.columns:
        existing_player_ids.update(existing_profiles["id"].unique())
    if not existing_careers.empty and "id" in existing_careers.columns:
        existing_player_ids.update(existing_careers["id"].unique())
    
    # Step 4: Extract profile/career/injury data (once per player)
    print(f"\nStep 4: Extracting profile/career/injury data...")
    new_profiles = []
    new_careers = []
    new_injuries = []
    
    players_to_process = [
        p for p in master_players.values() if p["player_id"] not in existing_player_ids
    ]
    
    if not players_to_process:
        print(f"  All {len(master_players)} players already exist in existing files")
        print(f"  Skipping profile/career/injury extraction and file writing")
    else:
        print(f"  Processing {len(players_to_process)} new players...")
        
        for player_idx, player in enumerate(players_to_process, 1):
            player_id = player["player_id"]
            player_slug = player.get("player_slug")
            player_name = player["player_name"]
            
            print(
                f"  [{player_idx}/{len(players_to_process)}] {player_name} (ID: {player_id})"
            )
            
            try:
                # Fetch profile
                print(f"    Fetching profile...", end=" ", flush=True)
                profile_dict = scraper.fetch_player_profile(player_slug, player_id)
                profile_dict.setdefault("name", player_name)
                
                # Enrich with transfer data
                latest_transfer = scraper.get_latest_completed_transfer(player_id)
                if latest_transfer:
                    transfer_details = latest_transfer.get("details", {})
                    # Use transferDestination (club player joined) for current_club, not transferSource (club player left)
                    dest_id = latest_transfer.get("transferDestination", {}).get("clubId")
                    dest_name = scraper.get_club_name(dest_id)
                    if dest_name:
                        profile_dict["current_club"] = dest_name
                    joined_date = transfer_details.get("date")
                    if joined_date and not profile_dict.get("joined_on"):
                        profile_dict["joined_on"] = joined_date.split("T", 1)[0]
                
                new_profiles.append(transform_profile(player_id, profile_dict))
                print("[OK]")
                
                # Fetch career
                print(f"    Fetching career (this may take 20-60 seconds due to rate limiting)...", end=" ", flush=True)
                try:
                    # Note: This can take 20-60 seconds due to rate limiting (1 sec per API call)
                    # Each transfer requires 2 club name lookups, so a player with 10 transfers = 20+ seconds
                    career_df = scraper.fetch_player_career(player_slug, player_id)
                    new_careers.append(transform_career(player_id, player_name, career_df))
                    print(f"[OK] ({len(career_df)} transfers)")
                except KeyboardInterrupt:
                    print(f"\n[WARNING] Interrupted during career fetch for {player_name}")
                    raise  # Re-raise to exit gracefully
                except Exception as e:
                    print(f"[ERROR] Error: {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue with next player even if career fetch fails
                    career_df = pd.DataFrame()
                    new_careers.append(transform_career(player_id, player_name, career_df))
                
                # Fetch injuries
                print(f"    Fetching injuries...", end=" ", flush=True)
                injuries_df = scraper.fetch_player_injuries(player_slug, player_id)
                # Enrich injuries with club data from career history
                if not injuries_df.empty and not career_df.empty:
                    injuries_df = _enrich_injuries_with_clubs(injuries_df, career_df)
                new_injuries.append(transform_injuries(player_id, injuries_df))
                print(f"[OK] ({len(injuries_df)} injuries)")
                
            except KeyboardInterrupt:
                print(f"\n[WARNING] Pipeline interrupted by user")
                raise  # Re-raise to exit gracefully
            except Exception as e:
                print(f"[ERROR] Error processing {player_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Combine with existing data and write files (only if we processed new players)
        if new_profiles or new_careers or new_injuries:
            print(f"\n  Writing consolidated files...", flush=True)
            
            if new_profiles:
                print(f"    Combining {len(existing_profiles)} existing + {len(new_profiles)} new profiles...", flush=True)
                all_profiles = pd.concat(
                    [existing_profiles] + new_profiles, ignore_index=True
                )
                print(f"    Writing {len(all_profiles)} profiles to {profile_path.name}...", flush=True)
                all_profiles.to_csv(profile_path, index=False, encoding="utf-8-sig")
                print(f"    [OK] Wrote {len(all_profiles)} profiles to {profile_path.name}")
            
            if new_careers:
                print(f"    Combining {len(existing_careers)} existing + {len(new_careers)} new careers...", flush=True)
                all_careers = pd.concat([existing_careers] + new_careers, ignore_index=True)
                print(f"    Writing {len(all_careers)} careers to {career_path.name}...", flush=True)
                all_careers.to_csv(career_path, index=False, encoding="utf-8-sig")
                print(f"    [OK] Wrote {len(all_careers)} careers to {career_path.name}")
            
            if new_injuries:
                print(f"    Combining {len(existing_injuries)} existing + {len(new_injuries)} new injuries...", flush=True)
                all_injuries = pd.concat(
                    [existing_injuries] + new_injuries, ignore_index=True
                )
                print(f"    Writing {len(all_injuries)} injuries to {injuries_path.name}...", flush=True)
                all_injuries.to_csv(injuries_path, index=False, encoding="utf-8-sig")
                print(f"    [OK] Wrote {len(all_injuries)} injuries to {injuries_path.name}")
    
    # Step 4.5: Expand player seasons to include ALL available seasons
    # (not just the seasons they were found in Premier League squads)
    print(f"\nStep 4.5: Determining all available seasons for each player...")
    
    # Load consolidated profile and career data (either just written or already existing)
    all_profiles_combined = pd.DataFrame()
    all_careers_combined = pd.DataFrame()
    
    if profile_path.exists():
        all_profiles_combined = pd.read_csv(profile_path, encoding="utf-8-sig")
    if career_path.exists():
        all_careers_combined = pd.read_csv(career_path, encoding="utf-8-sig")
    
    # If we just wrote files, they should exist. If not, log a warning but continue.
    if all_profiles_combined.empty or all_careers_combined.empty:
        print(f"  Warning: Profile or career CSV not found. Using original seasons from squad pages.")
    else:
        print(f"  Loaded {len(all_profiles_combined)} profiles and {len(all_careers_combined)} careers")
    
    # Expand seasons for each player
    for player_id, player in master_players.items():
        player_name = player["player_name"]
        original_seasons = set(player["seasons"])
        
        # Get profile and career for this player
        profile_row = None
        career_rows = pd.DataFrame()
        
        if not all_profiles_combined.empty and "id" in all_profiles_combined.columns:
            profile_rows = all_profiles_combined[all_profiles_combined["id"] == player_id]
            if not profile_rows.empty:
                profile_row = profile_rows.iloc[0].to_dict()
        
        if not all_careers_combined.empty and "id" in all_careers_combined.columns:
            career_rows = all_careers_combined[all_careers_combined["id"] == player_id]
        
        # Convert profile row to dict format expected by _get_all_available_seasons
        profile_dict = None
        if profile_row:
            profile_dict = {
                "date_of_birth": profile_row.get("date_of_birth"),
                # Add other fields if needed, but date_of_birth is the main one used
            }
        
        # Determine all available seasons
        try:
            all_seasons = _get_all_available_seasons(
                career_rows, profile_dict, as_of.year
            )
            
            # Update player's seasons (union of original and all available)
            all_seasons_set = set(all_seasons)
            player["seasons"] = sorted(list(all_seasons_set), reverse=True)
            
            if len(all_seasons_set) > len(original_seasons):
                added_seasons = sorted(list(all_seasons_set - original_seasons))
                print(f"  {player_name} (ID: {player_id}): Expanded from {len(original_seasons)} to {len(all_seasons_set)} seasons (added: {added_seasons})")
        except Exception as e:
            # If expansion fails, keep original seasons (safe fallback)
            print(f"  Warning: Could not expand seasons for {player_name} (ID: {player_id}): {e}")
            print(f"    Using original seasons: {sorted(list(original_seasons))}")
            continue
    
    print(f"  [OK] Updated seasons for all players")
    
    # Step 5: Extract match data (per player, per season)
    print(f"\nStep 5: Extracting match data (per player, per season)...")
    
    total_match_files = sum(len(p["seasons"]) for p in master_players.values())
    processed_files = 0
    
    for player_idx, player in enumerate(master_players.values(), 1):
        player_id = player["player_id"]
        player_slug = player.get("player_slug")
        player_name = player["player_name"]
        player_seasons = player["seasons"]
        clubs_seasons = player["clubs"]
        
        print(
            f"\n[{player_idx}/{len(master_players)}] {player_name} (ID: {player_id})"
        )
        print(f"  Player {player_idx}/{len(master_players)}")
        print(f"  Seasons: {len(player_seasons)}")
        
        for season_idx, season in enumerate(player_seasons, 1):
            season_label = _season_label(season)
            filename = f"match_{player_id}_{season_label}.csv"
            match_file_path = match_data_dir / filename
            
            # Skip if file already exists
            if args.resume and match_file_path.exists():
                print(f"    Season {season_label} ({season_idx}/{len(player_seasons)}): Skipping (file exists)")
                processed_files += 1
                continue
            
            print(f"    Season {season_label} ({season_idx}/{len(player_seasons)})...", end=" ", flush=True)
            
            try:
                # Fetch match log for this season
                raw_matches = scraper.fetch_player_match_log(
                    player_slug, player_id, season=season
                )
                
                if raw_matches.empty:
                    print("No matches")
                    processed_files += 1
                    continue
                
                # Transform and filter
                transformed = transform_matches(player_id, player_name, raw_matches)
                filtered = transformed[
                    transformed["date"].isna()
                    | (transformed["date"] <= pd.Timestamp(as_of))
                ]
                
                if filtered.empty:
                    print("No matches after date filter")
                    processed_files += 1
                    continue
                
                # Write to file
                filtered.to_csv(match_file_path, index=False, encoding="utf-8-sig", na_rep="")
                
                # Show stats
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
                print(f"[OK] {len(filtered)} matches ({stats_str})")
                processed_files += 1
                
            except Exception as e:
                print(f"Error: {e}")
                continue
    
    print(f"\n{'='*80}")
    print(f"Pipeline completed!")
    print(f"  Clubs processed: {len(unique_clubs)}")
    print(f"  Players processed: {len(master_players)}")
    print(f"  Match files created: {processed_files}")
    print(f"{'='*80}\n")
    
    # Close the scraper session to allow clean exit
    scraper.close()


if __name__ == "__main__":
    import sys
    import io
    # Force UTF-8 encoding for Windows compatibility (especially when redirecting output)
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    else:
        # Force unbuffered output
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    
    print("Pipeline script starting...", flush=True)
    try:
        main()
        # Only exit successfully if main() completes without exception
        sys.exit(0)
    except KeyboardInterrupt:
        print(f"\n[WARNING] Pipeline interrupted by user. Exiting...", flush=True)
        sys.exit(130)  # Standard exit code for SIGINT/KeyboardInterrupt
    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

