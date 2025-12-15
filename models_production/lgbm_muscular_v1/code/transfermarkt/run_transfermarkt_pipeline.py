#!/usr/bin/env python3
"""
Scrape Transfermarkt data for a single club and export tables following the
reference schema.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.data_collection.transfermarkt_scraper import (
    ScraperConfig,
    TransfermarktScraper,
    fetch_multiple_match_logs,
)
from scripts.data_collection.transformers import (
    transform_career,
    transform_competitions,
    transform_injuries,
    transform_matches,
    transform_profile,
    transform_teams,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--club-slug", required=True, help="Transfermarkt slug, e.g. 'sl-benfica'")
    parser.add_argument("--club-id", required=True, type=int, help="Transfermarkt club ID, e.g. 294")
    parser.add_argument("--club-name", required=True, help="Club name for logging")
    parser.add_argument(
        "--as-of-date",
        default="2025-11-09",
        help="ISO date – rows beyond this date are discarded.",
    )
    parser.add_argument(
        "--output-root",
        default="data_exports/transfermarkt",
        help="Directory to write dataset CSV files into a timestamped subfolder.",
    )
    parser.add_argument(
        "--player-manifest",
        help="Optional JSON file with [{'player_id', 'player_slug', 'player_name'}]. "
        "If omitted the current squad page is used.",
    )
    parser.add_argument(
        "--seasons-back",
        type=int,
        default=6,
        help="Number of historical seasons to fetch match logs for.",
    )
    parser.add_argument("--max-players", type=int, help="Cap number of players (debug/testing).")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    as_of = datetime.fromisoformat(args.as_of_date)
    output_dir = (
        Path(args.output_root)
        / args.club_name.replace(" ", "_").lower()
        / as_of.strftime("%Y%m%d")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    scraper = TransfermarktScraper(ScraperConfig())
    players = _load_player_manifest(args, scraper)
    if args.max_players:
        players = players[: args.max_players]

    profiles, careers, injuries, matches = [], [], [], []
    
    total_players = len(players)
    print(f"\n{'='*60}")
    print(f"Starting data collection for {total_players} players")
    print(f"{'='*60}\n")

    for idx, player in enumerate(players, 1):
        player_id = player["player_id"]
        player_slug = player.get("player_slug")
        player_name = player["player_name"]
        
        print(f"[{idx}/{total_players}] Processing: {player_name} (ID: {player_id})")
        print(f"  Fetching profile...", end=" ", flush=True)
        player_id = player["player_id"]
        player_slug = player.get("player_slug")
        player_name = player["player_name"]
        profile_dict = scraper.fetch_player_profile(player_slug, player_id)
        profile_dict.setdefault("name", player_name)
        print("✓")
        
        print(f"  Fetching career...", end=" ", flush=True)
        career_df = scraper.fetch_player_career(player_slug, player_id)
        print(f"✓ ({len(career_df)} transfers)")

        # Enrich signed_from/joined_on using official transfer history API
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
        
        profiles.append(transform_profile(player_id, profile_dict))
        careers.append(transform_career(player_id, player_name, career_df))

        print(f"  Fetching injuries...", end=" ", flush=True)
        injuries_df = scraper.fetch_player_injuries(player_slug, player_id)
        # Enrich injuries with club data from career history if not already present
        if not injuries_df.empty and not career_df.empty and "Club" not in injuries_df.columns:
            injuries_df = _enrich_injuries_with_clubs(injuries_df, career_df)
        injuries.append(transform_injuries(player_id, injuries_df))
        print(f"✓ ({len(injuries_df)} injuries)")

        # Determine all available seasons for this player
        # Extract from career data and also try a wide range to catch all matches
        available_seasons = _get_all_available_seasons(career_df, profile_dict, as_of.year)
        print(f"  Fetching match data ({len(available_seasons)} seasons)...", end=" ", flush=True)
        match_df = fetch_multiple_match_logs(scraper, player_slug, player_id, available_seasons)
        transformed_matches = transform_matches(player_id, player_name, match_df)
        filtered_matches = transformed_matches[
            transformed_matches["date"].isna()
            | (transformed_matches["date"] <= pd.Timestamp(as_of))
        ]
        matches.append(filtered_matches)
        # Show stats extraction status
        stats_count = filtered_matches["position"].notna().sum() if "position" in filtered_matches.columns else 0
        goals_count = filtered_matches["goals"].notna().sum() if "goals" in filtered_matches.columns else 0
        print(f"✓ ({len(filtered_matches)} matches, {stats_count} with position, {goals_count} with goals)")
        print()  # Blank line between players

    profile_table = pd.concat(profiles, ignore_index=True) if profiles else pd.DataFrame()
    career_table = pd.concat(careers, ignore_index=True) if careers else pd.DataFrame()
    injury_table = pd.concat(injuries, ignore_index=True) if injuries else pd.DataFrame()
    match_table = pd.concat(matches, ignore_index=True) if matches else pd.DataFrame()

    teams_table = transform_teams(_extract_team_rows(match_table, career_table, injury_table))
    competitions_table = transform_competitions(
        [{"competition": c} for c in sorted(match_table["competition"].dropna().unique())]
    )

    exports = {
        "players_profile": profile_table,
        "players_career": career_table,
        "injuries_data": injury_table,
        "match_data": match_table,
        "teams_data": teams_table,
        "competition_data": competitions_table,
    }
    print(f"\n{'='*60}")
    print("Exporting data files...")
    print(f"{'='*60}\n")
    
    for name, df in exports.items():
        path = output_dir / f"{as_of.strftime('%Y%m%d')}_{name}.csv"
        print(f"  Writing {name}...", end=" ", flush=True)
        # Use UTF-8 with BOM for Excel compatibility on Windows
        df_to_write = df.copy()
        
        # For injuries data, ensure lost_games is written as integers (no decimals)
        if name == "injuries_data" and "lost_games" in df_to_write.columns:
            # Convert to integers and format for CSV to avoid decimal display in Excel
            # Convert Int64 to regular integers, writing empty string for NaN
            lost_games_series = df_to_write["lost_games"]
            # Create a formatted version: integers as strings, NaN as empty
            formatted_lost_games = lost_games_series.apply(
                lambda x: "" if pd.isna(x) else str(int(float(x))) if pd.notna(x) else ""
            )
            df_to_write["lost_games"] = formatted_lost_games
        
        # Write CSV with special handling for lost_games to ensure integer format
        if name == "injuries_data" and "lost_games" in df_to_write.columns:
            # Write manually to ensure integers are formatted correctly
            with open(path, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.writer(f)
                # Write header
                writer.writerow(df_to_write.columns.tolist())
                # Write rows, ensuring lost_games is written as integer string
                for _, row in df_to_write.iterrows():
                    row_values = []
                    for col in df_to_write.columns:
                        val = row[col]
                        if col == "lost_games":
                            # Write as integer string (no decimals)
                            if pd.isna(val) or val == "":
                                row_values.append("")
                            else:
                                try:
                                    row_values.append(str(int(float(val))))
                                except (ValueError, TypeError):
                                    row_values.append("")
                        else:
                            # Write other columns normally
                            if pd.isna(val):
                                row_values.append("")
                            else:
                                row_values.append(val)
                    writer.writerow(row_values)
        else:
            df_to_write.to_csv(path, index=False, encoding="utf-8-sig", na_rep="")
        print(f"Wrote {path} ({len(df_to_write):,} rows)")
    
    # Close the scraper session to allow clean exit
    scraper.close()


def _enrich_injuries_with_clubs(injuries_df: pd.DataFrame, career_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich injuries DataFrame with club information from career history.
    
    For each injury, finds the club the player was at during the injury period
    based on the career transfer history.
    """
    if injuries_df.empty or career_df.empty:
        return injuries_df
    
    # Ensure we have date columns
    if "from" not in injuries_df.columns and "From" not in injuries_df.columns:
        return injuries_df
    
    from_col = "from" if "from" in injuries_df.columns else "From"
    
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
    
    injuries_df["Club"] = clubs_list
    return injuries_df


def _load_player_manifest(
    args: argparse.Namespace,
    scraper: TransfermarktScraper,
) -> List[Dict[str, str]]:
    if args.player_manifest:
        with open(args.player_manifest, "r", encoding="utf-8") as fh:
            return json.load(fh)
    # Extract season ID from as_of_date (e.g., 2025-11-09 -> 2025)
    as_of = datetime.fromisoformat(args.as_of_date)
    season_id = as_of.year
    players = scraper.get_squad_players(
        args.club_slug, 
        args.club_id, 
        path="kader", 
        season_id=season_id
    )
    if not players:
        raise RuntimeError("Could not detect players from squad page; supply --player-manifest.")
    return players


def _season_range(latest_year: int, seasons_back: int) -> List[int]:
    return list(range(latest_year, latest_year - seasons_back, -1))


def _get_all_available_seasons(
    career_df: pd.DataFrame,
    profile_dict: Dict[str, Any],
    current_year: int,
) -> List[int]:
    """
    Determine all available seasons for a player to fetch match data.
    
    Uses career data to find seasons, and also includes a range from
    player's debut year to current year to ensure we don't miss any.
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
    date_of_birth = profile_dict.get("date_of_birth")
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


def _extract_team_rows(*frames: pd.DataFrame) -> List[Dict[str, str]]:
    teams = set()
    for frame in frames:
        if frame is None or frame.empty:
            continue
        for column in frame.columns:
            if "team" in column:
                teams.update(frame[column].dropna().tolist())
    return [{"team": t, "country": None} for t in sorted(teams)]


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted by user.")

