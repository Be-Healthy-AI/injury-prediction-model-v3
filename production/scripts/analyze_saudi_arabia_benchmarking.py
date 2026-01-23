#!/usr/bin/env python3
"""
Al-Ahli SFC Benchmarking Analysis
Compares Al-Ahli SFC with other Saudi Pro League clubs across multiple KPIs.

KPIs:
- Injured vs Non-Injured Player/Weeks (absolute and percentage)
- Average Age per Club per Season
- Total Market Value per Club per Season
- Squad Size
- Injury Frequency
- Average Injury Duration
- Injury Severity Distribution

Output: CSV files and summary report in saudi_arabia/YYYYMMDD_Al-Ahli SFC Benchmarking/
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import warnings
import sys

# Add root directory to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent
ROOT_DIR = PRODUCTION_ROOT.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.data_collection.transfermarkt_scraper import (
    ScraperConfig,
    TransfermarktScraper,
)

warnings.filterwarnings('ignore')

# Paths are already calculated above

# Data directory - use latest folder
def get_latest_data_folder(country: str) -> Optional[Path]:
    """Get the latest data folder for a country."""
    country_dir = PRODUCTION_ROOT / "raw_data" / country.lower().replace(" ", "_")
    if not country_dir.exists():
        return None
    
    # Find all date folders (YYYYMMDD format)
    date_folders = []
    for folder in country_dir.iterdir():
        if folder.is_dir() and folder.name.isdigit() and len(folder.name) == 8:
            try:
                datetime.strptime(folder.name, "%Y%m%d")
                date_folders.append(folder)
            except ValueError:
                continue
    
    if not date_folders:
        return None
    
    # Return the most recent one
    return max(date_folders, key=lambda x: x.name)

# Get latest data folder
DATA_DIR = get_latest_data_folder("Saudi Arabia")
if DATA_DIR is None:
    raise ValueError("No data folder found for Saudi Arabia. Please run fetch_raw_data.py first.")

OUTPUT_DIR = DATA_DIR.parent / f"{DATA_DIR.name}_Al-Ahli SFC Benchmarking"

# Season definitions
SEASONS = {
    "2024/25": {
        "start": pd.Timestamp("2024-07-01"),
        "end": pd.Timestamp("2025-06-30"),
        "year": 2024,
        "league": "Saudi Pro League",
        "competition_slug": "saudi-pro-league",
        "competition_id": "SA1"
    },
    "2025/26": {
        "start": pd.Timestamp("2025-07-01"),
        "end": pd.Timestamp("2026-06-30"),
        "year": 2025,
        "league": "Saudi Pro League",
        "competition_slug": "saudi-pro-league",
        "competition_id": "SA1"
    }
}

# Analysis start date for injuries (all injuries since this date)
INJURY_ANALYSIS_START = pd.Timestamp("2025-07-01")

# Note: Club lists are fetched dynamically from Transfermarkt, so no hardcoded lists needed


def load_all_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load all CSV files from the data directory."""
    print("Loading data files...")
    
    data = {}
    
    # Load profiles (optional)
    profiles_path = data_dir / "players_profile.csv"
    if profiles_path.exists():
        data['profiles'] = pd.read_csv(profiles_path, sep=';', encoding='utf-8-sig')
        print(f"  Loaded {len(data['profiles'])} profiles")
    else:
        print(f"  Warning: Profiles file not found: {profiles_path}")
        print(f"  Creating empty profiles DataFrame...")
        data['profiles'] = pd.DataFrame(columns=['id', 'name', 'position', 'date_of_birth', 'current_club', 'joined_on'])
    
    # Load injuries
    injuries_path = data_dir / "injuries_data.csv"
    if injuries_path.exists():
        data['injuries'] = pd.read_csv(injuries_path, sep=';', encoding='utf-8-sig')
        print(f"  Loaded {len(data['injuries'])} injury records")
    else:
        print(f"  Warning: Injuries file not found: {injuries_path}")
        data['injuries'] = pd.DataFrame()
    
    # Load career (optional)
    career_path = data_dir / "players_career.csv"
    if career_path.exists():
        data['career'] = pd.read_csv(career_path, sep=';', encoding='utf-8-sig')
        print(f"  Loaded {len(data['career'])} career records")
    else:
        print(f"  Warning: Career file not found: {career_path}")
        print(f"  Creating empty career DataFrame...")
        data['career'] = pd.DataFrame(columns=['id', 'Date', 'From', 'To', 'VM'])
    
    return data


def parse_season_string(season_str: str) -> Optional[int]:
    """Parse season string like '24/25' or '2024/25' to year (2024)."""
    if pd.isna(season_str):
        return None
    
    season_str = str(season_str).strip()
    if '/' in season_str:
        parts = season_str.split('/')
        if len(parts) == 2:
            year1 = parts[0]
            if len(year1) == 2:
                year1 = '20' + year1
            try:
                return int(year1)
            except ValueError:
                return None
    return None


def fetch_players_from_transfermarkt(
    season_name: str,
    season_info: Dict,
    scraper: TransfermarktScraper
) -> Dict[int, str]:
    """
    Fetch all players and their clubs directly from Transfermarkt for a given season.
    Returns: {player_id: club_name}
    """
    competition_slug = season_info['competition_slug']
    competition_id = season_info['competition_id']
    season_year = season_info['year']
    
    print(f"Fetching clubs from Transfermarkt for {season_name}...")
    clubs = scraper.fetch_league_clubs(competition_slug, competition_id, season_year)
    print(f"  Found {len(clubs)} clubs")
    
    player_club_map = {}
    club_player_counts = {}  # Track player counts per club
    player_original_club = {}  # Track which club each player was originally fetched from
    
    for club_idx, club in enumerate(clubs, 1):
        club_name = club['club_name']
        club_id = club['club_id']
        club_slug = club['club_slug']
        
        # Improved exclusion logic - only exclude clubs that end with " B" or are specific B teams
        # Don't exclude clubs that just contain " B" in the middle of their name
        should_exclude = False
        
        # Check if club ends with " B" or " B."
        if club_name.endswith(' B') or club_name.endswith(' B.'):
            should_exclude = True
        
        # Youth/U19/U23 teams
        if ' U19' in club_name or ' U23' in club_name or ' U21' in club_name:
            should_exclude = True
        
        # Other reserve/youth team indicators
        exclude_patterns = [
            'Promesas', 'Vetusta', 'Mirandilla', 'Castilla', 'Juvenil', 
            'Youth', 'Reserve'
        ]
        for pattern in exclude_patterns:
            if pattern in club_name:
                should_exclude = True
                break
        
        if should_exclude:
            continue
        
        print(f"  [{club_idx}/{len(clubs)}] Fetching players from {club_name}...", end=" ", flush=True)
        
        try:
            players = scraper.get_squad_players(club_slug, club_id, "kader", season_year)
            print(f"[OK] {len(players)} players")
            club_player_counts[club_name] = len(players)
            
            added_count = 0
            overwritten_count = 0
            for player in players:
                player_id = player['player_id']
                # Track original club for this player
                if player_id not in player_original_club:
                    player_original_club[player_id] = club_name
                
                if player_id in player_club_map:
                    old_club = player_club_map[player_id]
                    if old_club != club_name:
                        overwritten_count += 1
                        # Use the original club, not the overwritten one
                        original_club = player_original_club[player_id]
                        if original_club != club_name:
                            # Keep the original club assignment
                            player_club_map[player_id] = original_club
                        else:
                            player_club_map[player_id] = club_name
                else:
                    player_club_map[player_id] = club_name
                added_count += 1
            
            if 'Al-Ahli' in club_name or 'Al Ahli' in club_name:
                print(f"    Added {added_count} Al-Ahli players to map (overwritten: {overwritten_count})")
        except Exception as e:
            print(f"[ERROR] {e}")
            continue
    
    # Log player counts per club
    print(f"\n  Player counts per club (from Transfermarkt):")
    for club_name, count in sorted(club_player_counts.items()):
        print(f"    {club_name}: {count} players")
    
    return player_club_map


def identify_players_from_match_data(
    data_dir: Path,
    season_label: str,
    target_clubs: Set[str]
) -> Set[int]:
    """
    Identify players who played for target clubs during a season by scanning match data files.
    Uses a heuristic: if a player appears in matches where a target club is involved,
    and that club appears more frequently than the opponent, assume the player was on that team.
    Returns set of player IDs.
    """
    players_in_clubs = set()
    
    # Check both match_data and previous_seasons folders
    match_data_dir = data_dir / "match_data"
    # previous_seasons is at saudi_arabia level (same level as date folder)
    previous_seasons_dir = data_dir.parent / "previous_seasons"
    
    match_files = []
    if match_data_dir.exists():
        match_files.extend(match_data_dir.glob(f"match_*_{season_label}.csv"))
    if previous_seasons_dir.exists():
        match_files.extend(previous_seasons_dir.glob(f"match_*_{season_label}.csv"))
    
    print(f"  Scanning {len(match_files)} match files for season {season_label}...")
    
    for match_file in match_files:
        try:
            # Extract player_id from filename: match_{player_id}_{season}.csv
            player_id = int(match_file.stem.split('_')[1])
            
            # Read full match file (not just first 100 rows)
            matches_df = pd.read_csv(match_file, encoding='utf-8-sig')
            if matches_df.empty:
                continue
            
            # Check if any matches are for target clubs
            # Match files have home_team and away_team columns
            if 'home_team' in matches_df.columns and 'away_team' in matches_df.columns:
                # For each target club, count how many times it appears as home vs away
                for target_club in target_clubs:
                    target_club_lower = target_club.lower()
                    
                    # Count matches where target club is home team
                    home_matches = matches_df[
                        matches_df['home_team'].str.lower().fillna('').str.contains(target_club_lower, case=False, na=False)
                    ]
                    # Count matches where target club is away team
                    away_matches = matches_df[
                        matches_df['away_team'].str.lower().fillna('').str.contains(target_club_lower, case=False, na=False)
                    ]
                    
                    home_count = len(home_matches)
                    away_count = len(away_matches)
                    
                    # If player has matches with this club, and it appears in at least 2 matches
                    # (to avoid false positives from single matches), include them
                    if home_count + away_count >= 2:
                        players_in_clubs.add(player_id)
                        break
                    # Also include if it's the only club that appears consistently
                    elif home_count + away_count >= 1 and len(matches_df) <= 5:
                        # For players with very few matches, be more lenient
                        players_in_clubs.add(player_id)
                        break
            elif 'club' in matches_df.columns:
                clubs = matches_df['club'].str.lower().fillna('')
                for target_club in target_clubs:
                    target_club_lower = target_club.lower()
                    if clubs.str.contains(target_club_lower, case=False, na=False).any():
                        players_in_clubs.add(player_id)
                        break
        except (ValueError, IndexError, Exception) as e:
            continue
    
    return players_in_clubs


def identify_players_in_league_clubs(
    profiles_df: pd.DataFrame,
    career_df: pd.DataFrame,
    season_start: pd.Timestamp,
    season_end: pd.Timestamp,
    target_clubs: Optional[Set[str]],
    data_dir: Optional[Path] = None,
    season_label: Optional[str] = None,
    scraper: Optional[TransfermarktScraper] = None,
    season_name: Optional[str] = None,
    season_info: Optional[Dict] = None
) -> Set[int]:
    """
    Identify players who were in target clubs during the specified season.
    Uses Transfermarkt scraper to fetch clubs and players directly (most accurate method).
    Falls back to career/profile/match data if scraper is not available.
    """
    # Method 1: Use Transfermarkt scraper (most accurate)
    if scraper and season_name and season_info:
        try:
            player_club_map = fetch_players_from_transfermarkt(season_name, season_info, scraper)
            players_in_clubs = set(player_club_map.keys())
            print(f"  Found {len(players_in_clubs)} players from Transfermarkt")
            return players_in_clubs
        except Exception as e:
            print(f"  Warning: Failed to fetch from Transfermarkt: {e}")
            print(f"  Falling back to local data...")
    
    # Method 2: Fallback to existing logic (career/profile/match data)
    players_in_clubs = set()
    
    # Check if we have the necessary data for fallback
    if profiles_df.empty:
        print(f"  Warning: Profiles DataFrame is empty. Cannot use fallback method.")
        return players_in_clubs
    
    career_df = career_df.copy()
    if not career_df.empty:
        career_df['Date'] = pd.to_datetime(career_df['Date'], errors='coerce')
    
    # Start with ALL players in profiles, not just those with career data
    all_player_ids = set(profiles_df['id'].unique())
    
    print(f"  Checking {len(all_player_ids)} players from profiles...")
    
    # Group by player
    for player_id in all_player_ids:
        profile_row = profiles_df[profiles_df['id'] == player_id]
        player_career = career_df[career_df['id'] == player_id]
        
        club_name = None
        
        # Method 1: Try to get club from profile data (current_club)
        if not profile_row.empty:
            current_club = profile_row.iloc[0].get('current_club', '')
            joined_on = profile_row.iloc[0].get('joined_on', '')
            
            if current_club and current_club.strip():
                # Check if current_club matches any target club
                if target_clubs and is_target_club(current_club, target_clubs):
                    # Check if player has transfer records showing they left this club before season end
                    left_club_before_season_end = False
                    if not player_career.empty:
                        player_career_sorted = player_career.sort_values('Date', ascending=False)
                        # Check if player transferred away from this club before or during season
                        transfers_out = player_career_sorted[
                            (player_career_sorted['Date'].notna()) & 
                            (player_career_sorted['From'].str.contains(current_club, case=False, na=False))
                        ]
                        if not transfers_out.empty:
                            latest_transfer_out = transfers_out.iloc[0]
                            transfer_out_date = latest_transfer_out['Date']
                            if pd.notna(transfer_out_date) and transfer_out_date < season_end:
                                left_club_before_season_end = True
                    
                    if not left_club_before_season_end:
                        # If joined_on is available, verify player was at club during season
                        if joined_on:
                            try:
                                joined_date = pd.to_datetime(joined_on, errors='coerce')
                                if pd.notna(joined_date):
                                    # If joined before season end, include them
                                    if joined_date <= season_end:
                                        club_name = current_club
                                    # If joined after season, don't include
                                else:
                                    # Can't parse joined_on, but current_club matches - include if no evidence they left
                                    club_name = current_club
                            except:
                                # Can't parse joined_on, but current_club matches - include if no evidence they left
                                club_name = current_club
                        else:
                            # No joined_on, but current_club matches target - include if season is recent enough
                            # (within 2 years) or if no evidence they left
                            if season_end >= pd.Timestamp.now() - pd.Timedelta(days=730):
                                club_name = current_club
                            # For older seasons, only include if we have evidence they were there
                            # (will be caught by career data check below)
        
        # Method 2: If no club from profile, try career data
        if (club_name is None or pd.isna(club_name)) and not player_career.empty:
            # Sort by date descending (most recent first)
            player_career = player_career.sort_values('Date', ascending=False)
            
            # Find the club at season start
            transfers_before_season = player_career[
                (player_career['Date'].notna()) & 
                (player_career['Date'] <= season_start)
            ]
            
            if not transfers_before_season.empty:
                # Get the most recent transfer before season start
                latest_transfer = transfers_before_season.iloc[0]
                club_name = latest_transfer['To']
                
                # Check if player transferred during the season
                transfers_during_season = player_career[
                    (player_career['Date'].notna()) & 
                    (player_career['Date'] > season_start) &
                    (player_career['Date'] <= season_end)
                ]
                
                if not transfers_during_season.empty:
                    # Player transferred mid-season - use the club they joined
                    mid_season_transfer = transfers_during_season.iloc[-1]
                    club_name = mid_season_transfer['To']
        
        # Check if club is in target clubs (with fuzzy matching)
        if club_name and target_clubs and is_target_club(club_name, target_clubs):
            players_in_clubs.add(player_id)
    
    # Method 3: Also use match data if available (this will add players not found by other methods)
    if data_dir and season_label and target_clubs:
        match_data_players = identify_players_from_match_data(data_dir, season_label, target_clubs)
        additional_players = match_data_players - players_in_clubs
        players_in_clubs.update(match_data_players)
        if additional_players:
            print(f"  Found {len(additional_players)} additional players from match data")
    
    return players_in_clubs


def is_target_club(club_name: str, target_clubs: Set[str]) -> bool:
    """Check if a club matches any target club (excludes B teams, U19, etc.)."""
    if pd.isna(club_name) or club_name == '':
        return False
    
    club_str = str(club_name).strip()
    
    # Improved exclusion logic - only exclude clubs that end with " B" or are specific B teams
    # Don't exclude clubs that just contain " B" in the middle of their name
    
    # Check if club ends with " B" or " B."
    if club_str.endswith(' B') or club_str.endswith(' B.'):
        return False
    
    # Youth/U19/U23 teams
    if ' U19' in club_str or ' U23' in club_str or ' U21' in club_str:
        return False
    
    # Other reserve/youth team indicators
    exclude_patterns = [
        'Promesas', 'Vetusta', 'Mirandilla', 'Castilla', 'Juvenil', 
        'Youth', 'Reserve'
    ]
    for pattern in exclude_patterns:
        if pattern in club_str:
            return False
    
    # Check if club matches any target club
    club_name_lower = club_str.lower()
    for target_club in target_clubs:
        target_club_lower = target_club.lower().strip()
        
        # Exact match (case-insensitive)
        if club_name_lower == target_club_lower:
            return True
        
        # Match without spaces
        if club_name_lower.replace(' ', '') == target_club_lower.replace(' ', ''):
            return True
        
        # Handle variations like "Al-Ahli" vs "Al Ahli"
        if club_name_lower.replace('-', ' ') == target_club_lower.replace('-', ' '):
            return True
        if target_club_lower.replace('-', ' ') == club_name_lower.replace('-', ' '):
            return True
    
    return False


def is_target_league_club(club_name: str, season_name: str) -> bool:
    """Check if a club is in the target league for the season (Saudi Pro League for both seasons)."""
    if pd.isna(club_name) or club_name == '':
        return False
    
    # For Saudi Pro League, we fetch clubs dynamically, so we can't use hardcoded lists
    # Instead, we'll rely on the fact that clubs fetched from Transfermarkt are already validated
    # This function is mainly used for filtering, so we'll be lenient here
    return True


def get_player_club_from_match_data(
    player_id: int,
    data_dir: Path,
    season_label: str,
    target_clubs: Set[str]
) -> Optional[str]:
    """
    Determine which target club a player was at during a season using match data.
    Returns the club name if found, None otherwise.
    """
    match_data_dir = data_dir / "match_data"
    previous_seasons_dir = data_dir.parent / "previous_seasons"
    
    match_files = []
    if match_data_dir.exists():
        match_files.extend(match_data_dir.glob(f"match_{player_id}_{season_label}.csv"))
    if previous_seasons_dir.exists():
        match_files.extend(previous_seasons_dir.glob(f"match_{player_id}_{season_label}.csv"))
    
    if not match_files:
        return None
    
    # Read the match file
    try:
        matches_df = pd.read_csv(match_files[0], encoding='utf-8-sig')
        if matches_df.empty:
            return None
        
        if 'home_team' in matches_df.columns and 'away_team' in matches_df.columns:
            # Count how many times each target club appears as home or away
            club_counts = {}
            for target_club in target_clubs:
                target_club_lower = target_club.lower()
                home_count = matches_df[
                    matches_df['home_team'].str.lower().fillna('').str.contains(target_club_lower, case=False, na=False)
                ].shape[0]
                away_count = matches_df[
                    matches_df['away_team'].str.lower().fillna('').str.contains(target_club_lower, case=False, na=False)
                ].shape[0]
                total_count = home_count + away_count
                if total_count > 0:
                    club_counts[target_club] = total_count
            
            # Return the club that appears most frequently
            if club_counts:
                return max(club_counts, key=club_counts.get)
    except Exception:
        pass
    
    return None


def map_players_to_clubs_by_season(
    career_df: pd.DataFrame,
    season_start: pd.Timestamp,
    season_end: pd.Timestamp,
    season_name: str,
    target_players: Optional[Set[int]] = None,
    profiles_df: Optional[pd.DataFrame] = None,
    data_dir: Optional[Path] = None,
    season_label: Optional[str] = None,
    target_clubs: Optional[Set[str]] = None,
    scraper: Optional[TransfermarktScraper] = None,
    season_info: Optional[Dict] = None
) -> Dict[int, Dict[str, any]]:
    """
    Determine which club each player was at during each season.
    Only includes players in target_players set and filters to target league clubs.
    Returns: {player_id: {'club': club_name, 'market_value': value, 'start_date': date, 'end_date': date}}
    """
    league_name = SEASONS[season_name]['league']
    print(f"Mapping players to clubs for season {season_start.year}/{season_end.year} ({league_name})...")
    
    player_clubs = {}
    
    # Method 1: Use Transfermarkt scraper (most accurate)
    if scraper and season_info:
        try:
            player_club_map = fetch_players_from_transfermarkt(season_name, season_info, scraper)
            initial_count = len(player_club_map)
            print(f"  Fetched {initial_count} players from Transfermarkt")
            
            # Check Al-Ahli players in the fetched map
            al_ahli_fetched = {pid: club for pid, club in player_club_map.items() if 'Al-Ahli' in club or 'Al Ahli' in club}
            print(f"  Al-Ahli players in fetched map: {len(al_ahli_fetched)}")
            
            # Don't filter based on target_players - keep all fetched players
            # This ensures we don't lose players due to timing differences between fetches
            # The target_players filter was causing 4 Al-Ahli players to be excluded
            if target_players is not None:
                before_filter = len(player_club_map)
                # Check which Al-Ahli players are in target_players
                al_ahli_in_target = {pid for pid in al_ahli_fetched.keys() if pid in target_players}
                print(f"  Al-Ahli players in target_players set: {len(al_ahli_in_target)}")
                if len(al_ahli_in_target) < len(al_ahli_fetched):
                    missing_from_target = set(al_ahli_fetched.keys()) - al_ahli_in_target
                    print(f"  WARNING: {len(missing_from_target)} Al-Ahli players not in target_players: {sorted(missing_from_target)}")
                
                # Keep all fetched players - don't filter
                # player_club_map = {pid: club for pid, club in player_club_map.items() if pid in target_players}
                after_filter = len(player_club_map)
                # Log if we would have filtered but didn't (for debugging)
                would_have_filtered = {pid: club for pid, club in player_club_map.items() if pid in target_players}
                if len(would_have_filtered) != len(player_club_map):
                    excluded_count = len(player_club_map) - len(would_have_filtered)
                    print(f"  Note: {excluded_count} players would have been excluded by target_players filter, but keeping all {len(player_club_map)} fetched players")
            
            # Convert to the expected format with market values
            career_df_copy = career_df.copy()
            if not career_df_copy.empty:
                career_df_copy['Date'] = pd.to_datetime(career_df_copy['Date'], errors='coerce')
            
            players_with_valid_club = 0
            al_ahli_in_map = []
            for player_id, club_name in player_club_map.items():
                if 'Al-Ahli' in club_name or 'Al Ahli' in club_name:
                    al_ahli_in_map.append(player_id)
                # Try to get market value from career data
                market_value = 0
                if not career_df_copy.empty:
                    player_career = career_df_copy[career_df_copy['id'] == player_id]
                    if not player_career.empty:
                        player_career = player_career.sort_values('Date', ascending=False)
                        transfers_before_season = player_career[
                            (player_career['Date'].notna()) & 
                            (player_career['Date'] <= season_start)
                        ]
                        if not transfers_before_season.empty:
                            latest_transfer = transfers_before_season.iloc[0]
                            mv = latest_transfer.get('VM', 0)
                            try:
                                if pd.notna(mv) and mv != '' and mv != '-':
                                    market_value = float(str(mv).replace(',', '').replace('€', '').strip())
                            except (ValueError, AttributeError):
                                pass
                
                player_clubs[player_id] = {
                    'club': club_name,
                    'market_value': market_value,
                    'start_date': season_start,
                    'end_date': season_end
                }
                players_with_valid_club += 1
            
            if al_ahli_in_map:
                print(f"  Al-Ahli players in player_club_map: {len(al_ahli_in_map)}")
                if len(al_ahli_in_map) < 38:
                    print(f"  WARNING: Only {len(al_ahli_in_map)} Al-Ahli players in player_club_map (expected 38)")
                    print(f"  Al-Ahli player IDs in player_club_map: {sorted(al_ahli_in_map)}")
            
            # Log player counts by club
            club_counts = {}
            al_ahli_players = []
            for player_id, info in player_clubs.items():
                club = info['club']
                club_counts[club] = club_counts.get(club, 0) + 1
                if 'Al-Ahli' in club or 'Al Ahli' in club:
                    al_ahli_players.append((player_id, info.get('club', 'Unknown')))
            
            print(f"  Mapped {len(player_clubs)} players to {league_name} clubs (from Transfermarkt)")
            print(f"  Player counts by club after mapping:")
            for club_name, count in sorted(club_counts.items()):
                print(f"    {club_name}: {count} players")
            
            # Special logging for Al-Ahli to identify missing players
            if al_ahli_players:
                print(f"  Al-Ahli SFC players after mapping: {len(al_ahli_players)}")
                if len(al_ahli_players) < 38:
                    print(f"  WARNING: Expected 38 Al-Ahli players, but only {len(al_ahli_players)} were mapped")
                    print(f"  Al-Ahli player IDs after mapping: {sorted([pid for pid, _ in al_ahli_players])}")
            
            return player_clubs
        except Exception as e:
            print(f"  Warning: Failed to fetch from Transfermarkt: {e}")
            print(f"  Falling back to local data...")
    
    # Method 2: Fallback to existing logic
    # Check if we have the necessary data for fallback
    if career_df.empty:
        print(f"  Warning: Career DataFrame is empty. Cannot use fallback method.")
        return player_clubs
    
    # Convert Date column to datetime
    career_df = career_df.copy()
    career_df['Date'] = pd.to_datetime(career_df['Date'], errors='coerce')
    
    # Group by player
    for player_id, player_career in career_df.groupby('id'):
        # Filter to only target players if list provided
        if target_players is not None and player_id not in target_players:
            continue
        
        # Sort by date descending (most recent first)
        player_career = player_career.sort_values('Date', ascending=False)
        
        # Find the club at season start
        # Get transfers before or at season start
        transfers_before_season = player_career[
            (player_career['Date'].notna()) & 
            (player_career['Date'] <= season_start)
        ]
        
        if not transfers_before_season.empty:
            # Get the most recent transfer before season start
            latest_transfer = transfers_before_season.iloc[0]
            club_name = latest_transfer['To']
            market_value = latest_transfer.get('VM', 0)
            
            # Try to convert market value to numeric
            try:
                if pd.isna(market_value) or market_value == '' or market_value == '-':
                    market_value = 0
                else:
                    market_value = float(str(market_value).replace(',', '').replace('€', '').strip())
            except (ValueError, AttributeError):
                market_value = 0
            
            # Check if player transferred during the season
            transfers_during_season = player_career[
                (player_career['Date'].notna()) & 
                (player_career['Date'] > season_start) &
                (player_career['Date'] <= season_end)
            ]
            
            if not transfers_during_season.empty:
                # Player transferred mid-season - use the club they joined
                mid_season_transfer = transfers_during_season.iloc[-1]  # First transfer during season
                club_name = mid_season_transfer['To']
                # Update market value if available
                try:
                    mv = mid_season_transfer.get('VM', 0)
                    if not (pd.isna(mv) or mv == '' or mv == '-'):
                        market_value = float(str(mv).replace(',', '').replace('€', '').strip())
                except (ValueError, AttributeError):
                    pass
            
            # Only include if club is in target league
            if is_target_league_club(club_name, season_name):
                player_clubs[player_id] = {
                    'club': club_name,
                    'market_value': market_value,
                    'start_date': season_start,
                    'end_date': season_end
                }
        
        # If still no club found, try match data
        if player_id not in player_clubs and data_dir and season_label and target_clubs:
            club_from_matches = get_player_club_from_match_data(player_id, data_dir, season_label, target_clubs)
            if club_from_matches and is_target_league_club(club_from_matches, season_name):
                # Try to get market value from profile or career
                market_value = 0
                if profiles_df is not None:
                    profile_row = profiles_df[profiles_df['id'] == player_id]
                    if not profile_row.empty:
                        # Market value would need to come from a different source
                        # For now, set to 0 if not available
                        pass
                
                player_clubs[player_id] = {
                    'club': club_from_matches,
                    'market_value': market_value,
                    'start_date': season_start,
                    'end_date': season_end
                }
    
    print(f"  Mapped {len(player_clubs)} players to {league_name} clubs")
    return player_clubs


def is_week_injured(week_start: pd.Timestamp, week_end: pd.Timestamp, injury_periods: List[Tuple[pd.Timestamp, pd.Timestamp]]) -> bool:
    """
    Check if a week overlaps with any injury period.
    Returns True if any injury overlaps with the week (even partially).
    """
    for injury_start, injury_end in injury_periods:
        # Check for overlap: injury overlaps if it starts before week ends 
        # and ends after week starts
        if injury_start <= week_end and injury_end >= week_start:
            return True
    return False


def calculate_weeks_between(start_date: pd.Timestamp, end_date: pd.Timestamp) -> int:
    """Calculate number of weeks between two dates (inclusive).
    For a full season year (365 days), returns exactly 52 weeks."""
    days = (end_date - start_date).days + 1
    
    # For a full year (365 days), return exactly 52 weeks
    if days == 365:
        return 52
    
    # For other periods, use standard calculation
    return (days + 6) // 7  # Round up to include partial weeks


def calculate_injury_weeks(
    player_id: int,
    week_calc_start: pd.Timestamp,  # Start date for calculating total weeks
    week_calc_end: pd.Timestamp,     # End date for calculating total weeks
    injury_filter_start: pd.Timestamp,  # Only count injuries since this date
    injury_filter_end: pd.Timestamp,    # End date for injury filtering
    injuries_df: pd.DataFrame
) -> Tuple[int, int, int]:
    """
    Calculate injured and non-injured weeks for a player.
    
    Args:
        week_calc_start: Start date for calculating total weeks (e.g., season start)
        week_calc_end: End date for calculating total weeks (e.g., season end or today)
        injury_filter_start: Only count injuries that occurred or were ongoing since this date
        injury_filter_end: End date for injury filtering (usually same as week_calc_end)
        injuries_df: DataFrame with injury data
    
    Only includes injuries that occurred or were ongoing since injury_filter_start.
    If a player was already injured at injury_filter_start, count that injury.
    If a player was injured for any part of a week (1-6 days), count that entire week as injured.
    
    Returns: (injured_weeks, non_injured_weeks, total_weeks)
    """
    # Get player's injuries
    player_injuries = injuries_df[injuries_df['player_id'] == player_id].copy()
    
    # Calculate total weeks for the period
    total_weeks = calculate_weeks_between(week_calc_start, week_calc_end)
    
    if player_injuries.empty:
        # No injuries - all weeks are non-injured
        return (0, total_weeks, total_weeks)
    
    # Parse injury dates
    player_injuries['fromDate'] = pd.to_datetime(player_injuries['fromDate'], errors='coerce')
    player_injuries['untilDate'] = pd.to_datetime(player_injuries['untilDate'], errors='coerce')
    
    # Filter injuries that overlap with the injury filter period
    injury_periods = []
    for _, injury in player_injuries.iterrows():
        injury_start = injury['fromDate']
        injury_end = injury['untilDate']
        
        if pd.isna(injury_start):
            continue
        
        # If end date is missing, use injury filter end
        if pd.isna(injury_end):
            injury_end = injury_filter_end
        
        # Include injury if:
        # 1. It started on or after injury_filter_start, OR
        # 2. It was ongoing at injury_filter_start (started before but ended after or is still ongoing)
        if injury_start >= injury_filter_start:
            # Injury started during filter period
            injury_start_clipped = injury_start
            injury_end_clipped = min(injury_end, injury_filter_end)
        elif injury_end >= injury_filter_start:
            # Injury was ongoing at injury_filter_start
            injury_start_clipped = injury_filter_start
            injury_end_clipped = min(injury_end, injury_filter_end)
        else:
            # Injury ended before injury_filter_start - skip it
            continue
        
        if injury_start_clipped <= injury_end_clipped:
            injury_periods.append((injury_start_clipped, injury_end_clipped))
    
    if not injury_periods:
        # No injuries in filter period
        return (0, total_weeks, total_weeks)
    
    # Divide week calculation period into weeks and check which are injured
    injured_weeks = set()
    current_date = week_calc_start
    
    while current_date <= week_calc_end:
        week_start = current_date
        week_end = min(current_date + timedelta(days=6), week_calc_end)
        
        # Check if this week overlaps with any injury period
        if is_week_injured(week_start, week_end, injury_periods):
            # Calculate week number (0-indexed)
            week_num = (current_date - week_calc_start).days // 7
            injured_weeks.add(week_num)
        
        current_date += timedelta(days=7)
    
    injured_weeks_count = len(injured_weeks)
    non_injured_weeks = total_weeks - injured_weeks_count
    
    return (injured_weeks_count, non_injured_weeks, total_weeks)


def calculate_player_age_at_season_start(
    player_id: int,
    season_start: pd.Timestamp,
    profiles_df: pd.DataFrame
) -> Optional[float]:
    """Get player age at season start."""
    if profiles_df.empty:
        return None
    
    player_profile = profiles_df[profiles_df['id'] == player_id]
    
    if player_profile.empty:
        return None
    
    date_of_birth = player_profile.iloc[0]['date_of_birth']
    
    if pd.isna(date_of_birth):
        return None
    
    try:
        if isinstance(date_of_birth, str):
            dob = pd.to_datetime(date_of_birth, errors='coerce')
        else:
            dob = pd.to_datetime(date_of_birth, errors='coerce')
        
        if pd.isna(dob):
            return None
        
        age = (season_start - dob).days / 365.25
        return age
    except (ValueError, TypeError):
        return None


def get_player_market_value_at_season_start(
    player_id: int,
    season_start: pd.Timestamp,
    career_df: pd.DataFrame
) -> float:
    """Get player market value at season start."""
    if career_df.empty:
        return 0.0
    
    player_career = career_df[career_df['id'] == player_id].copy()
    
    if player_career.empty:
        return 0.0
    
    player_career['Date'] = pd.to_datetime(player_career['Date'], errors='coerce')
    
    # Get transfers before or at season start
    transfers_before = player_career[
        (player_career['Date'].notna()) & 
        (player_career['Date'] <= season_start)
    ]
    
    if transfers_before.empty:
        return 0.0
    
    # Get most recent transfer
    latest_transfer = transfers_before.iloc[0]
    market_value = latest_transfer.get('VM', 0)
    
    try:
        if pd.isna(market_value) or market_value == '' or market_value == '-':
            return 0.0
        else:
            return float(str(market_value).replace(',', '').replace('€', '').strip())
    except (ValueError, AttributeError):
        return 0.0


def calculate_additional_injury_kpis(
    player_id: int,
    analysis_start: pd.Timestamp,
    analysis_end: pd.Timestamp,
    injuries_df: pd.DataFrame
) -> Dict[str, any]:
    """Calculate additional injury KPIs for a player since analysis_start."""
    player_injuries = injuries_df[injuries_df['player_id'] == player_id].copy()
    
    if player_injuries.empty:
        return {
            'num_injuries': 0,
            'avg_injury_duration': 0.0,
            'injuries_mild': 0,
            'injuries_moderate': 0,
            'injuries_severe': 0,
            'injuries_critical': 0
        }
    
    # Parse injury dates
    player_injuries['fromDate'] = pd.to_datetime(player_injuries['fromDate'], errors='coerce')
    player_injuries['untilDate'] = pd.to_datetime(player_injuries['untilDate'], errors='coerce')
    
    # Filter injuries that overlap with analysis period
    season_injuries = player_injuries[
        (player_injuries['fromDate'].notna()) &
        (
            # Injury started during analysis period
            ((player_injuries['fromDate'] >= analysis_start) & (player_injuries['fromDate'] <= analysis_end)) |
            # Injury was ongoing at analysis_start
            ((player_injuries['fromDate'] < analysis_start) & 
             ((player_injuries['untilDate'].isna()) | (player_injuries['untilDate'] >= analysis_start)))
        )
    ]
    
    num_injuries = len(season_injuries)
    
    # Calculate average duration (only the portion within analysis period)
    durations = []
    for _, injury in season_injuries.iterrows():
        injury_start = max(injury['fromDate'], analysis_start)
        injury_end = injury['untilDate'] if pd.notna(injury['untilDate']) else analysis_end
        injury_end = min(injury_end, analysis_end)
        
        if pd.notna(injury_start) and pd.notna(injury_end):
            duration_days = (injury_end - injury_start).days + 1
            if duration_days > 0:
                durations.append(duration_days)
    
    avg_duration = np.mean(durations) if durations else 0.0
    
    # Count by severity
    severity_counts = season_injuries['severity'].value_counts().to_dict()
    
    return {
        'num_injuries': num_injuries,
        'avg_injury_duration': avg_duration,
        'injuries_mild': severity_counts.get('mild', 0),
        'injuries_moderate': severity_counts.get('moderate', 0),
        'injuries_severe': severity_counts.get('severe', 0),
        'injuries_critical': severity_counts.get('critical', 0)
    }


def aggregate_club_kpis(player_level_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate player-level data to club-level."""
    print("Aggregating club-level KPIs...")
    
    club_level_data = []
    
    for (club, season), group in player_level_df.groupby(['club', 'season']):
        # Improved exclusion logic - only exclude clubs that end with " B" or are specific B teams
        # Don't exclude clubs that just contain " B" in the middle of their name
        should_exclude = False
        club_str = str(club)
        
        # Check if club ends with " B" or " B."
        if club_str.endswith(' B') or club_str.endswith(' B.'):
            should_exclude = True
        
        # Youth/U19/U23 teams
        if ' U19' in club_str or ' U23' in club_str or ' U21' in club_str:
            should_exclude = True
        
        # Other reserve/youth team indicators
        exclude_patterns = [
            'Promesas', 'Vetusta', 'Mirandilla', 'Castilla', 'Juvenil', 
            'Youth', 'Reserve'
        ]
        for pattern in exclude_patterns:
            if pattern in club_str:
                should_exclude = True
                break
        
        if should_exclude:
            continue
        
        club_data = {
            'club': club,
            'season': season,
            'total_injured_weeks': group['injured_weeks'].sum(),
            'total_non_injured_weeks': group['non_injured_weeks'].sum(),
            'total_player_weeks': group['total_weeks'].sum(),
            'club_injury_rate_pct': (group['injured_weeks'].sum() / group['total_weeks'].sum() * 100) if group['total_weeks'].sum() > 0 else 0.0,
            'average_age': group['age'].mean() if group['age'].notna().any() else None,
            'total_market_value': group['market_value'].sum(),
            'avg_market_value': group['market_value'].mean() if group['market_value'].notna().any() else 0.0,
            'squad_size': len(group),
            'num_injuries': group['num_injuries'].sum(),
            'avg_injury_duration': group['avg_injury_duration'].mean() if group['avg_injury_duration'].notna().any() else 0.0,
            'injuries_mild': group['injuries_mild'].sum(),
            'injuries_moderate': group['injuries_moderate'].sum(),
            'injuries_severe': group['injuries_severe'].sum(),
            'injuries_critical': group['injuries_critical'].sum()
        }
        club_level_data.append(club_data)
    
    return pd.DataFrame(club_level_data)


def generate_benchmarking_report(club_level_df: pd.DataFrame, output_dir: Path):
    """Generate summary report and comparisons."""
    print("Generating benchmarking report...")
    
    report_lines = []
    report_lines.append("# Al-Ahli SFC Benchmarking Analysis Report")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Filter Al-Ahli SFC data
    al_ahli_data = club_level_df[club_level_df['club'].str.contains('Al-Ahli|Al Ahli', case=False, na=False)]
    other_clubs = club_level_df[~club_level_df['club'].str.contains('Al-Ahli|Al Ahli', case=False, na=False)]
    
    for season in ['2024/25', '2025/26']:
        report_lines.append(f"## Season {season}")
        report_lines.append("")
        
        season_al_ahli = al_ahli_data[al_ahli_data['season'] == season]
        season_others = other_clubs[other_clubs['season'] == season]
        
        if season_al_ahli.empty:
            report_lines.append(f"  No data for Al-Ahli SFC in {season}")
            report_lines.append("")
            continue
        
        al_ahli_row = season_al_ahli.iloc[0]
        
        # Use all clubs in the data (they were already validated when fetched from Transfermarkt)
        # No need to filter by hardcoded list - use all clubs that appear in the data
        season_others_target = season_others.copy()
        
        # League averages (from target league clubs only)
        league_avg_injury_rate = season_others_target['club_injury_rate_pct'].mean()
        league_avg_age = season_others_target['average_age'].mean()
        league_avg_market_value = season_others_target['total_market_value'].mean()
        
        league_name = "Saudi Pro League"
        report_lines.append(f"### Al-Ahli SFC vs League Averages ({league_name})")
        report_lines.append("")
        report_lines.append(f"**Injury Rate:**")
        report_lines.append(f"  - Al-Ahli SFC: {al_ahli_row['club_injury_rate_pct']:.2f}%")
        report_lines.append(f"  - League Average: {league_avg_injury_rate:.2f}%")
        report_lines.append(f"  - Difference: {al_ahli_row['club_injury_rate_pct'] - league_avg_injury_rate:+.2f}%")
        report_lines.append("")
        
        report_lines.append(f"**Average Age:**")
        if pd.notna(al_ahli_row['average_age']):
            report_lines.append(f"  - Al-Ahli SFC: {al_ahli_row['average_age']:.1f} years")
        if pd.notna(league_avg_age):
            report_lines.append(f"  - League Average: {league_avg_age:.1f} years")
            if pd.notna(al_ahli_row['average_age']):
                report_lines.append(f"  - Difference: {al_ahli_row['average_age'] - league_avg_age:+.1f} years")
        report_lines.append("")
        
        report_lines.append(f"**Total Market Value:**")
        report_lines.append(f"  - Al-Ahli SFC: €{al_ahli_row['total_market_value']:,.0f}")
        report_lines.append(f"  - League Average: €{league_avg_market_value:,.0f}")
        report_lines.append(f"  - Difference: €{al_ahli_row['total_market_value'] - league_avg_market_value:+,.0f}")
        report_lines.append("")
        
        # Rankings
        report_lines.append("### Rankings")
        report_lines.append("")
        
        # Injury rate ranking (lower is better)
        sorted_injury = season_others_target.sort_values('club_injury_rate_pct')
        al_ahli_rank_injury = (sorted_injury['club_injury_rate_pct'] < al_ahli_row['club_injury_rate_pct']).sum() + 1
        total_clubs = len(season_others_target) + 1
        report_lines.append(f"**Injury Rate:** Al-Ahli SFC ranks #{al_ahli_rank_injury} out of {total_clubs} clubs (lower is better)")
        report_lines.append("")
        
        # Market value ranking (higher is better)
        sorted_market = season_others_target.sort_values('total_market_value', ascending=False)
        al_ahli_rank_market = (sorted_market['total_market_value'] > al_ahli_row['total_market_value']).sum() + 1
        report_lines.append(f"**Market Value:** Al-Ahli SFC ranks #{al_ahli_rank_market} out of {total_clubs} clubs (higher is better)")
        report_lines.append("")
        
        report_lines.append("=" * 80)
        report_lines.append("")
    
    # Write report
    report_path = output_dir / "benchmarking_summary_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"  Report saved to: {report_path}")


def main():
    print("=" * 80)
    print("Al-Ahli SFC Benchmarking Analysis")
    print("=" * 80)
    print()
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize scraper for fetching clubs/players from Transfermarkt
    scraper = TransfermarktScraper(ScraperConfig())
    
    try:
        # Load data
        data = load_all_data(DATA_DIR)
        
        # Prepare DataFrames
        profiles_df = data['profiles']
        injuries_df = data['injuries']
        career_df = data['career']
        
        # Analysis period: from 2025-07-01 onwards
        analysis_start = INJURY_ANALYSIS_START
        analysis_end = pd.Timestamp.now()  # Use current date as end
        
        print(f"Analysis period: {analysis_start.date()} to {analysis_end.date()}")
        print(f"Total weeks in analysis: {calculate_weeks_between(analysis_start, analysis_end)}")
        print()
        
        # Process each season
        player_level_results = []
        
        # Season 2024/25: Saudi Pro League
        season_2024_25 = SEASONS["2024/25"]
        print(f"\n{'='*80}")
        print(f"Processing Season: 2024/25 (Saudi Pro League)")
        print(f"{'='*80}")
        
        print(f"Identifying players in Saudi Pro League clubs for 2024/25...")
        saudi_players_2024_25 = identify_players_in_league_clubs(
            profiles_df,
            career_df,
            season_2024_25['start'],
            season_2024_25['end'],
            None,  # No hardcoded club list - fetch dynamically
            DATA_DIR,
            "2024_2025",
            scraper,
            "2024/25",
            season_2024_25
        )
        print(f"  Found {len(saudi_players_2024_25)} players in Saudi Pro League clubs")
        
        # Map players to clubs
        player_clubs_2024_25 = map_players_to_clubs_by_season(
            career_df,
            season_2024_25['start'],
            season_2024_25['end'],
            "2024/25",
            saudi_players_2024_25,
            profiles_df,
            DATA_DIR,
            "2024_2025",
            None,  # No hardcoded club list
            scraper,
            season_2024_25
        )
        
        print(f"Calculating KPIs for {len(player_clubs_2024_25)} Saudi Pro League players...")
        
        # Track processing stats
        players_processed = 0
        players_skipped = 0
        players_by_club = {}
        
        # Process each player for 2024/25
        for player_id, club_info in player_clubs_2024_25.items():
            club_name = club_info['club']
            
            # Skip if club name is missing or invalid
            if pd.isna(club_name) or club_name == '' or club_name == 'Unknown':
                players_skipped += 1
                continue
            
            players_processed += 1
            players_by_club[club_name] = players_by_club.get(club_name, 0) + 1
            
            # Calculate injury weeks
            # For 2024/25: Use full season period for both weeks and injury counting
            injured_weeks, non_injured_weeks, total_weeks = calculate_injury_weeks(
                player_id, 
                season_2024_25['start'],  # Week calculation: full season
                season_2024_25['end'],    # Week calculation: full season
                season_2024_25['start'],   # Injury filter: season start
                season_2024_25['end'],    # Injury filter: season end
                injuries_df
            )
            
            # Calculate age (at season start)
            age = calculate_player_age_at_season_start(
                player_id, season_2024_25['start'], profiles_df
            )
            
            # Get market value
            market_value = get_player_market_value_at_season_start(
                player_id, season_2024_25['start'], career_df
            )
            
            # Calculate additional injury KPIs (for the season period)
            injury_kpis = calculate_additional_injury_kpis(
                player_id, season_2024_25['start'], season_2024_25['end'], injuries_df
            )
            
            # Get player info
            player_profile = profiles_df[profiles_df['id'] == player_id]
            player_name = player_profile.iloc[0]['name'] if not player_profile.empty else f"Player {player_id}"
            position = player_profile.iloc[0].get('position', '') if not player_profile.empty else ''
            
            # Calculate injury rate
            injury_rate = (injured_weeks / total_weeks * 100) if total_weeks > 0 else 0.0
            
            player_result = {
                'player_id': player_id,
                'player_name': player_name,
                'club': club_name,
                'season': '2024/25',
                'league': 'Saudi Pro League',
                'position': position,
                'injured_weeks': injured_weeks,
                'non_injured_weeks': non_injured_weeks,
                'total_weeks': total_weeks,
                'injury_rate_pct': injury_rate,
                'age': age,
                'market_value': market_value,
                'num_injuries': injury_kpis['num_injuries'],
                'avg_injury_duration': injury_kpis['avg_injury_duration'],
                'injuries_mild': injury_kpis['injuries_mild'],
                'injuries_moderate': injury_kpis['injuries_moderate'],
                'injuries_severe': injury_kpis['injuries_severe'],
                'injuries_critical': injury_kpis['injuries_critical']
            }
            
            player_level_results.append(player_result)
        
        # Log processing stats for 2024/25
        print(f"\n  Processing stats for 2024/25:")
        print(f"    Total players mapped: {len(player_clubs_2024_25)}")
        print(f"    Players processed: {players_processed}")
        print(f"    Players skipped: {players_skipped}")
        print(f"    Players by club (processed):")
        for club_name, count in sorted(players_by_club.items()):
            print(f"      {club_name}: {count} players")
        
        # Season 2025/26: Saudi Pro League
        season_2025_26 = SEASONS["2025/26"]
        print(f"\n{'='*80}")
        print(f"Processing Season: 2025/26 (Saudi Pro League)")
        print(f"{'='*80}")
        
        print(f"Identifying players in Saudi Pro League clubs for 2025/26...")
        saudi_players_2025_26 = identify_players_in_league_clubs(
            profiles_df,
            career_df,
            season_2025_26['start'],
            season_2025_26['end'],
            None,  # No hardcoded club list - fetch dynamically
            DATA_DIR,
            "2025_2026",
            scraper,
            "2025/26",
            season_2025_26
        )
        print(f"  Found {len(saudi_players_2025_26)} players in Saudi Pro League clubs")
        
        # Map players to clubs
        player_clubs_2025_26 = map_players_to_clubs_by_season(
            career_df,
            season_2025_26['start'],
            season_2025_26['end'],
            "2025/26",
            saudi_players_2025_26,
            profiles_df,
            DATA_DIR,
            "2025_2026",
            None,  # No hardcoded club list
            scraper,
            season_2025_26
        )
        
        print(f"Calculating KPIs for {len(player_clubs_2025_26)} Saudi Pro League players...")
        
        # Track processing stats
        players_processed_2025_26 = 0
        players_skipped_2025_26 = 0
        players_by_club_2025_26 = {}
        
        # Process each player for 2025/26
        for player_id, club_info in player_clubs_2025_26.items():
            club_name = club_info['club']
            
            # Skip if club name is missing or invalid
            if pd.isna(club_name) or club_name == '' or club_name == 'Unknown':
                players_skipped_2025_26 += 1
                continue
            
            players_processed_2025_26 += 1
            players_by_club_2025_26[club_name] = players_by_club_2025_26.get(club_name, 0) + 1
            
            # Calculate injury weeks
            # For 2025/26: Use analysis period (2025-07-01 to today) for both weeks and injuries
            injured_weeks, non_injured_weeks, total_weeks = calculate_injury_weeks(
                player_id,
                analysis_start,           # Week calculation: from 2025-07-01
                analysis_end,             # Week calculation: until today
                analysis_start,           # Injury filter: only since 2025-07-01
                analysis_end,             # Injury filter: until today
                injuries_df
            )
            
            # Calculate age (at season start)
            age = calculate_player_age_at_season_start(
                player_id, season_2025_26['start'], profiles_df
            )
            
            # Get market value
            market_value = get_player_market_value_at_season_start(
                player_id, season_2025_26['start'], career_df
            )
            
            # Calculate additional injury KPIs (only since 2025-07-01)
            injury_kpis = calculate_additional_injury_kpis(
                player_id, analysis_start, analysis_end, injuries_df
            )
            
            # Get player info
            player_profile = profiles_df[profiles_df['id'] == player_id]
            player_name = player_profile.iloc[0]['name'] if not player_profile.empty else f"Player {player_id}"
            position = player_profile.iloc[0].get('position', '') if not player_profile.empty else ''
            
            # Calculate injury rate
            injury_rate = (injured_weeks / total_weeks * 100) if total_weeks > 0 else 0.0
            
            player_result = {
                'player_id': player_id,
                'player_name': player_name,
                'club': club_name,
                'season': '2025/26',
                'league': 'Saudi Pro League',
                'position': position,
                'injured_weeks': injured_weeks,
                'non_injured_weeks': non_injured_weeks,
                'total_weeks': total_weeks,
                'injury_rate_pct': injury_rate,
                'age': age,
                'market_value': market_value,
                'num_injuries': injury_kpis['num_injuries'],
                'avg_injury_duration': injury_kpis['avg_injury_duration'],
                'injuries_mild': injury_kpis['injuries_mild'],
                'injuries_moderate': injury_kpis['injuries_moderate'],
                'injuries_severe': injury_kpis['injuries_severe'],
                'injuries_critical': injury_kpis['injuries_critical']
            }
            
            player_level_results.append(player_result)
        
        # Log processing stats for 2025/26
        print(f"\n  Processing stats for 2025/26:")
        print(f"    Total players mapped: {len(player_clubs_2025_26)}")
        print(f"    Players processed: {players_processed_2025_26}")
        print(f"    Players skipped: {players_skipped_2025_26}")
        print(f"    Players by club (processed):")
        for club_name, count in sorted(players_by_club_2025_26.items()):
            print(f"      {club_name}: {count} players")
        
        # Create player-level DataFrame
        player_level_df = pd.DataFrame(player_level_results)
        
        # Aggregate to club level
        club_level_df = aggregate_club_kpis(player_level_df)
        
        # Save results
        print(f"\n{'='*80}")
        print("Saving results...")
        print(f"{'='*80}")
        
        # Player-level results
        player_output_path = OUTPUT_DIR / "benchmarking_player_level.csv"
        player_level_df.to_csv(player_output_path, index=False, encoding='utf-8-sig', sep=';')
        print(f"  Player-level data: {player_output_path} ({len(player_level_df)} records)")
        
        # Club-level results
        club_output_path = OUTPUT_DIR / "benchmarking_club_level.csv"
        club_level_df.to_csv(club_output_path, index=False, encoding='utf-8-sig', sep=';')
        print(f"  Club-level data: {club_output_path} ({len(club_level_df)} records)")
        
        # Al-Ahli SFC focus comparison
        al_ahli_focus = club_level_df.copy()
        al_ahli_clubs = al_ahli_focus[al_ahli_focus['club'].str.contains('Al-Ahli|Al Ahli', case=False, na=False)]['club'].unique()
        
        if len(al_ahli_clubs) > 0:
            al_ahli_focus_path = OUTPUT_DIR / "benchmarking_al_ahli_sfc_focus.csv"
            al_ahli_focus.to_csv(al_ahli_focus_path, index=False, encoding='utf-8-sig', sep=';')
            print(f"  Al-Ahli SFC focus: {al_ahli_focus_path}")
        
        # Generate report
        generate_benchmarking_report(club_level_df, OUTPUT_DIR)
        
        print(f"\n{'='*80}")
        print("Analysis Complete!")
        print(f"{'='*80}")
        print(f"Output directory: {OUTPUT_DIR}")
        print()
    
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        scraper.close()


if __name__ == "__main__":
    main()
