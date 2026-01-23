#!/usr/bin/env python3
"""
Analyze missing players for a specific club by comparing our config with Transfermarkt.
Checks if missing players have left the club since season start.
"""

import json
import sys
import io
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Optional
from datetime import datetime

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

# Add root directory to path
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent
ROOT_DIR = PRODUCTION_ROOT.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.data_collection.transfermarkt_scraper import ScraperConfig, TransfermarktScraper

# Paths
DEPLOYMENTS_DIR = PRODUCTION_ROOT / "deployments" / "England"

# Club name mapping (our names -> Transfermarkt names)
CLUB_NAME_MAPPING = {
    "Arsenal FC": "Arsenal FC",
    "Manchester City": "Manchester City",
    "Chelsea FC": "Chelsea FC",
    "Liverpool FC": "Liverpool FC",
    "Tottenham Hotspur": "Tottenham Hotspur",
    "Manchester United": "Manchester United",
    "Aston Villa": "Aston Villa",
    "Newcastle United": "Newcastle United",
    "Brighton & Hove Albion": "Brighton & Hove Albion",
    "West Ham United": "West Ham United",
    "Wolverhampton Wanderers": "Wolverhampton Wanderers",
    "Crystal Palace": "Crystal Palace",
    "Fulham FC": "Fulham FC",
    "AFC Bournemouth": "AFC Bournemouth",
    "Everton FC": "Everton FC",
    "Brentford FC": "Brentford FC",
    "Nottingham Forest": "Nottingham Forest",
    "Leeds United": "Leeds United",
    "Sunderland AFC": "Sunderland AFC",
    "Burnley FC": "Burnley FC",
}


def get_latest_raw_data_folder(country: str = "england") -> Optional[Path]:
    """Get the latest raw data folder."""
    base_dir = PRODUCTION_ROOT / "raw_data" / country.lower().replace(" ", "_")
    if not base_dir.exists():
        return None
    
    date_folders = [d for d in base_dir.iterdir() if d.is_dir() and d.name.isdigit() and len(d.name) == 8]
    if not date_folders:
        return None
    
    return max(date_folders, key=lambda x: x.name)


def load_our_player_ids(club_name: str) -> Set[int]:
    """Load player IDs from our config.json file."""
    config_path = DEPLOYMENTS_DIR / club_name / "config.json"
    
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        return set()
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        player_ids = set(config.get('player_ids', []))
        return player_ids
    except Exception as e:
        print(f"ERROR loading config: {e}")
        return set()


def fetch_transfermarkt_players(club_name: str, season_year: int = 2025) -> Dict[int, Dict]:
    """Fetch current players from Transfermarkt for a club."""
    scraper = TransfermarktScraper(ScraperConfig())
    
    # Get club name for Transfermarkt
    tm_club_name = CLUB_NAME_MAPPING.get(club_name, club_name)
    
    try:
        # Fetch clubs for the season
        clubs = scraper.fetch_league_clubs("premier-league", "GB1", season_year)
        
        # Find our club
        target_club = None
        for club in clubs:
            if club['club_name'] == tm_club_name:
                target_club = club
                break
        
        if not target_club:
            print(f"ERROR: Club {tm_club_name} not found on Transfermarkt")
            scraper.close()
            return {}
        
        print(f"Found {tm_club_name} on Transfermarkt (ID: {target_club['club_id']})")
        
        # Fetch players
        players = scraper.get_squad_players(
            target_club['club_slug'],
            target_club['club_id'],
            "kader",
            season_year
        )
        
        scraper.close()
        
        # Convert to dict with player_id as key
        players_dict = {}
        for player in players:
            player_id = player['player_id']
            players_dict[player_id] = {
                'player_id': player_id,
                'player_name': player.get('player_name', f'Player {player_id}'),
                'player_slug': player.get('player_slug'),
                'position': player.get('position', ''),
            }
        
        return players_dict
        
    except Exception as e:
        print(f"ERROR fetching from Transfermarkt: {e}")
        import traceback
        traceback.print_exc()
        return {}


def check_player_status(player_id: int, club_name: str, data_dir: Path) -> Optional[Dict]:
    """Check if a player was in the club at season start (2025-07-01) and current status."""
    # First check profile for current_club
    profile_path = data_dir / "players_profile.csv"
    current_club = None
    
    if profile_path.exists():
        try:
            profile_df = pd.read_csv(profile_path, sep=';', encoding='utf-8-sig', low_memory=False)
            player_id_col = 'player_id' if 'player_id' in profile_df.columns else 'id'
            player_profile = profile_df[profile_df[player_id_col] == player_id]
            
            if not player_profile.empty:
                current_club = player_profile.iloc[0].get('current_club', '')
        except Exception as e:
            pass
    
    # Check career data
    career_path = data_dir / "players_career.csv"
    
    if not career_path.exists():
        return {
            'current_club': current_club,
            'was_in_club_at_season_start': None,
        }
    
    try:
        career_df = pd.read_csv(career_path, sep=';', encoding='utf-8-sig', low_memory=False)
        
        # Filter for this player (career CSV uses 'id' column)
        player_id_col = 'player_id' if 'player_id' in career_df.columns else 'id'
        player_career = career_df[career_df[player_id_col] == player_id].copy()
        
        if player_career.empty:
            return {
                'current_club': current_club,
                'was_in_club_at_season_start': None,
            }
        
        # Convert Date column to datetime
        player_career['Date'] = pd.to_datetime(player_career['Date'], errors='coerce')
        
        # Season start date (2025-07-01)
        season_start = pd.Timestamp('2025-07-01')
        
        # Check if player was in this club at season start
        # Look for entries where Date <= season_start and club matches
        club_name_normalized = club_name.lower().strip()
        
        # Career CSV uses 'To' column for destination club
        club_col = 'To' if 'To' in player_career.columns else 'club'
        
        was_in_club_at_start = False
        last_date_in_club = None
        
        for _, row in player_career.iterrows():
            career_date = row.get('Date')
            career_club = str(row.get(club_col, '')).lower().strip()
            
            if pd.notna(career_date) and career_date <= season_start:
                if club_name_normalized in career_club or career_club in club_name_normalized:
                    # Player was in club at season start
                    was_in_club_at_start = True
                    if last_date_in_club is None or career_date > last_date_in_club:
                        last_date_in_club = career_date
        
        # Check latest entry for this player
        latest_club = None
        latest_date = None
        if not player_career.empty:
            latest_entry = player_career.sort_values('Date', na_position='last').iloc[-1]
            latest_club = latest_entry.get(club_col, '')
            latest_date = latest_entry.get('Date')
        
        # Use current_club from profile if available, otherwise use latest from career
        final_current_club = current_club if current_club else latest_club
        
        return {
            'current_club': final_current_club,
            'was_in_club_at_season_start': was_in_club_at_start,
            'last_date_in_club': last_date_in_club,
            'latest_date': latest_date,
            'latest_club': latest_club,
        }
        
    except Exception as e:
        return {
            'current_club': current_club,
            'was_in_club_at_season_start': None,
            'error': str(e),
        }


def analyze_club(club_name: str, data_date: str = "20260114"):
    """Analyze missing players for a specific club."""
    print("=" * 80)
    print(f"MISSING PLAYERS ANALYSIS: {club_name}")
    print("=" * 80)
    print(f"Data date: {data_date}")
    print()
    
    # Step 1: Load our player IDs
    print("STEP 1: Loading our player IDs from config.json...")
    our_player_ids = load_our_player_ids(club_name)
    print(f"  Found {len(our_player_ids)} players in our config")
    print()
    
    # Step 2: Fetch current players from Transfermarkt
    print("STEP 2: Fetching current players from Transfermarkt (2025/26 season)...")
    tm_players = fetch_transfermarkt_players(club_name, season_year=2025)
    tm_player_ids = set(tm_players.keys())
    print(f"  Found {len(tm_player_ids)} players on Transfermarkt")
    print()
    
    # Step 3: Identify missing players
    print("STEP 3: Identifying missing players...")
    missing_player_ids = tm_player_ids - our_player_ids
    extra_player_ids = our_player_ids - tm_player_ids
    
    print(f"  Players on Transfermarkt but NOT in our config: {len(missing_player_ids)}")
    print(f"  Players in our config but NOT on Transfermarkt: {len(extra_player_ids)}")
    print()
    
    # Step 4: Check career data for missing players
    print("STEP 4: Checking if missing players were in club at season start...")
    data_dir = get_latest_raw_data_folder("england")
    if data_dir:
        print(f"  Using data from: {data_dir.name}")
    else:
        print("  WARNING: No raw data folder found")
        data_dir = None
    
    print()
    
    # Step 5: Detailed analysis
    print("=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)
    print()
    
    # Missing players (on TM but not in our config)
    if missing_player_ids:
        print(f"MISSING PLAYERS ({len(missing_player_ids)}):")
        print(f"{'Player ID':<12} {'Player Name':<40} {'Status':<30} {'Details':<30}")
        print("-" * 112)
        
        still_in_squad = []
        left_club = []
        no_career_data = []
        
        for player_id in sorted(missing_player_ids):
            player_info = tm_players.get(player_id, {})
            player_name = player_info.get('player_name', f'Player {player_id}')
            
            if data_dir:
                player_info = check_player_status(player_id, club_name, data_dir)
                
                if player_info:
                    current_club = player_info.get('current_club', '')
                    was_in_club = player_info.get('was_in_club_at_season_start', False)
                    
                    # Check if player is currently in the club (from profile)
                    club_name_normalized = club_name.lower().strip()
                    current_club_normalized = str(current_club).lower().strip() if current_club else ''
                    is_currently_in_club = club_name_normalized in current_club_normalized or current_club_normalized in club_name_normalized
                    
                    if was_in_club and is_currently_in_club:
                        # Player was in club at season start and is still there - should be in our config
                        still_in_squad.append((player_id, player_name, player_info))
                        status = "[SHOULD ADD] Still in squad"
                        details = f"Was in club on {player_info.get('last_date_in_club', 'N/A')}"
                    elif was_in_club and not is_currently_in_club:
                        # Player was in club at season start but left
                        left_club.append((player_id, player_name, player_info))
                        status = "[LEFT CLUB]"
                        details = f"Now at: {current_club or 'Unknown'}"
                    elif not was_in_club and is_currently_in_club:
                        # Player joined after season start
                        left_club.append((player_id, player_name, player_info))
                        status = "[JOINED LATER]"
                        details = f"Joined after season start"
                    else:
                        # Player not currently in club
                        left_club.append((player_id, player_name, player_info))
                        status = "[NOT IN CLUB]"
                        details = f"Current: {current_club or 'Unknown'}"
                else:
                    no_career_data.append((player_id, player_name))
                    status = "[NO CAREER DATA]"
                    details = "No career data found"
            else:
                no_career_data.append((player_id, player_name))
                status = "[NO DATA FOLDER]"
                details = "Cannot check career"
            
            print(f"{player_id:<12} {player_name:<40} {status:<30} {details:<30}")
        
        print()
        print(f"Summary:")
        print(f"  - Still in squad (should add): {len(still_in_squad)}")
        print(f"  - Left club / joined later: {len(left_club)}")
        print(f"  - No career data: {len(no_career_data)}")
        print()
        
        if still_in_squad:
            print("PLAYERS TO ADD TO CONFIG:")
            for player_id, player_name, _ in still_in_squad:
                print(f"  - {player_id}: {player_name}")
            print()
    else:
        print("No missing players found!")
        print()
    
    # Extra players (in our config but not on TM)
    if extra_player_ids:
        print(f"EXTRA PLAYERS ({len(extra_player_ids)}):")
        print(f"{'Player ID':<12} {'Player Name':<40} {'Status':<30} {'Details':<30}")
        print("-" * 112)
        
        # Load player names from profile
        player_names = {}
        if data_dir:
            profile_path = data_dir / "players_profile.csv"
            if profile_path.exists():
                try:
                    profile_df = pd.read_csv(profile_path, sep=';', encoding='utf-8-sig', low_memory=False)
                    player_id_col = 'player_id' if 'player_id' in profile_df.columns else 'id'
                    name_col = 'name' if 'name' in profile_df.columns else 'player_name'
                    for _, row in profile_df.iterrows():
                        pid = row.get(player_id_col)
                        if pd.notna(pid) and int(pid) in extra_player_ids:
                            player_names[int(pid)] = row.get(name_col, f'Player {int(pid)}')
                except Exception as e:
                    print(f"  Warning: Could not load player names: {e}")
        
        for player_id in sorted(extra_player_ids):
            player_name = player_names.get(player_id, f'Player {player_id}')
            
            # Check if player left the club
            if data_dir:
                player_info = check_player_status(player_id, club_name, data_dir)
                if player_info:
                    current_club = player_info.get('current_club', '')
                    club_name_normalized = club_name.lower().strip()
                    current_club_normalized = str(current_club).lower().strip() if current_club else ''
                    is_currently_in_club = club_name_normalized in current_club_normalized or current_club_normalized in club_name_normalized
                    
                    if not is_currently_in_club:
                        status = "[LEFT CLUB] Can remove"
                        details = f"Now at: {current_club or 'Unknown'}"
                    else:
                        status = "[CHECK MANUALLY]"
                        details = "Still in club per data"
                else:
                    status = "[NO CAREER DATA]"
                    details = "Cannot verify"
            else:
                status = "[CHECK MANUALLY]"
                details = "No data folder"
            
            print(f"{player_id:<12} {player_name:<40} {status:<30} {details:<30}")
        print()
    else:
        print("No extra players found!")
        print()
    
    print("=" * 80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze missing players for a specific club'
    )
    parser.add_argument(
        '--club',
        type=str,
        required=True,
        help='Club name (e.g., "AFC Bournemouth")'
    )
    parser.add_argument(
        '--data-date',
        type=str,
        default=None,
        help='Data date (YYYYMMDD format, default: latest available)'
    )
    
    args = parser.parse_args()
    
    data_date = args.data_date or datetime.now().strftime("%Y%m%d")
    
    analyze_club(args.club, data_date)


if __name__ == "__main__":
    main()
