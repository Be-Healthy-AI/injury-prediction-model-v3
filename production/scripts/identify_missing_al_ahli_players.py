#!/usr/bin/env python3
"""
Identify which 4 Al-Ahli SFC players are fetched but not processed in benchmarking.
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

from pathlib import Path
import pandas as pd

# Add root directory to path
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent
ROOT_DIR = PRODUCTION_ROOT.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.data_collection.transfermarkt_scraper import ScraperConfig, TransfermarktScraper

def get_latest_data_folder(country: str) -> Path:
    """Get the latest data folder for a country."""
    raw_data_dir = PRODUCTION_ROOT / "raw_data" / country.lower().replace(" ", "_")
    if not raw_data_dir.exists():
        return None
    
    date_folders = [d for d in raw_data_dir.iterdir() 
                   if d.is_dir() and d.name.isdigit() and len(d.name) == 8]
    if not date_folders:
        return None
    
    return max(date_folders, key=lambda x: x.name)

def main():
    print("=" * 80)
    print("Identify Missing Al-Ahli SFC Players")
    print("=" * 80)
    
    # Get latest data folder
    data_dir = get_latest_data_folder("saudi arabia")
    if data_dir is None:
        print("Error: No data folder found for Saudi Arabia")
        return
    
    print(f"\nUsing data folder: {data_dir}")
    
    # Load existing data
    profiles_path = data_dir / "players_profile.csv"
    career_path = data_dir / "players_career.csv"
    
    profiles_df = pd.DataFrame()
    career_df = pd.DataFrame()
    
    if profiles_path.exists():
        profiles_df = pd.read_csv(profiles_path, encoding='utf-8-sig', sep=';')
        print(f"Loaded {len(profiles_df)} profiles")
    
    if career_path.exists():
        career_df = pd.read_csv(career_path, encoding='utf-8-sig', sep=';')
        print(f"Loaded {len(career_df)} career records")
    
    # Initialize scraper
    scraper = TransfermarktScraper(ScraperConfig())
    
    # Fetch players from Al-Ahli SFC
    print("\nFetching players from Al-Ahli SFC for 2024/25 season...")
    clubs = scraper.fetch_league_clubs("saudi-pro-league", "SA1", 2024)
    
    al_ahli_club = None
    for club in clubs:
        if 'Al-Ahli' in club['club_name'] or 'Al Ahli' in club['club_name']:
            al_ahli_club = club
            break
    
    if not al_ahli_club:
        print("Error: Could not find Al-Ahli SFC")
        return
    
    players = scraper.get_squad_players(
        al_ahli_club['club_slug'], 
        al_ahli_club['club_id'], 
        "kader", 
        2024
    )
    
    print(f"Fetched {len(players)} players from Transfermarkt")
    
    # Get player IDs
    fetched_player_ids = {p['player_id']: p for p in players}
    
    # Check which players have profiles
    if not profiles_df.empty and 'id' in profiles_df.columns:
        profile_ids = set(profiles_df['id'].unique())
    else:
        profile_ids = set()
    
    # Check which players have career data
    if not career_df.empty and 'id' in career_df.columns:
        career_ids = set(career_df['id'].unique())
    else:
        career_ids = set()
    
    print(f"\nAnalysis:")
    print(f"  Players fetched from Transfermarkt: {len(fetched_player_ids)}")
    print(f"  Players with profiles: {len(profile_ids)}")
    print(f"  Players with career data: {len(career_ids)}")
    
    # Find missing players
    missing_profiles = []
    missing_careers = []
    
    for player_id, player_info in fetched_player_ids.items():
        player_name = player_info.get('player_name', f'Player {player_id}')
        if player_id not in profile_ids:
            missing_profiles.append((player_id, player_name))
        if player_id not in career_ids:
            missing_careers.append((player_id, player_name))
    
    print(f"\nMissing profiles: {len(missing_profiles)}")
    for player_id, player_name in missing_profiles:
        print(f"  - {player_name} (ID: {player_id})")
    
    print(f"\nMissing careers: {len(missing_careers)}")
    for player_id, player_name in missing_careers:
        print(f"  - {player_name} (ID: {player_id})")
    
    # Check which players are in both profiles and careers
    players_with_both = profile_ids & career_ids & set(fetched_player_ids.keys())
    print(f"\nPlayers with both profile and career data: {len(players_with_both)}")
    
    # The 4 missing players should be those that are fetched but not in players_with_both
    missing_from_processing = set(fetched_player_ids.keys()) - players_with_both
    if missing_from_processing:
        print(f"\nPlayers fetched but missing from processing (should be 4): {len(missing_from_processing)}")
        for player_id in missing_from_processing:
            player_info = fetched_player_ids[player_id]
            player_name = player_info.get('player_name', f'Player {player_id}')
            has_profile = player_id in profile_ids
            has_career = player_id in career_ids
            print(f"  - {player_name} (ID: {player_id}) - Profile: {has_profile}, Career: {has_career}")

if __name__ == "__main__":
    main()
