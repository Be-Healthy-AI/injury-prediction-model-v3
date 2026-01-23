#!/usr/bin/env python3
"""
Fetch missing career data for Al-Ahli SFC players in 2024/25 season.
Identifies players that were fetched from Transfermarkt but don't have career data,
then fetches their career data and updates the career CSV.
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

from pathlib import Path
from typing import Dict, Set, Optional
import pandas as pd

# Add root directory to path
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent
ROOT_DIR = PRODUCTION_ROOT.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.data_collection.transfermarkt_scraper import ScraperConfig, TransfermarktScraper
from scripts.data_collection.transformers import transform_career

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

def identify_missing_players(
    scraper: TransfermarktScraper,
    club_name: str,
    season_year: int,
    existing_career_df: pd.DataFrame,
    missing_player_ids: Optional[Set[int]] = None
) -> Dict[int, Dict[str, str]]:
    """
    Identify players from Al-Ahli SFC that don't have career data.
    Returns: {player_id: {'player_slug': slug, 'player_name': name, 'club_name': club}}
    """
    print(f"Fetching players from {club_name} for season {season_year}...")
    
    # Fetch clubs for the season
    clubs = scraper.fetch_league_clubs("saudi-pro-league", "SA1", season_year)
    
    # Find Al-Ahli SFC
    al_ahli_club = None
    for club in clubs:
        if 'Al-Ahli' in club['club_name'] or 'Al Ahli' in club['club_name']:
            al_ahli_club = club
            break
    
    if not al_ahli_club:
        print(f"  Error: Could not find {club_name} in league clubs")
        return {}
    
    # Fetch all players from Al-Ahli SFC
    players = scraper.get_squad_players(
        al_ahli_club['club_slug'], 
        al_ahli_club['club_id'], 
        "kader", 
        season_year
    )
    
    print(f"  Found {len(players)} players from {club_name}")
    
    # Check which players don't have career data
    existing_player_ids = set()
    if not existing_career_df.empty and 'id' in existing_career_df.columns:
        existing_player_ids = set(existing_career_df['id'].unique())
    
    missing_players = {}
    for player in players:
        player_id = player['player_id']
        # Include if: 1) missing career data, OR 2) in the specific missing_player_ids set
        if player_id not in existing_player_ids or (missing_player_ids and player_id in missing_player_ids):
            missing_players[player_id] = {
                'player_slug': player.get('player_slug'),
                'player_name': player.get('player_name', f'Player {player_id}'),
                'club_name': club_name
            }
    
    if missing_player_ids:
        print(f"  Found {len(missing_players)} players to fetch (including {len(missing_player_ids)} specific missing players)")
    else:
        print(f"  Found {len(missing_players)} players without career data")
    return missing_players

def fetch_career_for_players(
    scraper: TransfermarktScraper,
    missing_players: Dict[int, Dict[str, str]],
    existing_career_path: Path
) -> None:
    """Fetch career data for missing players and update the career CSV."""
    if not missing_players:
        print("No missing players to fetch.")
        return
    
    # Load existing career data
    existing_careers = []
    if existing_career_path.exists():
        try:
            existing_career_df = pd.read_csv(existing_career_path, encoding='utf-8-sig', sep=';')
            existing_careers.append(existing_career_df)
            print(f"  Loaded {len(existing_career_df)} existing career records")
        except Exception as e:
            print(f"  Warning: Could not load existing careers: {e}")
    
    # Fetch career data for missing players
    new_careers = []
    for idx, (player_id, player_info) in enumerate(missing_players.items(), 1):
        player_slug = player_info['player_slug']
        player_name = player_info['player_name']
        
        print(f"\n  [{idx}/{len(missing_players)}] Fetching career for {player_name} (ID: {player_id})...")
        
        try:
            print(f"    Fetching career...", end=" ", flush=True)
            career_df = scraper.fetch_player_career(player_slug, player_id)
            transformed = transform_career(player_id, player_name, career_df)
            new_careers.append(transformed)
            print(f"[OK] ({len(career_df)} transfers)")
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()
            # Still add empty career to maintain consistency
            empty_career = transform_career(player_id, player_name, pd.DataFrame())
            new_careers.append(empty_career)
    
    # Combine and save
    if new_careers:
        new_careers_df = pd.concat(new_careers, ignore_index=True)
        if existing_careers:
            all_careers = pd.concat(existing_careers + [new_careers_df], ignore_index=True)
        else:
            all_careers = new_careers_df
        
        # Remove duplicates (keep last)
        all_careers = all_careers.drop_duplicates(subset=['id', 'Date', 'From', 'To'], keep='last')
        
        all_careers.to_csv(existing_career_path, index=False, encoding='utf-8-sig', sep=';')
        print(f"\n  [OK] Updated career file: {existing_career_path}")
        print(f"    Total career records: {len(all_careers)}")
        print(f"    New career records: {len(new_careers_df)}")

def main():
    print("=" * 80)
    print("Fetch Missing Career Data for Al-Ahli SFC Players (2024/25)")
    print("=" * 80)
    
    # Get latest data folder
    data_dir = get_latest_data_folder("saudi arabia")
    if data_dir is None:
        print("Error: No data folder found for Saudi Arabia")
        return
    
    print(f"\nUsing data folder: {data_dir}")
    
    # Load existing career data
    career_path = data_dir / "players_career.csv"
    existing_career_df = pd.DataFrame()
    if career_path.exists():
        try:
            existing_career_df = pd.read_csv(career_path, encoding='utf-8-sig', sep=';')
            print(f"Loaded {len(existing_career_df)} existing career records")
        except Exception as e:
            print(f"Warning: Could not load existing careers: {e}")
    
    # Initialize scraper
    scraper = TransfermarktScraper(ScraperConfig())
    
    # The 4 specific player IDs that are missing from mapping
    missing_player_ids = {478528, 796803, 1061025, 1130616}
    
    # Identify missing players (including the 4 specific ones)
    missing_players = identify_missing_players(
        scraper,
        "Al-Ahli SFC",
        2024,  # Season year for 2024/25
        existing_career_df,
        missing_player_ids
    )
    
    if not missing_players:
        print("\n[OK] All players already have career data!")
        return
    
    print(f"\nMissing players to fetch:")
    for player_id, info in missing_players.items():
        print(f"  - {info['player_name']} (ID: {player_id})")
    
    # Fetch career data
    fetch_career_for_players(scraper, missing_players, career_path)
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)

if __name__ == "__main__":
    main()
