#!/usr/bin/env python3
"""
Diagnose why 4 Al-Ahli SFC players are missing from benchmarking.
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
    print("Diagnose Missing Al-Ahli SFC Players")
    print("=" * 80)
    
    # Get latest data folder
    data_dir = get_latest_data_folder("saudi arabia")
    if data_dir is None:
        print("Error: No data folder found for Saudi Arabia")
        return
    
    print(f"\nUsing data folder: {data_dir}")
    
    # Initialize scraper
    scraper = TransfermarktScraper(ScraperConfig())
    
    # Simulate the two fetches that happen in benchmarking
    print("\n" + "=" * 80)
    print("Fetch 1: identify_players_in_league_clubs (all clubs)")
    print("=" * 80)
    
    clubs1 = scraper.fetch_league_clubs("saudi-pro-league", "SA1", 2024)
    print(f"Found {len(clubs1)} clubs")
    
    all_players_fetch1 = {}
    for club in clubs1:
        club_name = club['club_name']
        if 'Al-Ahli' in club_name or 'Al Ahli' in club_name:
            players = scraper.get_squad_players(club['club_slug'], club['club_id'], "kader", 2024)
            for player in players:
                all_players_fetch1[player['player_id']] = {
                    'name': player.get('player_name', f"Player {player['player_id']}"),
                    'club': club_name
                }
            print(f"  Al-Ahli SFC: {len(players)} players")
    
    print(f"\nTotal players from all clubs (fetch 1): {len(all_players_fetch1)}")
    al_ahli_fetch1 = {pid: info for pid, info in all_players_fetch1.items() if 'Al-Ahli' in info['club']}
    print(f"Al-Ahli players in fetch 1: {len(al_ahli_fetch1)}")
    
    print("\n" + "=" * 80)
    print("Fetch 2: map_players_to_clubs_by_season (all clubs again)")
    print("=" * 80)
    
    clubs2 = scraper.fetch_league_clubs("saudi-pro-league", "SA1", 2024)
    print(f"Found {len(clubs2)} clubs")
    
    all_players_fetch2 = {}
    for club in clubs2:
        club_name = club['club_name']
        if 'Al-Ahli' in club_name or 'Al Ahli' in club_name:
            players = scraper.get_squad_players(club['club_slug'], club['club_id'], "kader", 2024)
            for player in players:
                all_players_fetch2[player['player_id']] = {
                    'name': player.get('player_name', f"Player {player['player_id']}"),
                    'club': club_name
                }
            print(f"  Al-Ahli SFC: {len(players)} players")
    
    print(f"\nTotal players from all clubs (fetch 2): {len(all_players_fetch2)}")
    al_ahli_fetch2 = {pid: info for pid, info in all_players_fetch2.items() if 'Al-Ahli' in info['club']}
    print(f"Al-Ahli players in fetch 2: {len(al_ahli_fetch2)}")
    
    # Compare
    print("\n" + "=" * 80)
    print("Comparison")
    print("=" * 80)
    
    fetch1_ids = set(al_ahli_fetch1.keys())
    fetch2_ids = set(al_ahli_fetch2.keys())
    
    in_fetch1_not_fetch2 = fetch1_ids - fetch2_ids
    in_fetch2_not_fetch1 = fetch2_ids - fetch1_ids
    in_both = fetch1_ids & fetch2_ids
    
    print(f"Players in fetch 1: {len(fetch1_ids)}")
    print(f"Players in fetch 2: {len(fetch2_ids)}")
    print(f"Players in both: {len(in_both)}")
    print(f"Players in fetch 1 but not fetch 2: {len(in_fetch1_not_fetch2)}")
    print(f"Players in fetch 2 but not fetch 1: {len(in_fetch2_not_fetch1)}")
    
    if in_fetch1_not_fetch2:
        print("\nPlayers in fetch 1 but not fetch 2:")
        for pid in in_fetch1_not_fetch2:
            info = al_ahli_fetch1[pid]
            print(f"  - {info['name']} (ID: {pid})")
    
    if in_fetch2_not_fetch1:
        print("\nPlayers in fetch 2 but not fetch 1:")
        for pid in in_fetch2_not_fetch1:
            info = al_ahli_fetch2[pid]
            print(f"  - {info['name']} (ID: {pid})")
    
    # Now simulate the filtering in map_players_to_clubs_by_season
    print("\n" + "=" * 80)
    print("Simulating map_players_to_clubs_by_season filtering")
    print("=" * 80)
    
    # target_players would be all players from fetch 1
    target_players = set(all_players_fetch1.keys())
    print(f"target_players set size: {len(target_players)}")
    
    # Filter fetch 2 to only target_players
    filtered_fetch2 = {pid: info for pid, info in all_players_fetch2.items() if pid in target_players}
    print(f"After filtering fetch 2 to target_players: {len(filtered_fetch2)} players")
    
    al_ahli_filtered = {pid: info for pid, info in filtered_fetch2.items() if 'Al-Ahli' in info['club']}
    print(f"Al-Ahli players after filtering: {len(al_ahli_filtered)}")
    
    missing_after_filter = fetch1_ids - set(al_ahli_filtered.keys())
    if missing_after_filter:
        print(f"\nAl-Ahli players missing after filtering (should be 4): {len(missing_after_filter)}")
        for pid in missing_after_filter:
            info = al_ahli_fetch1[pid]
            print(f"  - {info['name']} (ID: {pid})")

if __name__ == "__main__":
    main()
