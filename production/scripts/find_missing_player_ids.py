#!/usr/bin/env python3
"""
Find the 4 missing Al-Ahli SFC player IDs.
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

from pathlib import Path

# Add root directory to path
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent
ROOT_DIR = PRODUCTION_ROOT.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.data_collection.transfermarkt_scraper import ScraperConfig, TransfermarktScraper

def main():
    # Initialize scraper
    scraper = TransfermarktScraper(ScraperConfig())
    
    # Fetch all Al-Ahli players
    clubs = scraper.fetch_league_clubs("saudi-pro-league", "SA1", 2024)
    
    al_ahli_club = None
    for club in clubs:
        if 'Al-Ahli' in club['club_name'] or 'Al Ahli' in club['club_name']:
            al_ahli_club = club
            break
    
    players = scraper.get_squad_players(
        al_ahli_club['club_slug'], 
        al_ahli_club['club_id'], 
        "kader", 
        2024
    )
    
    all_player_ids = {p['player_id']: p.get('player_name', f"Player {p['player_id']}") for p in players}
    
    # The 34 mapped player IDs from the benchmarking output
    mapped_ids = {129604, 131789, 171424, 193159, 251664, 294808, 340879, 369090, 442531, 454121, 
                  478531, 509898, 509912, 524481, 554034, 631922, 654918, 672391, 699704, 743946, 
                  821382, 901076, 901079, 901336, 922536, 951628, 997714, 1020859, 1178314, 1180065, 
                  1188907, 1229318, 1259162, 1340559}
    
    missing_ids = set(all_player_ids.keys()) - mapped_ids
    
    print(f"Total players fetched: {len(all_player_ids)}")
    print(f"Players mapped: {len(mapped_ids)}")
    print(f"Missing players: {len(missing_ids)}")
    print("\nMissing player IDs and names:")
    for player_id in sorted(missing_ids):
        player_name = all_player_ids[player_id]
        print(f"  - {player_name} (ID: {player_id})")

if __name__ == "__main__":
    main()
