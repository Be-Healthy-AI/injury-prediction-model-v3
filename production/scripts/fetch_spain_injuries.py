#!/usr/bin/env python3
"""
Fetch fresh injury data for all players in LaLiga (2024/25) and LaLiga2 (2025/26).

This script:
1. Identifies all clubs in LaLiga 2024/25 and LaLiga2 2025/26
2. Identifies all players in those clubs
3. Fetches injury data for each player from Transfermarkt
4. Saves consolidated injury data to spain/raw_data/{date}/injuries_data.csv
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set
import pandas as pd
import json

# Add root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent
ROOT_DIR = PRODUCTION_ROOT.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.data_collection.transfermarkt_scraper import (
    ScraperConfig,
    TransfermarktScraper,
)
from scripts.data_collection.transformers import transform_injuries

# Season definitions (same as in analyze_spain_benchmarking.py)
SEASONS = {
    "2024/25": {
        "start": pd.Timestamp("2024-07-01"),
        "end": pd.Timestamp("2025-06-30"),
        "year": 2024,
        "league": "LaLiga",
        "competition_slug": "laliga",
        "competition_id": "ES1"
    },
    "2025/26": {
        "start": pd.Timestamp("2025-07-01"),
        "end": pd.Timestamp("2026-06-30"),
        "year": 2025,
        "league": "LaLiga2",
        "competition_slug": "laliga2",
        "competition_id": "ES2"
    }
}

def fetch_players_from_transfermarkt(
    season_name: str,
    season_info: Dict,
    scraper: TransfermarktScraper
) -> Dict[int, Dict[str, any]]:
    """
    Fetch all players and their clubs directly from Transfermarkt for a given season.
    Returns: {player_id: {'club_name': str, 'player_slug': str, 'player_name': str}}
    """
    competition_slug = season_info['competition_slug']
    competition_id = season_info['competition_id']
    season_year = season_info['year']
    
    print(f"Fetching clubs from Transfermarkt for {season_name}...")
    clubs = scraper.fetch_league_clubs(competition_slug, competition_id, season_year)
    print(f"  Found {len(clubs)} clubs")
    
    players_info = {}
    
    for club_idx, club in enumerate(clubs, 1):
        club_name = club['club_name']
        club_id = club['club_id']
        club_slug = club['club_slug']
        
        # Exclude B teams, U19, etc.
        exclude_patterns = [' B', ' U19', ' U23', ' U21', 'Promesas', 'Vetusta', 
                           'Mirandilla', 'Castilla', 'Juvenil', 'Youth', 'Reserve',
                           'Atlético B', 'Atletico B', 'Barcelona B', 'Real Madrid Castilla']
        should_exclude = any(pattern in club_name for pattern in exclude_patterns)
        if should_exclude:
            continue
        
        print(f"  [{club_idx}/{len(clubs)}] Fetching players from {club_name}...", end=" ", flush=True)
        
        try:
            players = scraper.get_squad_players(club_slug, club_id, "kader", season_year)
            print(f"[OK] {len(players)} players")
            
            for player in players:
                player_id = player['player_id']
                player_slug = player.get('player_slug')
                player_name = player.get('player_name', f'Player {player_id}')
                
                players_info[player_id] = {
                    'club_name': club_name,
                    'player_slug': player_slug,
                    'player_name': player_name,
                    'season': season_name
                }
        except Exception as e:
            print(f"[ERROR] {e}")
            continue
    
    return players_info


def fetch_injuries_for_players(
    players_info: Dict[int, Dict[str, any]],
    scraper: TransfermarktScraper,
    injury_mappings: Dict = None
) -> List[pd.DataFrame]:
    """
    Fetch injury data for all players.
    Returns list of transformed injury DataFrames.
    """
    all_injuries = []
    total_players = len(players_info)
    
    for player_idx, (player_id, player_info) in enumerate(players_info.items(), 1):
        player_name = player_info['player_name']
        player_slug = player_info.get('player_slug')
        club_name = player_info['club_name']
        season = player_info['season']
        
        print(f"\n  [{player_idx}/{total_players}] {player_name} ({club_name}, {season})")
        print(f"    Fetching injuries...", end=" ", flush=True)
        
        try:
            injuries_df = scraper.fetch_player_injuries(player_slug, player_id)
            
            if not injuries_df.empty:
                # Transform injuries
                transformed_injuries = transform_injuries(player_id, injuries_df, injury_mappings)
                all_injuries.append(transformed_injuries)
                print(f"[OK] {len(injuries_df)} injuries")
            else:
                print(f"[OK] No injuries")
                
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return all_injuries


def main():
    print("=" * 80)
    print("Fetching Fresh Injury Data for Spain (LaLiga 2024/25 & LaLiga2 2025/26)")
    print("=" * 80)
    print()
    
    # Setup output directory
    as_of_date = datetime.today()
    output_dir = PRODUCTION_ROOT / "raw_data" / "spain" / as_of_date.strftime("%Y%m%d")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize scraper
    scraper = TransfermarktScraper(ScraperConfig())
    
    try:
        # Load injury mappings if available
        injury_mappings = None
        mapping_file = PRODUCTION_ROOT / "config" / "injury_mappings.json"
        if mapping_file.exists():
            with open(mapping_file, 'r', encoding='utf-8') as f:
                injury_mappings = json.load(f)
            print(f"Loaded {len(injury_mappings)} injury type mappings")
        else:
            print(f"Warning: Injury mappings file not found: {mapping_file}")
        
        # Step 1: Identify all players for both seasons
        all_players_info = {}
        
        for season_name, season_info in SEASONS.items():
            print(f"\n{'='*80}")
            print(f"Processing Season: {season_name} ({season_info['league']})")
            print(f"{'='*80}")
            
            players_info = fetch_players_from_transfermarkt(season_name, season_info, scraper)
            print(f"  Found {len(players_info)} players")
            
            # Merge into all_players_info (handle duplicates - same player in both seasons)
            for player_id, info in players_info.items():
                if player_id in all_players_info:
                    # Player appears in both seasons - update season info
                    existing_seasons = all_players_info[player_id].get('seasons', [])
                    if season_name not in existing_seasons:
                        existing_seasons.append(season_name)
                    all_players_info[player_id]['seasons'] = existing_seasons
                else:
                    info['seasons'] = [season_name]
                    all_players_info[player_id] = info
        
        print(f"\n{'='*80}")
        print(f"Total unique players across both seasons: {len(all_players_info)}")
        print(f"{'='*80}")
        
        # Step 2: Fetch injuries for all players
        print(f"\n{'='*80}")
        print("Fetching Injury Data")
        print(f"{'='*80}")
        
        all_injuries = fetch_injuries_for_players(all_players_info, scraper, injury_mappings)
        
        # Step 3: Consolidate and save
        print(f"\n{'='*80}")
        print("Saving Results")
        print(f"{'='*80}")
        
        if all_injuries:
            consolidated_injuries = pd.concat(all_injuries, ignore_index=True)
            output_path = output_dir / "injuries_data.csv"
            consolidated_injuries.to_csv(output_path, index=False, encoding="utf-8-sig", sep=';')
            print(f"  Saved {len(consolidated_injuries)} injury records to: {output_path}")
            print(f"  Unique players with injuries: {consolidated_injuries['id'].nunique()}")
        else:
            print("  No injuries found")
    
    finally:
        scraper.close()
    
    print(f"\n{'='*80}")
    print("Complete!")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()

