#!/usr/bin/env python3
"""
Compare player counts per club between our configs and Transfermarkt.
"""

import json
import sys
import io
from pathlib import Path
from typing import Dict, List

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

# Transfermarkt data from the web page (2025/26 season)
# Source: https://www.transfermarkt.pt/premier-league/startseite/wettbewerb/GB1
TRANSFERMARKT_PLAYER_COUNTS = {
    "FC Arsenal": 25,
    "Manchester City FC": 29,
    "Chelsea FC": 32,
    "Liverpool FC": 26,
    "Tottenham Hotspur": 27,
    "Manchester United FC": 26,
    "Aston Villa": 28,
    "Newcastle United": 28,
    "Brighton & Hove Albion": 28,
    "West Ham United": 28,
    "Wolverhampton Wanderers": 28,
    "Crystal Palace FC": 28,
    "FC Fulham": 28,
    "AFC Bournemouth": 28,
    "Everton FC": 28,
    "Brentford FC": 28,
    "Nottingham Forest": 28,
    "Leeds United FC": 28,
    "AFC Sunderland": 28,
    "FC Burnley": 28,
}

# Club name mapping (our names -> Transfermarkt names)
CLUB_NAME_MAPPING = {
    "Arsenal FC": "FC Arsenal",
    "Manchester City": "Manchester City FC",
    "Chelsea FC": "Chelsea FC",
    "Liverpool FC": "Liverpool FC",
    "Tottenham Hotspur": "Tottenham Hotspur",
    "Manchester United": "Manchester United FC",
    "Aston Villa": "Aston Villa",
    "Newcastle United": "Newcastle United",
    "Brighton & Hove Albion": "Brighton & Hove Albion",
    "West Ham United": "West Ham United",
    "Wolverhampton Wanderers": "Wolverhampton Wanderers",
    "Crystal Palace": "Crystal Palace FC",
    "Fulham FC": "FC Fulham",
    "AFC Bournemouth": "AFC Bournemouth",
    "Everton FC": "Everton FC",
    "Brentford FC": "Brentford FC",
    "Nottingham Forest": "Nottingham Forest",
    "Leeds United": "Leeds United FC",
    "Sunderland AFC": "AFC Sunderland",
    "Burnley FC": "FC Burnley",
}


def load_our_player_counts() -> Dict[str, int]:
    """Load player counts from our config.json files."""
    our_counts = {}
    
    for club_folder in sorted(DEPLOYMENTS_DIR.iterdir()):
        if not club_folder.is_dir():
            continue
        
        config_path = club_folder / "config.json"
        if not config_path.exists():
            continue
        
        club_name = club_folder.name
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            player_ids = config.get('player_ids', [])
            our_counts[club_name] = len(player_ids)
        except Exception as e:
            print(f"ERROR loading {club_name}: {e}")
            our_counts[club_name] = 0
    
    return our_counts


def fetch_transfermarkt_counts() -> Dict[str, int]:
    """Fetch player counts from Transfermarkt for current season (2025/26)."""
    scraper = TransfermarktScraper(ScraperConfig())
    
    # Fetch clubs for 2025/26 season (season key 2025)
    season_year = 2025
    print(f"Fetching clubs from Transfermarkt for season {season_year-1}/{str(season_year)[-2:]}...")
    
    try:
        clubs = scraper.fetch_league_clubs("premier-league", "GB1", season_year)
        print(f"Found {len(clubs)} clubs")
        
        transfermarkt_counts = {}
        
        for club_idx, club in enumerate(clubs, 1):
            club_name = club['club_name']
            club_id = club['club_id']
            club_slug = club['club_slug']
            
            print(f"  [{club_idx}/{len(clubs)}] Fetching players from {club_name}...", end=" ", flush=True)
            
            try:
                players = scraper.get_squad_players(club_slug, club_id, "kader", season_year)
                player_count = len(players)
                transfermarkt_counts[club_name] = player_count
                print(f"[OK] {player_count} players")
            except Exception as e:
                print(f"[ERROR] {e}")
                transfermarkt_counts[club_name] = 0
        
        scraper.close()
        return transfermarkt_counts
        
    except Exception as e:
        print(f"ERROR fetching from Transfermarkt: {e}")
        print("Falling back to static data from web page...")
        return TRANSFERMARKT_PLAYER_COUNTS


def main():
    print("=" * 80)
    print("PLAYER COUNT COMPARISON: Our Configs vs Transfermarkt")
    print("=" * 80)
    print()
    
    # Load our player counts
    print("STEP 1: Loading player counts from our config.json files...")
    our_counts = load_our_player_counts()
    print(f"  Loaded {len(our_counts)} clubs")
    print()
    
    # Fetch Transfermarkt counts
    print("STEP 2: Fetching player counts from Transfermarkt...")
    transfermarkt_counts = fetch_transfermarkt_counts()
    print(f"  Loaded {len(transfermarkt_counts)} clubs")
    print()
    
    # Compare
    print("=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print(f"{'Club':<30} {'Our Count':<12} {'TM Count':<12} {'Difference':<12} {'Status':<15}")
    print("-" * 80)
    
    total_our = 0
    total_tm = 0
    missing_clubs = []
    extra_clubs = []
    
    # Sort clubs for consistent output
    all_clubs = sorted(set(list(our_counts.keys()) + list(transfermarkt_counts.keys())))
    
    for club_name in all_clubs:
        our_count = our_counts.get(club_name, 0)
        tm_name = CLUB_NAME_MAPPING.get(club_name, club_name)
        tm_count = transfermarkt_counts.get(tm_name, transfermarkt_counts.get(club_name, 0))
        
        if our_count == 0 and tm_count > 0:
            missing_clubs.append(club_name)
        elif our_count > 0 and tm_count == 0:
            extra_clubs.append(club_name)
        
        diff = our_count - tm_count
        total_our += our_count
        total_tm += tm_count
        
        if diff == 0:
            status = "[OK] Match"
        elif diff > 0:
            status = f"[+] +{diff} extra"
        else:
            status = f"[-] {diff} missing"
        
        print(f"{club_name:<30} {our_count:<12} {tm_count:<12} {diff:<12} {status:<15}")
    
    print("-" * 80)
    print(f"{'TOTAL':<30} {total_our:<12} {total_tm:<12} {total_our - total_tm:<12}")
    print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total players in our configs: {total_our}")
    print(f"Total players on Transfermarkt: {total_tm}")
    print(f"Difference: {total_our - total_tm} ({total_our - total_tm:+d})")
    print()
    
    if missing_clubs:
        print(f"Clubs in Transfermarkt but not in our configs: {len(missing_clubs)}")
        for club in missing_clubs:
            print(f"  - {club}")
        print()
    
    if extra_clubs:
        print(f"Clubs in our configs but not on Transfermarkt: {len(extra_clubs)}")
        for club in extra_clubs:
            print(f"  - {club}")
        print()
    
    # Clubs with mismatches
    mismatches = []
    for club_name in all_clubs:
        our_count = our_counts.get(club_name, 0)
        tm_name = CLUB_NAME_MAPPING.get(club_name, club_name)
        tm_count = transfermarkt_counts.get(tm_name, transfermarkt_counts.get(club_name, 0))
        
        if our_count > 0 and tm_count > 0 and our_count != tm_count:
            mismatches.append((club_name, our_count, tm_count))
    
    if mismatches:
        print(f"Clubs with player count mismatches: {len(mismatches)}")
        for club_name, our_count, tm_count in sorted(mismatches, key=lambda x: abs(x[1] - x[2]), reverse=True):
            diff = our_count - tm_count
            print(f"  - {club_name}: We have {our_count}, Transfermarkt has {tm_count} (difference: {diff:+d})")
    else:
        print("[OK] All clubs match!")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
