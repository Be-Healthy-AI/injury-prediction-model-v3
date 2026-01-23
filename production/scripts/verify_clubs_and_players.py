#!/usr/bin/env python3
"""
Verify clubs and players in deployments vs Premier League.
"""

import json
from pathlib import Path
from collections import defaultdict

# Paths
PRODUCTION_ROOT = Path(__file__).resolve().parent.parent
DEPLOYMENTS_DIR = PRODUCTION_ROOT / "deployments" / "England"

# Premier League clubs (2025-2026 season) - 20 clubs
PREMIER_LEAGUE_CLUBS = {
    "Arsenal FC", "Aston Villa", "AFC Bournemouth", "Brentford FC", 
    "Brighton & Hove Albion", "Burnley FC", "Chelsea FC", "Crystal Palace",
    "Everton FC", "Fulham FC", "Ipswich Town", "Leicester City",
    "Liverpool FC", "Luton Town", "Manchester City", "Manchester United",
    "Newcastle United", "Nottingham Forest", "Sheffield United", 
    "Southampton FC", "Tottenham Hotspur", "West Ham United",
    "Wolverhampton Wanderers"
}

def main():
    print("=" * 80)
    print("VERIFICATION: Clubs and Players in Deployments")
    print("=" * 80)
    print()
    
    # Get all club folders
    club_folders = [d for d in DEPLOYMENTS_DIR.iterdir() if d.is_dir() and (d / "config.json").exists()]
    
    print(f"Total club folders found: {len(club_folders)}")
    print()
    
    # Load all configs and collect player IDs
    all_player_ids = set()
    club_player_counts = {}
    clubs_in_pl = []
    clubs_not_in_pl = []
    
    for club_folder in sorted(club_folders):
        club_name = club_folder.name
        config_path = club_folder / "config.json"
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            player_ids = set(config.get('player_ids', []))
            club_player_counts[club_name] = len(player_ids)
            all_player_ids.update(player_ids)
            
            if club_name in PREMIER_LEAGUE_CLUBS:
                clubs_in_pl.append(club_name)
            else:
                clubs_not_in_pl.append(club_name)
        except Exception as e:
            print(f"ERROR: Could not read config for {club_name}: {e}")
            continue
    
    print("=" * 80)
    print("QUESTION 1: Unique Player IDs Across All Club Configs")
    print("=" * 80)
    print(f"Total unique player IDs: {len(all_player_ids)}")
    print()
    
    # Show breakdown by club
    print("Player count per club:")
    for club_name in sorted(club_player_counts.keys()):
        count = club_player_counts[club_name]
        status = "[PL]" if club_name in PREMIER_LEAGUE_CLUBS else "[NOT PL]"
        print(f"  {club_name:<30} {count:>3} players  {status}")
    print()
    
    print("=" * 80)
    print("QUESTION 2: Compare to 543 Players Fetched")
    print("=" * 80)
    print(f"Players in configs: {len(all_player_ids)}")
    print(f"Players fetched: 543")
    print(f"Difference: {len(all_player_ids) - 543}")
    print()
    
    if len(all_player_ids) == 543:
        print("[OK] Match! We have exactly 543 players in configs.")
    elif len(all_player_ids) > 543:
        print(f"[WARNING] We have {len(all_player_ids) - 543} MORE players in configs than fetched.")
    else:
        print(f"[WARNING] We have {543 - len(all_player_ids)} FEWER players in configs than fetched.")
    print()
    
    print("=" * 80)
    print("QUESTION 3: Identify Clubs Not in Current Premier League")
    print("=" * 80)
    print(f"Clubs in Premier League (20): {len(clubs_in_pl)}")
    print(f"Clubs NOT in Premier League: {len(clubs_not_in_pl)}")
    print()
    
    if clubs_in_pl:
        print("[OK] Clubs that ARE in Premier League:")
        for club in sorted(clubs_in_pl):
            print(f"  - {club} ({club_player_counts[club]} players)")
        print()
    
    if clubs_not_in_pl:
        print("[EXTRA] Clubs that are NOT in Premier League:")
        for club in sorted(clubs_not_in_pl):
            print(f"  - {club} ({club_player_counts[club]} players)")
        print()
        print(f"These {len(clubs_not_in_pl)} clubs should be removed or are incorrectly included.")
    else:
        print("[OK] All clubs in deployments are Premier League clubs.")
    print()
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total clubs in deployments: {len(club_folders)}")
    print(f"  - Premier League clubs: {len(clubs_in_pl)}")
    print(f"  - Non-Premier League clubs: {len(clubs_not_in_pl)}")
    print(f"Total unique players: {len(all_player_ids)}")
    print(f"Expected players (from fetch): 543")
    print()
    
    if len(clubs_in_pl) == 20 and len(clubs_not_in_pl) == 3 and len(all_player_ids) == 543:
        print("[OK] All checks passed! We have exactly 20 PL clubs and 543 players.")
    else:
        print("[WARNING] Discrepancies found. See details above.")

if __name__ == '__main__':
    main()

