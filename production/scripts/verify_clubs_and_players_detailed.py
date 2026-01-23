#!/usr/bin/env python3
"""
Detailed verification:
1. Cross-reference clubs in deployments with clubs in raw data to find 3 extra clubs
2. For each of the 20 correct clubs, verify all players exist in players_profile.csv
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Paths
PRODUCTION_ROOT = Path(__file__).resolve().parent.parent
DEPLOYMENTS_DIR = PRODUCTION_ROOT / "deployments" / "England"
RAW_DATA_DIR = PRODUCTION_ROOT / "raw_data" / "england" / "20251226"
PLAYERS_PROFILE_FILE = RAW_DATA_DIR / "players_profile.csv"

def main():
    print("=" * 80)
    print("DETAILED VERIFICATION: Clubs and Players")
    print("=" * 80)
    print()
    
    # Step 1: Load clubs from deployments
    print("STEP 1: Loading clubs from deployments...")
    deployment_clubs = {}
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
            player_ids = set(config.get('player_ids', []))
            deployment_clubs[club_name] = player_ids
            print(f"  Loaded {club_name}: {len(player_ids)} players")
        except Exception as e:
            print(f"  ERROR loading {club_name}: {e}")
            continue
    
    print(f"\nTotal clubs in deployments: {len(deployment_clubs)}")
    print()
    
    # Step 2: Load clubs from raw data (players_profile.csv)
    print("STEP 2: Loading clubs from raw data (players_profile.csv)...")
    if not PLAYERS_PROFILE_FILE.exists():
        print(f"ERROR: Players profile file not found: {PLAYERS_PROFILE_FILE}")
        return
    
    try:
        # Read the CSV file
        df_players = pd.read_csv(PLAYERS_PROFILE_FILE, sep=';', encoding='utf-8-sig', low_memory=False)
        print(f"  Loaded {len(df_players)} players from players_profile.csv")
        
        # Group players by club
        players_by_club = defaultdict(set)
        for _, row in df_players.iterrows():
            club = row.get('current_club')
            player_id = row.get('id')
            if pd.notna(club) and pd.notna(player_id):
                players_by_club[str(club).strip()].add(int(player_id))
        
        print(f"  Found {len(players_by_club)} unique clubs in raw data")
        print()
        
    except Exception as e:
        print(f"ERROR loading players_profile.csv: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Identify 3 extra clubs
    print("=" * 80)
    print("STEP 3: Identifying Extra Clubs")
    print("=" * 80)
    print()
    
    # Find clubs in deployments that are NOT in raw data
    clubs_in_deployments = set(deployment_clubs.keys())
    clubs_in_raw_data = set(players_by_club.keys())
    
    # Try to match club names (they might have slight variations)
    matched_clubs = {}
    unmatched_deployment = []
    
    for dep_club in clubs_in_deployments:
        # Try exact match first
        if dep_club in clubs_in_raw_data:
            matched_clubs[dep_club] = dep_club
        else:
            # Try fuzzy matching (case-insensitive, remove extra spaces)
            found = False
            dep_club_normalized = dep_club.lower().strip()
            for raw_club in clubs_in_raw_data:
                raw_club_normalized = raw_club.lower().strip()
                if dep_club_normalized == raw_club_normalized:
                    matched_clubs[dep_club] = raw_club
                    found = True
                    break
            if not found:
                unmatched_deployment.append(dep_club)
    
    print(f"Clubs in deployments: {len(clubs_in_deployments)}")
    print(f"Clubs in raw data: {len(clubs_in_raw_data)}")
    print(f"Matched clubs: {len(matched_clubs)}")
    print()
    
    if unmatched_deployment:
        print(f"[EXTRA CLUBS] {len(unmatched_deployment)} clubs in deployments but NOT in raw data:")
        for club in sorted(unmatched_deployment):
            print(f"  - {club} ({len(deployment_clubs[club])} players)")
        print()
        print("These are the clubs that should be removed from deployments.")
    else:
        print("[OK] All clubs in deployments are found in raw data.")
    print()
    
    # Show all clubs in raw data
    print("All clubs in raw data (from players_profile.csv):")
    for club in sorted(clubs_in_raw_data):
        print(f"  - {club} ({len(players_by_club[club])} players)")
    print()
    
    # Step 4: Verify players for each of the 20 matched clubs
    print("=" * 80)
    print("STEP 4: Verifying Players for Each Club")
    print("=" * 80)
    print()
    
    all_raw_player_ids = set(df_players['id'].dropna().unique())
    print(f"Total unique player IDs in raw data: {len(all_raw_player_ids)}")
    print()
    
    clubs_with_missing_players = []
    clubs_with_extra_players = []
    clubs_perfect_match = []
    
    for dep_club, raw_club in sorted(matched_clubs.items()):
        dep_player_ids = deployment_clubs[dep_club]
        raw_player_ids = players_by_club.get(raw_club, set())
        
        missing_in_raw = dep_player_ids - raw_player_ids
        extra_in_raw = raw_player_ids - dep_player_ids
        
        if missing_in_raw:
            clubs_with_missing_players.append((dep_club, missing_in_raw))
        if extra_in_raw:
            clubs_with_extra_players.append((dep_club, extra_in_raw))
        if not missing_in_raw and not extra_in_raw:
            clubs_perfect_match.append(dep_club)
        
        status = "OK" if not missing_in_raw and not extra_in_raw else "ISSUES"
        print(f"{dep_club:<30} [Deploy: {len(dep_player_ids):>3}] [Raw: {len(raw_player_ids):>3}] [{status}]")
        if missing_in_raw:
            print(f"  {'':30} Missing in raw: {sorted(missing_in_raw)[:5]}{'...' if len(missing_in_raw) > 5 else ''}")
        if extra_in_raw:
            print(f"  {'':30} Extra in raw: {sorted(extra_in_raw)[:5]}{'...' if len(extra_in_raw) > 5 else ''}")
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total clubs in deployments: {len(deployment_clubs)}")
    print(f"Total clubs in raw data: {len(players_by_club)}")
    print(f"Matched clubs: {len(matched_clubs)}")
    print(f"Extra clubs (in deployments, not in raw): {len(unmatched_deployment)}")
    print()
    print(f"Clubs with perfect player match: {len(clubs_perfect_match)}")
    print(f"Clubs with missing players: {len(clubs_with_missing_players)}")
    print(f"Clubs with extra players in raw: {len(clubs_with_extra_players)}")
    print()
    
    if unmatched_deployment:
        print("=" * 80)
        print("RECOMMENDATION: Extra Clubs to Remove")
        print("=" * 80)
        for club in sorted(unmatched_deployment):
            print(f"  - {club}")
        print()
    
    # Detailed missing players report
    if clubs_with_missing_players:
        print("=" * 80)
        print("DETAILED: Clubs with Missing Players")
        print("=" * 80)
        for club, missing_ids in clubs_with_missing_players:
            print(f"\n{club}: {len(missing_ids)} missing players")
            print(f"  Player IDs: {sorted(missing_ids)}")

if __name__ == '__main__':
    main()







