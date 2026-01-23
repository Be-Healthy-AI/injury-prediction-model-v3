#!/usr/bin/env python3
"""
Fix clubs and players:
1. Remove 5 extra clubs
2. Add 2 missing clubs (Leeds United, Sunderland AFC)
3. Remove missing players from configs
"""

import json
import pandas as pd
import shutil
from pathlib import Path
from collections import defaultdict

# Paths
PRODUCTION_ROOT = Path(__file__).resolve().parent.parent
DEPLOYMENTS_DIR = PRODUCTION_ROOT / "deployments" / "England"
RAW_DATA_DIR = PRODUCTION_ROOT / "raw_data" / "england" / "20251226"
PLAYERS_PROFILE_FILE = RAW_DATA_DIR / "players_profile.csv"

# Clubs to remove
CLUBS_TO_REMOVE = [
    "Ipswich Town",
    "Leicester City",
    "Luton Town",
    "Sheffield United",
    "Southampton FC"
]

# Clubs to add
CLUBS_TO_ADD = [
    "Leeds United",
    "Sunderland AFC"
]

# Missing players to remove (club -> list of player IDs)
MISSING_PLAYERS = {
    "Brentford FC": [1144158],
    "Crystal Palace": [860078, 1004679, 1047256],
    "Liverpool FC": [340950]
}

def main():
    print("=" * 80)
    print("FIXING CLUBS AND PLAYERS")
    print("=" * 80)
    print()
    
    # Step 1: Remove 5 extra clubs
    print("=" * 80)
    print("STEP 1: Removing Extra Clubs")
    print("=" * 80)
    print()
    
    removed_count = 0
    for club_name in CLUBS_TO_REMOVE:
        club_dir = DEPLOYMENTS_DIR / club_name
        if club_dir.exists():
            try:
                shutil.rmtree(club_dir)
                print(f"[REMOVED] {club_name}")
                removed_count += 1
            except Exception as e:
                print(f"[ERROR] Failed to remove {club_name}: {e}")
        else:
            print(f"[SKIP] {club_name} - folder not found")
    
    print(f"\nRemoved {removed_count}/{len(CLUBS_TO_REMOVE)} clubs")
    print()
    
    # Step 2: Load raw data to get players for new clubs
    print("=" * 80)
    print("STEP 2: Adding Missing Clubs")
    print("=" * 80)
    print()
    
    if not PLAYERS_PROFILE_FILE.exists():
        print(f"ERROR: Players profile file not found: {PLAYERS_PROFILE_FILE}")
        return
    
    try:
        df_players = pd.read_csv(PLAYERS_PROFILE_FILE, sep=';', encoding='utf-8-sig', low_memory=False)
        print(f"Loaded {len(df_players)} players from players_profile.csv")
        
        # Group players by club
        players_by_club = defaultdict(set)
        for _, row in df_players.iterrows():
            club = row.get('current_club')
            player_id = row.get('id')
            if pd.notna(club) and pd.notna(player_id):
                players_by_club[str(club).strip()].add(int(player_id))
        
    except Exception as e:
        print(f"ERROR loading players_profile.csv: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create configs for new clubs
    added_count = 0
    for club_name in CLUBS_TO_ADD:
        if club_name not in players_by_club:
            print(f"[SKIP] {club_name} - not found in raw data")
            continue
        
        club_dir = DEPLOYMENTS_DIR / club_name
        if club_dir.exists():
            print(f"[SKIP] {club_name} - already exists")
            continue
        
        player_ids = sorted(list(players_by_club[club_name]))
        
        # Create club directory
        club_dir.mkdir(parents=True, exist_ok=True)
        
        # Create config.json
        config = {
            "club_name": club_name,
            "country": "England",
            "player_ids": player_ids,
            "models": {
                "lgbm_muscular_v2": "../../../models/lgbm_muscular_v2/model.joblib",
                "lgbm_columns": "../../../models/lgbm_muscular_v2/columns.json"
            },
            "default_date_range_days": 7
        }
        
        config_path = club_dir / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        # Create empty directories
        (club_dir / "daily_features").mkdir(exist_ok=True)
        (club_dir / "timelines").mkdir(exist_ok=True)
        (club_dir / "predictions").mkdir(exist_ok=True)
        (club_dir / "dashboards").mkdir(exist_ok=True)
        
        print(f"[ADDED] {club_name} - {len(player_ids)} players")
        added_count += 1
    
    print(f"\nAdded {added_count}/{len(CLUBS_TO_ADD)} clubs")
    print()
    
    # Step 3: Fix missing players in configs
    print("=" * 80)
    print("STEP 3: Removing Missing Players from Configs")
    print("=" * 80)
    print()
    
    fixed_count = 0
    for club_name, missing_player_ids in MISSING_PLAYERS.items():
        config_path = DEPLOYMENTS_DIR / club_name / "config.json"
        
        if not config_path.exists():
            print(f"[SKIP] {club_name} - config not found")
            continue
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            original_count = len(config['player_ids'])
            config['player_ids'] = [pid for pid in config['player_ids'] if pid not in missing_player_ids]
            new_count = len(config['player_ids'])
            removed_players = original_count - new_count
            
            if removed_players > 0:
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2)
                print(f"[FIXED] {club_name} - removed {removed_players} missing players: {missing_player_ids}")
                fixed_count += 1
            else:
                print(f"[SKIP] {club_name} - no missing players found")
                
        except Exception as e:
            print(f"[ERROR] Failed to fix {club_name}: {e}")
    
    print(f"\nFixed {fixed_count}/{len(MISSING_PLAYERS)} clubs")
    print()
    
    # Final summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Removed clubs: {removed_count}/{len(CLUBS_TO_REMOVE)}")
    print(f"Added clubs: {added_count}/{len(CLUBS_TO_ADD)}")
    print(f"Fixed clubs: {fixed_count}/{len(MISSING_PLAYERS)}")
    print()
    
    # Verify final state
    remaining_clubs = [d.name for d in DEPLOYMENTS_DIR.iterdir() if d.is_dir() and (d / "config.json").exists()]
    print(f"Total clubs remaining: {len(remaining_clubs)}")
    print(f"Clubs: {', '.join(sorted(remaining_clubs))}")
    print()

if __name__ == '__main__':
    main()

