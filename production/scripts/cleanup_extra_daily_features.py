#!/usr/bin/env python3
"""
Remove daily features files for players that are no longer in config.json
"""

import json
from pathlib import Path

PRODUCTION_ROOT = Path(__file__).resolve().parent.parent
DEPLOYMENTS_DIR = PRODUCTION_ROOT / "deployments" / "England"

# Clubs that had players removed
CLUBS_TO_CHECK = ["Brentford FC", "Crystal Palace", "Liverpool FC"]

def main():
    print("=" * 80)
    print("CLEANING UP EXTRA DAILY FEATURES FILES")
    print("=" * 80)
    print()
    
    total_removed = 0
    
    for club_name in CLUBS_TO_CHECK:
        config_path = DEPLOYMENTS_DIR / club_name / "config.json"
        daily_features_dir = DEPLOYMENTS_DIR / club_name / "daily_features"
        
        if not config_path.exists() or not daily_features_dir.exists():
            print(f"[SKIP] {club_name}: Config or daily_features directory not found")
            continue
        
        # Load player IDs from config
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        valid_player_ids = set(config.get('player_ids', []))
        
        # Find all daily features files
        all_files = list(daily_features_dir.glob("player_*_daily_features.csv"))
        
        removed_count = 0
        for file_path in all_files:
            # Extract player ID from filename
            try:
                player_id = int(file_path.stem.replace('player_', '').replace('_daily_features', ''))
                if player_id not in valid_player_ids:
                    file_path.unlink()
                    print(f"  [REMOVED] {club_name}: player_{player_id}_daily_features.csv")
                    removed_count += 1
            except ValueError:
                print(f"  [ERROR] {club_name}: Could not parse player ID from {file_path.name}")
        
        print(f"[OK] {club_name}: Removed {removed_count} extra file(s)")
        total_removed += removed_count
        print()
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files removed: {total_removed}")
    print()

if __name__ == '__main__':
    main()







