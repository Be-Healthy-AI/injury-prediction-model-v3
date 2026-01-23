#!/usr/bin/env python3
"""
Copy daily features files for Leeds United and Sunderland AFC players
from daily_features_output to their respective club folders.
"""

import json
import shutil
from pathlib import Path

# Paths
PRODUCTION_ROOT = Path(__file__).resolve().parent.parent
ROOT_DIR = PRODUCTION_ROOT.parent
SOURCE_DIR = ROOT_DIR / "daily_features_output"
DEPLOYMENTS_DIR = PRODUCTION_ROOT / "deployments" / "England"

# Clubs to check
CLUBS_TO_CHECK = ["Leeds United", "Sunderland AFC"]

def main():
    print("=" * 80)
    print("COPYING MISSING DAILY FEATURES FILES")
    print("=" * 80)
    print()
    
    if not SOURCE_DIR.exists():
        print(f"ERROR: Source directory not found: {SOURCE_DIR}")
        return 1
    
    print(f"Source directory: {SOURCE_DIR}")
    print()
    
    total_copied = 0
    total_missing = 0
    total_errors = 0
    
    for club_name in CLUBS_TO_CHECK:
        print("=" * 80)
        print(f"Processing: {club_name}")
        print("=" * 80)
        print()
        
        # Load player IDs from config
        config_path = DEPLOYMENTS_DIR / club_name / "config.json"
        if not config_path.exists():
            print(f"ERROR: Config file not found: {config_path}")
            continue
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        player_ids = config.get('player_ids', [])
        
        print(f"Player IDs in config: {len(player_ids)}")
        print()
        
        # Destination directory
        dest_dir = DEPLOYMENTS_DIR / club_name / "daily_features"
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        copied = 0
        missing = 0
        errors = 0
        
        for player_id in player_ids:
            source_file = SOURCE_DIR / f"player_{player_id}_daily_features.csv"
            dest_file = dest_dir / f"player_{player_id}_daily_features.csv"
            
            if not source_file.exists():
                print(f"  [MISSING] Player {player_id}: Source file not found")
                missing += 1
                continue
            
            if dest_file.exists():
                print(f"  [SKIP] Player {player_id}: Destination file already exists")
                continue
            
            try:
                shutil.copy2(source_file, dest_file)
                file_size = dest_file.stat().st_size / (1024 * 1024)  # MB
                print(f"  [COPIED] Player {player_id}: {file_size:.2f} MB")
                copied += 1
            except Exception as e:
                print(f"  [ERROR] Player {player_id}: Failed to copy - {e}")
                errors += 1
        
        print()
        print(f"Summary for {club_name}:")
        print(f"  Copied: {copied}")
        print(f"  Missing in source: {missing}")
        print(f"  Errors: {errors}")
        print()
        
        total_copied += copied
        total_missing += missing
        total_errors += errors
    
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Total copied: {total_copied}")
    print(f"Total missing in source: {total_missing}")
    print(f"Total errors: {total_errors}")
    print()
    
    if total_missing > 0:
        print(f"WARNING: {total_missing} player files are missing in source directory.")
        print("These will need to be generated from scratch using update_daily_features.py")
        print()
    
    return 0 if total_errors == 0 else 1

if __name__ == '__main__':
    exit(main())







