#!/usr/bin/env python3
"""
Verify that all daily features files have data until 2025-12-05.
"""

import pandas as pd
from pathlib import Path
import json

PRODUCTION_ROOT = Path(__file__).resolve().parent.parent
DEPLOYMENTS_DIR = PRODUCTION_ROOT / "deployments" / "England"
TARGET_DATE = "2025-12-05"

def main():
    print("=" * 80)
    print("VERIFYING DAILY FEATURES FILES - DATA UNTIL 2025-12-05")
    print("=" * 80)
    print()
    
    clubs_with_issues = []
    clubs_ok = []
    
    for club_dir in sorted(DEPLOYMENTS_DIR.iterdir()):
        if not club_dir.is_dir():
            continue
        
        config_path = club_dir / "config.json"
        if not config_path.exists():
            continue
        
        club_name = club_dir.name
        daily_features_dir = club_dir / "daily_features"
        
        if not daily_features_dir.exists():
            print(f"[MISSING] {club_name}: daily_features directory not found")
            clubs_with_issues.append(club_name)
            continue
        
        # Get player IDs from config
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        player_ids = config.get('player_ids', [])
        
        # Check a sample of files
        files = list(daily_features_dir.glob("player_*_daily_features.csv"))
        
        if len(files) != len(player_ids):
            print(f"[MISMATCH] {club_name}: {len(files)} files found, {len(player_ids)} players in config")
            clubs_with_issues.append(club_name)
            continue
        
        # Check latest date in a few sample files
        sample_files = files[:min(3, len(files))]
        all_have_target_date = True
        latest_dates = []
        
        for file_path in sample_files:
            try:
                df = pd.read_csv(file_path, parse_dates=['date'], low_memory=False)
                if df.empty:
                    all_have_target_date = False
                    break
                max_date = df['date'].max()
                latest_dates.append(str(max_date.date()))
                has_target = (df['date'] == pd.to_datetime(TARGET_DATE)).any()
                if not has_target:
                    all_have_target_date = False
            except Exception as e:
                print(f"[ERROR] {club_name}: Error reading {file_path.name}: {e}")
                all_have_target_date = False
                break
        
        if all_have_target_date:
            print(f"[OK] {club_name}: {len(files)} files, latest date: {max(latest_dates)}")
            clubs_ok.append(club_name)
        else:
            print(f"[ISSUE] {club_name}: Missing data for {TARGET_DATE} or earlier")
            clubs_with_issues.append(club_name)
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Clubs OK: {len(clubs_ok)}/{len(clubs_ok) + len(clubs_with_issues)}")
    print(f"Clubs with issues: {len(clubs_with_issues)}/{len(clubs_ok) + len(clubs_with_issues)}")
    
    if clubs_with_issues:
        print(f"\nClubs with issues: {', '.join(clubs_with_issues)}")
    else:
        print("\n[SUCCESS] All clubs have daily features files with data until 2025-12-05!")

if __name__ == '__main__':
    main()







