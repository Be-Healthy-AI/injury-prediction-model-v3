#!/usr/bin/env python3
"""
Aggregate all raw data files needed for V4 daily features generation.
Copies files from various sources into V4/data/raw_data/ directory.
"""

import sys
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import os
import shutil
from pathlib import Path

# V4 root (go up 3 levels: daily_features -> code -> lgbm_muscular_v4)
V4_ROOT = Path(__file__).resolve().parent.parent.parent
V4_RAW_DATA = V4_ROOT / "data" / "raw_data"

# Project root (go up 5 levels: daily_features -> code -> lgbm_muscular_v4 -> models_production -> root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

# Source paths
NEW_TEAMS_DATA = PROJECT_ROOT / "production" / "raw_data" / "england" / "20260106" / "teams_data.csv"
NEW_DATA_DIR = PROJECT_ROOT / "production" / "raw_data" / "england" / "20260106"

# V1 source (fallback if new data not available)
V1_DATA_DIR = PROJECT_ROOT / "data_exports" / "transfermarkt" / "england" / "20251205"

def aggregate_raw_data():
    """Copy all required raw data files to V4/data/raw_data/"""
    
    print("=" * 80)
    print("AGGREGATING RAW DATA FOR V4")
    print("=" * 80)
    
    # Create output directory
    V4_RAW_DATA.mkdir(parents=True, exist_ok=True)
    (V4_RAW_DATA / "match_data").mkdir(parents=True, exist_ok=True)
    
    # Files to copy
    # NOTE: Use V1 source (1509 players) as primary source for player data
    # The new data directory only has 549 players, which is insufficient
    files_to_copy = [
        ('players_profile.csv', 'players_profile.csv'),
        ('players_career.csv', 'players_career.csv'),
        ('injuries_data.csv', 'injuries_data.csv'),
    ]
    
    # Copy individual files (prioritize V1 source for full dataset, fallback to new data)
    for source_name, target_name in files_to_copy:
        primary_path = V1_DATA_DIR / source_name
        fallback_path = NEW_DATA_DIR / source_name
        
        if primary_path.exists():
            shutil.copy2(primary_path, V4_RAW_DATA / target_name)
            print(f"✅ Copied {source_name} from V1 source (full dataset)")
        elif fallback_path.exists():
            shutil.copy2(fallback_path, V4_RAW_DATA / target_name)
            print(f"⚠️  Copied {source_name} from new data (fallback - limited dataset)")
        else:
            print(f"❌ {source_name} not found in either location!")
    
    # Copy teams_data.csv (MUST use new reviewed version)
    if NEW_TEAMS_DATA.exists():
        shutil.copy2(NEW_TEAMS_DATA, V4_RAW_DATA / "teams_data.csv")
        print(f"✅ Copied teams_data.csv from new reviewed data")
    else:
        print(f"❌ teams_data.csv not found at {NEW_TEAMS_DATA}")
        print("   This file is REQUIRED - please ensure it exists!")
        return False
    
    # Copy match_data directory
    new_match_dir = NEW_DATA_DIR / "match_data"
    v1_match_dir = V1_DATA_DIR / "match_data"
    
    match_files_copied = 0
    if new_match_dir.exists():
        for match_file in new_match_dir.glob("match_*.csv"):
            shutil.copy2(match_file, V4_RAW_DATA / "match_data" / match_file.name)
            match_files_copied += 1
        print(f"✅ Copied {match_files_copied} match files from new data")
    
    # Also copy from V1 if needed (for historical matches)
    if v1_match_dir.exists():
        v1_files_copied = 0
        for match_file in v1_match_dir.glob("match_*.csv"):
            target_file = V4_RAW_DATA / "match_data" / match_file.name
            if not target_file.exists():  # Don't overwrite if already copied
                shutil.copy2(match_file, target_file)
                v1_files_copied += 1
        if v1_files_copied > 0:
            print(f"✅ Copied {v1_files_copied} additional match files from V1 data")
        print(f"📊 Total match files: {match_files_copied + v1_files_copied}")
    
    print("\n" + "=" * 80)
    print("RAW DATA AGGREGATION COMPLETE")
    print("=" * 80)
    print(f"Data location: {V4_RAW_DATA}")
    print(f"Files ready for daily features generation")
    return True

if __name__ == "__main__":
    success = aggregate_raw_data()
    exit(0 if success else 1)
