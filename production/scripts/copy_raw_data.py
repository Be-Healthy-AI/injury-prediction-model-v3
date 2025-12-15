#!/usr/bin/env python3
"""
Utility script to copy raw data exports to production structure.

Copies a date-stamped folder from source to production/raw_data/{country}/{date}/
and validates the file structure.
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

# Calculate paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent


def validate_raw_data_structure(data_dir: Path) -> bool:
    """Validate that raw data folder has expected structure."""
    required_files = [
        "players_profile.csv",
        "players_career.csv",
        "injuries_data.csv",
    ]
    required_dirs = [
        "match_data",
    ]
    
    missing_files = []
    for file in required_files:
        if not (data_dir / file).exists():
            missing_files.append(file)
    
    missing_dirs = []
    for dir_name in required_dirs:
        if not (data_dir / dir_name).is_dir():
            missing_dirs.append(dir_name)
    
    if missing_files or missing_dirs:
        print("❌ Validation failed:")
        if missing_files:
            print(f"   Missing files: {', '.join(missing_files)}")
        if missing_dirs:
            print(f"   Missing directories: {', '.join(missing_dirs)}")
        return False
    
    print("✅ File structure validated")
    return True


def separate_match_files_by_season(
    source_match_dir: Path, 
    dest_match_dir: Path, 
    previous_seasons_dir: Path, 
    current_season_key: int
) -> tuple[int, int]:
    """
    Separate match files: current season → dest_match_dir, old seasons → previous_seasons_dir.
    
    Returns:
        (current_season_count, old_season_count)
    """
    if not source_match_dir.exists():
        return 0, 0
    
    match_files = list(source_match_dir.glob("match_*.csv"))
    current_season_count = 0
    old_season_count = 0
    
    for match_file in match_files:
        # Extract season from filename: match_<player_id>_<season-1>_<season>.csv
        # e.g., match_12345_2024_2025.csv -> season key is 2024
        parts = match_file.stem.split('_')
        if len(parts) >= 3:
            try:
                season_start = int(parts[-2])  # e.g., 2024 from "2024_2025"
                season_key = season_start  # Season key is the starting year
                
                if season_key == current_season_key:
                    # Current season → date folder
                    dest_path = dest_match_dir / match_file.name
                    shutil.copy2(match_file, dest_path)
                    current_season_count += 1
                else:
                    # Old season → previous_seasons folder
                    dest_path = previous_seasons_dir / match_file.name
                    # Only copy if doesn't exist (preserve existing old season data)
                    if not dest_path.exists():
                        shutil.copy2(match_file, dest_path)
                        old_season_count += 1
                    else:
                        print(f"   ⏭️  Skipping {match_file.name} (already exists in previous_seasons)")
            except ValueError:
                print(f"   ⚠️  Could not parse season from {match_file.name}, copying to current season folder")
                dest_path = dest_match_dir / match_file.name
                shutil.copy2(match_file, dest_path)
                current_season_count += 1
        else:
            # If we can't parse the filename, copy to current season folder
            print(f"   ⚠️  Could not parse filename {match_file.name}, copying to current season folder")
            dest_path = dest_match_dir / match_file.name
            shutil.copy2(match_file, dest_path)
            current_season_count += 1
    
    return current_season_count, old_season_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Source directory containing raw data files'
    )
    parser.add_argument(
        '--country',
        type=str,
        required=True,
        help='Country name (e.g., "england")'
    )
    parser.add_argument(
        '--date',
        type=str,
        default=None,
        help='Date folder name (YYYYMMDD format). If not provided, uses folder name from source.'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate file structure after copying'
    )
    args = parser.parse_args()
    
    source_dir = Path(args.source)
    if not source_dir.exists():
        print(f"❌ Source directory not found: {source_dir}")
        return 1
    
    # Determine date folder name
    if args.date:
        date_folder = args.date
    else:
        # Use source folder name if it's a date folder, otherwise use today
        if source_dir.name.isdigit() and len(source_dir.name) == 8:
            date_folder = source_dir.name
        else:
            date_folder = datetime.today().strftime("%Y%m%d")
    
    # Determine current season
    now = datetime.now()
    if now.month >= 7:
        current_season_key = now.year
    else:
        current_season_key = now.year - 1
    
    # Setup destination
    dest_dir = PRODUCTION_ROOT / "raw_data" / args.country.lower().replace(" ", "_") / date_folder
    previous_seasons_dir = PRODUCTION_ROOT / "raw_data" / args.country.lower().replace(" ", "_") / "previous_seasons"
    previous_seasons_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("COPY RAW DATA TO PRODUCTION")
    print("=" * 70)
    print(f"📂 Source: {source_dir}")
    print(f"📂 Destination: {dest_dir}")
    print(f"📅 Date folder: {date_folder}")
    print()
    
    # Check if destination already exists
    if dest_dir.exists():
        print(f"⚠️  Destination already exists: {dest_dir}")
        response = input("   Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("   Cancelled")
            return 0
        shutil.rmtree(dest_dir)
    
    # Copy files
    print("📋 Copying files...")
    try:
        # Copy main CSV files
        for file_name in ["players_profile.csv", "players_career.csv", "injuries_data.csv"]:
            source_file = source_dir / file_name
            if source_file.exists():
                shutil.copy2(source_file, dest_dir / file_name)
                print(f"   ✅ {file_name}")
            else:
                print(f"   ⚠️  {file_name} not found in source")
        
        # Create match_data directory
        dest_match_dir = dest_dir / "match_data"
        dest_match_dir.mkdir(parents=True, exist_ok=True)
        
        # Separate match files by season
        source_match_dir = source_dir / "match_data"
        if source_match_dir.exists():
            print(f"   📂 Separating match files (current season: {current_season_key})...")
            current_count, old_count = separate_match_files_by_season(
                source_match_dir, dest_match_dir, previous_seasons_dir, current_season_key
            )
            print(f"   ✅ Current season files: {current_count}")
            print(f"   ✅ Previous season files: {old_count} (copied to previous_seasons/)")
        else:
            print(f"   ⚠️  No match_data folder in source")
        
        print(f"✅ Copied to: {dest_dir}")
    except Exception as e:
        print(f"❌ Error copying files: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Validate structure
    if args.validate:
        print("\n🔍 Validating file structure...")
        if not validate_raw_data_structure(dest_dir):
            return 1
    
    print("\n✅ Raw data copied successfully!")
    return 0


if __name__ == "__main__":
    exit(main())

