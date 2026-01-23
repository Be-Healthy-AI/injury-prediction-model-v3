#!/usr/bin/env python3
"""
Update existing daily features files with temporal feature changes:
- Remove 'season_year' column (temporal leak)
- Add 'is_early_season' column (first 30 days of season)

This script modifies files in place, so it's much faster than regenerating from raw data.
"""

import sys
from pathlib import Path
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

SCRIPT_DIR = Path(__file__).resolve().parent
# Use same path calculation as create_daily_features_v4_enhanced.py
DAILY_FEATURES_DIR = SCRIPT_DIR.parent.parent / 'data' / 'daily_features'


def update_one_file(file_path: Path, verbose: bool = False) -> tuple:
    """
    Update a single daily features file:
    - Remove 'season_year' column if it exists
    - Add 'is_early_season' column based on 'days_into_season'
    
    Returns: (success: bool, message: str)
    """
    try:
        # Read the file
        df = pd.read_csv(file_path, encoding="utf-8-sig", low_memory=False)
        
        changes_made = []
        
        # Remove 'season_year' column if it exists
        if 'season_year' in df.columns:
            df = df.drop(columns=['season_year'])
            changes_made.append("removed 'season_year'")
        
        # Add 'is_early_season' if 'days_into_season' exists and 'is_early_season' doesn't
        if 'days_into_season' in df.columns and 'is_early_season' not in df.columns:
            df['is_early_season'] = (df['days_into_season'] <= 30).astype(int)
            changes_made.append("added 'is_early_season'")
        elif 'days_into_season' not in df.columns:
            if verbose:
                print(f"   ⚠️  {file_path.name}: 'days_into_season' not found, skipping 'is_early_season'")
        
        # Save back to the same file
        if changes_made:
            df.to_csv(file_path, index=False, encoding="utf-8-sig")
            return True, f"Updated: {', '.join(changes_made)}"
        else:
            return True, "No changes needed"
            
    except Exception as e:
        return False, f"Error: {str(e)}"


def main():
    """Main function to update all daily features files."""
    print("=" * 80)
    print("UPDATE DAILY FEATURES - TEMPORAL FEATURES")
    print("=" * 80)
    print(f"Directory: {DAILY_FEATURES_DIR}")
    print("Changes:")
    print("  - Remove 'season_year' column (temporal leak)")
    print("  - Add 'is_early_season' column (first 30 days of season)")
    print("=" * 80)
    
    if not DAILY_FEATURES_DIR.exists():
        print(f"ERROR: Directory not found: {DAILY_FEATURES_DIR}")
        return 1
    
    # Find all CSV files
    all_files = sorted(DAILY_FEATURES_DIR.glob("*.csv"))
    if not all_files:
        print(f"No CSV files found in {DAILY_FEATURES_DIR}")
        return 1
    
    print(f"\nFound {len(all_files)} daily features files to process.")
    print("Processing...\n")
    
    successful = 0
    failed = 0
    no_changes = 0
    
    for file_path in tqdm(all_files, desc="Updating files"):
        success, message = update_one_file(file_path, verbose=False)
        
        if success:
            if "No changes needed" in message:
                no_changes += 1
            else:
                successful += 1
        else:
            failed += 1
            print(f"[ERROR] {file_path.name}: {message}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"[OK] Successfully updated: {successful}")
    print(f"[SKIP] No changes needed: {no_changes}")
    print(f"[ERROR] Failed: {failed}")
    print(f"Total: {len(all_files)}")
    print("=" * 80)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
