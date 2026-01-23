#!/usr/bin/env python3
"""
Deduplicate injuries in raw data file by removing duplicate entries.
Duplicates are identified by: player_id, fromDate, injury_type, untilDate
"""
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
from pathlib import Path
import argparse

# Calculate paths
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent

def deduplicate_injuries_file(injuries_file: Path):
    """Deduplicate injuries in a CSV file."""
    if not injuries_file.exists():
        print(f"[ERROR] File not found: {injuries_file}")
        return False
    
    try:
        # Load injuries file
        df = pd.read_csv(injuries_file, sep=';', low_memory=False)
        print(f"Loaded {len(df)} injuries from {injuries_file.name}")
        
        # Identify duplicates based on key fields
        dedup_cols = ['player_id', 'fromDate', 'injury_type', 'untilDate']
        
        # Check which columns exist
        missing_cols = [col for col in dedup_cols if col not in df.columns]
        if missing_cols:
            print(f"[ERROR] Missing required columns: {missing_cols}")
            return False
        
        before_count = len(df)
        
        # Remove duplicates, keeping first occurrence
        df_dedup = df.drop_duplicates(subset=dedup_cols, keep='first')
        after_count = len(df_dedup)
        
        duplicates_removed = before_count - after_count
        
        if duplicates_removed > 0:
            # Save deduplicated file
            df_dedup.to_csv(injuries_file, index=False, encoding='utf-8-sig', sep=';')
            print(f"[OK] Removed {duplicates_removed} duplicate injuries (kept {after_count} unique injuries)")
            
            # Show some examples of duplicates
            duplicates = df[df.duplicated(subset=dedup_cols, keep=False)].sort_values(dedup_cols)
            if not duplicates.empty:
                print(f"\nExamples of duplicates found:")
                for player_id in duplicates['player_id'].unique()[:5]:
                    player_dups = duplicates[duplicates['player_id'] == player_id]
                    print(f"  Player {player_id}: {len(player_dups)} duplicate entries")
                    for idx, row in player_dups.head(2).iterrows():
                        print(f"    - {row['injury_type']} on {row['fromDate']}")
        else:
            print(f"[OK] No duplicates found ({before_count} injuries)")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to deduplicate: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Deduplicate injuries in raw data file")
    parser.add_argument("--file", type=str, help="Path to injuries_data.csv file")
    parser.add_argument("--country", type=str, default="england", help="Country name")
    parser.add_argument("--date", type=str, help="Date folder (YYYYMMDD), e.g., 20260121")
    
    args = parser.parse_args()
    
    if args.file:
        injuries_file = Path(args.file)
    elif args.date:
        injuries_file = PRODUCTION_ROOT / "raw_data" / args.country / args.date / "injuries_data.csv"
    else:
        print("[ERROR] Must provide either --file or --date")
        return 1
    
    if deduplicate_injuries_file(injuries_file):
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())
