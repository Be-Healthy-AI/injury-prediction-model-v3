#!/usr/bin/env python3
"""
Add injury_class, body_part, and severity columns to an existing injuries_data.csv file.

This script reads an existing injuries CSV file, adds the three new columns based on
the injury mappings and severity calculation, and writes it back.

Usage:
    python production/scripts/add_injury_columns.py --file production/raw_data/england/20251205/injuries_data.csv
    python production/scripts/add_injury_columns.py --country england --date 20251205
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Calculate paths
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent
ROOT_DIR = PRODUCTION_ROOT.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.data_collection.transformers import calculate_severity_from_days


def add_injury_columns(injuries_file: Path, injury_mappings: dict) -> None:
    """
    Add injury_class, body_part, and severity columns to an injuries CSV file.
    
    Args:
        injuries_file: Path to the injuries_data.csv file
        injury_mappings: Dictionary mapping injury_type -> {injury_class, body_part}
    """
    print(f"Reading injuries file: {injuries_file}")
    df = pd.read_csv(injuries_file, sep=';', encoding='utf-8-sig')
    
    print(f"  Total injuries: {len(df)}")
    
    # Check if columns already exist
    if 'injury_class' in df.columns and 'body_part' in df.columns and 'severity' in df.columns:
        print("  [INFO] Columns already exist. Checking if they need updating...")
        # Check if any values are missing
        missing_class = df['injury_class'].isna().sum()
        missing_body_part = df['body_part'].isna().sum()
        missing_severity = df['severity'].isna().sum()
        
        if missing_class == 0 and missing_body_part == 0 and missing_severity == 0:
            print("  [OK] All columns are already populated. No update needed.")
            return
        else:
            print(f"  Found missing values: injury_class={missing_class}, body_part={missing_body_part}, severity={missing_severity}")
            print("  Updating missing values...")
    
    # Add/update injury_class
    if 'injury_class' not in df.columns:
        df['injury_class'] = None
    df['injury_class'] = df.apply(
        lambda row: injury_mappings.get(str(row['injury_type']), {}).get('injury_class', 'unknown')
        if pd.notna(row.get('injury_type')) else 'unknown',
        axis=1
    )
    
    # Add/update body_part
    if 'body_part' not in df.columns:
        df['body_part'] = None
    df['body_part'] = df.apply(
        lambda row: injury_mappings.get(str(row['injury_type']), {}).get('body_part', '')
        if pd.notna(row.get('injury_type')) else '',
        axis=1
    )
    
    # Add/update severity
    if 'severity' not in df.columns:
        df['severity'] = None
    df['severity'] = df['days'].apply(calculate_severity_from_days)
    
    # Save back to file
    df.to_csv(injuries_file, index=False, encoding='utf-8-sig', sep=';')
    
    print(f"  [OK] Updated {len(df)} injuries with new columns")
    print(f"  [OK] Saved to: {injuries_file}")
    
    # Show summary
    print("\n  Summary:")
    print(f"    Injury classes: {df['injury_class'].value_counts().to_dict()}")
    body_part_counts = df[df['body_part'] != '']['body_part'].value_counts()
    if len(body_part_counts) > 0:
        print(f"    Body parts: {body_part_counts.to_dict()}")
    print(f"    Severity levels: {df['severity'].value_counts().to_dict()}")


def main():
    parser = argparse.ArgumentParser(
        description="Add injury_class, body_part, and severity columns to an existing injuries_data.csv file"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to the injuries_data.csv file to update"
    )
    parser.add_argument(
        "--country",
        type=str,
        help="Country name (e.g., 'england'). Used with --date to construct file path"
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Date folder (e.g., '20251205'). Used with --country to construct file path"
    )
    
    args = parser.parse_args()
    
    # Determine the injuries file path
    if args.file:
        injuries_file = Path(args.file)
    elif args.country and args.date:
        injuries_file = PRODUCTION_ROOT / "raw_data" / args.country.lower() / args.date / "injuries_data.csv"
    else:
        print("[ERROR] Either --file or both --country and --date must be provided")
        return 1
    
    if not injuries_file.exists():
        print(f"[ERROR] Injuries file not found: {injuries_file}")
        return 1
    
    # Load injury mappings
    mapping_file = PRODUCTION_ROOT / "config" / "injury_mappings.json"
    if not mapping_file.exists():
        print(f"[ERROR] Injury mappings file not found: {mapping_file}")
        print(f"  Please run: python production/scripts/generate_injury_mappings.py")
        return 1
    
    with open(mapping_file, 'r', encoding='utf-8') as f:
        injury_mappings = json.load(f)
    
    print(f"Loaded {len(injury_mappings)} injury type mappings")
    print()
    
    # Add columns
    add_injury_columns(injuries_file, injury_mappings)
    
    return 0


if __name__ == "__main__":
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    
    sys.exit(main())

