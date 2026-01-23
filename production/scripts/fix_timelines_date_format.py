#!/usr/bin/env python3
"""
Fix reference_date format in timelines file - convert from datetime to date-only format.
"""

import pandas as pd
from pathlib import Path

# Paths
PRODUCTION_ROOT = Path(__file__).resolve().parent.parent
TIMELINES_FILE = PRODUCTION_ROOT / "deployments" / "England" / "Chelsea FC" / "timelines" / "timelines_35day_season_2025_2026_v4_muscular.csv"

def main():
    print("=" * 80)
    print("FIXING REFERENCE_DATE FORMAT IN TIMELINES")
    print("=" * 80)
    print()
    
    print(f"Loading timelines from: {TIMELINES_FILE}")
    df = pd.read_csv(TIMELINES_FILE, low_memory=False)
    print(f"Total timelines: {len(df):,}")
    print()
    
    # Check current format
    if 'reference_date' not in df.columns:
        print("ERROR: 'reference_date' column not found")
        return
    
    # Show sample of current format
    sample = df['reference_date'].head(5)
    print(f"Sample reference_date values (before fix):")
    for val in sample:
        print(f"  {val} (type: {type(val).__name__})")
    print()
    
    # Convert to datetime first (handles both date strings and datetime strings)
    print("Converting reference_date to date-only format...")
    df['reference_date'] = pd.to_datetime(df['reference_date'], errors='coerce')
    
    # Convert to date-only string format (YYYY-MM-DD)
    df['reference_date'] = df['reference_date'].dt.date.astype(str)
    
    # Show sample of new format
    sample = df['reference_date'].head(5)
    print(f"Sample reference_date values (after fix):")
    for val in sample:
        print(f"  {val} (type: {type(val).__name__})")
    print()
    
    # Check for any invalid dates
    invalid_count = df['reference_date'].isna().sum()
    if invalid_count > 0:
        print(f"WARNING: {invalid_count} rows have invalid dates")
    else:
        print("All dates are valid")
    print()
    
    # Save
    print(f"Saving fixed timelines to: {TIMELINES_FILE}")
    df.to_csv(TIMELINES_FILE, index=False)
    print(f"[OK] Saved {len(df):,} timelines with date-only format")
    print()
    
    # Verify date range
    date_range = pd.to_datetime(df['reference_date'], errors='coerce').dropna()
    if len(date_range) > 0:
        print(f"Date range: {date_range.min().date()} to {date_range.max().date()}")
    
    print()
    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()








