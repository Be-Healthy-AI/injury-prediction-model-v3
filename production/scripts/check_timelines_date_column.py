#!/usr/bin/env python3
"""Check for date column issues in timelines file."""

import pandas as pd
from pathlib import Path

PRODUCTION_ROOT = Path(__file__).resolve().parent.parent
TIMELINES_FILE = PRODUCTION_ROOT / "deployments" / "England" / "Chelsea FC" / "timelines" / "timelines_35day_season_2025_2026_v4_muscular.csv"

df = pd.read_csv(TIMELINES_FILE, low_memory=False, nrows=100)

# Find all date-related columns
date_cols = [c for c in df.columns if 'date' in c.lower()]

print("=" * 80)
print("CHECKING DATE COLUMNS IN TIMELINES")
print("=" * 80)
print(f"\nDate-related columns found: {date_cols}")
print()

if date_cols:
    for col in date_cols:
        print(f"Column: {col}")
        print(f"  Sample values (first 5):")
        for i in range(min(5, len(df))):
            val = df[col].iloc[i]
            print(f"    Row {i}: {val} (type: {type(val).__name__})")
        
        # Check for NaT/NaN
        nat_count = df[col].isna().sum()
        print(f"  NaT/NaN count: {nat_count} out of {len(df)} rows")
        print()
else:
    print("No date-related columns found!")

# Check full file for NaT
print("Checking full file for NaT values in date columns...")
df_full = pd.read_csv(TIMELINES_FILE, low_memory=False)
if date_cols:
    for col in date_cols:
        nat_count = df_full[col].isna().sum()
        print(f"  {col}: {nat_count} NaT/NaN values out of {len(df_full)} total rows")








