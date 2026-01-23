#!/usr/bin/env python3
"""Check reference_date format consistency in timelines file."""

import pandas as pd
from pathlib import Path

PRODUCTION_ROOT = Path(__file__).resolve().parent.parent
TIMELINES_FILE = PRODUCTION_ROOT / "deployments" / "England" / "Chelsea FC" / "timelines" / "timelines_35day_season_2025_2026_v4_muscular.csv"

df = pd.read_csv(TIMELINES_FILE, low_memory=False)

print("=" * 80)
print("CHECKING REFERENCE_DATE FORMAT CONSISTENCY")
print("=" * 80)
print(f"\nTotal rows: {len(df):,}")
print()

# Check format of reference_date
print("Sample reference_date values:")
for i in range(min(10, len(df))):
    val = df['reference_date'].iloc[i]
    print(f"  Row {i}: '{val}' (type: {type(val).__name__}, length: {len(str(val))})")

print()
print("Checking for datetime format (contains time):")
datetime_format_count = df['reference_date'].astype(str).str.contains(' ').sum()
date_only_count = len(df) - datetime_format_count
print(f"  Date-only format (YYYY-MM-DD): {date_only_count:,}")
print(f"  Datetime format (YYYY-MM-DD HH:MM:SS): {datetime_format_count:,}")

print()
print("Checking for NaT/NaN:")
nat_count = df['reference_date'].isna().sum()
print(f"  NaT/NaN count: {nat_count:,}")

print()
print("Parsing as datetime to check validity:")
df['reference_date_parsed'] = pd.to_datetime(df['reference_date'], errors='coerce')
invalid_count = df['reference_date_parsed'].isna().sum()
print(f"  Invalid dates: {invalid_count:,}")
if invalid_count > 0:
    print("  Sample invalid values:")
    invalid_samples = df[df['reference_date_parsed'].isna()]['reference_date'].head(5)
    for val in invalid_samples:
        print(f"    '{val}'")








