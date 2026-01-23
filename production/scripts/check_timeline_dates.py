#!/usr/bin/env python3
"""
Check the dates in a timeline file.
"""

import pandas as pd
from pathlib import Path
import sys

if len(sys.argv) < 2:
    print("Usage: python check_timeline_dates.py <club_name>")
    sys.exit(1)

club = sys.argv[1]
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent

timelines_file = PRODUCTION_ROOT / "deployments" / "England" / club / "timelines" / "timelines_35day_season_2025_2026_v4_muscular.csv"

if not timelines_file.exists():
    print(f"Timelines file not found for {club}")
    sys.exit(1)

df = pd.read_csv(timelines_file, usecols=['reference_date'], low_memory=False)
df['reference_date'] = pd.to_datetime(df['reference_date'], errors='coerce', format='mixed')

print(f"\n{club} timelines:")
print(f"Total rows: {len(df)}")
print(f"Date range: {df['reference_date'].min().date()} to {df['reference_date'].max().date()}")

# Check dates around 2025-12-05
dec_05 = pd.Timestamp('2025-12-05')
dec_06 = pd.Timestamp('2025-12-06')
dec_26 = pd.Timestamp('2025-12-26')

has_dec_05 = (df['reference_date'] == dec_05).any()
has_dec_06 = (df['reference_date'] >= dec_06).any()
has_dec_26 = (df['reference_date'] == dec_26).any()

print(f"\nHas 2025-12-05: {has_dec_05}")
print(f"Has dates >= 2025-12-06: {has_dec_06}")
if has_dec_06:
    max_date = df[df['reference_date'] >= dec_06]['reference_date'].max()
    print(f"  Max date >= 2025-12-06: {max_date.date()}")
print(f"Has 2025-12-26: {has_dec_26}")

# Count rows by month
df['year_month'] = df['reference_date'].dt.to_period('M')
month_counts = df['year_month'].value_counts().sort_index()
print(f"\nRows by month (last 3 months):")
for month, count in month_counts.tail(3).items():
    print(f"  {month}: {count} rows")

