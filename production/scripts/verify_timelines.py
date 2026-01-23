#!/usr/bin/env python3
"""Quick verification script for timelines file."""

import pandas as pd
from pathlib import Path

PRODUCTION_ROOT = Path(__file__).resolve().parent.parent
TIMELINES_FILE = PRODUCTION_ROOT / "deployments" / "England" / "Chelsea FC" / "timelines" / "timelines_35day_season_2025_2026_v4_muscular.csv"

df = pd.read_csv(TIMELINES_FILE, low_memory=False)
df['reference_date'] = pd.to_datetime(df['reference_date'], errors='coerce')

print(f"File: {TIMELINES_FILE.name}")
print(f"Total timelines: {len(df):,}")
date_range = df['reference_date'].dropna()
if len(date_range) > 0:
    print(f"Date range: {date_range.min().date()} to {date_range.max().date()}")
else:
    print("Date range: No valid dates found")
print(f"Timelines on 2025-12-05: {len(df[df['reference_date'] == '2025-12-05'])}")
print(f"Timelines on 2025-12-06: {len(df[df['reference_date'] == '2025-12-06'])}")
print(f"Timelines on 2025-12-22: {len(df[df['reference_date'] == '2025-12-22'])}")

# Check date distribution around the cutoff
print(f"\nDate distribution around cutoff:")
for date_str in ['2025-12-04', '2025-12-05', '2025-12-06', '2025-12-07']:
    count = len(df[df['reference_date'] == date_str])
    print(f"  {date_str}: {count} timelines")

