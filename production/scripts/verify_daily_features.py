#!/usr/bin/env python3
"""Quick verification script for daily features files."""

import pandas as pd
from pathlib import Path

PRODUCTION_ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = PRODUCTION_ROOT / "deployments" / "England" / "Chelsea FC" / "daily_features"

# Check a sample file
sample_file = TARGET_DIR / "player_258878_daily_features.csv"
df = pd.read_csv(sample_file, parse_dates=['date'])

print(f"File: {sample_file.name}")
print(f"Total rows: {len(df)}")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Rows on 2025-12-05: {len(df[df['date'] == '2025-12-05'])}")
print(f"Rows on 2025-12-06: {len(df[df['date'] == '2025-12-06'])}")
print(f"Rows on 2025-12-22: {len(df[df['date'] == '2025-12-22'])}")

# Check avg_injury_severity values
print(f"\nSample avg_injury_severity values:")
print(f"  2025-12-05: {df[df['date'] == '2025-12-05']['avg_injury_severity'].values[0] if len(df[df['date'] == '2025-12-05']) > 0 else 'N/A'}")
print(f"  2025-12-06: {df[df['date'] == '2025-12-06']['avg_injury_severity'].values[0] if len(df[df['date'] == '2025-12-06']) > 0 else 'N/A'}")
print(f"  2025-12-22: {df[df['date'] == '2025-12-22']['avg_injury_severity'].values[0] if len(df[df['date'] == '2025-12-22']) > 0 else 'N/A'}")








