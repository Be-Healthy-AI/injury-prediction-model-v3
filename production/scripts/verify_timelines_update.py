#!/usr/bin/env python3
"""
Verify that timelines were updated correctly for all clubs.
"""

import pandas as pd
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent

# Get all club folders
deployments_dir = PRODUCTION_ROOT / "deployments" / "England"
clubs = [d.name for d in deployments_dir.iterdir() if d.is_dir()]

print("Verifying timelines update for all clubs...")
print("=" * 80)

results = []
for club in sorted(clubs):
    timelines_file = deployments_dir / club / "timelines" / "timelines_35day_season_2025_2026_v4_muscular.csv"
    
    if not timelines_file.exists():
        results.append((club, "No timelines file", 0, None))
        continue
    
    try:
        df = pd.read_csv(timelines_file, usecols=['reference_date'], low_memory=False)
        if len(df) == 0:
            results.append((club, "Empty file", 0, None))
            continue
        
        df['reference_date'] = pd.to_datetime(df['reference_date'], errors='coerce')
        df = df.dropna(subset=['reference_date'])
        
        if len(df) == 0:
            results.append((club, "No valid dates", 0, None))
            continue
        
        max_date = df['reference_date'].max()
        min_date = df['reference_date'].min()
        results.append((club, "OK", len(df), max_date.date()))
    except Exception as e:
        results.append((club, f"Error: {str(e)}", 0, None))

# Print results
print(f"{'Club':<30} {'Status':<20} {'Rows':<10} {'Latest Date':<15}")
print("-" * 80)
for club, status, rows, max_date in results:
    date_str = str(max_date) if max_date else "N/A"
    print(f"{club:<30} {status:<20} {rows:<10} {date_str:<15}")

# Summary
total_clubs = len(results)
clubs_with_data = sum(1 for _, status, rows, _ in results if status == "OK" and rows > 0)
clubs_without_data = total_clubs - clubs_with_data

print("\n" + "=" * 80)
print(f"Summary: {clubs_with_data}/{total_clubs} clubs have timelines data")
if clubs_without_data > 0:
    print(f"Clubs without data: {clubs_without_data}")







