#!/usr/bin/env python3
"""
Truncate all timeline files to remove rows where reference_date > 2026-01-21
"""
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
from pathlib import Path

# Calculate paths
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent

# Get all clubs
country = "England"
deployments_dir = PRODUCTION_ROOT / "deployments" / country
max_date = pd.to_datetime("2026-01-21").normalize()

clubs = []
for item in deployments_dir.iterdir():
    if item.is_dir() and (item / "config.json").exists():
        clubs.append(item.name)

clubs = sorted(clubs)

print(f"Truncating timeline files for {len(clubs)} clubs...")
print(f"Max date: {max_date.date()}")

for club in clubs:
    timeline_file = deployments_dir / club / "timelines" / "timelines_35day_season_2025_2026_v4_muscular.csv"
    
    if not timeline_file.exists():
        print(f"[SKIP] {club} - Timeline file not found")
        continue
    
    try:
        # Load timeline file
        df = pd.read_csv(timeline_file)
        
        if 'reference_date' not in df.columns:
            print(f"[SKIP] {club} - No reference_date column")
            continue
        
        # Convert reference_date to datetime
        df['reference_date'] = pd.to_datetime(df['reference_date']).dt.normalize()
        
        before_count = len(df)
        # Filter to keep only rows where reference_date <= max_date
        df = df[df['reference_date'] <= max_date].copy()
        after_count = len(df)
        
        if before_count > after_count:
            # Save truncated file
            df.to_csv(timeline_file, index=False, encoding='utf-8-sig')
            print(f"[OK] {club} - Truncated: removed {before_count - after_count} rows (kept {after_count} rows)")
        else:
            print(f"[OK] {club} - No truncation needed ({before_count} rows)")
            
    except Exception as e:
        print(f"[ERROR] {club} - {str(e)}")

print("\nTruncation complete!")
