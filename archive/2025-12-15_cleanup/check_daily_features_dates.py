#!/usr/bin/env python3
import pandas as pd
import os
from pathlib import Path
from collections import Counter

daily_features_dir = Path('daily_features_output')
files = list(daily_features_dir.glob('player_*_daily_features.csv'))

print(f"Found {len(files)} daily feature files")
print("Checking date ranges across all files...\n")

min_dates = []
max_dates = []
all_years = set()
files_with_old_data = []

for i, file_path in enumerate(files):
    try:
        df = pd.read_csv(file_path)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            valid_dates = df['date'].dropna()
            if len(valid_dates) > 0:
                file_min = valid_dates.min()
                file_max = valid_dates.max()
                min_dates.append(file_min)
                max_dates.append(file_max)
                years = set(valid_dates.dt.year.unique())
                all_years.update(years)
                
                # Check if file has data before 2021
                if file_min < pd.Timestamp('2021-07-01'):
                    files_with_old_data.append((file_path.name, file_min, file_max))
                    
        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{len(files)} files...")
    except Exception as e:
        pass

if min_dates and max_dates:
    overall_min = min(min_dates)
    overall_max = max(max_dates)
    print(f"\n{'='*60}")
    print(f"Overall date range across ALL daily feature files:")
    print(f"  Minimum: {overall_min}")
    print(f"  Maximum: {overall_max}")
    print(f"  Years available: {sorted(all_years)}")
    
    print(f"\n{'='*60}")
    print(f"Files with data before 2021-07-01: {len(files_with_old_data)}")
    if files_with_old_data:
        print(f"\nSample files with old data (first 10):")
        for name, fmin, fmax in files_with_old_data[:10]:
            print(f"  {name}: {fmin.date()} to {fmax.date()}")
        
        # Count by earliest year
        earliest_years = Counter([fmin.year for _, fmin, _ in files_with_old_data])
        print(f"\nFiles by earliest year:")
        for year in sorted(earliest_years.keys()):
            print(f"  {year}: {earliest_years[year]} files")
