#!/usr/bin/env python3
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
from pathlib import Path

print("=" * 80)
print("COMPARING TIMELINE FILES - DATE RANGES")
print("=" * 80)

files_to_check = [
    'timelines_35day_enhanced_natural_v4_muscular_train.csv',
    'timelines_35day_enhanced_balanced_v4_train.csv'
]

results = {}

for file_path in files_to_check:
    if not Path(file_path).exists():
        print(f"\n⚠️  File not found: {file_path}")
        continue
    
    print(f"\n📂 Analyzing: {file_path}")
    
    try:
        # Read in chunks to get accurate min/max dates
        print("   Reading file in chunks to find date range...")
        chunk_size = 100000
        min_dates = []
        max_dates = []
        total_rows = 0
        year_counts = {}
        positives = 0
        negatives = 0
        
        for chunk in pd.read_csv(file_path, encoding='utf-8-sig', chunksize=chunk_size, low_memory=False):
            total_rows += len(chunk)
            
            if 'reference_date' in chunk.columns:
                chunk['reference_date'] = pd.to_datetime(chunk['reference_date'], errors='coerce')
                valid_dates = chunk['reference_date'].dropna()
                
                if len(valid_dates) > 0:
                    min_dates.append(valid_dates.min())
                    max_dates.append(valid_dates.max())
                    
                    # Count by year
                    chunk['year'] = chunk['reference_date'].dt.year
                    for year, count in chunk['year'].value_counts().items():
                        year_counts[year] = year_counts.get(year, 0) + count
            
            if 'target' in chunk.columns:
                positives += chunk['target'].sum()
                negatives += len(chunk) - chunk['target'].sum()
            
            if total_rows % 500000 == 0:
                print(f"   Processed {total_rows:,} rows...")
        
        if min_dates and max_dates:
            overall_min = min(min_dates)
            overall_max = max(max_dates)
            
            results[file_path] = {
                'total_rows': total_rows,
                'min_date': overall_min,
                'max_date': overall_max,
                'year_counts': year_counts,
                'positives': positives,
                'negatives': negatives
            }
            
            print(f"\n   ✅ Results:")
            print(f"      Total rows: {total_rows:,}")
            if positives > 0 or negatives > 0:
                print(f"      Positives: {positives:,} ({positives/total_rows:.2%})")
                print(f"      Negatives: {negatives:,} ({negatives/total_rows:.2%})")
            print(f"      Date range: {overall_min.date()} to {overall_max.date()}")
            print(f"      Years: {sorted(year_counts.keys())}")
            print(f"\n      Row counts by year:")
            for year in sorted(year_counts.keys()):
                print(f"         {year}: {year_counts[year]:,} rows")
        else:
            print(f"   ⚠️  No valid dates found")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()

# Comparison
if len(results) == 2:
    print("\n" + "=" * 80)
    print("📊 COMPARISON")
    print("=" * 80)
    
    file1 = list(results.keys())[0]
    file2 = list(results.keys())[1]
    
    r1 = results[file1]
    r2 = results[file2]
    
    print(f"\n   Row counts:")
    print(f"      {Path(file1).name}: {r1['total_rows']:,} rows")
    print(f"      {Path(file2).name}: {r2['total_rows']:,} rows")
    print(f"      Difference: {abs(r1['total_rows'] - r2['total_rows']):,} rows")
    
    print(f"\n   Date ranges:")
    print(f"      {Path(file1).name}: {r1['min_date'].date()} to {r1['max_date'].date()}")
    print(f"      {Path(file2).name}: {r2['min_date'].date()} to {r2['max_date'].date()}")
    
    print(f"\n   Year coverage:")
    years1 = set(r1['year_counts'].keys())
    years2 = set(r2['year_counts'].keys())
    print(f"      {Path(file1).name}: {sorted(years1)}")
    print(f"      {Path(file2).name}: {sorted(years2)}")
    
    if years1 != years2:
        only_in_1 = years1 - years2
        only_in_2 = years2 - years1
        if only_in_1:
            print(f"      Years only in {Path(file1).name}: {sorted(only_in_1)}")
        if only_in_2:
            print(f"      Years only in {Path(file2).name}: {sorted(only_in_2)}")
    else:
        print(f"      Both files have the same year coverage")

print("\n" + "=" * 80)
print("✅ Comparison complete!")
print("=" * 80)
