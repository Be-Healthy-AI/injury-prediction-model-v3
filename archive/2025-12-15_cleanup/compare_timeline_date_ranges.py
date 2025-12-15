#!/usr/bin/env python3
"""
Compare date ranges between natural and balanced timeline datasets.
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
from collections import Counter

# Files to compare
NATURAL_FILE = 'timelines_35day_enhanced_natural_v4_muscular_train.csv'
BALANCED_FILE = 'timelines_35day_enhanced_balanced_v4_train.csv'

def analyze_file(filepath, name):
    """Analyze date range and statistics for a timeline file."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {name}")
    print(f"File: {filepath}")
    print(f"{'='*80}")
    
    try:
        # Read file
        print(f"\n📂 Loading file...")
        df = pd.read_csv(filepath, encoding='utf-8-sig')
        print(f"   ✅ Loaded: {len(df):,} rows, {len(df.columns)} columns")
        
        # Check if reference_date column exists
        if 'reference_date' not in df.columns:
            print(f"\n   ⚠️  WARNING: 'reference_date' column not found!")
            print(f"   Available columns: {list(df.columns[:10])}...")
            return None
        
        # Convert reference_date to datetime
        print(f"\n📅 Analyzing date range...")
        df['reference_date'] = pd.to_datetime(df['reference_date'], errors='coerce')
        
        # Check for invalid dates
        invalid_dates = df['reference_date'].isna().sum()
        if invalid_dates > 0:
            print(f"   ⚠️  WARNING: {invalid_dates:,} rows have invalid dates ({invalid_dates/len(df):.2%})")
        
        # Get valid dates
        valid_dates = df['reference_date'].dropna()
        
        if len(valid_dates) == 0:
            print(f"   ⚠️  ERROR: No valid dates found!")
            return None
        
        # Date range statistics
        min_date = valid_dates.min()
        max_date = valid_dates.max()
        date_range_days = (max_date - min_date).days
        
        print(f"\n   Date Range:")
        print(f"      Minimum date: {min_date}")
        print(f"      Maximum date: {max_date}")
        print(f"      Range: {date_range_days:,} days ({date_range_days/365.25:.1f} years)")
        
        # Year distribution
        years = valid_dates.dt.year
        year_counts = Counter(years)
        print(f"\n   Year Distribution:")
        for year in sorted(year_counts.keys()):
            count = year_counts[year]
            pct = count / len(valid_dates) * 100
            print(f"      {year}: {count:,} samples ({pct:.1f}%)")
        
        # Target statistics (if available)
        if 'target' in df.columns:
            positives = df['target'].sum()
            negatives = len(df) - positives
            pos_pct = positives / len(df) * 100 if len(df) > 0 else 0
            print(f"\n   Target Statistics:")
            print(f"      Total samples: {len(df):,}")
            print(f"      Positives: {positives:,} ({pos_pct:.2f}%)")
            print(f"      Negatives: {negatives:,} ({100-pos_pct:.2f}%)")
        
        # Samples before 2017-07-01
        before_2017 = (valid_dates < pd.Timestamp('2017-07-01')).sum()
        after_2017 = (valid_dates >= pd.Timestamp('2017-07-01')).sum()
        print(f"\n   Samples relative to 2017-07-01 cutoff:")
        print(f"      Before 2017-07-01: {before_2017:,} ({before_2017/len(valid_dates)*100:.1f}%)")
        print(f"      >= 2017-07-01: {after_2017:,} ({after_2017/len(valid_dates)*100:.1f}%)")
        
        return {
            'min_date': min_date,
            'max_date': max_date,
            'total_rows': len(df),
            'valid_dates': len(valid_dates),
            'year_counts': year_counts,
            'before_2017': before_2017,
            'after_2017': after_2017
        }
        
    except FileNotFoundError:
        print(f"\n   ❌ ERROR: File not found: {filepath}")
        return None
    except Exception as e:
        print(f"\n   ❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("="*80)
    print("COMPARING TIMELINE DATASET DATE RANGES")
    print("="*80)
    
    # Analyze natural file
    natural_stats = analyze_file(NATURAL_FILE, "Natural Dataset (Muscular)")
    
    # Analyze balanced file
    balanced_stats = analyze_file(BALANCED_FILE, "Balanced Dataset")
    
    # Comparison
    if natural_stats and balanced_stats:
        print(f"\n{'='*80}")
        print("COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        print(f"\n   Date Range Comparison:")
        print(f"      Natural - Min: {natural_stats['min_date']}, Max: {natural_stats['max_date']}")
        print(f"      Balanced - Min: {balanced_stats['min_date']}, Max: {balanced_stats['max_date']}")
        
        date_range_diff = (natural_stats['max_date'] - natural_stats['min_date']).days - \
                         (balanced_stats['max_date'] - balanced_stats['min_date']).days
        print(f"      Range difference: {date_range_diff:,} days")
        
        print(f"\n   Row Count Comparison:")
        print(f"      Natural: {natural_stats['total_rows']:,} rows")
        print(f"      Balanced: {balanced_stats['total_rows']:,} rows")
        print(f"      Difference: {natural_stats['total_rows'] - balanced_stats['total_rows']:,} rows")
        
        print(f"\n   Samples before 2017-07-01:")
        print(f"      Natural: {natural_stats['before_2017']:,} ({natural_stats['before_2017']/natural_stats['valid_dates']*100:.1f}%)")
        print(f"      Balanced: {balanced_stats['before_2017']:,} ({balanced_stats['before_2017']/balanced_stats['valid_dates']*100:.1f}%)")
        
        # Year overlap
        natural_years = set(natural_stats['year_counts'].keys())
        balanced_years = set(balanced_stats['year_counts'].keys())
        common_years = natural_years & balanced_years
        only_natural = natural_years - balanced_years
        only_balanced = balanced_years - natural_years
        
        print(f"\n   Year Coverage:")
        print(f"      Natural years: {sorted(natural_years)}")
        print(f"      Balanced years: {sorted(balanced_years)}")
        print(f"      Common years: {sorted(common_years)}")
        if only_natural:
            print(f"      Only in Natural: {sorted(only_natural)}")
        if only_balanced:
            print(f"      Only in Balanced: {sorted(only_balanced)}")
    
    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()


