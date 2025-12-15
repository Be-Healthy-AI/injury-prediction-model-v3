#!/usr/bin/env python3
"""
Filter natural datasets to exclude samples before 2017-07-01 (season 2017/18 onwards).
Creates new datasets with '_post2017' suffix to keep original datasets intact.
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
from datetime import datetime

# Configuration
MIN_DATE = pd.Timestamp('2017-07-01')  # Exclude everything before this date
INPUT_TRAIN_FILE = 'timelines_35day_enhanced_natural_v4_muscular_train.csv'
INPUT_VAL_FILE = 'timelines_35day_enhanced_natural_v4_muscular_val.csv'
OUTPUT_TRAIN_FILE = 'timelines_35day_enhanced_natural_post2017_v4_muscular_train.csv'
OUTPUT_VAL_FILE = 'timelines_35day_enhanced_natural_post2017_v4_muscular_val.csv'

def diagnose_dataset(input_file, dataset_name):
    """Diagnose dataset to show date ranges and what would be filtered"""
    print(f"\n📂 Diagnosing {dataset_name}...")
    print(f"   Loading: {input_file}")
    
    df = pd.read_csv(input_file, encoding='utf-8-sig')
    original_count = len(df)
    original_positives = df['target'].sum()
    original_negatives = original_count - original_positives
    
    print(f"   Original: {original_count:,} samples ({original_positives:,} positives, {original_negatives:,} negatives)")
    
    # Convert reference_date to datetime
    print(f"   Converting reference_date to datetime...")
    df['reference_date'] = pd.to_datetime(df['reference_date'], errors='coerce')
    
    # Check for any parsing errors
    invalid_dates = df['reference_date'].isna().sum()
    if invalid_dates > 0:
        print(f"   ⚠️  WARNING: {invalid_dates:,} samples have invalid dates!")
    
    # Show date range statistics
    min_date = df['reference_date'].min()
    max_date = df['reference_date'].max()
    print(f"\n   Date range in original dataset:")
    print(f"      Minimum date: {min_date}")
    print(f"      Maximum date: {max_date}")
    
    # Count samples before and after the cutoff
    before_cutoff = df[df['reference_date'] < MIN_DATE]
    after_cutoff = df[df['reference_date'] >= MIN_DATE]
    
    before_count = len(before_cutoff)
    after_count = len(after_cutoff)
    before_positives = before_cutoff['target'].sum() if before_count > 0 else 0
    before_negatives = before_count - before_positives
    after_positives = after_cutoff['target'].sum() if after_count > 0 else 0
    after_negatives = after_count - after_positives
    
    print(f"\n   Samples before {MIN_DATE.date()} (would be removed):")
    print(f"      Total: {before_count:,} ({before_count/original_count:.2%})")
    print(f"      Positives: {before_positives:,}")
    print(f"      Negatives: {before_negatives:,}")
    
    print(f"\n   Samples >= {MIN_DATE.date()} (would be kept):")
    print(f"      Total: {after_count:,} ({after_count/original_count:.2%})")
    print(f"      Positives: {after_positives:,}")
    print(f"      Negatives: {after_negatives:,}")
    
    # Show date distribution by year for samples before cutoff
    if before_count > 0:
        before_cutoff['year'] = before_cutoff['reference_date'].dt.year
        year_counts = before_cutoff['year'].value_counts().sort_index()
        print(f"\n   Year distribution of samples that would be removed:")
        for year, count in year_counts.items():
            print(f"      {year}: {count:,} samples")
    
    return {
        'original_count': original_count,
        'before_count': before_count,
        'after_count': after_count,
        'min_date': min_date,
        'max_date': max_date
    }

def filter_dataset(input_file, output_file, dataset_name):
    """Filter dataset to exclude samples with reference_date <= 2017-06-30"""
    print(f"\n📂 Processing {dataset_name}...")
    print(f"   Loading: {input_file}")
    
    df = pd.read_csv(input_file, encoding='utf-8-sig')
    original_count = len(df)
    original_positives = df['target'].sum()
    original_negatives = original_count - original_positives
    
    print(f"   Original: {original_count:,} samples ({original_positives:,} positives, {original_negatives:,} negatives)")
    
    # Convert reference_date to datetime
    df['reference_date'] = pd.to_datetime(df['reference_date'], errors='coerce')
    
    # Check for any parsing errors
    invalid_dates = df['reference_date'].isna().sum()
    if invalid_dates > 0:
        print(f"   ⚠️  WARNING: {invalid_dates:,} samples have invalid dates!")
    
    # Show date range statistics
    min_date = df['reference_date'].min()
    max_date = df['reference_date'].max()
    print(f"   Date range: {min_date} to {max_date}")
    
    # Count samples before and after the cutoff
    before_cutoff = df[df['reference_date'] < MIN_DATE]
    after_cutoff = df[df['reference_date'] >= MIN_DATE]
    
    before_count = len(before_cutoff)
    after_count = len(after_cutoff)
    before_positives = before_cutoff['target'].sum() if before_count > 0 else 0
    before_negatives = before_count - before_positives
    after_positives = after_cutoff['target'].sum() if after_count > 0 else 0
    after_negatives = after_count - after_positives
    
    print(f"   Samples before {MIN_DATE.date()}: {before_count:,} ({before_count/original_count:.2%})")
    print(f"   Samples >= {MIN_DATE.date()}: {after_count:,} ({after_count/original_count:.2%})")
    
    # Filter: keep only samples with reference_date >= 2017-07-01
    df_filtered = df[df['reference_date'] >= MIN_DATE].copy()
    
    filtered_count = len(df_filtered)
    filtered_positives = df_filtered['target'].sum()
    filtered_negatives = filtered_count - filtered_positives
    
    print(f"   After filtering (>= {MIN_DATE.date()}): {filtered_count:,} samples ({filtered_positives:,} positives, {filtered_negatives:,} negatives)")
    print(f"   Removed: {original_count - filtered_count:,} samples ({(original_count - filtered_count) / original_count:.1%})")
    
    if filtered_count > 0:
        print(f"   Saving to: {output_file}")
        df_filtered.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"   ✅ Saved {filtered_count:,} samples")
        return filtered_count, filtered_positives, filtered_negatives
    else:
        print(f"   ⚠️  No samples remaining after filtering!")
        return 0, 0, 0

def main():
    import sys
    
    # Check if diagnostic-only mode is requested
    DIAGNOSTIC_ONLY = '--diagnostic' in sys.argv or '-d' in sys.argv
    
    print("=" * 80)
    if DIAGNOSTIC_ONLY:
        print("DIAGNOSTIC: ANALYZE NATURAL DATASETS - POST 2017-06-30")
    else:
        print("FILTER NATURAL DATASETS - POST 2017-06-30")
    print("=" * 80)
    print(f"\n📋 Configuration:")
    print(f"   Minimum date: {MIN_DATE.date()} (season 2017/18 onwards)")
    print(f"   Input train: {INPUT_TRAIN_FILE}")
    print(f"   Input val: {INPUT_VAL_FILE}")
    if not DIAGNOSTIC_ONLY:
        print(f"   Output train: {OUTPUT_TRAIN_FILE}")
        print(f"   Output val: {OUTPUT_VAL_FILE}")
    print("=" * 80)
    
    if DIAGNOSTIC_ONLY:
        # Diagnostic mode - only analyze, don't filter
        train_stats = diagnose_dataset(INPUT_TRAIN_FILE, "Training Dataset")
        val_stats = diagnose_dataset(INPUT_VAL_FILE, "Validation Dataset")
        
        # Summary
        print("\n" + "=" * 80)
        print("📊 DIAGNOSTIC SUMMARY")
        print("=" * 80)
        print(f"\n   Training Dataset:")
        print(f"      Original samples: {train_stats['original_count']:,}")
        print(f"      Would be removed: {train_stats['before_count']:,} ({train_stats['before_count']/train_stats['original_count']:.2%})")
        print(f"      Would be kept: {train_stats['after_count']:,} ({train_stats['after_count']/train_stats['original_count']:.2%})")
        print(f"      Date range: {train_stats['min_date']} to {train_stats['max_date']}")
        
        print(f"\n   Validation Dataset:")
        print(f"      Original samples: {val_stats['original_count']:,}")
        print(f"      Would be removed: {val_stats['before_count']:,} ({val_stats['before_count']/val_stats['original_count']:.2%})")
        print(f"      Would be kept: {val_stats['after_count']:,} ({val_stats['after_count']/val_stats['original_count']:.2%})")
        print(f"      Date range: {val_stats['min_date']} to {val_stats['max_date']}")
        
        print("\n✅ Diagnostic complete!")
        print("\n💡 To actually filter the datasets, run without --diagnostic flag")
    else:
        # Filter mode - actually filter and save
        train_count, train_pos, train_neg = filter_dataset(
            INPUT_TRAIN_FILE, 
            OUTPUT_TRAIN_FILE, 
            "Training Dataset"
        )
        
        val_count, val_pos, val_neg = filter_dataset(
            INPUT_VAL_FILE, 
            OUTPUT_VAL_FILE, 
            "Validation Dataset"
        )
        
        # Summary
        print("\n" + "=" * 80)
        print("📊 SUMMARY")
        print("=" * 80)
        print(f"\n   Training Dataset:")
        print(f"      Total: {train_count:,} samples")
        if train_count > 0:
            print(f"      Positives: {train_pos:,} ({train_pos/train_count:.2%})")
            print(f"      Negatives: {train_neg:,} ({train_neg/train_count:.2%})")
        else:
            print(f"      Positives: 0")
            print(f"      Negatives: 0")
        
        print(f"\n   Validation Dataset:")
        print(f"      Total: {val_count:,} samples")
        if val_count > 0:
            print(f"      Positives: {val_pos:,} ({val_pos/val_count:.2%})")
            print(f"      Negatives: {val_neg:,} ({val_neg/val_count:.2%})")
        else:
            print(f"      Positives: 0")
            print(f"      Negatives: 0")
        
        print("\n✅ Filtering complete!")
        print(f"\n💾 New datasets saved:")
        print(f"   - {OUTPUT_TRAIN_FILE}")
        print(f"   - {OUTPUT_VAL_FILE}")

if __name__ == '__main__':
    main()

