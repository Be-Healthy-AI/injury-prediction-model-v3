#!/usr/bin/env python3
"""
Create balanced datasets with specific target ratios from natural datasets.
Downsamples negative (non-injury) timelines while keeping ALL positive (injury) timelines.
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import random
import csv
from datetime import datetime
from tqdm import tqdm

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Target ratios to create
TARGET_RATIOS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]  # 5%, 10%, 15%, 20%, 25%, 30%, 40%, 50%

# Input files (natural datasets)
USE_POST2017_FILTER = True  # Set to True to use filtered datasets (post 2017-06-30)

if USE_POST2017_FILTER:
    NATURAL_TRAIN_FILE = 'timelines_35day_enhanced_natural_post2017_v4_muscular_train.csv'
    NATURAL_VAL_FILE = 'timelines_35day_enhanced_natural_post2017_v4_muscular_val.csv'
    OUTPUT_SUFFIX = '_post2017'  # Add this suffix to output filenames
else:
    NATURAL_TRAIN_FILE = 'timelines_35day_enhanced_natural_v4_muscular_train.csv'
    NATURAL_VAL_FILE = 'timelines_35day_enhanced_natural_v4_muscular_val.csv'
    OUTPUT_SUFFIX = ''  # No suffix for original datasets
# Test file is not modified

def save_timelines_to_csv_chunked(timelines, output_file, chunk_size=10000):
    """
    Save timelines to CSV in chunks to avoid memory issues.
    
    Args:
        timelines: List of dictionaries or DataFrame containing timeline data
        output_file: Output CSV file path
        chunk_size: Number of rows to process at a time
    """
    if len(timelines) == 0:
        return (0, 0)
    
    # Convert to list of dicts if DataFrame
    if isinstance(timelines, pd.DataFrame):
        timelines = timelines.to_dict('records')
    
    # Get all unique keys from all timelines to ensure consistent columns
    all_keys = set()
    for timeline in timelines:
        all_keys.update(timeline.keys())
    
    # Sort keys for consistent column order
    fieldnames = sorted(all_keys)
    
    # Write CSV in chunks
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        
        # Write in chunks to manage memory
        for i in tqdm(range(0, len(timelines), chunk_size), 
                     desc=f"   Saving {output_file}", 
                     unit="chunk",
                     leave=False):
            chunk = timelines[i:i + chunk_size]
            for timeline in chunk:
                # Ensure all keys are present (fill missing with None)
                row = {key: timeline.get(key, None) for key in fieldnames}
                writer.writerow(row)
    
    # Get shape info
    sample_df = pd.read_csv(output_file, nrows=1)
    num_cols = len(sample_df.columns)
    num_rows = len(timelines)
    del sample_df  # Free memory
    return (num_rows, num_cols)

def downsample_to_target_ratio(df, target_ratio, dataset_name, random_state=42):
    """
    Downsample negative class to achieve target ratio while keeping ALL positives.
    
    Args:
        df: DataFrame with 'target' column
        target_ratio: Desired ratio of positives (e.g., 0.05 for 5%)
        dataset_name: Name of dataset (for logging)
        random_state: Random seed for reproducibility
    
    Returns:
        DataFrame with downsampled negatives
    """
    print(f"\n   Processing {dataset_name}...")
    
    # Separate positives and negatives
    positives = df[df['target'] == 1].copy()
    negatives = df[df['target'] == 0].copy()
    
    n_positives = len(positives)
    n_negatives = len(negatives)
    
    print(f"      Original: {n_positives:,} positives, {n_negatives:,} negatives")
    print(f"      Original ratio: {n_positives / (n_positives + n_negatives):.2%}")
    
    # Calculate how many negatives we need
    # target_ratio = n_positives / (n_positives + n_negatives_needed)
    # Solving for n_negatives_needed:
    # n_negatives_needed = n_positives / target_ratio - n_positives
    n_negatives_needed = int(n_positives / target_ratio - n_positives)
    
    print(f"      Target ratio: {target_ratio:.1%}")
    print(f"      Negatives needed: {n_negatives_needed:,}")
    
    # Check if we have enough negatives
    if n_negatives_needed > n_negatives:
        print(f"      ⚠️  WARNING: Need {n_negatives_needed:,} negatives but only have {n_negatives:,}")
        print(f"      Using all available negatives (will result in ratio > {target_ratio:.1%})")
        n_negatives_needed = n_negatives
    
    # Randomly sample negatives
    if n_negatives_needed < n_negatives:
        negatives_sampled = negatives.sample(n=n_negatives_needed, random_state=random_state)
    else:
        negatives_sampled = negatives.copy()
    
    # Combine positives and sampled negatives
    balanced_df = pd.concat([positives, negatives_sampled], ignore_index=True)
    
    # Shuffle the combined dataset
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Calculate final ratio
    final_ratio = balanced_df['target'].mean()
    print(f"      Final: {len(positives):,} positives, {len(negatives_sampled):,} negatives")
    print(f"      Final ratio: {final_ratio:.2%}")
    print(f"      Total: {len(balanced_df):,} records")
    
    return balanced_df

def main():
    print("="*80)
    print("CREATING BALANCED DATASETS FROM NATURAL DATASETS")
    print("="*80)
    print("\n📋 Configuration:")
    print(f"   Target ratios: {[f'{r:.0%}' for r in TARGET_RATIOS]}")
    print(f"   Strategy: Keep ALL positives, downsample negatives only")
    print(f"   Random seed: 42 (for reproducibility)")
    if USE_POST2017_FILTER:
        print(f"   Using filtered datasets: Post 2017-06-30 only")
        print(f"   Output suffix: {OUTPUT_SUFFIX}")
    else:
        print(f"   Using original natural datasets")
    print("="*80)
    
    start_time = datetime.now()
    
    # Load natural datasets
    print("\n📂 Loading natural datasets...")
    print(f"   Loading {NATURAL_TRAIN_FILE}...")
    df_train_natural = pd.read_csv(NATURAL_TRAIN_FILE, encoding='utf-8-sig')
    print(f"   ✅ Loaded: {len(df_train_natural):,} records")
    print(f"      Positives: {df_train_natural['target'].sum():,} ({df_train_natural['target'].mean():.2%})")
    
    print(f"\n   Loading {NATURAL_VAL_FILE}...")
    df_val_natural = pd.read_csv(NATURAL_VAL_FILE, encoding='utf-8-sig')
    print(f"   ✅ Loaded: {len(df_val_natural):,} records")
    print(f"      Positives: {df_val_natural['target'].sum():,} ({df_val_natural['target'].mean():.2%})")
    
    # Process each target ratio
    for target_ratio in TARGET_RATIOS:
        print("\n" + "="*80)
        print(f"CREATING DATASETS WITH {target_ratio:.0%} TARGET RATIO")
        print("="*80)
        
        # Process training set
        df_train_balanced = downsample_to_target_ratio(
            df_train_natural, 
            target_ratio, 
            "Training set"
        )
        
        # Process validation set
        df_val_balanced = downsample_to_target_ratio(
            df_val_natural, 
            target_ratio, 
            "Validation set"
        )
        
        # Create output filenames
        ratio_str = f"{int(target_ratio * 100):02d}pc"  # e.g., "05pc", "10pc", "15pc"
        train_output = f'timelines_35day_enhanced_{ratio_str}_v4_muscular_train{OUTPUT_SUFFIX}.csv'
        val_output = f'timelines_35day_enhanced_{ratio_str}_v4_muscular_val{OUTPUT_SUFFIX}.csv'
        
        # Save datasets
        print(f"\n💾 Saving balanced datasets...")
        shape_train = save_timelines_to_csv_chunked(df_train_balanced, train_output)
        shape_val = save_timelines_to_csv_chunked(df_val_balanced, val_output)
        
        print(f"\n✅ Saved training set: {train_output}")
        print(f"   Shape: {shape_train}")
        print(f"✅ Saved validation set: {val_output}")
        print(f"   Shape: {shape_val}")
        
        # Free memory
        del df_train_balanced, df_val_balanced
    
    total_time = datetime.now() - start_time
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n✅ Created {len(TARGET_RATIOS)} sets of balanced datasets:")
    for ratio in TARGET_RATIOS:
        ratio_str = f"{int(ratio * 100):02d}pc"
        print(f"   - {ratio:.0%} ratio: train + val datasets")
    print(f"\n⏱️  Total execution time: {total_time}")
    print("\n📊 Final dataset inventory:")
    print("   Natural datasets (unchanged):")
    print("      - train, val, test")
    print("   Balanced datasets (new):")
    for ratio in TARGET_RATIOS:
        ratio_str = f"{int(ratio * 100):02d}pc"
        print(f"      - train_{ratio_str}, val_{ratio_str}")
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()

