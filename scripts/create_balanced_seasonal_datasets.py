#!/usr/bin/env python3
"""
Create balanced datasets from season-by-season timeline datasets.
For each season (except 2025-2026), creates 3 balanced versions:
- 10% target ratio
- 25% target ratio  
- 50% target ratio

All positives are kept, negatives are downsampled.
The 2025-2026 season is excluded as it's the test dataset (kept with natural ratio).
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import random
import glob
import os
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

# Configuration
TARGET_RATIOS = [0.10, 0.25, 0.50]  # 10%, 25%, 50%
SEASON_FILES_PATTERN = 'timelines_35day_season_*_v4_muscular.csv'
EXCLUDE_SEASON = '2025_2026'  # Test dataset - keep natural ratio only
RANDOM_STATE = 42

# Set random seed for reproducibility
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

def find_season_files(pattern=SEASON_FILES_PATTERN, exclude_season=EXCLUDE_SEASON):
    """
    Find all season timeline files, excluding the test season.
    
    Returns:
        List of (season_id, filepath) tuples, sorted chronologically
    """
    files = glob.glob(pattern)
    season_files = []
    
    for filepath in files:
        filename = os.path.basename(filepath)
        # Extract season from filename: timelines_35day_season_2000_2001_v4_muscular.csv
        if 'season_' in filename:
            parts = filename.split('season_')
            if len(parts) > 1:
                season_part = parts[1].split('_v4_muscular')[0]
                if season_part != exclude_season:
                    season_files.append((season_part, filepath))
    
    # Sort by season (chronologically)
    season_files.sort(key=lambda x: x[0])
    return season_files

def downsample_to_target_ratio(df, target_ratio, dataset_name, random_state=RANDOM_STATE):
    """
    Downsample negative class to achieve target ratio while keeping ALL positives.
    
    Args:
        df: DataFrame with 'target' column
        target_ratio: Desired ratio of positives (e.g., 0.10 for 10%)
        dataset_name: Name of dataset (for logging)
        random_state: Random seed for reproducibility
    
    Returns:
        DataFrame with downsampled negatives
    """
    # Separate positives and negatives
    positives = df[df['target'] == 1].copy()
    negatives = df[df['target'] == 0].copy()
    
    n_positives = len(positives)
    n_negatives = len(negatives)
    
    # Calculate how many negatives we need
    # target_ratio = n_positives / (n_positives + n_negatives_needed)
    # Solving for n_negatives_needed:
    # n_negatives_needed = n_positives / target_ratio - n_positives
    n_negatives_needed = int(n_positives / target_ratio - n_positives)
    
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
    
    return balanced_df

def process_season_file(season_id, input_file, target_ratios=TARGET_RATIOS):
    """
    Process a single season file to create balanced versions.
    
    Args:
        season_id: Season identifier (e.g., '2000_2001')
        input_file: Path to the season CSV file
        target_ratios: List of target ratios to create
    
    Returns:
        List of (output_file, stats_dict) tuples
    """
    print(f"\n📂 Processing season {season_id}...")
    print(f"   Input: {input_file}")
    
    # Load season dataset
    df = pd.read_csv(input_file, encoding='utf-8-sig')
    original_positives = df['target'].sum()
    original_negatives = len(df) - original_positives
    original_ratio = df['target'].mean()
    
    print(f"   Original: {len(df):,} records")
    print(f"      Positives: {original_positives:,} ({original_ratio:.2%})")
    print(f"      Negatives: {original_negatives:,}")
    
    results = []
    
    # Process each target ratio
    for target_ratio in target_ratios:
        ratio_str = f"{int(target_ratio * 100):02d}pc"  # e.g., "10pc", "25pc", "50pc"
        output_file = f'timelines_35day_season_{season_id}_{ratio_str}_v4_muscular.csv'
        
        print(f"\n   Creating {target_ratio:.0%} target ratio dataset...")
        
        # Downsample to target ratio
        balanced_df = downsample_to_target_ratio(
            df, 
            target_ratio, 
            f"Season {season_id} ({target_ratio:.0%})",
            random_state=RANDOM_STATE
        )
        
        # Calculate final stats
        final_positives = balanced_df['target'].sum()
        final_negatives = len(balanced_df) - final_positives
        final_ratio = balanced_df['target'].mean()
        
        # Save balanced dataset
        print(f"   💾 Saving to {output_file}...")
        balanced_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        stats = {
            'season': season_id,
            'target_ratio': target_ratio,
            'original_positives': original_positives,
            'original_negatives': original_negatives,
            'original_ratio': original_ratio,
            'final_positives': final_positives,
            'final_negatives': final_negatives,
            'final_ratio': final_ratio,
            'total_records': len(balanced_df),
            'output_file': output_file
        }
        
        results.append((output_file, stats))
        
        print(f"   ✅ Saved: {len(balanced_df):,} records ({final_positives:,} positives, {final_negatives:,} negatives, {final_ratio:.2%} ratio)")
        
        # Free memory
        del balanced_df
    
    return results

def main():
    print("="*80)
    print("CREATING BALANCED SEASONAL DATASETS")
    print("="*80)
    print("\n📋 Configuration:")
    print(f"   Target ratios: {[f'{r:.0%}' for r in TARGET_RATIOS]}")
    print(f"   Strategy: Keep ALL positives, downsample negatives only")
    print(f"   Random seed: {RANDOM_STATE} (for reproducibility)")
    print(f"   Excluding season: {EXCLUDE_SEASON} (test dataset - keeping natural ratio)")
    print("="*80)
    
    start_time = datetime.now()
    
    # Find all season files
    print("\n🔍 Discovering season files...")
    season_files = find_season_files()
    
    if not season_files:
        print("❌ No season files found!")
        print(f"   Looking for pattern: {SEASON_FILES_PATTERN}")
        return
    
    print(f"✅ Found {len(season_files)} season files to process:")
    for season_id, filepath in season_files:
        print(f"   - {season_id}: {filepath}")
    
    # Process each season
    all_results = []
    
    for idx, (season_id, input_file) in enumerate(season_files, 1):
        print("\n" + "="*80)
        print(f"PROCESSING SEASON {idx}/{len(season_files)}: {season_id}")
        print("="*80)
        
        try:
            results = process_season_file(season_id, input_file, TARGET_RATIOS)
            all_results.extend(results)
        except Exception as e:
            print(f"❌ Error processing {season_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    total_time = datetime.now() - start_time
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n✅ Processed {len(season_files)} seasons")
    print(f"✅ Created {len(all_results)} balanced datasets")
    print(f"\n📊 Breakdown by target ratio:")
    
    for ratio in TARGET_RATIOS:
        ratio_results = [r for r in all_results if r[1]['target_ratio'] == ratio]
        print(f"   {ratio:.0%} ratio: {len(ratio_results)} datasets")
    
    print(f"\n⏱️  Total execution time: {total_time}")
    print("\n📁 Generated files:")
    for output_file, stats in all_results:
        print(f"   - {output_file} ({stats['total_records']:,} records, {stats['final_ratio']:.2%} ratio)")
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()

