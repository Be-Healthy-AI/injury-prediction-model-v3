import pandas as pd
import os
from pathlib import Path

# Paths
timelines_train_dir = Path(r"C:\Users\joao.henriques\IPM V3\models_production\lgbm_muscular_v4\data\timelines\train")
timelines_test_dir = Path(r"C:\Users\joao.henriques\IPM V3\models_production\lgbm_muscular_v4\data\timelines\test")

results = []

# Check train files
for file in sorted(timelines_train_dir.glob("*.csv")):
    try:
        # First check if columns exist
        sample_df = pd.read_csv(file, nrows=1)
        if 'target1' not in sample_df.columns or 'target2' not in sample_df.columns:
            season = file.stem.replace('timelines_35day_season_', '').replace('_v4_muscular_train', '')
            results.append({
                'Season': season,
                'Type': 'Train',
                'Total': 'OLD FORMAT',
                'target1=1 (Muscular)': 'N/A',
                'target2=1 (Skeletal)': 'N/A',
                'target1 %': 'N/A',
                'target2 %': 'N/A'
            })
            continue
        
        # Read just the target columns to be efficient
        df = pd.read_csv(file, usecols=['target1', 'target2'], low_memory=False)
        
        season = file.stem.replace('timelines_35day_season_', '').replace('_v4_muscular_train', '')
        target1_count = (df['target1'] == 1).sum()
        target2_count = (df['target2'] == 1).sum()
        total = len(df)
        
        results.append({
            'Season': season,
            'Type': 'Train',
            'Total': total,
            'target1=1 (Muscular)': target1_count,
            'target2=1 (Skeletal)': target2_count,
            'target1 %': f"{(target1_count/total*100):.2f}%" if total > 0 else "N/A",
            'target2 %': f"{(target2_count/total*100):.2f}%" if total > 0 else "N/A"
        })
    except Exception as e:
        season = file.stem.replace('timelines_35day_season_', '').replace('_v4_muscular_train', '')
        results.append({
            'Season': season,
            'Type': 'Train',
            'Total': 'ERROR',
            'target1=1 (Muscular)': f"Error: {str(e)}",
            'target2=1 (Skeletal)': '',
            'target1 %': '',
            'target2 %': ''
        })

# Check test files
for file in sorted(timelines_test_dir.glob("*.csv")):
    try:
        # First check if columns exist
        sample_df = pd.read_csv(file, nrows=1)
        if 'target1' not in sample_df.columns or 'target2' not in sample_df.columns:
            season = file.stem.replace('timelines_35day_season_', '').replace('_v4_muscular_test', '')
            results.append({
                'Season': season,
                'Type': 'Test',
                'Total': 'OLD FORMAT',
                'target1=1 (Muscular)': 'N/A',
                'target2=1 (Skeletal)': 'N/A',
                'target1 %': 'N/A',
                'target2 %': 'N/A'
            })
            continue
        
        # Read just the target columns to be efficient
        df = pd.read_csv(file, usecols=['target1', 'target2'], low_memory=False)
        
        season = file.stem.replace('timelines_35day_season_', '').replace('_v4_muscular_test', '')
        target1_count = (df['target1'] == 1).sum()
        target2_count = (df['target2'] == 1).sum()
        total = len(df)
        
        results.append({
            'Season': season,
            'Type': 'Test',
            'Total': total,
            'target1=1 (Muscular)': target1_count,
            'target2=1 (Skeletal)': target2_count,
            'target1 %': f"{(target1_count/total*100):.2f}%" if total > 0 else "N/A",
            'target2 %': f"{(target2_count/total*100):.2f}%" if total > 0 else "N/A"
        })
    except Exception as e:
        season = file.stem.replace('timelines_35day_season_', '').replace('_v4_muscular_test', '')
        results.append({
            'Season': season,
            'Type': 'Test',
            'Total': 'ERROR',
            'target1=1 (Muscular)': f"Error: {str(e)}",
            'target2=1 (Skeletal)': '',
            'target1 %': '',
            'target2 %': ''
        })

# Display results
print("\n" + "="*80)
print("TARGET RATIOS (MUSCULAR & SKELETAL) PER SEASON")
print("="*80)
print(f"{'Season':<20} {'Type':<8} {'Total':<12} {'target1=1':<12} {'target2=1':<12} {'target1 %':<10} {'target2 %':<10}")
print("-"*80)

for r in results:
    print(f"{r['Season']:<20} {r['Type']:<8} {str(r['Total']):<12} {str(r['target1=1 (Muscular)']):<12} {str(r['target2=1 (Skeletal)']):<12} {r['target1 %']:<10} {r['target2 %']:<10}")

# Summary - manually calculate from results
import numpy as np

total_target1 = 0
total_target2 = 0
total_timelines = 0

for r in results:
    val1 = r['target1=1 (Muscular)']
    val2 = r['target2=1 (Skeletal)']
    val_total = r['Total']
    
    # Handle both Python int and numpy int64
    if isinstance(val1, (int, np.integer)) and not isinstance(val1, str):
        total_target1 += int(val1)
    if isinstance(val2, (int, np.integer)) and not isinstance(val2, str):
        total_target2 += int(val2)
    if isinstance(val_total, (int, np.integer)) and not isinstance(val_total, str):
        total_timelines += int(val_total)

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Total timelines: {total_timelines:,}")
print(f"Total target1=1 (Muscular): {total_target1:,}")
print(f"Total target2=1 (Skeletal): {total_target2:,}")
if total_timelines > 0:
    print(f"Overall target1 ratio: {(total_target1/total_timelines*100):.2f}%")
    print(f"Overall target2 ratio: {(total_target2/total_timelines*100):.2f}%")
