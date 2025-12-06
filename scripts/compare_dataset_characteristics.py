#!/usr/bin/env python3
"""
Compare Training, In-Sample, and Out-of-Sample datasets
- Number of unique players
- Distribution of career length
"""
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

def get_player_first_match_date(player_id):
    """Get player's first match date from daily features file"""
    daily_features_file = f'daily_features_output/player_{player_id}_daily_features.csv'
    if not Path(daily_features_file).exists():
        return None
    
    try:
        df_daily = pd.read_csv(daily_features_file)
        df_daily['date'] = pd.to_datetime(df_daily['date'])
        first_match = df_daily['date'].min()
        return first_match
    except:
        return None

def calculate_career_length(df):
    """Calculate career length for each player in the dataset
    Uses actual first match date from daily features, not dataset date range
    """
    player_stats = []
    
    for player_id in df['player_id'].unique():
        player_data = df[df['player_id'] == player_id].copy()
        player_data['reference_date'] = pd.to_datetime(player_data['reference_date'])
        
        # Get actual first match date from daily features
        first_match_date = get_player_first_match_date(player_id)
        
        if first_match_date is None:
            # Fallback: use earliest reference date in dataset
            first_match_date = player_data['reference_date'].min()
        
        # For each timeline, calculate career length at that point
        # Use median career length for the player in this dataset
        career_lengths = []
        for ref_date in player_data['reference_date']:
            career_length = (ref_date - first_match_date).days / 365.25
            career_lengths.append(career_length)
        
        median_career_length = np.median(career_lengths) if career_lengths else 0
        
        player_stats.append({
            'player_id': player_id,
            'first_match_date': first_match_date,
            'first_timeline_date': player_data['reference_date'].min(),
            'last_timeline_date': player_data['reference_date'].max(),
            'career_length_years': median_career_length,
            'career_length_min': min(career_lengths) if career_lengths else 0,
            'career_length_max': max(career_lengths) if career_lengths else 0,
            'num_timelines': len(player_data),
            'num_injuries': player_data['target'].sum()
        })
    
    return pd.DataFrame(player_stats)

def main():
    print("="*80)
    print("DATASET CHARACTERISTICS COMPARISON")
    print("="*80)
    print("📋 Comparing: Training, In-Sample Validation, Out-of-Sample Validation")
    print("="*80)
    
    # Load datasets
    print("\n📂 Loading datasets...")
    train_file = 'timelines_35day_enhanced_balanced_v4_train.csv'
    val_file = 'timelines_35day_enhanced_balanced_v4_val.csv'
    
    df_train_full = pd.read_csv(train_file, encoding='utf-8-sig')
    df_val_outsample = pd.read_csv(val_file, encoding='utf-8-sig')
    
    print(f"✅ Loaded training set: {len(df_train_full):,} records")
    print(f"✅ Loaded out-of-sample validation set: {len(df_val_outsample):,} records")
    
    # Convert dates
    df_train_full['reference_date'] = pd.to_datetime(df_train_full['reference_date'])
    df_val_outsample['reference_date'] = pd.to_datetime(df_val_outsample['reference_date'])
    
    # Split training into train and in-sample validation (80/20)
    from sklearn.model_selection import train_test_split
    
    train_indices, val_insample_indices = train_test_split(
        df_train_full.index,
        test_size=0.2,
        random_state=42,
        stratify=df_train_full['target']
    )
    
    df_train = df_train_full.loc[train_indices].copy()
    df_val_insample = df_train_full.loc[val_insample_indices].copy()
    
    print(f"✅ Split training set:")
    print(f"   Training (80%): {len(df_train):,} records")
    print(f"   In-sample validation (20%): {len(df_val_insample):,} records")
    
    # Calculate player statistics for each dataset
    print("\n📊 Calculating player statistics...")
    
    print("   Processing training set...")
    train_players = calculate_career_length(df_train)
    
    print("   Processing in-sample validation set...")
    insample_players = calculate_career_length(df_val_insample)
    
    print("   Processing out-of-sample validation set...")
    outsample_players = calculate_career_length(df_val_outsample)
    
    # Basic statistics
    print("\n" + "="*80)
    print("📊 DATASET SUMMARY")
    print("="*80)
    
    print(f"\n📈 Number of Unique Players:")
    print(f"   Training: {len(train_players):,} players")
    print(f"   In-Sample Validation: {len(insample_players):,} players")
    print(f"   Out-of-Sample Validation: {len(outsample_players):,} players")
    
    # Player overlap
    train_player_ids = set(train_players['player_id'])
    insample_player_ids = set(insample_players['player_id'])
    outsample_player_ids = set(outsample_players['player_id'])
    
    print(f"\n📊 Player Overlap:")
    print(f"   Training ∩ In-Sample: {len(train_player_ids & insample_player_ids):,} players")
    print(f"   Training ∩ Out-of-Sample: {len(train_player_ids & outsample_player_ids):,} players")
    print(f"   In-Sample ∩ Out-of-Sample: {len(insample_player_ids & outsample_player_ids):,} players")
    print(f"   All three: {len(train_player_ids & insample_player_ids & outsample_player_ids):,} players")
    
    # Career length statistics
    print("\n" + "="*80)
    print("📊 CAREER LENGTH DISTRIBUTION")
    print("="*80)
    
    datasets = {
        'Training': train_players,
        'In-Sample Validation': insample_players,
        'Out-of-Sample Validation': outsample_players
    }
    
    career_stats = {}
    
    for dataset_name, player_df in datasets.items():
        career_lengths = player_df['career_length_years']
        
        stats = {
            'count': len(career_lengths),
            'mean': career_lengths.mean(),
            'median': career_lengths.median(),
            'std': career_lengths.std(),
            'min': career_lengths.min(),
            'max': career_lengths.max(),
            'q25': career_lengths.quantile(0.25),
            'q75': career_lengths.quantile(0.75)
        }
        
        career_stats[dataset_name] = stats
        
        print(f"\n{dataset_name}:")
        print(f"   Count: {stats['count']:,} players")
        print(f"   Mean: {stats['mean']:.2f} years")
        print(f"   Median: {stats['median']:.2f} years")
        print(f"   Std Dev: {stats['std']:.2f} years")
        print(f"   Min: {stats['min']:.2f} years")
        print(f"   Max: {stats['max']:.2f} years")
        print(f"   Q25: {stats['q25']:.2f} years")
        print(f"   Q75: {stats['q75']:.2f} years")
    
    # Comparison
    print("\n" + "="*80)
    print("📊 COMPARISON")
    print("="*80)
    
    train_mean = career_stats['Training']['mean']
    insample_mean = career_stats['In-Sample Validation']['mean']
    outsample_mean = career_stats['Out-of-Sample Validation']['mean']
    
    print(f"\n📈 Mean Career Length:")
    print(f"   Training: {train_mean:.2f} years")
    print(f"   In-Sample Validation: {insample_mean:.2f} years ({insample_mean - train_mean:+.2f} vs training)")
    print(f"   Out-of-Sample Validation: {outsample_mean:.2f} years ({outsample_mean - train_mean:+.2f} vs training)")
    
    if abs(outsample_mean - train_mean) > 1.0:
        print(f"   ⚠️  SIGNIFICANT DIFFERENCE: Out-of-sample players have {outsample_mean - train_mean:+.2f} years different career length")
    
    # Date ranges
    print(f"\n📅 Date Ranges:")
    print(f"   Training:")
    print(f"      First date: {df_train['reference_date'].min()}")
    print(f"      Last date: {df_train['reference_date'].max()}")
    print(f"      Span: {(df_train['reference_date'].max() - df_train['reference_date'].min()).days / 365.25:.1f} years")
    
    print(f"   In-Sample Validation:")
    print(f"      First date: {df_val_insample['reference_date'].min()}")
    print(f"      Last date: {df_val_insample['reference_date'].max()}")
    print(f"      Span: {(df_val_insample['reference_date'].max() - df_val_insample['reference_date'].min()).days / 365.25:.1f} years")
    
    print(f"   Out-of-Sample Validation:")
    print(f"      First date: {df_val_outsample['reference_date'].min()}")
    print(f"      Last date: {df_val_outsample['reference_date'].max()}")
    print(f"      Span: {(df_val_outsample['reference_date'].max() - df_val_outsample['reference_date'].min()).days / 365.25:.1f} years")
    
    # Save results
    output_dir = Path('experiments')
    output_dir.mkdir(exist_ok=True)
    
    # Save player statistics
    train_players.to_csv(output_dir / 'train_players_stats.csv', index=False)
    insample_players.to_csv(output_dir / 'insample_players_stats.csv', index=False)
    outsample_players.to_csv(output_dir / 'outsample_players_stats.csv', index=False)
    
    # Save summary
    summary = {
        'num_players': {
            'training': int(len(train_players)),
            'insample': int(len(insample_players)),
            'outsample': int(len(outsample_players))
        },
        'player_overlap': {
            'train_insample': int(len(train_player_ids & insample_player_ids)),
            'train_outsample': int(len(train_player_ids & outsample_player_ids)),
            'insample_outsample': int(len(insample_player_ids & outsample_player_ids)),
            'all_three': int(len(train_player_ids & insample_player_ids & outsample_player_ids))
        },
        'career_length_stats': career_stats,
        'date_ranges': {
            'training': {
                'first': str(df_train['reference_date'].min()),
                'last': str(df_train['reference_date'].max())
            },
            'insample': {
                'first': str(df_val_insample['reference_date'].min()),
                'last': str(df_val_insample['reference_date'].max())
            },
            'outsample': {
                'first': str(df_val_outsample['reference_date'].min()),
                'last': str(df_val_outsample['reference_date'].max())
            }
        }
    }
    
    with open(output_dir / 'dataset_characteristics_comparison.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Create markdown summary
    md_file = output_dir / 'dataset_characteristics_comparison.md'
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# Dataset Characteristics Comparison\n\n")
        f.write("**Date:** 2025-11-27\n\n")
        
        f.write("## Number of Unique Players\n\n")
        f.write("| Dataset | Number of Players |\n")
        f.write("|---------|-------------------|\n")
        f.write(f"| Training | {len(train_players):,} |\n")
        f.write(f"| In-Sample Validation | {len(insample_players):,} |\n")
        f.write(f"| Out-of-Sample Validation | {len(outsample_players):,} |\n")
        
        f.write("\n## Player Overlap\n\n")
        f.write("| Overlap | Number of Players |\n")
        f.write("|---------|-------------------|\n")
        f.write(f"| Training ∩ In-Sample | {len(train_player_ids & insample_player_ids):,} |\n")
        f.write(f"| Training ∩ Out-of-Sample | {len(train_player_ids & outsample_player_ids):,} |\n")
        f.write(f"| In-Sample ∩ Out-of-Sample | {len(insample_player_ids & outsample_player_ids):,} |\n")
        f.write(f"| All Three | {len(train_player_ids & insample_player_ids & outsample_player_ids):,} |\n")
        
        f.write("\n## Career Length Distribution\n\n")
        f.write("| Dataset | Count | Mean | Median | Std Dev | Min | Max | Q25 | Q75 |\n")
        f.write("|---------|-------|------|--------|---------|-----|-----|-----|-----|\n")
        
        for dataset_name, stats in career_stats.items():
            f.write(f"| {dataset_name} | {stats['count']:,} | {stats['mean']:.2f} | "
                   f"{stats['median']:.2f} | {stats['std']:.2f} | {stats['min']:.2f} | "
                   f"{stats['max']:.2f} | {stats['q25']:.2f} | {stats['q75']:.2f} |\n")
        
        f.write("\n## Career Length Comparison\n\n")
        f.write(f"- **Training mean:** {train_mean:.2f} years\n")
        f.write(f"- **In-Sample Validation mean:** {insample_mean:.2f} years "
               f"({insample_mean - train_mean:+.2f} vs training)\n")
        f.write(f"- **Out-of-Sample Validation mean:** {outsample_mean:.2f} years "
               f"({outsample_mean - train_mean:+.2f} vs training)\n")
        
        if abs(outsample_mean - train_mean) > 1.0:
            f.write(f"\n⚠️ **SIGNIFICANT DIFFERENCE:** Out-of-sample players have "
                   f"{outsample_mean - train_mean:+.2f} years different career length on average.\n")
            f.write("This could contribute to distribution shift and model performance issues.\n")
        
        f.write("\n## Date Ranges\n\n")
        f.write("| Dataset | First Date | Last Date | Span (years) |\n")
        f.write("|---------|------------|-----------|--------------|\n")
        
        train_span = (df_train['reference_date'].max() - df_train['reference_date'].min()).days / 365.25
        insample_span = (df_val_insample['reference_date'].max() - df_val_insample['reference_date'].min()).days / 365.25
        outsample_span = (df_val_outsample['reference_date'].max() - df_val_outsample['reference_date'].min()).days / 365.25
        
        f.write(f"| Training | {df_train['reference_date'].min()} | {df_train['reference_date'].max()} | {train_span:.1f} |\n")
        f.write(f"| In-Sample Validation | {df_val_insample['reference_date'].min()} | "
               f"{df_val_insample['reference_date'].max()} | {insample_span:.1f} |\n")
        f.write(f"| Out-of-Sample Validation | {df_val_outsample['reference_date'].min()} | "
               f"{df_val_outsample['reference_date'].max()} | {outsample_span:.1f} |\n")
    
    print(f"\n✅ Results saved to {output_dir / 'dataset_characteristics_comparison.json'}")
    print(f"✅ Summary saved to {md_file}")
    print("\n" + "="*80)
    print("DATASET CHARACTERISTICS COMPARISON COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()

