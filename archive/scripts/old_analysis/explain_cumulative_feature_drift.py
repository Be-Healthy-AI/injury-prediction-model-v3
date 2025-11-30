#!/usr/bin/env python3
"""
Explain why cumulative features are higher in validation period
with concrete examples
"""
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from pathlib import Path

def main():
    print("="*80)
    print("EXPLAINING CUMULATIVE FEATURE DRIFT")
    print("="*80)
    
    # Load data
    print("\n📂 Loading data...")
    df_full = pd.read_csv('timelines_35day_enhanced_balanced_v4_train.csv', encoding='utf-8-sig')
    df_full['reference_date'] = pd.to_datetime(df_full['reference_date'])
    
    # Temporal split
    TRAIN_CUTOFF = pd.Timestamp('2024-06-30')
    VAL_END = pd.Timestamp('2025-06-30')
    
    train_mask = df_full['reference_date'] <= TRAIN_CUTOFF
    val_mask = (df_full['reference_date'] > TRAIN_CUTOFF) & (df_full['reference_date'] <= VAL_END)
    
    df_train = df_full[train_mask].copy()
    df_val = df_full[val_mask].copy()
    
    print(f"✅ Training: {len(df_train):,} records (<= 2024-06-30)")
    print(f"✅ Validation: {len(df_val):,} records (> 2024-06-30 and <= 2025-06-30)")
    
    # Analyze cum_inj_starts
    print("\n" + "="*80)
    print("ANALYZING cum_inj_starts")
    print("="*80)
    
    if 'cum_inj_starts' in df_train.columns:
        train_mean = df_train['cum_inj_starts'].mean()
        val_mean = df_val['cum_inj_starts'].mean()
        train_median = df_train['cum_inj_starts'].median()
        val_median = df_val['cum_inj_starts'].median()
        
        print(f"\n📊 Statistics:")
        print(f"   Training mean: {train_mean:.2f}")
        print(f"   Validation mean: {val_mean:.2f}")
        print(f"   Difference: {val_mean - train_mean:.2f} ({((val_mean - train_mean) / train_mean * 100):.1f}% higher)")
        print(f"\n   Training median: {train_median:.2f}")
        print(f"   Validation median: {val_median:.2f}")
        print(f"   Difference: {val_median - train_median:.2f} ({((val_median - train_median) / train_median * 100):.1f}% higher)")
        
        # Show distribution
        print(f"\n📊 Distribution percentiles:")
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        print(f"\n   Training:")
        for p in percentiles:
            val = df_train['cum_inj_starts'].quantile(p/100)
            print(f"      {p}th percentile: {val:.2f}")
        
        print(f"\n   Validation:")
        for p in percentiles:
            val = df_val['cum_inj_starts'].quantile(p/100)
            print(f"      {p}th percentile: {val:.2f}")
        
        # Find players that appear in both sets
        print("\n" + "="*80)
        print("EXAMPLE: SAME PLAYER IN TRAINING vs VALIDATION")
        print("="*80)
        
        train_players = set(df_train['player_id'].unique())
        val_players = set(df_val['player_id'].unique())
        common_players = train_players & val_players
        
        print(f"\n👥 Players in both sets: {len(common_players)}")
        
        if len(common_players) > 0:
            # Pick a few example players
            example_players = list(common_players)[:5]
            
            for player_id in example_players:
                player_train = df_train[df_train['player_id'] == player_id].sort_values('reference_date')
                player_val = df_val[df_val['player_id'] == player_id].sort_values('reference_date')
                
                if len(player_train) > 0 and len(player_val) > 0:
                    # Get earliest and latest records
                    train_earliest = player_train.iloc[0]
                    train_latest = player_train.iloc[-1]
                    val_earliest = player_val.iloc[0]
                    val_latest = player_val.iloc[-1]
                    
                    player_name = train_earliest.get('player_name', f'Player_{player_id}')
                    
                    print(f"\n   Player {player_id} ({player_name}):")
                    print(f"      Training period:")
                    print(f"         Earliest: {train_earliest['reference_date'].strftime('%Y-%m-%d')} - cum_inj_starts = {train_earliest['cum_inj_starts']:.1f}")
                    print(f"         Latest:   {train_latest['reference_date'].strftime('%Y-%m-%d')} - cum_inj_starts = {train_latest['cum_inj_starts']:.1f}")
                    if train_latest['cum_inj_starts'] > train_earliest['cum_inj_starts']:
                        print(f"         Increase: {train_latest['cum_inj_starts'] - train_earliest['cum_inj_starts']:.1f}")
                    print(f"      Validation period:")
                    print(f"         Earliest: {val_earliest['reference_date'].strftime('%Y-%m-%d')} - cum_inj_starts = {val_earliest['cum_inj_starts']:.1f}")
                    print(f"         Latest:   {val_latest['reference_date'].strftime('%Y-%m-%d')} - cum_inj_starts = {val_latest['cum_inj_starts']:.1f}")
                    if val_latest['cum_inj_starts'] > val_earliest['cum_inj_starts']:
                        print(f"         Increase: {val_latest['cum_inj_starts'] - val_earliest['cum_inj_starts']:.1f}")
                    print(f"      Jump from training end to validation start:")
                    print(f"         {train_latest['reference_date'].strftime('%Y-%m-%d')} → {val_earliest['reference_date'].strftime('%Y-%m-%d')}")
                    print(f"         cum_inj_starts: {train_latest['cum_inj_starts']:.1f} → {val_earliest['cum_inj_starts']:.1f}")
                    jump = val_earliest['cum_inj_starts'] - train_latest['cum_inj_starts']
                    if jump > 0:
                        print(f"         Increase: {jump:.1f} ({jump/train_latest['cum_inj_starts']*100:.1f}% higher)")
        
        # Show players with highest cum_inj_starts in each period
        print("\n" + "="*80)
        print("TOP 10 PLAYERS BY cum_inj_starts")
        print("="*80)
        
        print("\n   Training period (top 10):")
        top_train = df_train.nlargest(10, 'cum_inj_starts')[['player_id', 'player_name', 'reference_date', 'cum_inj_starts']]
        for idx, row in top_train.iterrows():
            player_name = row.get('player_name', f'Player_{row["player_id"]}')
            print(f"      {player_name} (ID: {row['player_id']}) - {row['reference_date'].strftime('%Y-%m-%d')}: {row['cum_inj_starts']:.1f}")
        
        print("\n   Validation period (top 10):")
        top_val = df_val.nlargest(10, 'cum_inj_starts')[['player_id', 'player_name', 'reference_date', 'cum_inj_starts']]
        for idx, row in top_val.iterrows():
            player_name = row.get('player_name', f'Player_{row["player_id"]}')
            print(f"      {player_name} (ID: {row['player_id']}) - {row['reference_date'].strftime('%Y-%m-%d')}: {row['cum_inj_starts']:.1f}")
        
        # Analyze by date ranges
        print("\n" + "="*80)
        print("CUMULATIVE FEATURE GROWTH OVER TIME")
        print("="*80)
        
        # Group training data by year
        df_train['year'] = df_train['reference_date'].dt.year
        df_val['year'] = df_val['reference_date'].dt.year
        
        print("\n   Training period (by year - showing last 10 years):")
        train_by_year = df_train.groupby('year')['cum_inj_starts'].agg(['mean', 'median', 'count'])
        for year, row in train_by_year.tail(10).iterrows():
            print(f"      {year}: mean={row['mean']:.2f}, median={row['median']:.2f}, count={row['count']:.0f}")
        
        print("\n   Validation period (by year):")
        val_by_year = df_val.groupby('year')['cum_inj_starts'].agg(['mean', 'median', 'count'])
        for year, row in val_by_year.iterrows():
            print(f"      {year}: mean={row['mean']:.2f}, median={row['median']:.2f}, count={row['count']:.0f}")
        
        # Show the trend
        print("\n   📈 Trend analysis:")
        print(f"      Training period (2001-2024):")
        train_early = df_train[df_train['year'] <= 2010]['cum_inj_starts'].mean()
        train_mid = df_train[(df_train['year'] > 2010) & (df_train['year'] <= 2020)]['cum_inj_starts'].mean()
        train_late = df_train[df_train['year'] > 2020]['cum_inj_starts'].mean()
        print(f"         2001-2010: {train_early:.2f}")
        print(f"         2011-2020: {train_mid:.2f}")
        print(f"         2021-2024: {train_late:.2f}")
        if train_early > 0:
            print(f"         Growth: {train_early:.2f} → {train_late:.2f} ({((train_late - train_early) / train_early * 100):.1f}% increase)")
        
        if len(val_by_year) > 0:
            val_mean = df_val['cum_inj_starts'].mean()
            print(f"      Validation period (2024-2025): {val_mean:.2f}")
            if train_late > 0:
                print(f"         Jump from training end: {train_late:.2f} → {val_mean:.2f} ({((val_mean - train_late) / train_late * 100):.1f}% increase)")
    
    # Analyze other cumulative features
    print("\n" + "="*80)
    print("OTHER CUMULATIVE FEATURES")
    print("="*80)
    
    cumulative_features = [col for col in df_train.columns if col.startswith('cum_')]
    
    print(f"\n📊 Found {len(cumulative_features)} cumulative features")
    print("\n   Comparison (Training vs Validation):")
    print(f"\n   {'Feature':<40} {'Train Mean':<12} {'Val Mean':<12} {'Diff %':<10}")
    print("   " + "-"*74)
    
    for feat in sorted(cumulative_features)[:20]:  # Show first 20
        if feat in df_train.columns and feat in df_val.columns:
            train_mean = df_train[feat].mean()
            val_mean = df_val[feat].mean()
            if train_mean > 0:
                diff_pct = ((val_mean - train_mean) / train_mean) * 100
                print(f"   {feat:<40} {train_mean:<12.2f} {val_mean:<12.2f} {diff_pct:>9.1f}%")
    
    # Explanation
    print("\n" + "="*80)
    print("EXPLANATION")
    print("="*80)
    
    print("""
Why are cumulative features higher in the validation period?

1. **Cumulative features accumulate over time:**
   - `cum_inj_starts` counts all injury starts up to the reference date
   - Players in 2024-2025 have had MORE TIME to accumulate injuries
   - A player who started in 2010 will have more cumulative injuries in 2024 than in 2015

2. **Temporal split creates a natural bias:**
   - Training: 2001-2024-06-30 (includes early career players with low cumulative counts)
   - Validation: 2024-07-01-2025-06-30 (includes players who have been playing longer)
   
3. **Example scenario:**
   - Player A starts career in 2015
   - In 2020 (training period): cum_inj_starts = 3 (5 years of career)
   - In 2024 (validation period): cum_inj_starts = 7 (9 years of career)
   - The model trained on 2020 data sees cum_inj_starts=3 as normal
   - But in validation, the same player has cum_inj_starts=7
   - Model thinks this is unusual (high) and may misclassify

4. **This is NOT a data quality issue:**
   - The feature is correctly calculated
   - The problem is that cumulative features are inherently time-dependent
   - They create distribution shift when splitting by time

5. **Solution approaches:**
   - Normalize by career length (cum_inj_starts / years_active)
   - Use rates instead of cumulative (injuries_per_year)
   - Use rolling windows instead of cumulative sums
   - Normalize features by time period
    """)

if __name__ == "__main__":
    main()

