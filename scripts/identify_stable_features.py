#!/usr/bin/env python3
"""
Identify stable features (low drift) for Phase 1 implementation
Features with drift < 0.10 are considered stable
"""
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import json
from pathlib import Path

def main():
    print("="*80)
    print("IDENTIFYING STABLE FEATURES (DRIFT < 0.10)")
    print("="*80)
    
    # Load drift data
    print("\n📂 Loading feature drift data...")
    drift_file = 'experiments/feature_correlation_comparison.csv'
    df_drift = pd.read_csv(drift_file)
    
    print(f"✅ Loaded {len(df_drift):,} features")
    
    # Filter stable features (drift < 0.10)
    DRIFT_THRESHOLD = 0.10
    stable_features = df_drift[df_drift['train_val_diff'].abs() < DRIFT_THRESHOLD].copy()
    
    print(f"\n📊 Feature Stability Analysis:")
    print(f"   Total features: {len(df_drift)}")
    print(f"   Stable features (drift < {DRIFT_THRESHOLD}): {len(stable_features)}")
    print(f"   High-drift features (drift >= {DRIFT_THRESHOLD}): {len(df_drift) - len(stable_features)}")
    
    # Show some statistics
    print(f"\n📈 Drift Statistics:")
    print(f"   Mean drift (all features): {df_drift['train_val_diff'].abs().mean():.4f}")
    print(f"   Mean drift (stable features): {stable_features['train_val_diff'].abs().mean():.4f}")
    print(f"   Max drift (stable features): {stable_features['train_val_diff'].abs().max():.4f}")
    
    # Get feature list
    stable_feature_list = stable_features['feature'].tolist()
    
    # Sort by drift (ascending) to show most stable
    stable_features_sorted = stable_features.sort_values('train_val_diff', key=lambda x: x.abs())
    
    print(f"\n📋 Top 20 Most Stable Features:")
    print(f"\n{'Feature':<50} {'Train Corr':<12} {'Val Corr':<12} {'Drift':<12}")
    print("-" * 90)
    for idx, row in stable_features_sorted.head(20).iterrows():
        print(f"{row['feature']:<50} {row['train_corr']:<12.4f} {row['val_corr']:<12.4f} {row['train_val_diff']:<12.4f}")
    
    # Show high-drift features that will be removed
    high_drift = df_drift[df_drift['train_val_diff'].abs() >= DRIFT_THRESHOLD].sort_values('train_val_diff', key=lambda x: x.abs(), ascending=False)
    
    print(f"\n📋 Top 20 Highest-Drift Features (Will Be Removed):")
    print(f"\n{'Feature':<50} {'Train Corr':<12} {'Val Corr':<12} {'Drift':<12}")
    print("-" * 90)
    for idx, row in high_drift.head(20).iterrows():
        print(f"{row['feature']:<50} {row['train_corr']:<12.4f} {row['val_corr']:<12.4f} {row['train_val_diff']:<12.4f}")
    
    # Save stable features list
    output_dir = Path('experiments')
    output_dir.mkdir(exist_ok=True)
    
    # Save as JSON
    output_file = output_dir / 'stable_features_phase1.json'
    with open(output_file, 'w') as f:
        json.dump(stable_feature_list, f, indent=2)
    
    # Save as CSV for reference
    csv_file = output_dir / 'stable_features_phase1.csv'
    stable_features_sorted.to_csv(csv_file, index=False)
    
    print(f"\n✅ Stable features saved to {output_file}")
    print(f"✅ Detailed list saved to {csv_file}")
    print(f"\n📊 Summary:")
    print(f"   Stable features: {len(stable_feature_list)}")
    print(f"   Removed features: {len(df_drift) - len(stable_features)}")
    print(f"   Reduction: {((len(df_drift) - len(stable_features)) / len(df_drift) * 100):.1f}%")
    
    return stable_feature_list

if __name__ == "__main__":
    main()

