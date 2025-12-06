#!/usr/bin/env python3
"""
Comprehensive analysis comparing 80/20 random split vs temporal split
to understand why F1-score drops so dramatically
"""
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def prepare_data(df, drop_week5=False):
    """Prepare data with same logic as training scripts"""
    feature_columns = [
        col for col in df.columns
        if col not in ['player_id', 'reference_date', 'player_name', 'target']
        and (drop_week5 is False or '_week_5' not in col)
    ]
    X = df[feature_columns].copy()
    y = df['target'].copy()
    
    # Handle missing values and encode categorical features
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    X_encoded = X.copy()
    
    # Encode categorical features
    for feature in categorical_features:
        X_encoded[feature] = X_encoded[feature].fillna('Unknown')
        dummies = pd.get_dummies(X_encoded[feature], prefix=feature, drop_first=True)
        X_encoded = pd.concat([X_encoded, dummies], axis=1)
        X_encoded = X_encoded.drop(columns=[feature])
    
    # Fill numeric missing values
    for feature in numeric_features:
        X_encoded[feature] = X_encoded[feature].fillna(0)
    
    return X_encoded, y

def apply_correlation_filter(X, threshold):
    """Drop one feature from each highly correlated pair."""
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    kept = [col for col in X.columns if col not in to_drop]
    return kept

def analyze_distribution_drift(X_train, X_val, feature_name, set1_name, set2_name):
    """Analyze distribution drift for a single feature"""
    train_vals = X_train[feature_name].values
    val_vals = X_val[feature_name].values
    
    # Remove NaN/inf
    train_vals = train_vals[np.isfinite(train_vals)]
    val_vals = val_vals[np.isfinite(val_vals)]
    
    if len(train_vals) == 0 or len(val_vals) == 0:
        return None
    
    # Statistical tests
    try:
        ks_stat, ks_p = stats.ks_2samp(train_vals, val_vals)
    except:
        ks_stat, ks_p = np.nan, np.nan
    
    try:
        mann_stat, mann_p = stats.mannwhitneyu(train_vals, val_vals, alternative='two-sided')
    except:
        mann_stat, mann_p = np.nan, np.nan
    
    return {
        'feature': feature_name,
        'train_mean': np.mean(train_vals),
        'val_mean': np.mean(val_vals),
        'train_std': np.std(train_vals),
        'val_std': np.std(val_vals),
        'mean_diff': np.mean(train_vals) - np.mean(val_vals),
        'mean_diff_pct': (np.mean(train_vals) - np.mean(val_vals)) / (np.abs(np.mean(train_vals)) + 1e-10) * 100,
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_p,
        'mann_statistic': mann_stat,
        'mann_pvalue': mann_p,
        'drift_severity': 'high' if ks_p < 0.01 and abs(ks_stat) > 0.1 else ('medium' if ks_p < 0.05 else 'low')
    }

def main():
    print("="*80)
    print("COMPREHENSIVE SPLIT ANALYSIS: 80/20 RANDOM vs TEMPORAL")
    print("="*80)
    
    # Load data
    print("\n📂 Loading data...")
    df_full = pd.read_csv('timelines_35day_enhanced_balanced_v4_train.csv', encoding='utf-8-sig')
    df_full['reference_date'] = pd.to_datetime(df_full['reference_date'])
    print(f"✅ Loaded {len(df_full):,} total records")
    
    # Prepare data
    print("\n🔧 Preparing data...")
    X_full, y_full = prepare_data(df_full, drop_week5=False)
    
    # Apply correlation filter
    CORR_THRESHOLD = 0.8
    print(f"\n🔎 Applying correlation filter (threshold={CORR_THRESHOLD})...")
    selected_columns = apply_correlation_filter(X_full, CORR_THRESHOLD)
    X_full = X_full[selected_columns]
    print(f"✅ Selected {len(selected_columns)} features")
    
    # ===== SPLIT 1: 80/20 RANDOM SPLIT =====
    print("\n" + "="*80)
    print("SPLIT 1: 80/20 RANDOM SPLIT (STRATIFIED)")
    print("="*80)
    
    X_train_random, X_val_random, y_train_random, y_val_random = train_test_split(
        X_full, y_full,
        test_size=0.2,
        random_state=42,
        stratify=y_full
    )
    
    print(f"Training: {len(X_train_random):,} samples, {y_train_random.mean():.1%} injury ratio")
    print(f"Validation: {len(X_val_random):,} samples, {y_val_random.mean():.1%} injury ratio")
    
    # ===== SPLIT 2: TEMPORAL SPLIT (2024-06-30) =====
    print("\n" + "="*80)
    print("SPLIT 2: TEMPORAL SPLIT (<= 2024-06-30 vs > 2024-06-30)")
    print("="*80)
    
    TRAIN_CUTOFF = pd.Timestamp('2024-06-30')
    VAL_END = pd.Timestamp('2025-06-30')
    
    train_mask = df_full['reference_date'] <= TRAIN_CUTOFF
    val_mask = (df_full['reference_date'] > TRAIN_CUTOFF) & (df_full['reference_date'] <= VAL_END)
    
    X_train_temporal = X_full[train_mask].copy()
    X_val_temporal = X_full[val_mask].copy()
    y_train_temporal = y_full[train_mask].copy()
    y_val_temporal = y_full[val_mask].copy()
    
    print(f"Training: {len(X_train_temporal):,} samples, {y_train_temporal.mean():.1%} injury ratio")
    print(f"Validation: {len(X_val_temporal):,} samples, {y_val_temporal.mean():.1%} injury ratio")
    
    # Date range analysis
    print(f"\n📅 Date ranges:")
    print(f"   Training: {df_full[train_mask]['reference_date'].min()} to {df_full[train_mask]['reference_date'].max()}")
    print(f"   Validation: {df_full[val_mask]['reference_date'].min()} to {df_full[val_mask]['reference_date'].max()}")
    
    # Player overlap analysis
    print(f"\n👥 Player overlap analysis:")
    players_train_random = set(df_full.iloc[X_train_random.index]['player_id'].unique())
    players_val_random = set(df_full.iloc[X_val_random.index]['player_id'].unique())
    players_train_temporal = set(df_full[train_mask]['player_id'].unique())
    players_val_temporal = set(df_full[val_mask]['player_id'].unique())
    
    print(f"   80/20 Split:")
    print(f"      Training players: {len(players_train_random):,}")
    print(f"      Validation players: {len(players_val_random):,}")
    print(f"      Overlap: {len(players_train_random & players_val_random):,} ({len(players_train_random & players_val_random)/len(players_val_random)*100:.1f}% of val)")
    
    print(f"   Temporal Split:")
    print(f"      Training players: {len(players_train_temporal):,}")
    print(f"      Validation players: {len(players_val_temporal):,}")
    print(f"      Overlap: {len(players_train_temporal & players_val_temporal):,} ({len(players_train_temporal & players_val_temporal)/len(players_val_temporal)*100:.1f}% of val)")
    
    # ===== FEATURE DISTRIBUTION ANALYSIS =====
    print("\n" + "="*80)
    print("FEATURE DISTRIBUTION DRIFT ANALYSIS")
    print("="*80)
    
    # Analyze numeric features only
    numeric_features = X_full.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\n📊 Analyzing {len(numeric_features)} numeric features...")
    
    # Random split drift
    print("\n🔍 Analyzing 80/20 random split drift...")
    random_drift_results = []
    for i, feat in enumerate(numeric_features[:100]):  # Sample first 100 for speed
        if i % 20 == 0:
            print(f"   Processing feature {i+1}/{min(100, len(numeric_features))}...")
        result = analyze_distribution_drift(
            X_train_random, X_val_random, feat,
            "Train (Random)", "Val (Random)"
        )
        if result:
            result['split_type'] = 'random'
            random_drift_results.append(result)
    
    # Temporal split drift
    print("\n🔍 Analyzing temporal split drift...")
    temporal_drift_results = []
    for i, feat in enumerate(numeric_features[:100]):  # Sample first 100 for speed
        if i % 20 == 0:
            print(f"   Processing feature {i+1}/{min(100, len(numeric_features))}...")
        result = analyze_distribution_drift(
            X_train_temporal, X_val_temporal, feat,
            "Train (Temporal)", "Val (Temporal)"
        )
        if result:
            result['split_type'] = 'temporal'
            temporal_drift_results.append(result)
    
    # Compare drift severity
    random_drift_df = pd.DataFrame(random_drift_results)
    temporal_drift_df = pd.DataFrame(temporal_drift_results)
    
    print("\n" + "="*80)
    print("DRIFT SEVERITY COMPARISON")
    print("="*80)
    
    if len(random_drift_df) > 0:
        print(f"\n80/20 Random Split:")
        print(f"   Low drift: {(random_drift_df['drift_severity']=='low').sum()} features")
        print(f"   Medium drift: {(random_drift_df['drift_severity']=='medium').sum()} features")
        print(f"   High drift: {(random_drift_df['drift_severity']=='high').sum()} features")
        print(f"   Mean KS statistic: {random_drift_df['ks_statistic'].abs().mean():.4f}")
        print(f"   Mean mean_diff_pct: {random_drift_df['mean_diff_pct'].abs().mean():.2f}%")
    
    if len(temporal_drift_df) > 0:
        print(f"\nTemporal Split:")
        print(f"   Low drift: {(temporal_drift_df['drift_severity']=='low').sum()} features")
        print(f"   Medium drift: {(temporal_drift_df['drift_severity']=='medium').sum()} features")
        print(f"   High drift: {(temporal_drift_df['drift_severity']=='high').sum()} features")
        print(f"   Mean KS statistic: {temporal_drift_df['ks_statistic'].abs().mean():.4f}")
        print(f"   Mean mean_diff_pct: {temporal_drift_df['mean_diff_pct'].abs().mean():.2f}%")
    
    # Top drifting features (temporal)
    if len(temporal_drift_df) > 0:
        print("\n" + "="*80)
        print("TOP 20 MOST DRIFTED FEATURES (Temporal Split)")
        print("="*80)
        top_drifted = temporal_drift_df.nlargest(20, 'ks_statistic')
        print("\n| Feature | Train Mean | Val Mean | Mean Diff % | KS Stat | KS p-value | Drift |")
        print("|---------|------------|----------|-------------|---------|------------|-------|")
        for _, row in top_drifted.iterrows():
            print(f"| {row['feature'][:40]:<40} | {row['train_mean']:.4f} | {row['val_mean']:.4f} | "
                  f"{row['mean_diff_pct']:.2f}% | {row['ks_statistic']:.4f} | {row['ks_pvalue']:.4e} | {row['drift_severity']} |")
    
    # ===== INJURY PATTERN ANALYSIS =====
    print("\n" + "="*80)
    print("INJURY PATTERN ANALYSIS")
    print("="*80)
    
    # Date distribution of injuries
    print("\n📅 Injury date distribution:")
    injuries_random_train = df_full.iloc[X_train_random.index][df_full.iloc[X_train_random.index]['target']==1]
    injuries_random_val = df_full.iloc[X_val_random.index][df_full.iloc[X_val_random.index]['target']==1]
    injuries_temporal_train = df_full[train_mask][df_full[train_mask]['target']==1]
    injuries_temporal_val = df_full[val_mask][df_full[val_mask]['target']==1]
    
    print(f"\n   80/20 Random Split:")
    print(f"      Training injuries: {len(injuries_random_train):,}")
    if len(injuries_random_train) > 0:
        print(f"         Date range: {injuries_random_train['reference_date'].min()} to {injuries_random_train['reference_date'].max()}")
    print(f"      Validation injuries: {len(injuries_random_val):,}")
    if len(injuries_random_val) > 0:
        print(f"         Date range: {injuries_random_val['reference_date'].min()} to {injuries_random_val['reference_date'].max()}")
    
    print(f"\n   Temporal Split:")
    print(f"      Training injuries: {len(injuries_temporal_train):,}")
    if len(injuries_temporal_train) > 0:
        print(f"         Date range: {injuries_temporal_train['reference_date'].min()} to {injuries_temporal_train['reference_date'].max()}")
    print(f"      Validation injuries: {len(injuries_temporal_val):,}")
    if len(injuries_temporal_val) > 0:
        print(f"         Date range: {injuries_temporal_val['reference_date'].min()} to {injuries_temporal_val['reference_date'].max()}")
    
    # ===== SAVE RESULTS =====
    output_dir = Path('experiments')
    output_dir.mkdir(exist_ok=True)
    
    # Save drift analysis
    if len(random_drift_df) > 0:
        random_drift_df.to_csv(output_dir / 'random_split_drift_analysis.csv', index=False)
    if len(temporal_drift_df) > 0:
        temporal_drift_df.to_csv(output_dir / 'temporal_split_drift_analysis.csv', index=False)
    
    # Create summary report
    summary_file = output_dir / 'split_analysis_summary.md'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("# Split Analysis: 80/20 Random vs Temporal\n\n")
        f.write("## Key Findings\n\n")
        f.write("### Dataset Characteristics\n\n")
        f.write("| Split Type | Training Size | Training Ratio | Validation Size | Validation Ratio |\n")
        f.write("|------------|---------------|----------------|-----------------|------------------|\n")
        f.write(f"| 80/20 Random | {len(X_train_random):,} | {y_train_random.mean():.1%} | {len(X_val_random):,} | {y_val_random.mean():.1%} |\n")
        f.write(f"| Temporal | {len(X_train_temporal):,} | {y_train_temporal.mean():.1%} | {len(X_val_temporal):,} | {y_val_temporal.mean():.1%} |\n")
        f.write("\n### Player Overlap\n\n")
        f.write(f"- **80/20 Random:** {len(players_train_random & players_val_random):,} overlapping players ({len(players_train_random & players_val_random)/len(players_val_random)*100:.1f}% of validation)\n")
        f.write(f"- **Temporal:** {len(players_train_temporal & players_val_temporal):,} overlapping players ({len(players_train_temporal & players_val_temporal)/len(players_val_temporal)*100:.1f}% of validation)\n")
        f.write("\n### Feature Drift Summary\n\n")
        if len(random_drift_df) > 0:
            f.write("**80/20 Random Split:**\n")
            f.write(f"- Low drift: {(random_drift_df['drift_severity']=='low').sum()} features\n")
            f.write(f"- Medium drift: {(random_drift_df['drift_severity']=='medium').sum()} features\n")
            f.write(f"- High drift: {(random_drift_df['drift_severity']=='high').sum()} features\n")
            f.write(f"- Mean KS statistic: {random_drift_df['ks_statistic'].abs().mean():.4f}\n\n")
        if len(temporal_drift_df) > 0:
            f.write("**Temporal Split:**\n")
            f.write(f"- Low drift: {(temporal_drift_df['drift_severity']=='low').sum()} features\n")
            f.write(f"- Medium drift: {(temporal_drift_df['drift_severity']=='medium').sum()} features\n")
            f.write(f"- High drift: {(temporal_drift_df['drift_severity']=='high').sum()} features\n")
            f.write(f"- Mean KS statistic: {temporal_drift_df['ks_statistic'].abs().mean():.4f}\n\n")
        f.write("## Explanation\n\n")
        f.write("The dramatic F1-score drop in temporal split vs 80/20 random split is likely due to:\n\n")
        f.write("1. **Temporal drift:** Features have different distributions in different time periods\n")
        f.write("2. **Player composition:** Different players in training vs validation periods\n")
        f.write("3. **Injury patterns:** Injury characteristics may change over time\n")
        f.write("4. **Data leakage:** Random split allows models to see patterns from all time periods during training\n")
    
    print(f"\n✅ Analysis complete!")
    print(f"✅ Results saved to {summary_file}")
    print(f"✅ Drift analysis saved to experiments/")

if __name__ == "__main__":
    main()



