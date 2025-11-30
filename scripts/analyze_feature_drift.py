#!/usr/bin/env python3
"""
Comprehensive Feature Distribution Drift Analysis
Compares feature distributions between training and out-of-sample validation sets
"""

import sys
import io
# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency
from pathlib import Path
import json

def prepare_data(df):
    """Prepare data with same encoding logic as training script"""
    feature_columns = [col for col in df.columns if col not in ['player_id', 'reference_date', 'player_name', 'target']]
    X = df[feature_columns]
    y = df['target']
    
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
    
    return X_encoded, y, feature_columns

def calculate_numeric_drift(train_series, val_series, feature_name):
    """Calculate drift metrics for numeric features"""
    # Remove any remaining NaN or inf values
    train_clean = train_series.replace([np.inf, -np.inf], np.nan).dropna()
    val_clean = val_series.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(train_clean) == 0 or len(val_clean) == 0:
        return {
            'feature': feature_name,
            'type': 'numeric',
            'drift_detected': False,
            'reason': 'Insufficient data'
        }
    
    # Statistical tests
    try:
        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = ks_2samp(train_clean, val_clean)
        
        # Mean difference
        mean_diff = val_clean.mean() - train_clean.mean()
        mean_diff_pct = (mean_diff / train_clean.mean() * 100) if train_clean.mean() != 0 else 0
        
        # Standard deviation ratio
        std_ratio = val_clean.std() / train_clean.std() if train_clean.std() != 0 else 1.0
        
        # Coefficient of variation difference
        cv_train = train_clean.std() / train_clean.mean() if train_clean.mean() != 0 else 0
        cv_val = val_clean.std() / val_clean.mean() if val_clean.mean() != 0 else 0
        cv_diff = abs(cv_val - cv_train)
        
        # Percentile shifts (25th, 50th, 75th)
        percentiles = [25, 50, 75]
        percentile_shifts = {}
        for p in percentiles:
            train_p = np.percentile(train_clean, p)
            val_p = np.percentile(val_clean, p)
            if train_p != 0:
                percentile_shifts[f'p{p}'] = (val_p - train_p) / train_p * 100
            else:
                percentile_shifts[f'p{p}'] = 0
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((train_clean.std()**2 + val_clean.std()**2) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std != 0 else 0
        
        # Drift severity classification
        drift_severity = 'none'
        if ks_pvalue < 0.001:
            if abs(cohens_d) > 0.8:
                drift_severity = 'severe'
            elif abs(cohens_d) > 0.5:
                drift_severity = 'moderate'
            elif abs(cohens_d) > 0.2:
                drift_severity = 'mild'
            else:
                drift_severity = 'minimal'
        
        return {
            'feature': feature_name,
            'type': 'numeric',
            'drift_detected': ks_pvalue < 0.05,
            'drift_severity': drift_severity,
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pvalue),
            'mean_train': float(train_clean.mean()),
            'mean_val': float(val_clean.mean()),
            'mean_diff': float(mean_diff),
            'mean_diff_pct': float(mean_diff_pct),
            'std_train': float(train_clean.std()),
            'std_val': float(val_clean.std()),
            'std_ratio': float(std_ratio),
            'cv_diff': float(cv_diff),
            'cohens_d': float(cohens_d),
            'percentile_shifts': percentile_shifts,
            'zero_count_train': int((train_clean == 0).sum()),
            'zero_count_val': int((val_clean == 0).sum()),
            'zero_pct_train': float((train_clean == 0).sum() / len(train_clean) * 100),
            'zero_pct_val': float((val_clean == 0).sum() / len(val_clean) * 100),
        }
    except Exception as e:
        return {
            'feature': feature_name,
            'type': 'numeric',
            'drift_detected': False,
            'error': str(e)
        }

def calculate_categorical_drift(train_series, val_series, feature_name):
    """Calculate drift metrics for categorical (binary) features"""
    # For one-hot encoded features, they're binary (0/1)
    train_clean = train_series.fillna(0)
    val_clean = val_series.fillna(0)
    
    # Calculate frequency differences
    train_freq = train_clean.mean()
    val_freq = val_clean.mean()
    freq_diff = val_freq - train_freq
    freq_diff_pct = (freq_diff / train_freq * 100) if train_freq != 0 else 0
    
    # Chi-square test for independence
    try:
        contingency = pd.crosstab(
            pd.concat([train_clean, val_clean]),
            pd.concat([pd.Series(['train'] * len(train_clean)), pd.Series(['val'] * len(val_clean))])
        )
        
        if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
            chi2, pvalue, dof, expected = chi2_contingency(contingency)
        else:
            chi2, pvalue = 0, 1.0
        
        # Drift severity
        drift_severity = 'none'
        if pvalue < 0.001:
            if abs(freq_diff_pct) > 50:
                drift_severity = 'severe'
            elif abs(freq_diff_pct) > 25:
                drift_severity = 'moderate'
            elif abs(freq_diff_pct) > 10:
                drift_severity = 'mild'
            else:
                drift_severity = 'minimal'
        
        return {
            'feature': feature_name,
            'type': 'categorical',
            'drift_detected': pvalue < 0.05,
            'drift_severity': drift_severity,
            'chi2_statistic': float(chi2) if 'chi2' in locals() else 0.0,
            'chi2_pvalue': float(pvalue) if 'pvalue' in locals() else 1.0,
            'freq_train': float(train_freq),
            'freq_val': float(val_freq),
            'freq_diff': float(freq_diff),
            'freq_diff_pct': float(freq_diff_pct),
            'presence_train': int((train_clean > 0).sum()),
            'presence_val': int((val_clean > 0).sum()),
            'presence_pct_train': float((train_clean > 0).sum() / len(train_clean) * 100),
            'presence_pct_val': float((val_clean > 0).sum() / len(val_clean) * 100),
        }
    except Exception as e:
        return {
            'feature': feature_name,
            'type': 'categorical',
            'drift_detected': False,
            'error': str(e)
        }

def analyze_feature_drift():
    """Main function to analyze feature drift"""
    print("=" * 80)
    print("🔍 COMPREHENSIVE FEATURE DISTRIBUTION DRIFT ANALYSIS")
    print("=" * 80)
    
    # Load datasets
    print("\n📂 Loading datasets...")
    train_file = 'timelines_35day_enhanced_balanced_v4_train.csv'
    val_file = 'timelines_35day_enhanced_balanced_v4_val.csv'
    
    if not os.path.exists(train_file):
        train_file = f'scripts/{train_file}'
    if not os.path.exists(val_file):
        val_file = f'scripts/{val_file}'
    
    if not os.path.exists(train_file) or not os.path.exists(val_file):
        print(f"❌ Error: Could not find timeline files")
        print(f"   Looking for: {train_file} and {val_file}")
        return
    
    df_train = pd.read_csv(train_file, encoding='utf-8-sig')
    df_val = pd.read_csv(val_file, encoding='utf-8-sig')
    
    print(f"✅ Training set: {len(df_train):,} records")
    print(f"✅ Validation set: {len(df_val):,} records")
    
    # Prepare data (same encoding as training)
    print("\n🔧 Preparing and encoding features...")
    X_train, y_train, _ = prepare_data(df_train)
    X_val, y_val, _ = prepare_data(df_val)
    
    # Align columns (same as training script)
    all_cols = sorted(set(X_train.columns) | set(X_val.columns))
    X_train = X_train.reindex(columns=all_cols, fill_value=0)
    X_val = X_val.reindex(columns=all_cols, fill_value=0)
    
    print(f"📊 Total features after encoding: {len(all_cols)}")
    print(f"   Training features: {X_train.shape[1]}")
    print(f"   Validation features: {X_val.shape[1]}")
    
    # Analyze each feature
    print("\n" + "=" * 80)
    print("🔍 ANALYZING FEATURE DRIFT")
    print("=" * 80)
    
    drift_results = []
    
    # Identify numeric vs categorical features
    # Categorical features are typically binary (0/1) after one-hot encoding
    # Numeric features have continuous or integer values
    
    for feature in all_cols:
        train_series = X_train[feature]
        val_series = X_val[feature]
        
        # Determine if feature is numeric or categorical
        # If feature has only 0/1 values, treat as categorical (one-hot encoded)
        unique_train = train_series.unique()
        unique_val = val_series.unique()
        all_unique = set(unique_train) | set(unique_val)
        
        if len(all_unique) <= 2 and all_unique.issubset({0, 1, 0.0, 1.0}):
            # Binary/categorical feature
            result = calculate_categorical_drift(train_series, val_series, feature)
        else:
            # Numeric feature
            result = calculate_numeric_drift(train_series, val_series, feature)
        
        drift_results.append(result)
        
        if len(drift_results) % 50 == 0:
            print(f"   Processed {len(drift_results)}/{len(all_cols)} features...")
    
    # Convert to DataFrame for analysis
    drift_df = pd.DataFrame(drift_results)
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("📊 DRIFT ANALYSIS SUMMARY")
    print("=" * 80)
    
    total_features = len(drift_df)
    drift_detected = drift_df['drift_detected'].sum()
    drift_pct = (drift_detected / total_features * 100) if total_features > 0 else 0
    
    print(f"\n📈 Overall Statistics:")
    print(f"   Total features analyzed: {total_features}")
    print(f"   Features with detected drift: {drift_detected} ({drift_pct:.1f}%)")
    print(f"   Features without drift: {total_features - drift_detected} ({100-drift_pct:.1f}%)")
    
    # Drift severity breakdown
    print(f"\n📊 Drift Severity Breakdown:")
    severity_counts = drift_df['drift_severity'].value_counts()
    for severity, count in severity_counts.items():
        pct = (count / total_features * 100) if total_features > 0 else 0
        print(f"   {severity.capitalize()}: {count} ({pct:.1f}%)")
    
    # Feature type breakdown
    print(f"\n📊 Feature Type Breakdown:")
    type_counts = drift_df['type'].value_counts()
    for ftype, count in type_counts.items():
        pct = (count / total_features * 100) if total_features > 0 else 0
        drift_in_type = drift_df[drift_df['type'] == ftype]['drift_detected'].sum()
        drift_pct_type = (drift_in_type / count * 100) if count > 0 else 0
        print(f"   {ftype.capitalize()}: {count} ({pct:.1f}%) - {drift_in_type} with drift ({drift_pct_type:.1f}%)")
    
    # Top drifted features
    print("\n" + "=" * 80)
    print("🔴 TOP 20 MOST DRIFTED FEATURES (Numeric)")
    print("=" * 80)
    
    numeric_drifted = drift_df[
        (drift_df['type'] == 'numeric') & 
        (drift_df['drift_detected'] == True)
    ].copy()
    
    if len(numeric_drifted) > 0:
        # Sort by absolute Cohen's d (effect size)
        numeric_drifted['abs_cohens_d'] = numeric_drifted['cohens_d'].abs()
        numeric_drifted = numeric_drifted.sort_values('abs_cohens_d', ascending=False)
        
        cohens_d_label = "Cohen's d"
        print(f"\n{'Feature':<50} {'Severity':<12} {'KS p-value':<12} {cohens_d_label:<12} {'Mean Diff %':<15}")
        print("-" * 100)
        for idx, row in numeric_drifted.head(20).iterrows():
            print(f"{row['feature'][:49]:<50} {row['drift_severity']:<12} {row['ks_pvalue']:<12.4f} {row['cohens_d']:<12.3f} {row['mean_diff_pct']:<15.1f}%")
    
    print("\n" + "=" * 80)
    print("🔴 TOP 20 MOST DRIFTED FEATURES (Categorical)")
    print("=" * 80)
    
    categorical_drifted = drift_df[
        (drift_df['type'] == 'categorical') & 
        (drift_df['drift_detected'] == True)
    ].copy()
    
    if len(categorical_drifted) > 0:
        # Sort by absolute frequency difference percentage
        categorical_drifted['abs_freq_diff_pct'] = categorical_drifted['freq_diff_pct'].abs()
        categorical_drifted = categorical_drifted.sort_values('abs_freq_diff_pct', ascending=False)
        
        print(f"\n{'Feature':<50} {'Severity':<12} {'Chi2 p-value':<12} {'Freq Diff %':<15} {'Train %':<12} {'Val %':<12}")
        print("-" * 110)
        for idx, row in categorical_drifted.head(20).iterrows():
            print(f"{row['feature'][:49]:<50} {row['drift_severity']:<12} {row['chi2_pvalue']:<12.4f} {row['freq_diff_pct']:<15.1f}% {row['presence_pct_train']:<12.1f}% {row['presence_pct_val']:<12.1f}%")
    
    # Features with zero value changes
    print("\n" + "=" * 80)
    print("⚠️  FEATURES WITH SIGNIFICANT ZERO-VALUE CHANGES")
    print("=" * 80)
    
    numeric_features = drift_df[drift_df['type'] == 'numeric'].copy()
    if 'zero_pct_train' in numeric_features.columns and 'zero_pct_val' in numeric_features.columns:
        numeric_features['zero_pct_diff'] = numeric_features['zero_pct_val'] - numeric_features['zero_pct_train']
        zero_changes = numeric_features[
            (numeric_features['zero_pct_diff'].abs() > 20) &  # More than 20% difference
            (numeric_features['drift_detected'] == True)
        ].sort_values('zero_pct_diff', key=abs, ascending=False)
        
        if len(zero_changes) > 0:
            print(f"\n{'Feature':<50} {'Zero % Train':<15} {'Zero % Val':<15} {'Difference':<15}")
            print("-" * 95)
            for idx, row in zero_changes.head(20).iterrows():
                print(f"{row['feature'][:49]:<50} {row['zero_pct_train']:<15.1f}% {row['zero_pct_val']:<15.1f}% {row['zero_pct_diff']:<15.1f}%")
        else:
            print("\n   No significant zero-value changes detected.")
    
    # Save detailed results
    print("\n" + "=" * 80)
    print("💾 SAVING RESULTS")
    print("=" * 80)
    
    os.makedirs('analysis', exist_ok=True)
    
    # Save full results as CSV
    drift_df.to_csv('analysis/feature_drift_analysis.csv', index=False, encoding='utf-8-sig')
    print(f"✅ Saved detailed results to: analysis/feature_drift_analysis.csv")
    
    # Save summary as JSON
    summary = {
        'total_features': int(total_features),
        'drift_detected_count': int(drift_detected),
        'drift_detected_pct': float(drift_pct),
        'severity_breakdown': severity_counts.to_dict(),
        'type_breakdown': {
            ftype: {
                'count': int(count),
                'drift_count': int(drift_df[drift_df['type'] == ftype]['drift_detected'].sum()),
                'drift_pct': float((drift_df[drift_df['type'] == ftype]['drift_detected'].sum() / count * 100) if count > 0 else 0)
            }
            for ftype, count in type_counts.items()
        },
        'top_drifted_numeric': numeric_drifted.head(20)[['feature', 'drift_severity', 'ks_pvalue', 'cohens_d', 'mean_diff_pct']].to_dict('records') if len(numeric_drifted) > 0 else [],
        'top_drifted_categorical': categorical_drifted.head(20)[['feature', 'drift_severity', 'chi2_pvalue', 'freq_diff_pct', 'presence_pct_train', 'presence_pct_val']].to_dict('records') if len(categorical_drifted) > 0 else []
    }
    
    with open('analysis/feature_drift_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Saved summary to: analysis/feature_drift_summary.json")
    
    # Feature groups analysis (if features follow naming patterns)
    print("\n" + "=" * 80)
    print("📊 FEATURE GROUP ANALYSIS")
    print("=" * 80)
    
    # Group features by common prefixes (e.g., week_1, week_2, etc.)
    feature_groups = {}
    for feature in all_cols:
        # Extract base name (remove week suffixes, etc.)
        base_name = feature
        for suffix in ['_week_1', '_week_2', '_week_3', '_week_4', '_week_5']:
            if feature.endswith(suffix):
                base_name = feature[:-len(suffix)]
                break
        
        if base_name not in feature_groups:
            feature_groups[base_name] = []
        feature_groups[base_name].append(feature)
    
    # Analyze groups with multiple features
    group_analysis = []
    for base_name, features in feature_groups.items():
        if len(features) > 1:
            group_drift = drift_df[drift_df['feature'].isin(features)]
            drift_count = group_drift['drift_detected'].sum()
            group_analysis.append({
                'group': base_name,
                'feature_count': len(features),
                'drift_count': int(drift_count),
                'drift_pct': float(drift_count / len(features) * 100) if len(features) > 0 else 0
            })
    
    if group_analysis:
        group_df = pd.DataFrame(group_analysis).sort_values('drift_pct', ascending=False)
        print(f"\n{'Feature Group':<50} {'Features':<12} {'Drifted':<12} {'Drift %':<12}")
        print("-" * 86)
        for idx, row in group_df.head(30).iterrows():
            print(f"{row['group'][:49]:<50} {row['feature_count']:<12} {row['drift_count']:<12} {row['drift_pct']:<12.1f}%")
    
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    
    return drift_df, summary

if __name__ == "__main__":
    drift_df, summary = analyze_feature_drift()

