#!/usr/bin/env python3
"""
Analyze low-probability injury cases for muscular injuries
- Compares injuries in low-probability bins (0.0-0.3) vs high-probability bins (0.7-1.0)
- Compares injuries in low-probability bins (0.0-0.3) vs ALL injuries in dataset
- Identifies missing signals and feature differences
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import json
import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent.parent.parent
MODELS_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'models'
TIMELINES_DIR = SCRIPT_DIR.parent / 'timelines'
TEST_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'data' / 'timelines' / 'test'
TRAIN_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'data' / 'timelines' / 'train'
OUTPUT_DIR = MODELS_DIR / 'low_probability_analysis'

# Copy necessary functions directly to avoid import issues
def filter_timelines_for_model(timelines_df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Filter timelines for a specific model, excluding other injury types from negatives."""
    if target_column == 'target1':
        # Model 1 (muscular): Only target1=1 (positives) or (target1=0 AND target2=0) (negatives)
        mask = (timelines_df['target1'] == 1) | ((timelines_df['target1'] == 0) & (timelines_df['target2'] == 0))
    elif target_column == 'target2':
        # Model 2 (skeletal): Only target2=1 (positives) or (target1=0 AND target2=0) (negatives)
        mask = (timelines_df['target2'] == 1) | ((timelines_df['target1'] == 0) & (timelines_df['target2'] == 0))
    else:
        raise ValueError(f"Invalid target_column: {target_column}. Must be 'target1' or 'target2'")
    return timelines_df[mask].copy()

def clean_categorical_value(value):
    """Clean categorical values to remove special characters"""
    if pd.isna(value) or value is None:
        return 'Unknown'
    value_str = str(value).strip()
    problematic_values = ['tel:', 'tel', 'phone', 'n/a', 'na', 'null', 'none', '', 'nan', 'address:', 'website:']
    if value_str.lower() in problematic_values:
        return 'Unknown'
    replacements = {
        ':': '_', "'": '_', ',': '_', '"': '_', ';': '_', '/': '_', '\\': '_',
        '{': '_', '}': '_', '[': '_', ']': '_', '(': '_', ')': '_', '|': '_',
        '&': '_', '?': '_', '!': '_', '*': '_', '+': '_', '=': '_', '@': '_',
        '#': '_', '$': '_', '%': '_', '^': '_', ' ': '_',
    }
    for old_char, new_char in replacements.items():
        value_str = value_str.replace(old_char, new_char)
    value_str = ''.join(char for char in value_str if ord(char) >= 32 or char in '\n\r\t')
    while '__' in value_str:
        value_str = value_str.replace('__', '_')
    value_str = value_str.strip('_')
    return value_str if value_str else 'Unknown'

def sanitize_feature_name(name):
    """Sanitize feature names to be JSON-safe"""
    name_str = str(name)
    replacements = {
        '"': '_quote_', '\\': '_backslash_', '/': '_slash_',
        '\b': '_bs_', '\f': '_ff_', '\n': '_nl_', '\r': '_cr_', '\t': '_tab_', ' ': '_',
        "'": '_apostrophe_', ':': '_colon_', ';': '_semicolon_', ',': '_comma_',
        '{': '_lbrace_', '}': '_rbrace_', '[': '_lbracket_', ']': '_rbracket_', '&': '_amp_',
    }
    for old, new in replacements.items():
        name_str = name_str.replace(old, new)
    name_str = ''.join(char for char in name_str if ord(char) >= 32 or char in '\n\r\t')
    while '__' in name_str:
        name_str = name_str.replace('__', '_')
    return name_str.strip('_')

# Create a simplified prepare_data that doesn't use tqdm
def prepare_data_simple(df, cache_file=None, use_cache=True):
    """Prepare data with basic preprocessing (simplified version without tqdm)"""
    # Check cache
    if use_cache and cache_file and os.path.exists(cache_file):
        print(f"   Loading from cache: {cache_file.name}...")
        try:
            df_preprocessed = pd.read_csv(cache_file, encoding='utf-8-sig', low_memory=False)
            if len(df_preprocessed) != len(df):
                print(f"   Cache length mismatch, preprocessing fresh...")
                use_cache = False
            else:
                target_cols = ['target1', 'target2', 'target', 'target_combined']
                cols_to_drop = [col for col in target_cols if col in df_preprocessed.columns]
                if cols_to_drop:
                    X = df_preprocessed.drop(columns=cols_to_drop)
                else:
                    X = df_preprocessed
                print(f"   Loaded preprocessed data from cache")
                return X
        except Exception as e:
            print(f"   Failed to load cache ({e}), preprocessing fresh...")
            use_cache = False
    
    # Drop non-feature columns
    feature_columns = [
        col for col in df.columns
        if col not in ['player_id', 'reference_date', 'date', 'player_name', 'target1', 'target2', 'target', 'has_minimum_activity', 'target_combined']
    ]
    X = df[feature_columns].copy()
    
    # Handle missing values and encode categorical features
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    X_encoded = X.copy()
    
    # Encode categorical features
    if len(categorical_features) > 0:
        print(f"   Processing {len(categorical_features)} categorical features...")
        for idx, feature in enumerate(categorical_features):
            if (idx + 1) % 5 == 0:
                print(f"      Progress: {idx + 1}/{len(categorical_features)} categorical features...")
            X_encoded[feature] = X_encoded[feature].fillna('Unknown')
            X_encoded[feature] = X_encoded[feature].apply(clean_categorical_value)
            dummies = pd.get_dummies(X_encoded[feature], prefix=feature, drop_first=True)
            dummies.columns = [sanitize_feature_name(col) for col in dummies.columns]
            X_encoded = pd.concat([X_encoded, dummies], axis=1)
            X_encoded = X_encoded.drop(columns=[feature])
    
    # Fill numeric missing values
    if len(numeric_features) > 0:
        print(f"   Processing {len(numeric_features)} numeric features...")
        for feature in numeric_features:
            X_encoded[feature] = X_encoded[feature].fillna(0)
    
    # Sanitize all column names
    X_encoded.columns = [sanitize_feature_name(col) for col in X_encoded.columns]
    
    return X_encoded

prepare_data = prepare_data_simple

def load_model(model_path, columns_path):
    """Load trained model and feature columns"""
    model = joblib.load(model_path)
    with open(columns_path, 'r') as f:
        columns = json.load(f)
    return model, columns

def generate_predictions(model, X, columns):
    """Generate predictions using model"""
    X_aligned = X.reindex(columns=columns, fill_value=0)
    X_aligned = X_aligned[columns]
    y_proba = model.predict_proba(X_aligned)[:, 1]
    return y_proba

def bin_probability(prob):
    """Bin probability into 0-9 range"""
    return min(int(prob * 10), 9)

def load_test_data():
    """Load test dataset"""
    test_file = TEST_DIR / 'timelines_35day_season_2025_2026_v4_muscular_test.csv'
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")
    print(f"\n📂 Loading test dataset: {test_file.name}")
    return pd.read_csv(test_file, encoding='utf-8-sig', low_memory=False)

def load_all_training_data():
    """Load all training data to get baseline injury statistics"""
    print("\n📂 Loading all training data for baseline comparison...")
    import glob
    import os
    
    pattern = str(TRAIN_DIR / 'timelines_35day_season_*_v4_muscular_train.csv')
    files = glob.glob(pattern)
    season_files = []
    
    for filepath in files:
        filename = os.path.basename(filepath)
        if '_10pc_' in filename or '_25pc_' in filename or '_50pc_' in filename:
            continue
        if 'season_' in filename:
            parts = filename.split('season_')
            if len(parts) > 1:
                season_part = parts[1].split('_v4_muscular_train')[0]
                if season_part != '2025_2026':
                    if season_part >= '2018_2019':
                        season_files.append(filepath)
    
    print(f"   Found {len(season_files)} season files")
    
    dfs = []
    for filepath in season_files:
        try:
            df = pd.read_csv(filepath, encoding='utf-8-sig', low_memory=False)
            if 'target1' in df.columns:
                dfs.append(df)
        except Exception as e:
            print(f"   ⚠️  Error loading {filepath}: {e}")
            continue
    
    if not dfs:
        return None
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"   ✅ Loaded {len(combined_df):,} training records")
    return combined_df

def compare_feature_groups(df_low, df_high, feature_cols, group1_name, group2_name):
    """Compare features between two groups of injuries"""
    print(f"\n📊 Comparing {group1_name} vs {group2_name}...")
    print(f"   Analyzing {len(feature_cols)} features...")
    
    comparison_results = []
    
    for idx, feature in enumerate(feature_cols):
        if (idx + 1) % 50 == 0:
            print(f"   Progress: {idx + 1}/{len(feature_cols)} features analyzed...")
        if feature not in df_low.columns or feature not in df_high.columns:
            continue
        
        low_vals = df_low[feature].dropna()
        high_vals = df_high[feature].dropna()
        
        if len(low_vals) == 0 or len(high_vals) == 0:
            continue
        
        # Skip if all values are the same
        if low_vals.nunique() == 1 and high_vals.nunique() == 1 and low_vals.iloc[0] == high_vals.iloc[0]:
            continue
        
        # Calculate statistics - handle errors gracefully
        try:
            if pd.api.types.is_numeric_dtype(low_vals):
                low_mean = low_vals.mean()
                high_mean = high_vals.mean()
                low_median = low_vals.median()
                high_median = high_vals.median()
                low_std = low_vals.std()
                high_std = high_vals.std()
            else:
                # Categorical - use mode instead
                low_mean = np.nan
                high_mean = np.nan
                low_median = np.nan
                high_median = np.nan
                low_std = np.nan
                high_std = np.nan
        except (TypeError, ValueError) as e:
            # Skip features that can't be converted to numeric
            continue
        
        # Statistical test
        try:
            if pd.api.types.is_numeric_dtype(low_vals):
                # Use t-test or Mann-Whitney U test
                if len(low_vals) >= 3 and len(high_vals) >= 3:
                    try:
                        stat, p_value = stats.mannwhitneyu(low_vals, high_vals, alternative='two-sided')
                    except:
                        stat, p_value = stats.ttest_ind(low_vals, high_vals)
                else:
                    p_value = np.nan
                    stat = np.nan
            else:
                # Categorical - use chi-square
                contingency = pd.crosstab(
                    pd.concat([df_low[feature], df_high[feature]]),
                    pd.concat([pd.Series([group1_name]*len(df_low)), pd.Series([group2_name]*len(df_high))])
                )
                if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                    try:
                        stat, p_value, _, _ = stats.chi2_contingency(contingency)
                    except:
                        p_value = np.nan
                        stat = np.nan
                else:
                    p_value = np.nan
                    stat = np.nan
        except:
            p_value = np.nan
            stat = np.nan
        
        # Calculate difference
        if pd.api.types.is_numeric_dtype(low_vals):
            mean_diff = high_mean - low_mean
            pct_diff = (mean_diff / abs(low_mean)) * 100 if low_mean != 0 else np.nan
        else:
            mean_diff = np.nan
            pct_diff = np.nan
        
        comparison_results.append({
            'feature': feature,
            f'{group1_name}_mean': low_mean if pd.api.types.is_numeric_dtype(low_vals) else np.nan,
            f'{group1_name}_median': low_median if pd.api.types.is_numeric_dtype(low_vals) else np.nan,
            f'{group1_name}_std': low_std if pd.api.types.is_numeric_dtype(low_vals) else np.nan,
            f'{group1_name}_count': len(low_vals),
            f'{group2_name}_mean': high_mean if pd.api.types.is_numeric_dtype(low_vals) else np.nan,
            f'{group2_name}_median': high_median if pd.api.types.is_numeric_dtype(low_vals) else np.nan,
            f'{group2_name}_std': high_std if pd.api.types.is_numeric_dtype(low_vals) else np.nan,
            f'{group2_name}_count': len(high_vals),
            'mean_difference': mean_diff,
            'pct_difference': pct_diff,
            'p_value': p_value,
            'significant': p_value < 0.05 if not pd.isna(p_value) else False
        })
    
    return pd.DataFrame(comparison_results)

def analyze_low_probability_injuries():
    """Main analysis function"""
    print("="*80)
    print("ANALYZING LOW-PROBABILITY INJURY CASES - MUSCULAR INJURIES")
    print("="*80)
    print(f"\n⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\n" + "="*80)
    print("STEP 1: LOADING MODEL")
    print("="*80)
    model_path = MODELS_DIR / 'lgbm_muscular_v4_enriched_model1.joblib'
    columns_path = MODELS_DIR / 'lgbm_muscular_v4_enriched_model1_columns.json'
    
    if not model_path.exists() or not columns_path.exists():
        print(f"❌ Model files not found!")
        return 1
    
    model, columns = load_model(model_path, columns_path)
    print(f"✅ Loaded model: {model_path.name}")
    print(f"   Features: {len(columns)}")
    
    # Load test data
    print("\n" + "="*80)
    print("STEP 2: LOADING TEST DATA")
    print("="*80)
    df_test_all = load_test_data()
    df_test = filter_timelines_for_model(df_test_all, 'target1')
    
    print(f"✅ Filtered test data: {len(df_test):,} records")
    print(f"   Injuries: {(df_test['target1'] == 1).sum():,}")
    
    # Prepare features - try to use cache first
    print("\n" + "="*80)
    print("STEP 3: PREPARING FEATURES")
    print("="*80)
    
    # Try to load from cache if available
    cache_file = ROOT_DIR / 'cache' / 'enriched_v4' / 'preprocessed_target1_test_enriched.csv'
    if cache_file.exists():
        print(f"   Loading from cache: {cache_file.name}")
        try:
            X_test = pd.read_csv(cache_file, encoding='utf-8-sig', low_memory=False)
            # Drop target columns if present
            target_cols = ['target1', 'target2', 'target', 'target_combined']
            cols_to_drop = [col for col in target_cols if col in X_test.columns]
            if cols_to_drop:
                X_test = X_test.drop(columns=cols_to_drop)
            print(f"   ✅ Loaded {len(X_test):,} preprocessed records from cache")
        except Exception as e:
            print(f"   ⚠️  Cache load failed: {e}, preprocessing fresh...")
            X_test = prepare_data(df_test, cache_file=None, use_cache=False)
    else:
        print("   No cache found, preprocessing fresh (this may take a few minutes)...")
        X_test = prepare_data(df_test, cache_file=None, use_cache=False)
    y_test = df_test['target1'].values
    
    # Align features
    X_test_aligned = X_test.reindex(columns=columns, fill_value=0)
    X_test_aligned = X_test_aligned[columns]
    
    # Generate predictions
    print("\n" + "="*80)
    print("STEP 4: GENERATING PREDICTIONS")
    print("="*80)
    y_proba = generate_predictions(model, X_test, columns)
    
    # Add predictions to dataframe
    df_test_with_proba = df_test.copy()
    df_test_with_proba['predicted_probability'] = y_proba
    df_test_with_proba['probability_bin'] = df_test_with_proba['predicted_probability'].apply(bin_probability)
    
    # Identify injury groups
    injuries = df_test_with_proba[df_test_with_proba['target1'] == 1].copy()
    low_prob_injuries = injuries[injuries['probability_bin'] <= 2].copy()  # 0.0-0.3
    high_prob_injuries = injuries[injuries['probability_bin'] >= 7].copy()  # 0.7-1.0
    
    print(f"\n📊 Injury Distribution:")
    print(f"   Total injuries: {len(injuries):,}")
    print(f"   Low-probability injuries (0.0-0.3): {len(low_prob_injuries):,} ({len(low_prob_injuries)/len(injuries)*100:.1f}%)")
    print(f"   High-probability injuries (0.7-1.0): {len(high_prob_injuries):,} ({len(high_prob_injuries)/len(injuries)*100:.1f}%)")
    print(f"   Average probability - Low: {low_prob_injuries['predicted_probability'].mean():.4f}")
    print(f"   Average probability - High: {high_prob_injuries['predicted_probability'].mean():.4f}")
    
    # Load all training injuries for baseline
    print("\n" + "="*80)
    print("STEP 5: LOADING TRAINING DATA FOR BASELINE")
    print("="*80)
    df_train_all = load_all_training_data()
    if df_train_all is not None:
        df_train = filter_timelines_for_model(df_train_all, 'target1')
        all_training_injuries = df_train[df_train['target1'] == 1].copy()
        print(f"✅ Loaded {len(all_training_injuries):,} training injuries for baseline")
    else:
        all_training_injuries = None
        print("⚠️  Could not load training data")
    
    # Prepare features for comparisons
    print("\n" + "="*80)
    print("STEP 6: PREPARING FEATURES FOR COMPARISON")
    print("="*80)
    
    # Get feature columns (exclude metadata)
    feature_cols = [col for col in df_test.columns 
                    if col not in ['player_id', 'reference_date', 'date', 'player_name', 
                                  'target1', 'target2', 'target', 'has_minimum_activity',
                                  'predicted_probability', 'probability_bin']]
    
    print(f"   Analyzing {len(feature_cols)} features")
    
    # Comparison 1: Low-probability vs High-probability injuries
    print("\n" + "="*80)
    print("STEP 7: COMPARISON 1 - Low vs High Probability Injuries")
    print("="*80)
    
    if len(low_prob_injuries) > 0 and len(high_prob_injuries) > 0:
        comparison1 = compare_feature_groups(
            low_prob_injuries, 
            high_prob_injuries,
            feature_cols,
            'Low_Probability',
            'High_Probability'
        )
        
        # Sort by significance and difference
        comparison1['abs_mean_difference'] = comparison1['mean_difference'].abs()
        comparison1 = comparison1.sort_values(['significant', 'p_value', 'abs_mean_difference'], 
                                              ascending=[False, True, False])
        
        print(f"\n✅ Found {len(comparison1)} comparable features")
        print(f"   Significant differences (p<0.05): {comparison1['significant'].sum()}")
        
        # Save comparison 1
        comp1_csv = OUTPUT_DIR / 'comparison_low_vs_high_probability.csv'
        comparison1.to_csv(comp1_csv, index=False, encoding='utf-8-sig')
        print(f"   ✅ Saved: {comp1_csv.name}")
        
        # Top differences
        print(f"\n📊 Top 20 Most Different Features (Low vs High):")
        print("-"*80)
        top_diff = comparison1.head(20)
        for _, row in top_diff.iterrows():
            sig = "*" if row['significant'] else " "
            low_val = f"{row['Low_Probability_mean']:.4f}" if not pd.isna(row['Low_Probability_mean']) else "N/A"
            high_val = f"{row['High_Probability_mean']:.4f}" if not pd.isna(row['High_Probability_mean']) else "N/A"
            diff_val = f"{row['mean_difference']:+.4f}" if not pd.isna(row['mean_difference']) else "N/A"
            p_val = f"{row['p_value']:.4f}" if not pd.isna(row['p_value']) else "N/A"
            print(f"{sig} {row['feature']:<50} | Low: {low_val:<8} | High: {high_val:<8} | Diff: {diff_val:<8} | p={p_val}")
    else:
        print("⚠️  Not enough injuries in both groups for comparison")
        comparison1 = None
    
    # Comparison 2: Low-probability injuries vs All training injuries
    print("\n" + "="*80)
    print("STEP 8: COMPARISON 2 - Low Probability vs All Training Injuries")
    print("="*80)
    
    if len(low_prob_injuries) > 0 and all_training_injuries is not None and len(all_training_injuries) > 0:
        # Prepare training injuries features (sample if too large)
        if len(all_training_injuries) > 10000:
            print(f"   Sampling 10,000 from {len(all_training_injuries):,} training injuries...")
            all_training_injuries = all_training_injuries.sample(n=10000, random_state=42)
        
        comparison2 = compare_feature_groups(
            low_prob_injuries,
            all_training_injuries,
            feature_cols,
            'Low_Probability_Test',
            'All_Training_Injuries'
        )
        
        comparison2['abs_mean_difference'] = comparison2['mean_difference'].abs()
        comparison2 = comparison2.sort_values(['significant', 'p_value', 'abs_mean_difference'],
                                              ascending=[False, True, False])
        
        print(f"\n✅ Found {len(comparison2)} comparable features")
        print(f"   Significant differences (p<0.05): {comparison2['significant'].sum()}")
        
        # Save comparison 2
        comp2_csv = OUTPUT_DIR / 'comparison_low_probability_vs_all_training.csv'
        comparison2.to_csv(comp2_csv, index=False, encoding='utf-8-sig')
        print(f"   ✅ Saved: {comp2_csv.name}")
        
        # Top differences
        print(f"\n📊 Top 20 Most Different Features (Low Prob Test vs All Training):")
        print("-"*80)
        top_diff2 = comparison2.head(20)
        for _, row in top_diff2.iterrows():
            sig = "*" if row['significant'] else " "
            test_val = f"{row['Low_Probability_Test_mean']:.4f}" if not pd.isna(row['Low_Probability_Test_mean']) else "N/A"
            train_val = f"{row['All_Training_Injuries_mean']:.4f}" if not pd.isna(row['All_Training_Injuries_mean']) else "N/A"
            diff_val = f"{row['mean_difference']:+.4f}" if not pd.isna(row['mean_difference']) else "N/A"
            p_val = f"{row['p_value']:.4f}" if not pd.isna(row['p_value']) else "N/A"
            print(f"{sig} {row['feature']:<50} | Test: {test_val:<8} | Training: {train_val:<8} | Diff: {diff_val:<8} | p={p_val}")
    else:
        print("⚠️  Not enough data for comparison")
        comparison2 = None
    
    # Save detailed injury cases
    print("\n" + "="*80)
    print("STEP 9: SAVING DETAILED INJURY CASES")
    print("="*80)
    
    # Select top features to include in detailed CSV
    top_features_to_save = feature_cols[:100]  # Save first 100 features
    
    low_prob_csv = OUTPUT_DIR / 'low_probability_injuries_detailed.csv'
    cols_to_save = ['player_id', 'reference_date', 'player_name', 'predicted_probability', 'probability_bin'] + top_features_to_save
    cols_to_save = [c for c in cols_to_save if c in low_prob_injuries.columns]
    low_prob_injuries[cols_to_save].to_csv(
        low_prob_csv, index=False, encoding='utf-8-sig'
    )
    print(f"   ✅ Saved: {low_prob_csv.name} ({len(low_prob_injuries)} cases)")
    
    if len(high_prob_injuries) > 0:
        high_prob_csv = OUTPUT_DIR / 'high_probability_injuries_detailed.csv'
        cols_to_save = ['player_id', 'reference_date', 'player_name', 'predicted_probability', 'probability_bin'] + top_features_to_save
        cols_to_save = [c for c in cols_to_save if c in high_prob_injuries.columns]
        high_prob_injuries[cols_to_save].to_csv(
            high_prob_csv, index=False, encoding='utf-8-sig'
        )
        print(f"   ✅ Saved: {high_prob_csv.name} ({len(high_prob_injuries)} cases)")
    
    # Generate summary report
    print("\n" + "="*80)
    print("STEP 10: GENERATING SUMMARY REPORT")
    print("="*80)
    
    summary_lines = [
        "# Low-Probability Injury Analysis - Muscular Injuries",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Overview",
        "",
        f"- **Total injuries in test set:** {len(injuries):,}",
        f"- **Low-probability injuries (0.0-0.3):** {len(low_prob_injuries):,} ({len(low_prob_injuries)/len(injuries)*100:.1f}%)",
        f"- **High-probability injuries (0.7-1.0):** {len(high_prob_injuries):,} ({len(high_prob_injuries)/len(injuries)*100:.1f}%)",
        "",
        "## Key Findings",
        ""
    ]
    
    if comparison1 is not None and len(comparison1) > 0:
        summary_lines.extend([
            "### Comparison 1: Low vs High Probability Injuries",
            "",
            f"- **Significant differences found:** {comparison1['significant'].sum()} features",
            "",
            "**Top 10 Most Different Features:**",
            ""
        ])
        
        for i, (_, row) in enumerate(comparison1.head(10).iterrows(), 1):
            low_val = f"{row['Low_Probability_mean']:.4f}" if not pd.isna(row['Low_Probability_mean']) else "N/A"
            high_val = f"{row['High_Probability_mean']:.4f}" if not pd.isna(row['High_Probability_mean']) else "N/A"
            diff_val = f"{row['mean_difference']:+.4f}" if not pd.isna(row['mean_difference']) else "N/A"
            p_val = f"{row['p_value']:.4f}" if not pd.isna(row['p_value']) else "N/A"
            summary_lines.append(
                f"{i}. **{row['feature']}**: "
                f"Low={low_val}, "
                f"High={high_val}, "
                f"Diff={diff_val} "
                f"(p={p_val})"
            )
        
        summary_lines.append("")
    
    if comparison2 is not None and len(comparison2) > 0:
        summary_lines.extend([
            "### Comparison 2: Low Probability Test vs All Training Injuries",
            "",
            f"- **Significant differences found:** {comparison2['significant'].sum()} features",
            "",
            "**Top 10 Most Different Features:**",
            ""
        ])
        
        for i, (_, row) in enumerate(comparison2.head(10).iterrows(), 1):
            test_val = f"{row['Low_Probability_Test_mean']:.4f}" if not pd.isna(row['Low_Probability_Test_mean']) else "N/A"
            train_val = f"{row['All_Training_Injuries_mean']:.4f}" if not pd.isna(row['All_Training_Injuries_mean']) else "N/A"
            diff_val = f"{row['mean_difference']:+.4f}" if not pd.isna(row['mean_difference']) else "N/A"
            p_val = f"{row['p_value']:.4f}" if not pd.isna(row['p_value']) else "N/A"
            summary_lines.append(
                f"{i}. **{row['feature']}**: "
                f"Test={test_val}, "
                f"Training={train_val}, "
                f"Diff={diff_val} "
                f"(p={p_val})"
            )
        
        summary_lines.append("")
    
    summary_lines.extend([
        "## Recommendations",
        "",
        "1. **Review top different features** to understand what signals are missing",
        "2. **Investigate low-probability injury cases** in detail",
        "3. **Consider adding features** that capture the differences identified",
        "4. **Review feature engineering** for features with large differences",
        "",
        "## Files Generated",
        "",
        "- `comparison_low_vs_high_probability.csv`: Detailed feature comparison",
        "- `comparison_low_probability_vs_all_training.csv`: Baseline comparison",
        "- `low_probability_injuries_detailed.csv`: All low-probability injury cases",
        "- `high_probability_injuries_detailed.csv`: All high-probability injury cases"
    ])
    
    summary_path = OUTPUT_DIR / 'analysis_summary.md'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
    print(f"   ✅ Saved: {summary_path.name}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\n📁 Output saved to: {OUTPUT_DIR}")
    
    return 0

if __name__ == "__main__":
    sys.exit(analyze_low_probability_injuries())
