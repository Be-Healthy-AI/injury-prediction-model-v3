#!/usr/bin/env python3
"""
Test different target ratios by downsampling positive class from train+val dataset
This allows quick testing without regenerating timelines

Tests ratios: 8%, 6%, 4%, 2%
Trains RF and GB models for each ratio
Evaluates on same test set for fair comparison
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import json
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix
)

def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types for JSON serialization"""
    if obj is None:
        return None
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, dict):
        return {str(key): convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

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
    """Sanitize feature names for LightGBM compatibility"""
    name_str = str(name)
    replacements = {
        '"': '_quote_', '\\': '_backslash_', '/': '_slash_',
        '\b': '_bs_', '\f': '_ff_', '\n': '_nl_', '\r': '_cr_', '\t': '_tab_',
        ' ': '_', "'": '_apostrophe_', ':': '_colon_', ';': '_semicolon_',
        ',': '_comma_', '{': '_lbrace_', '}': '_rbrace_',
        '[': '_lbracket_', ']': '_rbracket_', '&': '_amp_',
    }
    for old, new in replacements.items():
        name_str = name_str.replace(old, new)
    name_str = ''.join(char for char in name_str if ord(char) >= 32 or char in '\n\r\t')
    while '__' in name_str:
        name_str = name_str.replace('__', '_')
    name_str = name_str.strip('_')
    return name_str

def prepare_data(df):
    """Prepare data with basic preprocessing"""
    feature_columns = [
        col for col in df.columns
        if col not in ['player_id', 'reference_date', 'player_name', 'target']
    ]
    X = df[feature_columns].copy()
    y = df['target'].copy()
    
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    X_encoded = X.copy()
    
    for feature in categorical_features:
        X_encoded[feature] = X_encoded[feature].fillna('Unknown')
        X_encoded[feature] = X_encoded[feature].apply(clean_categorical_value)
        dummies = pd.get_dummies(X_encoded[feature], prefix=feature, drop_first=True)
        dummies.columns = [sanitize_feature_name(col) for col in dummies.columns]
        X_encoded = pd.concat([X_encoded, dummies], axis=1)
        X_encoded = X_encoded.drop(columns=[feature])
    
    for feature in numeric_features:
        X_encoded[feature] = X_encoded[feature].fillna(0)
    
    X_encoded.columns = [sanitize_feature_name(col) for col in X_encoded.columns]
    
    return X_encoded, y

def align_features(X_train, X_test):
    """Ensure all datasets have the same features"""
    common_features = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_features]
    X_test = X_test[common_features]
    return X_train, X_test

def apply_correlation_filter(X, threshold=0.8):
    """Apply correlation filter"""
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    kept = [col for col in X.columns if col not in to_drop]
    return kept

def evaluate_model(model, X, y, dataset_name):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.0,
        'gini': (2 * roc_auc_score(y, y_proba) - 1) if len(np.unique(y)) > 1 else 0.0
    }
    
    cm = confusion_matrix(y, y_pred)
    if cm.shape == (2, 2):
        metrics['confusion_matrix'] = {
            'tn': int(cm[0, 0]),
            'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]),
            'tp': int(cm[1, 1])
        }
    else:
        if len(cm) == 1:
            if y.sum() == 0:
                metrics['confusion_matrix'] = {'tn': int(cm[0, 0]), 'fp': 0, 'fn': 0, 'tp': 0}
            else:
                metrics['confusion_matrix'] = {'tn': 0, 'fp': 0, 'fn': int(y.sum()), 'tp': 0}
        else:
            metrics['confusion_matrix'] = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
    
    metrics = convert_numpy_types(metrics)
    
    print(f"\n   {dataset_name}:")
    print(f"      Accuracy: {metrics['accuracy']:.4f}")
    print(f"      Precision: {metrics['precision']:.4f}")
    print(f"      Recall: {metrics['recall']:.4f}")
    print(f"      F1-Score: {metrics['f1']:.4f}")
    print(f"      ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"      Gini: {metrics['gini']:.4f}")
    
    return metrics

def downsample_to_target_ratio(df, target_ratio, random_state=42):
    """
    Downsample positive class to achieve target ratio
    
    Args:
        df: DataFrame with 'target' column
        target_ratio: Desired ratio of positive class (e.g., 0.06 for 6%)
        random_state: Random seed for reproducibility
    
    Returns:
        Downsampled DataFrame
    """
    positive_samples = df[df['target'] == 1].copy()
    negative_samples = df[df['target'] == 0].copy()
    
    n_negative = len(negative_samples)
    n_positive_needed = int(n_negative * target_ratio / (1 - target_ratio))
    
    if n_positive_needed >= len(positive_samples):
        print(f"⚠️  Warning: Need {n_positive_needed} positives but only have {len(positive_samples)}")
        print(f"   Using all {len(positive_samples)} positives (actual ratio: {len(positive_samples)/(len(positive_samples)+n_negative):.2%})")
        return pd.concat([positive_samples, negative_samples], ignore_index=True)
    
    # Randomly sample positives
    positive_sampled = positive_samples.sample(
        n=n_positive_needed, 
        random_state=random_state
    )
    
    # Combine
    df_downsampled = pd.concat([positive_sampled, negative_samples], ignore_index=True)
    
    # Shuffle
    df_downsampled = df_downsampled.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    actual_ratio = df_downsampled['target'].mean()
    print(f"✅ Downsampled to {target_ratio:.1%} target ratio")
    print(f"   Positives: {len(positive_sampled):,} (from {len(positive_samples):,})")
    print(f"   Negatives: {len(negative_samples):,}")
    print(f"   Total: {len(df_downsampled):,}")
    print(f"   Actual ratio: {actual_ratio:.2%}")
    
    return df_downsampled

def train_rf(X_train, y_train, X_test, y_test):
    """Train Random Forest model"""
    print("\n" + "=" * 70)
    print("🌲 TRAINING RANDOM FOREST MODEL")
    print("=" * 70)
    
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    print("\n🌳 Training started...")
    train_start = datetime.now()
    rf_model.fit(X_train, y_train)
    train_time = datetime.now() - train_start
    print(f"✅ Training completed in {train_time}")
    
    train_metrics = evaluate_model(rf_model, X_train, y_train, "Training")
    test_metrics = evaluate_model(rf_model, X_test, y_test, "Test")
    
    return rf_model, train_metrics, test_metrics

def train_gb(X_train, y_train, X_test, y_test):
    """Train Gradient Boosting model"""
    print("\n" + "=" * 70)
    print("🚀 TRAINING GRADIENT BOOSTING MODEL")
    print("=" * 70)
    
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    
    print("\n🌳 Training started...")
    train_start = datetime.now()
    gb_model.fit(X_train, y_train)
    train_time = datetime.now() - train_start
    print(f"✅ Training completed in {train_time}")
    
    train_metrics = evaluate_model(gb_model, X_train, y_train, "Training")
    test_metrics = evaluate_model(gb_model, X_test, y_test, "Test")
    
    return gb_model, train_metrics, test_metrics

def main():
    print("="*80)
    print("TARGET RATIO DOWNSAMPLING TEST - MUSCULAR INJURIES")
    print("="*80)
    print("\n📋 Configuration:")
    print("   Testing ratios: 8%, 6%, 4%, 2%")
    print("   Models: Random Forest, Gradient Boosting")
    print("   Test set: >= 2025-07-01 (unchanged)")
    print("="*80)
    
    start_time = datetime.now()
    
    # Load data
    print("\n📂 Loading timeline data...")
    train_file = 'timelines_35day_enhanced_balanced_v4_muscular_train.csv'
    val_file = 'timelines_35day_enhanced_balanced_v4_muscular_val.csv'
    test_file = 'timelines_35day_enhanced_balanced_v4_muscular_test.csv'
    
    df_train = pd.read_csv(train_file, encoding='utf-8-sig')
    df_val = pd.read_csv(val_file, encoding='utf-8-sig')
    df_test = pd.read_csv(test_file, encoding='utf-8-sig')
    
    print(f"✅ Loaded training set: {len(df_train):,} records")
    print(f"   Injury ratio: {df_train['target'].mean():.1%}")
    print(f"✅ Loaded validation set: {len(df_val):,} records")
    print(f"   Injury ratio: {df_val['target'].mean():.1%}")
    print(f"✅ Loaded test set: {len(df_test):,} records")
    print(f"   Injury ratio: {df_test['target'].mean():.1%}")
    
    # Combine train and validation datasets
    print("\n📊 Combining training and validation datasets...")
    df_train_combined = pd.concat([df_train, df_val], ignore_index=True)
    print(f"✅ Combined training set: {len(df_train_combined):,} records")
    print(f"   Injury ratio: {df_train_combined['target'].mean():.1%}")
    
    # Prepare test data (only once, used for all ratios)
    print("\n📊 Preparing test data...")
    X_test, y_test = prepare_data(df_test)
    print(f"   Test features: {X_test.shape[1]}")
    
    # Test different target ratios
    target_ratios = [0.08, 0.06, 0.04, 0.02]
    all_results = {}
    
    for target_ratio in target_ratios:
        print("\n" + "="*80)
        print(f"TESTING TARGET RATIO: {target_ratio:.1%}")
        print("="*80)
        
        # Downsample to target ratio
        print(f"\n📉 Downsampling to {target_ratio:.1%} target ratio...")
        df_downsampled = downsample_to_target_ratio(
            df_train_combined, 
            target_ratio=target_ratio,
            random_state=42
        )
        
        # Prepare training data
        print("\n📊 Preparing training data...")
        X_train, y_train = prepare_data(df_downsampled)
        print(f"   Training features: {X_train.shape[1]}")
        
        # Align features
        print("\n🔧 Aligning features...")
        X_train, X_test_aligned = align_features(X_train, X_test)
        print(f"   Common features: {X_train.shape[1]}")
        
        # Apply correlation filter
        print("\n🔧 Applying correlation filter (threshold=0.8)...")
        CORR_THRESHOLD = 0.8
        selected_features = apply_correlation_filter(X_train, CORR_THRESHOLD)
        X_train = X_train[selected_features]
        X_test_aligned = X_test_aligned[selected_features]
        print(f"   Features after filtering: {len(selected_features)}")
        
        # Sanitize feature names
        X_train.columns = [sanitize_feature_name(col) for col in X_train.columns]
        X_test_aligned.columns = [sanitize_feature_name(col) for col in X_test_aligned.columns]
        
        ratio_key = f"{target_ratio:.0%}"
        all_results[ratio_key] = {}
        
        # Train Random Forest
        print(f"\n{'='*80}")
        print(f"RANDOM FOREST - {target_ratio:.1%} Ratio")
        print(f"{'='*80}")
        rf_model, rf_train_metrics, rf_test_metrics = train_rf(X_train, y_train, X_test_aligned, y_test)
        all_results[ratio_key]['RF'] = {
            'train': rf_train_metrics,
            'test': rf_test_metrics
        }
        
        # Train Gradient Boosting
        print(f"\n{'='*80}")
        print(f"GRADIENT BOOSTING - {target_ratio:.1%} Ratio")
        print(f"{'='*80}")
        gb_model, gb_train_metrics, gb_test_metrics = train_gb(X_train, y_train, X_test_aligned, y_test)
        all_results[ratio_key]['GB'] = {
            'train': gb_train_metrics,
            'test': gb_test_metrics
        }
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    os.makedirs('experiments', exist_ok=True)
    results_file = 'experiments/target_ratio_downsampling_test_results.json'
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Results saved to {results_file}")
    
    # Create summary
    print("\n" + "="*80)
    print("SUMMARY - TEST SET PERFORMANCE BY TARGET RATIO")
    print("="*80)
    
    summary_data = []
    for ratio_key in ['8%', '6%', '4%', '2%']:
        if ratio_key in all_results:
            for model_name in ['RF', 'GB']:
                if model_name in all_results[ratio_key]:
                    test_metrics = all_results[ratio_key][model_name]['test']
                    summary_data.append({
                        'Ratio': ratio_key,
                        'Model': model_name,
                        'F1-Score': test_metrics['f1'],
                        'Precision': test_metrics['precision'],
                        'Recall': test_metrics['recall'],
                        'ROC AUC': test_metrics['roc_auc'],
                        'Gini': test_metrics['gini']
                    })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))
    
    # Save summary
    summary_file = 'experiments/target_ratio_downsampling_test_summary.md'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("# Target Ratio Downsampling Test Results\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Test Set Performance by Target Ratio\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n## Detailed Results\n\n")
        f.write("See `target_ratio_downsampling_test_results.json` for full metrics.\n")
    
    print(f"\n✅ Summary saved to {summary_file}")
    
    total_time = datetime.now() - start_time
    print(f"\n✅ Completed in {total_time}")
    print("="*80)

if __name__ == '__main__':
    main()


