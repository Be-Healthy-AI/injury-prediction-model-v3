#!/usr/bin/env python3
"""
Combine metrics from all 4 models (RF, GB, XGB, LGBM) into a single file
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
    elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    elif isinstance(obj, float) and np.isnan(obj):
        return None
    else:
        return obj

def evaluate_model(model, X, y, dataset_name):
    """Evaluate model and return metrics"""
    print(f"   Evaluating {dataset_name} ({len(X):,} samples)...")
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
    return metrics

def clean_categorical_value(value):
    """Clean categorical values - but preserve spaces for existing models"""
    if pd.isna(value) or value is None:
        return 'Unknown'
    value_str = str(value).strip()
    problematic_values = ['tel:', 'tel', 'phone', 'n/a', 'na', 'null', 'none', '', 'nan', 'address:', 'website:']
    if value_str.lower() in problematic_values:
        return 'Unknown'
    # Only replace truly problematic characters, but KEEP SPACES for existing models
    replacements = {
        ':': '_',
        "'": '_',
        ',': '_',
        '"': '_',
        ';': '_',
        '/': '_',
        '\\': '_',
        '{': '_',
        '}': '_',
        '[': '_',
        ']': '_',
        '(': '_',
        ')': '_',
        '|': '_',
        '&': '_',
        '?': '_',
        '!': '_',
        '*': '_',
        '+': '_',
        '=': '_',
        '@': '_',
        '#': '_',
        '$': '_',
        '%': '_',
        '^': '_',
        # NOTE: We do NOT replace spaces here - models were trained with spaces
    }
    for old_char, new_char in replacements.items():
        value_str = value_str.replace(old_char, new_char)
    value_str = ''.join(char for char in value_str if ord(char) >= 32 or char in '\n\r\t')
    while '__' in value_str:
        value_str = value_str.replace('__', '_')
    value_str = value_str.strip('_')
    if not value_str:
        return 'Unknown'
    return value_str

def prepare_data(df, replace_spaces_in_values=False):
    """Prepare data - preserve spaces for existing models"""
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
        # Clean categorical values (preserving spaces if replace_spaces_in_values=False)
        if replace_spaces_in_values:
            # For LightGBM: replace spaces
            X_encoded[feature] = X_encoded[feature].apply(lambda x: clean_categorical_value(x).replace(' ', '_') if pd.notna(x) else 'Unknown')
        else:
            # For existing models: keep spaces
            X_encoded[feature] = X_encoded[feature].apply(lambda x: clean_categorical_value(x) if pd.notna(x) else 'Unknown')
        
        dummies = pd.get_dummies(X_encoded[feature], prefix=feature, drop_first=True)
        X_encoded = pd.concat([X_encoded, dummies], axis=1)
        X_encoded = X_encoded.drop(columns=[feature])
    
    for feature in numeric_features:
        X_encoded[feature] = X_encoded[feature].fillna(0)
    
    return X_encoded, y

def align_features(X_train, X_test):
    """Ensure all datasets have the same features"""
    common_features = list(set(X_train.columns) & set(X_test.columns))
    common_features = sorted(common_features)
    return X_train[common_features], X_test[common_features]

def main():
    print("="*80)
    print("COMBINING METRICS FROM ALL 4 MODELS")
    print("="*80)
    
    # Load data
    print("\n📂 Loading timeline data...")
    train_file = 'timelines_35day_enhanced_balanced_v4_muscular_train.csv'
    val_file = 'timelines_35day_enhanced_balanced_v4_muscular_val.csv'
    test_file = 'timelines_35day_enhanced_balanced_v4_muscular_test.csv'
    
    df_train = pd.read_csv(train_file, encoding='utf-8-sig')
    df_val = pd.read_csv(val_file, encoding='utf-8-sig')
    df_test = pd.read_csv(test_file, encoding='utf-8-sig')
    
    df_train_combined = pd.concat([df_train, df_val], ignore_index=True)
    print(f"✅ Loaded {len(df_train_combined):,} training records and {len(df_test):,} test records")
    
    # Prepare data WITHOUT replacing spaces (for RF, GB, XGB compatibility)
    print("\n📊 Preparing data (preserving spaces for existing models)...")
    X_train_orig, y_train = prepare_data(df_train_combined, replace_spaces_in_values=False)
    X_test_orig, y_test = prepare_data(df_test, replace_spaces_in_values=False)
    
    # Align features
    X_train_orig, X_test_orig = align_features(X_train_orig, X_test_orig)
    
    # Apply correlation filter
    print("\n🔎 Applying correlation filter...")
    def apply_correlation_filter(X, threshold=0.8):
        """Drop one feature from each highly correlated pair."""
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        kept = [col for col in X.columns if col not in to_drop]
        return kept
    
    CORR_THRESHOLD = 0.8
    selected_features = apply_correlation_filter(X_train_orig, CORR_THRESHOLD)
    X_train_orig = X_train_orig[selected_features]
    X_test_orig = X_test_orig[selected_features]
    print(f"✅ Using {len(selected_features)} features after correlation filtering")
    
    # Load existing LGBM metrics
    print("\n📂 Loading existing LGBM metrics...")
    lgbm_metrics_file = 'experiments/v4_muscular_combined_corr_metrics.json'
    if os.path.exists(lgbm_metrics_file):
        with open(lgbm_metrics_file, 'r', encoding='utf-8') as f:
            existing_metrics = json.load(f)
        if 'LGBM' in existing_metrics:
            print("✅ Found LGBM metrics")
            all_results = {'LGBM': existing_metrics['LGBM']}
        else:
            print("⚠️  LGBM metrics not found in file")
            all_results = {}
    else:
        print("⚠️  LGBM metrics file not found")
        all_results = {}
    
    # Evaluate RF, GB, XGB models
    models_to_evaluate = [
        ('RF', 'rf_model_v4_muscular_combined_corr.joblib'),
        ('GB', 'gb_model_v4_muscular_combined_corr.joblib'),
        ('XGB', 'xgb_model_v4_muscular_combined_corr.joblib')
    ]
    
    for model_name, model_file in models_to_evaluate:
        model_path = f'models/{model_file}'
        if not os.path.exists(model_path):
            print(f"\n⚠️  {model_name} model not found: {model_path}")
            continue
        
        print(f"\n📊 Evaluating {model_name} model...")
        model = joblib.load(model_path)
        
        # Load feature columns to ensure alignment
        columns_file = model_path.replace('.joblib', '_columns.json')
        if os.path.exists(columns_file):
            with open(columns_file, 'r', encoding='utf-8') as f:
                model_columns = json.load(f)
            # Align features
            X_train_aligned = X_train_orig.reindex(columns=model_columns, fill_value=0)
            X_test_aligned = X_test_orig.reindex(columns=model_columns, fill_value=0)
        else:
            X_train_aligned = X_train_orig
            X_test_aligned = X_test_orig
        
        train_metrics = evaluate_model(model, X_train_aligned, y_train, "Training (Train+Val Combined)")
        test_metrics = evaluate_model(model, X_test_aligned, y_test, "Test (>= 2025-07-01)")
        
        all_results[model_name] = {
            'train': train_metrics,
            'test': test_metrics
        }
        print(f"✅ {model_name} evaluation complete")
    
    # Save combined metrics
    print("\n💾 Saving combined metrics...")
    os.makedirs('experiments', exist_ok=True)
    with open('experiments/v4_muscular_combined_corr_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved combined metrics to experiments/v4_muscular_combined_corr_metrics.json")
    print(f"   Models in file: {', '.join(all_results.keys())}")
    
    # Display comparison table
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON - ALL 4 MODELS")
    print("="*80)
    print(f"\n{'Model':<10} {'Dataset':<30} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC AUC':<12} {'Gini':<12}")
    print("-" * 100)
    
    for model_name in ['RF', 'GB', 'XGB', 'LGBM']:
        if model_name not in all_results:
            print(f"{model_name:<10} {'N/A':<30} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
            continue
        
        results = all_results[model_name]
        print(f"{model_name:<10} {'Training (Train+Val Combined)':<30} {results['train']['precision']:<12.4f} {results['train']['recall']:<12.4f} {results['train']['f1']:<12.4f} {results['train']['roc_auc']:<12.4f} {results['train']['gini']:<12.4f}")
        print(f"{'':<10} {'Test (>= 2025-07-01)':<30} {results['test']['precision']:<12.4f} {results['test']['recall']:<12.4f} {results['test']['f1']:<12.4f} {results['test']['roc_auc']:<12.4f} {results['test']['gini']:<12.4f}")
        print("-" * 100)
    
    # Detailed table with confusion matrix
    print("\n" + "="*80)
    print("DETAILED PERFORMANCE TABLE (with Confusion Matrix)")
    print("="*80)
    print(f"\n{'Model':<10} {'Dataset':<30} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'ROC AUC':<12} {'TP':<8} {'FP':<8} {'TN':<8} {'FN':<8}")
    print("-" * 135)
    
    for model_name in ['RF', 'GB', 'XGB', 'LGBM']:
        if model_name not in all_results:
            continue
        
        results = all_results[model_name]
        for dataset_name, metrics in [('Training (Train+Val Combined)', results['train']), 
                                       ('Test (>= 2025-07-01)', results['test'])]:
            cm = metrics['confusion_matrix']
            print(f"{model_name:<10} {dataset_name:<30} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1']:<12.4f} {metrics['roc_auc']:<12.4f} {cm['tp']:<8} {cm['fp']:<8} {cm['tn']:<8} {cm['fn']:<8}")
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)

if __name__ == '__main__':
    main()

