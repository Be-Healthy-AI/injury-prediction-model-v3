#!/usr/bin/env python3
"""
Add LightGBM metrics to existing RF, GB, XGB metrics file
This script evaluates RF, GB, XGB models and combines with LGBM metrics
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

# Import functions from complete_lgbm_training.py
sys.path.insert(0, '.')
from complete_lgbm_training import (
    prepare_data, align_features,
    evaluate_model, convert_numpy_types
)

def main():
    print("="*80)
    print("COMBINING ALL MODEL METRICS (RF, GB, XGB, LGBM)")
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
    
    df_train_combined = pd.concat([df_train, df_val], ignore_index=True)
    print(f"✅ Loaded {len(df_train_combined):,} training records and {len(df_test):,} test records")
    
    # Prepare data WITHOUT replacing spaces (for RF, GB, XGB compatibility)
    print("\n📊 Preparing data (preserving spaces for existing models)...")
    prep_start = datetime.now()
    X_train_orig, y_train = prepare_data(df_train_combined, replace_spaces_in_values=False)
    X_test_orig, y_test = prepare_data(df_test, replace_spaces_in_values=False)
    prep_time = datetime.now() - prep_start
    print(f"✅ Data preparation completed in {prep_time}")
    
    # Align features
    print("\n🔧 Aligning features...")
    X_train_orig, X_test_orig = align_features(X_train_orig, X_test_orig)
    
    # Apply correlation filter
    print("\n🔎 Applying correlation filter...")
    CORR_THRESHOLD = 0.8
    
    def apply_correlation_filter_simple(X, threshold=0.8):
        """Drop one feature from each highly correlated pair"""
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        kept = [col for col in X.columns if col not in to_drop]
        return kept
    
    selected_features = apply_correlation_filter_simple(X_train_orig, CORR_THRESHOLD)
    X_train_orig = X_train_orig[selected_features]
    X_test_orig = X_test_orig[selected_features]
    print(f"✅ Using {len(selected_features)} features after correlation filtering")
    
    all_results = {}
    
    # Load existing LGBM metrics
    print("\n📂 Loading existing LGBM metrics...")
    lgbm_metrics_file = 'experiments/v4_muscular_combined_corr_metrics.json'
    if os.path.exists(lgbm_metrics_file):
        with open(lgbm_metrics_file, 'r', encoding='utf-8') as f:
            existing_metrics = json.load(f)
        if 'LGBM' in existing_metrics:
            all_results['LGBM'] = existing_metrics['LGBM']
            print("✅ Loaded LGBM metrics")
        else:
            print("⚠️  LGBM metrics not found in file")
    else:
        print("⚠️  LGBM metrics file not found")
    
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
        eval_start = datetime.now()
        model = joblib.load(model_path)
        
        # Load feature columns to ensure alignment
        columns_file = model_path.replace('.joblib', '_columns.json')
        if os.path.exists(columns_file):
            with open(columns_file, 'r', encoding='utf-8') as f:
                model_columns = json.load(f)
            # Align features - use reindex to match model's expected columns
            X_train_aligned = X_train_orig.reindex(columns=model_columns, fill_value=0)
            X_test_aligned = X_test_orig.reindex(columns=model_columns, fill_value=0)
            print(f"   Aligned to {len(model_columns)} model features")
        else:
            print(f"   ⚠️  Column file not found, using current features")
            X_train_aligned = X_train_orig
            X_test_aligned = X_test_orig
        
        train_metrics = evaluate_model(model, X_train_aligned, y_train, "Training (Train+Val Combined)")
        test_metrics = evaluate_model(model, X_test_aligned, y_test, "Test (>= 2025-07-01)")
        
        all_results[model_name] = {
            'train': train_metrics,
            'test': test_metrics
        }
        eval_time = datetime.now() - eval_start
        print(f"✅ {model_name} evaluation complete in {eval_time}")
    
    # Save combined metrics
    print("\n💾 Saving combined metrics...")
    os.makedirs('experiments', exist_ok=True)
    
    # Convert all results to ensure JSON serializable
    all_results_clean = convert_numpy_types(all_results, path="all_results")
    
    with open('experiments/v4_muscular_combined_corr_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(all_results_clean, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved combined metrics to experiments/v4_muscular_combined_corr_metrics.json")
    print(f"   Models in file: {', '.join(sorted(all_results.keys()))}")
    
    # Display comparison table
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON - ALL 4 MODELS")
    print("="*80)
    print(f"\n{'Model':<10} {'Dataset':<30} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC AUC':<12} {'Gini':<12}")
    print("-" * 100)
    
    for model_name in ['RF', 'GB', 'XGB', 'LGBM']:
        if model_name not in all_results:
            print(f"{model_name:<10} {'N/A (not found)':<30} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
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
    
    total_time = datetime.now() - start_time
    print(f"\n✅ Total execution time: {total_time}")
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)

if __name__ == '__main__':
    main()

