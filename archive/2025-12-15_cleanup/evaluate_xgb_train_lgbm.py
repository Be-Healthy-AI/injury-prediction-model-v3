#!/usr/bin/env python3
"""
Manually evaluate XGBoost and add to metrics, then train LightGBM
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
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix
)

def prepare_data(df):
    """Prepare data with basic preprocessing (same as main script)"""
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

def apply_correlation_filter(X, threshold=0.8):
    """Drop one feature from each highly correlated pair"""
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    kept = [col for col in X.columns if col not in to_drop]
    return kept

def evaluate_model(model, X, y, dataset_name):
    """Evaluate model and return metrics (with JSON-serializable types)"""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    # Convert numpy types to Python native types for JSON serialization
    metrics = {
        'accuracy': float(accuracy_score(y, y_pred)),
        'precision': float(precision_score(y, y_pred, zero_division=0)),
        'recall': float(recall_score(y, y_pred, zero_division=0)),
        'f1': float(f1_score(y, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y, y_proba)) if len(np.unique(y)) > 1 else 0.0,
        'gini': float(2 * roc_auc_score(y, y_proba) - 1) if len(np.unique(y)) > 1 else 0.0
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
    
    print(f"\n   {dataset_name}:")
    print(f"      Accuracy: {metrics['accuracy']:.4f}")
    print(f"      Precision: {metrics['precision']:.4f}")
    print(f"      Recall: {metrics['recall']:.4f}")
    print(f"      F1-Score: {metrics['f1']:.4f}")
    print(f"      ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"      Gini: {metrics['gini']:.4f}")
    print(f"      TP: {metrics['confusion_matrix']['tp']}, FP: {metrics['confusion_matrix']['fp']}, "
          f"TN: {metrics['confusion_matrix']['tn']}, FN: {metrics['confusion_matrix']['fn']}")
    
    return metrics

def main():
    print("="*80)
    print("EVALUATING XGBOOST AND TRAINING LIGHTGBM")
    print("="*80)
    
    start_time = datetime.now()
    
    # Load existing metrics
    print("\n📂 Loading existing metrics...")
    metrics_file = 'experiments/v4_muscular_combined_corr_metrics.json'
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            all_results = json.load(f)
        print(f"✅ Loaded existing metrics for: {list(all_results.keys())}")
    else:
        all_results = {}
        print("⚠️  No existing metrics file found, starting fresh")
    
    # Load data
    print("\n📂 Loading timeline data...")
    train_file = 'timelines_35day_enhanced_balanced_v4_muscular_train.csv'
    val_file = 'timelines_35day_enhanced_balanced_v4_muscular_val.csv'
    test_file = 'timelines_35day_enhanced_balanced_v4_muscular_test.csv'
    
    df_train = pd.read_csv(train_file, encoding='utf-8-sig')
    df_val = pd.read_csv(val_file, encoding='utf-8-sig')
    df_test = pd.read_csv(test_file, encoding='utf-8-sig')
    
    print(f"✅ Loaded training set: {len(df_train):,} records")
    print(f"✅ Loaded validation set: {len(df_val):,} records")
    print(f"✅ Loaded test set: {len(df_test):,} records")
    
    # Combine train and validation
    df_train_combined = pd.concat([df_train, df_val], ignore_index=True)
    print(f"✅ Combined training set: {len(df_train_combined):,} records")
    
    # Prepare data
    print("\n📊 Preparing data...")
    X_train, y_train = prepare_data(df_train_combined)
    X_test, y_test = prepare_data(df_test)
    
    # Align features
    print("\n🔧 Aligning features...")
    X_train, X_test = align_features(X_train, X_test)
    
    # Apply correlation filter
    print("\n🔎 Applying correlation filter...")
    CORR_THRESHOLD = 0.8
    selected_features = apply_correlation_filter(X_train, CORR_THRESHOLD)
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]
    print(f"✅ Using {len(selected_features)} features after correlation filtering")
    
    # Evaluate XGBoost model
    print("\n" + "="*70)
    print("📊 EVALUATING XGBOOST MODEL")
    print("="*70)
    
    xgb_model_file = 'models/xgb_model_v4_muscular_combined_corr.joblib'
    if os.path.exists(xgb_model_file):
        print(f"✅ Loading XGBoost model from {xgb_model_file}")
        xgb_model = joblib.load(xgb_model_file)
        
        # Load feature columns to ensure alignment
        xgb_columns_file = 'models/xgb_model_v4_muscular_combined_corr_columns.json'
        if os.path.exists(xgb_columns_file):
            with open(xgb_columns_file, 'r') as f:
                xgb_columns = json.load(f)
            # Align features to match model
            X_train_xgb = X_train.reindex(columns=xgb_columns, fill_value=0)
            X_test_xgb = X_test.reindex(columns=xgb_columns, fill_value=0)
        else:
            X_train_xgb = X_train
            X_test_xgb = X_test
        
        xgb_train_metrics = evaluate_model(xgb_model, X_train_xgb, y_train, "Training (Train+Val Combined)")
        xgb_test_metrics = evaluate_model(xgb_model, X_test_xgb, y_test, "Test (>= 2025-07-01)")
        all_results['XGB'] = {'train': xgb_train_metrics, 'test': xgb_test_metrics}
        print(f"\n✅ XGBoost evaluation complete")
    else:
        print(f"❌ XGBoost model file not found: {xgb_model_file}")
        print("   Skipping XGBoost evaluation")
    
    # Train LightGBM model
    print("\n" + "="*70)
    print("🚀 TRAINING LIGHTGBM MODEL")
    print("="*70)
    
    lgbm_model = LGBMClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    print("\n🔧 Model hyperparameters:")
    print(f"   n_estimators: {lgbm_model.n_estimators}")
    print(f"   max_depth: {lgbm_model.max_depth}")
    print(f"   learning_rate: {lgbm_model.learning_rate}")
    print(f"   min_child_samples: {lgbm_model.min_child_samples}")
    print(f"   subsample: {lgbm_model.subsample}")
    print(f"   colsample_bytree: {lgbm_model.colsample_bytree}")
    print(f"   reg_alpha: {lgbm_model.reg_alpha}")
    print(f"   reg_lambda: {lgbm_model.reg_lambda}")
    print(f"   class_weight: {lgbm_model.class_weight}")
    
    print("\n🌳 Training started...")
    train_start = datetime.now()
    lgbm_model.fit(X_train, y_train)
    train_time = datetime.now() - train_start
    print(f"✅ Training completed in {train_time}")
    
    # Evaluate LightGBM
    lgbm_train_metrics = evaluate_model(lgbm_model, X_train, y_train, "Training (Train+Val Combined)")
    lgbm_test_metrics = evaluate_model(lgbm_model, X_test, y_test, "Test (>= 2025-07-01)")
    all_results['LGBM'] = {'train': lgbm_train_metrics, 'test': lgbm_test_metrics}
    
    # Save LightGBM model
    print("\n💾 Saving LightGBM model...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(lgbm_model, 'models/lgbm_model_v4_muscular_combined_corr.joblib')
    # Convert column names to strings and ensure JSON serializable
    column_names = [str(col) for col in X_train.columns]
    with open('models/lgbm_model_v4_muscular_combined_corr_columns.json', 'w', encoding='utf-8') as f:
        json.dump(column_names, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved LightGBM model to models/lgbm_model_v4_muscular_combined_corr.joblib")
    
    # Save all metrics
    print("\n💾 Saving all metrics...")
    os.makedirs('experiments', exist_ok=True)
    with open('experiments/v4_muscular_combined_corr_metrics.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✅ Saved metrics to experiments/v4_muscular_combined_corr_metrics.json")
    print(f"   Models in metrics: {list(all_results.keys())}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY - ALL MODELS")
    print("="*80)
    print(f"\n{'Model':<10} {'Dataset':<30} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC AUC':<12}")
    print("-" * 100)
    
    for model_name in ['RF', 'GB', 'XGB', 'LGBM']:
        if model_name in all_results:
            results = all_results[model_name]
            print(f"{model_name:<10} {'Training (Train+Val Combined)':<30} {results['train']['precision']:<12.4f} {results['train']['recall']:<12.4f} {results['train']['f1']:<12.4f} {results['train']['roc_auc']:<12.4f}")
            print(f"{'':<10} {'Test (>= 2025-07-01)':<30} {results['test']['precision']:<12.4f} {results['test']['recall']:<12.4f} {results['test']['f1']:<12.4f} {results['test']['roc_auc']:<12.4f}")
            print("-" * 100)
        else:
            print(f"{model_name:<10} {'NOT FOUND':<30}")
    
    total_time = datetime.now() - start_time
    print(f"\n✅ Total execution time: {total_time}")
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)

if __name__ == '__main__':
    main()

