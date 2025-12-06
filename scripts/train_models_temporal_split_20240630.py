#!/usr/bin/env python3
"""
Quick test: Train models with data <= 2024-06-30, validate with data > 2024-06-30 and <= 2025-06-30
Uses existing training dataset, just splits it differently
"""
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import joblib
import json
import os
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
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
    print(f"\n🔎 Applying correlation filter (threshold={threshold:.2f})...")
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    kept = [col for col in X.columns if col not in to_drop]
    print(f"   Removed {len(to_drop)} features due to correlation > {threshold}")
    print(f"   Remaining features: {len(kept)}")
    return kept

def evaluate_model(model, X, y, set_name):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y, y_proba),
        'gini': 2 * roc_auc_score(y, y_proba) - 1
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
        metrics['confusion_matrix'] = {
            'tn': int(cm[0, 0]) if len(cm) > 0 else 0,
            'fp': 0,
            'fn': int(np.sum(y)) - int(cm[0, 0]) if len(cm) > 0 else int(np.sum(y)),
            'tp': 0
        }
    
    print(f"\n   {set_name}:")
    print(f"      Accuracy: {metrics['accuracy']:.4f}")
    print(f"      Precision: {metrics['precision']:.4f}")
    print(f"      Recall: {metrics['recall']:.4f}")
    print(f"      F1: {metrics['f1']:.4f}")
    print(f"      ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"      Gini: {metrics['gini']:.4f}")
    
    return metrics

def train_model(model_name, model_class, model_params, X_train, y_train, X_val, y_val, output_dir='models'):
    """Train a single model"""
    print(f"\n{'='*70}")
    print(f"🚀 TRAINING {model_name.upper()}")
    print(f"{'='*70}")
    
    print(f"\n📊 Dataset sizes:")
    print(f"   Training: {len(X_train):,} samples, {X_train.shape[1]} features")
    print(f"   Training injury ratio: {y_train.mean():.1%}")
    print(f"   Validation: {len(X_val):,} samples")
    print(f"   Validation injury ratio: {y_val.mean():.1%}")
    
    # Train model
    print(f"\n🌳 Training {model_name}...")
    model = model_class(**model_params)
    
    train_start = datetime.now()
    model.fit(X_train, y_train)
    train_time = datetime.now() - train_start
    print(f"✅ Training completed in {train_time}")
    
    # Evaluate
    print(f"\n📊 PERFORMANCE METRICS:")
    train_metrics = evaluate_model(model, X_train, y_train, "Training")
    val_metrics = evaluate_model(model, X_val, y_val, "Validation")
    
    # Calculate gaps
    gaps = {metric: train_metrics[metric] - val_metrics[metric] 
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'gini']}
    
    print(f"\n📊 GAPS (Train - Val):")
    print(f"   Accuracy gap: {gaps['accuracy']:.4f}")
    print(f"   Precision gap: {gaps['precision']:.4f}")
    print(f"   Recall gap: {gaps['recall']:.4f}")
    print(f"   F1 gap: {gaps['f1']:.4f}")
    print(f"   ROC AUC gap: {gaps['roc_auc']:.4f}")
    print(f"   Gini gap: {gaps['gini']:.4f}")
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model_file = f'{output_dir}/{model_name}_20240630_split.joblib'
    columns_file = f'{output_dir}/{model_name}_20240630_split_columns.json'
    metrics_file = f'{output_dir}/{model_name}_20240630_split_metrics.json'
    
    joblib.dump(model, model_file)
    json.dump(X_train.columns.tolist(), open(columns_file, 'w'))
    
    all_metrics = {
        'train': train_metrics,
        'validation': val_metrics,
        'gaps': gaps,
        'auc_gap': gaps['roc_auc'],
        'num_features': X_train.shape[1]
    }
    json.dump(all_metrics, open(metrics_file, 'w'), indent=2)
    
    print(f"\n✅ Saved model to {model_file}")
    print(f"✅ Saved columns to {columns_file}")
    print(f"✅ Saved metrics to {metrics_file}")
    
    return model, all_metrics

def main():
    print("="*80)
    print("TEMPORAL SPLIT TEST - 2024-06-30")
    print("="*80)
    print("\nSplit configuration:")
    print("  Training: reference_date <= 2024-06-30")
    print("  Validation: reference_date > 2024-06-30 and <= 2025-06-30")
    
    # Load training data
    train_file = 'timelines_35day_enhanced_balanced_v4_train.csv'
    if not os.path.exists(train_file):
        print(f"❌ Error: {train_file} not found")
        return
    
    print(f"\n📂 Loading training dataset...")
    df_full = pd.read_csv(train_file, encoding='utf-8-sig')
    df_full['reference_date'] = pd.to_datetime(df_full['reference_date'])
    
    print(f"✅ Loaded {len(df_full):,} total records")
    
    # Split by date
    TRAIN_CUTOFF = pd.Timestamp('2024-06-30')
    VAL_END = pd.Timestamp('2025-06-30')
    
    df_train = df_full[df_full['reference_date'] <= TRAIN_CUTOFF].copy()
    df_val = df_full[(df_full['reference_date'] > TRAIN_CUTOFF) & 
                     (df_full['reference_date'] <= VAL_END)].copy()
    
    print(f"\n📊 Dataset split:")
    print(f"   Training (<= 2024-06-30): {len(df_train):,} records")
    print(f"      Injury ratio: {df_train['target'].mean():.1%}")
    print(f"      Date range: {df_train['reference_date'].min()} to {df_train['reference_date'].max()}")
    print(f"   Validation (> 2024-06-30 and <= 2025-06-30): {len(df_val):,} records")
    print(f"      Injury ratio: {df_val['target'].mean():.1%}")
    if len(df_val) > 0:
        print(f"      Date range: {df_val['reference_date'].min()} to {df_val['reference_date'].max()}")
    
    if len(df_train) == 0:
        print("❌ Error: No training data found!")
        return
    
    if len(df_val) == 0:
        print("❌ Error: No validation data found!")
        return
    
    # Prepare data
    print(f"\n🔧 Preparing data...")
    X_train, y_train = prepare_data(df_train, drop_week5=False)
    X_val, y_val = prepare_data(df_val, drop_week5=False)
    
    # Apply correlation filter
    CORR_THRESHOLD = 0.8
    selected_columns = apply_correlation_filter(X_train, CORR_THRESHOLD)
    X_train = X_train[selected_columns]
    X_val = X_val.reindex(columns=selected_columns, fill_value=0)
    
    print(f"   Final feature count: {len(selected_columns)}")
    
    # Train models
    models_config = [
        ('rf', RandomForestClassifier, {
            'n_estimators': 300,
            'max_depth': 25,
            'min_samples_split': 8,
            'min_samples_leaf': 2,
            'random_state': 42,
            'class_weight': 'balanced',
            'n_jobs': -1
        }),
        ('gb', GradientBoostingClassifier, {
            'n_estimators': 250,
            'max_depth': 15,
            'learning_rate': 0.15,
            'min_samples_split': 15,
            'min_samples_leaf': 8,
            'subsample': 0.9,
            'max_features': 'sqrt',
            'random_state': 42
        }),
        ('lr', LogisticRegression, {
            'max_iter': 1000,
            'class_weight': 'balanced',
            'random_state': 42,
            'solver': 'liblinear'
        })
    ]
    
    results = {}
    for model_key, model_class, model_params in models_config:
        model, metrics = train_model(
            model_key, model_class, model_params,
            X_train, y_train,
            X_val, y_val
        )
        results[model_key] = metrics
    
    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY - PERFORMANCE METRICS")
    print(f"{'='*80}")
    
    print(f"\n{'Model':<10} {'Set':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'ROC AUC':<12}")
    print("-" * 80)
    for model_key, metrics in results.items():
        train = metrics['train']
        val = metrics['validation']
        print(f"{model_key.upper():<10} {'Training':<12} {train['precision']:<12.4f} {train['recall']:<12.4f} "
              f"{train['f1']:<12.4f} {train['roc_auc']:<12.4f}")
        print(f"{'':<10} {'Validation':<12} {val['precision']:<12.4f} {val['recall']:<12.4f} "
              f"{val['f1']:<12.4f} {val['roc_auc']:<12.4f}")
        print(f"{'':<10} {'Gap':<12} {metrics['gaps']['precision']:<12.4f} {metrics['gaps']['recall']:<12.4f} "
              f"{metrics['gaps']['f1']:<12.4f} {metrics['gaps']['roc_auc']:<12.4f}")
        print()
    
    # Save summary
    summary_file = 'experiments/temporal_split_20240630_summary.md'
    os.makedirs('experiments', exist_ok=True)
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("# Temporal Split Test - 2024-06-30\n\n")
        f.write("**Date:** 2025-11-27\n")
        f.write("**Split:** Training <= 2024-06-30, Validation > 2024-06-30 and <= 2025-06-30\n\n")
        f.write("## Dataset Characteristics\n\n")
        f.write(f"- **Training:** {len(df_train):,} records, {df_train['target'].mean():.1%} injury ratio\n")
        f.write(f"- **Validation:** {len(df_val):,} records, {df_val['target'].mean():.1%} injury ratio\n\n")
        f.write("## Performance Metrics\n\n")
        f.write("| Model | Set | Precision | Recall | F1-Score | ROC AUC |\n")
        f.write("|-------|-----|-----------|--------|----------|----------|\n")
        for model_key, metrics in results.items():
            train = metrics['train']
            val = metrics['validation']
            f.write(f"| {model_key.upper()} | Training | {train['precision']:.4f} | {train['recall']:.4f} | {train['f1']:.4f} | {train['roc_auc']:.4f} |\n")
            f.write(f"| {model_key.upper()} | Validation | {val['precision']:.4f} | {val['recall']:.4f} | {val['f1']:.4f} | {val['roc_auc']:.4f} |\n")
            f.write(f"| {model_key.upper()} | Gap | {metrics['gaps']['precision']:.4f} | {metrics['gaps']['recall']:.4f} | {metrics['gaps']['f1']:.4f} | {metrics['gaps']['roc_auc']:.4f} |\n")
    
    print(f"✅ Summary saved to {summary_file}")
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()



