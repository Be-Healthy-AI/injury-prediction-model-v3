#!/usr/bin/env python3
"""
Gradient Boosting Training Script V4 with Temporal Split
- Temporal train/validation split (train <= 2025-06-30, val >= 2025-07-01)
- Comprehensive validation metrics
- Overfitting analysis
"""
import sys
import io
# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import argparse
import os
import joblib
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)

def prepare_data(df, drop_week5=True):
    """Prepare data with same encoding logic as V3, optionally excluding week_5 features"""
    feature_columns = [
        col for col in df.columns
                      if col not in ['player_id', 'reference_date', 'player_name', 'target']
        and (drop_week5 is False or '_week_5' not in col)
    ]
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

def apply_correlation_filter(X, threshold):
    """Drop one feature from each highly correlated pair."""
    print(f"\n🔎 Applying correlation filter (threshold={threshold:.2f}) on training data...")
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
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'roc_auc': roc_auc_score(y, y_proba),
    }
    metrics['gini'] = 2 * metrics['roc_auc'] - 1
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    metrics['confusion_matrix'] = {
        'tn': int(cm[0, 0]),
        'fp': int(cm[0, 1]),
        'fn': int(cm[1, 0]),
        'tp': int(cm[1, 1])
    }
    
    return metrics, y_pred, y_proba

def main():
    parser = argparse.ArgumentParser(description="Train Gradient Boosting V4 with optional feature controls.")
    parser.add_argument("--keep-week5", action="store_true", help="Retain week_5 features in training.")
    parser.add_argument("--corr-threshold", type=float, default=None,
                        help="If provided, drop features with |corr| above this threshold (evaluated on training data).")
    args = parser.parse_args()

    print("🚀 GRADIENT BOOSTING TRAINING V4 (DUAL VALIDATION)")
    print("=" * 70)
    if args.keep_week5:
        print("📋 Features: Enhanced features with full 35-day windows (week_5 included)")
    else:
        print("📋 Features: Enhanced features with 28-day windows (excluding week_5)")
    print("📊 Validation: In-sample (80/20) + Out-of-sample (temporal)")
    print("=" * 70)
    
    start_time = datetime.now()
    
    # Load training and validation timelines
    print("\n📂 Loading V4 timelines data...")
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
    
    df_train_full = pd.read_csv(train_file, encoding='utf-8-sig')
    df_val_outsample = pd.read_csv(val_file, encoding='utf-8-sig')
    
    print(f"✅ Loaded training set: {len(df_train_full):,} records")
    print(f"✅ Loaded out-of-sample validation set: {len(df_val_outsample):,} records")
    
    drop_week5 = not args.keep_week5
    print("\n🔧 Preparing data...")
    X_train_full, y_train_full, _ = prepare_data(df_train_full, drop_week5=drop_week5)
    X_val_outsample, y_val_outsample, _ = prepare_data(df_val_outsample, drop_week5=drop_week5)

    if args.corr_threshold is not None:
        selected_columns = apply_correlation_filter(X_train_full, args.corr_threshold)
        X_train_full = X_train_full[selected_columns]
        X_val_outsample = X_val_outsample.reindex(columns=selected_columns, fill_value=0)
    else:
        selected_columns = X_train_full.columns.tolist()
    
    # Split training set 80/20 for in-sample validation
    X_train, X_val_insample, y_train, y_val_insample = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=y_train_full
    )
    
    # Ensure same columns (align)
    X_train = X_train.reindex(columns=selected_columns, fill_value=0)
    X_val_insample = X_val_insample.reindex(columns=selected_columns, fill_value=0)
    X_val_outsample = X_val_outsample.reindex(columns=selected_columns, fill_value=0)
    
    print(f"📊 Training set (80%): {len(X_train):,} samples, {X_train.shape[1]} features")
    print(f"   Injury ratio: {y_train.mean():.1%}")
    print(f"📊 In-sample validation (20%): {len(X_val_insample):,} samples")
    print(f"   Injury ratio: {y_val_insample.mean():.1%}")
    print(f"📊 Out-of-sample validation: {len(X_val_outsample):,} samples")
    print(f"   Injury ratio: {y_val_outsample.mean():.1%}")
    
    # Train model
    print("\n" + "=" * 70)
    print("🚀 TRAINING GRADIENT BOOSTING MODEL")
    print("=" * 70)
    
    print("\n🔧 Model hyperparameters:")
    gb_model = GradientBoostingClassifier(
        n_estimators=250,
        max_depth=15,
        learning_rate=0.15,
        min_samples_split=15,
        min_samples_leaf=8,
        subsample=0.9,
        max_features='sqrt',
        random_state=42
    )
    print(f"   n_estimators: {gb_model.n_estimators}")
    print(f"   max_depth: {gb_model.max_depth}")
    print(f"   learning_rate: {gb_model.learning_rate}")
    print(f"   min_samples_split: {gb_model.min_samples_split}")
    print(f"   min_samples_leaf: {gb_model.min_samples_leaf}")
    print(f"   subsample: {gb_model.subsample}")
    print(f"   max_features: {gb_model.max_features}")
    
    print("\n🌳 Training started...")
    train_start = datetime.now()
    gb_model.fit(X_train, y_train)
    train_time = datetime.now() - train_start
    print(f"✅ Training completed in {train_time}")
    
    # Evaluate
    print("\n" + "=" * 70)
    print("📊 TRAINING SET PERFORMANCE")
    print("=" * 70)
    train_metrics, _, _ = evaluate_model(gb_model, X_train, y_train, "Training")
    for metric, value in train_metrics.items():
        if metric != 'confusion_matrix':
            print(f"   {metric.capitalize()}: {value:.4f}")
    
    print("\n" + "=" * 70)
    print("📊 IN-SAMPLE VALIDATION PERFORMANCE (80/20 split)")
    print("=" * 70)
    val_insample_metrics, _, _ = evaluate_model(gb_model, X_val_insample, y_val_insample, "In-Sample Validation")
    for metric, value in val_insample_metrics.items():
        if metric != 'confusion_matrix':
            print(f"   {metric.capitalize()}: {value:.4f}")
    
    print("\n" + "=" * 70)
    print("📊 OUT-OF-SAMPLE VALIDATION PERFORMANCE (temporal split)")
    print("=" * 70)
    val_outsample_metrics, _, _ = evaluate_model(gb_model, X_val_outsample, y_val_outsample, "Out-of-Sample Validation")
    for metric, value in val_outsample_metrics.items():
        if metric != 'confusion_matrix':
            print(f"   {metric.capitalize()}: {value:.4f}")
    
    # Overfitting analysis (in-sample)
    print("\n" + "=" * 70)
    print("🔍 OVERFITTING ANALYSIS (In-Sample)")
    print("=" * 70)
    
    gaps_insample = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'gini']:
        gaps_insample[metric] = train_metrics[metric] - val_insample_metrics[metric]
        print(f"   {metric.capitalize()} gap: {gaps_insample[metric]:.4f}")
    
    auc_gap_insample = gaps_insample['roc_auc']
    if auc_gap_insample < 0.01:
        risk_level_insample = "LOW"
        risk_emoji_insample = "✅"
    elif auc_gap_insample < 0.05:
        risk_level_insample = "MODERATE"
        risk_emoji_insample = "⚠️"
    else:
        risk_level_insample = "HIGH"
        risk_emoji_insample = "❌"
    
    print(f"\n{risk_emoji_insample} Overfitting Risk (In-Sample): {risk_level_insample} (AUC gap: {auc_gap_insample:.4f})")
    
    # Out-of-sample gap
    gaps_outsample = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'gini']:
        gaps_outsample[metric] = train_metrics[metric] - val_outsample_metrics[metric]
    
    auc_gap_outsample = gaps_outsample['roc_auc']
    print(f"\n📊 Out-of-Sample AUC Gap: {auc_gap_outsample:.4f}")
    
    # Save model and metrics
    print("\n" + "=" * 70)
    print("💾 SAVING MODEL AND METRICS")
    print("=" * 70)
    
    os.makedirs('models', exist_ok=True)
    
    model_file = 'models/gb_model_v4.joblib'
    columns_file = 'models/gb_model_v4_columns.json'
    metrics_file = 'models/gb_model_v4_metrics.json'
    
    joblib.dump(gb_model, model_file)
    json.dump(X_train.columns.tolist(), open(columns_file, 'w'))
    
    all_metrics = {
        'train': train_metrics,
        'validation_insample': val_insample_metrics,
        'validation_outsample': val_outsample_metrics,
        'gaps_insample': gaps_insample,
        'gaps_outsample': gaps_outsample,
        'overfitting_risk_insample': risk_level_insample,
        'auc_gap_insample': auc_gap_insample,
        'auc_gap_outsample': auc_gap_outsample
    }
    json.dump(all_metrics, open(metrics_file, 'w'), indent=2)
    
    print(f"✅ Saved model to {model_file}")
    print(f"✅ Saved columns to {columns_file}")
    print(f"✅ Saved metrics to {metrics_file}")
    
    total_time = datetime.now() - start_time
    print(f"\n⏱️  Total time: {total_time}")
    print("\n🎉 TRAINING COMPLETED!")

if __name__ == "__main__":
    main()

