#!/usr/bin/env python3
"""
Retrain models using only stable features (low correlation drift)
to reduce data drift with out-of-sample validation
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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# Stability criteria
MAX_DRIFT = 0.05  # Maximum allowed correlation drift
MIN_TRAIN_CORR = 0.01  # Minimum training correlation (to avoid noise)

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

def get_stable_features(corr_file='experiments/feature_correlation_comparison.csv'):
    """Load stable features based on correlation drift"""
    if not Path(corr_file).exists():
        print(f"❌ Error: Correlation file not found: {corr_file}")
        return None
    
    df_corr = pd.read_csv(corr_file)
    
    # Filter stable features
    stable = df_corr[
        (df_corr['train_val_diff'].abs() < MAX_DRIFT) &
        (df_corr['train_corr'].abs() > MIN_TRAIN_CORR)
    ]
    
    stable_features = stable['feature'].tolist()
    
    print(f"\n📊 Stable Features Selection:")
    print(f"   Total features in correlation file: {len(df_corr)}")
    print(f"   Stable features (drift < {MAX_DRIFT}, train_corr > {MIN_TRAIN_CORR}): {len(stable_features)}")
    print(f"   Top 10 stable features by training correlation:")
    top_stable = stable.nlargest(10, 'train_corr')
    for idx, row in top_stable.iterrows():
        print(f"      {row['feature']}: train={row['train_corr']:.4f}, val={row['val_corr']:.4f}, drift={row['train_val_diff']:.4f}")
    
    return stable_features

def filter_stable_features(X, stable_features):
    """Filter dataframe to only include stable features"""
    # Get columns that match stable features (exact match or prefix match for one-hot encoded)
    available_cols = X.columns.tolist()
    filtered_cols = []
    
    for feat in stable_features:
        # Exact match
        if feat in available_cols:
            filtered_cols.append(feat)
        else:
            # Check for one-hot encoded versions (e.g., "current_club_Bayern" from "current_club")
            matching = [col for col in available_cols if col.startswith(feat + '_')]
            filtered_cols.extend(matching)
    
    # Remove duplicates and ensure all columns exist
    filtered_cols = list(set(filtered_cols))
    filtered_cols = [col for col in filtered_cols if col in available_cols]
    
    return X[filtered_cols]

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

def train_model(model_name, model_class, model_params, X_train, y_train, X_val_insample, y_val_insample, X_val_outsample, y_val_outsample, output_dir='models'):
    """Train a single model"""
    print(f"\n{'='*70}")
    print(f"🚀 TRAINING {model_name.upper()}")
    print(f"{'='*70}")
    
    print(f"\n📊 Dataset sizes:")
    print(f"   Training: {len(X_train):,} samples, {X_train.shape[1]} features")
    print(f"   In-sample validation: {len(X_val_insample):,} samples")
    print(f"   Out-of-sample validation: {len(X_val_outsample):,} samples")
    
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
    insample_metrics = evaluate_model(model, X_val_insample, y_val_insample, "In-sample validation")
    outsample_metrics = evaluate_model(model, X_val_outsample, y_val_outsample, "Out-of-sample validation")
    
    # Calculate gaps
    gaps_insample = {metric: train_metrics[metric] - insample_metrics[metric] 
                     for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'gini']}
    gaps_outsample = {metric: train_metrics[metric] - outsample_metrics[metric]
                      for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'gini']}
    
    print(f"\n📊 GAPS:")
    print(f"   In-sample AUC gap: {gaps_insample['roc_auc']:.4f}")
    print(f"   Out-of-sample AUC gap: {gaps_outsample['roc_auc']:.4f}")
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model_file = f'{output_dir}/{model_name}_stable_features.joblib'
    columns_file = f'{output_dir}/{model_name}_stable_features_columns.json'
    metrics_file = f'{output_dir}/{model_name}_stable_features_metrics.json'
    
    joblib.dump(model, model_file)
    json.dump(X_train.columns.tolist(), open(columns_file, 'w'))
    
    all_metrics = {
        'train': train_metrics,
        'validation_insample': insample_metrics,
        'validation_outsample': outsample_metrics,
        'gaps_insample': gaps_insample,
        'gaps_outsample': gaps_outsample,
        'auc_gap_insample': gaps_insample['roc_auc'],
        'auc_gap_outsample': gaps_outsample['roc_auc'],
        'num_features': X_train.shape[1],
        'stable_features_criteria': {
            'max_drift': MAX_DRIFT,
            'min_train_corr': MIN_TRAIN_CORR
        }
    }
    json.dump(all_metrics, open(metrics_file, 'w'), indent=2)
    
    print(f"\n✅ Saved model to {model_file}")
    print(f"✅ Saved columns to {columns_file}")
    print(f"✅ Saved metrics to {metrics_file}")
    
    return model, all_metrics

def main():
    print("="*80)
    print("TRAINING MODELS WITH STABLE FEATURES ONLY")
    print("="*80)
    print(f"\nStability criteria:")
    print(f"   Maximum correlation drift: {MAX_DRIFT}")
    print(f"   Minimum training correlation: {MIN_TRAIN_CORR}")
    
    # Load stable features
    stable_features = get_stable_features()
    if stable_features is None:
        return
    
    # Load data
    print(f"\n📂 Loading datasets...")
    train_file = 'timelines_35day_enhanced_balanced_v4_train.csv'
    val_file = 'timelines_35day_enhanced_balanced_v4_val.csv'
    
    df_train = pd.read_csv(train_file, encoding='utf-8-sig')
    df_val = pd.read_csv(val_file, encoding='utf-8-sig')
    
    print(f"✅ Training: {len(df_train):,} records")
    print(f"✅ Validation: {len(df_val):,} records")
    
    # Prepare data
    print(f"\n🔧 Preparing data...")
    X_train_full, y_train_full = prepare_data(df_train, drop_week5=False)
    X_val_outsample, y_val_outsample = prepare_data(df_val, drop_week5=False)
    
    # Filter to stable features
    print(f"\n🔍 Filtering to stable features...")
    X_train_full = filter_stable_features(X_train_full, stable_features)
    X_val_outsample = filter_stable_features(X_val_outsample, stable_features)
    
    print(f"   Training features after filtering: {X_train_full.shape[1]}")
    print(f"   Validation features after filtering: {X_val_outsample.shape[1]}")
    
    # Ensure same columns
    all_cols = sorted(set(X_train_full.columns) | set(X_val_outsample.columns))
    X_train_full = X_train_full.reindex(columns=all_cols, fill_value=0)
    X_val_outsample = X_val_outsample.reindex(columns=all_cols, fill_value=0)
    
    print(f"   Final feature count: {len(all_cols)}")
    
    # Split training
    X_train, X_val_insample, y_train, y_val_insample = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=y_train_full
    )
    
    # Ensure same columns
    X_train = X_train.reindex(columns=all_cols, fill_value=0)
    X_val_insample = X_val_insample.reindex(columns=all_cols, fill_value=0)
    X_val_outsample = X_val_outsample.reindex(columns=all_cols, fill_value=0)
    
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
            X_val_insample, y_val_insample,
            X_val_outsample, y_val_outsample
        )
        results[model_key] = metrics
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY - OUT-OF-SAMPLE PERFORMANCE")
    print(f"{'='*80}")
    print(f"\n{'Model':<10} {'Precision':<12} {'Recall':<12} {'F1':<12} {'ROC AUC':<12} {'AUC Gap':<12}")
    print("-" * 80)
    for model_key, metrics in results.items():
        outsample = metrics['validation_outsample']
        gap = metrics['auc_gap_outsample']
        print(f"{model_key.upper():<10} {outsample['precision']:<12.4f} {outsample['recall']:<12.4f} "
              f"{outsample['f1']:<12.4f} {outsample['roc_auc']:<12.4f} {gap:<12.4f}")
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()



