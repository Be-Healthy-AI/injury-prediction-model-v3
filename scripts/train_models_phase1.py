#!/usr/bin/env python3
"""
Phase 1 Training Scripts: Stable Features + Increased Regularization
- Uses only stable features (drift < 0.10)
- Increased regularization for RF and GB
- Trains RF, GB, and LR models
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
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix
)

def load_stable_features():
    """Load stable features list (drift < 0.10)"""
    stable_file = 'experiments/stable_features_phase1.json'
    if not os.path.exists(stable_file):
        raise FileNotFoundError(f"Stable features file not found: {stable_file}")
    
    with open(stable_file, 'r') as f:
        stable_features = json.load(f)
    
    return set(stable_features)

def prepare_data(df, drop_week5=False, stable_features_set=None):
    """Prepare data with stable feature filtering"""
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
    
    # Filter to stable features only (after encoding)
    if stable_features_set is not None:
        # Keep only columns that match stable features (exact match or prefix match for encoded features)
        stable_cols = []
        for col in X_encoded.columns:
            # Check if base feature name is in stable set
            base_feature = col.split('_week_')[0] if '_week_' in col else col
            # Remove encoded prefixes (e.g., "position_", "current_club_")
            for prefix in ['position_', 'current_club_', 'previous_club_', 'nationality1_', 'nationality2_', 
                          'current_club_country_', 'previous_club_country_', 'last_match_position_week_']:
                if col.startswith(prefix):
                    base_feature = col.replace(prefix, '')
                    break
            
            # Check if original feature or encoded version is stable
            if col in stable_features_set or base_feature in stable_features_set:
                stable_cols.append(col)
            # Also keep if it's a categorical encoding of a stable feature
            elif any(col.startswith(f"{feat}_") for feat in stable_features_set):
                stable_cols.append(col)
        
        # If we have stable features, filter to them
        if stable_cols:
            X_encoded = X_encoded[[col for col in stable_cols if col in X_encoded.columns]]
    
    return X_encoded, y

def evaluate_model(model, X, y, set_name):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.0,
    }
    metrics['gini'] = 2 * metrics['roc_auc'] - 1
    
    # Confusion matrix
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
            'tn': int((y == 0).sum() - (y_pred[y == 0] == 1).sum()),
            'fp': int((y_pred[y == 0] == 1).sum()),
            'fn': int((y[y_pred == 0] == 1).sum()),
            'tp': int((y[y_pred == 1] == 1).sum())
        }
    
    return metrics, y_pred, y_proba

def train_rf(X_train, y_train, X_val_insample, y_val_insample, X_val_outsample, y_val_outsample):
    """Train Random Forest with increased regularization"""
    print("\n" + "=" * 70)
    print("🚀 TRAINING RANDOM FOREST MODEL (PHASE 1)")
    print("=" * 70)
    
    print("\n🔧 Model hyperparameters (INCREASED REGULARIZATION):")
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,  # Reduced from 25
        min_samples_split=20,  # Increased from 8
        min_samples_leaf=10,  # Increased from 2
        max_features='sqrt',  # Added for regularization
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    print(f"   n_estimators: {rf_model.n_estimators}")
    print(f"   max_depth: {rf_model.max_depth} (reduced from 25)")
    print(f"   min_samples_split: {rf_model.min_samples_split} (increased from 8)")
    print(f"   min_samples_leaf: {rf_model.min_samples_leaf} (increased from 2)")
    print(f"   max_features: {rf_model.max_features} (added)")
    print(f"   class_weight: {rf_model.class_weight}")
    
    print("\n🌳 Training started...")
    train_start = datetime.now()
    rf_model.fit(X_train, y_train)
    train_time = datetime.now() - train_start
    print(f"✅ Training completed in {train_time}")
    
    # Evaluate
    train_metrics, _, _ = evaluate_model(rf_model, X_train, y_train, "Training")
    val_insample_metrics, _, _ = evaluate_model(rf_model, X_val_insample, y_val_insample, "In-Sample")
    val_outsample_metrics, _, _ = evaluate_model(rf_model, X_val_outsample, y_val_outsample, "Out-of-Sample")
    
    return rf_model, train_metrics, val_insample_metrics, val_outsample_metrics

def train_gb(X_train, y_train, X_val_insample, y_val_insample, X_val_outsample, y_val_outsample):
    """Train Gradient Boosting with increased regularization"""
    print("\n" + "=" * 70)
    print("🚀 TRAINING GRADIENT BOOSTING MODEL (PHASE 1)")
    print("=" * 70)
    
    print("\n🔧 Model hyperparameters (INCREASED REGULARIZATION):")
    gb_model = GradientBoostingClassifier(
        n_estimators=200,  # Reduced from 250
        max_depth=10,  # Reduced from 15
        learning_rate=0.1,  # Reduced from 0.15
        min_samples_split=20,  # Increased from 15
        min_samples_leaf=10,  # Increased from 8
        subsample=0.8,  # Reduced from 0.9
        max_features='sqrt',
        random_state=42
    )
    print(f"   n_estimators: {gb_model.n_estimators} (reduced from 250)")
    print(f"   max_depth: {gb_model.max_depth} (reduced from 15)")
    print(f"   learning_rate: {gb_model.learning_rate} (reduced from 0.15)")
    print(f"   min_samples_split: {gb_model.min_samples_split} (increased from 15)")
    print(f"   min_samples_leaf: {gb_model.min_samples_leaf} (increased from 8)")
    print(f"   subsample: {gb_model.subsample} (reduced from 0.9)")
    print(f"   max_features: {gb_model.max_features}")
    
    print("\n🌳 Training started...")
    train_start = datetime.now()
    gb_model.fit(X_train, y_train)
    train_time = datetime.now() - train_start
    print(f"✅ Training completed in {train_time}")
    
    # Evaluate
    train_metrics, _, _ = evaluate_model(gb_model, X_train, y_train, "Training")
    val_insample_metrics, _, _ = evaluate_model(gb_model, X_val_insample, y_val_insample, "In-Sample")
    val_outsample_metrics, _, _ = evaluate_model(gb_model, X_val_outsample, y_val_outsample, "Out-of-Sample")
    
    return gb_model, train_metrics, val_insample_metrics, val_outsample_metrics

def train_lr(X_train, y_train, X_val_insample, y_val_insample, X_val_outsample, y_val_outsample):
    """Train Logistic Regression"""
    print("\n" + "=" * 70)
    print("🚀 TRAINING LOGISTIC REGRESSION MODEL (PHASE 1)")
    print("=" * 70)
    
    print("\n🔧 Model hyperparameters:")
    lr_model = LogisticRegression(
        max_iter=1000,
        C=0.1,  # Increased regularization (lower C = more regularization)
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    print(f"   max_iter: {lr_model.max_iter}")
    print(f"   C: {lr_model.C} (increased regularization)")
    print(f"   class_weight: {lr_model.class_weight}")
    
    print("\n🌳 Training started...")
    train_start = datetime.now()
    lr_model.fit(X_train, y_train)
    train_time = datetime.now() - train_start
    print(f"✅ Training completed in {train_time}")
    
    # Evaluate
    train_metrics, _, _ = evaluate_model(lr_model, X_train, y_train, "Training")
    val_insample_metrics, _, _ = evaluate_model(lr_model, X_val_insample, y_val_insample, "In-Sample")
    val_outsample_metrics, _, _ = evaluate_model(lr_model, X_val_outsample, y_val_outsample, "Out-of-Sample")
    
    return lr_model, train_metrics, val_insample_metrics, val_outsample_metrics

def main():
    print("="*80)
    print("PHASE 1: STABLE FEATURES + INCREASED REGULARIZATION")
    print("="*80)
    print("📋 Strategy:")
    print("   1. Use only stable features (drift < 0.10)")
    print("   2. Increase regularization for RF and GB")
    print("   3. Train RF, GB, and LR models")
    print("="*80)
    
    start_time = datetime.now()
    
    # Load stable features
    print("\n📂 Loading stable features...")
    try:
        stable_features_set = load_stable_features()
        print(f"✅ Loaded {len(stable_features_set)} stable features")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("   Please run scripts/identify_stable_features.py first")
        return
    
    # Load data
    print("\n📂 Loading timeline data...")
    train_file = 'timelines_35day_enhanced_balanced_v4_train.csv'
    val_file = 'timelines_35day_enhanced_balanced_v4_val.csv'
    
    if not os.path.exists(train_file) or not os.path.exists(val_file):
        print(f"❌ Error: Could not find timeline files")
        return
    
    df_train_full = pd.read_csv(train_file, encoding='utf-8-sig')
    df_val_outsample = pd.read_csv(val_file, encoding='utf-8-sig')
    
    print(f"✅ Loaded training set: {len(df_train_full):,} records")
    print(f"✅ Loaded out-of-sample validation set: {len(df_val_outsample):,} records")
    
    # Prepare data with stable features
    print("\n🔧 Preparing data with stable features...")
    X_train_full, y_train_full = prepare_data(df_train_full, drop_week5=False, stable_features_set=stable_features_set)
    X_val_outsample, y_val_outsample = prepare_data(df_val_outsample, drop_week5=False, stable_features_set=stable_features_set)
    
    # Align columns
    all_cols = sorted(set(X_train_full.columns) | set(X_val_outsample.columns))
    X_train_full = X_train_full.reindex(columns=all_cols, fill_value=0)
    X_val_outsample = X_val_outsample.reindex(columns=all_cols, fill_value=0)
    
    print(f"✅ Prepared data: {X_train_full.shape[1]} features (stable features only)")
    
    # Split training set 80/20 for in-sample validation
    X_train, X_val_insample, y_train, y_val_insample = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=y_train_full
    )
    
    print(f"\n📊 Training set (80%): {len(X_train):,} samples, {X_train.shape[1]} features")
    print(f"   Injury ratio: {y_train.mean():.1%}")
    print(f"📊 In-sample validation (20%): {len(X_val_insample):,} samples")
    print(f"   Injury ratio: {y_val_insample.mean():.1%}")
    print(f"📊 Out-of-sample validation: {len(X_val_outsample):,} samples")
    print(f"   Injury ratio: {y_val_outsample.mean():.1%}")
    
    # Train models
    results = {}
    
    # Random Forest
    rf_model, rf_train, rf_insample, rf_outsample = train_rf(
        X_train, y_train, X_val_insample, y_val_insample, X_val_outsample, y_val_outsample
    )
    results['rf'] = {
        'model': rf_model,
        'train': rf_train,
        'insample': rf_insample,
        'outsample': rf_outsample
    }
    
    # Gradient Boosting
    gb_model, gb_train, gb_insample, gb_outsample = train_gb(
        X_train, y_train, X_val_insample, y_val_insample, X_val_outsample, y_val_outsample
    )
    results['gb'] = {
        'model': gb_model,
        'train': gb_train,
        'insample': gb_insample,
        'outsample': gb_outsample
    }
    
    # Logistic Regression
    lr_model, lr_train, lr_insample, lr_outsample = train_lr(
        X_train, y_train, X_val_insample, y_val_insample, X_val_outsample, y_val_outsample
    )
    results['lr'] = {
        'model': lr_model,
        'train': lr_train,
        'insample': lr_insample,
        'outsample': lr_outsample
    }
    
    # Print summary
    print("\n" + "="*80)
    print("📊 PERFORMANCE SUMMARY - PHASE 1")
    print("="*80)
    
    for model_name in ['rf', 'gb', 'lr']:
        print(f"\n{model_name.upper()}:")
        print(f"   Out-of-Sample - Precision: {results[model_name]['outsample']['precision']:.4f}, "
              f"Recall: {results[model_name]['outsample']['recall']:.4f}, "
              f"F1: {results[model_name]['outsample']['f1']:.4f}, "
              f"ROC AUC: {results[model_name]['outsample']['roc_auc']:.4f}")
    
    # Save models and metrics
    print("\n" + "="*80)
    print("💾 SAVING MODELS AND METRICS")
    print("="*80)
    
    os.makedirs('models', exist_ok=True)
    import joblib
    
    for model_name in ['rf', 'gb', 'lr']:
        model = results[model_name]['model']
        model_file = f'models/{model_name}_model_phase1.joblib'
        columns_file = f'models/{model_name}_model_phase1_columns.json'
        metrics_file = f'models/{model_name}_model_phase1_metrics.json'
        
        joblib.dump(model, model_file)
        json.dump(X_train.columns.tolist(), open(columns_file, 'w'))
        
        all_metrics = {
            'train': results[model_name]['train'],
            'validation_insample': results[model_name]['insample'],
            'validation_outsample': results[model_name]['outsample']
        }
        json.dump(all_metrics, open(metrics_file, 'w'), indent=2)
        
        print(f"✅ Saved {model_name.upper()} model to {model_file}")
    
    total_time = datetime.now() - start_time
    print(f"\n⏱️  Total time: {total_time}")
    print("\n🎉 PHASE 1 TRAINING COMPLETED!")

if __name__ == "__main__":
    main()



