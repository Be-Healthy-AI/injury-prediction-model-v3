#!/usr/bin/env python3
"""
Seasonal Temporal Split Training
- Training: <= 2024-06-30 (all seasons before 2024/25)
- Validation: > 2024-06-30 AND <= 2025-06-30 (2024/25 season)
- Test: >= 2025-07-01 (2025/26 season)
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
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix
)
from pathlib import Path

def load_low_shift_features():
    """Load low-shift features from adversarial validation"""
    results_file = 'experiments/adversarial_validation_results.json'
    if not os.path.exists(results_file):
        print(f"⚠️  Warning: Adversarial validation results not found: {results_file}")
        print("   Proceeding without feature filtering...")
        return None
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return set(results['low_shift_features'])

def load_sample_weights():
    """Load sample weights from covariate shift correction"""
    weights_file = 'experiments/covariate_shift_weights.npy'
    if not os.path.exists(weights_file):
        print(f"⚠️  Warning: Sample weights not found: {weights_file}")
        print("   Proceeding without sample weights...")
        return None
    
    weights = np.load(weights_file)
    return weights

def prepare_data(df, low_shift_features_set=None):
    """Prepare data with low-shift feature filtering (same as Phase 2)"""
    feature_columns = [
        col for col in df.columns
        if col not in ['player_id', 'reference_date', 'player_name', 'target']
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
    
    # Filter to low-shift features only (after encoding)
    if low_shift_features_set is not None:
        low_shift_cols = []
        for col in X_encoded.columns:
            base_feature = col.split('_week_')[0] if '_week_' in col else col
            for prefix in ['position_', 'current_club_', 'previous_club_', 'nationality1_', 'nationality2_', 
                          'current_club_country_', 'previous_club_country_', 'last_match_position_week_']:
                if col.startswith(prefix):
                    base_feature = col.replace(prefix, '')
                    break
            
            if col in low_shift_features_set or base_feature in low_shift_features_set:
                low_shift_cols.append(col)
            elif any(col.startswith(f"{feat}_") for feat in low_shift_features_set):
                low_shift_cols.append(col)
        
        if low_shift_cols:
            X_encoded = X_encoded[[col for col in X_encoded.columns if col in low_shift_cols]]
    
    return X_encoded, y

def align_features(X_train, X_val, X_test):
    """Ensure all datasets have the same features"""
    # Get common features
    common_features = list(set(X_train.columns) & set(X_val.columns) & set(X_test.columns))
    
    # Sort for consistency
    common_features = sorted(common_features)
    
    print(f"   Aligning features: {len(common_features)} common features")
    print(f"   Training: {X_train.shape[1]} -> {len(common_features)}")
    print(f"   Validation: {X_val.shape[1]} -> {len(common_features)}")
    print(f"   Test: {X_test.shape[1]} -> {len(common_features)}")
    
    return X_train[common_features], X_val[common_features], X_test[common_features]

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
        # Handle edge case
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

def train_rf(X_train, y_train, sample_weights, X_val, y_val, X_test, y_test):
    """Train Random Forest model"""
    print("\n" + "=" * 70)
    print("🚀 TRAINING RANDOM FOREST MODEL")
    print("=" * 70)
    
    base_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1,
        max_features='sqrt'
    )
    
    # Calibrate model
    rf_model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
    
    print("\n🌳 Training started...")
    train_start = datetime.now()
    if sample_weights is not None:
        rf_model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        rf_model.fit(X_train, y_train)
    train_time = datetime.now() - train_start
    print(f"✅ Training completed in {train_time}")
    
    # Evaluate
    train_metrics = evaluate_model(rf_model, X_train, y_train, "Training")
    val_metrics = evaluate_model(rf_model, X_val, y_val, "Validation (2024/25)") if X_val is not None else None
    test_metrics = evaluate_model(rf_model, X_test, y_test, "Test (2025/26)")
    
    return rf_model, train_metrics, val_metrics, test_metrics

def train_gb(X_train, y_train, sample_weights, X_val, y_val, X_test, y_test):
    """Train Gradient Boosting model"""
    print("\n" + "=" * 70)
    print("🚀 TRAINING GRADIENT BOOSTING MODEL")
    print("=" * 70)
    
    base_model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.8,
        max_features='sqrt',
        random_state=42
    )
    
    # Calibrate model
    gb_model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
    
    print("\n🌳 Training started...")
    train_start = datetime.now()
    if sample_weights is not None:
        gb_model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        gb_model.fit(X_train, y_train)
    train_time = datetime.now() - train_start
    print(f"✅ Training completed in {train_time}")
    
    # Evaluate
    train_metrics = evaluate_model(gb_model, X_train, y_train, "Training")
    val_metrics = evaluate_model(gb_model, X_val, y_val, "Validation (2024/25)") if X_val is not None else None
    test_metrics = evaluate_model(gb_model, X_test, y_test, "Test (2025/26)")
    
    return gb_model, train_metrics, val_metrics, test_metrics

def train_lr(X_train, y_train, sample_weights, X_val, y_val, X_test, y_test):
    """Train Logistic Regression model"""
    print("\n" + "=" * 70)
    print("🚀 TRAINING LOGISTIC REGRESSION MODEL")
    print("=" * 70)
    
    base_model = LogisticRegression(
        max_iter=2000,
        C=0.1,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # Calibrate model
    lr_model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
    
    print("\n🌳 Training started...")
    train_start = datetime.now()
    if sample_weights is not None:
        lr_model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        lr_model.fit(X_train, y_train)
    train_time = datetime.now() - train_start
    print(f"✅ Training completed in {train_time}")
    
    # Evaluate
    train_metrics = evaluate_model(lr_model, X_train, y_train, "Training")
    val_metrics = evaluate_model(lr_model, X_val, y_val, "Validation (2024/25)") if X_val is not None else None
    test_metrics = evaluate_model(lr_model, X_test, y_test, "Test (2025/26)")
    
    return lr_model, train_metrics, val_metrics, test_metrics

def main():
    print("="*80)
    print("SEASONAL TEMPORAL SPLIT TRAINING")
    print("="*80)
    print("\n📋 Split Configuration:")
    print("   Training: <= 2024-06-30 (all seasons before 2024/25)")
    print("   Validation: > 2024-06-30 AND <= 2025-06-30 (2024/25 season)")
    print("   Test: >= 2025-07-01 (2025/26 season)")
    print("="*80)
    
    start_time = datetime.now()
    
    # Load low-shift features (optional)
    print("\n📂 Loading low-shift features...")
    low_shift_features_set = load_low_shift_features()
    if low_shift_features_set:
        print(f"✅ Loaded {len(low_shift_features_set)} low-shift features")
    else:
        print("   Using all features")
    
    # Load sample weights (optional)
    print("\n📂 Loading sample weights...")
    sample_weights = load_sample_weights()
    if sample_weights is not None:
        print(f"✅ Loaded sample weights for {len(sample_weights):,} samples")
    else:
        print("   Proceeding without sample weights")
    
    # Load data
    print("\n📂 Loading timeline data...")
    train_file = 'timelines_35day_enhanced_balanced_v4_train.csv'
    val_file = 'timelines_35day_enhanced_balanced_v4_val.csv'
    
    df_train_full = pd.read_csv(train_file, encoding='utf-8-sig')
    df_train_full['reference_date'] = pd.to_datetime(df_train_full['reference_date'])
    
    df_test = pd.read_csv(val_file, encoding='utf-8-sig')
    df_test['reference_date'] = pd.to_datetime(df_test['reference_date'])
    
    print(f"✅ Training data: {len(df_train_full):,} records")
    print(f"✅ Test data: {len(df_test):,} records")
    
    # Temporal split
    TRAIN_CUTOFF = pd.Timestamp('2024-06-30')
    VAL_END = pd.Timestamp('2025-06-30')
    TEST_START = pd.Timestamp('2025-07-01')
    
    df_train = df_train_full[df_train_full['reference_date'] <= TRAIN_CUTOFF].copy()
    df_val = df_train_full[
        (df_train_full['reference_date'] > TRAIN_CUTOFF) & 
        (df_train_full['reference_date'] <= VAL_END)
    ].copy()
    df_test = df_test[df_test['reference_date'] >= TEST_START].copy()
    
    print(f"\n📊 Dataset Split:")
    print(f"   Training (<= 2024-06-30): {len(df_train):,} records")
    print(f"      Injury ratio: {df_train['target'].mean():.1%}")
    print(f"      Date range: {df_train['reference_date'].min().date()} to {df_train['reference_date'].max().date()}")
    print(f"   Validation (2024/25 season): {len(df_val):,} records")
    print(f"      Injury ratio: {df_val['target'].mean():.1%}")
    if len(df_val) > 0:
        print(f"      Date range: {df_val['reference_date'].min().date()} to {df_val['reference_date'].max().date()}")
    print(f"   Test (2025/26 season): {len(df_test):,} records")
    print(f"      Injury ratio: {df_test['target'].mean():.1%}")
    if len(df_test) > 0:
        print(f"      Date range: {df_test['reference_date'].min().date()} to {df_test['reference_date'].max().date()}")
    
    if len(df_train) == 0:
        print("❌ Error: No training data found!")
        return
    
    if len(df_val) == 0:
        print("⚠️  Warning: No validation data found in training file!")
        print("   This might mean all validation data is in the test file")
    
    if len(df_test) == 0:
        print("❌ Error: No test data found!")
        return
    
    # Prepare data
    print("\n🔧 Preparing data...")
    X_train, y_train = prepare_data(df_train, low_shift_features_set)
    X_val, y_val = prepare_data(df_val, low_shift_features_set) if len(df_val) > 0 else (None, None)
    X_test, y_test = prepare_data(df_test, low_shift_features_set)
    
    print(f"✅ Training features: {X_train.shape[1]}")
    if X_val is not None:
        print(f"✅ Validation features: {X_val.shape[1]}")
    print(f"✅ Test features: {X_test.shape[1]}")
    
    # Align features
    print("\n🔧 Aligning features across datasets...")
    if X_val is not None:
        X_train, X_val, X_test = align_features(X_train, X_val, X_test)
    else:
        # Just align train and test
        common_features = sorted(list(set(X_train.columns) & set(X_test.columns)))
        X_train = X_train[common_features]
        X_test = X_test[common_features]
        print(f"   Aligned to {len(common_features)} common features")
    
    # Adjust sample weights if needed
    if sample_weights is not None:
        if len(sample_weights) > len(X_train):
            sample_weights = sample_weights[:len(X_train)]
        elif len(sample_weights) < len(X_train):
            print(f"⚠️  Warning: Sample weights length ({len(sample_weights)}) < training data length ({len(X_train)})")
            print("   Using weights for available samples, padding with 1.0 for rest")
            padded_weights = np.ones(len(X_train))
            padded_weights[:len(sample_weights)] = sample_weights
            sample_weights = padded_weights
    
    # Train models
    print("\n" + "="*80)
    print("TRAINING MODELS")
    print("="*80)
    
    all_results = {}
    
    # Train RF
    rf_model, rf_train_metrics, rf_val_metrics, rf_test_metrics = train_rf(
        X_train, y_train, sample_weights,
        X_val, y_val,
        X_test, y_test
    )
    
    all_results['RF'] = {
        'train': rf_train_metrics,
        'validation': rf_val_metrics if X_val is not None else None,
        'test': rf_test_metrics
    }
    
    # Save RF model
    os.makedirs('models', exist_ok=True)
    joblib.dump(rf_model, 'models/rf_model_seasonal_split.joblib')
    with open('models/rf_model_seasonal_split_columns.json', 'w') as f:
        json.dump(list(X_train.columns), f, indent=2)
    print(f"\n✅ Saved RF model to models/rf_model_seasonal_split.joblib")
    
    # Train GB
    gb_model, gb_train_metrics, gb_val_metrics, gb_test_metrics = train_gb(
        X_train, y_train, sample_weights,
        X_val, y_val,
        X_test, y_test
    )
    
    all_results['GB'] = {
        'train': gb_train_metrics,
        'validation': gb_val_metrics if X_val is not None else None,
        'test': gb_test_metrics
    }
    
    # Save GB model
    joblib.dump(gb_model, 'models/gb_model_seasonal_split.joblib')
    with open('models/gb_model_seasonal_split_columns.json', 'w') as f:
        json.dump(list(X_train.columns), f, indent=2)
    print(f"\n✅ Saved GB model to models/gb_model_seasonal_split.joblib")
    
    # Train LR
    lr_model, lr_train_metrics, lr_val_metrics, lr_test_metrics = train_lr(
        X_train, y_train, sample_weights,
        X_val, y_val,
        X_test, y_test
    )
    
    all_results['LR'] = {
        'train': lr_train_metrics,
        'validation': lr_val_metrics if X_val is not None else None,
        'test': lr_test_metrics
    }
    
    # Save LR model
    joblib.dump(lr_model, 'models/lr_model_seasonal_split.joblib')
    with open('models/lr_model_seasonal_split_columns.json', 'w') as f:
        json.dump(list(X_train.columns), f, indent=2)
    print(f"\n✅ Saved LR model to models/lr_model_seasonal_split.joblib")
    
    # Save metrics
    with open('experiments/seasonal_split_metrics.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Saved metrics to experiments/seasonal_split_metrics.json")
    
    # Create summary report
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    summary_lines = [
        "# Seasonal Temporal Split - Model Performance",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Dataset Split",
        "",
        f"- **Training:** <= 2024-06-30 ({len(df_train):,} records, {df_train['target'].mean():.1%} injury ratio)",
        f"- **Validation:** 2024/25 season ({len(df_val):,} records, {df_val['target'].mean():.1%} injury ratio)" if len(df_val) > 0 else "- **Validation:** No data in training file",
        f"- **Test:** 2025/26 season ({len(df_test):,} records, {df_test['target'].mean():.1%} injury ratio)",
        "",
        "## Performance Metrics",
        ""
    ]
    
    # Create comparison table
    summary_lines.append("| Model | Dataset | Precision | Recall | F1-Score | ROC AUC | Gini |")
    summary_lines.append("|-------|---------|-----------|--------|----------|---------|------|")
    
    for model_name in ['RF', 'GB', 'LR']:
        results = all_results[model_name]
        summary_lines.append(f"| **{model_name}** | Training | {results['train']['precision']:.4f} | {results['train']['recall']:.4f} | {results['train']['f1']:.4f} | {results['train']['roc_auc']:.4f} | {results['train']['gini']:.4f} |")
        if results['validation']:
            summary_lines.append(f"| | Validation (2024/25) | {results['validation']['precision']:.4f} | {results['validation']['recall']:.4f} | {results['validation']['f1']:.4f} | {results['validation']['roc_auc']:.4f} | {results['validation']['gini']:.4f} |")
        summary_lines.append(f"| | Test (2025/26) | {results['test']['precision']:.4f} | {results['test']['recall']:.4f} | {results['test']['f1']:.4f} | {results['test']['roc_auc']:.4f} | {results['test']['gini']:.4f} |")
        summary_lines.append("| | | | | | | |")
    
    # Add gap analysis
    summary_lines.extend([
        "",
        "## Performance Gaps",
        ""
    ])
    
    for model_name in ['RF', 'GB', 'LR']:
        results = all_results[model_name]
        train_f1 = results['train']['f1']
        test_f1 = results['test']['f1']
        gap = train_f1 - test_f1
        
        summary_lines.append(f"### {model_name}")
        summary_lines.append(f"- **F1 Gap (Train → Test):** {gap:.4f} ({gap/train_f1*100:.1f}% relative)")
        if results['validation']:
            val_f1 = results['validation']['f1']
            val_gap = train_f1 - val_f1
            test_gap_from_val = val_f1 - test_f1
            summary_lines.append(f"- **F1 Gap (Train → Validation):** {val_gap:.4f} ({val_gap/train_f1*100:.1f}% relative)")
            summary_lines.append(f"- **F1 Gap (Validation → Test):** {test_gap_from_val:.4f} ({test_gap_from_val/val_f1*100:.1f}% relative)" if val_f1 > 0 else "- **F1 Gap (Validation → Test):** N/A")
        summary_lines.append("")
    
    # Save summary
    summary_text = "\n".join(summary_lines)
    with open('experiments/seasonal_split_summary.md', 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    print(summary_text)
    
    total_time = datetime.now() - start_time
    print(f"\n✅ Complete! Total time: {total_time}")
    print(f"✅ Summary saved to: experiments/seasonal_split_summary.md")

if __name__ == '__main__':
    main()

