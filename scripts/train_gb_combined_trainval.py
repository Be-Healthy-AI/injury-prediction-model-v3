#!/usr/bin/env python3
"""
Train GB model on combined training + validation datasets
Test on test dataset with threshold 0.5
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
from sklearn.ensemble import GradientBoostingClassifier
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

def align_features(X_trainval, X_test):
    """Ensure both datasets have the same features"""
    common_features = list(set(X_trainval.columns) & set(X_test.columns))
    common_features = sorted(common_features)
    
    print(f"   Aligning features: {len(common_features)} common features")
    print(f"   Train+Val: {X_trainval.shape[1]} -> {len(common_features)}")
    print(f"   Test: {X_test.shape[1]} -> {len(common_features)}")
    
    return X_trainval[common_features], X_test[common_features]

def evaluate_model(model, X, y, dataset_name, threshold=0.5):
    """Evaluate model and return metrics"""
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
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
    
    print(f"\n   {dataset_name} (Threshold: {threshold}):")
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
    print("TRAINING GB MODEL - COMBINED TRAINING + VALIDATION DATASETS")
    print("="*80)
    print("\n📋 Configuration:")
    print("   Training: Combined train + validation datasets")
    print("   Testing: Test dataset (2025/26 season)")
    print("   Threshold: 0.5 (Precision target: ~41.56%)")
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
        print("   ⚠️  Note: Sample weights were calculated for original training set only")
        print("   Will pad with 1.0 for validation samples")
    else:
        print("   Proceeding without sample weights")
    
    # Load data
    print("\n📂 Loading timeline data...")
    train_file = 'timelines_35day_enhanced_balanced_v4_train.csv'
    val_file = 'timelines_35day_enhanced_balanced_v4_val.csv'
    test_file = 'timelines_35day_enhanced_balanced_v4_test.csv'
    
    df_train = pd.read_csv(train_file, encoding='utf-8-sig')
    df_train['reference_date'] = pd.to_datetime(df_train['reference_date'])
    
    df_val = pd.read_csv(val_file, encoding='utf-8-sig')
    df_val['reference_date'] = pd.to_datetime(df_val['reference_date'])
    
    df_test = pd.read_csv(test_file, encoding='utf-8-sig')
    df_test['reference_date'] = pd.to_datetime(df_test['reference_date'])
    
    print(f"✅ Training data: {len(df_train):,} records")
    print(f"   Injury ratio: {df_train['target'].mean():.1%}")
    print(f"✅ Validation data: {len(df_val):,} records")
    print(f"   Injury ratio: {df_val['target'].mean():.1%}")
    print(f"✅ Test data: {len(df_test):,} records")
    print(f"   Injury ratio: {df_test['target'].mean():.1%}")
    
    # Combine training and validation datasets
    print("\n🔗 Combining training and validation datasets...")
    df_trainval = pd.concat([df_train, df_val], ignore_index=True)
    print(f"✅ Combined dataset: {len(df_trainval):,} records")
    print(f"   Injury ratio: {df_trainval['target'].mean():.1%}")
    print(f"   Date range: {df_trainval['reference_date'].min().date()} to {df_trainval['reference_date'].max().date()}")
    
    # Prepare data
    print("\n🔧 Preparing data...")
    X_trainval, y_trainval = prepare_data(df_trainval, low_shift_features_set)
    X_test, y_test = prepare_data(df_test, low_shift_features_set)
    
    print(f"✅ Train+Val features: {X_trainval.shape[1]}")
    print(f"✅ Test features: {X_test.shape[1]}")
    
    # Align features
    print("\n🔧 Aligning features across datasets...")
    X_trainval, X_test = align_features(X_trainval, X_test)
    
    # Adjust sample weights if needed
    if sample_weights is not None:
        original_train_size = len(df_train)
        if len(sample_weights) > original_train_size:
            sample_weights = sample_weights[:original_train_size]
        
        # Pad with 1.0 for validation samples
        if len(sample_weights) < len(X_trainval):
            print(f"   Padding sample weights: {len(sample_weights)} -> {len(X_trainval)}")
            padded_weights = np.ones(len(X_trainval))
            padded_weights[:len(sample_weights)] = sample_weights
            sample_weights = padded_weights
    
    # Train GB model
    print("\n" + "="*80)
    print("TRAINING GRADIENT BOOSTING MODEL")
    print("="*80)
    
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
    if sample_weights is not None and len(sample_weights) == len(X_trainval):
        gb_model.fit(X_trainval, y_trainval, sample_weight=sample_weights)
    else:
        gb_model.fit(X_trainval, y_trainval)
    train_time = datetime.now() - train_start
    print(f"✅ Training completed in {train_time}")
    
    # Evaluate on test set with threshold 0.5
    print("\n" + "="*80)
    print("EVALUATION ON TEST DATASET")
    print("="*80)
    test_metrics = evaluate_model(gb_model, X_test, y_test, "Test (2025/26)", threshold=0.5)
    
    # Also evaluate on training set for comparison
    trainval_metrics = evaluate_model(gb_model, X_trainval, y_trainval, "Train+Val", threshold=0.5)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(gb_model, 'models/gb_model_combined_trainval.joblib')
    with open('models/gb_model_combined_trainval_columns.json', 'w') as f:
        json.dump(list(X_trainval.columns), f, indent=2)
    print(f"\n✅ Saved GB model to models/gb_model_combined_trainval.joblib")
    
    # Save metrics
    all_results = {
        'trainval': trainval_metrics,
        'test': test_metrics,
        'threshold': 0.5,
        'training_samples': len(X_trainval),
        'test_samples': len(X_test)
    }
    
    with open('experiments/gb_combined_trainval_metrics.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✅ Saved metrics to experiments/gb_combined_trainval_metrics.json")
    
    # Create summary report
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    summary_lines = [
        "# GB Model - Combined Training + Validation - Performance Summary",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Configuration",
        "",
        f"- **Training Dataset:** Combined train + validation ({len(df_trainval):,} records, {df_trainval['target'].mean():.1%} injury ratio)",
        f"- **Test Dataset:** Test (2025/26 season) ({len(df_test):,} records, {df_test['target'].mean():.1%} injury ratio)",
        f"- **Threshold:** 0.5",
        "",
        "## Performance Metrics",
        ""
    ]
    
    # Create comparison table
    summary_lines.append("| Dataset | Accuracy | Precision | Recall | F1-Score | ROC AUC | Gini | TP | FP | TN | FN |")
    summary_lines.append("|---------|----------|-----------|--------|----------|---------|------|----|----|----|----|")
    
    summary_lines.append(
        f"| **Train+Val** | {trainval_metrics['accuracy']:.4f} | {trainval_metrics['precision']:.4f} | "
        f"{trainval_metrics['recall']:.4f} | {trainval_metrics['f1']:.4f} | "
        f"{trainval_metrics['roc_auc']:.4f} | {trainval_metrics['gini']:.4f} | "
        f"{trainval_metrics['confusion_matrix']['tp']} | {trainval_metrics['confusion_matrix']['fp']} | "
        f"{trainval_metrics['confusion_matrix']['tn']} | {trainval_metrics['confusion_matrix']['fn']} |"
    )
    
    summary_lines.append(
        f"| **Test (2025/26)** | {test_metrics['accuracy']:.4f} | {test_metrics['precision']:.4f} | "
        f"{test_metrics['recall']:.4f} | {test_metrics['f1']:.4f} | "
        f"{test_metrics['roc_auc']:.4f} | {test_metrics['gini']:.4f} | "
        f"{test_metrics['confusion_matrix']['tp']} | {test_metrics['confusion_matrix']['fp']} | "
        f"{test_metrics['confusion_matrix']['tn']} | {test_metrics['confusion_matrix']['fn']} |"
    )
    
    # Add comparison with previous model (if available)
    try:
        with open('experiments/8pct_seasonal_metrics.json', 'r') as f:
            prev_results = json.load(f)
        prev_test = prev_results['GB']['test']
        
        summary_lines.extend([
            "",
            "## Comparison with Previous Model (Trained on Training Only)",
            "",
            "| Model | Precision | Recall | F1-Score | Improvement |",
            "|-------|-----------|--------|----------|-------------|",
            f"| **Previous (Train only)** | {prev_test['precision']:.4f} | {prev_test['recall']:.4f} | {prev_test['f1']:.4f} | Baseline |",
            f"| **Current (Train+Val)** | {test_metrics['precision']:.4f} | {test_metrics['recall']:.4f} | {test_metrics['f1']:.4f} | "
            f"F1: {(test_metrics['f1'] - prev_test['f1']) / prev_test['f1'] * 100:+.1f}% |"
        ])
    except:
        pass
    
    summary_text = "\n".join(summary_lines)
    with open('experiments/gb_combined_trainval_summary.md', 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    print(summary_text)
    
    total_time = datetime.now() - start_time
    print(f"\n✅ Complete! Total time: {total_time}")
    print(f"✅ Summary saved to: experiments/gb_combined_trainval_summary.md")

if __name__ == '__main__':
    main()



