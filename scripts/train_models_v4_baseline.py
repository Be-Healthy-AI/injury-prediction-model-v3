#!/usr/bin/env python3
"""
Train baseline models on V4 timeline datasets
- Training: 2022-07-01 to 2024-06-30 (seasons 2022/23, 2023/24)
- Validation: 2024-07-01 to 2025-06-30 (season 2024/25)
- Test: >= 2025-07-01
- Simple baseline approach: no calibration, no feature selection, no correlation filtering
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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix
)
from pathlib import Path

def prepare_data(df):
    """Prepare data with basic preprocessing (no feature selection)"""
    # Drop non-feature columns
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

def train_rf(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train Random Forest model"""
    print("\n" + "=" * 70)
    print("🚀 TRAINING RANDOM FOREST MODEL")
    print("=" * 70)
    
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1,
        max_features='sqrt'
    )
    
    print("\n🔧 Model hyperparameters:")
    print(f"   n_estimators: {rf_model.n_estimators}")
    print(f"   max_depth: {rf_model.max_depth}")
    print(f"   min_samples_split: {rf_model.min_samples_split}")
    print(f"   min_samples_leaf: {rf_model.min_samples_leaf}")
    print(f"   max_features: {rf_model.max_features}")
    print(f"   class_weight: {rf_model.class_weight}")
    
    print("\n🌳 Training started...")
    train_start = datetime.now()
    rf_model.fit(X_train, y_train)
    train_time = datetime.now() - train_start
    print(f"✅ Training completed in {train_time}")
    
    # Evaluate
    train_metrics = evaluate_model(rf_model, X_train, y_train, "Training")
    val_metrics = evaluate_model(rf_model, X_val, y_val, "Validation (2024/25)")
    test_metrics = evaluate_model(rf_model, X_test, y_test, "Test (>= 2025-07-01)")
    
    return rf_model, train_metrics, val_metrics, test_metrics

def train_gb(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train Gradient Boosting model"""
    print("\n" + "=" * 70)
    print("🚀 TRAINING GRADIENT BOOSTING MODEL")
    print("=" * 70)
    
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.8,
        max_features='sqrt',
        random_state=42
    )
    
    print("\n🔧 Model hyperparameters:")
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
    train_metrics = evaluate_model(gb_model, X_train, y_train, "Training")
    val_metrics = evaluate_model(gb_model, X_val, y_val, "Validation (2024/25)")
    test_metrics = evaluate_model(gb_model, X_test, y_test, "Test (>= 2025-07-01)")
    
    return gb_model, train_metrics, val_metrics, test_metrics

def train_lr(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train Logistic Regression model"""
    print("\n" + "=" * 70)
    print("🚀 TRAINING LOGISTIC REGRESSION MODEL")
    print("=" * 70)
    
    lr_model = LogisticRegression(
        max_iter=2000,
        C=0.1,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        solver='liblinear'
    )
    
    print("\n🔧 Model hyperparameters:")
    print(f"   max_iter: {lr_model.max_iter}")
    print(f"   C: {lr_model.C}")
    print(f"   class_weight: {lr_model.class_weight}")
    print(f"   solver: {lr_model.solver}")
    
    print("\n🌳 Training started...")
    train_start = datetime.now()
    lr_model.fit(X_train, y_train)
    train_time = datetime.now() - train_start
    print(f"✅ Training completed in {train_time}")
    
    # Evaluate
    train_metrics = evaluate_model(lr_model, X_train, y_train, "Training")
    val_metrics = evaluate_model(lr_model, X_val, y_val, "Validation (2024/25)")
    test_metrics = evaluate_model(lr_model, X_test, y_test, "Test (>= 2025-07-01)")
    
    return lr_model, train_metrics, val_metrics, test_metrics

def main():
    print("="*80)
    print("TRAINING BASELINE MODELS - V4 TIMELINE DATASETS")
    print("="*80)
    print("\n📋 Dataset Configuration:")
    print("   Training: 2022-07-01 to 2024-06-30 (seasons 2022/23, 2023/24)")
    print("   Validation: 2024-07-01 to 2025-06-30 (season 2024/25)")
    print("   Test: >= 2025-07-01")
    print("   Approach: Baseline (no calibration, no feature selection, no correlation filtering)")
    print("="*80)
    
    start_time = datetime.now()
    
    # Load data
    print("\n📂 Loading timeline data...")
    train_file = 'timelines_35day_enhanced_balanced_v4_train.csv'
    val_file = 'timelines_35day_enhanced_balanced_v4_val.csv'
    test_file = 'timelines_35day_enhanced_balanced_v4_test.csv'
    
    df_train = pd.read_csv(train_file, encoding='utf-8-sig')
    df_val = pd.read_csv(val_file, encoding='utf-8-sig')
    df_test = pd.read_csv(test_file, encoding='utf-8-sig')
    
    print(f"✅ Loaded training set: {len(df_train):,} records")
    print(f"   Injury ratio: {df_train['target'].mean():.1%}")
    print(f"✅ Loaded validation set: {len(df_val):,} records")
    print(f"   Injury ratio: {df_val['target'].mean():.1%}")
    print(f"✅ Loaded test set: {len(df_test):,} records")
    print(f"   Injury ratio: {df_test['target'].mean():.1%}")
    
    # Prepare data
    print("\n📊 Preparing data...")
    X_train, y_train = prepare_data(df_train)
    X_val, y_val = prepare_data(df_val)
    X_test, y_test = prepare_data(df_test)
    
    print(f"   Training features: {X_train.shape[1]}")
    print(f"   Validation features: {X_val.shape[1]}")
    print(f"   Test features: {X_test.shape[1]}")
    
    # Align features
    print("\n🔧 Aligning features across datasets...")
    X_train, X_val, X_test = align_features(X_train, X_val, X_test)
    
    print(f"\n✅ Prepared data: {X_train.shape[1]} features (aligned across all datasets)")
    
    # Train models
    print("\n" + "="*80)
    print("TRAINING MODELS")
    print("="*80)
    
    all_results = {}
    
    # Train RF
    rf_model, rf_train_metrics, rf_val_metrics, rf_test_metrics = train_rf(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test
    )
    
    all_results['RF'] = {
        'train': rf_train_metrics,
        'validation': rf_val_metrics,
        'test': rf_test_metrics
    }
    
    # Save RF model
    os.makedirs('models', exist_ok=True)
    joblib.dump(rf_model, 'models/rf_model_v4_baseline.joblib')
    with open('models/rf_model_v4_baseline_columns.json', 'w') as f:
        json.dump(list(X_train.columns), f, indent=2)
    print(f"\n✅ Saved RF model to models/rf_model_v4_baseline.joblib")
    
    # Train GB
    gb_model, gb_train_metrics, gb_val_metrics, gb_test_metrics = train_gb(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test
    )
    
    all_results['GB'] = {
        'train': gb_train_metrics,
        'validation': gb_val_metrics,
        'test': gb_test_metrics
    }
    
    # Save GB model
    joblib.dump(gb_model, 'models/gb_model_v4_baseline.joblib')
    with open('models/gb_model_v4_baseline_columns.json', 'w') as f:
        json.dump(list(X_train.columns), f, indent=2)
    print(f"\n✅ Saved GB model to models/gb_model_v4_baseline.joblib")
    
    # Train LR
    lr_model, lr_train_metrics, lr_val_metrics, lr_test_metrics = train_lr(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test
    )
    
    all_results['LR'] = {
        'train': lr_train_metrics,
        'validation': lr_val_metrics,
        'test': lr_test_metrics
    }
    
    # Save LR model
    joblib.dump(lr_model, 'models/lr_model_v4_baseline.joblib')
    with open('models/lr_model_v4_baseline_columns.json', 'w') as f:
        json.dump(list(X_train.columns), f, indent=2)
    print(f"\n✅ Saved LR model to models/lr_model_v4_baseline.joblib")
    
    # Save metrics
    os.makedirs('experiments', exist_ok=True)
    with open('experiments/v4_baseline_metrics.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Saved metrics to experiments/v4_baseline_metrics.json")
    
    # Create summary report
    print("\n" + "="*80)
    print("SUMMARY REPORT - PERFORMANCE COMPARISON")
    print("="*80)
    
    # Create comparison table
    print("\n" + "="*80)
    print("PERFORMANCE METRICS TABLE")
    print("="*80)
    print(f"\n{'Model':<10} {'Dataset':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC AUC':<12} {'Gini':<12}")
    print("-" * 95)
    
    for model_name in ['RF', 'GB', 'LR']:
        results = all_results[model_name]
        print(f"{model_name:<10} {'Training':<25} {results['train']['precision']:<12.4f} {results['train']['recall']:<12.4f} {results['train']['f1']:<12.4f} {results['train']['roc_auc']:<12.4f} {results['train']['gini']:<12.4f}")
        print(f"{'':<10} {'Validation (2024/25)':<25} {results['validation']['precision']:<12.4f} {results['validation']['recall']:<12.4f} {results['validation']['f1']:<12.4f} {results['validation']['roc_auc']:<12.4f} {results['validation']['gini']:<12.4f}")
        print(f"{'':<10} {'Test (>= 2025-07-01)':<25} {results['test']['precision']:<12.4f} {results['test']['recall']:<12.4f} {results['test']['f1']:<12.4f} {results['test']['roc_auc']:<12.4f} {results['test']['gini']:<12.4f}")
        print("-" * 95)
    
    # Create markdown summary
    summary_lines = [
        "# V4 Timeline Datasets - Baseline Model Performance",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Dataset Split",
        "",
        f"- **Training:** 2022-07-01 to 2024-06-30 ({len(df_train):,} records, {df_train['target'].mean():.1%} injury ratio)",
        f"- **Validation:** 2024-07-01 to 2025-06-30 ({len(df_val):,} records, {df_val['target'].mean():.1%} injury ratio)",
        f"- **Test:** >= 2025-07-01 ({len(df_test):,} records, {df_test['target'].mean():.1%} injury ratio)",
        "",
        "## Approach",
        "",
        "- **Baseline approach:** No calibration, no feature selection, no correlation filtering",
        f"- **Features:** {X_train.shape[1]} features (after one-hot encoding)",
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
        summary_lines.append(f"| | Validation (2024/25) | {results['validation']['precision']:.4f} | {results['validation']['recall']:.4f} | {results['validation']['f1']:.4f} | {results['validation']['roc_auc']:.4f} | {results['validation']['gini']:.4f} |")
        summary_lines.append(f"| | Test (>= 2025-07-01) | {results['test']['precision']:.4f} | {results['test']['recall']:.4f} | {results['test']['f1']:.4f} | {results['test']['roc_auc']:.4f} | {results['test']['gini']:.4f} |")
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
        val_f1 = results['validation']['f1']
        test_f1 = results['test']['f1']
        
        summary_lines.append(f"### {model_name}")
        summary_lines.append(f"- **F1 Gap (Train → Validation):** {train_f1 - val_f1:.4f} ({(train_f1 - val_f1)/train_f1*100:.1f}% relative)" if train_f1 > 0 else "- **F1 Gap (Train → Validation):** N/A")
        summary_lines.append(f"- **F1 Gap (Validation → Test):** {val_f1 - test_f1:.4f} ({(val_f1 - test_f1)/val_f1*100:.1f}% relative)" if val_f1 > 0 else "- **F1 Gap (Validation → Test):** N/A")
        summary_lines.append(f"- **F1 Gap (Train → Test):** {train_f1 - test_f1:.4f} ({(train_f1 - test_f1)/train_f1*100:.1f}% relative)" if train_f1 > 0 else "- **F1 Gap (Train → Test):** N/A")
        summary_lines.append("")
    
    # Save summary
    summary_text = "\n".join(summary_lines)
    with open('experiments/v4_baseline_summary.md', 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    print(f"\n✅ Saved summary report to experiments/v4_baseline_summary.md")
    
    # Create detailed table with confusion matrix
    print("\n" + "="*80)
    print("DETAILED PERFORMANCE TABLE (with Confusion Matrix)")
    print("="*80)
    print(f"\n{'Model':<10} {'Dataset':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'ROC AUC':<12} {'TP':<8} {'FP':<8} {'TN':<8} {'FN':<8}")
    print("-" * 130)
    
    for model_name in ['RF', 'GB', 'LR']:
        results = all_results[model_name]
        for dataset_name, metrics in [('Training', results['train']), 
                                       ('Validation', results['validation']), 
                                       ('Test', results['test'])]:
            cm = metrics['confusion_matrix']
            print(f"{model_name:<10} {dataset_name:<25} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1']:<12.4f} {metrics['roc_auc']:<12.4f} {cm['tp']:<8} {cm['fp']:<8} {cm['tn']:<8} {cm['fn']:<8}")
    
    total_time = datetime.now() - start_time
    print(f"\n✅ Total execution time: {total_time}")
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()

