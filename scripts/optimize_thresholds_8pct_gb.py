#!/usr/bin/env python3
"""
Threshold Optimization for 8% Target Ratio GB Model
Evaluates GB model across a range of thresholds to find optimal operating points
"""
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_low_shift_features():
    """Load low-shift features from adversarial validation"""
    results_file = 'experiments/adversarial_validation_results.json'
    if not os.path.exists(results_file):
        return None
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return set(results['low_shift_features'])

def prepare_data(df, low_shift_features_set=None):
    """Prepare data with low-shift feature filtering (same as training)"""
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
    common_features = list(set(X_train.columns) & set(X_val.columns) & set(X_test.columns))
    common_features = sorted(common_features)
    return X_train[common_features], X_val[common_features], X_test[common_features]

def evaluate_threshold(y_true, y_proba, threshold):
    """Evaluate model at a specific threshold"""
    y_pred = (y_proba >= threshold).astype(int)
    
    metrics = {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0
    }
    
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        metrics['tp'] = int(cm[1, 1])
        metrics['fp'] = int(cm[0, 1])
        metrics['tn'] = int(cm[0, 0])
        metrics['fn'] = int(cm[1, 0])
    else:
        if y_pred.sum() == 0:
            metrics['tp'] = 0
            metrics['fp'] = 0
            metrics['tn'] = int((y_true == 0).sum())
            metrics['fn'] = int(y_true.sum())
        else:
            metrics['tp'] = int((y_true & y_pred).sum())
            metrics['fp'] = int((~y_true & y_pred).sum())
            metrics['tn'] = int((~y_true & ~y_pred).sum())
            metrics['fn'] = int((y_true & ~y_pred).sum())
    
    return metrics

def main():
    print("="*80)
    print("THRESHOLD OPTIMIZATION - 8% TARGET RATIO GB MODEL")
    print("="*80)
    
    # Load low-shift features
    print("\n📂 Loading low-shift features...")
    low_shift_features_set = load_low_shift_features()
    if low_shift_features_set:
        print(f"✅ Loaded {len(low_shift_features_set)} low-shift features")
    else:
        print("   Using all features")
    
    # Load datasets
    print("\n📂 Loading datasets...")
    df_train = pd.read_csv('timelines_35day_enhanced_balanced_v4_train.csv', encoding='utf-8-sig')
    df_val = pd.read_csv('timelines_35day_enhanced_balanced_v4_val.csv', encoding='utf-8-sig')
    df_test = pd.read_csv('timelines_35day_enhanced_balanced_v4_test.csv', encoding='utf-8-sig')
    
    print(f"✅ Training: {len(df_train):,} records ({df_train['target'].mean():.1%} injury ratio)")
    print(f"✅ Validation: {len(df_val):,} records ({df_val['target'].mean():.1%} injury ratio)")
    print(f"✅ Test: {len(df_test):,} records ({df_test['target'].mean():.1%} injury ratio)")
    
    # Prepare data
    print("\n🔧 Preparing data...")
    X_train, y_train = prepare_data(df_train, low_shift_features_set)
    X_val, y_val = prepare_data(df_val, low_shift_features_set)
    X_test, y_test = prepare_data(df_test, low_shift_features_set)
    
    # Align features
    print("\n🔧 Aligning features...")
    X_train, X_val, X_test = align_features(X_train, X_val, X_test)
    print(f"✅ Common features: {X_train.shape[1]}")
    
    # Load GB model
    print("\n📂 Loading GB model...")
    gb_model = joblib.load('models/gb_model_8pct_seasonal.joblib')
    print("✅ Model loaded")
    
    # Generate probabilities
    print("\n🔮 Generating probabilities...")
    y_proba_train = gb_model.predict_proba(X_train)[:, 1]
    y_proba_val = gb_model.predict_proba(X_val)[:, 1]
    y_proba_test = gb_model.predict_proba(X_test)[:, 1]
    print("✅ Probabilities generated")
    
    # Threshold sweep
    print("\n" + "="*80)
    print("THRESHOLD SWEEP")
    print("="*80)
    
    thresholds = np.arange(0.1, 1.0, 0.05)  # 0.1 to 0.95 in steps of 0.05
    results = {
        'train': [],
        'validation': [],
        'test': []
    }
    
    print(f"\n📊 Evaluating {len(thresholds)} thresholds...")
    for threshold in thresholds:
        # Training
        train_metrics = evaluate_threshold(y_train, y_proba_train, threshold)
        train_metrics['threshold'] = threshold
        results['train'].append(train_metrics)
        
        # Validation
        val_metrics = evaluate_threshold(y_val, y_proba_val, threshold)
        val_metrics['threshold'] = threshold
        results['validation'].append(val_metrics)
        
        # Test
        test_metrics = evaluate_threshold(y_test, y_proba_test, threshold)
        test_metrics['threshold'] = threshold
        results['test'].append(test_metrics)
    
    # Find best operating points
    print("\n" + "="*80)
    print("BEST OPERATING POINTS")
    print("="*80)
    
    # Best F1-Score
    test_f1_scores = [r['f1'] for r in results['test']]
    best_f1_idx = np.argmax(test_f1_scores)
    best_f1_threshold = thresholds[best_f1_idx]
    best_f1_metrics = results['test'][best_f1_idx]
    
    # Best Precision (with recall > 0.05)
    test_precisions = [r['precision'] for r in results['test'] if r['recall'] > 0.05]
    test_recalls_for_prec = [r['recall'] for r in results['test'] if r['recall'] > 0.05]
    if test_precisions:
        best_prec_idx = np.argmax(test_precisions)
        valid_indices = [i for i, r in enumerate(results['test']) if r['recall'] > 0.05]
        best_prec_threshold = thresholds[valid_indices[best_prec_idx]]
        best_prec_metrics = results['test'][valid_indices[best_prec_idx]]
    else:
        best_prec_threshold = None
        best_prec_metrics = None
    
    # Best Recall (with precision > 0.1)
    test_recalls = [r['recall'] for r in results['test'] if r['precision'] > 0.1]
    if test_recalls:
        best_recall_idx = np.argmax(test_recalls)
        valid_indices = [i for i, r in enumerate(results['test']) if r['precision'] > 0.1]
        best_recall_threshold = thresholds[valid_indices[best_recall_idx]]
        best_recall_metrics = results['test'][valid_indices[best_recall_idx]]
    else:
        best_recall_threshold = None
        best_recall_metrics = None
    
    # Balanced (precision * recall maximized)
    test_balanced = [r['precision'] * r['recall'] for r in results['test']]
    best_balanced_idx = np.argmax(test_balanced)
    best_balanced_threshold = thresholds[best_balanced_idx]
    best_balanced_metrics = results['test'][best_balanced_idx]
    
    print(f"\n🎯 Best F1-Score (Test):")
    print(f"   Threshold: {best_f1_threshold:.2f}")
    print(f"   Precision: {best_f1_metrics['precision']:.4f}")
    print(f"   Recall: {best_f1_metrics['recall']:.4f}")
    print(f"   F1-Score: {best_f1_metrics['f1']:.4f}")
    print(f"   TP: {best_f1_metrics['tp']}, FP: {best_f1_metrics['fp']}, TN: {best_f1_metrics['tn']}, FN: {best_f1_metrics['fn']}")
    
    if best_prec_metrics:
        print(f"\n🎯 Best Precision (Test, Recall > 0.05):")
        print(f"   Threshold: {best_prec_threshold:.2f}")
        print(f"   Precision: {best_prec_metrics['precision']:.4f}")
        print(f"   Recall: {best_prec_metrics['recall']:.4f}")
        print(f"   F1-Score: {best_prec_metrics['f1']:.4f}")
        print(f"   TP: {best_prec_metrics['tp']}, FP: {best_prec_metrics['fp']}, TN: {best_prec_metrics['tn']}, FN: {best_prec_metrics['fn']}")
    
    if best_recall_metrics:
        print(f"\n🎯 Best Recall (Test, Precision > 0.1):")
        print(f"   Threshold: {best_recall_threshold:.2f}")
        print(f"   Precision: {best_recall_metrics['precision']:.4f}")
        print(f"   Recall: {best_recall_metrics['recall']:.4f}")
        print(f"   F1-Score: {best_recall_metrics['f1']:.4f}")
        print(f"   TP: {best_recall_metrics['tp']}, FP: {best_recall_metrics['fp']}, TN: {best_recall_metrics['tn']}, FN: {best_recall_metrics['fn']}")
    
    print(f"\n🎯 Best Balanced (Precision × Recall, Test):")
    print(f"   Threshold: {best_balanced_threshold:.2f}")
    print(f"   Precision: {best_balanced_metrics['precision']:.4f}")
    print(f"   Recall: {best_balanced_metrics['recall']:.4f}")
    print(f"   F1-Score: {best_balanced_metrics['f1']:.4f}")
    print(f"   TP: {best_balanced_metrics['tp']}, FP: {best_balanced_metrics['fp']}, TN: {best_balanced_metrics['tn']}, FN: {best_balanced_metrics['fn']}")
    
    # Save results
    output_data = {
        'best_operating_points': {
            'best_f1': {
                'threshold': float(best_f1_threshold),
                'metrics': best_f1_metrics
            },
            'best_precision': {
                'threshold': float(best_prec_threshold) if best_prec_threshold else None,
                'metrics': best_prec_metrics if best_prec_metrics else None
            },
            'best_recall': {
                'threshold': float(best_recall_threshold) if best_recall_threshold else None,
                'metrics': best_recall_metrics if best_recall_metrics else None
            },
            'best_balanced': {
                'threshold': float(best_balanced_threshold),
                'metrics': best_balanced_metrics
            }
        },
        'all_thresholds': {
            'train': results['train'],
            'validation': results['validation'],
            'test': results['test']
        }
    }
    
    os.makedirs('experiments', exist_ok=True)
    with open('experiments/8pct_gb_threshold_optimization.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\n✅ Saved detailed results to: experiments/8pct_gb_threshold_optimization.json")
    
    # Create summary table
    summary_lines = [
        "# 8% Target Ratio GB Model - Threshold Optimization Results",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Best Operating Points (Test Dataset)",
        ""
    ]
    
    summary_lines.append("| Metric | Threshold | Precision | Recall | F1-Score | TP | FP | TN | FN |")
    summary_lines.append("|--------|-----------|-----------|--------|----------|----|----|----|----|")
    
    summary_lines.append(f"| **Best F1** | {best_f1_threshold:.2f} | {best_f1_metrics['precision']:.4f} | {best_f1_metrics['recall']:.4f} | {best_f1_metrics['f1']:.4f} | {best_f1_metrics['tp']} | {best_f1_metrics['fp']} | {best_f1_metrics['tn']} | {best_f1_metrics['fn']} |")
    
    if best_prec_metrics:
        summary_lines.append(f"| **Best Precision** | {best_prec_threshold:.2f} | {best_prec_metrics['precision']:.4f} | {best_prec_metrics['recall']:.4f} | {best_prec_metrics['f1']:.4f} | {best_prec_metrics['tp']} | {best_prec_metrics['fp']} | {best_prec_metrics['tn']} | {best_prec_metrics['fn']} |")
    
    if best_recall_metrics:
        summary_lines.append(f"| **Best Recall** | {best_recall_threshold:.2f} | {best_recall_metrics['precision']:.4f} | {best_recall_metrics['recall']:.4f} | {best_recall_metrics['f1']:.4f} | {best_recall_metrics['tp']} | {best_recall_metrics['fp']} | {best_recall_metrics['tn']} | {best_recall_metrics['fn']} |")
    
    summary_lines.append(f"| **Best Balanced** | {best_balanced_threshold:.2f} | {best_balanced_metrics['precision']:.4f} | {best_balanced_metrics['recall']:.4f} | {best_balanced_metrics['f1']:.4f} | {best_balanced_metrics['tp']} | {best_balanced_metrics['fp']} | {best_balanced_metrics['tn']} | {best_balanced_metrics['fn']} |")
    
    # Add full threshold table
    summary_lines.extend([
        "",
        "## Full Threshold Sweep (Test Dataset)",
        "",
        "| Threshold | Precision | Recall | F1-Score | Accuracy | TP | FP | TN | FN |",
        "|-----------|-----------|--------|----------|----------|----|----|----|----|"
    ])
    
    for i, threshold in enumerate(thresholds):
        test_metrics = results['test'][i]
        summary_lines.append(
            f"| {threshold:.2f} | {test_metrics['precision']:.4f} | {test_metrics['recall']:.4f} | "
            f"{test_metrics['f1']:.4f} | {test_metrics['accuracy']:.4f} | "
            f"{test_metrics['tp']} | {test_metrics['fp']} | {test_metrics['tn']} | {test_metrics['fn']} |"
        )
    
    # Add comparison with default threshold (0.5)
    default_idx = np.where(thresholds == 0.5)[0]
    if len(default_idx) > 0:
        default_metrics = results['test'][default_idx[0]]
        summary_lines.extend([
            "",
            "## Comparison: Default (0.5) vs Best F1",
            "",
            "| Threshold | Precision | Recall | F1-Score | Improvement |",
            "|-----------|-----------|--------|----------|-------------|",
            f"| **Default (0.5)** | {default_metrics['precision']:.4f} | {default_metrics['recall']:.4f} | {default_metrics['f1']:.4f} | Baseline |",
            f"| **Best F1 ({best_f1_threshold:.2f})** | {best_f1_metrics['precision']:.4f} | {best_f1_metrics['recall']:.4f} | {best_f1_metrics['f1']:.4f} | "
            f"{(best_f1_metrics['f1'] - default_metrics['f1']) / default_metrics['f1'] * 100:+.1f}% |"
        ])
    
    summary_text = "\n".join(summary_lines)
    with open('experiments/8pct_gb_threshold_optimization_summary.md', 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    print(f"✅ Summary saved to: experiments/8pct_gb_threshold_optimization_summary.md")
    
    print("\n" + "="*80)
    print("✅ THRESHOLD OPTIMIZATION COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()



