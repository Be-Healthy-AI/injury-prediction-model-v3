#!/usr/bin/env python3
"""
Test Cascading Ensemble Approaches:
1. RF (low threshold, high recall) → GB (high threshold, high precision)
2. GB (low threshold, high recall) → RF (high threshold, high precision)

Compare with existing ensemble methods
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

def align_features(X_trainval, X_test):
    """Ensure both datasets have the same features"""
    common_features = list(set(X_trainval.columns) & set(X_test.columns))
    common_features = sorted(common_features)
    return X_trainval[common_features], X_test[common_features]

def evaluate_cascading(y_true, y_pred):
    """Evaluate cascading ensemble predictions"""
    metrics = {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred)
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

def cascading_rf_then_gb(proba_rf, proba_gb, threshold_rf, threshold_gb):
    """
    Cascading: RF (screening) → GB (filtering)
    Stage 1: RF flags potential injuries (low threshold = high recall)
    Stage 2: GB filters RF's positives (high threshold = high precision)
    """
    # Stage 1: RF screening (low threshold for high recall)
    rf_positives = (proba_rf >= threshold_rf).astype(int)
    
    # Stage 2: GB filtering (high threshold for high precision)
    # Only evaluate GB on samples that RF flagged as positive
    gb_filtered = np.zeros_like(rf_positives)
    gb_filtered[rf_positives == 1] = (proba_gb[rf_positives == 1] >= threshold_gb).astype(int)
    
    return gb_filtered

def cascading_gb_then_rf(proba_gb, proba_rf, threshold_gb, threshold_rf):
    """
    Cascading: GB (screening) → RF (filtering)
    Stage 1: GB flags potential injuries (low threshold = high recall)
    Stage 2: RF filters GB's positives (high threshold = high precision)
    """
    # Stage 1: GB screening (low threshold for high recall)
    gb_positives = (proba_gb >= threshold_gb).astype(int)
    
    # Stage 2: RF filtering (high threshold for high precision)
    # Only evaluate RF on samples that GB flagged as positive
    rf_filtered = np.zeros_like(gb_positives)
    rf_filtered[gb_positives == 1] = (proba_rf[gb_positives == 1] >= threshold_rf).astype(int)
    
    return rf_filtered

def main():
    print("="*80)
    print("CASCADING ENSEMBLE APPROACHES - TESTING")
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
    
    # Combine train and val
    df_trainval = pd.concat([df_train, df_val], ignore_index=True)
    
    print(f"✅ Train+Val: {len(df_trainval):,} records ({df_trainval['target'].mean():.1%} injury ratio)")
    print(f"✅ Test: {len(df_test):,} records ({df_test['target'].mean():.1%} injury ratio)")
    
    # Prepare data
    print("\n🔧 Preparing data...")
    X_trainval, y_trainval = prepare_data(df_trainval, low_shift_features_set)
    X_test, y_test = prepare_data(df_test, low_shift_features_set)
    
    # Align features
    print("\n🔧 Aligning features...")
    X_trainval, X_test = align_features(X_trainval, X_test)
    print(f"✅ Common features: {X_trainval.shape[1]}")
    
    # Load models
    print("\n📂 Loading models...")
    gb_model = joblib.load('models/gb_model_combined_trainval.joblib')
    rf_model = joblib.load('models/rf_model_combined_trainval.joblib')
    print("✅ Models loaded")
    
    # Generate probabilities
    print("\n🔮 Generating probabilities...")
    proba_gb_test = gb_model.predict_proba(X_test)[:, 1]
    proba_rf_test = rf_model.predict_proba(X_test)[:, 1]
    print("✅ Probabilities generated")
    
    # Test cascading approaches
    print("\n" + "="*80)
    print("TESTING CASCADING APPROACHES")
    print("="*80)
    
    # Threshold ranges
    low_thresholds = [0.1, 0.15, 0.2, 0.25, 0.3]  # For screening (high recall)
    high_thresholds = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85]  # For filtering (high precision)
    
    cascading_results = {
        'rf_then_gb': [],
        'gb_then_rf': []
    }
    
    # 1. RF → GB Cascading
    print("\n📊 Testing RF (screening) → GB (filtering)...")
    for thresh_rf in low_thresholds:
        for thresh_gb in high_thresholds:
            y_pred = cascading_rf_then_gb(proba_rf_test, proba_gb_test, thresh_rf, thresh_gb)
            metrics = evaluate_cascading(y_test, y_pred)
            metrics['threshold_rf'] = thresh_rf
            metrics['threshold_gb'] = thresh_gb
            cascading_results['rf_then_gb'].append(metrics)
    
    # 2. GB → RF Cascading
    print("\n📊 Testing GB (screening) → RF (filtering)...")
    for thresh_gb in low_thresholds:
        for thresh_rf in high_thresholds:
            y_pred = cascading_gb_then_rf(proba_gb_test, proba_rf_test, thresh_gb, thresh_rf)
            metrics = evaluate_cascading(y_test, y_pred)
            metrics['threshold_gb'] = thresh_gb
            metrics['threshold_rf'] = thresh_rf
            cascading_results['gb_then_rf'].append(metrics)
    
    # Find best configurations
    print("\n" + "="*80)
    print("BEST CASCADING CONFIGURATIONS")
    print("="*80)
    
    # Best F1 for each approach
    best_rf_gb = max(cascading_results['rf_then_gb'], key=lambda x: x['f1'])
    best_gb_rf = max(cascading_results['gb_then_rf'], key=lambda x: x['f1'])
    
    print(f"\n🎯 Best RF → GB (by F1):")
    print(f"   RF Threshold: {best_rf_gb['threshold_rf']:.2f} (screening)")
    print(f"   GB Threshold: {best_rf_gb['threshold_gb']:.2f} (filtering)")
    print(f"   Precision: {best_rf_gb['precision']:.4f}")
    print(f"   Recall: {best_rf_gb['recall']:.4f}")
    print(f"   F1-Score: {best_rf_gb['f1']:.4f}")
    print(f"   TP: {best_rf_gb['tp']}, FP: {best_rf_gb['fp']}, TN: {best_rf_gb['tn']}, FN: {best_rf_gb['fn']}")
    
    print(f"\n🎯 Best GB → RF (by F1):")
    print(f"   GB Threshold: {best_gb_rf['threshold_gb']:.2f} (screening)")
    print(f"   RF Threshold: {best_gb_rf['threshold_rf']:.2f} (filtering)")
    print(f"   Precision: {best_gb_rf['precision']:.4f}")
    print(f"   Recall: {best_gb_rf['recall']:.4f}")
    print(f"   F1-Score: {best_gb_rf['f1']:.4f}")
    print(f"   TP: {best_gb_rf['tp']}, FP: {best_gb_rf['fp']}, TN: {best_gb_rf['tn']}, FN: {best_gb_rf['fn']}")
    
    # Load existing ensemble results for comparison
    print("\n" + "="*80)
    print("COMPARISON WITH EXISTING ENSEMBLE METHODS")
    print("="*80)
    
    with open('experiments/ensemble_combined_threshold_optimization.json', 'r') as f:
        existing_ensembles = json.load(f)
    
    # Get best from existing ensembles
    existing_best = existing_ensembles['overall_best']
    
    print(f"\n📊 Existing Best Ensemble: {existing_best['ensemble'].replace('_', ' ').title()}")
    print(f"   Precision: {existing_best['metrics']['precision']:.4f}")
    print(f"   Recall: {existing_best['metrics']['recall']:.4f}")
    print(f"   F1-Score: {existing_best['metrics']['f1']:.4f}")
    print(f"   TP: {existing_best['metrics']['tp']}, FP: {existing_best['metrics']['fp']}, "
          f"TN: {existing_best['metrics']['tn']}, FN: {existing_best['metrics']['fn']}")
    
    # Create comparison table
    comparison_data = {
        'method': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'tp': [],
        'fp': [],
        'tn': [],
        'fn': []
    }
    
    # Add cascading results
    comparison_data['method'].append('RF → GB (Cascading)')
    comparison_data['precision'].append(best_rf_gb['precision'])
    comparison_data['recall'].append(best_rf_gb['recall'])
    comparison_data['f1'].append(best_rf_gb['f1'])
    comparison_data['tp'].append(best_rf_gb['tp'])
    comparison_data['fp'].append(best_rf_gb['fp'])
    comparison_data['tn'].append(best_rf_gb['tn'])
    comparison_data['fn'].append(best_rf_gb['fn'])
    
    comparison_data['method'].append('GB → RF (Cascading)')
    comparison_data['precision'].append(best_gb_rf['precision'])
    comparison_data['recall'].append(best_gb_rf['recall'])
    comparison_data['f1'].append(best_gb_rf['f1'])
    comparison_data['tp'].append(best_gb_rf['tp'])
    comparison_data['fp'].append(best_gb_rf['fp'])
    comparison_data['tn'].append(best_gb_rf['tn'])
    comparison_data['fn'].append(best_gb_rf['fn'])
    
    # Add existing best
    comparison_data['method'].append(f"{existing_best['ensemble'].replace('_', ' ').title()} (Weighted Avg)")
    comparison_data['precision'].append(existing_best['metrics']['precision'])
    comparison_data['recall'].append(existing_best['metrics']['recall'])
    comparison_data['f1'].append(existing_best['metrics']['f1'])
    comparison_data['tp'].append(existing_best['metrics']['tp'])
    comparison_data['fp'].append(existing_best['metrics']['fp'])
    comparison_data['tn'].append(existing_best['metrics']['tn'])
    comparison_data['fn'].append(existing_best['metrics']['fn'])
    
    # Sort by F1
    sorted_indices = sorted(range(len(comparison_data['f1'])), key=lambda i: comparison_data['f1'][i], reverse=True)
    
    print("\n📊 COMPARISON TABLE (Sorted by F1-Score):")
    print("="*80)
    print(f"{'Method':<35} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'TP':<6} {'FP':<6} {'TN':<6} {'FN':<6}")
    print("="*80)
    for idx in sorted_indices:
        print(f"{comparison_data['method'][idx]:<35} "
              f"{comparison_data['precision'][idx]:<12.4f} "
              f"{comparison_data['recall'][idx]:<12.4f} "
              f"{comparison_data['f1'][idx]:<12.4f} "
              f"{comparison_data['tp'][idx]:<6} "
              f"{comparison_data['fp'][idx]:<6} "
              f"{comparison_data['tn'][idx]:<6} "
              f"{comparison_data['fn'][idx]:<6}")
    
    # Save results
    output_data = {
        'cascading_results': cascading_results,
        'best_configurations': {
            'rf_then_gb': best_rf_gb,
            'gb_then_rf': best_gb_rf
        },
        'comparison': {
            'methods': [comparison_data['method'][i] for i in sorted_indices],
            'metrics': {
                'precision': [comparison_data['precision'][i] for i in sorted_indices],
                'recall': [comparison_data['recall'][i] for i in sorted_indices],
                'f1': [comparison_data['f1'][i] for i in sorted_indices],
                'tp': [comparison_data['tp'][i] for i in sorted_indices],
                'fp': [comparison_data['fp'][i] for i in sorted_indices],
                'tn': [comparison_data['tn'][i] for i in sorted_indices],
                'fn': [comparison_data['fn'][i] for i in sorted_indices]
            }
        }
    }
    
    os.makedirs('experiments', exist_ok=True)
    with open('experiments/cascading_ensemble_results.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\n✅ Saved detailed results to: experiments/cascading_ensemble_results.json")
    
    # Create summary markdown
    summary_lines = [
        "# Cascading Ensemble Approaches - Results",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Best Cascading Configurations",
        "",
        "### RF → GB (RF Screening, GB Filtering)",
        "",
        f"- **RF Threshold:** {best_rf_gb['threshold_rf']:.2f} (screening for high recall)",
        f"- **GB Threshold:** {best_rf_gb['threshold_gb']:.2f} (filtering for high precision)",
        f"- **Precision:** {best_rf_gb['precision']:.4f}",
        f"- **Recall:** {best_rf_gb['recall']:.4f}",
        f"- **F1-Score:** {best_rf_gb['f1']:.4f}",
        f"- **TP:** {best_rf_gb['tp']}, **FP:** {best_rf_gb['fp']}, **TN:** {best_rf_gb['tn']}, **FN:** {best_rf_gb['fn']}",
        "",
        "### GB → RF (GB Screening, RF Filtering)",
        "",
        f"- **GB Threshold:** {best_gb_rf['threshold_gb']:.2f} (screening for high recall)",
        f"- **RF Threshold:** {best_gb_rf['threshold_rf']:.2f} (filtering for high precision)",
        f"- **Precision:** {best_gb_rf['precision']:.4f}",
        f"- **Recall:** {best_gb_rf['recall']:.4f}",
        f"- **F1-Score:** {best_gb_rf['f1']:.4f}",
        f"- **TP:** {best_gb_rf['tp']}, **FP:** {best_gb_rf['fp']}, **TN:** {best_gb_rf['tn']}, **FN:** {best_gb_rf['fn']}",
        "",
        "## Comparison with Existing Ensemble Methods",
        "",
        "| Method | Precision | Recall | F1-Score | TP | FP | TN | FN |",
        "|--------|-----------|--------|----------|----|----|----|----|"
    ]
    
    for idx in sorted_indices:
        summary_lines.append(
            f"| **{comparison_data['method'][idx]}** | "
            f"{comparison_data['precision'][idx]:.4f} | "
            f"{comparison_data['recall'][idx]:.4f} | "
            f"{comparison_data['f1'][idx]:.4f} | "
            f"{comparison_data['tp'][idx]} | "
            f"{comparison_data['fp'][idx]} | "
            f"{comparison_data['tn'][idx]} | "
            f"{comparison_data['fn'][idx]} |"
        )
    
    summary_text = "\n".join(summary_lines)
    with open('experiments/cascading_ensemble_summary.md', 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    print(f"✅ Summary saved to: experiments/cascading_ensemble_summary.md")
    
    print("\n" + "="*80)
    print("✅ CASCADING ENSEMBLE TESTING COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()

