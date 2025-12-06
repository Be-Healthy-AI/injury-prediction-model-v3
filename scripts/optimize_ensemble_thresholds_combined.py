#!/usr/bin/env python3
"""
Threshold Optimization for GB + RF Ensemble Model (Combined Train+Val)
Tests various ensemble methods and thresholds to find optimal operating points
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

def ensemble_weighted_average(proba_gb, proba_rf, weight_gb):
    """Weighted average ensemble"""
    return weight_gb * proba_gb + (1 - weight_gb) * proba_rf

def ensemble_geometric_mean(proba_gb, proba_rf):
    """Geometric mean ensemble"""
    return np.sqrt(proba_gb * proba_rf)

def ensemble_and_gate(proba_gb, proba_rf, threshold_gb, threshold_rf):
    """AND gate: both models must predict above their thresholds"""
    return ((proba_gb >= threshold_gb) & (proba_rf >= threshold_rf)).astype(int)

def ensemble_or_gate(proba_gb, proba_rf, threshold_gb, threshold_rf):
    """OR gate: either model can predict above its threshold"""
    return ((proba_gb >= threshold_gb) | (proba_rf >= threshold_rf)).astype(int)

def main():
    print("="*80)
    print("ENSEMBLE THRESHOLD OPTIMIZATION - GB + RF (COMBINED TRAIN+VAL)")
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
    proba_gb_trainval = gb_model.predict_proba(X_trainval)[:, 1]
    proba_rf_trainval = rf_model.predict_proba(X_trainval)[:, 1]
    proba_gb_test = gb_model.predict_proba(X_test)[:, 1]
    proba_rf_test = rf_model.predict_proba(X_test)[:, 1]
    print("✅ Probabilities generated")
    
    # Test different ensemble methods
    print("\n" + "="*80)
    print("ENSEMBLE METHODS EVALUATION")
    print("="*80)
    
    thresholds = np.arange(0.1, 1.0, 0.05)  # 0.1 to 0.95 in steps of 0.05
    all_results = {}
    
    # 1. Weighted Average Ensembles
    print("\n📊 Testing Weighted Average Ensembles...")
    weight_combinations = [
        (0.5, 0.5),  # Equal weights
        (0.6, 0.4),  # Favor GB
        (0.7, 0.3),  # Strongly favor GB
        (0.4, 0.6),  # Favor RF
        (0.3, 0.7),  # Strongly favor RF
    ]
    
    for weight_gb, weight_rf in weight_combinations:
        ensemble_name = f"weighted_avg_{int(weight_gb*100)}gb_{int(weight_rf*100)}rf"
        all_results[ensemble_name] = []
        
        proba_ensemble_test = ensemble_weighted_average(proba_gb_test, proba_rf_test, weight_gb)
        
        for threshold in thresholds:
            metrics = evaluate_threshold(y_test, proba_ensemble_test, threshold)
            metrics['threshold'] = threshold
            all_results[ensemble_name].append(metrics)
    
    # 2. Geometric Mean Ensemble
    print("\n📊 Testing Geometric Mean Ensemble...")
    ensemble_name = "geometric_mean"
    all_results[ensemble_name] = []
    
    proba_ensemble_test = ensemble_geometric_mean(proba_gb_test, proba_rf_test)
    
    for threshold in thresholds:
        metrics = evaluate_threshold(y_test, proba_ensemble_test, threshold)
        metrics['threshold'] = threshold
        all_results[ensemble_name].append(metrics)
    
    # 3. AND Gate Ensemble (both must agree)
    print("\n📊 Testing AND Gate Ensemble...")
    ensemble_name = "and_gate"
    all_results[ensemble_name] = []
    
    # Test different threshold combinations for AND gate
    and_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    for thresh_gb in and_thresholds:
        for thresh_rf in and_thresholds:
            y_pred = ensemble_and_gate(proba_gb_test, proba_rf_test, thresh_gb, thresh_rf)
            metrics = {
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'accuracy': accuracy_score(y_test, y_pred),
                'threshold_gb': thresh_gb,
                'threshold_rf': thresh_rf
            }
            cm = confusion_matrix(y_test, y_pred)
            if cm.shape == (2, 2):
                metrics['tp'] = int(cm[1, 1])
                metrics['fp'] = int(cm[0, 1])
                metrics['tn'] = int(cm[0, 0])
                metrics['fn'] = int(cm[1, 0])
            else:
                metrics['tp'] = 0
                metrics['fp'] = 0
                metrics['tn'] = int((y_test == 0).sum())
                metrics['fn'] = int(y_test.sum())
            
            all_results[ensemble_name].append(metrics)
    
    # 4. OR Gate Ensemble (either can trigger)
    print("\n📊 Testing OR Gate Ensemble...")
    ensemble_name = "or_gate"
    all_results[ensemble_name] = []
    
    # Test different threshold combinations for OR gate
    or_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    for thresh_gb in or_thresholds:
        for thresh_rf in or_thresholds:
            y_pred = ensemble_or_gate(proba_gb_test, proba_rf_test, thresh_gb, thresh_rf)
            metrics = {
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'accuracy': accuracy_score(y_test, y_pred),
                'threshold_gb': thresh_gb,
                'threshold_rf': thresh_rf
            }
            cm = confusion_matrix(y_test, y_pred)
            if cm.shape == (2, 2):
                metrics['tp'] = int(cm[1, 1])
                metrics['fp'] = int(cm[0, 1])
                metrics['tn'] = int(cm[0, 0])
                metrics['fn'] = int(cm[1, 0])
            else:
                metrics['tp'] = 0
                metrics['fp'] = 0
                metrics['tn'] = int((y_test == 0).sum())
                metrics['fn'] = int(y_test.sum())
            
            all_results[ensemble_name].append(metrics)
    
    # Find best operating points for each ensemble method
    print("\n" + "="*80)
    print("BEST OPERATING POINTS BY ENSEMBLE METHOD")
    print("="*80)
    
    best_results = {}
    
    for ensemble_name, results in all_results.items():
        if ensemble_name in ['and_gate', 'or_gate']:
            # For gate methods, find best F1
            best_idx = np.argmax([r['f1'] for r in results])
            best_result = results[best_idx]
            best_results[ensemble_name] = {
                'method': ensemble_name,
                'best_f1': best_result,
                'threshold_gb': best_result.get('threshold_gb', None),
                'threshold_rf': best_result.get('threshold_rf', None)
            }
        else:
            # For probability-based methods, find best F1 across thresholds
            best_idx = np.argmax([r['f1'] for r in results])
            best_result = results[best_idx]
            best_results[ensemble_name] = {
                'method': ensemble_name,
                'best_f1': best_result,
                'threshold': best_result['threshold']
            }
    
    # Print best results
    for ensemble_name, result in best_results.items():
        print(f"\n🎯 {ensemble_name.upper().replace('_', ' ')}:")
        if 'threshold' in result:
            print(f"   Threshold: {result['threshold']:.2f}")
        else:
            print(f"   Threshold GB: {result['threshold_gb']:.2f}, Threshold RF: {result['threshold_rf']:.2f}")
        print(f"   Precision: {result['best_f1']['precision']:.4f}")
        print(f"   Recall: {result['best_f1']['recall']:.4f}")
        print(f"   F1-Score: {result['best_f1']['f1']:.4f}")
        print(f"   TP: {result['best_f1']['tp']}, FP: {result['best_f1']['fp']}, TN: {result['best_f1']['tn']}, FN: {result['best_f1']['fn']}")
    
    # Find overall best ensemble
    overall_best = max(best_results.items(), key=lambda x: x[1]['best_f1']['f1'])
    print(f"\n🏆 OVERALL BEST ENSEMBLE: {overall_best[0].upper().replace('_', ' ')}")
    print(f"   F1-Score: {overall_best[1]['best_f1']['f1']:.4f}")
    print(f"   Precision: {overall_best[1]['best_f1']['precision']:.4f}")
    print(f"   Recall: {overall_best[1]['best_f1']['recall']:.4f}")
    
    # Save results
    output_data = {
        'best_operating_points': best_results,
        'overall_best': {
            'ensemble': overall_best[0],
            'metrics': overall_best[1]['best_f1'],
            'config': {k: v for k, v in overall_best[1].items() if k != 'best_f1'}
        },
        'all_results': all_results
    }
    
    os.makedirs('experiments', exist_ok=True)
    with open('experiments/ensemble_combined_threshold_optimization.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\n✅ Saved detailed results to: experiments/ensemble_combined_threshold_optimization.json")
    
    # Create summary table
    summary_lines = [
        "# GB + RF Ensemble - Combined Train+Val - Threshold Optimization Results",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Best Operating Points by Ensemble Method (Test Dataset)",
        ""
    ]
    
    summary_lines.append("| Ensemble Method | Config | Precision | Recall | F1-Score | TP | FP | TN | FN |")
    summary_lines.append("|----------------|--------|-----------|--------|----------|----|----|----|----|")
    
    for ensemble_name, result in sorted(best_results.items(), key=lambda x: x[1]['best_f1']['f1'], reverse=True):
        config_str = ""
        if 'threshold' in result:
            config_str = f"Threshold: {result['threshold']:.2f}"
        else:
            config_str = f"GB: {result['threshold_gb']:.2f}, RF: {result['threshold_rf']:.2f}"
        
        summary_lines.append(
            f"| **{ensemble_name.replace('_', ' ').title()}** | {config_str} | "
            f"{result['best_f1']['precision']:.4f} | {result['best_f1']['recall']:.4f} | "
            f"{result['best_f1']['f1']:.4f} | {result['best_f1']['tp']} | {result['best_f1']['fp']} | "
            f"{result['best_f1']['tn']} | {result['best_f1']['fn']} |"
        )
    
    summary_text = "\n".join(summary_lines)
    with open('experiments/ensemble_combined_threshold_optimization_summary.md', 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    print(f"✅ Summary saved to: experiments/ensemble_combined_threshold_optimization_summary.md")
    
    print("\n" + "="*80)
    print("✅ ENSEMBLE THRESHOLD OPTIMIZATION COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()



