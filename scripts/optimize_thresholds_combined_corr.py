#!/usr/bin/env python3
"""
Threshold Optimization for Combined Train+Val Models - Muscular Injuries Only (with Correlation Filtering)
Evaluates RF and GB models across a range of thresholds to find optimal operating points
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

def prepare_data(df):
    """Prepare data with basic preprocessing (same as training)"""
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

def align_features(X_train, X_test):
    """Ensure all datasets have the same features"""
    # Get common features
    common_features = list(set(X_train.columns) & set(X_test.columns))
    
    # Sort for consistency
    common_features = sorted(common_features)
    
    return X_train[common_features], X_test[common_features]

def apply_correlation_filter(X, threshold=0.8):
    """Drop one feature from each highly correlated pair (same as training)"""
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    kept = [col for col in X.columns if col not in to_drop]
    return kept

def evaluate_threshold(y_true, y_proba, threshold):
    """Evaluate model at a specific threshold"""
    y_pred = (y_proba >= threshold).astype(int)
    
    metrics = {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0,
        'gini': (2 * roc_auc_score(y_true, y_proba) - 1) if len(np.unique(y_true)) > 1 else 0.0
    }
    
    cm = confusion_matrix(y_true, y_pred)
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
            if y_true.sum() == 0:
                metrics['confusion_matrix'] = {'tn': int(cm[0, 0]), 'fp': 0, 'fn': 0, 'tp': 0}
            else:
                metrics['confusion_matrix'] = {'tn': 0, 'fp': 0, 'fn': int(y_true.sum()), 'tp': 0}
        else:
            metrics['confusion_matrix'] = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
    
    return metrics

def main():
    print("="*80)
    print("THRESHOLD OPTIMIZATION - MUSCULAR INJURIES ONLY (COMBINED TRAIN+VAL, CORRELATION FILTERING)")
    print("="*80)
    
    start_time = datetime.now()
    
    # Load test data
    print("\n📂 Loading test data...")
    test_file = 'timelines_35day_enhanced_balanced_v4_muscular_test.csv'
    df_test = pd.read_csv(test_file, encoding='utf-8-sig')
    print(f"✅ Loaded test set: {len(df_test):,} records")
    print(f"   Injury ratio: {df_test['target'].mean():.1%}")
    
    # Prepare test data
    print("\n📊 Preparing test data...")
    X_test, y_test = prepare_data(df_test)
    
    # Load models and use their saved feature columns (already filtered)
    print("\n📂 Loading models...")
    models = {}
    for model_name in ['rf', 'gb']:
        model_file = f'models/{model_name}_model_v4_muscular_combined_corr.joblib'
        columns_file = f'models/{model_name}_model_v4_muscular_combined_corr_columns.json'
        
        if not os.path.exists(model_file):
            print(f"❌ Model file not found: {model_file}")
            continue
        
        model = joblib.load(model_file)
        with open(columns_file, 'r') as f:
            model_columns = json.load(f)
        
        print(f"✅ Loaded {model_name.upper()} model with {len(model_columns)} features")
        
        # Align test data to model columns (these are already correlation-filtered)
        X_test_aligned = X_test.reindex(columns=model_columns, fill_value=0)
        
        # Generate probabilities
        print(f"   Generating probabilities for {model_name.upper()}...")
        y_proba = model.predict_proba(X_test_aligned)[:, 1]
        
        models[model_name] = {
            'model': model,
            'proba': y_proba,
            'name': model_name.upper(),
            'columns': model_columns
        }
        print(f"✅ Generated probabilities for {model_name.upper()}")
    
    if not models:
        print("❌ No models loaded. Exiting.")
        return
    
    # Threshold ranges to test
    thresholds = np.concatenate([
        np.arange(0.01, 0.10, 0.01),  # 0.01 to 0.09
        np.arange(0.10, 0.30, 0.02),  # 0.10 to 0.28
        np.arange(0.30, 0.60, 0.05),  # 0.30 to 0.55
        [0.60, 0.70, 0.80, 0.90, 0.95]  # Higher thresholds
    ])
    
    print(f"\n🔍 Testing {len(thresholds)} thresholds per model")
    print(f"   Threshold range: {thresholds.min():.2f} to {thresholds.max():.2f}")
    
    all_results = {}
    
    print("\n" + "="*80)
    print("THRESHOLD EVALUATION")
    print("="*80)
    
    for model_key, model_data in models.items():
        print(f"\n{'='*80}")
        print(f"{model_data['name']} - THRESHOLD SWEEP")
        print(f"{'='*80}")
        
        y_proba = model_data['proba']
        model_results = []
        
        print(f"\n{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Accuracy':<12} {'TP':<6} {'FP':<6} {'TN':<8} {'FN':<6}")
        print("-" * 100)
        
        for threshold in thresholds:
            metrics = evaluate_threshold(y_test, y_proba, threshold)
            
            model_results.append({
                'threshold': threshold,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'accuracy': metrics['accuracy'],
                'roc_auc': metrics['roc_auc'],
                'gini': metrics['gini'],
                'tp': metrics['confusion_matrix']['tp'],
                'fp': metrics['confusion_matrix']['fp'],
                'tn': metrics['confusion_matrix']['tn'],
                'fn': metrics['confusion_matrix']['fn']
            })
            
            cm = metrics['confusion_matrix']
            print(f"{threshold:<12.2f} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1']:<12.4f} {metrics['accuracy']:<12.4f} {cm['tp']:<6} {cm['fp']:<6} {cm['tn']:<8} {cm['fn']:<6}")
        
        all_results[model_key] = model_results
        
        # Find best operating points
        df_results = pd.DataFrame(model_results)
        
        # Best F1
        best_f1_idx = df_results['f1'].idxmax()
        best_f1 = df_results.loc[best_f1_idx]
        
        # Best Precision (with recall > 0.05)
        valid_prec = df_results[df_results['recall'] > 0.05]
        if len(valid_prec) > 0:
            best_prec_idx = valid_prec['precision'].idxmax()
            best_prec = df_results.loc[best_prec_idx]
        else:
            best_prec = None
        
        # Best Recall (with precision > 0.05)
        valid_recall = df_results[df_results['precision'] > 0.05]
        if len(valid_recall) > 0:
            best_recall_idx = valid_recall['recall'].idxmax()
            best_recall = df_results.loc[best_recall_idx]
        else:
            best_recall = None
        
        print(f"\n{'='*80}")
        print(f"{model_data['name']} - BEST OPERATING POINTS")
        print(f"{'='*80}")
        
        print(f"\n✅ Best F1-Score:")
        print(f"   Threshold: {best_f1['threshold']:.3f}")
        print(f"   Precision: {best_f1['precision']:.4f}")
        print(f"   Recall: {best_f1['recall']:.4f}")
        print(f"   F1-Score: {best_f1['f1']:.4f}")
        print(f"   Accuracy: {best_f1['accuracy']:.4f}")
        print(f"   ROC AUC: {best_f1['roc_auc']:.4f}")
        print(f"   Gini: {best_f1['gini']:.4f}")
        print(f"   TP: {int(best_f1['tp'])}, FP: {int(best_f1['fp'])}, TN: {int(best_f1['tn'])}, FN: {int(best_f1['fn'])}")
        
        if best_prec is not None:
            print(f"\n✅ Best Precision (recall > 0.05):")
            print(f"   Threshold: {best_prec['threshold']:.3f}")
            print(f"   Precision: {best_prec['precision']:.4f}")
            print(f"   Recall: {best_prec['recall']:.4f}")
            print(f"   F1-Score: {best_prec['f1']:.4f}")
        
        if best_recall is not None:
            print(f"\n✅ Best Recall (precision > 0.05):")
            print(f"   Threshold: {best_recall['threshold']:.3f}")
            print(f"   Precision: {best_recall['precision']:.4f}")
            print(f"   Recall: {best_recall['recall']:.4f}")
            print(f"   F1-Score: {best_recall['f1']:.4f}")
    
    # Save results
    os.makedirs('experiments', exist_ok=True)
    with open('experiments/v4_muscular_combined_corr_threshold_optimization.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Saved detailed results to: experiments/v4_muscular_combined_corr_threshold_optimization.json")
    
    # Create summary
    summary_lines = [
        "# Muscular Injuries Only - Combined Train+Val Models (with Correlation Filtering) - Threshold Optimization Results",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Test Dataset",
        "",
        f"- **Records:** {len(df_test):,}",
        f"- **Injury ratio:** {df_test['target'].mean():.1%}",
        f"- **Target:** Muscular injuries only",
        "",
        "## Best Operating Points",
        ""
    ]
    
    for model_key, model_data in models.items():
        df_results = pd.DataFrame(all_results[model_key])
        
        # Best F1
        best_f1_idx = df_results['f1'].idxmax()
        best_f1 = df_results.loc[best_f1_idx]
        
        summary_lines.append(f"### {model_data['name']} - Best F1-Score")
        summary_lines.append(f"- **Threshold:** {best_f1['threshold']:.3f}")
        summary_lines.append(f"- **Precision:** {best_f1['precision']:.4f}")
        summary_lines.append(f"- **Recall:** {best_f1['recall']:.4f}")
        summary_lines.append(f"- **F1-Score:** {best_f1['f1']:.4f}")
        summary_lines.append(f"- **Accuracy:** {best_f1['accuracy']:.4f}")
        summary_lines.append(f"- **ROC AUC:** {best_f1['roc_auc']:.4f}")
        summary_lines.append(f"- **Gini:** {best_f1['gini']:.4f}")
        summary_lines.append(f"- **TP:** {int(best_f1['tp'])}, **FP:** {int(best_f1['fp'])}, **TN:** {int(best_f1['tn'])}, **FN:** {int(best_f1['fn'])}")
        summary_lines.append("")
    
    # Create comparison table
    summary_lines.extend([
        "## Performance Comparison Table",
        "",
        "| Model | Threshold | Precision | Recall | F1-Score | Accuracy | ROC AUC | Gini | TP | FP | TN | FN |",
        "|-------|-----------|-----------|--------|----------|----------|---------|------|----|----|----|----|"
    ])
    
    for model_key, model_data in models.items():
        df_results = pd.DataFrame(all_results[model_key])
        best_f1_idx = df_results['f1'].idxmax()
        best_f1 = df_results.loc[best_f1_idx]
        
        summary_lines.append(
            f"| **{model_data['name']}** | {best_f1['threshold']:.3f} | {best_f1['precision']:.4f} | "
            f"{best_f1['recall']:.4f} | {best_f1['f1']:.4f} | {best_f1['accuracy']:.4f} | "
            f"{best_f1['roc_auc']:.4f} | {best_f1['gini']:.4f} | {int(best_f1['tp'])} | "
            f"{int(best_f1['fp'])} | {int(best_f1['tn'])} | {int(best_f1['fn'])} |"
        )
    
    summary_text = "\n".join(summary_lines)
    with open('experiments/v4_muscular_combined_corr_threshold_optimization_summary.md', 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    print(f"✅ Summary saved to: experiments/v4_muscular_combined_corr_threshold_optimization_summary.md")
    
    total_time = datetime.now() - start_time
    print(f"\n✅ Total execution time: {total_time}")
    print("\n" + "="*80)
    print("THRESHOLD OPTIMIZATION COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()

