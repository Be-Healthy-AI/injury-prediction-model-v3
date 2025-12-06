#!/usr/bin/env python3
"""
Threshold Optimization for Phase 2 Models
Evaluates models across a range of thresholds to find optimal operating points
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
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

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
    print("THRESHOLD OPTIMIZATION - PHASE 2 MODELS")
    print("="*80)
    
    # Load validation data
    print("\n📂 Loading validation data...")
    df_val = pd.read_csv('timelines_35day_enhanced_balanced_v4_val.csv', encoding='utf-8-sig')
    print(f"✅ Loaded {len(df_val):,} validation records")
    print(f"   Injury ratio: {df_val['target'].mean():.1%}")
    
    # Prepare data
    print("\n🔧 Preparing data...")
    X_val, y_val = prepare_data(df_val, drop_week5=False)
    
    # Load models and generate probabilities
    print("\n📂 Loading Phase 2 models and generating probabilities...")
    model_probas = {}
    
    for model_name in ['rf', 'gb', 'lr']:
        model_file = f'models/{model_name}_model_phase2.joblib'
        columns_file = f'models/{model_name}_model_phase2_columns.json'
        
        if not Path(model_file).exists():
            print(f"⚠️  {model_name.upper()} Phase 2 model not found, skipping...")
            continue
        
        model = joblib.load(model_file)
        with open(columns_file, 'r') as f:
            model_columns = json.load(f)
        
        # Align validation data to model columns
        X_val_aligned = X_val.reindex(columns=model_columns, fill_value=0)
        
        # Generate probabilities
        y_proba = model.predict_proba(X_val_aligned)[:, 1]
        model_probas[model_name] = y_proba
        
        print(f"✅ {model_name.upper()} probabilities generated")
        print(f"   Probability range: [{y_proba.min():.4f}, {y_proba.max():.4f}]")
        print(f"   Mean probability: {y_proba.mean():.4f}")
    
    if not model_probas:
        print("❌ No Phase 2 models found!")
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
    
    all_results = []
    
    print("\n" + "="*80)
    print("THRESHOLD EVALUATION")
    print("="*80)
    
    for model_name, y_proba in model_probas.items():
        print(f"\n📊 Evaluating {model_name.upper()} across thresholds...")
        
        for threshold in thresholds:
            metrics = evaluate_threshold(y_val, y_proba, threshold)
            
            all_results.append({
                'model': model_name.upper(),
                'threshold': threshold,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'roc_auc': metrics['roc_auc'],
                'tp': metrics['tp'],
                'fp': metrics['fp'],
                'tn': metrics['tn'],
                'fn': metrics['fn']
            })
    
    # Convert to DataFrame for analysis
    df_results = pd.DataFrame(all_results)
    
    print(f"\n✅ Evaluated {len(df_results):,} model-threshold combinations")
    
    # Find best operating points for each model
    print("\n" + "="*80)
    print("BEST OPERATING POINTS")
    print("="*80)
    
    best_points = {}
    
    for model_name in model_probas.keys():
        model_results = df_results[df_results['model'] == model_name.upper()]
        
        # Best F1
        best_f1_idx = model_results['f1'].idxmax()
        best_f1 = model_results.loc[best_f1_idx]
        
        # Best balanced (precision > 0.1 and recall > 0.1)
        balanced = model_results[(model_results['precision'] > 0.1) & (model_results['recall'] > 0.1)]
        if len(balanced) > 0:
            best_balanced_idx = balanced['f1'].idxmax()
            best_balanced = balanced.loc[best_balanced_idx]
        else:
            best_balanced = None
        
        # Best precision (with recall > 0.05)
        precision_candidates = model_results[model_results['recall'] > 0.05]
        if len(precision_candidates) > 0:
            best_precision_idx = precision_candidates['precision'].idxmax()
            best_precision = precision_candidates.loc[best_precision_idx]
        else:
            best_precision = None
        
        # Best recall (with precision > 0.05)
        recall_candidates = model_results[model_results['precision'] > 0.05]
        if len(recall_candidates) > 0:
            best_recall_idx = recall_candidates['recall'].idxmax()
            best_recall = recall_candidates.loc[best_recall_idx]
        else:
            best_recall = None
        
        best_points[model_name] = {
            'best_f1': best_f1,
            'best_balanced': best_balanced,
            'best_precision': best_precision,
            'best_recall': best_recall
        }
        
        print(f"\n{model_name.upper()}:")
        print(f"   Best F1:")
        print(f"      Threshold: {best_f1['threshold']:.3f}")
        print(f"      Precision: {best_f1['precision']:.4f}, Recall: {best_f1['recall']:.4f}, F1: {best_f1['f1']:.4f}")
        print(f"      TP: {int(best_f1['tp'])}, FP: {int(best_f1['fp'])}, TN: {int(best_f1['tn'])}, FN: {int(best_f1['fn'])}")
        
        if best_balanced is not None:
            print(f"   Best Balanced (P>0.1, R>0.1):")
            print(f"      Threshold: {best_balanced['threshold']:.3f}")
            print(f"      Precision: {best_balanced['precision']:.4f}, Recall: {best_balanced['recall']:.4f}, F1: {best_balanced['f1']:.4f}")
        
        if best_precision is not None:
            print(f"   Best Precision (R>0.05):")
            print(f"      Threshold: {best_precision['threshold']:.3f}")
            print(f"      Precision: {best_precision['precision']:.4f}, Recall: {best_precision['recall']:.4f}, F1: {best_precision['f1']:.4f}")
        
        if best_recall is not None:
            print(f"   Best Recall (P>0.05):")
            print(f"      Threshold: {best_recall['threshold']:.3f}")
            print(f"      Precision: {best_recall['precision']:.4f}, Recall: {best_recall['recall']:.4f}, F1: {best_recall['f1']:.4f}")
    
    # Save results
    output_dir = Path('experiments')
    output_dir.mkdir(exist_ok=True)
    
    # Save JSON
    with open(output_dir / 'phase2_threshold_optimization.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Save summary markdown
    summary_file = output_dir / 'phase2_threshold_optimization_summary.md'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("# Threshold Optimization - Phase 2 Models\n\n")
        f.write("**Date:** 2025-11-27\n")
        f.write("**Models:** RF, GB, LR (Phase 2)\n")
        f.write("**Validation Dataset:** Out-of-sample (12,218 records, 1.8% injury ratio)\n\n")
        
        for model_name in model_probas.keys():
            points = best_points[model_name]
            f.write(f"## {model_name.upper()} Model\n\n")
            
            f.write("### Best F1-Score\n\n")
            best_f1 = points['best_f1']
            f.write(f"- **Threshold:** {best_f1['threshold']:.3f}\n")
            f.write(f"- **Precision:** {best_f1['precision']:.4f}\n")
            f.write(f"- **Recall:** {best_f1['recall']:.4f}\n")
            f.write(f"- **F1-Score:** {best_f1['f1']:.4f}\n")
            f.write(f"- **ROC AUC:** {best_f1['roc_auc']:.4f}\n")
            f.write(f"- **Confusion Matrix:** TP={int(best_f1['tp'])}, FP={int(best_f1['fp'])}, TN={int(best_f1['tn'])}, FN={int(best_f1['fn'])}\n\n")
            
            if points['best_balanced'] is not None:
                f.write("### Best Balanced (Precision > 0.1, Recall > 0.1)\n\n")
                best_balanced = points['best_balanced']
                f.write(f"- **Threshold:** {best_balanced['threshold']:.3f}\n")
                f.write(f"- **Precision:** {best_balanced['precision']:.4f}\n")
                f.write(f"- **Recall:** {best_balanced['recall']:.4f}\n")
                f.write(f"- **F1-Score:** {best_balanced['f1']:.4f}\n\n")
            
            if points['best_precision'] is not None:
                f.write("### Best Precision (Recall > 0.05)\n\n")
                best_precision = points['best_precision']
                f.write(f"- **Threshold:** {best_precision['threshold']:.3f}\n")
                f.write(f"- **Precision:** {best_precision['precision']:.4f}\n")
                f.write(f"- **Recall:** {best_precision['recall']:.4f}\n")
                f.write(f"- **F1-Score:** {best_precision['f1']:.4f}\n\n")
            
            if points['best_recall'] is not None:
                f.write("### Best Recall (Precision > 0.05)\n\n")
                best_recall = points['best_recall']
                f.write(f"- **Threshold:** {best_recall['threshold']:.3f}\n")
                f.write(f"- **Precision:** {best_recall['precision']:.4f}\n")
                f.write(f"- **Recall:** {best_recall['recall']:.4f}\n")
                f.write(f"- **F1-Score:** {best_recall['f1']:.4f}\n\n")
    
    # Create comparison table
    table_file = output_dir / 'phase2_threshold_optimization_table.md'
    with open(table_file, 'w', encoding='utf-8') as f:
        f.write("# Threshold Optimization - Phase 2 Models - Summary Table\n\n")
        f.write("**Date:** 2025-11-27\n\n")
        
        f.write("## Best Operating Points\n\n")
        f.write("| Model | Objective | Threshold | Precision | Recall | F1-Score | ROC AUC | TP | FP | TN | FN |\n")
        f.write("|-------|-----------|-----------|-----------|--------|----------|---------|----|----|----|----|\n")
        
        for model_name in model_probas.keys():
            points = best_points[model_name]
            
            # Best F1
            best_f1 = points['best_f1']
            f.write(f"| {model_name.upper()} | **Best F1** | **{best_f1['threshold']:.3f}** | **{best_f1['precision']:.4f}** | "
                   f"**{best_f1['recall']:.4f}** | **{best_f1['f1']:.4f}** | {best_f1['roc_auc']:.4f} | "
                   f"{int(best_f1['tp'])} | {int(best_f1['fp'])} | {int(best_f1['tn'])} | {int(best_f1['fn'])} |\n")
            
            if points['best_balanced'] is not None:
                best_balanced = points['best_balanced']
                f.write(f"| {model_name.upper()} | Balanced | {best_balanced['threshold']:.3f} | "
                       f"{best_balanced['precision']:.4f} | {best_balanced['recall']:.4f} | {best_balanced['f1']:.4f} | "
                       f"{best_balanced['roc_auc']:.4f} | {int(best_balanced['tp'])} | {int(best_balanced['fp'])} | "
                       f"{int(best_balanced['tn'])} | {int(best_balanced['fn'])} |\n")
            
            if points['best_precision'] is not None:
                best_precision = points['best_precision']
                f.write(f"| {model_name.upper()} | Precision | {best_precision['threshold']:.3f} | "
                       f"{best_precision['precision']:.4f} | {best_precision['recall']:.4f} | {best_precision['f1']:.4f} | "
                       f"{best_precision['roc_auc']:.4f} | {int(best_precision['tp'])} | {int(best_precision['fp'])} | "
                       f"{int(best_precision['tn'])} | {int(best_precision['fn'])} |\n")
            
            if points['best_recall'] is not None:
                best_recall = points['best_recall']
                f.write(f"| {model_name.upper()} | Recall | {best_recall['threshold']:.3f} | "
                       f"{best_recall['precision']:.4f} | {best_recall['recall']:.4f} | {best_recall['f1']:.4f} | "
                       f"{best_recall['roc_auc']:.4f} | {int(best_recall['tp'])} | {int(best_recall['fp'])} | "
                       f"{int(best_recall['tn'])} | {int(best_recall['fn'])} |\n")
    
    print(f"\n✅ Results saved to {output_dir / 'phase2_threshold_optimization.json'}")
    print(f"✅ Summary saved to {summary_file}")
    print(f"✅ Table saved to {table_file}")
    print("\n" + "="*80)
    print("THRESHOLD OPTIMIZATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()



