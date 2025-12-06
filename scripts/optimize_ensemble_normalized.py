#!/usr/bin/env python3
"""
Ensemble optimization for normalized cumulative features models
Tests various ensemble combinations to find optimal Precision & Recall
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

def evaluate_ensemble(y_true, y_pred, y_proba):
    """Evaluate ensemble predictions"""
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
    print("ENSEMBLE OPTIMIZATION - NORMALIZED CUMULATIVE FEATURES")
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
    print("\n📂 Loading models and generating probabilities...")
    model_probas = {}
    
    for model_name in ['rf', 'gb', 'lr']:
        model_file = f'models/{model_name}_model_v4.joblib'
        columns_file = f'models/{model_name}_model_v4_columns.json'
        
        model = joblib.load(model_file)
        with open(columns_file, 'r') as f:
            model_columns = json.load(f)
        
        # Align validation data to model columns
        X_val_aligned = X_val.reindex(columns=model_columns, fill_value=0)
        
        # Generate probabilities
        y_proba = model.predict_proba(X_val_aligned)[:, 1]
        model_probas[model_name] = y_proba
        
        print(f"✅ {model_name.upper()} probabilities generated")
    
    rf_proba = model_probas['rf']
    gb_proba = model_probas['gb']
    lr_proba = model_probas['lr']
    
    # Define ensemble configurations
    ensemble_configs = [
        # Individual models (baseline)
        {"name": "RF_only", "type": "single", "model": "rf"},
        {"name": "GB_only", "type": "single", "model": "gb"},
        {"name": "LR_only", "type": "single", "model": "lr"},
        
        # Weighted averages (RF + GB)
        {"name": "RF_50_GB_50", "type": "weighted", "weights": {"rf": 0.5, "gb": 0.5}},
        {"name": "RF_60_GB_40", "type": "weighted", "weights": {"rf": 0.6, "gb": 0.4}},
        {"name": "RF_40_GB_60", "type": "weighted", "weights": {"rf": 0.4, "gb": 0.6}},
        {"name": "RF_70_GB_30", "type": "weighted", "weights": {"rf": 0.7, "gb": 0.3}},
        {"name": "RF_30_GB_70", "type": "weighted", "weights": {"rf": 0.3, "gb": 0.7}},
        
        # Weighted averages (RF + LR)
        {"name": "RF_50_LR_50", "type": "weighted", "weights": {"rf": 0.5, "lr": 0.5}},
        {"name": "RF_60_LR_40", "type": "weighted", "weights": {"rf": 0.6, "lr": 0.4}},
        {"name": "RF_40_LR_60", "type": "weighted", "weights": {"rf": 0.4, "lr": 0.6}},
        
        # Weighted averages (GB + LR)
        {"name": "GB_50_LR_50", "type": "weighted", "weights": {"gb": 0.5, "lr": 0.5}},
        {"name": "GB_60_LR_40", "type": "weighted", "weights": {"gb": 0.6, "lr": 0.4}},
        {"name": "GB_40_LR_60", "type": "weighted", "weights": {"gb": 0.4, "lr": 0.6}},
        
        # Weighted averages (RF + GB + LR)
        {"name": "RF_33_GB_33_LR_33", "type": "weighted", "weights": {"rf": 0.33, "gb": 0.33, "lr": 0.34}},
        {"name": "RF_40_GB_30_LR_30", "type": "weighted", "weights": {"rf": 0.4, "gb": 0.3, "lr": 0.3}},
        {"name": "RF_30_GB_40_LR_30", "type": "weighted", "weights": {"rf": 0.3, "gb": 0.4, "lr": 0.3}},
        {"name": "RF_30_GB_30_LR_40", "type": "weighted", "weights": {"rf": 0.3, "gb": 0.3, "lr": 0.4}},
        {"name": "RF_50_GB_25_LR_25", "type": "weighted", "weights": {"rf": 0.5, "gb": 0.25, "lr": 0.25}},
        {"name": "RF_25_GB_50_LR_25", "type": "weighted", "weights": {"rf": 0.25, "gb": 0.5, "lr": 0.25}},
        {"name": "RF_25_GB_25_LR_50", "type": "weighted", "weights": {"rf": 0.25, "gb": 0.25, "lr": 0.5}},
        
        # Logical ensembles
        {"name": "RF_AND_GB", "type": "and_gate"},
        {"name": "RF_OR_GB", "type": "or_gate"},
        {"name": "RF_GB_GeometricMean", "type": "geometric_mean"},
        {"name": "GB_AND_LR", "type": "and_gate", "models": ["gb", "lr"]},
        {"name": "GB_OR_LR", "type": "or_gate", "models": ["gb", "lr"]},
        {"name": "RF_AND_LR", "type": "and_gate", "models": ["rf", "lr"]},
        {"name": "RF_OR_LR", "type": "or_gate", "models": ["rf", "lr"]},
    ]
    
    # Threshold ranges to test
    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 
                  0.12, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    
    print(f"\n🔍 Testing {len(ensemble_configs)} ensemble configurations")
    print(f"🔍 Testing {len(thresholds)} thresholds per ensemble")
    print(f"🔍 Total evaluations: {len(ensemble_configs) * len(thresholds):,}")
    
    all_results = []
    
    print("\n" + "="*80)
    print("ENSEMBLE EVALUATION")
    print("="*80)
    
    for config in ensemble_configs:
        ensemble_name = config['name']
        ensemble_type = config['type']
        
        # Calculate ensemble probability
        if ensemble_type == "single":
            ensemble_proba = model_probas[config['model']]
        elif ensemble_type == "weighted":
            ensemble_proba = np.zeros(len(y_val))
            for model_key, weight in config['weights'].items():
                ensemble_proba += weight * model_probas[model_key]
        elif ensemble_type == "and_gate":
            models = config.get('models', ['rf', 'gb'])
            if len(models) == 2:
                ensemble_proba = np.minimum(model_probas[models[0]], model_probas[models[1]])
            else:
                ensemble_proba = np.minimum(rf_proba, gb_proba)
        elif ensemble_type == "or_gate":
            models = config.get('models', ['rf', 'gb'])
            if len(models) == 2:
                ensemble_proba = np.maximum(model_probas[models[0]], model_probas[models[1]])
            else:
                ensemble_proba = np.maximum(rf_proba, gb_proba)
        elif ensemble_type == "geometric_mean":
            ensemble_proba = np.sqrt(rf_proba * gb_proba)
        else:
            continue
        
        # Evaluate across thresholds
        for threshold in thresholds:
            y_pred = (ensemble_proba >= threshold).astype(int)
            metrics = evaluate_ensemble(y_val, y_pred, ensemble_proba)
            
            all_results.append({
                'ensemble': ensemble_name,
                'type': ensemble_type,
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
    
    print(f"\n✅ Evaluated {len(df_results):,} ensemble-threshold combinations")
    
    # Find best ensembles
    print("\n" + "="*80)
    print("BEST ENSEMBLE CONFIGURATIONS")
    print("="*80)
    
    # Best F1
    best_f1_idx = df_results['f1'].idxmax()
    best_f1 = df_results.loc[best_f1_idx]
    
    # Best balanced (precision > 0.1 and recall > 0.1)
    balanced = df_results[(df_results['precision'] > 0.1) & (df_results['recall'] > 0.1)]
    if len(balanced) > 0:
        best_balanced_idx = balanced['f1'].idxmax()
        best_balanced = balanced.loc[best_balanced_idx]
    else:
        best_balanced = None
    
    # Best precision (with recall > 0.05)
    precision_candidates = df_results[df_results['recall'] > 0.05]
    if len(precision_candidates) > 0:
        best_precision_idx = precision_candidates['precision'].idxmax()
        best_precision = precision_candidates.loc[best_precision_idx]
    else:
        best_precision = None
    
    # Best recall (with precision > 0.05)
    recall_candidates = df_results[df_results['precision'] > 0.05]
    if len(recall_candidates) > 0:
        best_recall_idx = recall_candidates['recall'].idxmax()
        best_recall = recall_candidates.loc[best_recall_idx]
    else:
        best_recall = None
    
    # Top 10 by F1
    top_10_f1 = df_results.nlargest(10, 'f1')
    
    print(f"\n📊 Best Operating Points:")
    print(f"\n   Best F1:")
    print(f"      Ensemble: {best_f1['ensemble']} @ Threshold {best_f1['threshold']:.3f}")
    print(f"      Precision: {best_f1['precision']:.4f}, Recall: {best_f1['recall']:.4f}, F1: {best_f1['f1']:.4f}")
    print(f"      TP: {int(best_f1['tp'])}, FP: {int(best_f1['fp'])}, TN: {int(best_f1['tn'])}, FN: {int(best_f1['fn'])}")
    
    if best_balanced is not None:
        print(f"\n   Best Balanced (P>0.1, R>0.1):")
        print(f"      Ensemble: {best_balanced['ensemble']} @ Threshold {best_balanced['threshold']:.3f}")
        print(f"      Precision: {best_balanced['precision']:.4f}, Recall: {best_balanced['recall']:.4f}, F1: {best_balanced['f1']:.4f}")
        print(f"      TP: {int(best_balanced['tp'])}, FP: {int(best_balanced['fp'])}, TN: {int(best_balanced['tn'])}, FN: {int(best_balanced['fn'])}")
    
    if best_precision is not None:
        print(f"\n   Best Precision (R>0.05):")
        print(f"      Ensemble: {best_precision['ensemble']} @ Threshold {best_precision['threshold']:.3f}")
        print(f"      Precision: {best_precision['precision']:.4f}, Recall: {best_precision['recall']:.4f}, F1: {best_precision['f1']:.4f}")
        print(f"      TP: {int(best_precision['tp'])}, FP: {int(best_precision['fp'])}, TN: {int(best_precision['tn'])}, FN: {int(best_precision['fn'])}")
    
    if best_recall is not None:
        print(f"\n   Best Recall (P>0.05):")
        print(f"      Ensemble: {best_recall['ensemble']} @ Threshold {best_recall['threshold']:.3f}")
        print(f"      Precision: {best_recall['precision']:.4f}, Recall: {best_recall['recall']:.4f}, F1: {best_recall['f1']:.4f}")
        print(f"      TP: {int(best_recall['tp'])}, FP: {int(best_recall['fp'])}, TN: {int(best_recall['tn'])}, FN: {int(best_recall['fn'])}")
    
    print(f"\n📊 Top 10 Ensembles by F1-Score:")
    print(f"\n{'Ensemble':<30} {'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 80)
    for idx, row in top_10_f1.iterrows():
        print(f"{row['ensemble']:<30} {row['threshold']:<12.3f} {row['precision']:<12.4f} {row['recall']:<12.4f} {row['f1']:<12.4f}")
    
    # Save results
    output_dir = Path('experiments')
    output_dir.mkdir(exist_ok=True)
    
    # Save JSON
    with open(output_dir / 'normalized_ensemble_optimization.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Save summary markdown
    summary_file = output_dir / 'normalized_ensemble_optimization_summary.md'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("# Ensemble Optimization - Normalized Cumulative Features\n\n")
        f.write("**Date:** 2025-11-27\n")
        f.write("**Models:** RF, GB, LR with normalized cumulative features\n")
        f.write("**Validation Dataset:** Out-of-sample (12,218 records, 1.8% injury ratio)\n\n")
        
        f.write("## Best Operating Points\n\n")
        f.write("### Best F1-Score\n\n")
        f.write(f"- **Ensemble:** {best_f1['ensemble']}\n")
        f.write(f"- **Threshold:** {best_f1['threshold']:.3f}\n")
        f.write(f"- **Precision:** {best_f1['precision']:.4f}\n")
        f.write(f"- **Recall:** {best_f1['recall']:.4f}\n")
        f.write(f"- **F1-Score:** {best_f1['f1']:.4f}\n")
        f.write(f"- **ROC AUC:** {best_f1['roc_auc']:.4f}\n")
        f.write(f"- **Confusion Matrix:** TP={int(best_f1['tp'])}, FP={int(best_f1['fp'])}, TN={int(best_f1['tn'])}, FN={int(best_f1['fn'])}\n\n")
        
        if best_balanced is not None:
            f.write("### Best Balanced (Precision > 0.1, Recall > 0.1)\n\n")
            f.write(f"- **Ensemble:** {best_balanced['ensemble']}\n")
            f.write(f"- **Threshold:** {best_balanced['threshold']:.3f}\n")
            f.write(f"- **Precision:** {best_balanced['precision']:.4f}\n")
            f.write(f"- **Recall:** {best_balanced['recall']:.4f}\n")
            f.write(f"- **F1-Score:** {best_balanced['f1']:.4f}\n\n")
        
        if best_precision is not None:
            f.write("### Best Precision (Recall > 0.05)\n\n")
            f.write(f"- **Ensemble:** {best_precision['ensemble']}\n")
            f.write(f"- **Threshold:** {best_precision['threshold']:.3f}\n")
            f.write(f"- **Precision:** {best_precision['precision']:.4f}\n")
            f.write(f"- **Recall:** {best_precision['recall']:.4f}\n")
            f.write(f"- **F1-Score:** {best_precision['f1']:.4f}\n\n")
        
        if best_recall is not None:
            f.write("### Best Recall (Precision > 0.05)\n\n")
            f.write(f"- **Ensemble:** {best_recall['ensemble']}\n")
            f.write(f"- **Threshold:** {best_recall['threshold']:.3f}\n")
            f.write(f"- **Precision:** {best_recall['precision']:.4f}\n")
            f.write(f"- **Recall:** {best_recall['recall']:.4f}\n")
            f.write(f"- **F1-Score:** {best_recall['f1']:.4f}\n\n")
        
        f.write("## Top 10 Ensembles by F1-Score\n\n")
        f.write("| Rank | Ensemble | Threshold | Precision | Recall | F1-Score | ROC AUC |\n")
        f.write("|------|----------|-----------|-----------|--------|----------|----------|\n")
        for rank, (idx, row) in enumerate(top_10_f1.iterrows(), 1):
            f.write(f"| {rank} | {row['ensemble']} | {row['threshold']:.3f} | {row['precision']:.4f} | "
                   f"{row['recall']:.4f} | {row['f1']:.4f} | {row['roc_auc']:.4f} |\n")
    
    # Create comparison table
    comparison_file = output_dir / 'normalized_ensemble_comparison_table.md'
    with open(comparison_file, 'w', encoding='utf-8') as f:
        f.write("# Ensemble Optimization - Comparison Table\n\n")
        f.write("**Date:** 2025-11-27\n\n")
        
        f.write("## Best Ensemble Configurations\n\n")
        f.write("| Ensemble | Threshold | Precision | Recall | F1-Score | ROC AUC | TP | FP | TN | FN |\n")
        f.write("|----------|-----------|-----------|--------|----------|---------|----|----|----|----|\n")
        
        # Best F1
        f.write(f"| **{best_f1['ensemble']}** | **{best_f1['threshold']:.3f}** | **{best_f1['precision']:.4f}** | "
               f"**{best_f1['recall']:.4f}** | **{best_f1['f1']:.4f}** | {best_f1['roc_auc']:.4f} | "
               f"{int(best_f1['tp'])} | {int(best_f1['fp'])} | {int(best_f1['tn'])} | {int(best_f1['fn'])} |\n")
        
        if best_balanced is not None:
            f.write(f"| {best_balanced['ensemble']} (Balanced) | {best_balanced['threshold']:.3f} | "
                   f"{best_balanced['precision']:.4f} | {best_balanced['recall']:.4f} | {best_balanced['f1']:.4f} | "
                   f"{best_balanced['roc_auc']:.4f} | {int(best_balanced['tp'])} | {int(best_balanced['fp'])} | "
                   f"{int(best_balanced['tn'])} | {int(best_balanced['fn'])} |\n")
        
        if best_precision is not None:
            f.write(f"| {best_precision['ensemble']} (Precision) | {best_precision['threshold']:.3f} | "
                   f"{best_precision['precision']:.4f} | {best_precision['recall']:.4f} | {best_precision['f1']:.4f} | "
                   f"{best_precision['roc_auc']:.4f} | {int(best_precision['tp'])} | {int(best_precision['fp'])} | "
                   f"{int(best_precision['tn'])} | {int(best_precision['fn'])} |\n")
        
        if best_recall is not None:
            f.write(f"| {best_recall['ensemble']} (Recall) | {best_recall['threshold']:.3f} | "
                   f"{best_recall['precision']:.4f} | {best_recall['recall']:.4f} | {best_recall['f1']:.4f} | "
                   f"{best_recall['roc_auc']:.4f} | {int(best_recall['tp'])} | {int(best_recall['fp'])} | "
                   f"{int(best_recall['tn'])} | {int(best_recall['fn'])} |\n")
        
        f.write("\n## Comparison with Individual Models (Best Thresholds)\n\n")
        f.write("| Model/Ensemble | Threshold | Precision | Recall | F1-Score |\n")
        f.write("|----------------|-----------|-----------|--------|----------|\n")
        
        # Individual model best F1s
        for model_name in ['rf', 'gb', 'lr']:
            model_results = df_results[df_results['ensemble'] == f'{model_name.upper()}_only']
            if len(model_results) > 0:
                best_model = model_results.loc[model_results['f1'].idxmax()]
                f.write(f"| {model_name.upper()} (best) | {best_model['threshold']:.3f} | "
                       f"{best_model['precision']:.4f} | {best_model['recall']:.4f} | {best_model['f1']:.4f} |\n")
        
        # Best ensemble
        f.write(f"| **{best_f1['ensemble']} (best)** | **{best_f1['threshold']:.3f}** | "
               f"**{best_f1['precision']:.4f}** | **{best_f1['recall']:.4f}** | **{best_f1['f1']:.4f}** |\n")
    
    print(f"\n✅ Results saved to {output_dir / 'normalized_ensemble_optimization.json'}")
    print(f"✅ Summary saved to {summary_file}")
    print(f"✅ Comparison table saved to {comparison_file}")
    print("\n" + "="*80)
    print("ENSEMBLE OPTIMIZATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()



