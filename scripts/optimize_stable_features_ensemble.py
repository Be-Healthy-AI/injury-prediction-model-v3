#!/usr/bin/env python3
"""
Optimize ensemble combinations of stable-features models (RF, GB, LR)
to find best Precision & Recall balance on out-of-sample validation
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
from pathlib import Path
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    confusion_matrix, roc_auc_score
)
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# Thresholds to evaluate
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]

def prepare_data(df, columns_file):
    """Prepare validation data with same columns as training"""
    with open(columns_file, 'r') as f:
        training_columns = json.load(f)
    
    # Get feature columns (exclude metadata)
    feature_columns = [col for col in training_columns 
                      if col not in ['player_id', 'reference_date', 'player_name', 'target']]
    
    # Prepare X and y
    X = df[feature_columns].copy() if all(col in df.columns for col in feature_columns) else df.reindex(columns=feature_columns, fill_value=0)
    y = df['target'].values if 'target' in df.columns else None
    
    # Ensure same columns as training
    X = X.reindex(columns=training_columns, fill_value=0)
    
    # Handle missing values
    X = X.fillna(0)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    return X, y

def evaluate_ensemble(y_true, ensemble_proba, threshold):
    """Evaluate ensemble at a specific threshold"""
    y_pred = (ensemble_proba >= threshold).astype(int)
    
    if len(np.unique(y_pred)) == 1:
        if y_pred[0] == 0:
            precision = 0.0 if np.sum(y_pred) == 0 else precision_score(y_true, y_pred, zero_division=0)
            recall = 0.0
        else:
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
    else:
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
    
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    try:
        roc_auc = roc_auc_score(y_true, ensemble_proba)
    except:
        roc_auc = 0.0
    
    return {
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'tp': int(cm[1, 1]) if cm.shape == (2, 2) else 0,
        'fp': int(cm[0, 1]) if cm.shape == (2, 2) else 0,
        'tn': int(cm[0, 0]) if cm.shape == (2, 2) else int(cm[0, 0]),
        'fn': int(cm[1, 0]) if cm.shape == (2, 2) else int(np.sum(y_true))
    }

def main():
    print("="*80)
    print("ENSEMBLE OPTIMIZATION - STABLE FEATURES MODELS")
    print("="*80)
    
    # Load validation data
    val_file = 'timelines_35day_enhanced_balanced_v4_val.csv'
    if not os.path.exists(val_file):
        print(f"❌ Error: Validation file not found: {val_file}")
        return
    
    print(f"\n📂 Loading validation data...")
    df_val = pd.read_csv(val_file, encoding='utf-8-sig')
    print(f"✅ Loaded {len(df_val):,} validation records")
    print(f"   Injury ratio: {df_val['target'].mean():.1%}\n")
    
    # Load models and generate probabilities
    print("📂 Loading models and generating probabilities...")
    models = {}
    model_configs = [
        ('rf', 'models/rf_stable_features.joblib', 'models/rf_stable_features_columns.json'),
        ('gb', 'models/gb_stable_features.joblib', 'models/gb_stable_features_columns.json'),
        ('lr', 'models/lr_stable_features.joblib', 'models/lr_stable_features_columns.json'),
    ]
    
    probabilities = {}
    y_val = None
    
    for model_key, model_path, cols_path in model_configs:
        if not os.path.exists(model_path) or not os.path.exists(cols_path):
            print(f"⚠️  {model_key.upper()} model not found, skipping...")
            continue
        
        model = joblib.load(model_path)
        X_val, y_val = prepare_data(df_val, cols_path)
        proba = model.predict_proba(X_val)[:, 1]
        probabilities[model_key] = proba
        print(f"✅ {model_key.upper()}: Generated probabilities (mean: {proba.mean():.4f})")
    
    if len(probabilities) == 0:
        print("❌ No models loaded!")
        return
    
    if y_val is None:
        print("❌ Could not load target values!")
        return
    
    print(f"\n📊 Testing ensemble combinations...")
    
    # Ensemble configurations to test
    ensemble_configs = []
    
    # Weighted probability blending (2-model combinations)
    if 'rf' in probabilities and 'gb' in probabilities:
        for rf_w in [0.3, 0.4, 0.5, 0.6, 0.7]:
            gb_w = 1.0 - rf_w
            ensemble_configs.append({
                'name': f'RF_{int(rf_w*100)}_GB_{int(gb_w*100)}',
                'type': 'weighted',
                'weights': {'rf': rf_w, 'gb': gb_w},
                'models': ['rf', 'gb']
            })
    
    if 'rf' in probabilities and 'lr' in probabilities:
        for rf_w in [0.3, 0.4, 0.5, 0.6, 0.7]:
            lr_w = 1.0 - rf_w
            ensemble_configs.append({
                'name': f'RF_{int(rf_w*100)}_LR_{int(lr_w*100)}',
                'type': 'weighted',
                'weights': {'rf': rf_w, 'lr': lr_w},
                'models': ['rf', 'lr']
            })
    
    if 'gb' in probabilities and 'lr' in probabilities:
        for gb_w in [0.3, 0.4, 0.5, 0.6, 0.7]:
            lr_w = 1.0 - gb_w
            ensemble_configs.append({
                'name': f'GB_{int(gb_w*100)}_LR_{int(lr_w*100)}',
                'type': 'weighted',
                'weights': {'gb': gb_w, 'lr': lr_w},
                'models': ['gb', 'lr']
            })
    
    # 3-model weighted ensemble
    if len(probabilities) == 3:
        for rf_w in [0.2, 0.3, 0.4]:
            for gb_w in [0.2, 0.3, 0.4]:
                lr_w = 1.0 - rf_w - gb_w
                if lr_w > 0:
                    ensemble_configs.append({
                        'name': f'RF_{int(rf_w*100)}_GB_{int(gb_w*100)}_LR_{int(lr_w*100)}',
                        'type': 'weighted',
                        'weights': {'rf': rf_w, 'gb': gb_w, 'lr': lr_w},
                        'models': ['rf', 'gb', 'lr']
                    })
    
    # AND gate ensembles (both/all models must agree)
    if 'rf' in probabilities and 'gb' in probabilities:
        ensemble_configs.append({
            'name': 'RF_AND_GB',
            'type': 'and_gate',
            'models': ['rf', 'gb']
        })
    
    if 'rf' in probabilities and 'gb' in probabilities and 'lr' in probabilities:
        ensemble_configs.append({
            'name': 'RF_AND_GB_AND_LR',
            'type': 'and_gate',
            'models': ['rf', 'gb', 'lr']
        })
    
    # OR gate ensembles (any model predicts positive)
    if 'rf' in probabilities and 'gb' in probabilities:
        ensemble_configs.append({
            'name': 'RF_OR_GB',
            'type': 'or_gate',
            'models': ['rf', 'gb']
        })
    
    # Geometric mean
    if 'rf' in probabilities and 'gb' in probabilities:
        ensemble_configs.append({
            'name': 'RF_GB_GeometricMean',
            'type': 'geometric_mean',
            'models': ['rf', 'gb']
        })
    
    print(f"   Testing {len(ensemble_configs)} ensemble configurations...")
    print(f"   Evaluating at {len(THRESHOLDS)} thresholds each...")
    print(f"   Total evaluations: {len(ensemble_configs) * len(THRESHOLDS)}\n")
    
    all_results = []
    
    for config in ensemble_configs:
        # Create ensemble probabilities
        if config['type'] == 'weighted':
            ensemble_proba = np.zeros(len(y_val))
            for model_key, weight in config['weights'].items():
                ensemble_proba += weight * probabilities[model_key]
        
        elif config['type'] == 'and_gate':
            # Use minimum probability (both must be high)
            ensemble_proba = np.minimum.reduce([probabilities[m] for m in config['models']])
        
        elif config['type'] == 'or_gate':
            # Use maximum probability (either can be high)
            ensemble_proba = np.maximum.reduce([probabilities[m] for m in config['models']])
        
        elif config['type'] == 'geometric_mean':
            # Geometric mean of probabilities
            ensemble_proba = np.ones(len(y_val))
            for m in config['models']:
                ensemble_proba *= probabilities[m]
            ensemble_proba = np.power(ensemble_proba, 1.0 / len(config['models']))
        
        # Evaluate at each threshold
        for threshold in THRESHOLDS:
            metrics = evaluate_ensemble(y_val, ensemble_proba, threshold)
            metrics['ensemble'] = config['name']
            metrics['ensemble_type'] = config['type']
            all_results.append(metrics)
    
    # Create DataFrame
    df_results = pd.DataFrame(all_results)
    
    # Find best ensembles
    print("\n" + "="*80)
    print("BEST ENSEMBLES BY METRIC")
    print("="*80)
    
    # Best F1-Score
    best_f1 = df_results.loc[df_results['f1'].idxmax()]
    print(f"\n🏆 Best F1-Score:")
    print(f"   Ensemble: {best_f1['ensemble']}")
    print(f"   Threshold: {best_f1['threshold']:.1f}")
    print(f"   Precision: {best_f1['precision']:.4f}")
    print(f"   Recall: {best_f1['recall']:.4f}")
    print(f"   F1-Score: {best_f1['f1']:.4f}")
    print(f"   ROC AUC: {best_f1['roc_auc']:.4f}")
    print(f"   TP: {int(best_f1['tp'])}, FP: {int(best_f1['fp'])}, TN: {int(best_f1['tn'])}, FN: {int(best_f1['fn'])}")
    
    # Best Precision (with reasonable recall > 0.05)
    best_precision = df_results[df_results['recall'] > 0.05].nlargest(1, 'precision')
    if len(best_precision) > 0:
        bp = best_precision.iloc[0]
        print(f"\n🎯 Best Precision (recall > 5%):")
        print(f"   Ensemble: {bp['ensemble']}")
        print(f"   Threshold: {bp['threshold']:.1f}")
        print(f"   Precision: {bp['precision']:.4f}")
        print(f"   Recall: {bp['recall']:.4f}")
        print(f"   F1-Score: {bp['f1']:.4f}")
        print(f"   ROC AUC: {bp['roc_auc']:.4f}")
    
    # Best Recall (with reasonable precision > 0.05)
    best_recall = df_results[df_results['precision'] > 0.05].nlargest(1, 'recall')
    if len(best_recall) > 0:
        br = best_recall.iloc[0]
        print(f"\n📈 Best Recall (precision > 5%):")
        print(f"   Ensemble: {br['ensemble']}")
        print(f"   Threshold: {br['threshold']:.1f}")
        print(f"   Precision: {br['precision']:.4f}")
        print(f"   Recall: {br['recall']:.4f}")
        print(f"   F1-Score: {br['f1']:.4f}")
        print(f"   ROC AUC: {br['roc_auc']:.4f}")
    
    # Best balanced (precision * recall product)
    df_results['precision_recall_product'] = df_results['precision'] * df_results['recall']
    best_balanced = df_results.nlargest(1, 'precision_recall_product')
    if len(best_balanced) > 0:
        bb = best_balanced.iloc[0]
        print(f"\n⚖️  Best Balanced (precision × recall):")
        print(f"   Ensemble: {bb['ensemble']}")
        print(f"   Threshold: {bb['threshold']:.1f}")
        print(f"   Precision: {bb['precision']:.4f}")
        print(f"   Recall: {bb['recall']:.4f}")
        print(f"   F1-Score: {bb['f1']:.4f}")
        print(f"   ROC AUC: {bb['roc_auc']:.4f}")
        print(f"   Product: {bb['precision_recall_product']:.4f}")
    
    # Top 10 by F1-Score
    print(f"\n📊 Top 10 Ensembles by F1-Score:")
    print("-"*80)
    top_10 = df_results.nlargest(10, 'f1')
    print(f"{'Ensemble':<30} {'Thresh':<8} {'Precision':<10} {'Recall':<10} {'F1':<10} {'ROC AUC':<10}")
    print("-"*80)
    for idx, row in top_10.iterrows():
        print(f"{row['ensemble']:<30} {row['threshold']:<8.1f} {row['precision']:<10.4f} {row['recall']:<10.4f} {row['f1']:<10.4f} {row['roc_auc']:<10.4f}")
    
    # Save results
    output_dir = Path('experiments')
    output_dir.mkdir(exist_ok=True)
    
    # Save JSON
    output_json = output_dir / 'stable_features_ensemble_optimization.json'
    df_results.to_json(output_json, orient='records', indent=2)
    print(f"\n✅ Results saved to {output_json}")
    
    # Save summary markdown
    output_md = output_dir / 'stable_features_ensemble_optimization.md'
    with open(output_md, 'w', encoding='utf-8') as f:
        f.write("# Stable-Features Ensemble Optimization Results\n\n")
        f.write("## Best Ensembles\n\n")
        f.write(f"### Best F1-Score\n\n")
        f.write(f"- **Ensemble:** {best_f1['ensemble']}\n")
        f.write(f"- **Threshold:** {best_f1['threshold']:.1f}\n")
        f.write(f"- **Precision:** {best_f1['precision']:.4f}\n")
        f.write(f"- **Recall:** {best_f1['recall']:.4f}\n")
        f.write(f"- **F1-Score:** {best_f1['f1']:.4f}\n")
        f.write(f"- **ROC AUC:** {best_f1['roc_auc']:.4f}\n\n")
        
        if len(best_precision) > 0:
            f.write(f"### Best Precision (recall > 5%)\n\n")
            f.write(f"- **Ensemble:** {bp['ensemble']}\n")
            f.write(f"- **Threshold:** {bp['threshold']:.1f}\n")
            f.write(f"- **Precision:** {bp['precision']:.4f}\n")
            f.write(f"- **Recall:** {bp['recall']:.4f}\n\n")
        
        if len(best_balanced) > 0:
            f.write(f"### Best Balanced (precision × recall)\n\n")
            f.write(f"- **Ensemble:** {bb['ensemble']}\n")
            f.write(f"- **Threshold:** {bb['threshold']:.1f}\n")
            f.write(f"- **Precision:** {bb['precision']:.4f}\n")
            f.write(f"- **Recall:** {bb['recall']:.4f}\n")
            f.write(f"- **F1-Score:** {bb['f1']:.4f}\n\n")
        
        f.write("## Top 10 Ensembles by F1-Score\n\n")
        f.write("| Ensemble | Threshold | Precision | Recall | F1-Score | ROC AUC |\n")
        f.write("|----------|-----------|-----------|--------|----------|----------|\n")
        for idx, row in top_10.iterrows():
            f.write(f"| {row['ensemble']} | {row['threshold']:.1f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['roc_auc']:.4f} |\n")
    
    print(f"✅ Summary saved to {output_md}")
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()



