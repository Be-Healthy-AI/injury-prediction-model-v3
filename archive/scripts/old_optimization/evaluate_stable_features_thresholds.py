#!/usr/bin/env python3
"""
Evaluate stable-features models across multiple thresholds (0.1, 0.2, 0.3, 0.4, 0.5)
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
    
    # Handle missing values - fill NaN with 0
    X = X.fillna(0)
    
    # Convert to numeric, replacing any non-numeric values with 0
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    return X, y

def evaluate_at_threshold(y_true, y_proba, threshold):
    """Evaluate metrics at a specific threshold"""
    y_pred = (y_proba >= threshold).astype(int)
    
    if len(np.unique(y_pred)) == 1:
        # All predictions are the same class
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
    
    # Calculate ROC AUC (same regardless of threshold)
    try:
        roc_auc = roc_auc_score(y_true, y_proba)
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

def evaluate_model_experiment(model_path, columns_path, val_data, model_name):
    """Evaluate a single model across all thresholds"""
    print(f"  Loading {model_name} stable-features model...")
    
    if not os.path.exists(model_path) or not os.path.exists(columns_path):
        print(f"    ⚠️  Model or columns file not found, skipping...")
        return None
    
    # Load model and columns
    model = joblib.load(model_path)
    X_val, y_val = prepare_data(val_data, columns_path)
    
    # Generate probability scores
    y_proba = model.predict_proba(X_val)[:, 1]
    
    # Evaluate at each threshold
    results = []
    for threshold in THRESHOLDS:
        metrics = evaluate_at_threshold(y_val, y_proba, threshold)
        metrics['model'] = model_name.upper()
        results.append(metrics)
    
    return results

def main():
    print("=" * 80)
    print("STABLE-FEATURES MODELS THRESHOLD SWEEP")
    print("=" * 80)
    print("\nEvaluating models across thresholds: ", THRESHOLDS)
    
    # Load validation data
    val_file = 'timelines_35day_enhanced_balanced_v4_val.csv'
    if not os.path.exists(val_file):
        print(f"❌ Error: Validation file not found: {val_file}")
        return
    
    print(f"\n📂 Loading validation data from {val_file}...")
    df_val = pd.read_csv(val_file, encoding='utf-8-sig')
    print(f"✅ Loaded {len(df_val):,} validation records")
    print(f"   Injury ratio: {df_val['target'].mean():.1%}\n")
    
    # Model configurations
    models_config = [
        ('rf', 'models/rf_stable_features.joblib', 'models/rf_stable_features_columns.json'),
        ('gb', 'models/gb_stable_features.joblib', 'models/gb_stable_features_columns.json'),
        ('lr', 'models/lr_stable_features.joblib', 'models/lr_stable_features_columns.json'),
    ]
    
    # Evaluate all models
    all_results = []
    
    for model_key, model_path, cols_path in models_config:
        print(f"\n{'='*80}")
        print(f"MODEL: {model_key.upper()}")
        print(f"{'='*80}")
        
        results = evaluate_model_experiment(
            model_path, cols_path, df_val, model_key
        )
        
        if results:
            all_results.extend(results)
            print(f"  ✅ Completed {model_key.upper()}")
    
    if not all_results:
        print("\n❌ No results generated. Check model paths.")
        return
    
    # Create summary DataFrame
    df_results = pd.DataFrame(all_results)
    
    # Create summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE - STABLE FEATURES MODELS")
    print("=" * 80)
    
    # Format for display
    summary_cols = ['model', 'threshold', 'precision', 'recall', 'f1', 'roc_auc', 'tp', 'fp', 'tn', 'fn']
    df_display = df_results[summary_cols].copy()
    df_display['precision'] = df_display['precision'].apply(lambda x: f"{x:.4f}")
    df_display['recall'] = df_display['recall'].apply(lambda x: f"{x:.4f}")
    df_display['f1'] = df_display['f1'].apply(lambda x: f"{x:.4f}")
    df_display['roc_auc'] = df_display['roc_auc'].apply(lambda x: f"{x:.4f}")
    
    # Print table
    print("\n| Model | Threshold | Precision | Recall | F1-Score | ROC AUC | TP | FP | TN | FN |")
    print("|" + "|".join(["---"] * 10) + "|")
    for _, row in df_display.iterrows():
        print(f"| {row['model']} | {row['threshold']:.1f} | {row['precision']} | {row['recall']} | {row['f1']} | {row['roc_auc']} | {int(row['tp'])} | {int(row['fp'])} | {int(row['tn'])} | {int(row['fn'])} |")
    
    # Find best operating points
    print("\n" + "=" * 80)
    print("BEST OPERATING POINTS BY METRIC")
    print("=" * 80)
    
    for model_key in ['RF', 'GB', 'LR']:
        model_results = df_results[df_results['model'] == model_key].copy()
        
        if len(model_results) == 0:
            continue
        
        print(f"\n{model_key} Model:")
        print(f"  Best F1-Score: {model_results.loc[model_results['f1'].idxmax(), 'threshold']:.1f} threshold")
        print(f"    Precision: {model_results.loc[model_results['f1'].idxmax(), 'precision']:.4f}, "
              f"Recall: {model_results.loc[model_results['f1'].idxmax(), 'recall']:.4f}, "
              f"F1: {model_results.loc[model_results['f1'].idxmax(), 'f1']:.4f}")
        
        print(f"  Best Precision: {model_results.loc[model_results['precision'].idxmax(), 'threshold']:.1f} threshold")
        print(f"    Precision: {model_results.loc[model_results['precision'].idxmax(), 'precision']:.4f}, "
              f"Recall: {model_results.loc[model_results['precision'].idxmax(), 'recall']:.4f}, "
              f"F1: {model_results.loc[model_results['precision'].idxmax(), 'f1']:.4f}")
        
        print(f"  Best Recall: {model_results.loc[model_results['recall'].idxmax(), 'threshold']:.1f} threshold")
        print(f"    Precision: {model_results.loc[model_results['recall'].idxmax(), 'precision']:.4f}, "
              f"Recall: {model_results.loc[model_results['recall'].idxmax(), 'recall']:.4f}, "
              f"F1: {model_results.loc[model_results['recall'].idxmax(), 'f1']:.4f}")
    
    # Save results
    output_file = 'experiments/stable_features_threshold_sweep_results.json'
    output_md = 'experiments/stable_features_threshold_sweep_results.md'
    
    os.makedirs('experiments', exist_ok=True)
    
    # Save JSON
    df_results.to_json(output_file, orient='records', indent=2)
    print(f"\n✅ Results saved to {output_file}")
    
    # Save Markdown
    with open(output_md, 'w', encoding='utf-8') as f:
        f.write("# Stable-Features Models Threshold Sweep Results\n\n")
        f.write("## Configuration\n\n")
        f.write(f"- Thresholds evaluated: {THRESHOLDS}\n")
        f.write(f"- Validation dataset: {len(df_val):,} records\n")
        f.write(f"- Validation injury ratio: {df_val['target'].mean():.1%}\n")
        f.write(f"- Stable features: 712 features (drift < 0.05, train_corr > 0.01)\n\n")
        f.write("## Results\n\n")
        f.write("| Model | Threshold | Precision | Recall | F1-Score | ROC AUC | TP | FP | TN | FN |\n")
        f.write("|" + "|".join(["---"] * 10) + "|\n")
        for _, row in df_display.iterrows():
            f.write(f"| {row['model']} | {row['threshold']:.1f} | {row['precision']} | {row['recall']} | {row['f1']} | {row['roc_auc']} | {int(row['tp'])} | {int(row['fp'])} | {int(row['tn'])} | {int(row['fn'])} |\n")
    
    print(f"✅ Markdown report saved to {output_md}")
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()

