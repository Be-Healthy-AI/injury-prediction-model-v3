#!/usr/bin/env python3
"""
Threshold optimization for normalized cumulative features models
Evaluates RF, GB, and LR across a range of thresholds to find optimal Precision/Recall combinations
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

def main():
    print("="*80)
    print("THRESHOLD OPTIMIZATION - NORMALIZED CUMULATIVE FEATURES")
    print("="*80)
    
    # Load validation data
    print("\n📂 Loading validation data...")
    df_val = pd.read_csv('timelines_35day_enhanced_balanced_v4_val.csv', encoding='utf-8-sig')
    print(f"✅ Loaded {len(df_val):,} validation records")
    print(f"   Injury ratio: {df_val['target'].mean():.1%}")
    
    # Prepare data
    print("\n🔧 Preparing data...")
    X_val, y_val = prepare_data(df_val, drop_week5=False)
    
    # Load models and their feature columns
    print("\n📂 Loading models...")
    models = {}
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
        
        models[model_name] = {
            'model': model,
            'proba': y_proba,
            'name': model_name.upper()
        }
        print(f"✅ Loaded {model_name.upper()} model")
    
    # Define threshold ranges for each model
    threshold_configs = {
        'rf': np.arange(0.05, 0.55, 0.05),  # 0.05 to 0.50
        'gb': np.arange(0.05, 0.55, 0.05),  # 0.05 to 0.50
        'lr': np.arange(0.01, 0.21, 0.01),  # 0.01 to 0.20 (LR tends to have lower probabilities)
    }
    
    # Evaluate each model across thresholds
    results = {}
    
    print("\n" + "="*80)
    print("THRESHOLD SWEEP RESULTS")
    print("="*80)
    
    for model_key, model_data in models.items():
        print(f"\n{'='*80}")
        print(f"{model_data['name']} - THRESHOLD SWEEP")
        print(f"{'='*80}")
        
        y_proba = model_data['proba']
        thresholds = threshold_configs[model_key]
        
        model_results = []
        
        print(f"\n{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'TP':<6} {'FP':<6} {'TN':<8} {'FN':<6}")
        print("-" * 80)
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            
            cm = confusion_matrix(y_val, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                # Handle edge case where all predictions are one class
                if y_pred.sum() == 0:
                    tn, fp, fn, tp = len(y_val) - y_val.sum(), 0, y_val.sum(), 0
                else:
                    tn, fp, fn, tp = 0, y_pred.sum() - y_val.sum(), y_val.sum() - y_pred.sum(), y_pred.sum()
            
            model_results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': int(tp),
                'fp': int(fp),
                'tn': int(tn),
                'fn': int(fn)
            })
            
            print(f"{threshold:<12.2f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {int(tp):<6} {int(fp):<6} {int(tn):<8} {int(fn):<6}")
        
        results[model_key] = model_results
        
        # Find best thresholds
        df_results = pd.DataFrame(model_results)
        
        # Best F1
        best_f1_idx = df_results['f1'].idxmax()
        best_f1 = df_results.loc[best_f1_idx]
        
        # Best balanced (precision and recall both > 0.1)
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
        
        print(f"\n📊 Best Operating Points:")
        print(f"   Best F1 (@ {best_f1['threshold']:.2f}): Precision={best_f1['precision']:.4f}, Recall={best_f1['recall']:.4f}, F1={best_f1['f1']:.4f}")
        
        if best_balanced is not None:
            print(f"   Best Balanced (@ {best_balanced['threshold']:.2f}): Precision={best_balanced['precision']:.4f}, Recall={best_balanced['recall']:.4f}, F1={best_balanced['f1']:.4f}")
        
        if best_precision is not None:
            print(f"   Best Precision (@ {best_precision['threshold']:.2f}): Precision={best_precision['precision']:.4f}, Recall={best_precision['recall']:.4f}, F1={best_precision['f1']:.4f}")
        
        if best_recall is not None:
            print(f"   Best Recall (@ {best_recall['threshold']:.2f}): Precision={best_recall['precision']:.4f}, Recall={best_recall['recall']:.4f}, F1={best_recall['f1']:.4f}")
    
    # Save results
    output_dir = Path('experiments')
    output_dir.mkdir(exist_ok=True)
    
    # Save JSON
    with open(output_dir / 'normalized_threshold_optimization.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE - BEST OPERATING POINTS")
    print("="*80)
    
    summary_data = []
    for model_key, model_data in models.items():
        df_results = pd.DataFrame(results[model_key])
        
        # Current (0.5 threshold)
        current = df_results[df_results['threshold'] == 0.5]
        if len(current) == 0:
            # Find closest to 0.5
            current = df_results.iloc[(df_results['threshold'] - 0.5).abs().argsort()[:1]]
        
        if len(current) > 0:
            current_row = current.iloc[0]
            summary_data.append({
                'Model': model_data['name'],
                'Threshold': 0.5,
                'Precision': current_row['precision'],
                'Recall': current_row['recall'],
                'F1': current_row['f1'],
                'Type': 'Current (0.5)'
            })
        
        # Best F1
        best_f1 = df_results.loc[df_results['f1'].idxmax()]
        summary_data.append({
            'Model': model_data['name'],
            'Threshold': best_f1['threshold'],
            'Precision': best_f1['precision'],
            'Recall': best_f1['recall'],
            'F1': best_f1['f1'],
            'Type': 'Best F1'
        })
        
        # Best balanced
        balanced = df_results[(df_results['precision'] > 0.1) & (df_results['recall'] > 0.1)]
        if len(balanced) > 0:
            best_balanced = balanced.loc[balanced['f1'].idxmax()]
            summary_data.append({
                'Model': model_data['name'],
                'Threshold': best_balanced['threshold'],
                'Precision': best_balanced['precision'],
                'Recall': best_balanced['recall'],
                'F1': best_balanced['f1'],
                'Type': 'Best Balanced'
            })
    
    df_summary = pd.DataFrame(summary_data)
    
    print("\n| Model | Threshold | Precision | Recall | F1-Score | Type |")
    print("|-------|-----------|-----------|--------|----------|------|")
    for _, row in df_summary.iterrows():
        print(f"| {row['Model']} | {row['Threshold']:.2f} | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1']:.4f} | {row['Type']} |")
    
    # Save summary to markdown
    summary_file = output_dir / 'normalized_threshold_optimization_summary.md'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("# Threshold Optimization - Normalized Cumulative Features\n\n")
        f.write("**Date:** 2025-11-27\n")
        f.write("**Models:** RF, GB, LR with normalized cumulative features\n")
        f.write("**Validation Dataset:** Out-of-sample (12,218 records, 1.8% injury ratio)\n\n")
        f.write("## Summary Table - Best Operating Points\n\n")
        f.write("| Model | Threshold | Precision | Recall | F1-Score | Type |\n")
        f.write("|-------|-----------|-----------|--------|----------|------|\n")
        for _, row in df_summary.iterrows():
            f.write(f"| {row['Model']} | {row['Threshold']:.2f} | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1']:.4f} | {row['Type']} |\n")
        
        f.write("\n## Detailed Results\n\n")
        for model_key, model_data in models.items():
            f.write(f"### {model_data['name']}\n\n")
            f.write("| Threshold | Precision | Recall | F1-Score | TP | FP | TN | FN |\n")
            f.write("|-----------|-----------|--------|----------|----|----|----|----|\n")
            for result in results[model_key]:
                f.write(f"| {result['threshold']:.2f} | {result['precision']:.4f} | {result['recall']:.4f} | "
                       f"{result['f1']:.4f} | {result['tp']} | {result['fp']} | {result['tn']} | {result['fn']} |\n")
            f.write("\n")
    
    print(f"\n✅ Results saved to {output_dir / 'normalized_threshold_optimization.json'}")
    print(f"✅ Summary saved to {summary_file}")
    print("\n" + "="*80)
    print("THRESHOLD OPTIMIZATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()



