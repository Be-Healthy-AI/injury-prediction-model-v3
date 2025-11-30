#!/usr/bin/env python3
"""
Model Comparison Script V4
Compares Random Forest, Gradient Boosting, and Logistic Regression models
"""
import sys
import io
# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import json
import pandas as pd
from datetime import datetime

def load_metrics(model_name):
    """Load metrics for a model"""
    metrics_file = f'models/{model_name}_model_v4_metrics.json'
    if not os.path.exists(metrics_file):
        metrics_file = f'scripts/{metrics_file}'
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return None

def main():
    print("📊 MODEL COMPARISON V4")
    print("=" * 80)
    print("Comparing: Random Forest, Gradient Boosting, Logistic Regression")
    print("=" * 80)
    
    models = ['rf', 'gb', 'lr']
    model_names = {
        'rf': 'Random Forest',
        'gb': 'Gradient Boosting',
        'lr': 'Logistic Regression'
    }
    
    all_metrics = {}
    for model in models:
        metrics = load_metrics(model)
        if metrics:
            all_metrics[model] = metrics
            print(f"\n✅ Loaded metrics for {model_names[model]}")
        else:
            print(f"\n❌ Could not load metrics for {model_names[model]}")
    
    if not all_metrics:
        print("\n❌ No metrics found! Please train models first.")
        return
    
    # Create comparison table for IN-SAMPLE validation
    print("\n" + "=" * 80)
    print("📊 IN-SAMPLE VALIDATION PERFORMANCE (80/20 split)")
    print("=" * 80)
    
    comparison_data_insample = []
    for model in models:
        if model in all_metrics and 'validation_insample' in all_metrics[model]:
            val_metrics = all_metrics[model]['validation_insample']
            gaps = all_metrics[model]['gaps_insample']
            comparison_data_insample.append({
                'Model': model_names[model],
                'ROC AUC': f"{val_metrics['roc_auc']:.4f}",
                'Gini': f"{val_metrics['gini']:.4f}",
                'Precision': f"{val_metrics['precision']:.4f}",
                'Recall': f"{val_metrics['recall']:.4f}",
                'F1-Score': f"{val_metrics['f1']:.4f}",
                'Accuracy': f"{val_metrics['accuracy']:.4f}",
                'AUC Gap': f"{gaps['roc_auc']:.4f}",
                'Overfitting Risk': all_metrics[model].get('overfitting_risk_insample', 'N/A')
            })
    
    comparison_df_insample = pd.DataFrame(comparison_data_insample)
    print("\n" + comparison_df_insample.to_string(index=False))
    
    # Create comparison table for OUT-OF-SAMPLE validation
    print("\n" + "=" * 80)
    print("📊 OUT-OF-SAMPLE VALIDATION PERFORMANCE (temporal split)")
    print("=" * 80)
    
    comparison_data_outsample = []
    for model in models:
        if model in all_metrics and 'validation_outsample' in all_metrics[model]:
            val_metrics = all_metrics[model]['validation_outsample']
            gaps = all_metrics[model]['gaps_outsample']
            comparison_data_outsample.append({
                'Model': model_names[model],
                'ROC AUC': f"{val_metrics['roc_auc']:.4f}",
                'Gini': f"{val_metrics['gini']:.4f}",
                'Precision': f"{val_metrics['precision']:.4f}",
                'Recall': f"{val_metrics['recall']:.4f}",
                'F1-Score': f"{val_metrics['f1']:.4f}",
                'Accuracy': f"{val_metrics['accuracy']:.4f}",
                'AUC Gap': f"{gaps['roc_auc']:.4f}"
            })
    
    comparison_df_outsample = pd.DataFrame(comparison_data_outsample)
    print("\n" + comparison_df_outsample.to_string(index=False))
    
    # Find best model by ROC AUC (in-sample)
    best_model_insample = None
    best_auc_insample = 0
    for model in models:
        if model in all_metrics and 'validation_insample' in all_metrics[model]:
            auc = all_metrics[model]['validation_insample']['roc_auc']
            if auc > best_auc_insample:
                best_auc_insample = auc
                best_model_insample = model
    
    if best_model_insample:
        print(f"\n🏆 Best Model (In-Sample, by ROC AUC): {model_names[best_model_insample]} (AUC: {best_auc_insample:.4f})")
    
    # Find best model by ROC AUC (out-of-sample)
    best_model_outsample = None
    best_auc_outsample = 0
    for model in models:
        if model in all_metrics and 'validation_outsample' in all_metrics[model]:
            auc = all_metrics[model]['validation_outsample']['roc_auc']
            if auc > best_auc_outsample:
                best_auc_outsample = auc
                best_model_outsample = model
    
    if best_model_outsample:
        print(f"🏆 Best Model (Out-of-Sample, by ROC AUC): {model_names[best_model_outsample]} (AUC: {best_auc_outsample:.4f})")
    
    # Overfitting comparison (in-sample)
    print("\n" + "=" * 80)
    print("🔍 OVERFITTING ANALYSIS COMPARISON (In-Sample)")
    print("=" * 80)
    
    overfitting_data_insample = []
    for model in models:
        if model in all_metrics and 'gaps_insample' in all_metrics[model]:
            gaps = all_metrics[model]['gaps_insample']
            risk = all_metrics[model].get('overfitting_risk_insample', 'N/A')
            overfitting_data_insample.append({
                'Model': model_names[model],
                'AUC Gap': f"{gaps['roc_auc']:.4f}",
                'F1 Gap': f"{gaps['f1']:.4f}",
                'Recall Gap': f"{gaps['recall']:.4f}",
                'Risk Level': risk
            })
    
    overfitting_df_insample = pd.DataFrame(overfitting_data_insample)
    print("\n" + overfitting_df_insample.to_string(index=False))
    
    # Save comparison report
    print("\n" + "=" * 80)
    print("💾 SAVING COMPARISON REPORT")
    print("=" * 80)
    
    os.makedirs('models', exist_ok=True)
    
    # Save as JSON
    comparison_json = {
        'generated_at': datetime.now().isoformat(),
        'models': all_metrics,
        'best_model_insample': best_model_insample,
        'best_auc_insample': best_auc_insample,
        'best_model_outsample': best_model_outsample,
        'best_auc_outsample': best_auc_outsample,
        'comparison_table_insample': comparison_data_insample,
        'comparison_table_outsample': comparison_data_outsample,
        'overfitting_table_insample': overfitting_data_insample
    }
    
    json_file = 'models/model_comparison_v4.json'
    with open(json_file, 'w') as f:
        json.dump(comparison_json, f, indent=2)
    print(f"✅ Saved JSON report to {json_file}")
    
    # Save as Markdown
    md_file = 'models/model_comparison_v4.md'
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# Model Comparison V4 (Dual Validation)\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## In-Sample Validation Performance (80/20 split)\n\n")
        try:
            f.write(comparison_df_insample.to_markdown(index=False))
        except ImportError:
            f.write(comparison_df_insample.to_string(index=False))
        f.write("\n\n")
        
        f.write("## Out-of-Sample Validation Performance (temporal split)\n\n")
        try:
            f.write(comparison_df_outsample.to_markdown(index=False))
        except ImportError:
            f.write(comparison_df_outsample.to_string(index=False))
        f.write("\n\n")
        
        f.write("## Overfitting Analysis (In-Sample)\n\n")
        try:
            f.write(overfitting_df_insample.to_markdown(index=False))
        except ImportError:
            f.write(overfitting_df_insample.to_string(index=False))
        f.write("\n\n")
        
        if best_model_insample:
            f.write(f"## Best Model (In-Sample)\n\n")
            f.write(f"**{model_names[best_model_insample]}** with ROC AUC: {best_auc_insample:.4f}\n\n")
        
        if best_model_outsample:
            f.write(f"## Best Model (Out-of-Sample)\n\n")
            f.write(f"**{model_names[best_model_outsample]}** with ROC AUC: {best_auc_outsample:.4f}\n\n")
        
        f.write("## Detailed Metrics\n\n")
        for model in models:
            if model in all_metrics:
                f.write(f"### {model_names[model]}\n\n")
                f.write("#### Training Set\n")
                train = all_metrics[model]['train']
                for metric, value in train.items():
                    if metric != 'confusion_matrix':
                        f.write(f"- {metric}: {value:.4f}\n")
                f.write("\n#### In-Sample Validation Set\n")
                if 'validation_insample' in all_metrics[model]:
                    val_insample = all_metrics[model]['validation_insample']
                    for metric, value in val_insample.items():
                        if metric != 'confusion_matrix':
                            f.write(f"- {metric}: {value:.4f}\n")
                f.write("\n#### Out-of-Sample Validation Set\n")
                if 'validation_outsample' in all_metrics[model]:
                    val_outsample = all_metrics[model]['validation_outsample']
                    for metric, value in val_outsample.items():
                        if metric != 'confusion_matrix':
                            f.write(f"- {metric}: {value:.4f}\n")
                f.write("\n")
    
    print(f"✅ Saved Markdown report to {md_file}")
    
    print("\n🎉 COMPARISON COMPLETED!")

if __name__ == "__main__":
    main()

