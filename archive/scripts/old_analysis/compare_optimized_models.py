#!/usr/bin/env python3
"""
Compare Optimized Models Script
Analyzes and ranks all trained models, generates comprehensive report
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
from pathlib import Path

def load_all_metrics(models_dir):
    """Load all metrics from saved model files"""
    all_results = []
    
    if not os.path.exists(models_dir):
        print(f"❌ Models directory not found: {models_dir}")
        return []
    
    metrics_files = [f for f in os.listdir(models_dir) if f.endswith('_metrics.json')]
    
    print(f"📂 Found {len(metrics_files)} model metrics files")
    
    for metrics_file in metrics_files:
        try:
            with open(f"{models_dir}/{metrics_file}", 'r') as f:
                metrics = json.load(f)
            
            # Extract model name and feature set from filename
            # Format: {model_name}_{feature_set_name}_metrics.json
            parts = metrics_file.replace('_metrics.json', '').split('_', 1)
            if len(parts) == 2:
                model_name, feature_set_name = parts
            else:
                continue
            
            all_results.append({
                'model_name': model_name,
                'feature_set_name': feature_set_name,
                'metrics': metrics
            })
        except Exception as e:
            print(f"⚠️  Error loading {metrics_file}: {e}")
    
    return all_results

def create_comparison_dataframe(all_results):
    """Create a comparison DataFrame from all results"""
    rows = []
    
    for result in all_results:
        metrics = result['metrics']
        outsample = metrics.get('validation_outsample', {})
        insample = metrics.get('validation_insample', {})
        train = metrics.get('train', {})
        gaps_outsample = metrics.get('gaps_outsample', {})
        
        row = {
            'model_name': result['model_name'],
            'feature_set_name': result['feature_set_name'],
            'best_f1': metrics.get('best_f1', 0),
            # Out-of-sample metrics (primary)
            'outsample_f1': outsample.get('f1', 0),
            'outsample_precision': outsample.get('precision', 0),
            'outsample_recall': outsample.get('recall', 0),
            'outsample_roc_auc': outsample.get('roc_auc', 0),
            'outsample_gini': outsample.get('gini', 0),
            'outsample_accuracy': outsample.get('accuracy', 0),
            # In-sample metrics
            'insample_f1': insample.get('f1', 0),
            'insample_precision': insample.get('precision', 0),
            'insample_recall': insample.get('recall', 0),
            'insample_roc_auc': insample.get('roc_auc', 0),
            # Training metrics
            'train_f1': train.get('f1', 0),
            'train_roc_auc': train.get('roc_auc', 0),
            # Gaps
            'f1_gap_outsample': gaps_outsample.get('f1', 0),
            'roc_auc_gap_outsample': gaps_outsample.get('roc_auc', 0),
            # Confusion matrix (out-of-sample)
            'tp': outsample.get('confusion_matrix', {}).get('tp', 0),
            'fp': outsample.get('confusion_matrix', {}).get('fp', 0),
            'tn': outsample.get('confusion_matrix', {}).get('tn', 0),
            'fn': outsample.get('confusion_matrix', {}).get('fn', 0),
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df

def generate_report(df, output_file):
    """Generate comprehensive markdown report"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Model Selection Report V5\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # Summary
        f.write("## Summary\n\n")
        f.write(f"Total models evaluated: **{len(df)}**\n\n")
        
        # Top 10 by F1-Score
        f.write("## Top 10 Models by Out-of-Sample F1-Score\n\n")
        top_10 = df.nlargest(10, 'outsample_f1')
        
        f.write("| Rank | Model | Feature Set | F1 | Precision | Recall | ROC AUC | Gini |\n")
        f.write("|------|-------|-------------|----|-----------|--------|---------|------|\n")
        
        for idx, row in enumerate(top_10.itertuples(), 1):
            f.write(f"| {idx} | {row.model_name} | {row.feature_set_name} | "
                   f"{row.outsample_f1:.4f} | {row.outsample_precision:.4f} | "
                   f"{row.outsample_recall:.4f} | {row.outsample_roc_auc:.4f} | "
                   f"{row.outsample_gini:.4f} |\n")
        
        f.write("\n---\n\n")
        
        # Model type analysis
        f.write("## Model Type Analysis\n\n")
        model_stats = df.groupby('model_name').agg({
            'outsample_f1': ['mean', 'max', 'min', 'std'],
            'outsample_precision': 'mean',
            'outsample_recall': 'mean',
            'outsample_roc_auc': 'mean'
        }).round(4)
        
        f.write("| Model | Avg F1 | Max F1 | Min F1 | Std F1 | Avg Precision | Avg Recall | Avg ROC AUC |\n")
        f.write("|-------|--------|--------|--------|--------|---------------|------------|-------------|\n")
        
        for model_name in model_stats.index:
            stats = model_stats.loc[model_name]
            f.write(f"| {model_name} | {stats[('outsample_f1', 'mean')]:.4f} | "
                   f"{stats[('outsample_f1', 'max')]:.4f} | {stats[('outsample_f1', 'min')]:.4f} | "
                   f"{stats[('outsample_f1', 'std')]:.4f} | "
                   f"{stats[('outsample_precision', 'mean')]:.4f} | "
                   f"{stats[('outsample_recall', 'mean')]:.4f} | "
                   f"{stats[('outsample_roc_auc', 'mean')]:.4f} |\n")
        
        f.write("\n---\n\n")
        
        # Feature set analysis
        f.write("## Feature Set Analysis\n\n")
        feature_stats = df.groupby('feature_set_name').agg({
            'outsample_f1': ['mean', 'max', 'min'],
            'outsample_roc_auc': 'mean'
        }).round(4)
        
        f.write("| Feature Set | Avg F1 | Max F1 | Min F1 | Avg ROC AUC |\n")
        f.write("|-------------|--------|--------|--------|-------------|\n")
        
        for feature_set in feature_stats.index:
            stats = feature_stats.loc[feature_set]
            f.write(f"| {feature_set} | {stats[('outsample_f1', 'mean')]:.4f} | "
                   f"{stats[('outsample_f1', 'max')]:.4f} | {stats[('outsample_f1', 'min')]:.4f} | "
                   f"{stats[('outsample_roc_auc', 'mean')]:.4f} |\n")
        
        f.write("\n---\n\n")
        
        # Best model details
        f.write("## Best Model Details\n\n")
        best_model = df.loc[df['outsample_f1'].idxmax()]
        
        f.write(f"**Model:** {best_model['model_name']}\n\n")
        f.write(f"**Feature Set:** {best_model['feature_set_name']}\n\n")
        f.write("### Performance Metrics\n\n")
        f.write(f"- **F1-Score:** {best_model['outsample_f1']:.4f}\n")
        f.write(f"- **Precision:** {best_model['outsample_precision']:.4f}\n")
        f.write(f"- **Recall:** {best_model['outsample_recall']:.4f}\n")
        f.write(f"- **ROC AUC:** {best_model['outsample_roc_auc']:.4f}\n")
        f.write(f"- **Gini:** {best_model['outsample_gini']:.4f}\n")
        f.write(f"- **Accuracy:** {best_model['outsample_accuracy']:.4f}\n\n")
        
        f.write("### Confusion Matrix (Out-of-Sample)\n\n")
        f.write(f"- True Positives: {best_model['tp']}\n")
        f.write(f"- False Positives: {best_model['fp']}\n")
        f.write(f"- True Negatives: {best_model['tn']}\n")
        f.write(f"- False Negatives: {best_model['fn']}\n\n")
        
        f.write("### Overfitting Analysis\n\n")
        f.write(f"- F1 Gap (Train - Out-of-Sample): {best_model['f1_gap_outsample']:.4f}\n")
        f.write(f"- ROC AUC Gap: {best_model['roc_auc_gap_outsample']:.4f}\n\n")
        
        f.write("---\n\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        
        # Best model for precision
        best_precision = df.loc[df['outsample_precision'].idxmax()]
        f.write(f"### For High Precision Use Cases\n")
        f.write(f"- **Model:** {best_precision['model_name']} with {best_precision['feature_set_name']}\n")
        f.write(f"- **Precision:** {best_precision['outsample_precision']:.4f}\n")
        f.write(f"- **Recall:** {best_precision['outsample_recall']:.4f}\n\n")
        
        # Best model for recall
        best_recall = df.loc[df['outsample_recall'].idxmax()]
        f.write(f"### For High Recall Use Cases\n")
        f.write(f"- **Model:** {best_recall['model_name']} with {best_recall['feature_set_name']}\n")
        f.write(f"- **Precision:** {best_recall['outsample_precision']:.4f}\n")
        f.write(f"- **Recall:** {best_recall['outsample_recall']:.4f}\n\n")
        
        # Best model for balanced
        f.write(f"### For Balanced Performance (F1-Score)\n")
        f.write(f"- **Model:** {best_model['model_name']} with {best_model['feature_set_name']}\n")
        f.write(f"- **F1-Score:** {best_model['outsample_f1']:.4f}\n")
        f.write(f"- **Precision:** {best_model['outsample_precision']:.4f}\n")
        f.write(f"- **Recall:** {best_model['outsample_recall']:.4f}\n\n")
        
        f.write("---\n\n")
        f.write("## Full Results Table\n\n")
        f.write("(Sorted by Out-of-Sample F1-Score)\n\n")
        
        # Full table (top 50)
        df_sorted = df.sort_values('outsample_f1', ascending=False).head(50)
        f.write(df_sorted[['model_name', 'feature_set_name', 'outsample_f1', 
                          'outsample_precision', 'outsample_recall', 'outsample_roc_auc']].to_markdown(index=False))
        
        f.write("\n\n---\n\n")
        f.write("*Report generated by compare_optimized_models.py*\n")

def main():
    print("=" * 80)
    print("COMPARE OPTIMIZED MODELS")
    print("=" * 80)
    
    # Load config
    config_file = 'config/model_selection_config.json'
    if not os.path.exists(config_file):
        config_file = f'scripts/{config_file}'
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    models_dir = config['paths']['models_dir']
    analysis_dir = config['paths']['analysis_dir']
    
    # Load all metrics
    print("\n📂 Loading model metrics...")
    all_results = load_all_metrics(models_dir)
    
    if len(all_results) == 0:
        print("❌ No model metrics found. Please run model training first.")
        return
    
    print(f"✅ Loaded {len(all_results)} model results")
    
    # Create comparison DataFrame
    print("\n🔧 Creating comparison DataFrame...")
    df = create_comparison_dataframe(all_results)
    
    print(f"✅ Created DataFrame with {len(df)} models")
    
    # Save DataFrame to CSV
    csv_file = f"{analysis_dir}/model_comparison_results.csv"
    os.makedirs(analysis_dir, exist_ok=True)
    df.to_csv(csv_file, index=False)
    print(f"✅ Saved comparison CSV to {csv_file}")
    
    # Generate report
    print("\n📊 Generating report...")
    report_file = f"{analysis_dir}/model_selection_report_v5.md"
    generate_report(df, report_file)
    print(f"✅ Generated report: {report_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"\n📊 Top 5 Models by F1-Score:")
    top_5 = df.nlargest(5, 'outsample_f1')
    for idx, row in enumerate(top_5.itertuples(), 1):
        print(f"   {idx}. {row.model_name} + {row.feature_set_name}")
        print(f"      F1: {row.outsample_f1:.4f}, Precision: {row.outsample_precision:.4f}, "
              f"Recall: {row.outsample_recall:.4f}, ROC AUC: {row.outsample_roc_auc:.4f}")
    
    print(f"\n✅ Analysis complete!")
    print(f"   CSV: {csv_file}")
    print(f"   Report: {report_file}")

if __name__ == "__main__":
    main()


