#!/usr/bin/env python3
"""
Compare V4 Baseline models vs V4 Enriched models
- Loads metrics from both versions
- Creates detailed comparison tables
- Generates markdown report
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent.parent.parent
MODELS_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'models'
OUTPUT_DIR = MODELS_DIR / 'comparison'

def load_metrics(filepath):
    """Load metrics from JSON file"""
    if not filepath.exists():
        raise FileNotFoundError(f"Metrics file not found: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_improvement(baseline, enriched):
    """Calculate absolute and percentage improvement"""
    if baseline == 0:
        return None, None
    abs_diff = enriched - baseline
    pct_diff = (abs_diff / baseline) * 100
    return abs_diff, pct_diff

def format_metric(value, decimals=4):
    """Format metric value"""
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"

def create_comparison_table(baseline_metrics, enriched_metrics):
    """Create detailed comparison table"""
    comparison_data = []
    
    models = ['model1_muscular', 'model2_skeletal']
    datasets = ['train', 'test']
    metrics_list = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'gini']
    
    for model_key in models:
        if model_key not in baseline_metrics or model_key not in enriched_metrics:
            continue
        
        baseline_model = baseline_metrics[model_key]
        enriched_model = enriched_metrics[model_key]
        
        for dataset in datasets:
            if dataset not in baseline_model or dataset not in enriched_model:
                continue
            
            baseline_data = baseline_model[dataset]
            enriched_data = enriched_model[dataset]
            
            for metric in metrics_list:
                if metric not in baseline_data or metric not in enriched_data:
                    continue
                
                baseline_val = baseline_data[metric]
                enriched_val = enriched_data[metric]
                abs_diff, pct_diff = calculate_improvement(baseline_val, enriched_val)
                
                comparison_data.append({
                    'Model': model_key.replace('model1_muscular', 'Model 1 (Muscular)').replace('model2_skeletal', 'Model 2 (Skeletal)'),
                    'Dataset': dataset.capitalize(),
                    'Metric': metric.upper(),
                    'Baseline': baseline_val,
                    'Enriched': enriched_val,
                    'Absolute Change': abs_diff if abs_diff is not None else None,
                    'Percentage Change': pct_diff if pct_diff is not None else None,
                    'Improvement': 'Yes' if (abs_diff is not None and abs_diff > 0) else ('No' if abs_diff is not None else 'N/A')
                })
    
    return pd.DataFrame(comparison_data)

def create_confusion_matrix_comparison(baseline_metrics, enriched_metrics):
    """Create confusion matrix comparison"""
    cm_data = []
    
    models = ['model1_muscular', 'model2_skeletal']
    datasets = ['train', 'test']
    
    for model_key in models:
        if model_key not in baseline_metrics or model_key not in enriched_metrics:
            continue
        
        baseline_model = baseline_metrics[model_key]
        enriched_model = enriched_metrics[model_key]
        
        for dataset in datasets:
            if dataset not in baseline_model or dataset not in enriched_model:
                continue
            
            baseline_cm = baseline_model[dataset].get('confusion_matrix', {})
            enriched_cm = enriched_model[dataset].get('confusion_matrix', {})
            
            for cm_key in ['tp', 'fp', 'tn', 'fn']:
                baseline_val = baseline_cm.get(cm_key, 0)
                enriched_val = enriched_cm.get(cm_key, 0)
                abs_diff = enriched_val - baseline_val
                
                cm_data.append({
                    'Model': model_key.replace('model1_muscular', 'Model 1 (Muscular)').replace('model2_skeletal', 'Model 2 (Skeletal)'),
                    'Dataset': dataset.capitalize(),
                    'Metric': cm_key.upper(),
                    'Baseline': baseline_val,
                    'Enriched': enriched_val,
                    'Change': abs_diff
                })
    
    return pd.DataFrame(cm_data)

def generate_markdown_report(comparison_df, cm_df, baseline_metrics, enriched_metrics):
    """Generate markdown comparison report"""
    lines = []
    
    lines.append("# V4 Baseline vs Enriched Models Comparison")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("This report compares the performance of V4 baseline models (trained on original features) vs V4 enriched models (trained on Layer 2 enriched features).")
    lines.append("")
    
    # Configuration comparison
    lines.append("## Configuration")
    lines.append("")
    baseline_config = baseline_metrics.get('configuration', {})
    enriched_config = enriched_metrics.get('configuration', {})
    
    lines.append("| Setting | Baseline | Enriched |")
    lines.append("|---------|----------|----------|")
    lines.append(f"| Training Date | {baseline_config.get('training_date', 'N/A')} | {enriched_config.get('training_date', 'N/A')} |")
    lines.append(f"| Min Season | {baseline_config.get('min_season', 'N/A')} | {enriched_config.get('min_season', 'N/A')} |")
    lines.append(f"| Features | Original | {enriched_config.get('features', 'Enriched')} |")
    lines.append(f"| Model 1 Features | N/A | {enriched_config.get('num_features_model1', 'N/A')} |")
    lines.append(f"| Model 2 Features | N/A | {enriched_config.get('num_features_model2', 'N/A')} |")
    lines.append("")
    
    # Performance metrics comparison
    lines.append("## Performance Metrics Comparison")
    lines.append("")
    
    for model_key in ['model1_muscular', 'model2_skeletal']:
        model_name = model_key.replace('model1_muscular', 'Model 1 (Muscular)').replace('model2_skeletal', 'Model 2 (Skeletal)')
        lines.append(f"### {model_name}")
        lines.append("")
        
        for dataset in ['train', 'test']:
            dataset_name = dataset.capitalize()
            lines.append(f"#### {dataset_name} Dataset")
            lines.append("")
            lines.append("| Metric | Baseline | Enriched | Change | % Change |")
            lines.append("|--------|----------|----------|--------|----------|")
            
            model_df = comparison_df[
                (comparison_df['Model'] == model_name) & 
                (comparison_df['Dataset'] == dataset_name)
            ]
            
            for _, row in model_df.iterrows():
                metric = row['Metric']
                baseline = format_metric(row['Baseline'])
                enriched = format_metric(row['Enriched'])
                abs_change = format_metric(row['Absolute Change']) if row['Absolute Change'] is not None else "N/A"
                pct_change = format_metric(row['Percentage Change'], 2) if row['Percentage Change'] is not None else "N/A"
                
                lines.append(f"| {metric} | {baseline} | {enriched} | {abs_change} | {pct_change}% |")
            
            lines.append("")
    
    # Confusion matrix comparison
    lines.append("## Confusion Matrix Comparison")
    lines.append("")
    
    for model_key in ['model1_muscular', 'model2_skeletal']:
        model_name = model_key.replace('model1_muscular', 'Model 1 (Muscular)').replace('model2_skeletal', 'Model 2 (Skeletal)')
        lines.append(f"### {model_name}")
        lines.append("")
        
        for dataset in ['train', 'test']:
            dataset_name = dataset.capitalize()
            lines.append(f"#### {dataset_name} Dataset")
            lines.append("")
            lines.append("| Metric | Baseline | Enriched | Change |")
            lines.append("|--------|----------|----------|--------|")
            
            model_cm_df = cm_df[
                (cm_df['Model'] == model_name) & 
                (cm_df['Dataset'] == dataset_name)
            ]
            
            for _, row in model_cm_df.iterrows():
                metric = row['Metric']
                baseline = row['Baseline']
                enriched = row['Enriched']
                change = row['Change']
                
                lines.append(f"| {metric} | {baseline:,} | {enriched:,} | {change:+,} |")
            
            lines.append("")
    
    # Summary
    lines.append("## Summary")
    lines.append("")
    
    # Calculate key improvements
    test_df = comparison_df[comparison_df['Dataset'] == 'Test']
    
    for metric in ['PRECISION', 'F1', 'ROC_AUC', 'GINI']:
        metric_df = test_df[test_df['Metric'] == metric]
        if len(metric_df) > 0:
            avg_improvement = metric_df['Percentage Change'].mean()
            if avg_improvement is not None and not pd.isna(avg_improvement):
                lines.append(f"- **Average {metric} improvement (Test):** {avg_improvement:.2f}%")
    
    lines.append("")
    lines.append("## Key Findings")
    lines.append("")
    lines.append("1. **Feature Count:** Enriched models use Layer 2 features (workload, recovery, injury history)")
    lines.append("2. **Training:** Both models use same hyperparameters and training configuration")
    lines.append("3. **Test Set:** Both evaluated on 2025/26 season test dataset")
    lines.append("")
    
    return "\n".join(lines)

def main():
    print("="*80)
    print("V4 BASELINE vs ENRICHED MODELS COMPARISON")
    print("="*80)
    
    # Load metrics
    print("\n📂 Loading metrics files...")
    baseline_file = MODELS_DIR / 'lgbm_muscular_v4_natural_metrics.json'
    enriched_file = MODELS_DIR / 'lgbm_muscular_v4_enriched_metrics.json'
    
    baseline_metrics = load_metrics(baseline_file)
    print(f"✅ Loaded baseline metrics: {baseline_file.name}")
    
    enriched_metrics = load_metrics(enriched_file)
    print(f"✅ Loaded enriched metrics: {enriched_file.name}")
    
    # Create comparison tables
    print("\n📊 Creating comparison tables...")
    comparison_df = create_comparison_table(baseline_metrics, enriched_metrics)
    cm_df = create_confusion_matrix_comparison(baseline_metrics, enriched_metrics)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save comparison tables
    comparison_csv = OUTPUT_DIR / 'v4_baseline_vs_enriched_comparison.csv'
    comparison_df.to_csv(comparison_csv, index=False, encoding='utf-8-sig')
    print(f"✅ Saved comparison table: {comparison_csv}")
    
    cm_csv = OUTPUT_DIR / 'v4_baseline_vs_enriched_confusion_matrix.csv'
    cm_df.to_csv(cm_csv, index=False, encoding='utf-8-sig')
    print(f"✅ Saved confusion matrix comparison: {cm_csv}")
    
    # Generate markdown report
    print("\n📝 Generating markdown report...")
    markdown_report = generate_markdown_report(comparison_df, cm_df, baseline_metrics, enriched_metrics)
    
    report_path = OUTPUT_DIR / 'v4_baseline_vs_enriched_comparison.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(markdown_report)
    print(f"✅ Saved markdown report: {report_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    print("\n📊 Key Metrics Comparison (Test Dataset):")
    print("-" * 80)
    
    test_df = comparison_df[comparison_df['Dataset'] == 'Test']
    for model_name in test_df['Model'].unique():
        print(f"\n{model_name}:")
        model_test_df = test_df[test_df['Model'] == model_name]
        for metric in ['PRECISION', 'F1', 'ROC_AUC', 'GINI']:
            metric_row = model_test_df[model_test_df['Metric'] == metric]
            if len(metric_row) > 0:
                row = metric_row.iloc[0]
                baseline = row['Baseline']
                enriched = row['Enriched']
                pct_change = row['Percentage Change']
                improvement = "↑" if pct_change and pct_change > 0 else "↓" if pct_change and pct_change < 0 else "="
                print(f"  {metric:12s}: {baseline:.4f} → {enriched:.4f} ({pct_change:+.2f}%) {improvement}")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print(f"\n📁 Output files saved to: {OUTPUT_DIR}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
