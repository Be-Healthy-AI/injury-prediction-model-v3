#!/usr/bin/env python3
"""
Compare Phase 2 models at default (0.5) vs optimized thresholds
"""
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import json
import pandas as pd
from pathlib import Path

def main():
    print("="*80)
    print("PHASE 2 THRESHOLD OPTIMIZATION COMPARISON")
    print("="*80)
    
    # Load Phase 2 metrics (default threshold 0.5)
    phase2_default = {}
    for model_name in ['rf', 'gb', 'lr']:
        metrics_file = f'models/{model_name}_model_phase2_metrics.json'
        if Path(metrics_file).exists():
            with open(metrics_file, 'r') as f:
                phase2_default[model_name] = json.load(f)['validation_outsample']
    
    # Load threshold optimization results
    opt_file = 'experiments/phase2_threshold_optimization.json'
    if not Path(opt_file).exists():
        print(f"❌ Threshold optimization results not found: {opt_file}")
        return
    
    with open(opt_file, 'r') as f:
        opt_results = json.load(f)
    
    df_opt = pd.DataFrame(opt_results)
    
    # Find best F1 for each model
    best_f1 = {}
    for model_name in ['RF', 'GB', 'LR']:
        model_results = df_opt[df_opt['model'] == model_name]
        if len(model_results) > 0:
            best_idx = model_results['f1'].idxmax()
            best_f1[model_name] = model_results.loc[best_idx]
    
    # Create comparison
    print("\n📊 COMPARISON: Default (0.5) vs Optimized Threshold")
    print("="*80)
    
    comparison_data = []
    
    for model_name in ['rf', 'gb', 'lr']:
        model_upper = model_name.upper()
        if model_name in phase2_default and model_upper in best_f1:
            default = phase2_default[model_name]
            optimized = best_f1[model_upper]
            
            comparison_data.append({
                'Model': model_upper,
                'Threshold': 'Default (0.5)',
                'Precision': default['precision'],
                'Recall': default['recall'],
                'F1': default['f1'],
                'ROC AUC': default['roc_auc']
            })
            
            comparison_data.append({
                'Model': model_upper,
                'Threshold': f"Optimized ({optimized['threshold']:.3f})",
                'Precision': optimized['precision'],
                'Recall': optimized['recall'],
                'F1': optimized['f1'],
                'ROC AUC': optimized['roc_auc']
            })
            
            comparison_data.append({
                'Model': model_upper,
                'Threshold': 'Improvement',
                'Precision': optimized['precision'] - default['precision'],
                'Recall': optimized['recall'] - default['recall'],
                'F1': optimized['f1'] - default['f1'],
                'ROC AUC': optimized['roc_auc'] - default['roc_auc']
            })
    
    df_comparison = pd.DataFrame(comparison_data)
    print("\n" + df_comparison.to_string(index=False))
    
    # Summary by model
    print("\n" + "="*80)
    print("📈 SUMMARY BY MODEL")
    print("="*80)
    
    for model_name in ['rf', 'gb', 'lr']:
        model_upper = model_name.upper()
        if model_name in phase2_default and model_upper in best_f1:
            default = phase2_default[model_name]
            optimized = best_f1[model_upper]
            
            print(f"\n{model_upper}:")
            print(f"   Default (0.5):")
            print(f"      Precision: {default['precision']:.4f}, Recall: {default['recall']:.4f}, F1: {default['f1']:.4f}")
            print(f"   Optimized (threshold={optimized['threshold']:.3f}):")
            print(f"      Precision: {optimized['precision']:.4f}, Recall: {optimized['recall']:.4f}, F1: {optimized['f1']:.4f}")
            print(f"   Improvement:")
            print(f"      Precision: {optimized['precision'] - default['precision']:+.4f} "
                  f"({((optimized['precision'] - default['precision']) / default['precision'] * 100):+.1f}%)")
            print(f"      Recall: {optimized['recall'] - default['recall']:+.4f} "
                  f"({((optimized['recall'] - default['recall']) / default['recall'] * 100):+.1f}%)")
            print(f"      F1: {optimized['f1'] - default['f1']:+.4f} "
                  f"({((optimized['f1'] - default['f1']) / default['f1'] * 100):+.1f}%)")
            print(f"      Detects: {int(optimized['tp'])} injuries (vs {int(default['confusion_matrix']['tp'])} at default)")
    
    # Save comparison
    output_dir = Path('experiments')
    output_dir.mkdir(exist_ok=True)
    
    csv_file = output_dir / 'phase2_threshold_comparison.csv'
    df_comparison.to_csv(csv_file, index=False)
    print(f"\n✅ Comparison saved to {csv_file}")
    
    # Save markdown summary
    md_file = output_dir / 'phase2_threshold_comparison_summary.md'
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# Phase 2 Threshold Optimization - Comparison\n\n")
        f.write("**Comparison:** Default threshold (0.5) vs Optimized threshold\n\n")
        f.write("## Summary Table\n\n")
        f.write("| Model | Threshold | Precision | Recall | F1-Score | ROC AUC |\n")
        f.write("|-------|-----------|-----------|--------|----------|----------|\n")
        
        for _, row in df_comparison.iterrows():
            if row['Threshold'] == 'Improvement':
                f.write(f"| {row['Model']} | **{row['Threshold']}** | **{row['Precision']:+.4f}** | "
                       f"**{row['Recall']:+.4f}** | **{row['F1']:+.4f}** | {row['ROC AUC']:+.4f} |\n")
            else:
                f.write(f"| {row['Model']} | {row['Threshold']} | {row['Precision']:.4f} | "
                       f"{row['Recall']:.4f} | {row['F1']:.4f} | {row['ROC AUC']:.4f} |\n")
        
        f.write("\n## Key Findings\n\n")
        
        for model_name in ['rf', 'gb', 'lr']:
            model_upper = model_name.upper()
            if model_name in phase2_default and model_upper in best_f1:
                default = phase2_default[model_name]
                optimized = best_f1[model_upper]
                
                f.write(f"### {model_upper}\n\n")
                f.write(f"- **Optimal Threshold:** {optimized['threshold']:.3f}\n")
                f.write(f"- **F1 Improvement:** {optimized['f1'] - default['f1']:+.4f} "
                       f"({((optimized['f1'] - default['f1']) / default['f1'] * 100):+.1f}%)\n")
                f.write(f"- **Precision:** {default['precision']:.4f} → {optimized['precision']:.4f} "
                       f"({optimized['precision'] - default['precision']:+.4f})\n")
                f.write(f"- **Recall:** {default['recall']:.4f} → {optimized['recall']:.4f} "
                       f"({optimized['recall'] - default['recall']:+.4f})\n")
                f.write(f"- **Injuries Detected:** {int(default['confusion_matrix']['tp'])} → "
                       f"{int(optimized['tp'])} ({int(optimized['tp']) - int(default['confusion_matrix']['tp']):+d})\n\n")
        
        # Overall recommendation
        gb_optimized = best_f1.get('GB', None)
        if gb_optimized is not None and isinstance(gb_optimized, pd.Series):
            f.write("## Recommendation\n\n")
            f.write(f"**Use GB model at threshold {gb_optimized['threshold']:.3f}**\n\n")
            f.write(f"- **F1-Score:** {gb_optimized['f1']:.4f}\n")
            f.write(f"- **Precision:** {gb_optimized['precision']:.4f}\n")
            f.write(f"- **Recall:** {gb_optimized['recall']:.4f}\n")
            f.write(f"- **Injuries Detected:** {int(gb_optimized['tp'])} out of 220\n")
    
    print(f"✅ Summary saved to {md_file}")

if __name__ == "__main__":
    main()

