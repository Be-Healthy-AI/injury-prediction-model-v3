#!/usr/bin/env python3
"""
Compare Phase 1 results with baseline models
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
    print("PHASE 1 RESULTS COMPARISON")
    print("="*80)
    
    # Load baseline metrics
    baseline_metrics = {}
    phase1_metrics = {}
    
    for model_name in ['rf', 'gb', 'lr']:
        # Baseline (v4)
        baseline_file = f'models/{model_name}_model_v4_metrics.json'
        if Path(baseline_file).exists():
            with open(baseline_file, 'r') as f:
                baseline_metrics[model_name] = json.load(f)
        
        # Phase 1
        phase1_file = f'models/{model_name}_model_phase1_metrics.json'
        if Path(phase1_file).exists():
            with open(phase1_file, 'r') as f:
                phase1_metrics[model_name] = json.load(f)
    
    # Create comparison table
    print("\n📊 OUT-OF-SAMPLE VALIDATION COMPARISON")
    print("="*80)
    
    comparison_data = []
    
    for model_name in ['rf', 'gb', 'lr']:
        if model_name in baseline_metrics and model_name in phase1_metrics:
            baseline = baseline_metrics[model_name]['validation_outsample']
            phase1 = phase1_metrics[model_name]['validation_outsample']
            
            comparison_data.append({
                'Model': model_name.upper(),
                'Metric': 'Precision',
                'Baseline': baseline['precision'],
                'Phase1': phase1['precision'],
                'Change': phase1['precision'] - baseline['precision'],
                'Change %': ((phase1['precision'] - baseline['precision']) / baseline['precision'] * 100) if baseline['precision'] > 0 else 0
            })
            
            comparison_data.append({
                'Model': model_name.upper(),
                'Metric': 'Recall',
                'Baseline': baseline['recall'],
                'Phase1': phase1['recall'],
                'Change': phase1['recall'] - baseline['recall'],
                'Change %': ((phase1['recall'] - baseline['recall']) / baseline['recall'] * 100) if baseline['recall'] > 0 else 0
            })
            
            comparison_data.append({
                'Model': model_name.upper(),
                'Metric': 'F1-Score',
                'Baseline': baseline['f1'],
                'Phase1': phase1['f1'],
                'Change': phase1['f1'] - baseline['f1'],
                'Change %': ((phase1['f1'] - baseline['f1']) / baseline['f1'] * 100) if baseline['f1'] > 0 else 0
            })
            
            comparison_data.append({
                'Model': model_name.upper(),
                'Metric': 'ROC AUC',
                'Baseline': baseline['roc_auc'],
                'Phase1': phase1['roc_auc'],
                'Change': phase1['roc_auc'] - baseline['roc_auc'],
                'Change %': ((phase1['roc_auc'] - baseline['roc_auc']) / baseline['roc_auc'] * 100) if baseline['roc_auc'] > 0 else 0
            })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    print("\n" + df_comparison.to_string(index=False))
    
    # Summary by model
    print("\n" + "="*80)
    print("📈 SUMMARY BY MODEL")
    print("="*80)
    
    for model_name in ['rf', 'gb', 'lr']:
        if model_name in baseline_metrics and model_name in phase1_metrics:
            baseline = baseline_metrics[model_name]['validation_outsample']
            phase1 = phase1_metrics[model_name]['validation_outsample']
            
            print(f"\n{model_name.upper()}:")
            print(f"   Precision: {baseline['precision']:.4f} → {phase1['precision']:.4f} "
                  f"({phase1['precision'] - baseline['precision']:+.4f}, "
                  f"{((phase1['precision'] - baseline['precision']) / baseline['precision'] * 100):+.1f}%)")
            print(f"   Recall: {baseline['recall']:.4f} → {phase1['recall']:.4f} "
                  f"({phase1['recall'] - baseline['recall']:+.4f}, "
                  f"{((phase1['recall'] - baseline['recall']) / baseline['recall'] * 100):+.1f}%)")
            print(f"   F1-Score: {baseline['f1']:.4f} → {phase1['f1']:.4f} "
                  f"({phase1['f1'] - baseline['f1']:+.4f}, "
                  f"{((phase1['f1'] - baseline['f1']) / baseline['f1'] * 100):+.1f}%)")
            print(f"   ROC AUC: {baseline['roc_auc']:.4f} → {phase1['roc_auc']:.4f} "
                  f"({phase1['roc_auc'] - baseline['roc_auc']:+.4f}, "
                  f"{((phase1['roc_auc'] - baseline['roc_auc']) / baseline['roc_auc'] * 100):+.1f}%)")
    
    # Save comparison
    output_dir = Path('experiments')
    output_dir.mkdir(exist_ok=True)
    
    # Save as CSV
    csv_file = output_dir / 'phase1_comparison.csv'
    df_comparison.to_csv(csv_file, index=False)
    print(f"\n✅ Comparison saved to {csv_file}")
    
    # Save as markdown
    md_file = output_dir / 'phase1_comparison_summary.md'
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# Phase 1 Results Comparison\n\n")
        f.write("**Strategy:** Stable Features (drift < 0.10) + Increased Regularization\n\n")
        f.write("## Out-of-Sample Validation Comparison\n\n")
        f.write("| Model | Metric | Baseline | Phase 1 | Change | Change % |\n")
        f.write("|-------|--------|----------|---------|--------|----------|\n")
        
        for _, row in df_comparison.iterrows():
            f.write(f"| {row['Model']} | {row['Metric']} | {row['Baseline']:.4f} | "
                   f"{row['Phase1']:.4f} | {row['Change']:+.4f} | {row['Change %']:+.1f}% |\n")
        
        f.write("\n## Key Findings\n\n")
        
        # Calculate overall improvements
        rf_f1_change = df_comparison[(df_comparison['Model'] == 'RF') & (df_comparison['Metric'] == 'F1-Score')]['Change'].values[0]
        gb_f1_change = df_comparison[(df_comparison['Model'] == 'GB') & (df_comparison['Metric'] == 'F1-Score')]['Change'].values[0]
        lr_f1_change = df_comparison[(df_comparison['Model'] == 'LR') & (df_comparison['Metric'] == 'F1-Score')]['Change'].values[0]
        
        f.write(f"- **RF F1-Score:** {rf_f1_change:+.4f} change\n")
        f.write(f"- **GB F1-Score:** {gb_f1_change:+.4f} change\n")
        f.write(f"- **LR F1-Score:** {lr_f1_change:+.4f} change\n")
        
        if rf_f1_change > 0 or gb_f1_change > 0:
            f.write("\n✅ **Phase 1 shows improvement in tree-based models!**\n")
        else:
            f.write("\n⚠️ **Phase 1 shows mixed results - may need further optimization.**\n")
    
    print(f"✅ Summary saved to {md_file}")

if __name__ == "__main__":
    main()



