#!/usr/bin/env python3
"""
Compare Phase 2 results with Baseline and Phase 1
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
    print("PHASE 2 RESULTS COMPARISON")
    print("="*80)
    
    # Load metrics
    baseline_metrics = {}
    phase1_metrics = {}
    phase2_metrics = {}
    
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
        
        # Phase 2
        phase2_file = f'models/{model_name}_model_phase2_metrics.json'
        if Path(phase2_file).exists():
            with open(phase2_file, 'r') as f:
                phase2_metrics[model_name] = json.load(f)
    
    # Create comparison table
    print("\n📊 OUT-OF-SAMPLE VALIDATION COMPARISON")
    print("="*80)
    
    comparison_data = []
    
    for model_name in ['rf', 'gb', 'lr']:
        if model_name in baseline_metrics and model_name in phase2_metrics:
            baseline = baseline_metrics[model_name]['validation_outsample']
            phase1 = phase1_metrics.get(model_name, {}).get('validation_outsample', {})
            phase2 = phase2_metrics[model_name]['validation_outsample']
            
            for metric in ['precision', 'recall', 'f1', 'roc_auc']:
                baseline_val = baseline.get(metric, 0)
                phase1_val = phase1.get(metric, baseline_val) if phase1 else baseline_val
                phase2_val = phase2.get(metric, 0)
                
                comparison_data.append({
                    'Model': model_name.upper(),
                    'Metric': metric.capitalize(),
                    'Baseline': baseline_val,
                    'Phase1': phase1_val,
                    'Phase2': phase2_val,
                    'Phase1 vs Baseline': phase1_val - baseline_val,
                    'Phase2 vs Baseline': phase2_val - baseline_val,
                    'Phase2 vs Phase1': phase2_val - phase1_val
                })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    print("\n" + df_comparison.to_string(index=False))
    
    # Summary by model
    print("\n" + "="*80)
    print("📈 SUMMARY BY MODEL")
    print("="*80)
    
    for model_name in ['rf', 'gb', 'lr']:
        if model_name in baseline_metrics and model_name in phase2_metrics:
            baseline = baseline_metrics[model_name]['validation_outsample']
            phase1 = phase1_metrics.get(model_name, {}).get('validation_outsample', {})
            phase2 = phase2_metrics[model_name]['validation_outsample']
            
            print(f"\n{model_name.upper()}:")
            for metric in ['precision', 'recall', 'f1', 'roc_auc']:
                baseline_val = baseline.get(metric, 0)
                phase1_val = phase1.get(metric, baseline_val) if phase1 else baseline_val
                phase2_val = phase2.get(metric, 0)
                
                print(f"   {metric.capitalize()}:")
                print(f"      Baseline: {baseline_val:.4f}")
                if phase1:
                    print(f"      Phase1:   {phase1_val:.4f} ({phase1_val - baseline_val:+.4f})")
                print(f"      Phase2:   {phase2_val:.4f} ({phase2_val - baseline_val:+.4f})")
                if phase1:
                    print(f"      Change (P2 vs P1): {phase2_val - phase1_val:+.4f}")
    
    # Save comparison
    output_dir = Path('experiments')
    output_dir.mkdir(exist_ok=True)
    
    # Save as CSV
    csv_file = output_dir / 'phase2_comparison.csv'
    df_comparison.to_csv(csv_file, index=False)
    print(f"\n✅ Comparison saved to {csv_file}")
    
    # Save as markdown
    md_file = output_dir / 'phase2_comparison_summary.md'
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# Phase 2 Results Comparison\n\n")
        f.write("**Strategy:** Low-Shift Features + Covariate Shift Correction + Calibration\n\n")
        f.write("## Out-of-Sample Validation Comparison\n\n")
        f.write("| Model | Metric | Baseline | Phase 1 | Phase 2 | P2 vs Baseline | P2 vs P1 |\n")
        f.write("|-------|--------|----------|---------|---------|----------------|----------|\n")
        
        for _, row in df_comparison.iterrows():
            f.write(f"| {row['Model']} | {row['Metric']} | {row['Baseline']:.4f} | "
                   f"{row['Phase1']:.4f} | {row['Phase2']:.4f} | "
                   f"{row['Phase2 vs Baseline']:+.4f} | {row['Phase2 vs Phase1']:+.4f} |\n")
        
        f.write("\n## Key Findings\n\n")
        
        # Calculate overall improvements
        for model_name in ['rf', 'gb', 'lr']:
            model_data = df_comparison[df_comparison['Model'] == model_name.upper()]
            f1_row = model_data[model_data['Metric'] == 'F1']
            if len(f1_row) > 0:
                f1_baseline = f1_row['Baseline'].values[0]
                f1_phase1 = f1_row['Phase1'].values[0]
                f1_phase2 = f1_row['Phase2'].values[0]
                f.write(f"### {model_name.upper()}\n\n")
                f.write(f"- **F1-Score:** Baseline={f1_baseline:.4f}, Phase1={f1_phase1:.4f}, Phase2={f1_phase2:.4f}\n")
                f.write(f"- **Improvement (P2 vs Baseline):** {f1_phase2 - f1_baseline:+.4f} ({(f1_phase2 - f1_baseline) / f1_baseline * 100:+.1f}%)\n")
                if f1_phase1 != f1_baseline:
                    f.write(f"- **Improvement (P2 vs Phase1):** {f1_phase2 - f1_phase1:+.4f} ({(f1_phase2 - f1_phase1) / f1_phase1 * 100:+.1f}%)\n")
                f.write("\n")
        
        # Overall assessment
        gb_f1_phase2 = df_comparison[(df_comparison['Model'] == 'GB') & (df_comparison['Metric'] == 'F1')]['Phase2'].values[0]
        gb_f1_baseline = df_comparison[(df_comparison['Model'] == 'GB') & (df_comparison['Metric'] == 'F1')]['Baseline'].values[0]
        
        if gb_f1_phase2 > gb_f1_baseline:
            f.write("✅ **Phase 2 shows improvement over baseline!**\n")
        else:
            f.write("⚠️ **Phase 2 shows mixed results - may need further optimization.**\n")
    
    print(f"✅ Summary saved to {md_file}")

if __name__ == "__main__":
    main()

