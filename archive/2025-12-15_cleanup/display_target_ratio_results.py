#!/usr/bin/env python3
"""Display target ratio downsampling test results"""
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import json

with open('experiments/target_ratio_downsampling_test_results.json', 'r') as f:
    results = json.load(f)

print("="*100)
print("TARGET RATIO DOWNSAMPLING TEST RESULTS - SUMMARY")
print("="*100)

print("\n📊 TEST SET PERFORMANCE (>= 2025-07-01)")
print("-"*100)
print(f"{'Ratio':<8} {'Model':<6} {'F1-Score':<12} {'Precision':<12} {'Recall':<12} {'ROC AUC':<12} {'Gini':<12}")
print("-"*100)

for ratio in ['8%', '6%', '4%', '2%']:
    for model in ['RF', 'GB']:
        if ratio in results and model in results[ratio]:
            metrics = results[ratio][model]['test']
            print(f"{ratio:<8} {model:<6} {metrics['f1']:<12.4f} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['roc_auc']:<12.4f} {metrics['gini']:<12.4f}")

print("\n" + "="*100)
print("📊 DETAILED TEST SET METRICS (with Confusion Matrix)")
print("="*100)
print(f"{'Ratio':<8} {'Model':<6} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'ROC AUC':<12} {'TP':<8} {'FP':<8} {'TN':<8} {'FN':<8}")
print("-"*135)

for ratio in ['8%', '6%', '4%', '2%']:
    for model in ['RF', 'GB']:
        if ratio in results and model in results[ratio]:
            metrics = results[ratio][model]['test']
            cm = metrics['confusion_matrix']
            print(f"{ratio:<8} {model:<6} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1']:<12.4f} {metrics['roc_auc']:<12.4f} {cm['tp']:<8} {cm['fp']:<8} {cm['tn']:<8} {cm['fn']:<8}")

print("\n" + "="*100)
print("📈 KEY FINDINGS")
print("="*100)

# Find best F1 for each model
best_rf = max([(ratio, results[ratio]['RF']['test']['f1']) for ratio in ['8%', '6%', '4%', '2%']], key=lambda x: x[1])
best_gb = max([(ratio, results[ratio]['GB']['test']['f1']) for ratio in ['8%', '6%', '4%', '2%']], key=lambda x: x[1])

print(f"\n✅ Best RF F1-Score: {best_rf[1]:.4f} at {best_rf[0]} ratio")
print(f"✅ Best GB F1-Score: {best_gb[1]:.4f} at {best_gb[0]} ratio")

# Compare with baseline (8%)
baseline_rf_f1 = results['8%']['RF']['test']['f1']
baseline_gb_f1 = results['8%']['GB']['test']['f1']

print(f"\n📊 Comparison with 8% baseline:")
for ratio in ['6%', '4%', '2%']:
    rf_f1 = results[ratio]['RF']['test']['f1']
    gb_f1 = results[ratio]['GB']['test']['f1']
    rf_change = ((rf_f1 - baseline_rf_f1) / baseline_rf_f1) * 100 if baseline_rf_f1 > 0 else 0
    gb_change = ((gb_f1 - baseline_gb_f1) / baseline_gb_f1) * 100 if baseline_gb_f1 > 0 else 0
    print(f"   {ratio}: RF F1 = {rf_f1:.4f} ({rf_change:+.1f}%), GB F1 = {gb_f1:.4f} ({gb_change:+.1f}%)")

print("\n" + "="*100)

