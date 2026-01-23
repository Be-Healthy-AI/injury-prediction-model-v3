#!/usr/bin/env python3
"""Display ensemble optimization results in a formatted table"""

import pandas as pd
from pathlib import Path

# Load results
results_file = Path(__file__).parent.parent.parent / 'models' / 'comparison' / 'ensemble_optimization' / 'ensemble_optimization_results.csv'
df = pd.read_csv(results_file)

print("\n" + "="*140)
print("ENSEMBLE OPTIMIZATION RESULTS - COMPARISON TABLE")
print("="*140)

# Best individual models
print("\nBEST INDIVIDUAL MODELS:")
print("-"*140)
individual_best = df[df['n_models'] == 1].sort_values('test_roc_auc', ascending=False).head(3)
display_cols = ['ensemble_name', 'method', 'test_roc_auc', 'test_gini', 'test_f1', 'test_precision', 'test_recall', 'test_accuracy']
print(individual_best[display_cols].to_string(index=False))

# Top 15 ensembles
print("\nTOP 15 ENSEMBLE COMBINATIONS:")
print("-"*140)
top_15 = df.sort_values('test_roc_auc', ascending=False).head(15)
display_cols = ['ensemble_name', 'n_models', 'method', 'test_roc_auc', 'test_gini', 'test_f1', 'test_precision', 'test_recall', 'test_accuracy']
print(top_15[display_cols].to_string(index=False))

# Improvement summary
print("\nIMPROVEMENT SUMMARY:")
print("-"*140)
best_individual = individual_best.iloc[0]['test_roc_auc']
best_ensemble = top_15.iloc[0]['test_roc_auc']
print(f"   Best Individual Model ROC-AUC: {best_individual:.4f}")
print(f"   Best Ensemble ROC-AUC: {best_ensemble:.4f}")
print(f"   Improvement: +{(best_ensemble - best_individual):.4f} ({(best_ensemble/best_individual - 1)*100:.2f}%)")
print(f"   Best Individual Model Gini: {individual_best.iloc[0]['test_gini']:.4f}")
print(f"   Best Ensemble Gini: {top_15.iloc[0]['test_gini']:.4f}")
print(f"   Gini Improvement: +{(top_15.iloc[0]['test_gini'] - individual_best.iloc[0]['test_gini']):.4f}")

# Best by method
print("\nBEST ENSEMBLE BY METHOD:")
print("-"*140)
best_by_method = df.sort_values('test_roc_auc', ascending=False).groupby('method').first().reset_index()
print(best_by_method[display_cols].to_string(index=False))

print("\n" + "="*140)
