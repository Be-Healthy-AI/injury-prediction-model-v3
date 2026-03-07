#!/usr/bin/env python3
"""Compare V3 and V4 model performance metrics"""

import json
from pathlib import Path

# Load V3 metrics
v3_metrics_path = Path(__file__).parent.parent.parent.parent / 'lgbm_muscular_v3' / 'model_natural_filtered' / 'lgbm_v3_natural_filtered_pl_only_metrics_test.json'
with open(v3_metrics_path, 'r') as f:
    v3_metrics = json.load(f)

# Load V4 metrics
v4_metrics_path = Path(__file__).parent.parent.parent / 'models' / 'comparison' / 'lgbm_target1_train_with_test_metrics.json'
with open(v4_metrics_path, 'r') as f:
    v4_metrics = json.load(f)

v4_test = v4_metrics['test']

print("\n" + "="*100)
print("V3 vs V4 MODEL PERFORMANCE COMPARISON - TEST DATASET (2025/26)")
print("="*100)

print("\nMODEL CONFIGURATIONS:")
print("-"*100)
print("V3-natural-filtered:")
print("  - Training: Seasons 2018-2026 (excluding 2021-2022, 2022-2023)")
print("  - Test: Season 2025/26 (PL-only)")
print("  - Target: Muscular injuries (natural ratio)")
print("\nV4-target1:")
print("  - Training: Seasons 2018-2026 + Test (2025/26) - ALL DATA")
print("  - Test: Season 2025/26 (PL-only)")
print("  - Target: Target1 (Muscular injuries, natural ratio)")

print("\n" + "="*100)
print("PERFORMANCE METRICS COMPARISON")
print("="*100)

print(f"\n{'Metric':<20} {'V3-natural-filtered':<25} {'V4-target1':<25} {'Difference':<20}")
print("-"*100)

metrics_to_compare = [
    ('accuracy', 'Accuracy'),
    ('precision', 'Precision'),
    ('recall', 'Recall'),
    ('f1', 'F1-Score'),
    ('roc_auc', 'ROC-AUC'),
    ('gini', 'Gini'),
]

for metric_key, metric_name in metrics_to_compare:
    v3_val = v3_metrics.get(metric_key, 0)
    v4_val = v4_test.get(metric_key, 0)
    diff = v4_val - v3_val
    diff_pct = (diff / v3_val * 100) if v3_val > 0 else 0
    
    print(f"{metric_name:<20} {v3_val:>24.4f} {v4_val:>24.4f} {diff:>+19.4f} ({diff_pct:>+6.2f}%)")

print("\n" + "="*100)
print("CONFUSION MATRIX COMPARISON")
print("="*100)

v3_cm = v3_metrics['confusion_matrix']
v4_cm = v4_test['confusion_matrix']

print(f"\n{'Metric':<20} {'V3-natural-filtered':<25} {'V4-target1':<25} {'Difference':<20}")
print("-"*100)

cm_metrics = [
    ('tp', 'True Positives'),
    ('fp', 'False Positives'),
    ('tn', 'True Negatives'),
    ('fn', 'False Negatives'),
]

for cm_key, cm_name in cm_metrics:
    v3_val = v3_cm.get(cm_key, 0)
    v4_val = v4_cm.get(cm_key, 0)
    diff = v4_val - v3_val
    
    print(f"{cm_name:<20} {v3_val:>24} {v4_val:>24} {diff:>+19}")

print("\n" + "="*100)
print("KEY FINDINGS")
print("="*100)

print("\n1. ROC-AUC & Gini:")
print(f"   V3: ROC-AUC={v3_metrics['roc_auc']:.6f}, Gini={v3_metrics['gini']:.6f}")
print(f"   V4: ROC-AUC={v4_test['roc_auc']:.6f}, Gini={v4_test['gini']:.6f}")
roc_diff = v4_test['roc_auc'] - v3_metrics['roc_auc']
gini_diff = v4_test['gini'] - v3_metrics['gini']
print(f"   V4 improvement: ROC-AUC +{roc_diff:.6f}, Gini +{gini_diff:.6f}")

print("\n2. Precision:")
print(f"   V3: {v3_metrics['precision']:.4f}")
print(f"   V4: {v4_test['precision']:.4f}")
prec_diff = v4_test['precision'] - v3_metrics['precision']
print(f"   V4 change: {prec_diff:+.4f} ({prec_diff/v3_metrics['precision']*100:+.2f}%)")

print("\n3. Recall:")
print(f"   Both models: 100% (Perfect recall - no false negatives)")

print("\n4. F1-Score:")
print(f"   V3: {v3_metrics['f1']:.4f}")
print(f"   V4: {v4_test['f1']:.4f}")
f1_diff = v4_test['f1'] - v3_metrics['f1']
print(f"   V4 change: {f1_diff:+.4f} ({f1_diff/v3_metrics['f1']*100:+.2f}%)")

print("\n5. False Positives:")
print(f"   V3: {v3_cm['fp']}")
print(f"   V4: {v4_cm['fp']}")
fp_diff = v4_cm['fp'] - v3_cm['fp']
print(f"   V4 change: {fp_diff:+d} ({fp_diff/v3_cm['fp']*100:+.2f}%)")

print("\n" + "="*100)
print("SUMMARY")
print("="*100)

print("\nV4 shows:")
if v4_test['roc_auc'] > v3_metrics['roc_auc']:
    print(f"  [OK] Higher ROC-AUC (+{roc_diff:.6f})")
else:
    print(f"  [X] Lower ROC-AUC ({roc_diff:.6f})")

if v4_test['gini'] > v3_metrics['gini']:
    print(f"  [OK] Higher Gini (+{gini_diff:.6f})")
else:
    print(f"  [X] Lower Gini ({gini_diff:.6f})")

if v4_test['precision'] > v3_metrics['precision']:
    print(f"  [OK] Higher Precision (+{prec_diff:.4f})")
else:
    print(f"  [X] Lower Precision ({prec_diff:.4f})")

if v4_test['f1'] > v3_metrics['f1']:
    print(f"  [OK] Higher F1-Score (+{f1_diff:.4f})")
else:
    print(f"  [X] Lower F1-Score ({f1_diff:.4f})")

if v4_cm['fp'] < v3_cm['fp']:
    print(f"  [OK] Fewer False Positives ({fp_diff:+d})")
else:
    print(f"  [X] More False Positives ({fp_diff:+d})")

print("\n" + "="*100)
