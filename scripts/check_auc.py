import json

with open('experiments/ensemble_combined_threshold_optimization.json', 'r') as f:
    data = json.load(f)

results = []
for k, v in data['best_operating_points'].items():
    auc = v['best_f1'].get('roc_auc')
    if auc is not None:
        results.append((k, auc))

results.sort(key=lambda x: x[1], reverse=True)

print("Ensemble Models by AUC (highest first):\n")
for k, v in results:
    name = k.replace('_', ' ').title()
    print(f"{name}: {v:.6f}")

print(f"\n🏆 Highest AUC: {results[0][0].replace('_', ' ').title()} with AUC = {results[0][1]:.6f}")

