#!/usr/bin/env python3
"""
Quick script to check if iterative training process is running
"""
import os
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent.parent.parent
MODEL_OUTPUT_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'models'
RESULTS_FILE = MODEL_OUTPUT_DIR / 'iterative_feature_selection_results.json'

print("="*80)
print("ITERATIVE TRAINING STATUS CHECK")
print("="*80)

# Check if results file exists
if RESULTS_FILE.exists():
    print(f"\n✅ Results file exists: {RESULTS_FILE}")
    try:
        with open(RESULTS_FILE, 'r') as f:
            data = json.load(f)
        
        if 'iterations' in data:
            n_iterations = len(data['iterations'])
            print(f"   📊 Completed iterations: {n_iterations}")
            
            if n_iterations > 0:
                last_iter = data['iterations'][-1]
                print(f"   📈 Last iteration: {last_iter.get('iteration', 'N/A')}")
                print(f"   🔢 Features used: {last_iter.get('n_features', 'N/A')}")
                print(f"   ⏱️  Training time: {last_iter.get('training_time_seconds', 0):.1f} seconds")
                
                if 'combined_score' in last_iter:
                    print(f"   📊 Combined score: {last_iter['combined_score']:.4f}")
        
        if 'best_iteration' in data and data['best_iteration']:
            print(f"\n   🏆 Best iteration: {data['best_iteration']}")
            print(f"   🎯 Best features: {data['best_n_features']}")
            print(f"   ⭐ Best score: {data['best_combined_score']:.4f}")
        
        if 'configuration' in data and 'end_time' in data['configuration']:
            print(f"\n   ✅ Process completed at: {data['configuration']['end_time']}")
        else:
            print(f"\n   ⏳ Process still running...")
            
    except Exception as e:
        print(f"   ⚠️  Error reading results file: {e}")
else:
    print(f"\n❌ Results file not found: {RESULTS_FILE}")
    print("   This means:")
    print("   - Process hasn't started yet, OR")
    print("   - Process crashed before first iteration, OR")
    print("   - Process is still in first iteration (takes ~8-10 minutes)")

print("\n" + "="*80)
