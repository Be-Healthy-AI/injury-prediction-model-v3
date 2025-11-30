#!/usr/bin/env python3
"""
Parallel Model Selection Execution Script
Runs model training in parallel for all feature set and model combinations
"""
import sys
import os
# Note: Not wrapping stdout/stderr here to avoid conflicts with parallel execution
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from tqdm import tqdm

# Import training functions
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
from train_models_optimized_v5 import (
    prepare_data, train_model_with_optuna,
    XGBOOST_AVAILABLE, LIGHTGBM_AVAILABLE, CATBOOST_AVAILABLE
)

def load_data_and_config():
    """Load data and configuration"""
    # Load config
    config_file = 'config/model_selection_config.json'
    if not os.path.exists(config_file):
        config_file = f'scripts/{config_file}'
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Load timelines
    train_file = 'timelines_35day_enhanced_balanced_v4_train.csv'
    val_file = 'timelines_35day_enhanced_balanced_v4_val.csv'
    
    if not os.path.exists(train_file):
        train_file = f'scripts/{train_file}'
    if not os.path.exists(val_file):
        val_file = f'scripts/{val_file}'
    
    df_train_full = pd.read_csv(train_file, encoding='utf-8-sig')
    df_val_outsample = pd.read_csv(val_file, encoding='utf-8-sig')
    
    # Prepare base data
    X_train_full_base, y_train_full, _ = prepare_data(df_train_full)
    X_val_outsample_base, y_val_outsample, _ = prepare_data(df_val_outsample)
    
    # Split training set
    X_train_base, X_val_insample_base, y_train, y_val_insample = train_test_split(
        X_train_full_base, y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=y_train_full
    )
    
    # Align columns
    all_cols = sorted(set(X_train_base.columns) | set(X_val_insample_base.columns) | set(X_val_outsample_base.columns))
    X_train_base = X_train_base.reindex(columns=all_cols, fill_value=0)
    X_val_insample_base = X_val_insample_base.reindex(columns=all_cols, fill_value=0)
    X_val_outsample_base = X_val_outsample_base.reindex(columns=all_cols, fill_value=0)
    
    return (config, X_train_base, y_train, X_val_insample_base, y_val_insample,
            X_val_outsample_base, y_val_outsample)

def train_single_model(model_name, feature_set_name, feature_list,
                       X_train_base, y_train, X_val_insample_base, y_val_insample,
                       X_val_outsample_base, y_val_outsample, config):
    """Train a single model (for parallel execution)"""
    try:
        # Filter data to feature set
        available_features = [f for f in feature_list if f in X_train_base.columns]
        
        if len(available_features) == 0:
            print(f"⚠️  No features available for {model_name}_{feature_set_name}")
            return None
        
        X_train = X_train_base[available_features]
        X_val_insample = X_val_insample_base[available_features]
        X_val_outsample = X_val_outsample_base[available_features]
        
        result = train_model_with_optuna(
            model_name, feature_set_name, available_features,
            X_train, y_train, X_val_insample, y_val_insample,
            X_val_outsample, y_val_outsample, config
        )
        
        return {
            'model_name': model_name,
            'feature_set_name': feature_set_name,
            'status': 'success',
            'result': result
        }
    except Exception as e:
        print(f"❌ Error training {model_name}_{feature_set_name}: {str(e)}")
        return {
            'model_name': model_name,
            'feature_set_name': feature_set_name,
            'status': 'error',
            'error': str(e)
        }

def main():
    print("=" * 80)
    print("PARALLEL MODEL SELECTION EXECUTION")
    print("=" * 80)
    
    start_time = datetime.now()
    
    # Load data and config
    print("\n📂 Loading data and configuration...")
    (config, X_train_base, y_train, X_val_insample_base, y_val_insample,
     X_val_outsample_base, y_val_outsample) = load_data_and_config()
    
    print(f"✅ Data loaded: {len(X_train_base):,} training samples, {X_train_base.shape[1]} features")
    
    # Load feature sets
    feature_sets_dir = config['paths']['feature_sets_dir']
    if not os.path.exists(feature_sets_dir):
        print(f"❌ Feature sets directory not found: {feature_sets_dir}")
        print("   Please run feature_selection_v5.py first")
        return
    
    feature_set_files = [f for f in os.listdir(feature_sets_dir) if f.endswith('.json')]
    print(f"\n📂 Found {len(feature_set_files)} feature sets")
    
    # Load feature sets
    feature_sets = {}
    for feature_set_file in feature_set_files:
        feature_set_name = feature_set_file.replace('.json', '')
        with open(f"{feature_sets_dir}/{feature_set_file}", 'r') as f:
            feature_sets[feature_set_name] = json.load(f)
        print(f"   - {feature_set_name}: {len(feature_sets[feature_set_name])} features")
    
    # Determine model types
    model_types = ['random_forest', 'gradient_boosting', 'logistic_regression']
    if XGBOOST_AVAILABLE:
        model_types.append('xgboost')
        print("   ✅ XGBoost available")
    else:
        print("   ⚠️  XGBoost not available")
    
    if LIGHTGBM_AVAILABLE:
        model_types.append('lightgbm')
        print("   ✅ LightGBM available")
    else:
        print("   ⚠️  LightGBM not available")
    
    if CATBOOST_AVAILABLE:
        model_types.append('catboost')
        print("   ✅ CatBoost available")
    else:
        print("   ⚠️  CatBoost not available")
    
    # Create task list
    tasks = [
        (model_name, feature_set_name, feature_sets[feature_set_name])
        for feature_set_name in feature_sets.keys()
        for model_name in model_types
    ]
    
    print(f"\n📊 Total tasks: {len(tasks)}")
    print(f"   Models: {len(model_types)}")
    print(f"   Feature sets: {len(feature_sets)}")
    print(f"   Combinations: {len(tasks)}")
    
    # Check for existing results to skip
    models_dir = config['paths']['models_dir']
    os.makedirs(models_dir, exist_ok=True)
    
    existing_models = set()
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith('.joblib'):
                # Extract model_name and feature_set_name
                parts = file.replace('.joblib', '').split('_', 1)
                if len(parts) == 2:
                    existing_models.add((parts[0], parts[1]))
    
    # Filter out already completed tasks
    tasks_to_run = [
        task for task in tasks
        if (task[0], task[1]) not in existing_models
    ]
    
    print(f"\n📊 Tasks to run: {len(tasks_to_run)}")
    print(f"   Already completed: {len(tasks) - len(tasks_to_run)}")
    
    if len(tasks_to_run) == 0:
        print("\n✅ All models already trained!")
        return
    
    # Execute tasks in parallel
    print(f"\n🚀 Starting parallel execution...")
    print(f"   Using {config['parallel_execution']['n_jobs']} parallel jobs")
    
    results = Parallel(
        n_jobs=config['parallel_execution']['n_jobs'],
        backend=config['parallel_execution'].get('backend', 'loky'),
        verbose=config['parallel_execution'].get('verbose', 10)
    )(
        delayed(train_single_model)(
            model_name, feature_set_name, feature_list,
            X_train_base, y_train, X_val_insample_base, y_val_insample,
            X_val_outsample_base, y_val_outsample, config
        )
        for model_name, feature_set_name, feature_list in tqdm(tasks_to_run, desc="Training models")
    )
    
    # Summary
    print("\n" + "=" * 80)
    print("EXECUTION SUMMARY")
    print("=" * 80)
    
    successful = sum(1 for r in results if r and r.get('status') == 'success')
    failed = sum(1 for r in results if r and r.get('status') == 'error')
    
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    
    if failed > 0:
        print(f"\n⚠️  Failed tasks:")
        for r in results:
            if r and r.get('status') == 'error':
                print(f"   {r['model_name']}_{r['feature_set_name']}: {r.get('error', 'Unknown error')}")
    
    total_time = datetime.now() - start_time
    print(f"\n⏱️  Total time: {total_time}")
    print("\n🎉 PARALLEL EXECUTION COMPLETED!")
    print("\n💡 Next step: Run compare_optimized_models.py to analyze results")

if __name__ == "__main__":
    main()

