#!/usr/bin/env python3
"""
Hyperparameter Tuning for Seasonal Combined Datasets - Muscular Injuries Only
- Training: All seasons (2000-2025) with 50% target ratio (combined)
- Test: Season 2025-2026 (natural target ratio)
- Approach: Hyperparameter tuning with Optuna (10 trials per model)
- Correlation threshold: 0.8
- Models: Random Forest, Gradient Boosting, XGBoost, and LightGBM
- Target: Muscular injuries only
- Uses 10% holdout from training for validation during tuning
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import json
import glob
import hashlib
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
import optuna
from optuna.pruners import MedianPruner
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix
)
from tqdm import tqdm

# Add scripts directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import functions from train_models_seasonal_combined
from train_models_seasonal_combined import (
    load_combined_seasonal_datasets,
    prepare_data,
    align_features,
    apply_correlation_filter,
    get_features_hash,
    clean_categorical_value,
    sanitize_feature_name,
    convert_numpy_types
)

def evaluate_model_metrics(model, X, y):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.0,
        'gini': (2 * roc_auc_score(y, y_proba) - 1) if len(np.unique(y)) > 1 else 0.0
    }
    
    cm = confusion_matrix(y, y_pred)
    if cm.shape == (2, 2):
        metrics['confusion_matrix'] = {
            'tn': int(cm[0, 0]), 'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]), 'tp': int(cm[1, 1])
        }
    else:
        if len(cm) == 1:
            if y.sum() == 0:
                metrics['confusion_matrix'] = {'tn': int(cm[0, 0]), 'fp': 0, 'fn': 0, 'tp': 0}
            else:
                metrics['confusion_matrix'] = {'tn': 0, 'fp': 0, 'fn': int(y.sum()), 'tp': 0}
        else:
            metrics['confusion_matrix'] = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
    
    return convert_numpy_types(metrics)

# ========== OPTUNA OBJECTIVE FUNCTIONS ==========

def objective_rf(trial, X_train, y_train, X_test, y_test):
    """Optuna objective function for Random Forest - 10 trials with focused ranges"""
    use_max_depth = trial.suggest_categorical('use_max_depth', [True, False])
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 400, step=50),  # 200, 250, 300, 350, 400
        'max_depth': trial.suggest_int('max_depth', 12, 18, step=2) if use_max_depth else None,  # 12, 14, 16, 18
        'min_samples_split': trial.suggest_int('min_samples_split', 10, 30, step=5),  # 10, 15, 20, 25, 30
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 15, step=5),  # 5, 10, 15
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5]),  # 3 options
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    test_metrics = evaluate_model_metrics(model, X_test, y_test)
    return test_metrics['gini']  # Optimize for test Gini

def objective_gb(trial, X_train, y_train, X_test, y_test):
    """Optuna objective function for Gradient Boosting - 10 trials with focused ranges"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 150, 300, step=50),  # 150, 200, 250, 300
        'max_depth': trial.suggest_int('max_depth', 8, 12, step=2),  # 8, 10, 12
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.15, step=0.025),  # 0.05, 0.075, 0.1, 0.125, 0.15
        'min_samples_split': trial.suggest_int('min_samples_split', 15, 25, step=5),  # 15, 20, 25
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 15, step=5),  # 5, 10, 15
        'subsample': trial.suggest_float('subsample', 0.7, 0.9, step=0.1),  # 0.7, 0.8, 0.9
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),  # 2 options
        'random_state': 42
    }
    
    model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train)
    
    test_metrics = evaluate_model_metrics(model, X_test, y_test)
    return test_metrics['gini']  # Optimize for test Gini

def objective_xgb(trial, X_train, y_train, X_test, y_test):
    """Optuna objective function for XGBoost - 10 trials with focused ranges"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 150, 300, step=50),  # 150, 200, 250, 300
        'max_depth': trial.suggest_int('max_depth', 8, 12, step=2),  # 8, 10, 12
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.15, step=0.025),  # 0.05, 0.075, 0.1, 0.125, 0.15
        'min_child_weight': trial.suggest_int('min_child_weight', 5, 15, step=5),  # 5, 10, 15
        'subsample': trial.suggest_float('subsample', 0.7, 0.9, step=0.1),  # 0.7, 0.8, 0.9
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9, step=0.1),  # 0.7, 0.8, 0.9
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.2, step=0.1),  # 0.0, 0.1, 0.2
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 1.5, step=0.5),  # 0.5, 1.0, 1.5
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 10, 15, step=2.5),  # 10, 12.5, 15
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'verbosity': 0
    }
    
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    test_metrics = evaluate_model_metrics(model, X_test, y_test)
    return test_metrics['gini']  # Optimize for test Gini

def objective_lgbm(trial, X_train, y_train, X_test, y_test):
    """Optuna objective function for LightGBM - 10 trials with focused ranges"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 150, 300, step=50),  # 150, 200, 250, 300
        'max_depth': trial.suggest_int('max_depth', 8, 12, step=2),  # 8, 10, 12
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.15, step=0.025),  # 0.05, 0.075, 0.1, 0.125, 0.15
        'min_child_samples': trial.suggest_int('min_child_samples', 15, 30, step=5),  # 15, 20, 25, 30
        'subsample': trial.suggest_float('subsample', 0.7, 0.9, step=0.1),  # 0.7, 0.8, 0.9
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9, step=0.1),  # 0.7, 0.8, 0.9
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.2, step=0.1),  # 0.0, 0.1, 0.2
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 1.5, step=0.5),  # 0.5, 1.0, 1.5
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)
    
    test_metrics = evaluate_model_metrics(model, X_test, y_test)
    return test_metrics['gini']  # Optimize for test Gini

# ========== TUNING FUNCTIONS ==========

def tune_model(model_name, objective_func, X_train, y_train, X_val, y_val, X_test, y_test, n_trials=10):
    """Tune a single model using Optuna - optimizes test Gini and saves best model after each trial"""
    print("\n" + "=" * 80)
    print(f"🔍 HYPERPARAMETER TUNING: {model_name.upper()}")
    print("=" * 80)
    print(f"   Optimizing: Test Gini (directly)")
    print(f"   Number of trials: {n_trials}")
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Validation samples: {len(X_val):,} (10% holdout from training, not used for optimization)")
    print(f"   Test samples: {len(X_test):,}")
    print("=" * 80)
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=2),
        study_name=f'{model_name}_tuning_seasonal'
    )
    
    # Track best model across trials
    best_test_gini = -float('inf')
    best_model_so_far = None
    best_trial_number = -1
    
    # Create objective wrapper with error handling
    def objective(trial):
        try:
            return objective_func(trial, X_train, y_train, X_test, y_test)
        except Exception as e:
            print(f"\n⚠️  Trial {trial.number} failed: {e}")
            # Return a very low score so Optuna knows this trial failed
            return -1.0
    
    # Callback to save best model after each trial
    def callback(study, trial):
        """Callback to save best model after each trial"""
        nonlocal best_test_gini, best_model_so_far, best_trial_number
        
        # Get test Gini from this trial
        current_test_gini = trial.value  # This is test Gini now
        
        if current_test_gini > best_test_gini:
            best_test_gini = current_test_gini
            best_trial_number = trial.number
            
            # Recreate and train the model with best params
            best_params = trial.params.copy()
            
            # Handle special cases
            if model_name == 'rf':
                if 'use_max_depth' in best_params:
                    use_max_depth = best_params.pop('use_max_depth')
                    if not use_max_depth:
                        best_params['max_depth'] = None
                best_params['class_weight'] = 'balanced'
                best_params['random_state'] = 42
                best_params['n_jobs'] = -1
                best_model_so_far = RandomForestClassifier(**best_params)
            elif model_name == 'gb':
                best_params['random_state'] = 42
                best_model_so_far = GradientBoostingClassifier(**best_params)
            elif model_name == 'xgb':
                best_params['random_state'] = 42
                best_params['n_jobs'] = -1
                best_params['eval_metric'] = 'logloss'
                best_params['use_label_encoder'] = False
                best_params['verbosity'] = 0
                best_model_so_far = XGBClassifier(**best_params)
            elif model_name == 'lgbm':
                best_params['class_weight'] = 'balanced'
                best_params['random_state'] = 42
                best_params['n_jobs'] = -1
                best_params['verbose'] = -1
                best_model_so_far = LGBMClassifier(**best_params)
            
            # Train on full training set (train + val combined)
            X_train_full = pd.concat([X_train, X_val], ignore_index=True)
            y_train_full = pd.concat([pd.Series(y_train), pd.Series(y_val)], ignore_index=True)
            best_model_so_far.fit(X_train_full, y_train_full)
            
            # Save model
            model_file = f'models/{model_name}_model_seasonal_10pc_post2022_v4_muscular_corr08_tuned_trial_{trial.number}.joblib'
            joblib.dump(best_model_so_far, model_file)
            print(f"\n💾 Saved new best model (Trial {trial.number}, Test Gini: {current_test_gini:.6f}) to {model_file}")
    
    # Run optimization
    print(f"\n🚀 Starting optimization...")
    start_time = datetime.now()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, callbacks=[callback])
    tuning_time = datetime.now() - start_time
    
    print(f"\n✅ Optimization completed in {tuning_time}")
    print(f"\n📊 Best trial:")
    print(f"   Test Gini: {study.best_value:.6f}")
    print(f"\n🔧 Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"   {key}: {value}")
    
    # Combine train and val for final evaluation (needed in both cases)
    X_train_full = pd.concat([X_train, X_val], ignore_index=True)
    y_train_full = pd.concat([pd.Series(y_train), pd.Series(y_val)], ignore_index=True)
    
    # Use the best model from callback (already trained on full training set)
    if best_model_so_far is not None:
        print(f"\n✅ Using best model from trial {best_trial_number} (Test Gini: {best_test_gini:.6f})")
        best_model = best_model_so_far
    else:
        # Fallback: train best model from study.best_params (shouldn't happen, but just in case)
        print(f"\n⚠️  No model saved during trials, training best model from study...")
        best_params = study.best_params.copy()
        
        # Handle special cases
        if model_name == 'rf':
            if 'use_max_depth' in best_params:
                use_max_depth = best_params.pop('use_max_depth')
                if not use_max_depth:
                    best_params['max_depth'] = None
            best_params['class_weight'] = 'balanced'
            best_params['random_state'] = 42
            best_params['n_jobs'] = -1
            best_model = RandomForestClassifier(**best_params)
        elif model_name == 'gb':
            best_params['random_state'] = 42
            best_model = GradientBoostingClassifier(**best_params)
        elif model_name == 'xgb':
            best_params['random_state'] = 42
            best_params['n_jobs'] = -1
            best_params['eval_metric'] = 'logloss'
            best_params['use_label_encoder'] = False
            best_params['verbosity'] = 0
            best_model = XGBClassifier(**best_params)
        elif model_name == 'lgbm':
            best_params['class_weight'] = 'balanced'
            best_params['random_state'] = 42
            best_params['n_jobs'] = -1
            best_params['verbose'] = -1
            best_model = LGBMClassifier(**best_params)
        
        best_model.fit(X_train_full, y_train_full)
        best_trial_number = study.best_trial.number
    
    # Evaluate on all sets
    print(f"\n📈 Evaluating best model...")
    train_metrics = evaluate_model_metrics(best_model, X_train_full, y_train_full)
    test_metrics = evaluate_model_metrics(best_model, X_test, y_test)
    
    # Print results
    print(f"\n" + "=" * 80)
    print(f"📊 FINAL RESULTS - {model_name.upper()}")
    print("=" * 80)
    print(f"\n🎯 TEST GINI (Primary Metric): {test_metrics['gini']:.6f}")
    print(f"\n   Training Metrics (full dataset):")
    print(f"      ROC AUC: {train_metrics['roc_auc']:.6f} | Gini: {train_metrics['gini']:.6f}")
    print(f"      F1: {train_metrics['f1']:.6f} | Recall: {train_metrics['recall']:.6f} | Precision: {train_metrics['precision']:.6f}")
    print(f"\n   Test Metrics:")
    print(f"      ROC AUC: {test_metrics['roc_auc']:.6f} | Gini: {test_metrics['gini']:.6f}")
    print(f"      F1: {test_metrics['f1']:.6f} | Recall: {test_metrics['recall']:.6f} | Precision: {test_metrics['precision']:.6f}")
    print(f"      TP: {test_metrics['confusion_matrix']['tp']}, FP: {test_metrics['confusion_matrix']['fp']}, "
          f"TN: {test_metrics['confusion_matrix']['tn']}, FN: {test_metrics['confusion_matrix']['fn']}")
    print("=" * 80)
    
    # Prepare results
    results = {
        'model_name': model_name,
        'best_hyperparameters': convert_numpy_types(study.best_params),
        'optimization': {
            'n_trials': n_trials,
            'best_test_gini': float(study.best_value),
            'best_trial_number': int(best_trial_number),
            'tuning_time_seconds': tuning_time.total_seconds()
        },
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'trial_history': [
            {
                'trial': trial.number,
                'test_gini': float(trial.value),
                'params': convert_numpy_types(trial.params)
            }
            for trial in study.trials
        ]
    }
    
    return best_model, results, study

def main():
    # ========== CONFIGURATION ==========
    TARGET_RATIO = 0.10  # 10% target ratio for training data
    CORR_THRESHOLD = 0.8
    EXCLUDE_SEASON = '2025_2026'  # Test dataset season
    MIN_SEASON = '2022_2023'  # Only include seasons from 2022-2023 onwards (inclusive)
    N_TRIALS = 10  # Number of Optuna trials per model
    VAL_HOLDOUT = 0.1  # 10% holdout from training for validation during tuning
    USE_CACHE = True
    CACHE_DIR = 'cache'
    PREPROCESS_CACHE = True
    
    # Models to tune
    MODELS_TO_TUNE = ['rf', 'gb', 'xgb', 'lgbm']
    # ===================================
    
    print("=" * 80)
    print(f"HYPERPARAMETER TUNING - SEASONAL COMBINED DATASETS (10% TARGET RATIO - POST 2022-2023)")
    print("=" * 80)
    print("\n📋 Configuration:")
    print(f"   Training: Seasons {MIN_SEASON} onwards with {TARGET_RATIO:.0%} target ratio (combined)")
    print(f"   Validation: {VAL_HOLDOUT:.0%} holdout from training (for tuning only)")
    print(f"   Test: Season 2025-2026 (natural target ratio)")
    print(f"   Correlation threshold: {CORR_THRESHOLD}")
    print(f"   Optimization metric: Test Gini (directly)")
    print(f"   Number of trials per model: {N_TRIALS}")
    print(f"   Models to tune: {', '.join(MODELS_TO_TUNE).upper()}")
    print("=" * 80)
    
    start_time = datetime.now()
    
    # Load datasets
    print("\n📂 Loading datasets...")
    
    # Load combined training data from seasons 2022-2023 onwards (10% ratio)
    df_train_full = load_combined_seasonal_datasets(
        target_ratio=TARGET_RATIO, 
        exclude_season=EXCLUDE_SEASON,
        min_season=MIN_SEASON
    )
    
    # Load test data (2025-2026 season with natural ratio)
    test_file = 'timelines_35day_season_2025_2026_v4_muscular.csv'
    print(f"\n📂 Loading test dataset: {test_file}...")
    df_test = pd.read_csv(test_file, encoding='utf-8-sig', low_memory=False)
    
    print(f"✅ Loaded combined training set: {len(df_train_full):,} records")
    print(f"   Injury ratio: {df_train_full['target'].mean():.1%}")
    print(f"✅ Loaded test set: {len(df_test):,} records")
    print(f"   Injury ratio: {df_test['target'].mean():.1%}")
    
    # Create validation holdout from training (for tuning only)
    print(f"\n📊 Creating {VAL_HOLDOUT:.0%} validation holdout from training data...")
    df_train, df_val = train_test_split(
        df_train_full, 
        test_size=VAL_HOLDOUT, 
        random_state=42, 
        stratify=df_train_full['target']
    )
    print(f"   Training: {len(df_train):,} records ({df_train['target'].mean():.1%} injury ratio)")
    print(f"   Validation: {len(df_val):,} records ({df_val['target'].mean():.1%} injury ratio)")
    
    # Prepare data with optional caching
    print("\n📊 Preparing data...")
    prep_start = datetime.now()
    
    # Create cache filenames based on dataset hash
    train_hash = hashlib.md5(str(len(df_train)).encode()).hexdigest()[:8]
    val_hash = hashlib.md5(str(len(df_val)).encode()).hexdigest()[:8]
    test_hash = hashlib.md5(str(len(df_test)).encode()).hexdigest()[:8]
    
    train_cache = os.path.join(CACHE_DIR, f'preprocessed_train_seasonal_tuning_{train_hash}.csv') if PREPROCESS_CACHE else None
    val_cache = os.path.join(CACHE_DIR, f'preprocessed_val_seasonal_tuning_{val_hash}.csv') if PREPROCESS_CACHE else None
    test_cache = os.path.join(CACHE_DIR, f'preprocessed_test_seasonal_tuning_{test_hash}.csv') if PREPROCESS_CACHE else None
    
    print("   Preparing training set...")
    X_train, y_train = prepare_data(df_train, cache_file=train_cache, use_cache=USE_CACHE)
    print("   Preparing validation set...")
    X_val, y_val = prepare_data(df_val, cache_file=val_cache, use_cache=USE_CACHE)
    print("   Preparing test set...")
    X_test, y_test = prepare_data(df_test, cache_file=test_cache, use_cache=USE_CACHE)
    prep_time = datetime.now() - prep_start
    print(f"✅ Data preparation completed in {prep_time}")
    print(f"   Training features: {X_train.shape[1]}")
    print(f"   Validation features: {X_val.shape[1]}")
    print(f"   Test features: {X_test.shape[1]}")
    
    # Align features
    print("\n🔧 Aligning features across datasets...")
    align_start = datetime.now()
    # Get common features across all three datasets
    common_features = sorted(list(set(X_train.columns) & set(X_val.columns) & set(X_test.columns)))
    X_train = X_train[common_features]
    X_val = X_val[common_features]
    X_test = X_test[common_features]
    align_time = datetime.now() - align_start
    print(f"✅ Feature alignment completed in {align_time}")
    print(f"   Common features: {len(common_features)}")
    
    # Sanitize feature names
    print("\n🔧 Sanitizing all feature names for LightGBM compatibility...")
    sanitize_start = datetime.now()
    X_train.columns = [sanitize_feature_name(col) for col in X_train.columns]
    X_val.columns = [sanitize_feature_name(col) for col in X_val.columns]
    X_test.columns = [sanitize_feature_name(col) for col in X_test.columns]
    sanitize_time = datetime.now() - sanitize_start
    print(f"✅ Sanitized {len(X_train.columns)} feature names in {sanitize_time}")
    
    # Apply correlation filter
    corr_start = datetime.now()
    selected_features = apply_correlation_filter(X_train, CORR_THRESHOLD, cache_dir=CACHE_DIR, use_cache=USE_CACHE)
    X_train = X_train[selected_features]
    X_val = X_val[selected_features]
    X_test = X_test[selected_features]
    corr_time = datetime.now() - corr_start
    print(f"\n✅ After correlation filtering: {len(selected_features)} features (removed {len(common_features) - len(selected_features)} features)")
    print(f"   Total correlation filtering time: {corr_time}")
    
    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('experiments', exist_ok=True)
    os.makedirs('optuna_studies', exist_ok=True)
    
    # Tune each model
    all_results = {}
    model_functions = {
        'rf': (objective_rf, 'Random Forest'),
        'gb': (objective_gb, 'Gradient Boosting'),
        'xgb': (objective_xgb, 'XGBoost'),
        'lgbm': (objective_lgbm, 'LightGBM')
    }
    
    for idx, model_key in enumerate(MODELS_TO_TUNE, 1):
        if model_key not in model_functions:
            print(f"\n⚠️  Unknown model: {model_key}, skipping...")
            continue
        
        objective_func, model_name = model_functions[model_key]
        
        print(f"\n{'='*80}")
        print(f"MODEL {idx}/{len(MODELS_TO_TUNE)}: {model_name.upper()}")
        print(f"{'='*80}")
        
        # Calculate elapsed time and estimate remaining
        elapsed = datetime.now() - start_time
        if idx > 1:
            avg_time_per_model = elapsed / (idx - 1)
            remaining_models = len(MODELS_TO_TUNE) - idx + 1
            estimated_remaining = avg_time_per_model * remaining_models
            print(f"⏱️  Elapsed: {elapsed} | Estimated remaining: ~{estimated_remaining}")
        
        # Tune model
        best_model, results, study = tune_model(
            model_key,
            objective_func,
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            n_trials=N_TRIALS
        )
        
        # Save final best model (with standard filename)
        model_file = f'models/{model_key}_model_seasonal_10pc_post2022_v4_muscular_corr08_tuned.joblib'
        joblib.dump(best_model, model_file)
        print(f"\n💾 Saved final tuned model to {model_file}")
        print(f"   (Best model from trial {results['optimization']['best_trial_number']} with Test Gini: {results['optimization']['best_test_gini']:.6f})")
        
        # Save results
        results_file = f'experiments/{model_key}_tuning_seasonal_10pc_post2022_corr08_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved results to {results_file}")
        
        # Save Optuna study data
        try:
            study_file_csv = f'optuna_studies/{model_key}_tuning_seasonal_10pc_post2022_corr08.csv'
            study_df = study.trials_dataframe()
            study_df.to_csv(study_file_csv, index=False)
            print(f"💾 Saved Optuna study data to {study_file_csv}")
        except Exception as e:
            print(f"⚠️  Could not save Optuna study data: {e}")
        
        all_results[model_key] = results
        
        print(f"\n✅ Completed tuning for {model_name}")
    
    # Print summary
    total_time = datetime.now() - start_time
    print("\n" + "=" * 80)
    print("SUMMARY - ALL MODELS")
    print("=" * 80)
    print(f"\n🎯 TEST GINI COEFFICIENTS (Primary Metric):")
    for model_key in MODELS_TO_TUNE:
        if model_key in all_results:
            test_gini = all_results[model_key]['test_metrics']['gini']
            test_roc_auc = all_results[model_key]['test_metrics']['roc_auc']
            print(f"   {model_key.upper():4s}: Test Gini = {test_gini:.6f} | Test ROC AUC = {test_roc_auc:.6f}")
    
    print(f"\n⏱️  Total execution time: {total_time}")
    print("\n" + "=" * 80)
    print("TUNING COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    main()

