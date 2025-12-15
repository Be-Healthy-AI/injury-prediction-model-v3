#!/usr/bin/env python3
"""
Hyperparameter Tuning for V4 Muscular Injuries Models
- Optimizes validation ROC AUC (to maximize test Gini)
- Uses Optuna for Bayesian optimization
- Tunes models sequentially: RF, GB, XGB, LGBM
- Works with 50% target ratio datasets
- Uses 0.8 correlation threshold
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import json
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
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

# Import preprocessing functions from train_models_v4_combined
# We'll copy the necessary functions here to avoid circular imports
def clean_categorical_value(value):
    """Clean categorical values to remove special characters"""
    if pd.isna(value) or value is None:
        return 'Unknown'
    
    value_str = str(value).strip()
    
    problematic_values = ['tel:', 'tel', 'phone', 'n/a', 'na', 'null', 'none', '', 'nan', 'address:', 'website:']
    if value_str.lower() in problematic_values:
        return 'Unknown'
    
    replacements = {
        ':': '_', "'": '_', ',': '_', '"': '_', ';': '_', '/': '_', '\\': '_',
        '{': '_', '}': '_', '[': '_', ']': '_', '(': '_', ')': '_', '|': '_',
        '&': '_', '?': '_', '!': '_', '*': '_', '+': '_', '=': '_', '@': '_',
        '#': '_', '$': '_', '%': '_', '^': '_', ' ': '_',
    }
    
    for old_char, new_char in replacements.items():
        value_str = value_str.replace(old_char, new_char)
    
    value_str = ''.join(char for char in value_str if ord(char) >= 32 or char in '\n\r\t')
    while '__' in value_str:
        value_str = value_str.replace('__', '_')
    value_str = value_str.strip('_')
    
    if not value_str:
        return 'Unknown'
    
    return value_str

def sanitize_feature_name(name):
    """Sanitize feature names to be JSON-safe for LightGBM"""
    name_str = str(name)
    replacements = {
        '"': '_quote_', '\\': '_backslash_', '/': '_slash_', '\b': '_bs_',
        '\f': '_ff_', '\n': '_nl_', '\r': '_cr_', '\t': '_tab_', ' ': '_',
        "'": '_apostrophe_', ':': '_colon_', ';': '_semicolon_', ',': '_comma_',
        '{': '_lbrace_', '}': '_rbrace_', '[': '_lbracket_', ']': '_rbracket_',
        '&': '_amp_',
    }
    for old, new in replacements.items():
        name_str = name_str.replace(old, new)
    name_str = ''.join(char for char in name_str if ord(char) >= 32 or char in '\n\r\t')
    while '__' in name_str:
        name_str = name_str.replace('__', '_')
    name_str = name_str.strip('_')
    return name_str

def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types for JSON serialization"""
    if obj is None:
        return None
    
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif hasattr(obj, 'item'):
        try:
            return convert_numpy_types(obj.item())
        except (AttributeError, ValueError):
            pass
    elif isinstance(obj, dict):
        return {str(key): convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    elif isinstance(obj, float) and np.isnan(obj):
        return None
    else:
        return obj

def prepare_data(df):
    """Prepare data with basic preprocessing"""
    feature_columns = [
        col for col in df.columns
        if col not in ['player_id', 'reference_date', 'player_name', 'target']
    ]
    X = df[feature_columns].copy()
    y = df['target'].copy()
    
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    X_encoded = X.copy()
    
    if len(categorical_features) > 0:
        for feature in categorical_features:
            X_encoded[feature] = X_encoded[feature].fillna('Unknown')
            X_encoded[feature] = X_encoded[feature].apply(clean_categorical_value)
            dummies = pd.get_dummies(X_encoded[feature], prefix=feature, drop_first=True)
            dummies.columns = [sanitize_feature_name(col) for col in dummies.columns]
            X_encoded = pd.concat([X_encoded, dummies], axis=1)
            X_encoded = X_encoded.drop(columns=[feature])
    
    if len(numeric_features) > 0:
        for feature in numeric_features:
            X_encoded[feature] = X_encoded[feature].fillna(0)
    
    X_encoded.columns = [sanitize_feature_name(col) for col in X_encoded.columns]
    
    return X_encoded, y

def align_features(X_train, X_val, X_test):
    """Ensure all datasets have the same features"""
    common_features = sorted(list(set(X_train.columns) & set(X_val.columns) & set(X_test.columns)))
    return X_train[common_features], X_val[common_features], X_test[common_features]

def apply_correlation_filter(X, threshold=0.8, cache_dir='cache', use_cache=True):
    """Drop one feature from each highly correlated pair with optional caching"""
    cache_file = None
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f'corr_matrix_{X.shape[0]}_{X.shape[1]}_{threshold}.npy')
        
        if os.path.exists(cache_file):
            try:
                corr_matrix = np.load(cache_file)
                corr_matrix = pd.DataFrame(corr_matrix, index=X.columns, columns=X.columns)
            except Exception:
                cache_file = None
    
    if cache_file is None or not os.path.exists(cache_file):
        corr_matrix = X.corr().abs()
        if use_cache and cache_file:
            try:
                np.save(cache_file, corr_matrix.values)
            except Exception:
                pass
    
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    kept = [col for col in X.columns if col not in to_drop]
    
    return kept

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

def objective_rf(trial, X_train, y_train, X_val, y_val):
    """Optuna objective function for Random Forest"""
    use_max_depth = trial.suggest_categorical('use_max_depth', [True, False])
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
        'max_depth': trial.suggest_int('max_depth', 10, 20) if use_max_depth else None,
        'min_samples_split': trial.suggest_int('min_samples_split', 5, 50, step=5),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 20, step=2),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.7]),
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    val_metrics = evaluate_model_metrics(model, X_val, y_val)
    return val_metrics['roc_auc']  # Optimize for validation ROC AUC

def objective_gb(trial, X_train, y_train, X_val, y_val):
    """Optuna objective function for Gradient Boosting"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 400, step=50),
        'max_depth': trial.suggest_int('max_depth', 5, 15, step=2),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'min_samples_split': trial.suggest_int('min_samples_split', 5, 30, step=5),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 15, step=2),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.1),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.7]),
        'random_state': 42
    }
    
    model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train)
    
    val_metrics = evaluate_model_metrics(model, X_val, y_val)
    return val_metrics['roc_auc']

def objective_xgb(trial, X_train, y_train, X_val, y_val):
    """Optuna objective function for XGBoost"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 400, step=50),
        'max_depth': trial.suggest_int('max_depth', 5, 15, step=2),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 15, step=2),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.1),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0, step=0.1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 3.0, step=0.5),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 10, 20, step=1),
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'verbosity': 0
    }
    
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    val_metrics = evaluate_model_metrics(model, X_val, y_val)
    return val_metrics['roc_auc']

def objective_lgbm(trial, X_train, y_train, X_val, y_val):
    """Optuna objective function for LightGBM"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 400, step=50),
        'max_depth': trial.suggest_int('max_depth', 5, 15, step=2),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100, step=10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.1),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0, step=0.1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 3.0, step=0.5),
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)
    
    val_metrics = evaluate_model_metrics(model, X_val, y_val)
    return val_metrics['roc_auc']

# ========== TUNING FUNCTIONS ==========

def tune_model(model_name, objective_func, X_train, y_train, X_val, y_val, X_test, y_test, n_trials=100):
    """Tune a single model using Optuna"""
    print("\n" + "=" * 80)
    print(f"🔍 HYPERPARAMETER TUNING: {model_name.upper()}")
    print("=" * 80)
    print(f"   Optimizing: Validation ROC AUC (to maximize Test Gini)")
    print(f"   Number of trials: {n_trials}")
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Validation samples: {len(X_val):,}")
    print(f"   Test samples: {len(X_test):,}")
    print("=" * 80)
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        study_name=f'{model_name}_tuning'
    )
    
    # Create objective wrapper with error handling
    def objective(trial):
        try:
            return objective_func(trial, X_train, y_train, X_val, y_val)
        except Exception as e:
            print(f"\n⚠️  Trial {trial.number} failed: {e}")
            # Return a very low score so Optuna knows this trial failed
            return 0.0
    
    # Run optimization
    print(f"\n🚀 Starting optimization...")
    start_time = datetime.now()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    tuning_time = datetime.now() - start_time
    
    print(f"\n✅ Optimization completed in {tuning_time}")
    print(f"\n📊 Best trial:")
    print(f"   Validation ROC AUC: {study.best_value:.6f}")
    print(f"   Validation Gini: {2 * study.best_value - 1:.6f}")
    print(f"\n🔧 Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"   {key}: {value}")
    
    # Train best model
    print(f"\n🌳 Training best model...")
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
    
    best_model.fit(X_train, y_train)
    
    # Evaluate on all sets
    print(f"\n📈 Evaluating best model...")
    train_metrics = evaluate_model_metrics(best_model, X_train, y_train)
    val_metrics = evaluate_model_metrics(best_model, X_val, y_val)
    test_metrics = evaluate_model_metrics(best_model, X_test, y_test)
    
    # Print results
    print(f"\n" + "=" * 80)
    print(f"📊 FINAL RESULTS - {model_name.upper()}")
    print("=" * 80)
    print(f"\n🎯 TEST GINI (Primary Metric): {test_metrics['gini']:.6f}")
    print(f"\n   Training Metrics:")
    print(f"      ROC AUC: {train_metrics['roc_auc']:.6f} | Gini: {train_metrics['gini']:.6f}")
    print(f"      F1: {train_metrics['f1']:.6f} | Recall: {train_metrics['recall']:.6f} | Precision: {train_metrics['precision']:.6f}")
    print(f"\n   Validation Metrics:")
    print(f"      ROC AUC: {val_metrics['roc_auc']:.6f} | Gini: {val_metrics['gini']:.6f}")
    print(f"      F1: {val_metrics['f1']:.6f} | Recall: {val_metrics['recall']:.6f} | Precision: {val_metrics['precision']:.6f}")
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
            'best_validation_roc_auc': float(study.best_value),
            'best_validation_gini': float(2 * study.best_value - 1),
            'tuning_time_seconds': tuning_time.total_seconds()
        },
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'trial_history': [
            {
                'trial': trial.number,
                'validation_roc_auc': float(trial.value),
                'validation_gini': float(2 * trial.value - 1),
                'params': convert_numpy_types(trial.params)
            }
            for trial in study.trials
        ]
    }
    
    return best_model, results, study

def main():
    # ========== CONFIGURATION ==========
    # Options for TARGET_RATIO: 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, or 'natural'
    TARGET_RATIO = 0.50  # 50% datasets
    CORR_THRESHOLD = 0.8
    N_TRIALS = 25  # Number of Optuna trials per model
    USE_POST2017_FILTER = False  # Set to True to use filtered datasets (post 2017-06-30)
    
    # Models to tune (can be modified to tune specific models)
    MODELS_TO_TUNE = ['rf', 'gb', 'xgb', 'lgbm']  # or ['rf'] to tune only RF
    
    # ===================================
    
    # Determine ratio string for filenames and display
    if TARGET_RATIO == 'natural':
        if USE_POST2017_FILTER:
            ratio_str = 'natural_post2017'
            ratio_display = 'Natural (post 2017-06-30)'
            ratio_title = 'NATURAL TARGET RATIO (POST 2017-06-30)'
        else:
            ratio_str = 'natural'
            ratio_display = 'Natural'
            ratio_title = 'NATURAL TARGET RATIO'
    else:
        if USE_POST2017_FILTER:
            ratio_str = f"{int(TARGET_RATIO * 100):02d}pc_post2017"
            ratio_display = f"{TARGET_RATIO:.0%} (post 2017-06-30)"
            ratio_title = f"{TARGET_RATIO:.0%} TARGET RATIO (POST 2017-06-30)"
        else:
            ratio_str = f"{int(TARGET_RATIO * 100):02d}pc"
            ratio_display = f"{TARGET_RATIO:.0%}"
            ratio_title = f"{TARGET_RATIO:.0%} TARGET RATIO"
    
    print("=" * 80)
    print(f"HYPERPARAMETER TUNING - V4 MUSCULAR INJURIES ({ratio_title})")
    print("=" * 80)
    print("\n📋 Configuration:")
    print(f"   Target ratio: {ratio_display}")
    print(f"   Correlation threshold: {CORR_THRESHOLD}")
    print(f"   Optimization metric: Validation ROC AUC (to maximize Test Gini)")
    print(f"   Number of trials per model: {N_TRIALS}")
    print(f"   Models to tune: {', '.join(MODELS_TO_TUNE).upper()}")
    print("=" * 80)
    
    start_time = datetime.now()
    
    # Load datasets
    print("\n📂 Loading datasets...")
    train_file = f'timelines_35day_enhanced_{ratio_str}_v4_muscular_train.csv'
    val_file = f'timelines_35day_enhanced_{ratio_str}_v4_muscular_val.csv'
    test_file = 'timelines_35day_enhanced_natural_v4_muscular_test.csv'
    
    print(f"   Loading {train_file}...")
    df_train = pd.read_csv(train_file, encoding='utf-8-sig')
    print(f"   ✅ Loaded: {len(df_train):,} records")
    
    print(f"   Loading {val_file}...")
    df_val = pd.read_csv(val_file, encoding='utf-8-sig')
    print(f"   ✅ Loaded: {len(df_val):,} records")
    
    print(f"   Loading {test_file}...")
    df_test = pd.read_csv(test_file, encoding='utf-8-sig')
    print(f"   ✅ Loaded: {len(df_test):,} records")
    
    # Prepare data
    print("\n📊 Preparing data...")
    print("   Preparing training set...")
    X_train, y_train = prepare_data(df_train)
    print("   Preparing validation set...")
    X_val, y_val = prepare_data(df_val)
    print("   Preparing test set...")
    X_test, y_test = prepare_data(df_test)
    
    # Align features
    print("\n🔧 Aligning features across datasets...")
    X_train, X_val, X_test = align_features(X_train, X_val, X_test)
    print(f"   ✅ Aligned: {X_train.shape[1]} features")
    
    # Apply correlation filter
    print("\n🔎 Applying correlation filter...")
    kept_features = apply_correlation_filter(X_train, threshold=CORR_THRESHOLD)
    X_train = X_train[kept_features]
    X_val = X_val[kept_features]
    X_test = X_test[kept_features]
    print(f"   ✅ After filtering: {len(kept_features)} features")
    
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
    
    for model_key in MODELS_TO_TUNE:
        if model_key not in model_functions:
            print(f"\n⚠️  Unknown model: {model_key}, skipping...")
            continue
        
        objective_func, model_name = model_functions[model_key]
        
        # Tune model
        best_model, results, study = tune_model(
            model_key,
            objective_func,
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            n_trials=N_TRIALS
        )
        
        # Save model
        model_file = f'models/{model_key}_model_v4_muscular_{ratio_str}_corr08_tuned.joblib'
        joblib.dump(best_model, model_file)
        print(f"\n💾 Saved tuned model to {model_file}")
        
        # Save results
        results_file = f'experiments/{model_key}_tuning_{ratio_str}_corr08_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Saved results to {results_file}")
        
        # Save Optuna study data (optional - for later analysis)
        try:
            study_file_csv = f'optuna_studies/{model_key}_tuning_{ratio_str}_corr08.csv'
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
            val_gini = all_results[model_key]['val_metrics']['gini']
            print(f"   {model_key.upper():4s}: Test Gini = {test_gini:.6f} | Val Gini = {val_gini:.6f}")
    
    print(f"\n⏱️  Total execution time: {total_time}")
    print("\n" + "=" * 80)
    print("TUNING COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    main()

