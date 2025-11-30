#!/usr/bin/env python3
"""
Optimized Model Training Script V5 with Optuna
Trains multiple models with hyperparameter optimization using Optuna
"""
import sys
import io
# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import optuna
from optuna.pruners import MedianPruner

# Try to import optional models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# Reuse prepare_data from train_model_v4
def prepare_data(df, feature_list=None):
    """Prepare data with same encoding logic as V4, excluding week_5 features"""
    # Exclude metadata columns and all week_5 features
    feature_columns = [col for col in df.columns 
                      if col not in ['player_id', 'reference_date', 'player_name', 'target']
                      and '_week_5' not in col]
    
    # If feature_list is provided, filter to only those features
    if feature_list:
        # Keep only features that exist in both feature_list and feature_columns
        feature_columns = [col for col in feature_columns if col in feature_list]
    
    X = df[feature_columns]
    y = df['target']
    
    # Handle missing values and encode categorical features
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    X_encoded = X.copy()
    
    # Encode categorical features
    for feature in categorical_features:
        X_encoded[feature] = X_encoded[feature].fillna('Unknown')
        dummies = pd.get_dummies(X_encoded[feature], prefix=feature, drop_first=True)
        X_encoded = pd.concat([X_encoded, dummies], axis=1)
        X_encoded = X_encoded.drop(columns=[feature])
    
    # Fill numeric missing values
    for feature in numeric_features:
        X_encoded[feature] = X_encoded[feature].fillna(0)
    
    return X_encoded, y, feature_columns

def evaluate_model(model, X, y):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.0,
    }
    metrics['gini'] = 2 * metrics['roc_auc'] - 1
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    metrics['confusion_matrix'] = {
        'tn': int(cm[0, 0]),
        'fp': int(cm[0, 1]),
        'fn': int(cm[1, 0]),
        'tp': int(cm[1, 1])
    }
    
    return metrics

def create_random_forest_objective(X_train, y_train, X_val_insample, y_val_insample, 
                                   X_val_outsample, y_val_outsample):
    """Create Optuna objective for Random Forest"""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'max_depth': trial.suggest_int('max_depth', 10, 30, step=5),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 15, step=3),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 6, step=2),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': 42,
            'class_weight': 'balanced',
            'n_jobs': -1
        }
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Evaluate on out-of-sample (primary metric)
        metrics = evaluate_model(model, X_val_outsample, y_val_outsample)
        return metrics['f1']
    
    return objective

def create_gradient_boosting_objective(X_train, y_train, X_val_insample, y_val_insample,
                                       X_val_outsample, y_val_outsample):
    """Create Optuna objective for Gradient Boosting"""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=50),
            'max_depth': trial.suggest_int('max_depth', 5, 20, step=5),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'min_samples_split': trial.suggest_int('min_samples_split', 5, 20, step=5),
            'subsample': trial.suggest_float('subsample', 0.8, 1.0),
            'random_state': 42
        }
        
        model = GradientBoostingClassifier(**params)
        model.fit(X_train, y_train)
        
        metrics = evaluate_model(model, X_val_outsample, y_val_outsample)
        return metrics['f1']
    
    return objective

def create_xgboost_objective(X_train, y_train, X_val_insample, y_val_insample,
                             X_val_outsample, y_val_outsample):
    """Create Optuna objective for XGBoost"""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 15, step=2),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'subsample': trial.suggest_float('subsample', 0.8, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
            'random_state': 42,
            'eval_metric': 'logloss',
            'use_label_encoder': False
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        metrics = evaluate_model(model, X_val_outsample, y_val_outsample)
        return metrics['f1']
    
    return objective

def create_lightgbm_objective(X_train, y_train, X_val_insample, y_val_insample,
                              X_val_outsample, y_val_outsample):
    """Create Optuna objective for LightGBM"""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 15, step=2),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 100, step=15),
            'subsample': trial.suggest_float('subsample', 0.8, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
            'random_state': 42,
            'verbose': -1
        }
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        
        metrics = evaluate_model(model, X_val_outsample, y_val_outsample)
        return metrics['f1']
    
    return objective

def create_catboost_objective(X_train, y_train, X_val_insample, y_val_insample,
                              X_val_outsample, y_val_outsample):
    """Create Optuna objective for CatBoost"""
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 300, step=50),
            'depth': trial.suggest_int('depth', 4, 10, step=2),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 7, step=2),
            'border_count': trial.suggest_int('border_count', 32, 128, step=32),
            'random_state': 42,
            'verbose': False
        }
        
        model = cb.CatBoostClassifier(**params)
        model.fit(X_train, y_train)
        
        metrics = evaluate_model(model, X_val_outsample, y_val_outsample)
        return metrics['f1']
    
    return objective

def create_logistic_regression_objective(X_train, y_train, X_val_insample, y_val_insample,
                                         X_val_outsample, y_val_outsample):
    """Create Optuna objective for Logistic Regression"""
    def objective(trial):
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
        
        if penalty == 'l1':
            solver = 'liblinear'
        elif penalty == 'l2':
            solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear'])
        else:  # elasticnet
            solver = 'saga'
        
        params = {
            'C': trial.suggest_float('C', 0.001, 100.0, log=True),
            'penalty': penalty,
            'solver': solver,
            'max_iter': trial.suggest_int('max_iter', 1000, 5000, step=1000),
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }
        
        if penalty == 'elasticnet':
            params['l1_ratio'] = trial.suggest_float('l1_ratio', 0, 1)
        
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        
        metrics = evaluate_model(model, X_val_outsample, y_val_outsample)
        return metrics['f1']
    
    return objective

def train_model_with_optuna(model_name, feature_set_name, feature_list,
                            X_train, y_train, X_val_insample, y_val_insample,
                            X_val_outsample, y_val_outsample, config):
    """Train a model with Optuna optimization"""
    print(f"\n{'='*80}")
    print(f"Training {model_name} with feature set: {feature_set_name}")
    print(f"{'='*80}")
    
    # Create objective function based on model type
    if model_name == 'random_forest':
        objective = create_random_forest_objective(
            X_train, y_train, X_val_insample, y_val_insample,
            X_val_outsample, y_val_outsample
        )
    elif model_name == 'gradient_boosting':
        objective = create_gradient_boosting_objective(
            X_train, y_train, X_val_insample, y_val_insample,
            X_val_outsample, y_val_outsample
        )
    elif model_name == 'xgboost' and XGBOOST_AVAILABLE:
        objective = create_xgboost_objective(
            X_train, y_train, X_val_insample, y_val_insample,
            X_val_outsample, y_val_outsample
        )
    elif model_name == 'lightgbm' and LIGHTGBM_AVAILABLE:
        objective = create_lightgbm_objective(
            X_train, y_train, X_val_insample, y_val_insample,
            X_val_outsample, y_val_outsample
        )
    elif model_name == 'catboost' and CATBOOST_AVAILABLE:
        objective = create_catboost_objective(
            X_train, y_train, X_val_insample, y_val_insample,
            X_val_outsample, y_val_outsample
        )
    elif model_name == 'logistic_regression':
        objective = create_logistic_regression_objective(
            X_train, y_train, X_val_insample, y_val_insample,
            X_val_outsample, y_val_outsample
        )
    else:
        print(f"⚠️  Model {model_name} not available or not implemented")
        return None
    
    # Create study
    study_name = f"{model_name}_{feature_set_name}"
    storage_path = config['optuna']['study_storage']
    os.makedirs(os.path.dirname(storage_path.replace('sqlite:///', '')), exist_ok=True)
    
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        storage=storage_path,
        load_if_exists=True,
        pruner=MedianPruner() if config['optuna']['pruning'] else None
    )
    
    # Optimize
    print(f"   Running {config['optuna']['n_trials']} trials...")
    study.optimize(
        objective,
        n_trials=config['optuna']['n_trials'],
        timeout=config['optuna'].get('timeout'),
        n_jobs=1  # Optuna handles parallelization differently
    )
    
    # Get best parameters and train final model
    best_params = study.best_params
    print(f"   Best F1-Score: {study.best_value:.4f}")
    print(f"   Best parameters: {best_params}")
    
    # Train final model with best parameters
    if model_name == 'random_forest':
        best_model = RandomForestClassifier(**best_params, random_state=42, class_weight='balanced', n_jobs=-1)
    elif model_name == 'gradient_boosting':
        best_model = GradientBoostingClassifier(**best_params, random_state=42)
    elif model_name == 'xgboost' and XGBOOST_AVAILABLE:
        best_model = xgb.XGBClassifier(**best_params, random_state=42, eval_metric='logloss', use_label_encoder=False)
    elif model_name == 'lightgbm' and LIGHTGBM_AVAILABLE:
        best_model = lgb.LGBMClassifier(**best_params, random_state=42, verbose=-1)
    elif model_name == 'catboost' and CATBOOST_AVAILABLE:
        best_model = cb.CatBoostClassifier(**best_params, random_state=42, verbose=False)
    elif model_name == 'logistic_regression':
        best_model = LogisticRegression(**best_params, class_weight='balanced', random_state=42, n_jobs=-1)
    else:
        return None
    
    best_model.fit(X_train, y_train)
    
    # Evaluate on all sets
    train_metrics = evaluate_model(best_model, X_train, y_train)
    insample_metrics = evaluate_model(best_model, X_val_insample, y_val_insample)
    outsample_metrics = evaluate_model(best_model, X_val_outsample, y_val_outsample)
    
    # Calculate gaps
    gaps_insample = {metric: train_metrics[metric] - insample_metrics[metric] 
                     for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'gini']}
    gaps_outsample = {metric: train_metrics[metric] - outsample_metrics[metric]
                   for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'gini']}
    
    results = {
        'model_name': model_name,
        'feature_set_name': feature_set_name,
        'best_params': best_params,
        'best_f1': study.best_value,
        'train': train_metrics,
        'validation_insample': insample_metrics,
        'validation_outsample': outsample_metrics,
        'gaps_insample': gaps_insample,
        'gaps_outsample': gaps_outsample
    }
    
    # Save model and results
    models_dir = config['paths']['models_dir']
    os.makedirs(models_dir, exist_ok=True)
    
    model_file = f"{models_dir}/{model_name}_{feature_set_name}.joblib"
    params_file = f"{models_dir}/{model_name}_{feature_set_name}_params.json"
    metrics_file = f"{models_dir}/{model_name}_{feature_set_name}_metrics.json"
    
    joblib.dump(best_model, model_file)
    with open(params_file, 'w') as f:
        json.dump(best_params, f, indent=2)
    with open(metrics_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"   ✅ Saved model to {model_file}")
    print(f"   ✅ Saved metrics to {metrics_file}")
    
    return results

def main():
    """Main function - can be called from parallel execution script"""
    print("=" * 80)
    print("OPTIMIZED MODEL TRAINING V5")
    print("=" * 80)
    
    # Load config
    config_file = 'config/model_selection_config.json'
    if not os.path.exists(config_file):
        config_file = f'scripts/{config_file}'
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Load data
    print("\n📂 Loading timelines...")
    train_file = 'timelines_35day_enhanced_balanced_v4_train.csv'
    val_file = 'timelines_35day_enhanced_balanced_v4_val.csv'
    
    if not os.path.exists(train_file):
        train_file = f'scripts/{train_file}'
    if not os.path.exists(val_file):
        val_file = f'scripts/{val_file}'
    
    df_train_full = pd.read_csv(train_file, encoding='utf-8-sig')
    df_val_outsample = pd.read_csv(val_file, encoding='utf-8-sig')
    
    print(f"✅ Loaded training set: {len(df_train_full):,} records")
    print(f"✅ Loaded validation set: {len(df_val_outsample):,} records")
    
    # Load feature sets
    feature_sets_dir = config['paths']['feature_sets_dir']
    if not os.path.exists(feature_sets_dir):
        print(f"❌ Feature sets directory not found: {feature_sets_dir}")
        print("   Please run feature_selection_v5.py first")
        return
    
    feature_set_files = [f for f in os.listdir(feature_sets_dir) if f.endswith('.json')]
    print(f"\n📂 Found {len(feature_set_files)} feature sets")
    
    # Prepare base data (without feature filtering)
    print("\n🔧 Preparing base data...")
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
    
    # Model types
    model_types = ['random_forest', 'gradient_boosting', 'logistic_regression']
    if XGBOOST_AVAILABLE:
        model_types.append('xgboost')
    if LIGHTGBM_AVAILABLE:
        model_types.append('lightgbm')
    if CATBOOST_AVAILABLE:
        model_types.append('catboost')
    
    print(f"\n📊 Available models: {', '.join(model_types)}")
    
    # Process each feature set and model combination
    # This is typically called from the parallel execution script
    # For standalone execution, process first feature set as example
    if len(feature_set_files) > 0:
        feature_set_file = feature_set_files[0]
        feature_set_name = feature_set_file.replace('.json', '')
        
        with open(f"{feature_sets_dir}/{feature_set_file}", 'r') as f:
            feature_list = json.load(f)
        
        # Filter data to feature set
        available_features = [f for f in feature_list if f in X_train_base.columns]
        X_train = X_train_base[available_features]
        X_val_insample = X_val_insample_base[available_features]
        X_val_outsample = X_val_outsample_base[available_features]
        
        print(f"\n📊 Using feature set: {feature_set_name} ({len(available_features)} features)")
        
        # Train one model as example
        train_model_with_optuna(
            'random_forest', feature_set_name, available_features,
            X_train, y_train, X_val_insample, y_val_insample,
            X_val_outsample, y_val_outsample, config
        )
    
    print("\n✅ Training script ready for parallel execution")

if __name__ == "__main__":
    main()


