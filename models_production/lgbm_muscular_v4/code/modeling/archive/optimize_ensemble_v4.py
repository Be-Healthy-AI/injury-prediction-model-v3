#!/usr/bin/env python3
"""
Ensemble Optimization Script for V4 Models
Tests different combinations of the 6 trained models and optimizes weights
using validation set, then evaluates on test set.
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from itertools import combinations
from tqdm import tqdm

# TensorFlow/Keras imports
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Add paths for imports
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent.parent.parent
TIMELINES_DIR = SCRIPT_DIR.parent / 'timelines'
sys.path.insert(0, str(TIMELINES_DIR.resolve()))

# Import filter function
import importlib.util
timeline_script_path = TIMELINES_DIR / 'create_35day_timelines_v4_enhanced.py'
if not timeline_script_path.exists():
    raise FileNotFoundError(f"Timeline script not found: {timeline_script_path}")
spec = importlib.util.spec_from_file_location("create_35day_timelines_v4_enhanced", timeline_script_path)
timeline_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timeline_module)
filter_timelines_for_model = timeline_module.filter_timelines_for_model

# ========== CONFIGURATION ==========
TRAIN_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'data' / 'timelines' / 'train'
TEST_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'data' / 'timelines' / 'test'
MODEL_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'models' / 'comparison'
CACHE_DIR = ROOT_DIR / 'cache'
OUTPUT_DIR = MODEL_DIR / 'ensemble_optimization'
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42
# ===================================

# Copy necessary functions from train_all_models_v4_comparison to avoid import issues
import glob

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
    return value_str if value_str else 'Unknown'

def sanitize_feature_name(name):
    """Sanitize feature names to be JSON-safe"""
    name_str = str(name)
    replacements = {
        '"': '_quote_', '\\': '_backslash_', '/': '_slash_',
        '\b': '_bs_', '\f': '_ff_', '\n': '_nl_', '\r': '_cr_', '\t': '_tab_', ' ': '_',
        "'": '_apostrophe_', ':': '_colon_', ';': '_semicolon_', ',': '_comma_',
        '{': '_lbrace_', '}': '_rbrace_', '[': '_lbracket_', ']': '_rbracket_', '&': '_amp_',
    }
    for old, new in replacements.items():
        name_str = name_str.replace(old, new)
    name_str = ''.join(char for char in name_str if ord(char) >= 32 or char in '\n\r\t')
    while '__' in name_str:
        name_str = name_str.replace('__', '_')
    return name_str.strip('_')

def load_combined_seasonal_datasets_natural(min_season=None, exclude_season='2025_2026'):
    """Load and combine all seasonal datasets with natural target ratio"""
    pattern = str(TRAIN_DIR / 'timelines_35day_season_*_v4_muscular_train.csv')
    files = glob.glob(pattern)
    season_files = []
    
    for filepath in files:
        filename = os.path.basename(filepath)
        if '_10pc_' in filename or '_25pc_' in filename or '_50pc_' in filename:
            continue
        if 'season_' in filename:
            parts = filename.split('season_')
            if len(parts) > 1:
                season_part = parts[1].split('_v4_muscular_train')[0]
                if season_part != exclude_season:
                    if min_season is None or season_part >= min_season:
                        season_files.append((season_part, filepath))
    
    season_files.sort(key=lambda x: x[0])
    
    print(f"\n📂 Loading {len(season_files)} season files with natural target ratio...")
    if min_season:
        print(f"   (Filtering: Only seasons >= {min_season})")
    
    dfs = []
    total_records = 0
    total_target1 = 0
    total_target2 = 0
    
    for season_id, filepath in season_files:
        try:
            df = pd.read_csv(filepath, encoding='utf-8-sig', low_memory=False)
            if 'target1' not in df.columns or 'target2' not in df.columns:
                print(f"   ⚠️  {season_id}: Missing target1/target2 columns - skipping")
                continue
            
            target1_count = (df['target1'] == 1).sum()
            target2_count = (df['target2'] == 1).sum()
            
            if len(df) > 0:
                dfs.append(df)
                total_records += len(df)
                total_target1 += target1_count
                total_target2 += target2_count
                print(f"   ✅ {season_id}: {len(df):,} records (target1: {target1_count:,}, target2: {target2_count:,})")
        except Exception as e:
            print(f"   ⚠️  Error loading {season_id}: {e}")
            continue
    
    if not dfs:
        raise ValueError("No valid season files found!")
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    print(f"✅ Combined dataset: {len(combined_df):,} records")
    print(f"   Total target1=1 (Muscular): {total_target1:,} ({total_target1/len(combined_df)*100:.4f}%)")
    print(f"   Total target2=1 (Skeletal): {total_target2:,} ({total_target2/len(combined_df)*100:.4f}%)")
    
    return combined_df

def load_test_dataset():
    """Load test dataset (2025/26 season)"""
    test_file = TEST_DIR / 'timelines_35day_season_2025_2026_v4_muscular_test.csv'
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")
    print(f"\n📂 Loading test dataset: {test_file.name}")
    df_test = pd.read_csv(test_file, encoding='utf-8-sig', low_memory=False)
    if 'target1' not in df_test.columns or 'target2' not in df_test.columns:
        raise ValueError("Test dataset missing target1/target2 columns")
    return df_test

def prepare_data(df, cache_file=None, use_cache=True):
    """Prepare data with basic preprocessing"""
    # Check cache
    if use_cache and cache_file and os.path.exists(cache_file):
        print(f"   📦 Loading preprocessed data from cache: {cache_file}...")
        try:
            df_preprocessed = pd.read_csv(cache_file, encoding='utf-8-sig', low_memory=False)
            if len(df_preprocessed) != len(df):
                print(f"   ⚠️  Cache length mismatch, preprocessing fresh...")
                use_cache = False
            else:
                target_cols = ['target1', 'target2', 'target', 'target_combined']
                cols_to_drop = [col for col in target_cols if col in df_preprocessed.columns]
                if cols_to_drop:
                    X = df_preprocessed.drop(columns=cols_to_drop)
                else:
                    X = df_preprocessed
                print(f"   ✅ Loaded preprocessed data from cache")
                return X
        except Exception as e:
            print(f"   ⚠️  Failed to load cache ({e}), preprocessing fresh...")
            use_cache = False
    
    # Drop non-feature columns
    feature_columns = [
        col for col in df.columns
        if col not in ['player_id', 'reference_date', 'date', 'player_name', 'target1', 'target2', 'target', 'has_minimum_activity', 'target_combined']
    ]
    X = df[feature_columns].copy()
    
    # Handle missing values and encode categorical features
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    X_encoded = X.copy()
    
    # Encode categorical features
    if len(categorical_features) > 0:
        print(f"   Processing {len(categorical_features)} categorical features...")
        for feature in tqdm(categorical_features, desc="   Encoding categorical features", unit="feature", leave=False):
            X_encoded[feature] = X_encoded[feature].fillna('Unknown')
            X_encoded[feature] = X_encoded[feature].apply(clean_categorical_value)
            dummies = pd.get_dummies(X_encoded[feature], prefix=feature, drop_first=True)
            dummies.columns = [sanitize_feature_name(col) for col in dummies.columns]
            X_encoded = pd.concat([X_encoded, dummies], axis=1)
            X_encoded = X_encoded.drop(columns=[feature])
    
    # Fill numeric missing values
    if len(numeric_features) > 0:
        print(f"   Processing {len(numeric_features)} numeric features...")
        for feature in tqdm(numeric_features, desc="   Filling numeric missing values", unit="feature", leave=False):
            X_encoded[feature] = X_encoded[feature].fillna(0)
    
    # Sanitize all column names
    X_encoded.columns = [sanitize_feature_name(col) for col in X_encoded.columns]
    
    # Cache preprocessed data
    if use_cache and cache_file:
        print(f"   💾 Caching preprocessed data to {cache_file}...")
        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            X_encoded.to_csv(cache_file, index=False, encoding='utf-8-sig')
            print(f"   ✅ Preprocessed data cached")
        except Exception as e:
            print(f"   ⚠️  Failed to cache ({e})")
    
    return X_encoded

def align_features(X_train, X_test):
    """Ensure all datasets have the same features"""
    common_features = list(set(X_train.columns) & set(X_test.columns))
    common_features = sorted(common_features)
    print(f"   Aligning features: {len(common_features)} common features")
    print(f"   Training: {X_train.shape[1]} -> {len(common_features)}")
    print(f"   Test: {X_test.shape[1]} -> {len(common_features)}")
    return X_train[common_features], X_test[common_features]

def create_combined_target(df):
    """Create combined target: target1=1 OR target2=1"""
    df = df.copy()
    df['target_combined'] = ((df['target1'] == 1) | (df['target2'] == 1)).astype(int)
    return df

def filter_for_combined_target(df):
    """Filter dataset for combined target model"""
    mask = ((df['target1'] == 1) | (df['target2'] == 1)) | ((df['target1'] == 0) & (df['target2'] == 0))
    return df[mask].copy()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_model(model_path, model_type='lgbm'):
    """Load a trained model"""
    if model_type == 'lgbm':
        return joblib.load(model_path)
    else:  # neural network
        return keras.models.load_model(model_path)

def load_model_columns(columns_path):
    """Load model feature columns"""
    with open(columns_path, 'r') as f:
        return json.load(f)

def generate_predictions(model, X, model_type='lgbm'):
    """Generate probability predictions from a model"""
    if model_type == 'lgbm':
        if isinstance(X, pd.DataFrame):
            return model.predict_proba(X)[:, 1]
        else:
            return model.predict_proba(X)[:, 1]
    else:  # neural network
        if isinstance(X, pd.DataFrame):
            X_array = X.values.astype(np.float32)
        else:
            X_array = np.array(X, dtype=np.float32)
        return model.predict(X_array, verbose=0).flatten()

def ensemble_weighted_average(probabilities_list, weights):
    """Create weighted average ensemble"""
    weights = np.array(weights)
    weights = weights / weights.sum()  # Normalize weights
    ensemble_proba = np.zeros(len(probabilities_list[0]))
    for proba, weight in zip(probabilities_list, weights):
        ensemble_proba += weight * proba
    return ensemble_proba

def ensemble_gini_weighted(probabilities_list, gini_scores):
    """Create Gini-weighted ensemble"""
    gini_array = np.array(gini_scores)
    # Convert Gini to weights (higher Gini = higher weight)
    weights = gini_array / gini_array.sum()
    return ensemble_weighted_average(probabilities_list, weights)

def ensemble_rank_average(probabilities_list):
    """Create rank-based ensemble"""
    from scipy.stats import rankdata
    n = len(probabilities_list[0])
    ranks = []
    for proba in probabilities_list:
        ranks.append(rankdata(proba, method='average'))
    avg_rank = np.mean(ranks, axis=0)
    # Convert ranks back to [0,1] scores
    return (n - avg_rank + 1) / n

def ensemble_geometric_mean(probabilities_list):
    """Create geometric mean ensemble"""
    eps = 1e-10
    proba_safe = [np.clip(p, eps, 1 - eps) for p in probabilities_list]
    product = np.ones(len(proba_safe[0]))
    for p in proba_safe:
        product *= p
    return np.power(product, 1.0 / len(proba_safe))

def evaluate_ensemble(y_true, y_proba, threshold=0.5):
    """Evaluate ensemble predictions"""
    y_pred = (y_proba >= threshold).astype(int)
    
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else 0.0,
        'gini': float((2 * roc_auc_score(y_true, y_proba) - 1)) if len(np.unique(y_true)) > 1 else 0.0
    }
    
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        metrics['confusion_matrix'] = {
            'tn': int(cm[0, 0]),
            'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]),
            'tp': int(cm[1, 1])
        }
    else:
        metrics['confusion_matrix'] = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
    
    return metrics

def optimize_weights_grid_search(probabilities_list, y_true, metric='roc_auc', n_points=11):
    """Optimize ensemble weights using grid search"""
    n_models = len(probabilities_list)
    
    if n_models == 1:
        return [1.0], evaluate_ensemble(y_true, probabilities_list[0])
    
    if n_models == 2:
        best_score = -1
        best_weights = None
        best_metrics = None
        
        for w1 in np.linspace(0.0, 1.0, n_points):
            w2 = 1.0 - w1
            ensemble_proba = ensemble_weighted_average(probabilities_list, [w1, w2])
            metrics = evaluate_ensemble(y_true, ensemble_proba)
            
            score = metrics[metric]
            if score > best_score:
                best_score = score
                best_weights = [w1, w2]
                best_metrics = metrics
        
        return best_weights, best_metrics
    
    else:  # 3+ models - use more efficient search
        # Try equal weights first
        equal_weights = [1.0 / n_models] * n_models
        ensemble_proba = ensemble_weighted_average(probabilities_list, equal_weights)
        best_metrics = evaluate_ensemble(y_true, ensemble_proba)
        best_weights = equal_weights
        best_score = best_metrics[metric]
        
        # Try Gini-weighted
        # We need individual model Gini scores - estimate from validation
        gini_scores = []
        for proba in probabilities_list:
            gini = (2 * roc_auc_score(y_true, proba) - 1) if len(np.unique(y_true)) > 1 else 0.0
            gini_scores.append(max(0, gini))  # Ensure non-negative
        
        if sum(gini_scores) > 0:
            gini_weights = [g / sum(gini_scores) for g in gini_scores]
            ensemble_proba = ensemble_weighted_average(probabilities_list, gini_weights)
            metrics = evaluate_ensemble(y_true, ensemble_proba)
            if metrics[metric] > best_score:
                best_score = metrics[metric]
                best_weights = gini_weights
                best_metrics = metrics
        
        # Try some manual combinations for 3 models
        if n_models == 3:
            test_combinations = [
                [0.5, 0.3, 0.2],
                [0.4, 0.4, 0.2],
                [0.6, 0.2, 0.2],
                [0.33, 0.33, 0.34],
            ]
            for weights in test_combinations:
                ensemble_proba = ensemble_weighted_average(probabilities_list, weights)
                metrics = evaluate_ensemble(y_true, ensemble_proba)
                if metrics[metric] > best_score:
                    best_score = metrics[metric]
                    best_weights = weights
                    best_metrics = metrics
        
        return best_weights, best_metrics

def test_ensemble_combination(model_names, model_predictions_val, y_val, 
                              model_predictions_test, y_test, 
                              individual_metrics_val, individual_metrics_test):
    """Test a specific ensemble combination"""
    results = {}
    
    # Get predictions for this combination (already as lists or dicts)
    if isinstance(model_predictions_val, dict):
        proba_val = [model_predictions_val[name] for name in model_names]
    else:
        proba_val = model_predictions_val
    
    if isinstance(model_predictions_test, dict):
        proba_test = [model_predictions_test[name] for name in model_names]
    else:
        proba_test = model_predictions_test
    
    # Test different ensemble methods
    ensemble_methods = {}
    
    # 1. Optimized weighted average (on validation)
    weights, val_metrics = optimize_weights_grid_search(proba_val, y_val, metric='roc_auc')
    ensemble_proba_test = ensemble_weighted_average(proba_test, weights)
    test_metrics = evaluate_ensemble(y_test, ensemble_proba_test)
    
    ensemble_methods['optimized_weighted'] = {
        'weights': weights,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics
    }
    
    # 2. Gini-weighted (using validation Gini scores)
    gini_scores = [individual_metrics_val[name]['gini'] for name in model_names]
    ensemble_proba_test = ensemble_gini_weighted(proba_test, gini_scores)
    test_metrics = evaluate_ensemble(y_test, ensemble_proba_test)
    
    ensemble_methods['gini_weighted'] = {
        'weights': [g / sum(gini_scores) for g in gini_scores],
        'val_metrics': None,  # Not optimized on val, just using Gini
        'test_metrics': test_metrics
    }
    
    # 3. Rank averaging
    ensemble_proba_test = ensemble_rank_average(proba_test)
    test_metrics = evaluate_ensemble(y_test, ensemble_proba_test)
    
    ensemble_methods['rank_average'] = {
        'weights': None,
        'val_metrics': None,
        'test_metrics': test_metrics
    }
    
    # 4. Geometric mean
    ensemble_proba_test = ensemble_geometric_mean(proba_test)
    test_metrics = evaluate_ensemble(y_test, ensemble_proba_test)
    
    ensemble_methods['geometric_mean'] = {
        'weights': None,
        'val_metrics': None,
        'test_metrics': test_metrics
    }
    
    # 5. Equal weights
    equal_weights = [1.0 / len(model_names)] * len(model_names)
    ensemble_proba_test = ensemble_weighted_average(proba_test, equal_weights)
    test_metrics = evaluate_ensemble(y_test, ensemble_proba_test)
    
    ensemble_methods['equal_weights'] = {
        'weights': equal_weights,
        'val_metrics': None,
        'test_metrics': test_metrics
    }
    
    return ensemble_methods

def main():
    print("="*80)
    print("ENSEMBLE OPTIMIZATION FOR V4 MODELS")
    print("="*80)
    print(f"\n📋 Strategy:")
    print(f"   1. Split training data into train/validation ({VALIDATION_SPLIT*100:.0f}% validation)")
    print(f"   2. Generate predictions from all 6 models on validation and test sets")
    print(f"   3. Test different ensemble combinations (2-6 models)")
    print(f"   4. Optimize weights on validation set")
    print(f"   5. Evaluate best ensembles on test set")
    print("="*80)
    
    start_time = datetime.now()
    print(f"\n⏰ Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)
    
    df_train_all = load_combined_seasonal_datasets_natural(
        min_season='2018_2019',
        exclude_season='2025_2026'
    )
    df_test_all = load_test_dataset()
    
    # Prepare data for different targets
    print("\n" + "="*80)
    print("STEP 2: PREPARING DATA FOR DIFFERENT TARGETS")
    print("="*80)
    
    # Target1 data
    print("\nPreparing data for target1...")
    df_train_target1 = filter_timelines_for_model(df_train_all, 'target1')
    df_test_target1 = filter_timelines_for_model(df_test_all, 'target1')
    
    # Target2 data
    print("\nPreparing data for target2...")
    df_train_target2 = filter_timelines_for_model(df_train_all, 'target2')
    df_test_target2 = filter_timelines_for_model(df_test_all, 'target2')
    
    # Combined target data
    print("\nPreparing data for combined target...")
    df_train_combined = filter_for_combined_target(df_train_all)
    df_train_combined = create_combined_target(df_train_combined)
    df_test_combined = filter_for_combined_target(df_test_all)
    df_test_combined = create_combined_target(df_test_combined)
    
    # Prepare features
    print("\n" + "="*80)
    print("STEP 3: PREPARING FEATURES")
    print("="*80)
    
    # Use cached features if available
    print("\nPreparing features for target1...")
    X_train_target1 = prepare_data(df_train_target1, cache_file=str(CACHE_DIR / 'preprocessed_target1_train.csv'), use_cache=True)
    y_train_target1 = df_train_target1['target1'].values
    X_test_target1 = prepare_data(df_test_target1, cache_file=str(CACHE_DIR / 'preprocessed_target1_test.csv'), use_cache=True)
    y_test_target1 = df_test_target1['target1'].values
    
    print("\nPreparing features for target2...")
    X_train_target2 = prepare_data(df_train_target2, cache_file=str(CACHE_DIR / 'preprocessed_target2_train.csv'), use_cache=True)
    y_train_target2 = df_train_target2['target2'].values
    X_test_target2 = prepare_data(df_test_target2, cache_file=str(CACHE_DIR / 'preprocessed_target2_test.csv'), use_cache=True)
    y_test_target2 = df_test_target2['target2'].values
    
    print("\nPreparing features for combined target...")
    X_train_combined = prepare_data(df_train_combined, cache_file=str(CACHE_DIR / 'preprocessed_combined_train.csv'), use_cache=True)
    y_train_combined = df_train_combined['target_combined'].values
    X_test_combined = prepare_data(df_test_combined, cache_file=str(CACHE_DIR / 'preprocessed_combined_test.csv'), use_cache=True)
    y_test_combined = df_test_combined['target_combined'].values
    
    # Align features
    print("\nAligning features...")
    X_train_target1, X_test_target1 = align_features(X_train_target1, X_test_target1)
    X_train_target2, X_test_target2 = align_features(X_train_target2, X_test_target2)
    X_train_combined, X_test_combined = align_features(X_train_combined, X_test_combined)
    
    # Create validation splits
    print("\n" + "="*80)
    print("STEP 4: CREATING VALIDATION SPLITS")
    print("="*80)
    
    X_train_t1, X_val_t1, y_train_t1, y_val_t1 = train_test_split(
        X_train_target1, y_train_target1, 
        test_size=VALIDATION_SPLIT, 
        random_state=RANDOM_STATE,
        stratify=y_train_target1
    )
    
    X_train_t2, X_val_t2, y_train_t2, y_val_t2 = train_test_split(
        X_train_target2, y_train_target2, 
        test_size=VALIDATION_SPLIT, 
        random_state=RANDOM_STATE,
        stratify=y_train_target2
    )
    
    X_train_comb, X_val_comb, y_train_comb, y_val_comb = train_test_split(
        X_train_combined, y_train_combined, 
        test_size=VALIDATION_SPLIT, 
        random_state=RANDOM_STATE,
        stratify=y_train_combined
    )
    
    print(f"✅ Created validation splits ({VALIDATION_SPLIT*100:.0f}% validation)")
    
    # Load models
    print("\n" + "="*80)
    print("STEP 5: LOADING TRAINED MODELS")
    print("="*80)
    
    models = {}
    model_columns = {}
    
    # LGBM models
    print("\nLoading LGBM models...")
    models['model1_lgbm_target1'] = load_model(MODEL_DIR / 'lgbm_target1.joblib', 'lgbm')
    model_columns['model1_lgbm_target1'] = load_model_columns(MODEL_DIR / 'lgbm_target1_columns.json')
    
    models['model2_lgbm_target2'] = load_model(MODEL_DIR / 'lgbm_target2.joblib', 'lgbm')
    model_columns['model2_lgbm_target2'] = load_model_columns(MODEL_DIR / 'lgbm_target2_columns.json')
    
    models['model3_lgbm_combined'] = load_model(MODEL_DIR / 'lgbm_combined.joblib', 'lgbm')
    model_columns['model3_lgbm_combined'] = load_model_columns(MODEL_DIR / 'lgbm_combined_columns.json')
    
    # Neural Network models
    if TENSORFLOW_AVAILABLE:
        print("\nLoading Neural Network models...")
        models['model4_nn_target1'] = load_model(MODEL_DIR / 'nn_target1.h5', 'nn')
        model_columns['model4_nn_target1'] = load_model_columns(MODEL_DIR / 'nn_target1_columns.json')
        
        models['model5_nn_target2'] = load_model(MODEL_DIR / 'nn_target2.h5', 'nn')
        model_columns['model5_nn_target2'] = load_model_columns(MODEL_DIR / 'nn_target2_columns.json')
        
        models['model6_nn_combined'] = load_model(MODEL_DIR / 'nn_combined.h5', 'nn')
        model_columns['model6_nn_combined'] = load_model_columns(MODEL_DIR / 'nn_combined_columns.json')
    
    print(f"✅ Loaded {len(models)} models")
    
    # Generate predictions
    print("\n" + "="*80)
    print("STEP 6: GENERATING PREDICTIONS")
    print("="*80)
    
    # Map models to their data
    model_data_map = {
        'model1_lgbm_target1': (X_val_t1, y_val_t1, X_test_target1, y_test_target1),
        'model2_lgbm_target2': (X_val_t2, y_val_t2, X_test_target2, y_test_target2),
        'model3_lgbm_combined': (X_val_comb, y_val_comb, X_test_combined, y_test_combined),
        'model4_nn_target1': (X_val_t1, y_val_t1, X_test_target1, y_test_target1),
        'model5_nn_target2': (X_val_t2, y_val_t2, X_test_target2, y_test_target2),
        'model6_nn_combined': (X_val_comb, y_val_comb, X_test_combined, y_test_combined),
    }
    
    predictions_val = {}
    predictions_test = {}
    individual_metrics_val = {}
    individual_metrics_test = {}
    
    print("\nGenerating predictions from individual models...")
    for model_name, model in models.items():
        if model_name not in model_data_map:
            continue
            
        X_val, y_val, X_test, y_test = model_data_map[model_name]
        model_type = 'nn' if 'nn' in model_name else 'lgbm'
        
        # Align features
        X_val_aligned = X_val.reindex(columns=model_columns[model_name], fill_value=0)
        X_test_aligned = X_test.reindex(columns=model_columns[model_name], fill_value=0)
        
        # Generate predictions
        print(f"   {model_name}...")
        proba_val = generate_predictions(model, X_val_aligned, model_type)
        proba_test = generate_predictions(model, X_test_aligned, model_type)
        
        predictions_val[model_name] = proba_val
        predictions_test[model_name] = proba_test
        
        # Evaluate individual model
        individual_metrics_val[model_name] = evaluate_ensemble(y_val, proba_val)
        individual_metrics_test[model_name] = evaluate_ensemble(y_test, proba_test)
    
    print(f"✅ Generated predictions from {len(predictions_val)} models")
    
    # Define ensemble combinations to test
    print("\n" + "="*80)
    print("STEP 7: TESTING ENSEMBLE COMBINATIONS")
    print("="*80)
    
    # Group models by target
    target1_models = ['model1_lgbm_target1', 'model4_nn_target1']
    target2_models = ['model2_lgbm_target2', 'model5_nn_target2']
    combined_models = ['model3_lgbm_combined', 'model6_nn_combined']
    all_lgbm = ['model1_lgbm_target1', 'model2_lgbm_target2', 'model3_lgbm_combined']
    all_nn = ['model4_nn_target1', 'model5_nn_target2', 'model6_nn_combined']
    all_models = list(models.keys())
    
    # Define combinations to test
    combinations_to_test = [
        # Single models (baseline)
        (['model1_lgbm_target1'], 'target1'),
        (['model2_lgbm_target2'], 'target2'),
        (['model3_lgbm_combined'], 'combined'),
        (['model4_nn_target1'], 'target1'),
        (['model5_nn_target2'], 'target2'),
        (['model6_nn_combined'], 'combined'),
        
        # 2-model ensembles - same target
        (target1_models, 'target1'),
        (target2_models, 'target2'),
        (combined_models, 'combined'),
        
        # 2-model ensembles - cross target
        (['model1_lgbm_target1', 'model3_lgbm_combined'], 'target1'),
        (['model4_nn_target1', 'model6_nn_combined'], 'target1'),
        (['model1_lgbm_target1', 'model4_nn_target1'], 'target1'),
        (['model3_lgbm_combined', 'model6_nn_combined'], 'combined'),
        
        # 3-model ensembles
        (all_lgbm, 'combined'),
        (all_nn, 'combined'),
        (['model1_lgbm_target1', 'model3_lgbm_combined', 'model6_nn_combined'], 'combined'),
        (['model1_lgbm_target1', 'model4_nn_target1', 'model6_nn_combined'], 'combined'),
        
        # 4+ model ensembles
        (['model1_lgbm_target1', 'model2_lgbm_target2', 'model4_nn_target1', 'model5_nn_target2'], 'combined'),
        (['model1_lgbm_target1', 'model3_lgbm_combined', 'model4_nn_target1', 'model6_nn_combined'], 'combined'),
        (all_models, 'combined'),
    ]
    
    # Filter to only include models that exist
    available_models = list(models.keys())
    valid_combinations = []
    for combo, target_type in combinations_to_test:
        if all(m in available_models for m in combo):
            valid_combinations.append((combo, target_type))
    
    print(f"\n📊 Testing {len(valid_combinations)} ensemble combinations...")
    
    all_results = []
    
    for combo, target_type in tqdm(valid_combinations, desc="Testing ensembles"):
        # Get appropriate validation and test data based on target_type
        # For cross-target ensembles, we need to regenerate predictions on the combined dataset
        if target_type == 'target1':
            y_val = y_val_t1
            y_test = y_test_target1
            # Get predictions - ensure all models use target1 data
            proba_val_dict = {}
            proba_test_dict = {}
            for name in combo:
                if name in ['model1_lgbm_target1', 'model4_nn_target1']:
                    # Already on target1 data
                    proba_val_dict[name] = predictions_val[name]
                    proba_test_dict[name] = predictions_test[name]
                else:
                    # Need to regenerate on target1 data
                    X_val_use, y_val_use, X_test_use, y_test_use = model_data_map['model1_lgbm_target1']
                    model_type = 'nn' if 'nn' in name else 'lgbm'
                    X_val_aligned = X_val_use.reindex(columns=model_columns[name], fill_value=0)
                    X_test_aligned = X_test_use.reindex(columns=model_columns[name], fill_value=0)
                    proba_val_dict[name] = generate_predictions(models[name], X_val_aligned, model_type)
                    proba_test_dict[name] = generate_predictions(models[name], X_test_aligned, model_type)
        elif target_type == 'target2':
            y_val = y_val_t2
            y_test = y_test_target2
            # Get predictions - ensure all models use target2 data
            proba_val_dict = {}
            proba_test_dict = {}
            for name in combo:
                if name in ['model2_lgbm_target2', 'model5_nn_target2']:
                    # Already on target2 data
                    proba_val_dict[name] = predictions_val[name]
                    proba_test_dict[name] = predictions_test[name]
                else:
                    # Need to regenerate on target2 data
                    X_val_use, y_val_use, X_test_use, y_test_use = model_data_map['model2_lgbm_target2']
                    model_type = 'nn' if 'nn' in name else 'lgbm'
                    X_val_aligned = X_val_use.reindex(columns=model_columns[name], fill_value=0)
                    X_test_aligned = X_test_use.reindex(columns=model_columns[name], fill_value=0)
                    proba_val_dict[name] = generate_predictions(models[name], X_val_aligned, model_type)
                    proba_test_dict[name] = generate_predictions(models[name], X_test_aligned, model_type)
        else:  # combined
            y_val = y_val_comb
            y_test = y_test_combined
            # Get predictions - ensure all models use combined data
            proba_val_dict = {}
            proba_test_dict = {}
            for name in combo:
                if name in ['model3_lgbm_combined', 'model6_nn_combined']:
                    # Already on combined data
                    proba_val_dict[name] = predictions_val[name]
                    proba_test_dict[name] = predictions_test[name]
                else:
                    # Need to regenerate on combined data
                    X_val_use, y_val_use, X_test_use, y_test_use = model_data_map['model3_lgbm_combined']
                    model_type = 'nn' if 'nn' in name else 'lgbm'
                    X_val_aligned = X_val_use.reindex(columns=model_columns[name], fill_value=0)
                    X_test_aligned = X_test_use.reindex(columns=model_columns[name], fill_value=0)
                    proba_val_dict[name] = generate_predictions(models[name], X_val_aligned, model_type)
                    proba_test_dict[name] = generate_predictions(models[name], X_test_aligned, model_type)
        
        # Get predictions as lists
        proba_val = [proba_val_dict[name] for name in combo]
        proba_test = [proba_test_dict[name] for name in combo]
        
        # Get individual metrics (recalculate on the correct data)
        ind_metrics_val = {}
        ind_metrics_test = {}
        for name in combo:
            ind_metrics_val[name] = evaluate_ensemble(y_val, proba_val_dict[name])
            ind_metrics_test[name] = evaluate_ensemble(y_test, proba_test_dict[name])
        
        # Test ensemble methods
        ensemble_results = test_ensemble_combination(
            combo, 
            proba_val_dict, y_val,
            proba_test_dict, y_test,
            ind_metrics_val, ind_metrics_test
        )
        
        # Store results
        for method_name, method_results in ensemble_results.items():
            all_results.append({
                'ensemble_name': '+'.join(combo),
                'n_models': len(combo),
                'target_type': target_type,
                'method': method_name,
                'weights': method_results['weights'],
                'val_roc_auc': method_results['val_metrics']['roc_auc'] if method_results['val_metrics'] else None,
                'val_f1': method_results['val_metrics']['f1'] if method_results['val_metrics'] else None,
                'val_gini': method_results['val_metrics']['gini'] if method_results['val_metrics'] else None,
                'val_precision': method_results['val_metrics']['precision'] if method_results['val_metrics'] else None,
                'val_recall': method_results['val_metrics']['recall'] if method_results['val_metrics'] else None,
                'test_roc_auc': method_results['test_metrics']['roc_auc'],
                'test_f1': method_results['test_metrics']['f1'],
                'test_gini': method_results['test_metrics']['gini'],
                'test_precision': method_results['test_metrics']['precision'],
                'test_recall': method_results['test_metrics']['recall'],
                'test_accuracy': method_results['test_metrics']['accuracy'],
                'test_tp': method_results['test_metrics']['confusion_matrix']['tp'],
                'test_fp': method_results['test_metrics']['confusion_matrix']['fp'],
                'test_tn': method_results['test_metrics']['confusion_matrix']['tn'],
                'test_fn': method_results['test_metrics']['confusion_matrix']['fn'],
            })
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    results_file = OUTPUT_DIR / 'ensemble_optimization_results.csv'
    results_df.to_csv(results_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ Saved results to: {results_file}")
    
    # Create summary table
    print("\n" + "="*80)
    print("ENSEMBLE OPTIMIZATION RESULTS - SUMMARY")
    print("="*80)
    
    # Sort by test ROC-AUC
    results_df_sorted = results_df.sort_values('test_roc_auc', ascending=False)
    
    # Show top 20 ensembles
    print("\n📊 TOP 20 ENSEMBLES BY TEST ROC-AUC:")
    print("="*80)
    top_20 = results_df_sorted.head(20)
    
    display_cols = ['ensemble_name', 'n_models', 'method', 'test_roc_auc', 'test_gini', 
                   'test_f1', 'test_precision', 'test_recall']
    print(top_20[display_cols].to_string(index=False))
    
    # Show best ensemble for each method
    print("\n📊 BEST ENSEMBLE BY METHOD:")
    print("="*80)
    best_by_method = results_df_sorted.groupby('method').first().reset_index()
    if 'method' in best_by_method.columns:
        print(best_by_method[display_cols].to_string(index=False))
    else:
        print(best_by_method[display_cols].to_string())
    
    # Show best overall
    best_overall = results_df_sorted.iloc[0]
    print("\n" + "="*80)
    print("🏆 BEST OVERALL ENSEMBLE:")
    print("="*80)
    print(f"   Ensemble: {best_overall['ensemble_name']}")
    print(f"   Method: {best_overall['method']}")
    print(f"   Weights: {best_overall['weights']}")
    print(f"\n   Validation Metrics:")
    if best_overall['val_roc_auc']:
        print(f"      ROC-AUC: {best_overall['val_roc_auc']:.4f}")
        print(f"      F1: {best_overall['val_f1']:.4f}")
        print(f"      Gini: {best_overall['val_gini']:.4f}")
    print(f"\n   Test Metrics:")
    print(f"      ROC-AUC: {best_overall['test_roc_auc']:.4f}")
    print(f"      F1: {best_overall['test_f1']:.4f}")
    print(f"      Gini: {best_overall['test_gini']:.4f}")
    print(f"      Precision: {best_overall['test_precision']:.4f}")
    print(f"      Recall: {best_overall['test_recall']:.4f}")
    print(f"      Accuracy: {best_overall['test_accuracy']:.4f}")
    print(f"      TP: {best_overall['test_tp']}, FP: {best_overall['test_fp']}, "
          f"TN: {best_overall['test_tn']}, FN: {best_overall['test_fn']}")
    
    # Save best ensemble configuration
    best_config = {
        'ensemble_name': best_overall['ensemble_name'],
        'method': best_overall['method'],
        'weights': best_overall['weights'],
        'target_type': best_overall['target_type'],
        'validation_metrics': {
            'roc_auc': best_overall['val_roc_auc'],
            'f1': best_overall['val_f1'],
            'gini': best_overall['val_gini'],
            'precision': best_overall['val_precision'],
            'recall': best_overall['val_recall'],
        },
        'test_metrics': {
            'roc_auc': best_overall['test_roc_auc'],
            'f1': best_overall['test_f1'],
            'gini': best_overall['test_gini'],
            'precision': best_overall['test_precision'],
            'recall': best_overall['test_recall'],
            'accuracy': best_overall['test_accuracy'],
            'confusion_matrix': {
                'tp': int(best_overall['test_tp']),
                'fp': int(best_overall['test_fp']),
                'tn': int(best_overall['test_tn']),
                'fn': int(best_overall['test_fn']),
            }
        }
    }
    
    config_file = OUTPUT_DIR / 'best_ensemble_config.json'
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(best_config, f, indent=2)
    print(f"\n✅ Saved best ensemble configuration to: {config_file}")
    
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds() / 60
    print(f"\n⏰ Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Total time: {total_time:.1f} minutes")
    
    return results_df

if __name__ == "__main__":
    results = main()
