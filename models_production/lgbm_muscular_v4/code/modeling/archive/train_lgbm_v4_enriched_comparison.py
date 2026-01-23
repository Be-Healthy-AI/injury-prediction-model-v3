#!/usr/bin/env python3
"""
Train 3 LGBM Models with Enriched Features (Layer 2) for Comparison:
1. LGBM on target1 (muscular injuries)
2. LGBM on target2 (skeletal injuries)
3. LGBM on combined target (target1=1 OR target2=1)

This script trains models using the newly regenerated timelines with Layer 2 enriched features
and compares results with previous training runs.

All models tested on 2025/26 season test dataset.
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import json
import glob
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from pathlib import Path
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix
)
from tqdm import tqdm

# Add paths for imports
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent.parent.parent
TIMELINES_DIR = SCRIPT_DIR.parent / 'timelines'
sys.path.insert(0, str(TIMELINES_DIR.resolve()))

# Import filter function from timeline generation script
import importlib.util
timeline_script_path = TIMELINES_DIR / 'create_35day_timelines_v4_enhanced.py'
if not timeline_script_path.exists():
    raise FileNotFoundError(f"Timeline script not found: {timeline_script_path}")
spec = importlib.util.spec_from_file_location("create_35day_timelines_v4_enhanced", timeline_script_path)
timeline_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timeline_module)
filter_timelines_for_model = timeline_module.filter_timelines_for_model

# ========== CONFIGURATION ==========
MIN_SEASON = '2018_2019'  # Start from 2018/19 season (inclusive)
EXCLUDE_SEASON = '2025_2026'  # Test dataset season
TRAIN_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'data' / 'timelines' / 'train'
TEST_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'data' / 'timelines' / 'test'
MODEL_OUTPUT_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'models' / 'enriched_comparison'
CACHE_DIR = ROOT_DIR / 'cache' / 'enriched_v4'
USE_CACHE = False  # Disable cache to ensure fresh training with new features

# ========== FEATURE EXCLUSION CONFIGURATION ==========
# Features to exclude to reduce over-reliance on specific feature groups

# Injury history features to exclude (reduce over-reliance)
EXCLUDE_INJURY_HISTORY_FEATURES = [
    'muscular_injuries_last_2_years',
    'injuries_last_2_years',
    'injuries_last_365d',
    'injuries_last_90d',
    'muscular_injury_count',  # Career total - too correlated with history
    'muscular_to_total_ratio',  # Derived from history
    'upper_leg_injuries',  # If this is a history feature
    'skeletal_injuries_last_2_years',  # For skeletal model
]

# Workload spike/acceleration features to exclude (reduce over-reliance)
EXCLUDE_WORKLOAD_FEATURES = [
    # All workload spike features (all week windows)
    'workload_spike_7d_week_1',
    'workload_spike_7d_week_2',
    'workload_spike_7d_week_3',
    'workload_spike_7d_week_4',
    'workload_spike_7d_week_5',
    'workload_spike_14d_week_1',
    'workload_spike_14d_week_2',
    'workload_spike_14d_week_3',
    'workload_spike_14d_week_4',
    'workload_spike_14d_week_5',
    # All workload acceleration features (all week windows)
    'workload_acceleration_7d_week_1',
    'workload_acceleration_7d_week_2',
    'workload_acceleration_7d_week_3',
    'workload_acceleration_7d_week_4',
    'workload_acceleration_7d_week_5',
    'workload_acceleration_14d_week_1',
    'workload_acceleration_14d_week_2',
    'workload_acceleration_14d_week_3',
    'workload_acceleration_14d_week_4',
    'workload_acceleration_14d_week_5',
    # Base workload acceleration features (if they exist without week suffix)
    'workload_spike_7d',
    'workload_spike_14d',
    'workload_acceleration_7d',
    'workload_acceleration_14d',
]

# Combine all exclusions
FEATURES_TO_EXCLUDE = list(set(EXCLUDE_INJURY_HISTORY_FEATURES + EXCLUDE_WORKLOAD_FEATURES + ['season_year']))
# ===================================

# Copy helper functions from existing script
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

# Data loading functions
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
    
    # Exclude features that cause over-reliance
    feature_columns = [col for col in feature_columns if col not in FEATURES_TO_EXCLUDE]
    
    # Also exclude any columns that start with excluded feature names (for one-hot encoded versions)
    excluded_patterns = []
    for excluded_feat in FEATURES_TO_EXCLUDE:
        excluded_patterns.extend([col for col in feature_columns if col.startswith(excluded_feat + '_')])
    
    feature_columns = [col for col in feature_columns if col not in excluded_patterns]
    
    if len(FEATURES_TO_EXCLUDE) > 0:
        print(f"   ⚠️  Excluding {len(FEATURES_TO_EXCLUDE)} features to reduce over-reliance")
        excluded_display = ', '.join(FEATURES_TO_EXCLUDE[:10])
        if len(FEATURES_TO_EXCLUDE) > 10:
            excluded_display += f'... (+{len(FEATURES_TO_EXCLUDE) - 10} more)'
        print(f"      Excluded: {excluded_display}")
        if len(excluded_patterns) > 0:
            print(f"      Also excluded {len(excluded_patterns)} one-hot encoded variants")
    
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
    """
    Create combined target: target1=1 OR target2=1
    Negatives: target1=0 AND target2=0 (only non-injuries)
    """
    df = df.copy()
    df['target_combined'] = ((df['target1'] == 1) | (df['target2'] == 1)).astype(int)
    return df

def filter_for_combined_target(df):
    """
    Filter dataset for combined target model.
    Includes: target1=1 OR target2=1 (positives) OR (target1=0 AND target2=0) (negatives)
    """
    # Positives: any injury (target1=1 OR target2=1)
    # Negatives: no injuries (target1=0 AND target2=0)
    mask = ((df['target1'] == 1) | (df['target2'] == 1)) | ((df['target1'] == 0) & (df['target2'] == 0))
    return df[mask].copy()

def evaluate_model(model, X, y, dataset_name):
    """Evaluate model and return metrics"""
    if len(X) > 100000:
        chunk_size = 50000
        y_pred_list = []
        y_proba_list = []
        num_chunks = (len(X) + chunk_size - 1) // chunk_size
        for i in tqdm(range(0, len(X), chunk_size), 
                     desc=f"      Predicting {dataset_name}", 
                     unit="chunk",
                     total=num_chunks,
                     leave=False):
            chunk_X = X.iloc[i:i+chunk_size]
            y_pred_list.append(model.predict(chunk_X))
            y_proba_list.append(model.predict_proba(chunk_X)[:, 1])
        y_pred = np.concatenate(y_pred_list)
        y_proba = np.concatenate(y_proba_list)
    else:
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': float(accuracy_score(y, y_pred)),
        'precision': float(precision_score(y, y_pred, zero_division=0)),
        'recall': float(recall_score(y, y_pred, zero_division=0)),
        'f1': float(f1_score(y, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y, y_proba)) if len(np.unique(y)) > 1 else 0.0,
        'gini': float((2 * roc_auc_score(y, y_proba) - 1)) if len(np.unique(y)) > 1 else 0.0
    }
    
    cm = confusion_matrix(y, y_pred)
    if cm.shape == (2, 2):
        metrics['confusion_matrix'] = {
            'tn': int(cm[0, 0]),
            'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]),
            'tp': int(cm[1, 1])
        }
    else:
        if len(cm) == 1:
            if y.sum() == 0:
                metrics['confusion_matrix'] = {'tn': int(cm[0, 0]), 'fp': 0, 'fn': 0, 'tp': 0}
            else:
                metrics['confusion_matrix'] = {'tn': 0, 'fp': 0, 'fn': int(y.sum()), 'tp': 0}
        else:
            metrics['confusion_matrix'] = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
    
    print(f"\n   📊 {dataset_name} Results:")
    print(f"      Accuracy: {metrics['accuracy']:.4f}")
    print(f"      Precision: {metrics['precision']:.4f}")
    print(f"      Recall: {metrics['recall']:.4f}")
    print(f"      F1-Score: {metrics['f1']:.4f}")
    print(f"      ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"      Gini: {metrics['gini']:.4f}")
    print(f"      TP: {metrics['confusion_matrix']['tp']}, FP: {metrics['confusion_matrix']['fp']}, "
          f"TN: {metrics['confusion_matrix']['tn']}, FN: {metrics['confusion_matrix']['fn']}")
    
    return metrics

def train_lgbm_model(X_train, y_train, X_test, y_test, model_name, target_name):
    """Train a LightGBM model and evaluate it"""
    print(f"\n{'='*80}")
    print(f"🚀 TRAINING LGBM MODEL: {model_name} ({target_name})")
    print(f"{'='*80}")
    
    lgbm_model = LGBMClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.5,  # Increased from 1.0 to reduce over-reliance on specific features
        min_gain_to_split=0.1,  # NEW: Minimum gain to split - helps reduce overfitting
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    print(f"\n🔧 Model hyperparameters:")
    print(f"   n_estimators: {lgbm_model.n_estimators}")
    print(f"   max_depth: {lgbm_model.max_depth}")
    print(f"   learning_rate: {lgbm_model.learning_rate}")
    print(f"   class_weight: {lgbm_model.class_weight}")
    
    print(f"\n📊 Training data:")
    print(f"   Total samples: {len(X_train):,}")
    print(f"   Positives: {y_train.sum():,} ({y_train.mean()*100:.4f}%)")
    print(f"   Negatives: {(y_train == 0).sum():,}")
    
    print(f"\n⏳ Training model...")
    start_time = datetime.now()
    lgbm_model.fit(X_train, y_train)
    training_time = (datetime.now() - start_time).total_seconds()
    print(f"✅ Training completed in {training_time:.1f} seconds")
    
    train_metrics = evaluate_model(lgbm_model, X_train, y_train, "Training")
    test_metrics = evaluate_model(lgbm_model, X_test, y_test, "Test")
    
    return lgbm_model, train_metrics, test_metrics

def load_previous_results():
    """Load previous model results for comparison"""
    previous_file = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'models' / 'comparison' / 'all_models_comparison_metrics.json'
    if previous_file.exists():
        try:
            with open(previous_file, 'r') as f:
                prev_results = json.load(f)
            print(f"✅ Loaded previous results from {previous_file.name}")
            return prev_results
        except Exception as e:
            print(f"⚠️  Could not load previous results: {e}")
            return None
    else:
        print(f"⚠️  Previous results file not found: {previous_file}")
        return None

def main():
    print("="*80)
    print("TRAINING 3 LGBM MODELS WITH ENRICHED FEATURES (LAYER 2)")
    print("="*80)
    print(f"\n📋 Models to train:")
    print(f"   1. LGBM on target1 (muscular injuries)")
    print(f"   2. LGBM on target2 (skeletal injuries)")
    print(f"   3. LGBM on combined target (target1=1 OR target2=1)")
    print(f"\n📋 Configuration:")
    print(f"   Training: Seasons {MIN_SEASON} onwards (natural target ratio)")
    print(f"   Test: Season {EXCLUDE_SEASON} (natural target ratio)")
    print(f"   Features: Layer 2 enriched features (workload, recovery, injury history)")
    print(f"   Cache: {'Enabled' if USE_CACHE else 'Disabled (fresh training)'}")
    print("="*80)
    
    start_time = datetime.now()
    print(f"\n⏰ Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directories
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load previous results for comparison
    print("\n" + "="*80)
    print("LOADING PREVIOUS RESULTS FOR COMPARISON")
    print("="*80)
    prev_results = load_previous_results()
    
    # Load data
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)
    df_train_all = load_combined_seasonal_datasets_natural(
        min_season=MIN_SEASON,
        exclude_season=EXCLUDE_SEASON
    )
    df_test_all = load_test_dataset()
    
    # Prepare data for different targets
    print("\n" + "="*80)
    print("STEP 2: PREPARING DATA FOR DIFFERENT TARGETS")
    print("="*80)
    
    # Model 1: target1 (muscular)
    print("\nPreparing data for Model 1 (target1)...")
    df_train_target1 = filter_timelines_for_model(df_train_all, 'target1')
    df_test_target1 = filter_timelines_for_model(df_test_all, 'target1')
    
    # Model 2: target2 (skeletal)
    print("\nPreparing data for Model 2 (target2)...")
    df_train_target2 = filter_timelines_for_model(df_train_all, 'target2')
    df_test_target2 = filter_timelines_for_model(df_test_all, 'target2')
    
    # Model 3: combined target
    print("\nPreparing data for Model 3 (combined target)...")
    df_train_combined = create_combined_target(df_train_all)
    df_train_combined = filter_for_combined_target(df_train_combined)
    df_test_combined = create_combined_target(df_test_all)
    df_test_combined = filter_for_combined_target(df_test_combined)
    
    # Prepare features
    print("\n" + "="*80)
    print("STEP 3: PREPARING FEATURES")
    print("="*80)
    
    print("\nPreparing features for Model 1 (target1)...")
    cache_file_target1_train = str(CACHE_DIR / 'preprocessed_target1_train_enriched.csv')
    cache_file_target1_test = str(CACHE_DIR / 'preprocessed_target1_test_enriched.csv')
    X_train_target1 = prepare_data(df_train_target1, cache_file=cache_file_target1_train, use_cache=USE_CACHE)
    y_train_target1 = df_train_target1['target1'].values
    X_test_target1 = prepare_data(df_test_target1, cache_file=cache_file_target1_test, use_cache=USE_CACHE)
    y_test_target1 = df_test_target1['target1'].values
    
    print("\nPreparing features for Model 2 (target2)...")
    cache_file_target2_train = str(CACHE_DIR / 'preprocessed_target2_train_enriched.csv')
    cache_file_target2_test = str(CACHE_DIR / 'preprocessed_target2_test_enriched.csv')
    X_train_target2 = prepare_data(df_train_target2, cache_file=cache_file_target2_train, use_cache=USE_CACHE)
    y_train_target2 = df_train_target2['target2'].values
    X_test_target2 = prepare_data(df_test_target2, cache_file=cache_file_target2_test, use_cache=USE_CACHE)
    y_test_target2 = df_test_target2['target2'].values
    
    print("\nPreparing features for Model 3 (combined target)...")
    cache_file_combined_train = str(CACHE_DIR / 'preprocessed_combined_train_enriched.csv')
    cache_file_combined_test = str(CACHE_DIR / 'preprocessed_combined_test_enriched.csv')
    X_train_combined = prepare_data(df_train_combined, cache_file=cache_file_combined_train, use_cache=USE_CACHE)
    y_train_combined = df_train_combined['target_combined'].values
    X_test_combined = prepare_data(df_test_combined, cache_file=cache_file_combined_test, use_cache=USE_CACHE)
    y_test_combined = df_test_combined['target_combined'].values
    
    # Align features
    print("\nAligning features between train and test sets...")
    X_train_target1, X_test_target1 = align_features(X_train_target1, X_test_target1)
    X_train_target2, X_test_target2 = align_features(X_train_target2, X_test_target2)
    X_train_combined, X_test_combined = align_features(X_train_combined, X_test_combined)
    
    print(f"   Target1 features: {len(X_train_target1.columns)}")
    print(f"   Target2 features: {len(X_train_target2.columns)}")
    print(f"   Combined features: {len(X_train_combined.columns)}")
    
    # Store all results
    all_results = {}
    
    # Train Model 1: LGBM on target1 (Muscular)
    print("\n" + "="*80)
    print("STEP 4: TRAINING MODEL 1 - LGBM ON TARGET1 (MUSCULAR)")
    print("="*80)
    model1, train_metrics1, test_metrics1 = train_lgbm_model(
        X_train_target1, y_train_target1,
        X_test_target1, y_test_target1,
        "Model 1: LGBM - Muscular Injuries",
        "target1"
    )
    all_results['model1_lgbm_target1'] = {
        'train': train_metrics1,
        'test': test_metrics1
    }
    
    # Train Model 2: LGBM on target2 (Skeletal)
    print("\n" + "="*80)
    print("STEP 5: TRAINING MODEL 2 - LGBM ON TARGET2 (SKELETAL)")
    print("="*80)
    model2, train_metrics2, test_metrics2 = train_lgbm_model(
        X_train_target2, y_train_target2,
        X_test_target2, y_test_target2,
        "Model 2: LGBM - Skeletal Injuries",
        "target2"
    )
    all_results['model2_lgbm_target2'] = {
        'train': train_metrics2,
        'test': test_metrics2
    }
    
    # Train Model 3: LGBM on combined target (optional, for completeness)
    print("\n" + "="*80)
    print("STEP 6: TRAINING MODEL 3 - LGBM ON COMBINED TARGET (OPTIONAL)")
    print("="*80)
    model3, train_metrics3, test_metrics3 = train_lgbm_model(
        X_train_combined, y_train_combined,
        X_test_combined, y_test_combined,
        "Model 3: LGBM - Combined Injuries (target1 OR target2)",
        "target_combined"
    )
    all_results['model3_lgbm_combined'] = {
        'train': train_metrics3,
        'test': test_metrics3
    }
    
    # Save models
    print("\n" + "="*80)
    print("STEP 7: SAVING MODELS")
    print("="*80)
    
    # Save models with naming that matches baseline for comparison
    MODEL_OUTPUT_BASE = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'models'
    
    model1_path = MODEL_OUTPUT_BASE / 'lgbm_muscular_v4_enriched_model1.joblib'
    model1_cols_path = MODEL_OUTPUT_BASE / 'lgbm_muscular_v4_enriched_model1_columns.json'
    joblib.dump(model1, model1_path)
    with open(model1_cols_path, 'w') as f:
        json.dump(list(X_train_target1.columns), f, indent=2)
    print(f"✅ Saved Model 1 (Muscular): {model1_path}")
    
    model2_path = MODEL_OUTPUT_BASE / 'lgbm_muscular_v4_enriched_model2.joblib'
    model2_cols_path = MODEL_OUTPUT_BASE / 'lgbm_muscular_v4_enriched_model2_columns.json'
    joblib.dump(model2, model2_path)
    with open(model2_cols_path, 'w') as f:
        json.dump(list(X_train_target2.columns), f, indent=2)
    print(f"✅ Saved Model 2 (Skeletal): {model2_path}")
    
    # Also save combined model
    joblib.dump(model3, MODEL_OUTPUT_DIR / 'lgbm_combined_enriched.joblib')
    with open(MODEL_OUTPUT_DIR / 'lgbm_combined_enriched_columns.json', 'w') as f:
        json.dump(list(X_train_combined.columns), f, indent=2)
    print(f"✅ Saved Model 3 (Combined): {MODEL_OUTPUT_DIR / 'lgbm_combined_enriched.joblib'}")
    
    # Save all metrics with baseline-compatible structure
    metrics_baseline_format = {
        'model1_muscular': {
            'train': train_metrics1,
            'test': test_metrics1
        },
        'model2_skeletal': {
            'train': train_metrics2,
            'test': test_metrics2
        },
        'configuration': {
            'version': 'v4_enriched',
            'min_season': MIN_SEASON,
            'exclude_season': EXCLUDE_SEASON,
            'training_date': start_time.isoformat(),
            'features': 'Layer 2 enriched (workload, recovery, injury history)',
            'num_features_model1': len(X_train_target1.columns),
            'num_features_model2': len(X_train_target2.columns)
        }
    }
    
    metrics_path = MODEL_OUTPUT_BASE / 'lgbm_muscular_v4_enriched_metrics.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_types(metrics_baseline_format), f, indent=2)
    print(f"✅ Saved metrics (baseline format): {metrics_path}")
    
    # Also save full results
    all_results['configuration'] = {
        'min_season': MIN_SEASON,
        'exclude_season': EXCLUDE_SEASON,
        'training_date': start_time.isoformat(),
        'features': 'Layer 2 enriched (workload, recovery, injury history)',
        'num_features_target1': len(X_train_target1.columns),
        'num_features_target2': len(X_train_target2.columns),
        'num_features_combined': len(X_train_combined.columns)
    }
    
    full_metrics_path = MODEL_OUTPUT_DIR / 'enriched_models_metrics.json'
    with open(full_metrics_path, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_types(all_results), f, indent=2)
    print(f"✅ Saved full metrics: {full_metrics_path}")
    
    # Create comparison table
    print("\n" + "="*80)
    print("FINAL SUMMARY - ENRICHED MODELS RESULTS")
    print("="*80)
    
    # Create DataFrame for comparison
    comparison_data = []
    for model_name, results in all_results.items():
        if 'configuration' in model_name:
            continue
        if 'train' in results and 'test' in results:
            comparison_data.append({
                'Model': model_name,
                'Train_ROC_AUC': results['train']['roc_auc'],
                'Train_F1': results['train']['f1'],
                'Train_Gini': results['train']['gini'],
                'Train_Precision': results['train']['precision'],
                'Train_Recall': results['train']['recall'],
                'Test_ROC_AUC': results['test']['roc_auc'],
                'Test_F1': results['test']['f1'],
                'Test_Gini': results['test']['gini'],
                'Test_Precision': results['test']['precision'],
                'Test_Recall': results['test']['recall'],
            })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        print("\n📊 ENRICHED MODELS RESULTS TABLE:")
        print("="*80)
        print(comparison_df.to_string(index=False))
        
        # Save comparison table
        comparison_csv = MODEL_OUTPUT_DIR / 'enriched_models_comparison_table.csv'
        comparison_df.to_csv(comparison_csv, index=False, encoding='utf-8-sig')
        print(f"\n✅ Saved comparison table: {comparison_csv}")
    
    # Compare with previous results if available
    if prev_results:
        print("\n" + "="*80)
        print("COMPARISON WITH PREVIOUS RESULTS")
        print("="*80)
        
        comparison_with_prev = []
        model_mapping = {
            'model1_lgbm_target1': 'model1_lgbm_target1',
            'model2_lgbm_target2': 'model2_lgbm_target2',
            'model3_lgbm_combined': 'model3_lgbm_combined'
        }
        
        for new_name, prev_name in model_mapping.items():
            if new_name in all_results and prev_name in prev_results:
                new_res = all_results[new_name]
                prev_res = prev_results[prev_name]
                
                if 'train' in new_res and 'test' in new_res and 'train' in prev_res and 'test' in prev_res:
                    comparison_with_prev.append({
                        'Model': new_name,
                        'Metric': 'ROC_AUC',
                        'Train_Previous': prev_res['train']['roc_auc'],
                        'Train_Enriched': new_res['train']['roc_auc'],
                        'Train_Diff': new_res['train']['roc_auc'] - prev_res['train']['roc_auc'],
                        'Test_Previous': prev_res['test']['roc_auc'],
                        'Test_Enriched': new_res['test']['roc_auc'],
                        'Test_Diff': new_res['test']['roc_auc'] - prev_res['test']['roc_auc'],
                    })
                    comparison_with_prev.append({
                        'Model': new_name,
                        'Metric': 'F1',
                        'Train_Previous': prev_res['train']['f1'],
                        'Train_Enriched': new_res['train']['f1'],
                        'Train_Diff': new_res['train']['f1'] - prev_res['train']['f1'],
                        'Test_Previous': prev_res['test']['f1'],
                        'Test_Enriched': new_res['test']['f1'],
                        'Test_Diff': new_res['test']['f1'] - prev_res['test']['f1'],
                    })
                    comparison_with_prev.append({
                        'Model': new_name,
                        'Metric': 'Gini',
                        'Train_Previous': prev_res['train']['gini'],
                        'Train_Enriched': new_res['train']['gini'],
                        'Train_Diff': new_res['train']['gini'] - prev_res['train']['gini'],
                        'Test_Previous': prev_res['test']['gini'],
                        'Test_Enriched': new_res['test']['gini'],
                        'Test_Diff': new_res['test']['gini'] - prev_res['test']['gini'],
                    })
        
        if comparison_with_prev:
            comp_df = pd.DataFrame(comparison_with_prev)
            print("\n📊 COMPARISON TABLE (Previous vs Enriched):")
            print("="*80)
            print(comp_df.to_string(index=False))
            
            comp_csv = MODEL_OUTPUT_DIR / 'comparison_previous_vs_enriched.csv'
            comp_df.to_csv(comp_csv, index=False, encoding='utf-8-sig')
            print(f"\n✅ Saved comparison table: {comp_csv}")
    
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds() / 60
    print(f"\n⏰ Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Total time: {total_time:.1f} minutes")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
