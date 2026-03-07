#!/usr/bin/env python3
"""
Train V4 620-Feature Model - Fair Comparison with V3

This script trains 3 versions of V4 with 620 optimal features:
1. V4 620 (Current): Training only (2018/19-2024/25, excluding test)
2. V4 620 (With Test): Training + Test (2018/19-2025/26, all seasons)
3. V4 620 (With Test, Excl 2021/22-2022/23): Training + Test, excluding low-injury-rate seasons

Then compares all 3 with V3 Production model.

This script avoids importlib by including all necessary functions directly.
"""

import sys
import os
import json
import traceback
import glob
import hashlib
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

# Add paths
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent.parent.parent
MODEL_OUTPUT_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'models'
V3_MODEL_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v3' / 'model'
TIMELINES_DIR = SCRIPT_DIR.parent / 'timelines'
CACHE_DIR = ROOT_DIR / 'cache'

# ========== CONFIGURATION ==========
ITERATIVE_RESULTS_FILE = MODEL_OUTPUT_DIR / 'iterative_feature_selection_results.json'
V3_METRICS_FILE = V3_MODEL_DIR / 'lgbm_v3_pl_only_metrics_test.json'
OUTPUT_METRICS_FILE = MODEL_OUTPUT_DIR / 'v4_620_comparison_metrics.json'
OUTPUT_TABLE_FILE = MODEL_OUTPUT_DIR / 'v4_620_comparison_table.md'
LOG_FILE = MODEL_OUTPUT_DIR / 'v4_620_comparison.log'

# Training configuration
MIN_SEASON = '2018_2019'
TRAIN_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'data' / 'timelines' / 'train'
TEST_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'data' / 'timelines' / 'test'
USE_CACHE = True
# ===================================

# Initialize log file
try:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(f"=== V4 620 Features Fair Comparison Started at {datetime.now().isoformat()} ===\n")
except Exception as e:
    print(f"Warning: Could not initialize log file: {e}")

def log_message(message, level="INFO"):
    """Log a message to both console and log file"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] [{level}] {message}"
    try:
        print(log_entry)
    except UnicodeEncodeError:
        safe_message = message.encode('ascii', 'ignore').decode('ascii')
        print(f"[{timestamp}] [{level}] {safe_message}")
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')
    except Exception as e:
        try:
            print(f"Warning: Could not write to log file: {e}")
        except:
            pass

def log_error(message, exception=None):
    """Log an error message with optional exception details"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] [ERROR] {message}"
    if exception:
        log_entry += f"\nException: {str(exception)}"
        log_entry += f"\nTraceback:\n{traceback.format_exc()}"
    try:
        print(log_entry)
    except UnicodeEncodeError:
        safe_message = message.encode('ascii', 'ignore').decode('ascii')
        print(f"[{timestamp}] [ERROR] {safe_message}")
        if exception:
            print(f"Exception: {str(exception)}")
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')
    except Exception:
        pass

# ============================================================================
# HELPER FUNCTIONS (from train_iterative_feature_selection_standalone.py)
# ============================================================================

def clean_categorical_value(value):
    """Clean categorical values to remove special characters that cause issues in feature names"""
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
        '"': '_quote_', '\\': '_backslash_', '/': '_slash_',
        '\b': '_bs_', '\f': '_ff_', '\n': '_nl_', '\r': '_cr_', '\t': '_tab_',
        ' ': '_', "'": '_apostrophe_', ':': '_colon_', ';': '_semicolon_',
        ',': '_comma_', '&': '_amp_', '?': '_qmark_', '!': '_excl_',
        '*': '_star_', '+': '_plus_', '=': '_eq_', '@': '_at_',
        '#': '_hash_', '$': '_dollar_', '%': '_pct_', '^': '_caret_',
    }
    
    for old_char, new_char in replacements.items():
        name_str = name_str.replace(old_char, new_char)
    
    name_str = ''.join(char for char in name_str if ord(char) >= 32 or char in '\n\r\t')
    
    while '__' in name_str:
        name_str = name_str.replace('__', '_')
    
    name_str = name_str.strip('_')
    
    if not name_str:
        return 'Unknown'
    
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
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def filter_timelines_for_model(timelines_df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Filter timelines for a specific model, excluding other injury types from negatives.
    """
    if target_column not in ['target1', 'target2']:
        raise ValueError(f"Invalid target_column: {target_column}. Must be 'target1' or 'target2'")
    
    if 'target1' not in timelines_df.columns or 'target2' not in timelines_df.columns:
        raise ValueError("DataFrame must contain both 'target1' and 'target2' columns")
    
    if target_column == 'target1':
        mask = (timelines_df['target1'] == 1) | ((timelines_df['target1'] == 0) & (timelines_df['target2'] == 0))
        filtered_df = timelines_df[mask].copy()
    else:
        mask = (timelines_df['target2'] == 1) | ((timelines_df['target1'] == 0) & (timelines_df['target2'] == 0))
        filtered_df = timelines_df[mask].copy()
    
    return filtered_df

def extract_season_from_date(reference_date):
    """Extract season identifier (YYYY_YYYY) from reference_date"""
    if pd.isna(reference_date):
        return None
    
    try:
        date = pd.to_datetime(reference_date)
        year = date.year
        month = date.month
        
        # Season runs from July to June
        # July-December: season is YEAR_YEAR+1
        # January-June: season is YEAR-1_YEAR
        if month >= 7:
            season = f"{year}_{year+1}"
        else:
            season = f"{year-1}_{year}"
        
        return season
    except:
        return None

def load_combined_seasonal_datasets_natural(min_season=None, exclude_seasons=None, include_test=False):
    """
    Load and combine seasonal datasets with natural (unbalanced) target ratio.
    
    Args:
        min_season: Minimum season to include (e.g., '2018_2019'). If None, includes all seasons.
        exclude_seasons: List of seasons to exclude (e.g., ['2025_2026'] or ['2021_2022', '2022_2023'])
        include_test: If True, also load and include test dataset
    
    Returns:
        Combined DataFrame
    """
    if exclude_seasons is None:
        exclude_seasons = []
    
    pattern = str(TRAIN_DIR / 'timelines_35day_season_*_v4_muscular_train.csv')
    files = glob.glob(pattern)
    season_files = []
    
    for filepath in files:
        filename = os.path.basename(filepath)
        # Exclude files with ratio suffixes
        if '_10pc_' in filename or '_25pc_' in filename or '_50pc_' in filename:
            continue
        if 'season_' in filename:
            parts = filename.split('season_')
            if len(parts) > 1:
                season_part = parts[1].split('_v4_muscular_train')[0]
                if season_part not in exclude_seasons:
                    if min_season is None or season_part >= min_season:
                        season_files.append((season_part, filepath))
    
    # Sort chronologically
    season_files.sort(key=lambda x: x[0])
    
    log_message(f"\n📂 Loading {len(season_files)} season files with natural target ratio...")
    if min_season:
        log_message(f"   (Filtering: Only seasons >= {min_season})")
    if exclude_seasons:
        log_message(f"   (Excluding seasons: {exclude_seasons})")
    
    dfs = []
    total_records = 0
    total_target1 = 0
    total_target2 = 0
    
    for season_id, filepath in season_files:
        try:
            df = pd.read_csv(filepath, encoding='utf-8-sig', low_memory=False)
            if 'target1' not in df.columns or 'target2' not in df.columns:
                log_message(f"   ⚠️  {season_id}: Missing target1/target2 columns - skipping")
                continue
            
            target1_count = (df['target1'] == 1).sum()
            target2_count = (df['target2'] == 1).sum()
            
            if len(df) > 0:
                dfs.append(df)
                total_records += len(df)
                total_target1 += target1_count
                total_target2 += target2_count
                log_message(f"   ✅ {season_id}: {len(df):,} records (target1: {target1_count:,}, target2: {target2_count:,})")
        except Exception as e:
            log_message(f"   ⚠️  Error loading {season_id}: {e}")
            continue
    
    # Include test dataset if requested
    if include_test:
        try:
            test_file = TEST_DIR / 'timelines_35day_season_2025_2026_v4_muscular_test.csv'
            if test_file.exists():
                log_message(f"\n📂 Loading test dataset: {test_file.name}")
                df_test = pd.read_csv(test_file, encoding='utf-8-sig', low_memory=False)
                
                if 'target1' in df_test.columns and 'target2' in df_test.columns:
                    target1_count = (df_test['target1'] == 1).sum()
                    target2_count = (df_test['target2'] == 1).sum()
                    
                    dfs.append(df_test)
                    total_records += len(df_test)
                    total_target1 += target1_count
                    total_target2 += target2_count
                    log_message(f"   ✅ Test dataset: {len(df_test):,} records (target1: {target1_count:,}, target2: {target2_count:,})")
        except Exception as e:
            log_message(f"   ⚠️  Error loading test dataset: {e}")
    
    if not dfs:
        raise ValueError("No valid season files found!")
    
    # Combine all dataframes
    log_message(f"\n📊 Combining {len(dfs)} datasets...")
    combined_df = pd.concat(dfs, ignore_index=True)
    
    log_message(f"✅ Combined dataset: {len(combined_df):,} records")
    log_message(f"   Total target1=1 (Muscular): {total_target1:,} ({total_target1/len(combined_df)*100:.4f}%)")
    log_message(f"   Total target2=1 (Skeletal): {total_target2:,} ({total_target2/len(combined_df)*100:.4f}%)")
    
    return combined_df

def filter_by_seasons(df, exclude_seasons):
    """
    Filter dataframe to exclude specific seasons based on reference_date.
    
    Args:
        df: DataFrame with 'reference_date' column
        exclude_seasons: List of seasons to exclude (e.g., ['2021_2022', '2022_2023'])
    
    Returns:
        Filtered DataFrame
    """
    if not exclude_seasons:
        return df
    
    log_message(f"   Filtering out seasons: {exclude_seasons}")
    
    # Extract season from reference_date
    df['_season'] = df['reference_date'].apply(extract_season_from_date)
    
    initial_count = len(df)
    df_filtered = df[~df['_season'].isin(exclude_seasons)].copy()
    df_filtered = df_filtered.drop(columns=['_season'])
    
    excluded_count = initial_count - len(df_filtered)
    log_message(f"   Excluded {excluded_count:,} records ({excluded_count/initial_count*100:.2f}%)")
    log_message(f"   Remaining: {len(df_filtered):,} records")
    
    return df_filtered

def load_test_dataset():
    """Load test dataset (2025/26 season)"""
    test_file = TEST_DIR / 'timelines_35day_season_2025_2026_v4_muscular_test.csv'
    
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    log_message(f"\n📂 Loading test dataset: {test_file.name}")
    df_test = pd.read_csv(test_file, encoding='utf-8-sig', low_memory=False)
    
    if 'target1' not in df_test.columns or 'target2' not in df_test.columns:
        raise ValueError("Test dataset missing target1/target2 columns")
    
    target1_count = (df_test['target1'] == 1).sum()
    target2_count = (df_test['target2'] == 1).sum()
    
    log_message(f"✅ Test dataset: {len(df_test):,} records")
    log_message(f"   target1=1 (Muscular): {target1_count:,} ({target1_count/len(df_test)*100:.4f}%)")
    log_message(f"   target2=1 (Skeletal): {target2_count:,} ({target2_count/len(df_test)*100:.4f}%)")
    
    return df_test

def prepare_data(df, cache_file=None, use_cache=True):
    """Prepare data with basic preprocessing (no feature selection) and optional caching"""
    
    if use_cache and cache_file and os.path.exists(cache_file):
        log_message(f"   📦 Loading preprocessed data from cache: {cache_file}...")
        try:
            df_preprocessed = pd.read_csv(cache_file, encoding='utf-8-sig', low_memory=False)
            if len(df_preprocessed) != len(df):
                log_message(f"   ⚠️  Cache length ({len(df_preprocessed)}) doesn't match data length ({len(df)}), preprocessing fresh...")
                use_cache = False
            else:
                target_cols = ['target1', 'target2', 'target']
                cols_to_drop = [col for col in target_cols if col in df_preprocessed.columns]
                if cols_to_drop:
                    X = df_preprocessed.drop(columns=cols_to_drop)
                else:
                    X = df_preprocessed
                log_message(f"   ✅ Loaded preprocessed data from cache")
                return X
        except Exception as e:
            log_message(f"   ⚠️  Failed to load cache ({e}), preprocessing fresh...")
            use_cache = False
    
    feature_columns = [
        col for col in df.columns
        if col not in ['player_id', 'reference_date', 'date', 'player_name', 'target1', 'target2', 'target', 'has_minimum_activity']
    ]
    X = df[feature_columns].copy()
    
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    X_encoded = X.copy()
    
    if len(categorical_features) > 0:
        log_message(f"   Processing {len(categorical_features)} categorical features...")
        for feature in tqdm(categorical_features, desc="   Encoding categorical features", unit="feature", leave=False):
            X_encoded[feature] = X_encoded[feature].fillna('Unknown')
            
            problematic_mask = X_encoded[feature].astype(str).str.contains(r'[:,\'"\\/;|&?!*+=@#$%^\s]', regex=True, na=False)
            problematic_count = problematic_mask.sum()
            if problematic_count > 0:
                problematic_values = X_encoded[feature][problematic_mask].unique()[:10]
                log_message(f"\n⚠️  Found {problematic_count} problematic values in '{feature}': {list(problematic_values)}")
            
            X_encoded[feature] = X_encoded[feature].apply(clean_categorical_value)
            
            dummies = pd.get_dummies(X_encoded[feature], prefix=feature, drop_first=True)
            dummies.columns = [sanitize_feature_name(col) for col in dummies.columns]
            
            X_encoded = pd.concat([X_encoded, dummies], axis=1)
            X_encoded = X_encoded.drop(columns=[feature])
    
    if len(numeric_features) > 0:
        log_message(f"   Processing {len(numeric_features)} numeric features...")
        for feature in tqdm(numeric_features, desc="   Filling numeric missing values", unit="feature", leave=False):
            X_encoded[feature] = X_encoded[feature].fillna(0)
    
    X_encoded.columns = [sanitize_feature_name(col) for col in X_encoded.columns]
    
    if use_cache and cache_file:
        log_message(f"   💾 Caching preprocessed data to {cache_file}...")
        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            df_preprocessed = X_encoded.copy()
            df_preprocessed.to_csv(cache_file, index=False, encoding='utf-8-sig')
            log_message(f"   ✅ Preprocessed data cached")
        except Exception as e:
            log_message(f"   ⚠️  Failed to cache ({e})")
    
    return X_encoded

def align_features(X_train, X_test):
    """Ensure all datasets have the same features"""
    common_features = list(set(X_train.columns) & set(X_test.columns))
    common_features = sorted(common_features)
    
    log_message(f"   Aligning features: {len(common_features)} common features")
    log_message(f"   Training: {X_train.shape[1]} -> {len(common_features)}")
    log_message(f"   Test: {X_test.shape[1]} -> {len(common_features)}")
    
    return X_train[common_features], X_test[common_features]

def filter_to_feature_subset(X_train, X_test, feature_subset):
    """
    Filter datasets to only include specified features.
    
    Args:
        X_train: Training feature DataFrame
        X_test: Test feature DataFrame
        feature_subset: List of feature names to keep
        
    Returns:
        Filtered X_train and X_test DataFrames
    """
    available_features = list(set(X_train.columns) & set(X_test.columns))
    requested_features = [f for f in feature_subset if f in available_features]
    
    missing_features = [f for f in feature_subset if f not in available_features]
    if missing_features:
        log_message(f"   ⚠️  Warning: {len(missing_features)} requested features not found in datasets")
        if len(missing_features) <= 10:
            log_message(f"      Missing: {missing_features}")
    
    if len(requested_features) == 0:
        raise ValueError("No requested features found in datasets!")
    
    log_message(f"   Using {len(requested_features)}/{len(feature_subset)} requested features")
    
    return X_train[requested_features], X_test[requested_features]

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
    
    metrics = convert_numpy_types(metrics)
    
    log_message(f"\n   📊 {dataset_name} Results:")
    log_message(f"      Accuracy: {metrics['accuracy']:.4f}")
    log_message(f"      Precision: {metrics['precision']:.4f}")
    log_message(f"      Recall: {metrics['recall']:.4f}")
    log_message(f"      F1-Score: {metrics['f1']:.4f}")
    log_message(f"      ROC AUC: {metrics['roc_auc']:.4f}")
    log_message(f"      Gini: {metrics['gini']:.4f}")
    log_message(f"      TP: {metrics['confusion_matrix']['tp']}, FP: {metrics['confusion_matrix']['fp']}, "
              f"TN: {metrics['confusion_matrix']['tn']}, FN: {metrics['confusion_matrix']['fn']}")
    
    return metrics

# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def load_optimal_features():
    """Load the 620 optimal features from iteration 31"""
    log_message(f"Loading optimal features from: {ITERATIVE_RESULTS_FILE}")
    
    if not ITERATIVE_RESULTS_FILE.exists():
        raise FileNotFoundError(f"Iterative results file not found: {ITERATIVE_RESULTS_FILE}")
    
    try:
        with open(ITERATIVE_RESULTS_FILE, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # Find iteration 31
        iteration_31 = None
        for iteration in results['iterations']:
            if iteration['iteration'] == 31:
                iteration_31 = iteration
                break
        
        if iteration_31 is None:
            raise ValueError("Iteration 31 not found in results file")
        
        features = iteration_31['features']
        log_message(f"✅ Loaded {len(features)} optimal features from iteration 31")
        log_message(f"   Combined score: {iteration_31['combined_score']:.4f}")
        
        return features
        
    except Exception as e:
        log_error("Failed to load optimal features", e)
        raise

def load_v3_metrics():
    """Load V3 production metrics"""
    log_message(f"Loading V3 production metrics from: {V3_METRICS_FILE}")
    
    if not V3_METRICS_FILE.exists():
        log_message(f"⚠️  V3 metrics file not found: {V3_METRICS_FILE}")
        return None
    
    try:
        with open(V3_METRICS_FILE, 'r', encoding='utf-8') as f:
            v3_metrics = json.load(f)
        
        log_message("✅ Loaded V3 production metrics")
        return v3_metrics
        
    except Exception as e:
        log_error("Failed to load V3 metrics", e)
        return None

def train_and_evaluate_version(version_name, df_train_all, df_test_all, optimal_features, cache_suffix):
    """
    Train and evaluate a model version.
    
    Args:
        version_name: Name of the version (for logging)
        df_train_all: Training data (all timelines)
        df_test_all: Test data (for evaluation)
        optimal_features: List of 620 optimal features
        cache_suffix: Suffix for cache files (to avoid conflicts)
    
    Returns:
        Dictionary with train_metrics, test_metrics, and model
    """
    log_message(f"\n{'='*80}")
    log_message(f"TRAINING {version_name}")
    log_message(f"{'='*80}")
    
    # Filter for Model 1 (Muscular)
    df_train_muscular = filter_timelines_for_model(df_train_all, 'target1')
    df_test_muscular = filter_timelines_for_model(df_test_all, 'target1')
    
    # Prepare features
    cache_file_train = str(CACHE_DIR / f'preprocessed_muscular_train_{cache_suffix}.csv')
    cache_file_test = str(CACHE_DIR / f'preprocessed_muscular_test_{cache_suffix}.csv')
    
    df_train_muscular = df_train_muscular.reset_index(drop=True)
    df_test_muscular = df_test_muscular.reset_index(drop=True)
    
    log_message("   Preparing features...")
    X_train = prepare_data(df_train_muscular, cache_file=cache_file_train, use_cache=USE_CACHE)
    y_train = df_train_muscular['target1'].values
    X_test = prepare_data(df_test_muscular, cache_file=cache_file_test, use_cache=USE_CACHE)
    y_test = df_test_muscular['target1'].values
    
    # Align features
    X_train, X_test = align_features(X_train, X_test)
    
    # Filter to optimal features
    log_message(f"   Filtering to {len(optimal_features)} optimal features...")
    X_train, X_test = filter_to_feature_subset(X_train, X_test, optimal_features)
    
    log_message(f"   ✅ Using {len(X_train.columns)} features")
    log_message(f"   Training samples: {len(X_train):,}")
    log_message(f"   Test samples: {len(X_test):,}")
    
    # Train model
    log_message(f"\n🚀 Training Model 1 (Muscular Injuries)...")
    
    lgbm_model = LGBMClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    lgbm_model.fit(X_train, y_train)
    
    # Evaluate
    train_metrics = evaluate_model(lgbm_model, X_train, y_train, "Training")
    test_metrics = evaluate_model(lgbm_model, X_test, y_test, "Test")
    
    return {
        'model': lgbm_model,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'features_used': list(X_train.columns)
    }

def generate_comparison_table(v3_metrics, v4_results):
    """Generate a markdown comparison table"""
    
    table = """# V4 620-Feature Model - Fair Comparison with V3

## Model Configurations

| Model | Training Seasons | Test Dataset | Excluded Seasons |
|-------|------------------|--------------|------------------|
| **V3 Production** | 2018/19, 2019/20, 2020/21, 2024/25, 2025/26 | 2025/26 (in-sample) | 2021/22, 2022/23, 2023/24 |
| **V4 620 (Current)** | 2018/19-2024/25 | 2025/26 (out-of-sample) | 2025/26 only |
| **V4 620 (With Test)** | 2018/19-2025/26 | 2025/26 (in-sample) | None |
| **V4 620 (With Test, Excl 2021/22-2022/23)** | 2018/19, 2019/20, 2020/21, 2023/24, 2024/25, 2025/26 | 2025/26 (in-sample) | 2021/22, 2022/23 |

## Test Dataset Performance Comparison (Model 1 - Muscular Injuries)

| Metric | V3 Production | V4 620 (Current) | V4 620 (With Test) | V4 620 (With Test, Excl 2021/22-2022/23) |
|--------|---------------|------------------|-------------------|------------------------------------------|
"""
    
    metrics_list = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'gini']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC', 'Gini']
    
    for metric, metric_name in zip(metrics_list, metric_names):
        row = f"| **{metric_name}** |"
        
        # V3
        if v3_metrics:
            v3_val = v3_metrics.get(metric, 0)
            row += f" {v3_val:.4f} |"
        else:
            row += " N/A |"
        
        # V4 versions
        for version_key in ['v4_current', 'v4_with_test', 'v4_with_test_excl']:
            if version_key in v4_results:
                val = v4_results[version_key]['test_metrics'].get(metric, 0)
                row += f" {val:.4f} |"
            else:
                row += " N/A |"
        
        table += row + "\n"
    
    # Confusion Matrix
    table += "\n### Confusion Matrix\n\n"
    table += "| Metric | V3 Production | V4 620 (Current) | V4 620 (With Test) | V4 620 (With Test, Excl 2021/22-2022/23) |\n"
    table += "|--------|---------------|------------------|-------------------|------------------------------------------|\n"
    
    cm_metrics = ['tp', 'fp', 'tn', 'fn']
    cm_names = ['True Positives (TP)', 'False Positives (FP)', 'True Negatives (TN)', 'False Negatives (FN)']
    
    for cm_metric, cm_name in zip(cm_metrics, cm_names):
        row = f"| **{cm_name}** |"
        
        # V3
        if v3_metrics and 'confusion_matrix' in v3_metrics:
            v3_val = v3_metrics['confusion_matrix'].get(cm_metric.upper(), 0)
            row += f" {v3_val} |"
        else:
            row += " N/A |"
        
        # V4 versions
        for version_key in ['v4_current', 'v4_with_test', 'v4_with_test_excl']:
            if version_key in v4_results:
                val = v4_results[version_key]['test_metrics']['confusion_matrix'].get(cm_metric, 0)
                row += f" {val} |"
            else:
                row += " N/A |"
        
        table += row + "\n"
    
    return table

def main():
    """Main function to train and compare all models"""
    log_message("="*80)
    log_message("V4 620-FEATURE MODEL - FAIR COMPARISON WITH V3")
    log_message("="*80)
    
    start_time = datetime.now()
    log_message(f"\n⏰ Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load optimal features
    log_message("\n" + "="*80)
    log_message("STEP 1: LOADING OPTIMAL FEATURES")
    log_message("="*80)
    optimal_features = load_optimal_features()
    
    # Load V3 metrics
    log_message("\n" + "="*80)
    log_message("STEP 2: LOADING V3 PRODUCTION METRICS")
    log_message("="*80)
    v3_metrics = load_v3_metrics()
    
    # Load test dataset (for evaluation)
    log_message("\n" + "="*80)
    log_message("STEP 3: LOADING TEST DATASET")
    log_message("="*80)
    df_test_all = load_test_dataset()
    
    # Store results
    v4_results = {}
    
    # ========================================================================
    # VERSION A: V4 620 (Current - without test dataset)
    # ========================================================================
    log_message("\n" + "="*80)
    log_message("STEP 4: TRAINING VERSION A - V4 620 (CURRENT)")
    log_message("="*80)
    log_message("Configuration: Training 2018/19-2024/25, excluding test dataset")
    
    try:
        df_train_all = load_combined_seasonal_datasets_natural(
            min_season=MIN_SEASON,
            exclude_seasons=['2025_2026'],
            include_test=False
        )
        
        result_a = train_and_evaluate_version(
            "V4 620 (Current)",
            df_train_all,
            df_test_all,
            optimal_features,
            cache_suffix='v4_620_current'
        )
        v4_results['v4_current'] = result_a
        
    except Exception as e:
        log_error("Failed to train Version A", e)
        v4_results['v4_current'] = None
    
    # ========================================================================
    # VERSION B: V4 620 (With test dataset)
    # ========================================================================
    log_message("\n" + "="*80)
    log_message("STEP 5: TRAINING VERSION B - V4 620 (WITH TEST)")
    log_message("="*80)
    log_message("Configuration: Training 2018/19-2025/26, including test dataset")
    
    try:
        df_train_all = load_combined_seasonal_datasets_natural(
            min_season=MIN_SEASON,
            exclude_seasons=[],
            include_test=True
        )
        
        result_b = train_and_evaluate_version(
            "V4 620 (With Test)",
            df_train_all,
            df_test_all,
            optimal_features,
            cache_suffix='v4_620_with_test'
        )
        v4_results['v4_with_test'] = result_b
        
    except Exception as e:
        log_error("Failed to train Version B", e)
        v4_results['v4_with_test'] = None
    
    # ========================================================================
    # VERSION C: V4 620 (With test dataset, excluding 2021/22 and 2022/23)
    # ========================================================================
    log_message("\n" + "="*80)
    log_message("STEP 6: TRAINING VERSION C - V4 620 (WITH TEST, EXCL 2021/22-2022/23)")
    log_message("="*80)
    log_message("Configuration: Training 2018/19-2025/26, excluding 2021/22 and 2022/23 seasons")
    
    try:
        # Load all data including test
        df_train_all = load_combined_seasonal_datasets_natural(
            min_season=MIN_SEASON,
            exclude_seasons=[],
            include_test=True
        )
        
        # Filter out 2021/22 and 2022/23 seasons
        df_train_all = filter_by_seasons(df_train_all, ['2021_2022', '2022_2023'])
        
        result_c = train_and_evaluate_version(
            "V4 620 (With Test, Excl 2021/22-2022/23)",
            df_train_all,
            df_test_all,
            optimal_features,
            cache_suffix='v4_620_with_test_excl_2021_2022'
        )
        v4_results['v4_with_test_excl'] = result_c
        
    except Exception as e:
        log_error("Failed to train Version C", e)
        v4_results['v4_with_test_excl'] = None
    
    # ========================================================================
    # GENERATE COMPARISON
    # ========================================================================
    log_message("\n" + "="*80)
    log_message("STEP 7: GENERATING COMPARISON")
    log_message("="*80)
    
    # Prepare results for JSON
    comparison_results = {
        'v3_production': v3_metrics,
        'v4_620_current': {
            'train_metrics': v4_results['v4_current']['train_metrics'] if v4_results.get('v4_current') else None,
            'test_metrics': v4_results['v4_current']['test_metrics'] if v4_results.get('v4_current') else None,
        } if v4_results.get('v4_current') else None,
        'v4_620_with_test': {
            'train_metrics': v4_results['v4_with_test']['train_metrics'] if v4_results.get('v4_with_test') else None,
            'test_metrics': v4_results['v4_with_test']['test_metrics'] if v4_results.get('v4_with_test') else None,
        } if v4_results.get('v4_with_test') else None,
        'v4_620_with_test_excl': {
            'train_metrics': v4_results['v4_with_test_excl']['train_metrics'] if v4_results.get('v4_with_test_excl') else None,
            'test_metrics': v4_results['v4_with_test_excl']['test_metrics'] if v4_results.get('v4_with_test_excl') else None,
        } if v4_results.get('v4_with_test_excl') else None,
        'optimal_features': optimal_features,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save JSON results
    try:
        with open(OUTPUT_METRICS_FILE, 'w', encoding='utf-8') as f:
            json.dump(comparison_results, f, indent=2)
        log_message(f"✅ Saved metrics to: {OUTPUT_METRICS_FILE}")
    except Exception as e:
        log_error("Failed to save metrics", e)
    
    # Generate comparison table
    try:
        table = generate_comparison_table(v3_metrics, v4_results)
        
        with open(OUTPUT_TABLE_FILE, 'w', encoding='utf-8') as f:
            f.write(table)
        log_message(f"✅ Saved comparison table to: {OUTPUT_TABLE_FILE}")
        
        # Also print the table
        log_message("\n" + "="*80)
        log_message("COMPARISON TABLE")
        log_message("="*80)
        print("\n" + table)
        
    except Exception as e:
        log_error("Failed to generate comparison table", e)
    
    # Print summary
    log_message("\n" + "="*80)
    log_message("SUMMARY")
    log_message("="*80)
    
    if v3_metrics:
        log_message(f"\nV3 Production - Test Gini: {v3_metrics.get('gini', 0):.4f}, F1: {v3_metrics.get('f1', 0):.4f}")
    
    if v4_results.get('v4_current'):
        log_message(f"V4 620 (Current) - Test Gini: {v4_results['v4_current']['test_metrics']['gini']:.4f}, "
                   f"F1: {v4_results['v4_current']['test_metrics']['f1']:.4f}")
    
    if v4_results.get('v4_with_test'):
        log_message(f"V4 620 (With Test) - Test Gini: {v4_results['v4_with_test']['test_metrics']['gini']:.4f}, "
                   f"F1: {v4_results['v4_with_test']['test_metrics']['f1']:.4f}")
    
    if v4_results.get('v4_with_test_excl'):
        log_message(f"V4 620 (With Test, Excl 2021/22-2022/23) - Test Gini: {v4_results['v4_with_test_excl']['test_metrics']['gini']:.4f}, "
                   f"F1: {v4_results['v4_with_test_excl']['test_metrics']['f1']:.4f}")
    
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds() / 60
    log_message(f"\n⏰ Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(f"⏱️  Total time: {total_time:.1f} minutes")
    log_message(f"Log file saved to: {LOG_FILE}")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        log_message("\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        log_error("Fatal error in main execution", e)
        sys.exit(1)
