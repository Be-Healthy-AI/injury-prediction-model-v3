#!/usr/bin/env python3
"""
Iterative Feature Selection Training for Model 1 (Muscular Injuries Only) - Standalone

This script completely avoids importlib by including all necessary functions directly.
This version should work within Cursor's terminal environment.

This script:
1. Loads ranked features from feature_ranking.json
2. Trains Model 1 (Muscular) iteratively with increasing feature sets (20, 40, 60, ...)
3. Tracks performance metrics (Gini and F1-Score on test set) for Model 1 only
4. Stops when 3 consecutive drops in performance are detected
5. Identifies the optimal number of features for Model 1

Performance Metric: weighted combination of Gini coefficient and F1-Score for Model 1
    combined_score = gini_weight * gini + f1_weight * f1_score
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
TIMELINES_DIR = SCRIPT_DIR.parent / 'timelines'
CACHE_DIR = ROOT_DIR / 'cache'

# ========== CONFIGURATION ==========
FEATURES_PER_ITERATION = 20
INITIAL_FEATURES = 20
CONSECUTIVE_DROPS_THRESHOLD = 3
PERFORMANCE_DROP_THRESHOLD = 0.001  # Minimum drop to count as a drop (0.1%)
GINI_WEIGHT = 0.6
F1_WEIGHT = 0.4
RANKING_FILE = MODEL_OUTPUT_DIR / 'feature_ranking.json'
RESULTS_FILE = MODEL_OUTPUT_DIR / 'iterative_feature_selection_results_muscular.json'
LOG_FILE = MODEL_OUTPUT_DIR / 'iterative_training_muscular.log'

# Training configuration
MIN_SEASON = '2018_2019'  # Start from 2018/19 season (inclusive)
EXCLUDE_SEASON = '2025_2026'  # Test dataset season
TRAIN_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'data' / 'timelines' / 'train'
TEST_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'data' / 'timelines' / 'test'
USE_CACHE = True
# ===================================

# Initialize log file
try:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(f"=== Iterative Training Log (Model 1 - Muscular) Started at {datetime.now().isoformat()} ===\n")
except Exception as e:
    print(f"Warning: Could not initialize log file: {e}")

def log_message(message, level="INFO"):
    """Log a message to both console and log file"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] [{level}] {message}"
    try:
        print(log_entry)
    except UnicodeEncodeError:
        # Fallback: remove emojis and special characters for console
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
        # Fallback: remove emojis and special characters for console
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
# HELPER FUNCTIONS (from train_lgbm_v4_dual_targets_natural.py)
# ============================================================================

def clean_categorical_value(value):
    """Clean categorical values to remove special characters that cause issues in feature names"""
    if pd.isna(value) or value is None:
        return 'Unknown'
    
    value_str = str(value).strip()
    
    # Handle data quality issues - common problematic values
    problematic_values = ['tel:', 'tel', 'phone', 'n/a', 'na', 'null', 'none', '', 'nan', 'address:', 'website:']
    if value_str.lower() in problematic_values:
        return 'Unknown'
    
    # Replace special characters that cause issues in column names
    replacements = {
        ':': '_', "'": '_', ',': '_', '"': '_', ';': '_', '/': '_', '\\': '_',
        '{': '_', '}': '_', '[': '_', ']': '_', '(': '_', ')': '_', '|': '_',
        '&': '_', '?': '_', '!': '_', '*': '_', '+': '_', '=': '_', '@': '_',
        '#': '_', '$': '_', '%': '_', '^': '_', ' ': '_',
    }
    
    for old_char, new_char in replacements.items():
        value_str = value_str.replace(old_char, new_char)
    
    # Remove any remaining control characters
    value_str = ''.join(char for char in value_str if ord(char) >= 32 or char in '\n\r\t')
    
    # Remove multiple consecutive underscores
    while '__' in value_str:
        value_str = value_str.replace('__', '_')
    
    # Remove leading/trailing underscores
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
    
    # Remove control characters
    name_str = ''.join(char for char in name_str if ord(char) >= 32 or char in '\n\r\t')
    
    # Remove multiple consecutive underscores
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

# ============================================================================
# TIMELINE FILTER FUNCTION (from create_35day_timelines_v4_enhanced.py)
# ============================================================================

def filter_timelines_for_model(timelines_df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Filter timelines for Model 1 (muscular injuries), excluding skeletal injuries from negatives.
    
    This ensures Model 1 only learns to distinguish muscular injuries from non-injuries,
    not from skeletal injuries.
    
    Only includes timelines where target1=1 (positives) 
    or (target1=0 AND target2=0) (negatives - only non-injuries)
    
    Args:
        timelines_df: DataFrame with target1 and target2 columns
        target_column: Must be 'target1' for this Model 1-only script
        
    Returns:
        Filtered DataFrame with only relevant timelines for Model 1
    """
    if target_column != 'target1':
        raise ValueError(f"This script is for Model 1 only. target_column must be 'target1', got: {target_column}")
    
    # Validate required columns exist
    if 'target1' not in timelines_df.columns or 'target2' not in timelines_df.columns:
        raise ValueError("DataFrame must contain both 'target1' and 'target2' columns")
    
    # Model 1 (muscular): Include muscular injuries (target1=1) and non-injuries (both=0)
    # Exclude skeletal injuries (target2=1, target1=0)
    mask = (timelines_df['target1'] == 1) | ((timelines_df['target1'] == 0) & (timelines_df['target2'] == 0))
    filtered_df = timelines_df[mask].copy()
    
    excluded_count = ((timelines_df['target1'] == 0) & (timelines_df['target2'] == 1)).sum()
    positives = filtered_df['target1'].sum()
    negatives = ((filtered_df['target1'] == 0) & (filtered_df['target2'] == 0)).sum()
    
    log_message(f"\n📊 Filtered for Model 1 (Muscular Injuries):")
    log_message(f"   Original timelines: {len(timelines_df):,}")
    log_message(f"   After filtering: {len(filtered_df):,}")
    log_message(f"   Positives (target1=1): {positives:,}")
    log_message(f"   Negatives (target1=0, target2=0): {negatives:,}")
    log_message(f"   Excluded (skeletal injuries): {excluded_count:,}")
    if len(filtered_df) > 0:
        log_message(f"   Target ratio: {positives / len(filtered_df) * 100:.2f}%")
    
    return filtered_df

# ============================================================================
# DATA LOADING FUNCTIONS (from train_lgbm_v4_dual_targets_natural.py)
# ============================================================================

def load_combined_seasonal_datasets_natural(min_season=None, exclude_season='2025_2026'):
    """
    Load and combine all seasonal datasets with natural (unbalanced) target ratio.
    
    Args:
        min_season: Minimum season to include (e.g., '2018_2019'). If None, includes all seasons.
        exclude_season: Season to exclude (default: '2025_2026' for test)
    
    Returns:
        Combined DataFrame
    """
    pattern = str(TRAIN_DIR / 'timelines_35day_season_*_v4_muscular_train.csv')
    files = glob.glob(pattern)
    season_files = []
    
    for filepath in files:
        filename = os.path.basename(filepath)
        # Exclude files with ratio suffixes (10pc, 25pc, 50pc)
        if '_10pc_' in filename or '_25pc_' in filename or '_50pc_' in filename:
            continue
        if 'season_' in filename:
            parts = filename.split('season_')
            if len(parts) > 1:
                # Extract season from pattern: timelines_35day_season_YYYY_YYYY_v4_muscular_train.csv
                season_part = parts[1].split('_v4_muscular_train')[0]
                if season_part != exclude_season:
                    # Filter by minimum season if specified
                    if min_season is None or season_part >= min_season:
                        season_files.append((season_part, filepath))
    
    # Sort chronologically
    season_files.sort(key=lambda x: x[0])
    
    log_message(f"\n📂 Loading {len(season_files)} season files with natural target ratio...")
    if min_season:
        log_message(f"   (Filtering: Only seasons >= {min_season})")
    
    dfs = []
    total_records = 0
    total_target1 = 0
    total_target2 = 0
    
    for season_id, filepath in season_files:
        try:
            df = pd.read_csv(filepath, encoding='utf-8-sig', low_memory=False)
            # Check for target1 and target2 columns
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
    
    if not dfs:
        raise ValueError("No valid season files found!")
    
    # Combine all dataframes
    log_message(f"\n📊 Combining {len(dfs)} season datasets...")
    combined_df = pd.concat(dfs, ignore_index=True)
    
    log_message(f"✅ Combined dataset: {len(combined_df):,} records")
    log_message(f"   Total target1=1 (Muscular): {total_target1:,} ({total_target1/len(combined_df)*100:.4f}%)")
    log_message(f"   Total target2=1 (Skeletal): {total_target2:,} ({total_target2/len(combined_df)*100:.4f}%)")
    
    return combined_df

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

# ============================================================================
# DATA PREPARATION FUNCTIONS (from train_lgbm_v4_dual_targets_natural.py)
# ============================================================================

def prepare_data(df, cache_file=None, use_cache=True):
    """Prepare data with basic preprocessing (no feature selection) and optional caching"""
    
    # Check cache - but verify it matches the current dataframe length
    if use_cache and cache_file and os.path.exists(cache_file):
        log_message(f"   📦 Loading preprocessed data from cache: {cache_file}...")
        try:
            df_preprocessed = pd.read_csv(cache_file, encoding='utf-8-sig', low_memory=False)
            # Verify cache matches current dataframe length
            if len(df_preprocessed) != len(df):
                log_message(f"   ⚠️  Cache length ({len(df_preprocessed)}) doesn't match data length ({len(df)}), preprocessing fresh...")
                use_cache = False
            else:
                # Drop target columns if they exist (we'll add them back)
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
    
    # Drop non-feature columns
    feature_columns = [
        col for col in df.columns
        if col not in ['player_id', 'reference_date', 'date', 'player_name', 'target1', 'target2', 'target', 'has_minimum_activity']
    ]
    X = df[feature_columns].copy()
    
    # Handle missing values and encode categorical features
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    X_encoded = X.copy()
    
    # Encode categorical features
    if len(categorical_features) > 0:
        log_message(f"   Processing {len(categorical_features)} categorical features...")
        for feature in tqdm(categorical_features, desc="   Encoding categorical features", unit="feature", leave=False):
            # Fill NaN first
            X_encoded[feature] = X_encoded[feature].fillna('Unknown')
            
            # Check for problematic values before cleaning
            problematic_mask = X_encoded[feature].astype(str).str.contains(r'[:,\'"\\/;|&?!*+=@#$%^\s]', regex=True, na=False)
            problematic_count = problematic_mask.sum()
            if problematic_count > 0:
                problematic_values = X_encoded[feature][problematic_mask].unique()[:10]
                log_message(f"\n⚠️  Found {problematic_count} problematic values in '{feature}': {list(problematic_values)}")
            
            # Clean categorical values BEFORE one-hot encoding
            X_encoded[feature] = X_encoded[feature].apply(clean_categorical_value)
            
            # Now one-hot encode
            dummies = pd.get_dummies(X_encoded[feature], prefix=feature, drop_first=True)
            
            # Sanitize dummy column names
            dummies.columns = [sanitize_feature_name(col) for col in dummies.columns]
            
            X_encoded = pd.concat([X_encoded, dummies], axis=1)
            X_encoded = X_encoded.drop(columns=[feature])
    
    # Fill numeric missing values
    if len(numeric_features) > 0:
        log_message(f"   Processing {len(numeric_features)} numeric features...")
        for feature in tqdm(numeric_features, desc="   Filling numeric missing values", unit="feature", leave=False):
            X_encoded[feature] = X_encoded[feature].fillna(0)
    
    # Sanitize all column names
    X_encoded.columns = [sanitize_feature_name(col) for col in X_encoded.columns]
    
    # Cache preprocessed data
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
    # Get common features across both datasets
    common_features = list(set(X_train.columns) & set(X_test.columns))
    
    # Sort for consistency
    common_features = sorted(common_features)
    
    log_message(f"   Aligning features: {len(common_features)} common features")
    log_message(f"   Training: {X_train.shape[1]} -> {len(common_features)}")
    log_message(f"   Test: {X_test.shape[1]} -> {len(common_features)}")
    
    return X_train[common_features], X_test[common_features]

# ============================================================================
# MODEL EVALUATION FUNCTION (from train_lgbm_v4_dual_targets_natural.py)
# ============================================================================

def evaluate_model(model, X, y, dataset_name):
    """Evaluate model and return metrics"""
    # For large datasets, show progress during prediction
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
    
    # Convert all numpy types to Python native types
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
# FEATURE FILTERING AND TRAINING FUNCTIONS
# ============================================================================

def filter_features(X_train, X_test, feature_subset):
    """
    Filter datasets to only include specified features.
    
    Args:
        X_train: Training feature DataFrame
        X_test: Test feature DataFrame
        feature_subset: List of feature names to keep
        
    Returns:
        Filtered X_train and X_test DataFrames
    """
    # Get features that exist in both datasets
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

def train_model_with_feature_subset(feature_subset, verbose=True):
    """
    Train Model 1 (Muscular) using a specific feature subset.
    
    Args:
        feature_subset: List of feature names to use for training
        verbose: Whether to print progress messages
        
    Returns:
        Dictionary containing:
        - model: Trained muscular model
        - train_metrics: Training metrics for Model 1
        - test_metrics: Test metrics for Model 1
        - feature_names_used: List of features actually used
    """
    if verbose:
        log_message(f"\n{'='*80}")
        log_message(f"TRAINING MODEL 1 (MUSCULAR) WITH {len(feature_subset)} FEATURES")
        log_message(f"{'='*80}")
    
    # Load and prepare data (reuse cached preprocessed data if available)
    if verbose:
        log_message("\n📂 Loading datasets...")
    
    try:
        df_train_all = load_combined_seasonal_datasets_natural(
            min_season=MIN_SEASON,
            exclude_season=EXCLUDE_SEASON
        )
        df_test_all = load_test_dataset()
    except Exception as e:
        log_error("Failed to load datasets", e)
        raise
    
    # Filter for Model 1 only
    try:
        df_train_muscular = filter_timelines_for_model(df_train_all, 'target1')
        df_test_muscular = filter_timelines_for_model(df_test_all, 'target1')
    except Exception as e:
        log_error("Failed to filter timelines", e)
        raise
    
    # Prepare features (use cache with unique names to avoid conflicts)
    cache_suffix = hashlib.md5(str(sorted(feature_subset)).encode()).hexdigest()[:8]
    
    cache_file_muscular_train = str(CACHE_DIR / f'preprocessed_muscular_train_subset_{cache_suffix}.csv')
    cache_file_muscular_test = str(CACHE_DIR / f'preprocessed_muscular_test_subset_{cache_suffix}.csv')
    
    df_train_muscular = df_train_muscular.reset_index(drop=True)
    df_test_muscular = df_test_muscular.reset_index(drop=True)
    
    if verbose:
        log_message("   Preparing features for Model 1 (Muscular)...")
    try:
        X_train_muscular = prepare_data(df_train_muscular, cache_file=cache_file_muscular_train, use_cache=USE_CACHE)
        y_train_muscular = df_train_muscular['target1'].values
        X_test_muscular = prepare_data(df_test_muscular, cache_file=cache_file_muscular_test, use_cache=USE_CACHE)
        y_test_muscular = df_test_muscular['target1'].values
    except Exception as e:
        log_error("Failed to prepare muscular features", e)
        raise
    
    # Align features
    try:
        X_train_muscular, X_test_muscular = align_features(X_train_muscular, X_test_muscular)
    except Exception as e:
        log_error("Failed to align features", e)
        raise
    
    # Filter to feature subset
    if verbose:
        log_message(f"\n   Filtering to {len(feature_subset)} requested features...")
    try:
        X_train_muscular, X_test_muscular = filter_features(X_train_muscular, X_test_muscular, feature_subset)
    except Exception as e:
        log_error("Failed to filter features", e)
        raise
    
    # Get actual features used
    features_used = sorted(list(X_train_muscular.columns))
    
    if verbose:
        log_message(f"   ✅ Using {len(features_used)} features")
    
    # Train Model 1 (Muscular)
    if verbose:
        log_message(f"\n🚀 Training Model 1 (Muscular Injuries)...")
    
    try:
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
        
        lgbm_model.fit(X_train_muscular, y_train_muscular)
        train_metrics = evaluate_model(lgbm_model, X_train_muscular, y_train_muscular, "Training")
        test_metrics = evaluate_model(lgbm_model, X_test_muscular, y_test_muscular, "Test")
    except Exception as e:
        log_error("Failed to train Model 1 (Muscular)", e)
        raise
    
    # Convert numpy types for JSON serialization
    train_metrics = convert_numpy_types(train_metrics)
    test_metrics = convert_numpy_types(test_metrics)
    
    return {
        'model': lgbm_model,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'feature_names_used': features_used
    }

# ============================================================================
# ITERATIVE TRAINING FUNCTIONS
# ============================================================================

def load_feature_ranking():
    """Load ranked features from JSON file"""
    log_message(f"Loading feature ranking from: {RANKING_FILE}")
    
    if not RANKING_FILE.exists():
        error_msg = f"Feature ranking file not found: {RANKING_FILE}\nPlease run rank_features_by_importance.py first."
        log_error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        log_message(f"Opening ranking file: {RANKING_FILE}")
        with open(RANKING_FILE, 'r', encoding='utf-8') as f:
            ranking_data = json.load(f)
        
        if 'ranked_features' not in ranking_data:
            raise KeyError("'ranked_features' key not found in ranking data")
        
        ranked_features = ranking_data['ranked_features']
        log_message(f"Successfully loaded {len(ranked_features)} ranked features")
        
        if len(ranked_features) == 0:
            log_error("Ranked features list is empty!")
            raise ValueError("No features found in ranking file")
        
        return ranked_features
        
    except json.JSONDecodeError as e:
        log_error(f"Failed to parse JSON from ranking file: {RANKING_FILE}", e)
        raise
    except Exception as e:
        log_error(f"Error loading feature ranking", e)
        raise

def calculate_combined_score(test_metrics, gini_weight=GINI_WEIGHT, f1_weight=F1_WEIGHT):
    """
    Calculate weighted combination of Gini and F1-Score for Model 1 (Muscular).
    
    Args:
        test_metrics: Test metrics for Model 1 (muscular)
        gini_weight: Weight for Gini coefficient (default 0.6)
        f1_weight: Weight for F1-Score (default 0.4)
    
    Returns:
        Combined performance score for Model 1
    """
    try:
        combined_score = (
            gini_weight * test_metrics['gini'] + 
            f1_weight * test_metrics['f1']
        )
        return combined_score
    except KeyError as e:
        log_error(f"Missing metric in test_metrics: {e}")
        raise
    except Exception as e:
        log_error(f"Error calculating combined score", e)
        raise

def has_consecutive_drops(scores, threshold=PERFORMANCE_DROP_THRESHOLD, 
                          consecutive_count=CONSECUTIVE_DROPS_THRESHOLD):
    """
    Check if there are N consecutive drops in performance.
    
    Args:
        scores: List of performance scores (most recent last)
        threshold: Minimum drop to consider significant
        consecutive_count: Number of consecutive drops required
    
    Returns:
        True if N consecutive drops detected, False otherwise
    """
    if len(scores) < consecutive_count + 1:
        return False
    
    # Check last N+1 scores for N consecutive drops
    drops = 0
    for i in range(len(scores) - consecutive_count, len(scores) - 1):
        if scores[i+1] < scores[i] - threshold:
            drops += 1
            if drops >= consecutive_count:
                return True
        else:
            drops = 0
    
    return False

def run_iterative_training():
    """Main function to run iterative feature selection training for Model 1"""
    log_message("="*80)
    log_message("ITERATIVE FEATURE SELECTION TRAINING - MODEL 1 (MUSCULAR) ONLY")
    log_message("="*80)
    log_message(f"\n📋 Configuration:")
    log_message(f"   Features per iteration: {FEATURES_PER_ITERATION}")
    log_message(f"   Initial features: {INITIAL_FEATURES}")
    log_message(f"   Stop after: {CONSECUTIVE_DROPS_THRESHOLD} consecutive drops")
    log_message(f"   Performance metric: {GINI_WEIGHT} * Gini + {F1_WEIGHT} * F1-Score (Model 1 only)")
    log_message(f"   Drop threshold: {PERFORMANCE_DROP_THRESHOLD}")
    log_message("="*80)
    
    start_time = datetime.now()
    log_message(f"\n⏰ Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load ranked features
    log_message("\n" + "="*80)
    log_message("STEP 1: LOADING FEATURE RANKING")
    log_message("="*80)
    
    try:
        ranked_features = load_feature_ranking()
    except Exception as e:
        log_error("Failed to load feature ranking", e)
        return 1
    
    # Initialize results storage
    log_message("Initializing results storage")
    results = {
        'iterations': [],
        'configuration': {
            'features_per_iteration': FEATURES_PER_ITERATION,
            'initial_features': INITIAL_FEATURES,
            'consecutive_drops_threshold': CONSECUTIVE_DROPS_THRESHOLD,
            'performance_drop_threshold': PERFORMANCE_DROP_THRESHOLD,
            'gini_weight': GINI_WEIGHT,
            'f1_weight': F1_WEIGHT,
            'model': 'Model 1 (Muscular) Only',
            'start_time': start_time.isoformat()
        },
        'best_iteration': None,
        'best_n_features': None,
        'best_combined_score': None
    }
    
    # Track performance scores for drop detection
    combined_scores = []
    
    # Iterative training
    log_message("\n" + "="*80)
    log_message("STEP 2: ITERATIVE TRAINING")
    log_message("="*80)
    
    iteration = 0
    best_score = -np.inf
    best_iteration = None
    best_n_features = None
    
    # Calculate number of iterations needed
    max_iterations = (len(ranked_features) - INITIAL_FEATURES) // FEATURES_PER_ITERATION + 1
    if (len(ranked_features) - INITIAL_FEATURES) % FEATURES_PER_ITERATION != 0:
        max_iterations += 1
    
    log_message(f"\n   Will train up to {max_iterations} iterations")
    log_message(f"   (from {INITIAL_FEATURES} to {len(ranked_features)} features)")
    
    # Main iteration loop
    while True:
        iteration += 1
        n_features = INITIAL_FEATURES + (iteration - 1) * FEATURES_PER_ITERATION
        
        # Check if we've used all features
        if n_features > len(ranked_features):
            log_message(f"\n✅ Reached maximum number of features ({len(ranked_features)})")
            break
        
        # Select feature subset
        feature_subset = ranked_features[:n_features]
        
        log_message(f"\n{'='*80}")
        log_message(f"ITERATION {iteration}: Training Model 1 with {n_features} features")
        log_message(f"{'='*80}")
        log_message(f"   Features: Top {n_features} from ranked list")
        
        iteration_start = datetime.now()
        
        try:
            log_message(f"Starting training for iteration {iteration} with {n_features} features")
            # Train model with this feature subset
            training_results = train_model_with_feature_subset(
                feature_subset, 
                verbose=True
            )
            log_message(f"Training completed for iteration {iteration}")
            
            # Validate training results
            required_keys = ['test_metrics', 'feature_names_used', 'train_metrics']
            for key in required_keys:
                if key not in training_results:
                    raise KeyError(f"Missing key in training results: {key}")
            
            log_message("Calculating combined performance score")
            # Calculate combined performance score (Model 1 only)
            combined_score = calculate_combined_score(
                training_results['test_metrics'],
                gini_weight=GINI_WEIGHT,
                f1_weight=F1_WEIGHT
            )
            
            combined_scores.append(combined_score)
            log_message(f"Combined score for iteration {iteration}: {combined_score:.4f}")
            
            # Check if this is the best so far
            if combined_score > best_score:
                best_score = combined_score
                best_iteration = iteration
                best_n_features = n_features
                log_message(f"New best score! Iteration {iteration} with {n_features} features: {best_score:.4f}")
            
            # Store iteration results
            iteration_data = {
                'iteration': iteration,
                'n_features': n_features,
                'features': training_results['feature_names_used'],
                'model1_muscular': {
                    'train': training_results['train_metrics'],
                    'test': training_results['test_metrics']
                },
                'combined_score': float(combined_score),
                'timestamp': iteration_start.isoformat(),
                'training_time_seconds': (datetime.now() - iteration_start).total_seconds()
            }
            
            results['iterations'].append(iteration_data)
            
            # Print iteration summary
            log_message(f"\n📊 Iteration {iteration} Results:")
            log_message(f"   Features: {n_features}")
            log_message(f"   Model 1 (Muscular) - Test: Gini={training_results['test_metrics']['gini']:.4f}, "
                      f"F1={training_results['test_metrics']['f1']:.4f}")
            log_message(f"   Combined Score: {combined_score:.4f}")
            log_message(f"   Training time: {iteration_data['training_time_seconds']:.1f} seconds")
            
            # Check for consecutive drops
            if has_consecutive_drops(combined_scores, 
                                    threshold=PERFORMANCE_DROP_THRESHOLD,
                                    consecutive_count=CONSECUTIVE_DROPS_THRESHOLD):
                log_message(f"\n⚠️  Detected {CONSECUTIVE_DROPS_THRESHOLD} consecutive drops in performance!")
                log_message(f"   Stopping iterative training.")
                break
            
            # Save intermediate results (after each iteration)
            log_message(f"Saving intermediate results to: {RESULTS_FILE}")
            try:
                with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2)
                log_message("Intermediate results saved successfully")
            except Exception as e:
                log_error(f"Failed to save intermediate results", e)
            
        except Exception as e:
            log_error(f"Error in iteration {iteration}", e)
            # Continue to next iteration
            continue
    
    # Finalize results
    log_message("Finalizing results")
    results['best_iteration'] = best_iteration
    results['best_n_features'] = best_n_features
    results['best_combined_score'] = float(best_score) if best_score != -np.inf else None
    results['configuration']['end_time'] = datetime.now().isoformat()
    results['configuration']['total_iterations'] = iteration
    results['configuration']['total_time_minutes'] = (datetime.now() - start_time).total_seconds() / 60
    
    # Save final results
    log_message("\n" + "="*80)
    log_message("STEP 3: SAVING RESULTS")
    log_message("="*80)
    
    try:
        with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        log_message(f"✅ Saved results to: {RESULTS_FILE}")
    except Exception as e:
        log_error(f"Failed to save final results", e)
        return 1
    
    # Print final summary
    log_message("\n" + "="*80)
    log_message("FINAL SUMMARY")
    log_message("="*80)
    
    log_message(f"\n📊 Training Summary:")
    log_message(f"   Total iterations: {iteration}")
    log_message(f"   Best iteration: {best_iteration}")
    log_message(f"   Best number of features: {best_n_features}")
    log_message(f"   Best combined score: {best_score:.4f}")
    
    if best_iteration:
        best_iter_data = results['iterations'][best_iteration - 1]
        log_message(f"\n📈 Best Performance (Iteration {best_iteration}):")
        log_message(f"   Model 1 (Muscular) - Test:")
        log_message(f"      Gini: {best_iter_data['model1_muscular']['test']['gini']:.4f}")
        log_message(f"      F1-Score: {best_iter_data['model1_muscular']['test']['f1']:.4f}")
        log_message(f"      ROC AUC: {best_iter_data['model1_muscular']['test']['roc_auc']:.4f}")
        log_message(f"   Combined Score: {best_iter_data['combined_score']:.4f}")
    
    # Plot performance progression (if matplotlib available)
    try:
        import matplotlib.pyplot as plt
        
        if len(combined_scores) > 1:
            iterations = [it['iteration'] for it in results['iterations']]
            scores = [it['combined_score'] for it in results['iterations']]
            
            plt.figure(figsize=(12, 6))
            plt.plot(iterations, scores, marker='o', linewidth=2, markersize=8)
            if best_iteration:
                plt.axvline(x=best_iteration, color='r', linestyle='--', 
                           label=f'Best: Iteration {best_iteration}')
            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel('Combined Score (0.6*Gini + 0.4*F1)', fontsize=12)
            plt.title('Model 1 (Muscular) Performance vs Number of Features', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            plot_file = MODEL_OUTPUT_DIR / 'iterative_feature_selection_muscular_plot.png'
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            log_message(f"\n✅ Saved performance plot to: {plot_file}")
            plt.close()
    except ImportError:
        log_message("\n   (Skipping plot - matplotlib not available)")
    except Exception as e:
        log_error("Error creating plot", e)
    
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds() / 60
    log_message(f"\n⏰ Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(f"⏱️  Total time: {total_time:.1f} minutes")
    log_message(f"Log file saved to: {LOG_FILE}")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = run_iterative_training()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        log_message("\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        log_error("Fatal error in main execution", e)
        sys.exit(1)
