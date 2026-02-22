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
import argparse
import traceback
import glob
import hashlib
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from pathlib import Path
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
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

# Exclude granular club/country features (too granular / uninformative for Premier League–focused use)
EXCLUDED_RAW_FEATURES = ('current_club', 'current_club_country', 'previous_club', 'previous_club_country')
EXCLUDED_FEATURE_PREFIXES = ('current_club_country_', 'current_club_', 'previous_club_country_', 'previous_club_')

# Training configuration
MIN_SEASON = '2018_2019'  # Start from 2018/19 season (inclusive)
EXCLUDE_SEASON = '2025_2026'  # Test dataset season
TRAIN_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'data' / 'timelines' / 'train'
TEST_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'data' / 'timelines' / 'test'
USE_CACHE = True
# Train/validation split (80% train, 20% validation by randomly picked timelines/rows)
TRAIN_VAL_RATIO = 0.8
SPLIT_RANDOM_STATE = 42
# Optimize-on: 'validation' or 'test' (set from CLI; used to pick best iteration)
OPTIMIZE_ON_DEFAULT = 'validation'

# Hyperparameter presets for sensitivity testing: standard (current), below (more regularized), above (less regularized)
LGBM_HP_STANDARD = {
    'n_estimators': 200, 'max_depth': 10, 'learning_rate': 0.1,
    'min_child_samples': 20, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'reg_alpha': 0.1, 'reg_lambda': 1.0,
}
LGBM_HP_BELOW = {
    'n_estimators': 120, 'max_depth': 6, 'learning_rate': 0.05,
    'min_child_samples': 40, 'subsample': 0.6, 'colsample_bytree': 0.6,
    'reg_alpha': 0.5, 'reg_lambda': 2.0,
}
# Between standard and below (more regularized than standard, less than below)
LGBM_HP_BELOW_MID = {
    'n_estimators': 160, 'max_depth': 8, 'learning_rate': 0.075,
    'min_child_samples': 30, 'subsample': 0.7, 'colsample_bytree': 0.7,
    'reg_alpha': 0.25, 'reg_lambda': 1.5,
}
# Below the below (more regularized than below)
LGBM_HP_BELOW_STRONG = {
    'n_estimators': 80, 'max_depth': 4, 'learning_rate': 0.03,
    'min_child_samples': 60, 'subsample': 0.5, 'colsample_bytree': 0.5,
    'reg_alpha': 1.0, 'reg_lambda': 3.0,
}
LGBM_HP_ABOVE = {
    'n_estimators': 320, 'max_depth': 14, 'learning_rate': 0.15,
    'min_child_samples': 10, 'subsample': 0.95, 'colsample_bytree': 0.95,
    'reg_alpha': 0.02, 'reg_lambda': 0.5,
}
HP_PRESETS_LGBM = {
    'standard': LGBM_HP_STANDARD, 'below': LGBM_HP_BELOW,
    'below_mid': LGBM_HP_BELOW_MID, 'below_strong': LGBM_HP_BELOW_STRONG,
    'above': LGBM_HP_ABOVE,
}

GB_HP_STANDARD = {
    'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1, 'min_samples_leaf': 20,
}
GB_HP_BELOW = {
    'n_estimators': 60, 'max_depth': 3, 'learning_rate': 0.05, 'min_samples_leaf': 40,
    'subsample': 0.7, 'max_features': 0.7,
}
# Between standard and below (more regularized than standard, less than below)
GB_HP_BELOW_MID = {
    'n_estimators': 80, 'max_depth': 4, 'learning_rate': 0.075, 'min_samples_leaf': 30,
    'subsample': 0.75, 'max_features': 0.75,
}
# More regularized than below
GB_HP_BELOW_STRONG = {
    'n_estimators': 40, 'max_depth': 2, 'learning_rate': 0.03, 'min_samples_leaf': 60,
    'subsample': 0.6, 'max_features': 0.6,
}
GB_HP_ABOVE = {
    'n_estimators': 160, 'max_depth': 7, 'learning_rate': 0.15, 'min_samples_leaf': 10,
}
HP_PRESETS_GB = {
    'standard': GB_HP_STANDARD, 'below': GB_HP_BELOW,
    'below_mid': GB_HP_BELOW_MID, 'below_strong': GB_HP_BELOW_STRONG,
    'above': GB_HP_ABOVE,
}
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
    Filter timelines for Model 1 (muscular injuries). Dataset is muscular-only (target1);
    all rows are used (no target2).
    
    Args:
        timelines_df: DataFrame with at least target1 column
        target_column: Must be 'target1' for this Model 1-only script
        
    Returns:
        DataFrame (unchanged; all rows have target1 only)
    """
    if target_column != 'target1':
        raise ValueError(f"This script is for Model 1 only. target_column must be 'target1', got: {target_column}")
    
    if 'target1' not in timelines_df.columns:
        raise ValueError("DataFrame must contain 'target1' column")
    
    # Muscular-only data: use all rows
    filtered_df = timelines_df.copy()
    positives = int(filtered_df['target1'].sum())
    negatives = int((filtered_df['target1'] == 0).sum())
    
    log_message(f"\n📊 Model 1 (Muscular) - target1 only:")
    log_message(f"   Total timelines: {len(filtered_df):,}")
    log_message(f"   Positives (target1=1): {positives:,}")
    log_message(f"   Negatives (target1=0): {negatives:,}")
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
    
    for season_id, filepath in season_files:
        try:
            df = pd.read_csv(filepath, encoding='utf-8-sig', low_memory=False)
            if 'target1' not in df.columns:
                log_message(f"   ⚠️  {season_id}: Missing target1 column - skipping")
                continue
            
            target1_count = (df['target1'] == 1).sum()
            
            if len(df) > 0:
                dfs.append(df)
                total_records += len(df)
                total_target1 += target1_count
                log_message(f"   ✅ {season_id}: {len(df):,} records (target1=1: {target1_count:,})")
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
    
    return combined_df

def load_test_dataset():
    """Load test dataset (2025/26 season)"""
    test_file = TEST_DIR / 'timelines_35day_season_2025_2026_v4_muscular_test.csv'
    
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    log_message(f"\n📂 Loading test dataset: {test_file.name}")
    df_test = pd.read_csv(test_file, encoding='utf-8-sig', low_memory=False)
    
    if 'target1' not in df_test.columns:
        raise ValueError("Test dataset missing target1 column")
    
    target1_count = (df_test['target1'] == 1).sum()
    
    log_message(f"✅ Test dataset: {len(df_test):,} records")
    log_message(f"   target1=1 (Muscular): {target1_count:,} ({target1_count/len(df_test)*100:.4f}%)")
    
    return df_test


def split_train_val_by_player(df, ratio=TRAIN_VAL_RATIO, random_state=SPLIT_RANDOM_STATE):
    """
    Split timeline data into train and validation by player (Option B).
    All rows of a given player go to either train or validation to avoid leakage.

    Args:
        df: DataFrame with at least 'player_id' column
        ratio: Fraction of players (and their rows) for training (default 0.8)
        random_state: For reproducible split

    Returns:
        df_train, df_val
    """
    if 'player_id' not in df.columns:
        raise ValueError("DataFrame must contain 'player_id' for split by player")
    player_ids = df['player_id'].unique()
    n_players = len(player_ids)
    rng = np.random.RandomState(random_state)
    rng.shuffle(player_ids)
    n_train = int(round(n_players * ratio))
    train_ids = set(player_ids[:n_train])
    val_ids = set(player_ids[n_train:])
    df_train = df[df['player_id'].isin(train_ids)].copy().reset_index(drop=True)
    df_val = df[df['player_id'].isin(val_ids)].copy().reset_index(drop=True)
    log_message(f"   Split by player: {n_train} players -> train ({len(df_train):,} rows), "
                f"{n_players - n_train} players -> val ({len(df_val):,} rows)")
    return df_train, df_val


def split_train_val_by_timeline(df, ratio=TRAIN_VAL_RATIO, random_state=SPLIT_RANDOM_STATE):
    """
    Split timeline data into train and validation by randomly picked rows (timelines).
    80% of rows go to train, 20% to validation, regardless of player.
    Same players can appear in both train and validation (aligned with production: same players across seasons).

    Args:
        df: DataFrame of timeline rows
        ratio: Fraction of rows for training (default 0.8)
        random_state: For reproducible split

    Returns:
        df_train, df_val
    """
    n = len(df)
    if n == 0:
        raise ValueError("DataFrame is empty")
    idx = np.arange(n)
    idx_train, idx_val = train_test_split(
        idx, train_size=ratio, random_state=random_state, shuffle=True
    )
    df_train = df.iloc[idx_train].copy().reset_index(drop=True)
    df_val = df.iloc[idx_val].copy().reset_index(drop=True)
    log_message(f"   Split by timeline (random rows): train {len(df_train):,} rows ({100*ratio:.0f}%), "
                f"val {len(df_val):,} rows ({100*(1-ratio):.0f}%)")
    return df_train, df_val


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
    
    # Drop non-feature columns (including excluded club/country raw features)
    exclude_cols = [
        'player_id', 'reference_date', 'date', 'player_name',
        'target1', 'target2', 'target', 'has_minimum_activity'
    ] + list(EXCLUDED_RAW_FEATURES)
    feature_columns = [col for col in df.columns if col not in exclude_cols]
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


def align_features_three(X_train, X_val, X_test):
    """Ensure train, validation, and test have the same feature columns. Missing columns filled with 0."""
    common = sorted(set(X_train.columns) & set(X_val.columns) & set(X_test.columns))
    log_message(f"   Aligning features: {len(common)} common features across train/val/test")
    X_train = X_train.reindex(columns=common, fill_value=0)
    X_val = X_val.reindex(columns=common, fill_value=0)
    X_test = X_test.reindex(columns=common, fill_value=0)
    return X_train, X_val, X_test


def align_features_two(X_train, X_test):
    """Ensure train and test have the same feature columns. Missing columns filled with 0."""
    common = sorted(set(X_train.columns) & set(X_test.columns))
    log_message(f"   Aligning features: {len(common)} common features across train/test")
    X_train = X_train.reindex(columns=common, fill_value=0)
    X_test = X_test.reindex(columns=common, fill_value=0)
    return X_train, X_test


def filter_features_two(X_train, X_test, feature_subset):
    """Filter train and test to feature_subset; add zeros for missing features."""
    available = set(X_train.columns) & set(X_test.columns)
    requested = [f for f in feature_subset if f in available]
    missing = [f for f in feature_subset if f not in available]
    if missing:
        log_message(f"   ⚠️  Warning: {len(missing)} requested features not in both sets")
        if len(missing) <= 10:
            log_message(f"      Missing: {missing}")
    if not requested:
        raise ValueError("No requested features found in train/test!")
    log_message(f"   Using {len(requested)}/{len(feature_subset)} requested features")
    X_train = X_train.reindex(columns=requested, fill_value=0)
    X_test = X_test.reindex(columns=requested, fill_value=0)
    return X_train, X_test


def filter_features_three(X_train, X_val, X_test, feature_subset):
    """Filter train, val, test to feature_subset; add zeros for missing features."""
    available = set(X_train.columns) & set(X_val.columns) & set(X_test.columns)
    requested = [f for f in feature_subset if f in available]
    missing = [f for f in feature_subset if f not in available]
    if missing:
        log_message(f"   ⚠️  Warning: {len(missing)} requested features not in all sets")
        if len(missing) <= 10:
            log_message(f"      Missing: {missing}")
    if not requested:
        raise ValueError("No requested features found in train/val/test!")
    log_message(f"   Using {len(requested)}/{len(feature_subset)} requested features")
    X_train = X_train.reindex(columns=requested, fill_value=0)
    X_val = X_val.reindex(columns=requested, fill_value=0)
    X_test = X_test.reindex(columns=requested, fill_value=0)
    return X_train, X_val, X_test


def train_model_with_feature_subset(feature_subset, verbose=True, use_cache=None, algorithm='lgbm', hyperparameter_set='standard', return_datasets=False, use_full_train=False):
    """
    Train Model 1 (Muscular) on pool data, test on 2025/26 holdout.

    Args:
        feature_subset: List of feature names to use for training
        verbose: Whether to print progress messages
        use_cache: If None, use USE_CACHE; else use this value for prepare_data cache
        algorithm: 'lgbm' (LightGBM) or 'gb' (sklearn GradientBoostingClassifier)
        hyperparameter_set: 'standard' (current), 'below' (more regularized), or 'above' (less regularized)
        return_datasets: If True, include X_val, y_val, X_test, y_test in returned dict (for threshold sweep)
        use_full_train: If True, use 100% of pool (train+val) for training; no validation set. For final production model.

    Returns:
        Dictionary containing:
        - model: Trained muscular model
        - train_metrics: Metrics on train (80% or 100% per use_full_train)
        - val_metrics: Metrics on 20% validation (or placeholder when use_full_train=True)
        - test_metrics: Metrics on 2025/26 test
        - feature_names_used: List of features actually used (may be fewer than requested)
        - X_val, y_val, X_test, y_test: (only if return_datasets=True) validation and test features/labels
    """
    if verbose:
        log_message(f"\n{'='*80}")
        log_message(f"TRAINING MODEL 1 (MUSCULAR) WITH {len(feature_subset)} FEATURES [hp={hyperparameter_set}]")
        log_message(f"{'='*80}")
    
    # Load pool (2018/19 .. 2024/25) and test (2025/26)
    if verbose:
        log_message("\n📂 Loading datasets...")
    
    try:
        df_pool = load_combined_seasonal_datasets_natural(
            min_season=MIN_SEASON,
            exclude_season=EXCLUDE_SEASON
        )
        df_test_all = load_test_dataset()
    except Exception as e:
        log_error("Failed to load datasets", e)
        raise
    
    # Filter for Model 1 only
    try:
        df_pool_muscular = filter_timelines_for_model(df_pool, 'target1')
        df_test_muscular = filter_timelines_for_model(df_test_all, 'target1')
    except Exception as e:
        log_error("Failed to filter timelines", e)
        raise
    
    df_test_muscular = df_test_muscular.reset_index(drop=True)
    _use_cache = USE_CACHE if use_cache is None else use_cache
    cache_suffix = hashlib.md5(str(sorted(feature_subset)).encode()).hexdigest()[:8]
    cache_test = str(CACHE_DIR / f'preprocessed_muscular_test_subset_{cache_suffix}.csv')

    if use_full_train:
        # Use 100% of pool (train+val) for training; no validation set
        if verbose:
            log_message("   Using 100% of pool (train+val) for training (no validation split)...")
        df_train_full = df_pool_muscular.reset_index(drop=True)
        cache_train_full = str(CACHE_DIR / f'preprocessed_muscular_full_train_subset_{cache_suffix}.csv')
        if verbose:
            log_message("   Preparing features for Model 1 (Muscular)...")
        try:
            X_train = prepare_data(df_train_full, cache_file=cache_train_full, use_cache=_use_cache)
            y_train = df_train_full['target1'].values
            X_test = prepare_data(df_test_muscular, cache_file=cache_test, use_cache=_use_cache)
            y_test = df_test_muscular['target1'].values
        except Exception as e:
            log_error("Failed to prepare muscular features", e)
            raise
        try:
            X_train, X_test = align_features_two(X_train, X_test)
        except Exception as e:
            log_error("Failed to align features", e)
            raise
        if verbose:
            log_message(f"\n   Filtering to {len(feature_subset)} requested features...")
        try:
            X_train, X_test = filter_features_two(X_train, X_test, feature_subset)
        except Exception as e:
            log_error("Failed to filter features", e)
            raise
        X_val = None
        y_val = None
    else:
        # 80/20 split by randomly picked timelines (rows)
        if verbose:
            log_message("   Splitting pool into train (80%) and validation (20%) by timeline (random rows)...")
        df_train_80, df_val_20 = split_train_val_by_timeline(
            df_pool_muscular, ratio=TRAIN_VAL_RATIO, random_state=SPLIT_RANDOM_STATE
        )
        cache_train = str(CACHE_DIR / f'preprocessed_muscular_train80_subset_{cache_suffix}.csv')
        cache_val = str(CACHE_DIR / f'preprocessed_muscular_val20_subset_{cache_suffix}.csv')
        if verbose:
            log_message("   Preparing features for Model 1 (Muscular)...")
        try:
            X_train = prepare_data(df_train_80, cache_file=cache_train, use_cache=_use_cache)
            y_train = df_train_80['target1'].values
            X_val = prepare_data(df_val_20, cache_file=cache_val, use_cache=_use_cache)
            y_val = df_val_20['target1'].values
            X_test = prepare_data(df_test_muscular, cache_file=cache_test, use_cache=_use_cache)
            y_test = df_test_muscular['target1'].values
        except Exception as e:
            log_error("Failed to prepare muscular features", e)
            raise
        try:
            X_train, X_val, X_test = align_features_three(X_train, X_val, X_test)
        except Exception as e:
            log_error("Failed to align features", e)
            raise
        if verbose:
            log_message(f"\n   Filtering to {len(feature_subset)} requested features...")
        try:
            X_train, X_val, X_test = filter_features_three(X_train, X_val, X_test, feature_subset)
        except Exception as e:
            log_error("Failed to filter features", e)
            raise

    features_used = sorted(list(X_train.columns))
    if verbose:
        log_message(f"   ✅ Using {len(features_used)} features")

    if verbose:
        if use_full_train:
            log_message(f"\n🚀 Training Model 1 (Muscular Injuries) on 100% train [algorithm={algorithm}]...")
        else:
            log_message(f"\n🚀 Training Model 1 (Muscular Injuries) on 80% train [algorithm={algorithm}]...")

    try:
        if algorithm == 'gb':
            hp = HP_PRESETS_GB.get(hyperparameter_set, GB_HP_STANDARD).copy()
            model = GradientBoostingClassifier(
                **hp,
                random_state=42,
            )
        else:
            hp = HP_PRESETS_LGBM.get(hyperparameter_set, LGBM_HP_STANDARD).copy()
            model = LGBMClassifier(
                **hp,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        model.fit(X_train, y_train)
        if use_full_train:
            train_metrics = evaluate_model(model, X_train, y_train, "Train (100%)")
            val_metrics = None  # No validation set
        else:
            train_metrics = evaluate_model(model, X_train, y_train, "Train (80%)")
            val_metrics = evaluate_model(model, X_val, y_val, "Validation (20%)")
        test_metrics = evaluate_model(model, X_test, y_test, "Test (2025/26)")
    except Exception as e:
        log_error("Failed to train Model 1 (Muscular)", e)
        raise

    train_metrics = convert_numpy_types(train_metrics)
    if val_metrics is not None:
        val_metrics = convert_numpy_types(val_metrics)
    else:
        # Placeholder so downstream code (e.g. export) that expects val_metrics does not break
        val_metrics = dict(train_metrics) if isinstance(train_metrics, dict) else {}
    test_metrics = convert_numpy_types(test_metrics)

    out = {
        'model': model,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'feature_names_used': features_used
    }
    if return_datasets:
        out['X_val'] = X_val
        out['y_val'] = y_val
        out['X_test'] = X_test
        out['y_test'] = y_test
    return out

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
        # Exclude granular club/country features even if present in an old ranking file
        before = len(ranked_features)
        ranked_features = [
            f for f in ranked_features
            if not any(f.startswith(p) for p in EXCLUDED_FEATURE_PREFIXES)
        ]
        if before > len(ranked_features):
            log_message(f"   Excluded {before - len(ranked_features)} club/country features from ranking")
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

def run_iterative_training(resume=True, optimize_on=None, use_cache=None, algorithm='lgbm', features_per_iteration=None, initial_features=None, hp_preset=None):
    """Main function to run iterative feature selection training for Model 1.
    If resume=True and RESULTS_FILE exists, load state and continue from the next iteration.
    optimize_on: 'validation' or 'test' - which metric to use for best iteration and early stopping.
    use_cache: If None, use USE_CACHE; else use this value for prepare_data in train_model_with_feature_subset.
    algorithm: 'lgbm' or 'gb'.
    features_per_iteration, initial_features: step size and first step; defaults from module constants if None.
    hp_preset: If set, use this hyperparameter preset (e.g. 'below_strong') for all iterations; else use 'standard'."""
    if optimize_on is None:
        optimize_on = OPTIMIZE_ON_DEFAULT
    if optimize_on not in ('validation', 'test'):
        raise ValueError(f"optimize_on must be 'validation' or 'test', got: {optimize_on}")
    if features_per_iteration is None:
        features_per_iteration = FEATURES_PER_ITERATION
    if initial_features is None:
        initial_features = INITIAL_FEATURES
    hp_presets = HP_PRESETS_GB if algorithm == 'gb' else HP_PRESETS_LGBM
    hyperparameter_set = hp_preset if hp_preset is not None else 'standard'
    if hyperparameter_set not in hp_presets:
        log_error(f"Unknown preset '{hyperparameter_set}' for {algorithm}. Valid: {list(hp_presets.keys())}")
        return 1

    log_message("="*80)
    log_message("ITERATIVE FEATURE SELECTION TRAINING - MODEL 1 (MUSCULAR) ONLY")
    log_message("="*80)
    log_message(f"\n📋 Configuration:")
    log_message(f"   Algorithm: {algorithm}")
    log_message(f"   Hyperparameter preset: {hyperparameter_set}")
    log_message(f"   Cache: {'disabled (preprocess from CSV)' if use_cache is False else 'enabled'}")
    log_message(f"   Train/val split: {TRAIN_VAL_RATIO:.0%} train / {1-TRAIN_VAL_RATIO:.0%} validation (by timeline / random rows)")
    log_message(f"   Optimize on: {optimize_on} (combined score used for best iteration and early stopping)")
    log_message(f"   Features per iteration: {features_per_iteration}")
    log_message(f"   Initial features: {initial_features}")
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
    
    # Initialize or resume results storage
    if resume and RESULTS_FILE.exists():
        try:
            with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
                results = json.load(f)
            # Use saved optimize_on and step sizes when resuming so best/score are comparable
            cfg = results.get('configuration', {})
            saved_optimize = cfg.get('optimize_on')
            if saved_optimize is not None and saved_optimize != optimize_on:
                log_message(f"Resuming: using saved optimize_on={saved_optimize} (ignoring current --optimize-on={optimize_on})")
                optimize_on = saved_optimize
            if cfg.get('features_per_iteration') is not None:
                features_per_iteration = cfg['features_per_iteration']
            if cfg.get('initial_features') is not None:
                initial_features = cfg['initial_features']
            if cfg.get('algorithm') is not None:
                algorithm = cfg['algorithm']
            if cfg.get('hyperparameter_preset') is not None:
                hyperparameter_set = cfg['hyperparameter_preset']
            combined_scores = [it['combined_score'] for it in results['iterations']]
            n_done = len(results['iterations'])
            if combined_scores:
                best_score = max(combined_scores)
                best_idx = combined_scores.index(best_score)
                best_iteration = results['iterations'][best_idx]['iteration']
                it = results['iterations'][best_idx]
                best_n_features = it.get('n_features_used', it.get('n_features'))
            else:
                best_score = -np.inf
                best_iteration = None
                best_n_features = None
            iteration = n_done
            log_message(f"Resuming from existing results: {RESULTS_FILE}")
            log_message(f"   Loaded {n_done} completed iterations; next iteration will be {iteration + 1}")
            if best_iteration is not None:
                log_message(f"   Best so far: iteration {best_iteration}, {best_n_features} features, score {best_score:.4f} (optimize_on={optimize_on})")
        except Exception as e:
            log_error("Failed to load results file for resume; starting fresh", e)
            results = None
    else:
        results = None

    if results is None:
        log_message("Initializing results storage (fresh run)")
        results = {
            'iterations': [],
            'configuration': {
                'train_val_ratio': TRAIN_VAL_RATIO,
                'split_random_state': SPLIT_RANDOM_STATE,
                'optimize_on': optimize_on,
                'algorithm': algorithm,
                'hyperparameter_preset': hyperparameter_set,
                'features_per_iteration': features_per_iteration,
                'initial_features': initial_features,
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
        combined_scores = []
        iteration = 0
        best_score = -np.inf
        best_iteration = None
        best_n_features = None

    # Ensure configuration has start_time when resuming (keep existing if present)
    if 'start_time' not in results.get('configuration', {}):
        results['configuration']['start_time'] = start_time.isoformat()

    # Iterative training
    log_message("\n" + "="*80)
    log_message("STEP 2: ITERATIVE TRAINING")
    log_message("="*80)
    
    # Calculate number of iterations needed
    max_iterations = (len(ranked_features) - initial_features) // features_per_iteration + 1
    if (len(ranked_features) - initial_features) % features_per_iteration != 0:
        max_iterations += 1
    
    log_message(f"\n   Will train up to {max_iterations} iterations")
    log_message(f"   (from {initial_features} to {len(ranked_features)} features, step {features_per_iteration})")
    
    # Main iteration loop
    while True:
        iteration += 1
        n_features_requested = initial_features + (iteration - 1) * features_per_iteration

        # Skip already-completed iterations when resuming
        if iteration <= len(results['iterations']):
            log_message(f"Resuming: skipping iteration {iteration} (already completed)")
            continue

        # Check if we've used all features
        if n_features_requested > len(ranked_features):
            log_message(f"\n✅ Reached maximum number of features ({len(ranked_features)})")
            break
        
        # Select feature subset
        feature_subset = ranked_features[:n_features_requested]
        
        log_message(f"\n{'='*80}")
        log_message(f"ITERATION {iteration}: Training Model 1 with top {n_features_requested} features (requested)")
        log_message(f"{'='*80}")
        log_message(f"   Features: Top {n_features_requested} from ranked list")
        
        iteration_start = datetime.now()
        
        try:
            log_message(f"Starting training for iteration {iteration} with {n_features_requested} requested features")
            # Train model with this feature subset
            training_results = train_model_with_feature_subset(
                feature_subset,
                verbose=True,
                use_cache=use_cache,
                algorithm=algorithm,
                hyperparameter_set=hyperparameter_set,
            )
            log_message(f"Training completed for iteration {iteration}")
            
            # Validate training results
            required_keys = ['train_metrics', 'val_metrics', 'test_metrics', 'feature_names_used']
            for key in required_keys:
                if key not in training_results:
                    raise KeyError(f"Missing key in training results: {key}")
            
            n_features_used = len(training_results['feature_names_used'])
            if n_features_used != n_features_requested:
                log_message(f"   ⚠️  Requested {n_features_requested} features; {n_features_used} actually used (missing in train/val/test intersection)")
            
            log_message("Calculating combined performance scores")
            combined_score_val = calculate_combined_score(
                training_results['val_metrics'],
                gini_weight=GINI_WEIGHT,
                f1_weight=F1_WEIGHT
            )
            combined_score_test = calculate_combined_score(
                training_results['test_metrics'],
                gini_weight=GINI_WEIGHT,
                f1_weight=F1_WEIGHT
            )
            combined_score = combined_score_val if optimize_on == 'validation' else combined_score_test
            
            combined_scores.append(combined_score)
            log_message(f"   Validation combined score: {combined_score_val:.4f}")
            log_message(f"   Test combined score: {combined_score_test:.4f}")
            log_message(f"   Using for selection (optimize_on={optimize_on}): {combined_score:.4f}")
            
            # Check if this is the best so far (use n_features_used for best count)
            if combined_score > best_score:
                best_score = combined_score
                best_iteration = iteration
                best_n_features = n_features_used
                log_message(f"New best score! Iteration {iteration} with {n_features_used} features used: {best_score:.4f}")
            
            # Store iteration results: real counts to avoid 759 vs 760 confusion
            iteration_data = {
                'iteration': iteration,
                'n_features_requested': n_features_requested,
                'n_features_used': n_features_used,
                'n_features': n_features_used,
                'features': training_results['feature_names_used'],
                'model1_muscular': {
                    'train': training_results['train_metrics'],
                    'val': training_results['val_metrics'],
                    'test': training_results['test_metrics']
                },
                'combined_score_val': float(combined_score_val),
                'combined_score_test': float(combined_score_test),
                'combined_score': float(combined_score),
                'timestamp': iteration_start.isoformat(),
                'training_time_seconds': (datetime.now() - iteration_start).total_seconds()
            }
            
            results['iterations'].append(iteration_data)
            
            # Print iteration summary
            log_message(f"\n📊 Iteration {iteration} Results:")
            log_message(f"   Features: requested={n_features_requested}, used={n_features_used}")
            log_message(f"   Validation: Gini={training_results['val_metrics']['gini']:.4f}, "
                      f"F1={training_results['val_metrics']['f1']:.4f} -> combined={combined_score_val:.4f}")
            log_message(f"   Test:      Gini={training_results['test_metrics']['gini']:.4f}, "
                      f"F1={training_results['test_metrics']['f1']:.4f} -> combined={combined_score_test:.4f}")
            log_message(f"   Best selection (optimize_on={optimize_on}): {combined_score:.4f}")
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
        best_iter_data = next(it for it in results['iterations'] if it['iteration'] == best_iteration)
        optimize_on = results.get('configuration', {}).get('optimize_on', OPTIMIZE_ON_DEFAULT)
        log_message(f"\n📈 Best Performance (Iteration {best_iteration}, optimize_on={optimize_on}):")
        log_message(f"   Validation: Gini={best_iter_data['model1_muscular']['val']['gini']:.4f}, "
                   f"F1={best_iter_data['model1_muscular']['val']['f1']:.4f} -> combined={best_iter_data['combined_score_val']:.4f}")
        log_message(f"   Test:      Gini={best_iter_data['model1_muscular']['test']['gini']:.4f}, "
                   f"F1={best_iter_data['model1_muscular']['test']['f1']:.4f} -> combined={best_iter_data['combined_score_test']:.4f}")
        log_message(f"   Selection score (used for best): {best_iter_data['combined_score']:.4f}")
    
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
            
            plot_suffix = '_gb' if algorithm == 'gb' else ''
            plot_file = MODEL_OUTPUT_DIR / f'iterative_feature_selection_muscular_plot{plot_suffix}.png'
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


def export_best_model(use_cache=None, algorithm='lgbm', export_iteration=None, hp_preset=None, train_on_full_data=False):
    """
    Load iterative results, find the best iteration by combined_score (val or test per optimize_on),
    or use a specific iteration if export_iteration is set. Re-train that model and save it (joblib)
    plus feature list (JSON).
    use_cache: If None, use USE_CACHE; else use this value for prepare_data.
    algorithm: 'lgbm' or 'gb' - must match the results file (use same as training run).
    export_iteration: If set, export this iteration number (e.g. 6 for 300 features); else export best by score.
    hp_preset: If set, use this hyperparameter preset (e.g. 'below_strong'); else use 'standard'.
    train_on_full_data: If True, train on 100% of pool (train+val); no validation set. For final production model.
    """
    if not RESULTS_FILE.exists():
        log_error(f"Results file not found: {RESULTS_FILE}. Run iterative training first.")
        return 1
    with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
        results = json.load(f)
    iterations = results.get('iterations', [])
    if not iterations:
        log_error("No iterations in results file.")
        return 1
    if export_iteration is not None:
        best_it = next((it for it in iterations if it['iteration'] == export_iteration), None)
        if best_it is None:
            log_error(f"Iteration {export_iteration} not found in results. Available: {[it['iteration'] for it in iterations]}")
            return 1
        log_message(f"Exporting specific iteration: {export_iteration} (--export-iteration)")
    else:
        best_it = max(iterations, key=lambda x: x['combined_score'])
    n_features = best_it.get('n_features_used', best_it.get('n_features'))
    feature_list = best_it['features']
    optimize_on = results.get('configuration', {}).get('optimize_on', OPTIMIZE_ON_DEFAULT)
    log_message(f"Best iteration: {best_it['iteration']} ({n_features} features used, "
                f"optimize_on={optimize_on}, selection score={best_it['combined_score']:.4f})")
    hp_presets = HP_PRESETS_GB if algorithm == 'gb' else HP_PRESETS_LGBM
    hyperparameter_set = hp_preset if hp_preset is not None else 'standard'
    if hyperparameter_set not in hp_presets:
        log_error(f"Unknown preset '{hyperparameter_set}' for {algorithm}. Valid: {list(hp_presets.keys())}")
        return 1
    if train_on_full_data:
        log_message(f"Re-training model with these features (100% train, hp={hyperparameter_set})...")
    else:
        log_message(f"Re-training model with these features (80% train, hp={hyperparameter_set})...")
    training_results = train_model_with_feature_subset(
        feature_list, verbose=True, use_cache=use_cache, algorithm=algorithm,
        hyperparameter_set=hyperparameter_set,
        use_full_train=train_on_full_data,
    )
    model = training_results['model']
    n_features_used = len(training_results['feature_names_used'])
    combined_score_test = calculate_combined_score(
        training_results['test_metrics'], gini_weight=GINI_WEIGHT, f1_weight=F1_WEIGHT
    )
    if train_on_full_data:
        combined_score_val = None
    else:
        combined_score_val = calculate_combined_score(
            training_results['val_metrics'], gini_weight=GINI_WEIGHT, f1_weight=F1_WEIGHT
        )
    suffix = '_gb' if algorithm == 'gb' else ''
    model_path = MODEL_OUTPUT_DIR / f'lgbm_muscular_best_iteration{suffix}.joblib'
    features_path = MODEL_OUTPUT_DIR / f'lgbm_muscular_best_iteration_features{suffix}.json'
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    meta = {
        'n_features': n_features_used,
        'algorithm': algorithm,
        'iteration': best_it['iteration'],
        'optimize_on': optimize_on,
        'hyperparameter_preset': hyperparameter_set,
        'trained_on_full_data': train_on_full_data,
        'combined_score_test': float(combined_score_test),
        'combined_score': best_it['combined_score'],
        'features': training_results['feature_names_used']
    }
    if combined_score_val is not None:
        meta['combined_score_val'] = float(combined_score_val)
    with open(features_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    log_message(f"Saved model to: {model_path}")
    log_message(f"Saved feature list to: {features_path} (n_features={n_features_used})")
    if train_on_full_data:
        log_message(f"   Trained on 100% data; Test combined score: {combined_score_test:.4f}")
    else:
        log_message(f"   Validation combined score: {combined_score_val:.4f}; Test combined score: {combined_score_test:.4f}")
    return 0


# Best model artifacts (for evaluate-best-on-test)
BEST_MODEL_PATH = MODEL_OUTPUT_DIR / 'lgbm_muscular_best_iteration.joblib'
BEST_FEATURES_PATH = MODEL_OUTPUT_DIR / 'lgbm_muscular_best_iteration_features.json'
PREVIOUS_COMBINED_SCORE_OLD_TEST = 0.4257  # ~0.42 from iterative results on old test set


def evaluate_best_model_on_test():
    """
    Load the saved 60-feature model, run it on the current test dataset (regenerated
    with all reference days), and compute the combined score for comparison with the
    previous ~0.42 value from the old test set.
    """
    if not BEST_MODEL_PATH.exists():
        log_error(f"Best model not found: {BEST_MODEL_PATH}. Run --export-best first.")
        return 1
    if not BEST_FEATURES_PATH.exists():
        log_error(f"Best features JSON not found: {BEST_FEATURES_PATH}.")
        return 1

    log_message("\n" + "=" * 80)
    log_message("EVALUATE 60-FEATURE MODEL ON (REGENERATED) TEST SET")
    log_message("=" * 80)

    # Load model and feature list
    log_message(f"\nLoading model: {BEST_MODEL_PATH.name}")
    model = joblib.load(BEST_MODEL_PATH)
    with open(BEST_FEATURES_PATH, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    feature_list = meta['features']
    prev_score = meta.get('combined_score', PREVIOUS_COMBINED_SCORE_OLD_TEST)
    log_message(f"Model has {len(feature_list)} features (iteration {meta.get('iteration', '?')})")
    log_message(f"Previous combined score (old test set): {prev_score:.4f}")

    # Use model's feature order (source of truth)
    model_features = list(getattr(model, 'feature_name_', None) or feature_list)

    # Load test dataset (current CSV = regenerated with all reference days)
    log_message("\nLoading test dataset...")
    df_test_all = load_test_dataset()
    df_test_muscular = filter_timelines_for_model(df_test_all, 'target1')
    df_test_muscular = df_test_muscular.reset_index(drop=True)

    # Preprocess test: use cache only if row count matches (new test will invalidate old cache)
    cache_suffix = hashlib.md5(str(sorted(feature_list)).encode()).hexdigest()[:8]
    cache_file_test = str(CACHE_DIR / f'preprocessed_muscular_test_subset_{cache_suffix}.csv')
    log_message("Preparing test features (cache invalidated if test set changed)...")
    X_test_full = prepare_data(df_test_muscular, cache_file=cache_file_test, use_cache=USE_CACHE)
    y_test = df_test_muscular['target1'].values

    # Build X_test with exactly the columns the model expects (same order); missing -> zeros
    missing = [f for f in model_features if f not in X_test_full.columns]
    if missing:
        log_message(f"   Adding {len(missing)} missing feature(s) as zeros (categories absent in test).")
        for f in missing:
            X_test_full[f] = 0
    X_test = X_test_full[model_features]

    # Evaluate
    log_message("\nEvaluating on test set...")
    test_metrics = evaluate_model(model, X_test, y_test, "Test")
    combined_score = calculate_combined_score(
        test_metrics,
        gini_weight=GINI_WEIGHT,
        f1_weight=F1_WEIGHT
    )

    # Summary and comparison
    log_message("\n" + "=" * 80)
    log_message("COMBINED SCORE COMPARISON")
    log_message("=" * 80)
    log_message(f"   Previous (old test set, fewer reference days): {prev_score:.4f}")
    log_message(f"   Current  (regenerated test, all reference days): {combined_score:.4f}")
    diff = combined_score - prev_score
    log_message(f"   Difference: {diff:+.4f}")
    log_message("=" * 80 + "\n")
    return 0


def run_hyperparameter_test(algorithm='lgbm', use_cache=None, presets_list=None, output_suffix=None):
    """
    Run hyperparameter sensitivity test: train with selected presets on the fixed best feature set
    (759 for LGBM, 300 for GB). Report and save comparison.
    If presets_list is None, uses ['below', 'standard', 'above'].
    If output_suffix is set (e.g. '_refinement'), appends it to the output JSON base name.
    """
    if use_cache is None:
        use_cache = USE_CACHE
    hp_presets = HP_PRESETS_GB if algorithm == 'gb' else HP_PRESETS_LGBM
    if presets_list is None:
        presets_list = ['below', 'standard', 'above']
    else:
        presets_list = [p.strip() for p in presets_list]
        invalid = [p for p in presets_list if p not in hp_presets]
        if invalid:
            log_error(f"Unknown preset(s) for {algorithm}: {invalid}. Valid: {list(hp_presets.keys())}")
            return 1
    if algorithm == 'gb':
        features_path = MODEL_OUTPUT_DIR / 'lgbm_muscular_best_iteration_features_gb.json'
        out_base = 'hyperparameter_test_gb_300'
        n_feat_label = '300'
    else:
        features_path = MODEL_OUTPUT_DIR / 'lgbm_muscular_best_iteration_features.json'
        out_base = 'hyperparameter_test_lgbm_759'
        n_feat_label = '759'
    out_json = MODEL_OUTPUT_DIR / f"{out_base}{output_suffix or ''}.json"
    if not features_path.exists():
        log_error(f"Features file not found: {features_path}. Run --algorithm {algorithm} --export-best first.")
        return 1
    with open(features_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    feature_list = meta['features']
    log_message("\n" + "=" * 80)
    log_message(f"HYPERPARAMETER TEST - {algorithm.upper()} ({n_feat_label} features)")
    log_message("=" * 80)
    log_message(f"   Presets: {', '.join(presets_list)}")
    log_message(f"   Results will be saved to: {out_json}")
    log_message("=" * 80 + "\n")

    results = []
    for preset in presets_list:
        log_message(f"\n--- Training with preset: {preset} ---")
        try:
            training_results = train_model_with_feature_subset(
                feature_list, verbose=True, use_cache=use_cache, algorithm=algorithm, hyperparameter_set=preset
            )
        except Exception as e:
            log_error(f"Hyperparameter test failed for preset={preset}", e)
            return 1
        combined_test = calculate_combined_score(
            training_results['test_metrics'], gini_weight=GINI_WEIGHT, f1_weight=F1_WEIGHT
        )
        rec = {
            'preset': preset,
            'hyperparameters': hp_presets[preset],
            'train_metrics': training_results['train_metrics'],
            'val_metrics': training_results['val_metrics'],
            'test_metrics': training_results['test_metrics'],
            'combined_score_test': float(combined_test),
        }
        results.append(rec)
        log_message(f"   {preset}: test Gini={training_results['test_metrics']['gini']:.4f}, test F1={training_results['test_metrics']['f1']:.4f}, combined={combined_test:.4f}")

    # Summary table
    log_message("\n" + "=" * 80)
    log_message("HYPERPARAMETER TEST SUMMARY")
    log_message("=" * 80)
    log_message(
        f"{'Preset':<10} {'Train Gini':>12} {'Val Gini':>12} {'Test Gini':>12} {'Test Prec':>10} {'Test Rec':>10} {'Test F1':>10} {'Combined':>10}"
    )
    log_message("-" * 96)
    for r in results:
        log_message(
            f"{r['preset']:<10} "
            f"{r['train_metrics']['gini']:>12.4f} "
            f"{r['val_metrics']['gini']:>12.4f} "
            f"{r['test_metrics']['gini']:>12.4f} "
            f"{r['test_metrics']['precision']:>10.4f} "
            f"{r['test_metrics']['recall']:>10.4f} "
            f"{r['test_metrics']['f1']:>10.4f} "
            f"{r['combined_score_test']:>10.4f}"
        )
    log_message("=" * 80 + "\n")

    report = {
        'algorithm': algorithm,
        'n_features': len(feature_list),
        'timestamp': datetime.now().isoformat(),
        'presets': results,
    }
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        log_message(f"Saved report to: {out_json}")
    except Exception as e:
        log_error(f"Failed to save hyperparameter test report", e)
        return 1
    return 0


def run_threshold_sweep(algorithm='lgbm', hp_preset='below', use_cache=None):
    """
    Train once with the given preset (e.g. 'below'), then sweep classification thresholds
    on val and test to find a better precision-recall combination. Saves CSV to models dir.
    """
    if use_cache is None:
        use_cache = USE_CACHE
    hp_presets = HP_PRESETS_GB if algorithm == 'gb' else HP_PRESETS_LGBM
    if hp_preset not in hp_presets:
        log_error(f"Preset '{hp_preset}' not available for {algorithm}. Valid: {list(hp_presets.keys())}")
        return 1
    if algorithm == 'gb':
        features_path = MODEL_OUTPUT_DIR / 'lgbm_muscular_best_iteration_features_gb.json'
        n_feat_label = '300'
    else:
        features_path = MODEL_OUTPUT_DIR / 'lgbm_muscular_best_iteration_features.json'
        n_feat_label = '759'
    if not features_path.exists():
        log_error(f"Features file not found: {features_path}. Run --algorithm {algorithm} --export-best first.")
        return 1
    with open(features_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    feature_list = meta['features']

    log_message("\n" + "=" * 80)
    log_message(f"THRESHOLD SWEEP - {algorithm.upper()} ({n_feat_label} features, hp={hp_preset})")
    log_message("=" * 80)
    log_message("   Training once, then sweeping thresholds on val and test for precision/recall/F1")
    log_message("=" * 80 + "\n")

    try:
        res = train_model_with_feature_subset(
            feature_list, verbose=True, use_cache=use_cache, algorithm=algorithm,
            hyperparameter_set=hp_preset, return_datasets=True
        )
    except Exception as e:
        log_error("Threshold sweep: training failed", e)
        return 1

    model = res['model']
    X_val = res['X_val']
    y_val = res['y_val']
    X_test = res['X_test']
    y_test = res['y_test']

    proba_val = model.predict_proba(X_val)[:, 1]
    proba_test = model.predict_proba(X_test)[:, 1]

    thresholds = np.linspace(0.05, 0.95, 19)
    rows = []
    for t in thresholds:
        pred_val = (proba_val >= t).astype(int)
        pred_test = (proba_test >= t).astype(int)
        row = {
            'threshold': round(t, 2),
            'val_precision': float(precision_score(y_val, pred_val, zero_division=0)),
            'val_recall': float(recall_score(y_val, pred_val, zero_division=0)),
            'val_f1': float(f1_score(y_val, pred_val, zero_division=0)),
            'test_precision': float(precision_score(y_test, pred_test, zero_division=0)),
            'test_recall': float(recall_score(y_test, pred_test, zero_division=0)),
            'test_f1': float(f1_score(y_test, pred_test, zero_division=0)),
        }
        rows.append(row)

    csv_path = MODEL_OUTPUT_DIR / f'threshold_sweep_{algorithm}_{hp_preset}_{len(feature_list)}.csv'
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    log_message(f"Saved threshold sweep to: {csv_path}")

    best_f1_val_idx = df['val_f1'].idxmax()
    best_row = df.loc[best_f1_val_idx]
    log_message("\n" + "=" * 80)
    log_message("THRESHOLD SWEEP SUMMARY")
    log_message("=" * 80)
    log_message(f"   Best F1 on validation at threshold {best_row['threshold']:.2f}: val_f1={best_row['val_f1']:.4f}, test_f1={best_row['test_f1']:.4f}, test_precision={best_row['test_precision']:.4f}, test_recall={best_row['test_recall']:.4f}")
    log_message(f"   First rows (threshold, val_prec, val_rec, val_f1, test_prec, test_rec, test_f1):")
    for _, r in df.head(6).iterrows():
        log_message(f"      {r['threshold']:.2f}  {r['val_precision']:.4f}  {r['val_recall']:.4f}  {r['val_f1']:.4f}  {r['test_precision']:.4f}  {r['test_recall']:.4f}  {r['test_f1']:.4f}")
    log_message("   ...")
    log_message("=" * 80 + "\n")
    return 0


def run_neighbourhood_feature_counts(feature_counts_list, use_cache=None, algorithm='gb', results_path=None):
    """
    Run training for specific feature counts (e.g. 270, 290, 310) using the same
    ranking as iterative training. Saves results to a dedicated neighbourhood JSON.
    """
    if use_cache is None:
        use_cache = USE_CACHE
    if results_path is None:
        results_path = MODEL_OUTPUT_DIR / 'iterative_feature_selection_results_muscular_gb_neighbourhood.json'
    log_message("\n" + "=" * 80)
    log_message("NEIGHBOURHOOD FEATURE COUNTS (fixed N from ranking)")
    log_message("=" * 80)
    log_message(f"   Feature counts: {feature_counts_list}")
    log_message(f"   Algorithm: {algorithm}")
    log_message(f"   Results will be saved to: {results_path}")
    try:
        ranked_features = load_feature_ranking()
    except Exception as e:
        log_error("Failed to load feature ranking", e)
        return 1
    runs = []
    for idx, n_features in enumerate(feature_counts_list, start=1):
        if n_features > len(ranked_features):
            log_message(f"Skipping {n_features} (ranking has only {len(ranked_features)} features)")
            continue
        feature_subset = ranked_features[:n_features]
        log_message(f"\n--- Run {idx}/{len(feature_counts_list)}: top {n_features} features ---")
        iteration_start = datetime.now()
        try:
            training_results = train_model_with_feature_subset(
                feature_subset,
                verbose=True,
                use_cache=use_cache,
                algorithm=algorithm
            )
        except Exception as e:
            log_error(f"Training failed for n_features={n_features}", e)
            continue
        n_features_used = len(training_results['feature_names_used'])
        combined_score_val = calculate_combined_score(
            training_results['val_metrics'], gini_weight=GINI_WEIGHT, f1_weight=F1_WEIGHT
        )
        combined_score_test = calculate_combined_score(
            training_results['test_metrics'], gini_weight=GINI_WEIGHT, f1_weight=F1_WEIGHT
        )
        elapsed = (datetime.now() - iteration_start).total_seconds()
        run_data = {
            'n_features_requested': n_features,
            'n_features_used': n_features_used,
            'n_features': n_features_used,
            'features': training_results['feature_names_used'],
            'model1_muscular': {
                'train': training_results['train_metrics'],
                'val': training_results['val_metrics'],
                'test': training_results['test_metrics']
            },
            'combined_score_val': float(combined_score_val),
            'combined_score_test': float(combined_score_test),
            'timestamp': iteration_start.isoformat(),
            'training_time_seconds': elapsed
        }
        runs.append(run_data)
        log_message(f"   Val combined: {combined_score_val:.4f}; Test combined: {combined_score_test:.4f}; Time: {elapsed:.1f}s")
    if not runs:
        log_error("No runs completed.")
        return 1
    results = {
        'description': 'Neighbourhood run (fixed feature counts from same ranking as iterative)',
        'algorithm': algorithm,
        'feature_counts': feature_counts_list,
        'runs': runs,
        'configuration': {
            'gini_weight': GINI_WEIGHT,
            'f1_weight': F1_WEIGHT,
        }
    }
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        log_message(f"\nSaved results to: {results_path}")
    except Exception as e:
        log_error(f"Failed to save neighbourhood results to {results_path}", e)
        return 1
    best_run = max(runs, key=lambda r: r['combined_score_test'])
    log_message(f"   Best test combined: {best_run['n_features_used']} features -> {best_run['combined_score_test']:.4f}")
    log_message("=" * 80 + "\n")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iterative feature selection for Model 1 (Muscular)")
    parser.add_argument("--no-resume", "--fresh", dest="resume", action="store_false",
                        help="Start from scratch (ignore existing results file)")
    parser.add_argument("--export-best", action="store_true",
                        help="Re-train the best iteration from results and save model + feature list, then exit")
    parser.add_argument("--export-iteration", type=int, default=None,
                        help="With --export-best: export this iteration number (e.g. 6 for 300 features) instead of best by score")
    parser.add_argument("--export-hp-preset", type=str, default=None,
                        help="With --export-best: use this HP preset (e.g. below_strong). If not set, uses standard.")
    parser.add_argument("--train-on-full-data", action="store_true",
                        help="With --export-best: train on 100%% of pool (train+val); no validation set. For final production model.")
    parser.add_argument("--hyperparameter-test", action="store_true",
                        help="Run sensitivity test: train with below/standard/above HP presets on best feature set, save comparison JSON")
    parser.add_argument("--hyperparameter-test-presets", type=str, default=None,
                        help="Comma-separated presets to run only these (e.g. below_mid,below_strong). Saves to ..._refinement.json. Implies HP test.")
    parser.add_argument("--hyperparameter-test-below-refinement", action="store_true",
                        help="Run HP test with only below_mid and below_strong presets; save to ..._refinement.json")
    parser.add_argument("--threshold-sweep", action="store_true",
                        help="Train with given HP preset, sweep classification thresholds on val/test for precision-recall, save CSV")
    parser.add_argument("--hp-preset", choices=["below", "below_mid", "below_strong", "standard", "above"], default="below",
                        help="HP preset for --threshold-sweep (default: below). LGBM-only: below_mid, below_strong.")
    parser.add_argument("--evaluate-best-on-test", action="store_true",
                        help="Load the saved 60-feature model, run on current test set, report combined score vs previous ~0.42")
    parser.add_argument("--optimize-on", choices=["validation", "test"], default=OPTIMIZE_ON_DEFAULT,
                        help="Which dataset to use for best iteration and early stopping (default: validation)")
    parser.add_argument("--algorithm", choices=["lgbm", "gb"], default="lgbm",
                        help="Algorithm: lgbm (LightGBM) or gb (sklearn GradientBoosting). Default: lgbm. For gb, --features-per-iteration defaults to 50.")
    parser.add_argument("--features-per-iteration", type=int, default=None,
                        help="Number of features to add per iteration (default: 20 for lgbm, 50 for gb)")
    parser.add_argument("--initial-features", type=int, default=None,
                        help="Number of features in first iteration (default: same as --features-per-iteration)")
    parser.add_argument("--iterative-hp-preset", type=str, default=None,
                        help="HP preset for iterative training (e.g. below_strong). If not set, uses standard. Saved in results when resuming.")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache; preprocess from timeline CSVs")
    parser.add_argument("--run-feature-counts", type=str, default=None,
                        help="Comma-separated feature counts to run (e.g. 270,290,310). Uses same ranking; trains once per count and saves to neighbourhood results file. Implies algorithm=gb.")
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    use_cache = not args.no_cache
    run_feature_counts = None
    if args.run_feature_counts:
        try:
            run_feature_counts = [int(x.strip()) for x in args.run_feature_counts.split(",") if x.strip()]
        except ValueError:
            print("Invalid --run-feature-counts; use comma-separated integers (e.g. 270,290,310)", file=sys.stderr)
            sys.exit(1)
        if not run_feature_counts:
            print("--run-feature-counts must list at least one feature count", file=sys.stderr)
            sys.exit(1)
    algorithm = args.algorithm
    if run_feature_counts is not None:
        algorithm = 'gb'
    features_per_iteration = args.features_per_iteration
    initial_features = args.initial_features
    if features_per_iteration is None:
        features_per_iteration = 50 if algorithm == 'gb' else FEATURES_PER_ITERATION
    if initial_features is None:
        initial_features = features_per_iteration

    # When using GB, write to separate results/log so LGBM and GB runs don't overwrite each other
    if algorithm == 'gb':
        RESULTS_FILE = MODEL_OUTPUT_DIR / 'iterative_feature_selection_results_muscular_gb.json'
        LOG_FILE = MODEL_OUTPUT_DIR / 'iterative_training_muscular_gb.log'
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            f.write(f"=== Iterative Training Log (Model 1 - Muscular, algorithm=gb) Started at {datetime.now().isoformat()} ===\n")

    try:
        if run_feature_counts is not None:
            exit_code = run_neighbourhood_feature_counts(
                run_feature_counts, use_cache=use_cache, algorithm=algorithm
            )
            sys.exit(exit_code)
        if args.evaluate_best_on_test:
            exit_code = evaluate_best_model_on_test()
            sys.exit(exit_code)
        if args.export_best:
            exit_code = export_best_model(
                use_cache=use_cache, algorithm=algorithm,
                export_iteration=args.export_iteration, hp_preset=args.export_hp_preset,
                train_on_full_data=args.train_on_full_data,
            )
            sys.exit(exit_code)
        if args.hyperparameter_test or args.hyperparameter_test_presets or args.hyperparameter_test_below_refinement:
            if args.hyperparameter_test_below_refinement:
                presets_list = ['below_mid', 'below_strong']
                output_suffix = '_refinement'
            elif args.hyperparameter_test_presets:
                presets_list = [p.strip() for p in args.hyperparameter_test_presets.split(',') if p.strip()]
                output_suffix = '_refinement'
                if not presets_list:
                    print("--hyperparameter-test-presets must list at least one preset", file=sys.stderr)
                    sys.exit(1)
            else:
                presets_list = None
                output_suffix = None
            exit_code = run_hyperparameter_test(
                algorithm=algorithm, use_cache=use_cache,
                presets_list=presets_list, output_suffix=output_suffix,
            )
            sys.exit(exit_code)
        if args.threshold_sweep:
            exit_code = run_threshold_sweep(algorithm=algorithm, hp_preset=args.hp_preset, use_cache=use_cache)
            sys.exit(exit_code)
        exit_code = run_iterative_training(
            resume=args.resume,
            optimize_on=args.optimize_on,
            use_cache=use_cache,
            algorithm=algorithm,
            features_per_iteration=features_per_iteration,
            initial_features=initial_features,
            hp_preset=args.iterative_hp_preset,
        )
        sys.exit(exit_code)
    except KeyboardInterrupt:
        log_message("\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        log_error("Fatal error in main execution", e)
        sys.exit(1)
