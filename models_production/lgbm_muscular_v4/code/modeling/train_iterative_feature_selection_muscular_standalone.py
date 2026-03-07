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

Note: The stored model lgbm_muscular_exp10_labeling.joblib (Exp 10, ~0.59 test Gini) is the
designated production best and must not be overwritten by this script. When using --exp10-data,
export writes to lgbm_muscular_best_iteration_exp10.joblib instead of lgbm_muscular_best_iteration.joblib.
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
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))
MODEL_OUTPUT_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'models'
TIMELINES_DIR = SCRIPT_DIR.parent / 'timelines'
CACHE_DIR = ROOT_DIR / 'cache'

# ========== CONFIGURATION ==========
FEATURES_PER_ITERATION = 20
INITIAL_FEATURES = 200  # Start at 200 features; step +20 (LGBM) or +50 (GB) per iteration
CONSECUTIVE_DROPS_THRESHOLD = 3
PERFORMANCE_DROP_THRESHOLD = 0.001  # Minimum drop to count as a drop (0.1%)
GINI_WEIGHT = 0.6
F1_WEIGHT = 0.4
RANKING_FILE = MODEL_OUTPUT_DIR / 'feature_ranking.json'
RESULTS_FILE = MODEL_OUTPUT_DIR / 'iterative_feature_selection_results_muscular.json'
LOG_FILE = MODEL_OUTPUT_DIR / 'iterative_training_muscular.log'
# Exp 10 enforcement: keep this exact 340-feature prefix in iterative ranking so
# iteration 17 reproduces the stored 340-feature model setup.
BEST_FEATURES_PATH = MODEL_OUTPUT_DIR / 'lgbm_muscular_best_iteration_features.json'
FIXED_PREFIX_FEATURE_COUNT = 340

# Exclude granular club/country features (too granular / uninformative for Premier League–focused use)
EXCLUDED_RAW_FEATURES = ('current_club', 'current_club_country', 'previous_club', 'previous_club_country')
EXCLUDED_FEATURE_PREFIXES = ('current_club_country_', 'current_club_', 'previous_club_country_', 'previous_club_')

# Training configuration
MIN_SEASON = '2018_2019'  # Start from 2018/19 season (inclusive)
EXCLUDE_SEASON = '2025_2026'  # Test dataset season
TRAIN_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'data' / 'timelines' / 'train'
TEST_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'data' / 'timelines' / 'test'
V4_RAW_DATA = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'data' / 'raw_data'
INJURIES_FILE = V4_RAW_DATA / 'injuries_data.csv'
USE_CACHE = True
# When True, load train from *LABELED_SUFFIX and test from timelines_35day_season_2025_2026 + LABELED_SUFFIX
USE_LABELED_TIMELINES = False
# Labeled file suffix (used when USE_LABELED_TIMELINES). Overridden to D-7 when --exp10-data.
LABELED_SUFFIX = '_v4_labeled.csv'
# Target column for labels (target1 = muscular only; target_msu = muscular/skeletal/unknown [D-7,D-1]). Set by --exp11-data.
TARGET_COLUMN = 'target1'
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
# Slightly above standard (less aggressive than LGBM_HP_ABOVE)
LGBM_HP_ABOVE_MID = {
    'n_estimators': 260, 'max_depth': 12, 'learning_rate': 0.12,
    'min_child_samples': 15, 'subsample': 0.9, 'colsample_bytree': 0.9,
    'reg_alpha': 0.05, 'reg_lambda': 0.8,
}
HP_PRESETS_LGBM = {
    'standard': LGBM_HP_STANDARD,
    'below': LGBM_HP_BELOW,
    'below_mid': LGBM_HP_BELOW_MID,
    'below_strong': LGBM_HP_BELOW_STRONG,
    'above_mid': LGBM_HP_ABOVE_MID,
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
    Filter timelines for a single target. For target1/target_msu all rows are used; target_column defines the label.
    For target2 (skeletal), excludes muscular-only rows so Model 2 learns skeletal vs non-injury only.

    Args:
        timelines_df: DataFrame with at least the target column(s)
        target_column: 'target1' (muscular), 'target_msu' (MSU), or 'target2' (skeletal)

    Returns:
        Filtered DataFrame
    """
    if target_column == 'target2':
        if 'target1' not in timelines_df.columns or 'target2' not in timelines_df.columns:
            raise ValueError("DataFrame must contain both 'target1' and 'target2' for skeletal (target2)")
        mask = (timelines_df['target2'] == 1) | ((timelines_df['target1'] == 0) & (timelines_df['target2'] == 0))
        filtered_df = timelines_df[mask].copy()
        excluded_count = ((timelines_df['target1'] == 1) & (timelines_df['target2'] == 0)).sum()
        positives = int(filtered_df['target2'].sum())
        negatives = int(((filtered_df['target1'] == 0) & (filtered_df['target2'] == 0)).sum())
        log_message(f"\n📊 Model 2 (Skeletal) - target2 only:")
        log_message(f"   Original timelines: {len(timelines_df):,}")
        log_message(f"   After filtering: {len(filtered_df):,}")
        log_message(f"   Positives (target2=1): {positives:,}")
        log_message(f"   Negatives (target1=0, target2=0): {negatives:,}")
        log_message(f"   Excluded (muscular injuries): {excluded_count:,}")
        if len(filtered_df) > 0:
            log_message(f"   Target ratio: {positives / len(filtered_df) * 100:.2f}%")
        return filtered_df

    if target_column not in ('target1', 'target_msu'):
        raise ValueError(f"target_column must be 'target1', 'target_msu', or 'target2', got: {target_column}")

    if target_column not in timelines_df.columns:
        raise ValueError(f"DataFrame must contain '{target_column}' column")

    filtered_df = timelines_df.copy()
    positives = int(filtered_df[target_column].sum())
    negatives = int((filtered_df[target_column] == 0).sum())
    label = "Model 1 (Muscular) - target1 only" if target_column == 'target1' else "Model (MSU) - target_msu"
    pos_label = f"Positives ({target_column}=1)"
    neg_label = f"Negatives ({target_column}=0)"
    log_message(f"\n📊 {label}:")
    log_message(f"   Total timelines: {len(filtered_df):,}")
    log_message(f"   {pos_label}: {positives:,}")
    log_message(f"   {neg_label}: {negatives:,}")
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
    if USE_LABELED_TIMELINES:
        pattern = str(TRAIN_DIR / ('timelines_35day_season_*' + LABELED_SUFFIX))
        suffix_for_split = LABELED_SUFFIX
    else:
        pattern = str(TRAIN_DIR / 'timelines_35day_season_*_v4_muscular_train.csv')
        suffix_for_split = '_v4_muscular_train'
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
                season_part = parts[1].split(suffix_for_split)[0].strip('_')
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
    total_positive = 0

    for season_id, filepath in season_files:
        try:
            df = pd.read_csv(filepath, encoding='utf-8-sig', low_memory=False)
            if TARGET_COLUMN not in df.columns:
                log_message(f"   ⚠️  {season_id}: Missing {TARGET_COLUMN} column - skipping")
                continue

            pos_count = (df[TARGET_COLUMN] == 1).sum()

            if len(df) > 0:
                dfs.append(df)
                total_records += len(df)
                total_positive += pos_count
                log_message(f"   ✅ {season_id}: {len(df):,} records ({TARGET_COLUMN}=1: {pos_count:,})")
        except Exception as e:
            log_message(f"   ⚠️  Error loading {season_id}: {e}")
            continue

    if not dfs:
        raise ValueError("No valid season files found!")

    # Combine all dataframes
    log_message(f"\n📊 Combining {len(dfs)} season datasets...")
    combined_df = pd.concat(dfs, ignore_index=True)

    log_message(f"✅ Combined dataset: {len(combined_df):,} records")
    log_message(f"   Total {TARGET_COLUMN}=1: {total_positive:,} ({total_positive/len(combined_df)*100:.4f}%)")
    
    return combined_df

def load_test_dataset():
    """Load test dataset (2025/26 season)"""
    if USE_LABELED_TIMELINES:
        test_file = TEST_DIR / ('timelines_35day_season_2025_2026' + LABELED_SUFFIX)
    else:
        test_file = TEST_DIR / 'timelines_35day_season_2025_2026_v4_muscular_test.csv'
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    log_message(f"\n📂 Loading test dataset: {test_file.name}")
    df_test = pd.read_csv(test_file, encoding='utf-8-sig', low_memory=False)
    
    if TARGET_COLUMN not in df_test.columns:
        raise ValueError(f"Test dataset missing {TARGET_COLUMN} column")

    pos_count = (df_test[TARGET_COLUMN] == 1).sum()

    log_message(f"✅ Test dataset: {len(df_test):,} records")
    log_message(f"   {TARGET_COLUMN}=1: {pos_count:,} ({pos_count/len(df_test)*100:.4f}%)")
    
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

def prepare_data(df, cache_file=None, use_cache=True, drop_first=True):
    """Prepare data with basic preprocessing (no feature selection) and optional caching.
    
    drop_first: If True (default), one-hot encoding drops first category (training convention).
                If False, no category is dropped so feature set matches production model (columns.json).
    """
    
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
        for feature in categorical_features:
            X_encoded[feature] = X_encoded[feature].fillna('Unknown')
            
            problematic_mask = X_encoded[feature].astype(str).str.contains(r'[:,\'"\\/;|&?!*+=@#$%^\s]', regex=True, na=False)
            problematic_count = problematic_mask.sum()
            if problematic_count > 0:
                problematic_values = X_encoded[feature][problematic_mask].unique()[:10]
                log_message(f"\n⚠️  Found {problematic_count} problematic values in '{feature}': {list(problematic_values)}")
            
            X_encoded[feature] = X_encoded[feature].apply(clean_categorical_value)
            
            dummies = pd.get_dummies(X_encoded[feature], prefix=feature, drop_first=drop_first)
            
            dummies.columns = [sanitize_feature_name(col) for col in dummies.columns]
            
            X_encoded = pd.concat([X_encoded, dummies], axis=1)
            X_encoded = X_encoded.drop(columns=[feature])
    
    # Fill numeric missing values
    if len(numeric_features) > 0:
        log_message(f"   Processing {len(numeric_features)} numeric features...")
        for feature in numeric_features:
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
    # For large datasets, avoid materializing all predictions at once
    if len(X) > 100000:
        chunk_size = 50000
        y_pred_list = []
        y_proba_list = []
        
        num_chunks = (len(X) + chunk_size - 1) // chunk_size
        for i in range(0, len(X), chunk_size):
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


def train_model_with_feature_subset(feature_subset, verbose=True, use_cache=None, algorithm='lgbm', hyperparameter_set='standard', return_datasets=False, use_full_train=False, exp10_data=False, test_negatives_before=None, use_single_thread=False, target_column='target1', exp12_data=False):
    """
    Train Model 1 (Muscular) or MSU model on pool data, test on 2025/26 holdout.

    Args:
        feature_subset: List of feature names to use for training
        verbose: Whether to print progress messages
        use_cache: If None, use USE_CACHE; else use this value for prepare_data cache
        algorithm: 'lgbm' (LightGBM) or 'gb' (sklearn GradientBoostingClassifier)
        exp10_data: If True, apply Exp 10 filter: keep all positives; drop negatives when [ref, ref+35] contains muscular/skeletal/unknown onset.
        exp12_data: If True (Exp 12), same as Exp 10 plus exclude all timelines (pool and test) when reference_date in [D, D+5] for muscular onset D.
        test_negatives_before: If set (e.g. '2025-11-01'), for test eval drop negatives with reference_date >= this date.
        hyperparameter_set: 'standard' (current), 'below' (more regularized), or 'above' (less regularized)
        return_datasets: If True, include X_val, y_val, X_test, y_test in returned dict (for threshold sweep)
        use_full_train: If True, use 100% of pool (train+val) for training; no validation set. For final production model.
        use_single_thread: If True, LGBM uses n_jobs=1 for deterministic, reproducible fit (e.g. when exporting best model).
        target_column: 'target1' (muscular only) or 'target_msu' (MSU [D-7,D-1]); used for labels and Exp10/test filters.

    Returns:
        Dictionary containing:
        - model: Trained muscular model
        - train_metrics: Metrics on train (80% or 100% per use_full_train)
        - val_metrics: Metrics on 20% validation (or placeholder when use_full_train=True)
        - test_metrics: Metrics on 2025/26 test
        - feature_names_used: List of features actually used (may be fewer than requested)
        - X_val, y_val, X_test, y_test: (only if return_datasets=True) validation and test features/labels
        - df_test_export: (only if return_datasets=True) DataFrame with columns player_id, reference_date for test rows (same order as X_test)
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

    # Exp 10 data: drop negatives when [ref, ref+35] contains muscular/skeletal/unknown onset. Exp 12 adds: drop all timelines when ref in [D, D+5] for muscular onset D.
    if exp10_data or exp12_data:
        if verbose:
            log_message("   Applying Exp 10 filter (negatives only if no muscular/skeletal/unknown onset in [ref, ref+35])...")
        df_pool['reference_date'] = pd.to_datetime(df_pool['reference_date'], errors='coerce')
        df_test_all['reference_date'] = pd.to_datetime(df_test_all['reference_date'], errors='coerce')
        from timelines.create_35day_timelines_v4_enhanced import load_injuries_data, build_exp10_onset_only_exclusion_set, build_ref_in_muscular_d_plus_5_exclusion_set
        injury_class_map = load_injuries_data(str(INJURIES_FILE))
        excl_set = build_exp10_onset_only_exclusion_set(injury_class_map)

        def _norm_ref(ts):
            t = pd.Timestamp(ts).normalize()
            if getattr(t, 'tz', None) is not None:
                t = t.tz_localize(None)
            return t

        def apply_exp10_filter(df):
            keep = []
            for i in range(len(df)):
                if df[target_column].iloc[i] == 1:
                    keep.append(True)
                    continue
                pid = int(df['player_id'].iloc[i])
                ref = df['reference_date'].iloc[i]
                if pd.isna(ref):
                    keep.append(False)
                    continue
                ref_n = _norm_ref(ref)
                keep.append((pid, ref_n) not in excl_set)
            return df.loc[keep].reset_index(drop=True)

        n_before_pool, n_before_test = len(df_pool), len(df_test_all)
        df_pool = apply_exp10_filter(df_pool)
        df_test_all = apply_exp10_filter(df_test_all)
        if verbose:
            log_message(f"   Pool: {n_before_pool:,} -> {len(df_pool):,} rows. Test: {n_before_test:,} -> {len(df_test_all):,} rows.")

        if exp12_data:
            ref_d5_set = build_ref_in_muscular_d_plus_5_exclusion_set(injury_class_map)
            def apply_ref_d5_filter(df):
                keep = []
                for i in range(len(df)):
                    pid = int(df['player_id'].iloc[i])
                    ref = df['reference_date'].iloc[i]
                    if pd.isna(ref):
                        keep.append(False)
                        continue
                    ref_n = _norm_ref(ref)
                    keep.append((pid, ref_n) not in ref_d5_set)
                return df.loc[keep].reset_index(drop=True)
            n_pool_before_d5, n_test_before_d5 = len(df_pool), len(df_test_all)
            df_pool = apply_ref_d5_filter(df_pool)
            df_test_all = apply_ref_d5_filter(df_test_all)
            if verbose:
                log_message("   Exp 12: excluded timelines with reference_date in [D, D+5] for muscular onset D.")
                log_message(f"   Pool: {n_pool_before_d5:,} -> {len(df_pool):,} rows. Test: {n_test_before_d5:,} -> {len(df_test_all):,} rows.")

        if test_negatives_before:
            cutoff = pd.Timestamp(test_negatives_before)
            ref = df_test_all['reference_date']
            keep_test = (df_test_all[target_column] == 1) | (ref < cutoff)
            n_drop = (~keep_test & (df_test_all[target_column] == 0)).sum()
            df_test_all = df_test_all.loc[keep_test].reset_index(drop=True)
            if verbose:
                log_message(f"   Test eval: dropped {int(n_drop):,} negatives with ref_date >= {test_negatives_before}; test rows: {len(df_test_all):,}")

    # Filter for model (target1 or target_msu)
    try:
        df_pool_muscular = filter_timelines_for_model(df_pool, target_column)
        df_test_muscular = filter_timelines_for_model(df_test_all, target_column)
    except Exception as e:
        log_error("Failed to filter timelines", e)
        raise

    df_test_muscular = df_test_muscular.reset_index(drop=True)
    _use_cache = USE_CACHE if use_cache is None else use_cache
    cache_suffix = hashlib.md5(str(sorted(feature_subset)).encode()).hexdigest()[:8]
    if exp10_data:
        cache_suffix += "_exp10"
    if exp12_data:
        cache_suffix += "_exp12"
    cache_prefix = "preprocessed_msu" if target_column == "target_msu" else "preprocessed_muscular"
    cache_test = str(CACHE_DIR / f'{cache_prefix}_test_subset_{cache_suffix}.csv')

    if use_full_train:
        # Use 100% of pool (train+val) for training; no validation set
        if verbose:
            log_message("   Using 100% of pool (train+val) for training (no validation split)...")
        df_train_full = df_pool_muscular.reset_index(drop=True)
        cache_train_full = str(CACHE_DIR / f'{cache_prefix}_full_train_subset_{cache_suffix}.csv')
        if verbose:
            log_message("   Preparing features for Model 1 (Muscular)...")
        try:
            X_train = prepare_data(df_train_full, cache_file=cache_train_full, use_cache=_use_cache)
            y_train = df_train_full[target_column].values
            X_test = prepare_data(df_test_muscular, cache_file=cache_test, use_cache=_use_cache)
            y_test = df_test_muscular[target_column].values
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
        cache_train = str(CACHE_DIR / f'{cache_prefix}_train80_subset_{cache_suffix}.csv')
        cache_val = str(CACHE_DIR / f'{cache_prefix}_val20_subset_{cache_suffix}.csv')
        if verbose:
            log_message("   Preparing features for Model 1 (Muscular)...")
        try:
            X_train = prepare_data(df_train_80, cache_file=cache_train, use_cache=_use_cache)
            y_train = df_train_80[target_column].values
            X_val = prepare_data(df_val_20, cache_file=cache_val, use_cache=_use_cache)
            y_val = df_val_20[target_column].values
            X_test = prepare_data(df_test_muscular, cache_file=cache_test, use_cache=_use_cache)
            y_test = df_test_muscular[target_column].values
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
            n_jobs = 1 if use_single_thread else -1
            model = LGBMClassifier(
                **hp,
                class_weight='balanced',
                random_state=42,
                n_jobs=n_jobs,
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

    _df_train = df_train_full if use_full_train else df_train_80
    exclude_cols = [
        'player_id', 'reference_date', 'date', 'player_name',
        'target1', 'target2', 'target', 'has_minimum_activity'
    ] + list(EXCLUDED_RAW_FEATURES)
    feature_cols_schema = [c for c in _df_train.columns if c not in exclude_cols]
    categorical_base_names = _df_train[feature_cols_schema].select_dtypes(include=['object']).columns.tolist()

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
        out['df_test_export'] = df_test_muscular[['player_id', 'reference_date']].copy()
        out['categorical_base_names'] = categorical_base_names
    return out


def train_and_evaluate_on_dataframes(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_list: list,
    *,
    cache_suffix: str = "default",
    use_cache: bool = True,
    verbose: bool = True,
    algorithm: str = "lgbm",
    hyperparameter_set: str = "standard",
) -> dict:
    """
    Train Model 1 (muscular) on in-memory train/test DataFrames and return test Gini.
    Used by labeling-experiment script to avoid reloading timelines; cache keys include
    cache_suffix and row counts so each experiment has its own cache.

    Args:
        df_train: DataFrame with target1 and all timeline columns.
        df_test: DataFrame with target1 and all timeline columns.
        feature_list: List of feature names (e.g. 500 from best iteration JSON).
        cache_suffix: Suffix for cache filenames (e.g. "exp1", "exp2").
        use_cache: Whether to use prepare_data cache.
        verbose: If True, log progress.
        algorithm: "lgbm" or "gb".
        hyperparameter_set: "standard", "below", "above", etc.

    Returns:
        Dict with keys: model, test_metrics, feature_names_used, test_gini (float).
    """
    if "target1" not in df_train.columns or "target1" not in df_test.columns:
        raise ValueError("Both df_train and df_test must have a 'target1' column")
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    df_train_m = filter_timelines_for_model(df_train.copy(), "target1")
    df_test_m = filter_timelines_for_model(df_test.copy(), "target1")
    df_test_m = df_test_m.reset_index(drop=True)

    _cache_suffix = hashlib.md5(str(sorted(feature_list)).encode()).hexdigest()[:8]
    cache_train = str(CACHE_DIR / f"preprocessed_muscular_exp_{cache_suffix}_train_{len(df_train_m)}_{_cache_suffix}.csv")
    cache_test = str(CACHE_DIR / f"preprocessed_muscular_exp_{cache_suffix}_test_{len(df_test_m)}_{_cache_suffix}.csv")

    if verbose:
        log_message(f"   Preparing features (train n={len(df_train_m):,}, test n={len(df_test_m):,})...")
    X_train = prepare_data(df_train_m, cache_file=cache_train, use_cache=use_cache)
    y_train = df_train_m["target1"].values
    X_test = prepare_data(df_test_m, cache_file=cache_test, use_cache=use_cache)
    y_test = df_test_m["target1"].values

    X_train, X_test = align_features_two(X_train, X_test)
    X_train, X_test = filter_features_two(X_train, X_test, feature_list)
    features_used = sorted(list(X_train.columns))

    if algorithm == "gb":
        hp = HP_PRESETS_GB.get(hyperparameter_set, GB_HP_STANDARD).copy()
        model = GradientBoostingClassifier(**hp, random_state=42)
    else:
        hp = HP_PRESETS_LGBM.get(hyperparameter_set, LGBM_HP_STANDARD).copy()
        model = LGBMClassifier(**hp, class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1)
    model.fit(X_train, y_train)
    test_metrics = evaluate_model(model, X_test, y_test, "Test")
    test_metrics = convert_numpy_types(test_metrics)
    gini = float(test_metrics.get("gini", 0.0))

    return {
        "model": model,
        "test_metrics": test_metrics,
        "feature_names_used": features_used,
        "test_gini": gini,
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

def run_iterative_training(resume=True, optimize_on=None, use_cache=None, algorithm='lgbm', features_per_iteration=None, initial_features=None, hp_preset=None, use_full_train=False, exp10_data=False, test_negatives_before=None, only_iteration=None, exp11_data=False, exp12_data=False, deploy_dir=None, model_type='muscular', only_iteration_chosen=False):
    """Main function to run iterative feature selection training.
    If resume=True and RESULTS_FILE exists, load state and continue from the next iteration.
    model_type: 'muscular' (target1), 'skeletal' (target2), or 'msu' (target_msu).
    optimize_on: 'validation' or 'test' - which metric to use for best iteration and early stopping.
    use_full_train: If True, use 100%% of training pool (no 80/20 split); optimize_on is forced to 'test'.
    use_cache: If None, use USE_CACHE; else use this value for prepare_data in train_model_with_feature_subset.
    algorithm: 'lgbm' or 'gb' (skeletal forces lgbm).
    features_per_iteration, initial_features: step size and first step; defaults from module constants if None.
    hp_preset: If set, use this hyperparameter preset (e.g. 'below_strong') for all iterations; else use 'standard'.
    exp10_data: If True, use D-7 labeled suffix, min_season 2020/21, and Exp 10 negative filter (onset-only in [ref, ref+35]).
    exp11_data: If True, use MSU labeled data (target_msu), same Exp10 negative filter, all seasons; no fixed-prefix; RANKING_FILE=exp11.
    exp12_data: If True (Exp 12), same as Exp 10 plus exclude timelines when ref in [D, D+5] for muscular onset D; no fixed-prefix; RANKING_FILE=exp12.
    test_negatives_before: If set (e.g. '2025-11-01'), drop test negatives with ref_date >= this for eval.
    only_iteration: If set (e.g. 20), run only this iteration number then exit. Uses same feature count and logic as that step. If results already have that iteration, re-runs and replaces it.
    only_iteration_chosen: When only_iteration is set, if True write all artifacts (model, features, deployment dir, reports) and update results file; if False run the iteration and print results only, without saving anything.
    deploy_dir: If set (Path or str), when saving from --only-iteration --chosen also write model and feature list here for deployment."""
    if optimize_on is None:
        optimize_on = OPTIMIZE_ON_DEFAULT
    if optimize_on not in ('validation', 'test'):
        raise ValueError(f"optimize_on must be 'validation' or 'test', got: {optimize_on}")
    if use_full_train and optimize_on == 'validation':
        log_message("   Note: use_full_train=True has no validation set; forcing optimize_on='test'")
        optimize_on = 'test'
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
    if model_type == 'skeletal':
        log_message("ITERATIVE FEATURE SELECTION TRAINING - MODEL 2 (SKELETAL) ONLY")
    else:
        log_message("ITERATIVE FEATURE SELECTION TRAINING - MODEL 1 (MUSCULAR) ONLY")
    log_message("="*80)
    log_message(f"\n📋 Configuration:")
    log_message(f"   Algorithm: {algorithm}")
    log_message(f"   Hyperparameter preset: {hyperparameter_set}")
    log_message(f"   Cache: {'disabled (preprocess from CSV)' if use_cache is False else 'enabled'}")
    if use_full_train:
        log_message(f"   Train data: 100% of pool (no validation split); optimize on test only")
    else:
        log_message(f"   Train/val split: {TRAIN_VAL_RATIO:.0%} train / {1-TRAIN_VAL_RATIO:.0%} validation (by timeline / random rows)")
    log_message(f"   Optimize on: {optimize_on} (combined score used for best iteration and early stopping)")
    log_message(f"   Features per iteration: {features_per_iteration}")
    log_message(f"   Initial features: {initial_features}")
    log_message(f"   Stop after: {CONSECUTIVE_DROPS_THRESHOLD} consecutive drops")
    log_message(f"   Performance metric: {GINI_WEIGHT} * Gini + {F1_WEIGHT} * F1-Score (Model 1 only)" if model_type != 'skeletal' else f"   Performance metric: {GINI_WEIGHT} * Gini + {F1_WEIGHT} * F1-Score (Model 2 only)")
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

    # Enforce exact 340-feature prefix from the stored best-iteration JSON so
    # iteration 17 (top 340) uses the same feature list and order. Skip for Exp 11 (MSU), Exp 12, and skeletal.
    # Skip when --only-iteration is set (e.g. 20): use ranking as-is for that iteration.
    if exp10_data and not exp11_data and not exp12_data and algorithm == 'lgbm' and only_iteration is None and model_type == 'muscular':
        if not BEST_FEATURES_PATH.exists():
            log_error(
                f"Exp10 fixed-prefix mode requires best features JSON: {BEST_FEATURES_PATH}"
            )
            return 1
        try:
            with open(BEST_FEATURES_PATH, 'r', encoding='utf-8') as f:
                best_meta = json.load(f)
            fixed_features = list(best_meta.get('features', []))
        except Exception as e:
            log_error(f"Failed to load fixed feature list from {BEST_FEATURES_PATH}", e)
            return 1

        if len(fixed_features) != FIXED_PREFIX_FEATURE_COUNT:
            log_error(
                f"Expected exactly {FIXED_PREFIX_FEATURE_COUNT} fixed features in "
                f"{BEST_FEATURES_PATH.name}, found {len(fixed_features)}"
            )
            return 1

        missing_in_ranking = [f for f in fixed_features if f not in ranked_features]
        if missing_in_ranking:
            log_message(
                f"   ⚠️  {len(missing_in_ranking)} fixed-prefix features are not in "
                f"{RANKING_FILE.name}; they will still be kept in the first "
                f"{FIXED_PREFIX_FEATURE_COUNT} positions."
            )

        fixed_set = set(fixed_features)
        ranked_features = fixed_features + [f for f in ranked_features if f not in fixed_set]
        log_message(
            f"   Applied fixed feature prefix from {BEST_FEATURES_PATH.name}: "
            f"first {FIXED_PREFIX_FEATURE_COUNT} features are locked."
        )
    
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
            if cfg.get('hyperparameter_preset') is not None and hp_preset is None:
                hyperparameter_set = cfg['hyperparameter_preset']
            if cfg.get('use_full_train') is not None:
                use_full_train = bool(cfg['use_full_train'])
                log_message(f"Resuming: using saved use_full_train={use_full_train}")
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
                'use_full_train': use_full_train,
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
                'fixed_prefix_source': BEST_FEATURES_PATH.name if (exp10_data and not exp11_data and not exp12_data and algorithm == 'lgbm') else None,
                'fixed_prefix_feature_count': FIXED_PREFIX_FEATURE_COUNT if (exp10_data and not exp11_data and not exp12_data and algorithm == 'lgbm') else None,
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

    if only_iteration is not None:
        iteration = only_iteration - 1
        if only_iteration_chosen:
            log_message(f"Only running iteration {only_iteration} (--only-iteration); this run is the chosen model: artifacts and deployment outputs will be written.")
        else:
            log_message(f"Only running iteration {only_iteration} (--only-iteration); experiment only: results will not be stored and no artifacts will be written.")

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

        # Skip already-completed iterations when resuming (unless re-running only this one)
        if iteration <= len(results['iterations']) and (only_iteration is None or iteration != only_iteration):
            log_message(f"Resuming: skipping iteration {iteration} (already completed)")
            continue

        # Check if we've used all features
        if n_features_requested > len(ranked_features):
            log_message(f"\n✅ Reached maximum number of features ({len(ranked_features)})")
            break
        
        # Select feature subset
        feature_subset = ranked_features[:n_features_requested]
        
        log_message(f"\n{'='*80}")
        model_label = "Model 2" if model_type == 'skeletal' else "Model 1"
        log_message(f"ITERATION {iteration}: Training {model_label} with top {n_features_requested} features (requested)")
        log_message(f"{'='*80}")
        log_message(f"   Features: Top {n_features_requested} from ranked list")
        
        iteration_start = datetime.now()
        
        try:
            log_message(f"Starting training for iteration {iteration} with {n_features_requested} requested features")
            target_column = 'target2' if model_type == 'skeletal' else ('target_msu' if exp11_data else 'target1')
            # Train model with this feature subset
            training_results = train_model_with_feature_subset(
                feature_subset,
                verbose=True,
                use_cache=use_cache,
                algorithm=algorithm,
                hyperparameter_set=hyperparameter_set,
                use_full_train=use_full_train,
                exp10_data=exp10_data or exp11_data or exp12_data,
                test_negatives_before=test_negatives_before,
                target_column=target_column,
                exp12_data=exp12_data,
                return_datasets=(only_iteration is not None and only_iteration_chosen),
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
            
            existing_idx = (next((i for i, it in enumerate(results['iterations']) if it['iteration'] == iteration), None)
                           if only_iteration is not None else None)
            if only_iteration is None or only_iteration_chosen:
                if existing_idx is not None:
                    combined_scores[existing_idx] = combined_score
                else:
                    combined_scores.append(combined_score)
            log_message(f"   Validation combined score: {combined_score_val:.4f}")
            log_message(f"   Test combined score: {combined_score_test:.4f}")
            log_message(f"   Using for selection (optimize_on={optimize_on}): {combined_score:.4f}")
            
            # Check if this is the best so far (use n_features_used for best count)
            if only_iteration is None or only_iteration_chosen:
                if combined_score > best_score and existing_idx is None:
                    best_score = combined_score
                    best_iteration = iteration
                    best_n_features = n_features_used
                    log_message(f"New best score! Iteration {iteration} with {n_features_used} features used: {best_score:.4f}")
            
            # Store iteration results: real counts to avoid 759 vs 760 confusion
            # Use model2_skeletal key for skeletal, model1_muscular for muscular/msu
            metrics_key = 'model2_skeletal' if model_type == 'skeletal' else 'model1_muscular'
            iteration_data = {
                'iteration': iteration,
                'n_features_requested': n_features_requested,
                'n_features_used': n_features_used,
                'n_features': n_features_used,
                'features': training_results['feature_names_used'],
                metrics_key: {
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
            
            if only_iteration is None or only_iteration_chosen:
                if existing_idx is not None:
                    results['iterations'][existing_idx] = iteration_data
                    # Recompute best from full results after replace
                    combined_scores_list = [it['combined_score'] for it in results['iterations']]
                    best_score = max(combined_scores_list)
                    best_idx = combined_scores_list.index(best_score)
                    best_iteration = results['iterations'][best_idx]['iteration']
                    best_n_features = results['iterations'][best_idx]['n_features_used']
                else:
                    results['iterations'].append(iteration_data)
            
            # Print iteration summary
            log_message(f"\n📊 Iteration {iteration} Results:")
            log_message(f"   Features: requested={n_features_requested}, used={n_features_used}")
            if use_full_train:
                log_message(f"   Train (100%): Gini={training_results['train_metrics']['gini']:.4f}, "
                          f"F1={training_results['train_metrics']['f1']:.4f} -> combined={combined_score_val:.4f}")
            else:
                log_message(f"   Validation: Gini={training_results['val_metrics']['gini']:.4f}, "
                          f"F1={training_results['val_metrics']['f1']:.4f} -> combined={combined_score_val:.4f}")
            log_message(f"   Test:      Gini={training_results['test_metrics']['gini']:.4f}, "
                      f"F1={training_results['test_metrics']['f1']:.4f} -> combined={combined_score_test:.4f}")
            log_message(f"   Best selection (optimize_on={optimize_on}): {combined_score:.4f}")
            log_message(f"   Training time: {iteration_data['training_time_seconds']:.1f} seconds")
            
            # When --only-iteration and not --chosen: experiment only, do not store or write artifacts
            if only_iteration is not None and iteration == only_iteration and not only_iteration_chosen:
                log_message("   Experiment only: results not stored, no artifacts written.")
                break
            
            # Check for consecutive drops
            if only_iteration is None or only_iteration_chosen:
                if has_consecutive_drops(combined_scores, 
                                    threshold=PERFORMANCE_DROP_THRESHOLD,
                                    consecutive_count=CONSECUTIVE_DROPS_THRESHOLD):
                    log_message(f"\n⚠️  Detected {CONSECUTIVE_DROPS_THRESHOLD} consecutive drops in performance!")
                    log_message(f"   Stopping iterative training.")
                    break
            
            # Save intermediate results (after each iteration)
            if only_iteration is None or only_iteration_chosen:
                log_message(f"Saving intermediate results to: {RESULTS_FILE}")
                try:
                    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2)
                    log_message("Intermediate results saved successfully")
                except Exception as e:
                    log_error(f"Failed to save intermediate results", e)
            
            if only_iteration is not None and iteration == only_iteration and only_iteration_chosen:
                # Save the model we just trained so we preserve this run's test Gini (e.g. 0.589)
                if model_type == 'skeletal':
                    suffix = '_exp12' if exp12_data else ''
                    artifact_base = 'lgbm_skeletal_best_iteration'
                    deployment_dir = MODEL_OUTPUT_DIR.parent / 'model_skeletal'
                    deploy_model_arg = 'skeletal'
                else:
                    suffix = '_gb' if algorithm == 'gb' else ''
                    if exp10_data and not exp12_data:
                        suffix = '_exp10'
                    if exp11_data:
                        suffix = '_exp11'
                    if exp12_data:
                        suffix = '_gb_exp12' if (exp12_data and algorithm == 'gb') else '_exp12'
                    artifact_base = 'lgbm_muscular_best_iteration'
                    # Deployment folder: one per model type (LGBM muscular, GB muscular, MSU)
                    if exp11_data:
                        deployment_dir = MODEL_OUTPUT_DIR.parent / 'model_msu_lgbm'
                        deploy_model_arg = 'msu_lgbm'
                    elif algorithm == 'gb':
                        deployment_dir = MODEL_OUTPUT_DIR.parent / 'model_muscular_gb'
                        deploy_model_arg = 'muscular_gb'
                    else:
                        deployment_dir = MODEL_OUTPUT_DIR.parent / 'model_muscular_lgbm'
                        deploy_model_arg = 'muscular_lgbm'
                deployment_dir.mkdir(parents=True, exist_ok=True)
                model_path = MODEL_OUTPUT_DIR / f'{artifact_base}{suffix}.joblib'
                features_path = MODEL_OUTPUT_DIR / f'{artifact_base}_features{suffix}.json'
                MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                joblib.dump(training_results['model'], model_path)
                meta = {
                    'n_features': n_features_used,
                    'algorithm': algorithm,
                    'iteration': iteration,
                    'optimize_on': optimize_on,
                    'hyperparameter_preset': hyperparameter_set,
                    'trained_on_full_data': use_full_train,
                    'combined_score_test': float(iteration_data['combined_score_test']),
                    'combined_score': float(iteration_data['combined_score']),
                    'features': training_results['feature_names_used'],
                    'train_metrics': training_results['train_metrics'],
                    'val_metrics': training_results['val_metrics'] if not use_full_train else None,
                    'test_metrics': training_results['test_metrics'],
                }
                if not use_full_train:
                    meta['combined_score_val'] = float(iteration_data['combined_score_val'])
                with open(features_path, 'w', encoding='utf-8') as f:
                    json.dump(meta, f, indent=2)
                log_message(f"Saved model (from --only-iteration {only_iteration}) to: {model_path}")
                log_message(f"Saved feature list to: {features_path} (n_features={n_features_used}, test Gini={training_results['test_metrics']['gini']:.4f})")
                # Export test predictions for comparison with production (same format as evaluate_production_lgbm_on_new_test.py)
                if 'X_test' in training_results and 'df_test_export' in training_results:
                    proba_test = training_results['model'].predict_proba(training_results['X_test'])[:, 1]
                    export_df = training_results['df_test_export'].copy()
                    export_df['predicted_probability'] = proba_test
                    label_col = 'target2' if model_type == 'skeletal' else ('target_msu' if deploy_model_arg == 'msu_lgbm' else 'target1')
                    export_df[label_col] = training_results['y_test']
                    export_path = deployment_dir / "test_predictions_from_training_pipeline.csv"
                    export_df.to_csv(export_path, index=False)
                    log_message(f"Exported test predictions to: {export_path}")
                    log_message(f"   Rows: {len(export_df):,} (player_id, reference_date, predicted_probability, {label_col})")
                if deploy_model_arg == 'msu_lgbm' and 'categorical_base_names' in training_results:
                    feature_names_used = training_results['feature_names_used']
                    encoding_schema = {}
                    for base in training_results['categorical_base_names']:
                        cols = [c for c in feature_names_used if c.startswith(base + "_")]
                        if cols:
                            encoding_schema[base] = sorted(cols)
                    if encoding_schema:
                        schema_path = deployment_dir / "encoding_schema.json"
                        with open(schema_path, 'w', encoding='utf-8') as f:
                            json.dump(encoding_schema, f, indent=2)
                        log_message(f"Exported encoding_schema.json to: {schema_path} ({len(encoding_schema)} categoricals)")
                # Write deployment guidelines for the deployment agent (sections 1-3 only; no validation step)
                guidelines_path = deployment_dir / "DEPLOYMENT_GUIDELINES.md"
                suffix_display = suffix if suffix else ""
                label_col = 'target2' if model_type == 'skeletal' else ('target_msu' if deploy_model_arg == 'msu_lgbm' else 'target1')
                if deploy_model_arg == 'skeletal':
                    step2_instructions = f"""- `{artifact_base}{suffix_display}.joblib` → `{artifact_base}.joblib`
- `{artifact_base}_features{suffix_display}.json` → `{artifact_base}_features.json`

(Artifacts live in `models_production/lgbm_muscular_v4/models/`; copy to expected names if using a suffix.)"""
                elif deploy_model_arg == 'msu_lgbm':
                    step2_instructions = "Artifacts are `lgbm_muscular_best_iteration_exp11.joblib` and `lgbm_muscular_best_iteration_features_exp11.json`. The deploy script reads these directly; no copy needed. Proceed to step 3."
                elif suffix:
                    step2_instructions = f"""- `lgbm_muscular_best_iteration{suffix}.joblib` → `lgbm_muscular_best_iteration.joblib`
- `lgbm_muscular_best_iteration_features{suffix}.json` → `lgbm_muscular_best_iteration_features.json`

(Replace SUFFIX with the actual suffix used in this run: `{suffix}`.)"""
                else:
                    step2_instructions = "Artifacts already have the expected non-suffixed names; no copy needed. Proceed to step 3."
                if deploy_model_arg == 'skeletal':
                    artifact_desc = f"  - `{artifact_base}{suffix_display}.joblib`\n  - `{artifact_base}_features{suffix_display}.json`"
                else:
                    artifact_desc = f"  - `lgbm_muscular_best_iteration{suffix_display}.joblib`\n  - `lgbm_muscular_best_iteration_features{suffix_display}.json`"
                guidelines_content = f"""# Deployment guidelines – {deployment_dir.name} (iteration {only_iteration})

Use these steps when deploying the model so production predictions match the training pipeline.

## 1. Training (already done)

- Model was trained with `--only-iteration {only_iteration}` and the selected options (e.g. Exp 10, D-7, train ≥ 2020/21, test negatives with ref_date before 2025-11-01).
- Artifacts produced in `models_production/lgbm_muscular_v4/models/`:
{artifact_desc}
- Test predictions from this training run have been exported to:
  - `models_production/lgbm_muscular_v4/{deployment_dir.name}/test_predictions_from_training_pipeline.csv`
  - Columns: `player_id`, `reference_date`, `predicted_probability`, `{label_col}`
  - Use this file to validate that production predictions match the training pipeline.

## 2. Prepare artifacts for the deploy script

The deploy script expects the correct artifact names. In `models_production/lgbm_muscular_v4/models/`:

{step2_instructions}

## 3. Deploy to production

From the repository root:

```
python models_production/lgbm_muscular_v4/code/modeling/deploy_gb_to_production.py --model {deploy_model_arg}
```

This refreshes `models_production/lgbm_muscular_v4/{deployment_dir.name}/` (e.g. `model.joblib`, `columns.json`, `MODEL_METADATA.json`).
"""
                guidelines_path.write_text(guidelines_content, encoding="utf-8")
                log_message(f"Exported deployment guidelines to: {guidelines_path}")
                # Selected model report: human-readable summary of chosen iteration and artifacts
                report_path = deployment_dir / "SELECTED_MODEL_REPORT.md"
                report_lines = [
                    "# Selected model report",
                    "",
                    f"**Model type:** {deploy_model_arg}",
                    f"**Source:** `--only-iteration {only_iteration}`",
                    "",
                    "## Selected iteration",
                    f"- Iteration: {iteration}",
                    f"- Features used: {n_features_used}",
                    f"- Optimize on: {optimize_on}",
                    "",
                    "## Performance (this run)",
                    f"- Train Gini: {training_results['train_metrics']['gini']:.4f}",
                    f"- Train F1: {training_results['train_metrics']['f1']:.4f}",
                ]
                if iteration_data.get('combined_score_val') is not None:
                    report_lines.append(f"- Validation combined score: {iteration_data['combined_score_val']:.4f}")
                report_lines.extend([
                    f"- Test Gini: {training_results['test_metrics']['gini']:.4f}",
                    f"- Test F1: {training_results['test_metrics']['f1']:.4f}",
                    f"- Test combined score: {iteration_data['combined_score_test']:.4f}",
                    "",
                    "## Artifacts",
                    f"- Model: `models_production/lgbm_muscular_v4/models/{artifact_base}{suffix_display}.joblib`",
                    f"- Features: `models_production/lgbm_muscular_v4/models/{artifact_base}_features{suffix_display}.json`",
                    f"- Test predictions: `models_production/lgbm_muscular_v4/{deployment_dir.name}/test_predictions_from_training_pipeline.csv` (label column: `{label_col}`)",
                    "",
                    "## Deploy",
                    "```",
                    f"python models_production/lgbm_muscular_v4/code/modeling/deploy_gb_to_production.py --model {deploy_model_arg}",
                    "```",
                    ""
                ])
                report_path.write_text("\n".join(report_lines), encoding="utf-8")
                log_message(f"Exported selected model report to: {report_path}")
                if deploy_dir is not None:
                    deploy_path = Path(deploy_dir) if isinstance(deploy_dir, str) else deploy_dir
                    deploy_path.mkdir(parents=True, exist_ok=True)
                    deploy_suffix = suffix  # same as model/features (e.g. _exp12)
                    deploy_model = deploy_path / f'{artifact_base}{deploy_suffix}.joblib'
                    deploy_features = deploy_path / f'{artifact_base}_features{deploy_suffix}.json'
                    joblib.dump(training_results['model'], deploy_model)
                    with open(deploy_features, 'w', encoding='utf-8') as f:
                        json.dump(meta, f, indent=2)
                    log_message(f"Deployment copy: {deploy_model}, {deploy_features}")
                log_message(f"Finished running only iteration {only_iteration}; exiting.")
                break
            
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
        metrics_key = 'model2_skeletal' if model_type == 'skeletal' else 'model1_muscular'
        best_metrics = best_iter_data.get(metrics_key, best_iter_data.get('model1_muscular'))
        log_message(f"\n📈 Best Performance (Iteration {best_iteration}, optimize_on={optimize_on}):")
        if best_metrics and best_metrics.get('val'):
            log_message(f"   Validation: Gini={best_metrics['val']['gini']:.4f}, "
                       f"F1={best_metrics['val']['f1']:.4f} -> combined={best_iter_data['combined_score_val']:.4f}")
        log_message(f"   Test:      Gini={best_metrics['test']['gini']:.4f}, "
                   f"F1={best_metrics['test']['f1']:.4f} -> combined={best_iter_data['combined_score_test']:.4f}")
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


def export_best_model(use_cache=None, algorithm='lgbm', export_iteration=None, hp_preset=None, train_on_full_data=False, exp10_data=False, test_negatives_before=None, exp11_data=False, exp12_data=False, model_type='muscular'):
    """
    Load iterative results, find the best iteration by combined_score (val or test per optimize_on),
    or use a specific iteration if export_iteration is set. Re-train that model and save it (joblib)
    plus feature list (JSON).
    use_cache: If None, use USE_CACHE; else use this value for prepare_data.
    algorithm: 'lgbm' or 'gb' - must match the results file (use same as training run). Skeletal uses lgbm only.
    export_iteration: If set, export this iteration number (e.g. 6 for 300 features); else export best by score.
    hp_preset: If set, use this hyperparameter preset (e.g. 'below_strong'); else use 'standard'.
    train_on_full_data: If True, train on 100% of pool (train+val); no validation set. For final production model.
    exp10_data: If True, use Exp 10 data (same as iterative run with --exp10-data); must match RESULTS_FILE.
    exp11_data: If True, use Exp 11 (MSU) data; export to exp11 artifact names and use target_msu + Exp10 filter.
    exp12_data: If True (Exp 12), use Exp 10 + ref in [D,D+5] exclusion; export to exp12 artifact names.
    test_negatives_before: If set (e.g. '2025-11-01'), drop test negatives with ref_date >= this when evaluating.
    model_type: 'muscular', 'skeletal', or 'msu'. When 'skeletal', uses target2 and writes to lgbm_skeletal_best_iteration*
    and model_skeletal/ (test predictions + DEPLOYMENT_GUIDELINES).
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
    # For skeletal, best iteration is stored under model2_skeletal; selection score is still combined_score
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
    target_column = 'target2' if model_type == 'skeletal' else ('target_msu' if exp11_data else 'target1')
    # Use cache so the same preprocessed data (and row order) as the iterative run is used (same feature list => same cache key).
    # Use same n_jobs as iterative run (multi-thread) to match iteration-20 training conditions.
    if use_cache and algorithm == 'lgbm':
        log_message("   Using cache + multi-thread fit (same as iterative run).")
    return_datasets_for_export = True  # Request X_test/df_test_export for CSV + deployment outputs for all model types
    training_results = train_model_with_feature_subset(
        feature_list,
        verbose=True,
        use_cache=use_cache,
        algorithm=algorithm,
        hyperparameter_set=hyperparameter_set,
        use_full_train=train_on_full_data,
        exp10_data=exp10_data or exp11_data or exp12_data,
        test_negatives_before=test_negatives_before,
        use_single_thread=False,
        target_column=target_column,
        exp12_data=exp12_data,
        return_datasets=return_datasets_for_export,
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
    if model_type == 'skeletal':
        suffix = '_exp12' if exp12_data else ''
        artifact_base = 'lgbm_skeletal_best_iteration'
        deployment_dir = MODEL_OUTPUT_DIR.parent / 'model_skeletal'
        deploy_model_arg = 'skeletal'
    else:
        suffix = '_gb' if algorithm == 'gb' else ''
        if exp12_data:
            suffix = '_gb_exp12' if (exp12_data and algorithm == 'gb') else '_exp12'
        elif exp11_data:
            suffix = '_exp11'
        elif exp10_data:
            suffix = '_exp10'  # Do not overwrite lgbm_muscular_best_iteration.* or lgbm_muscular_exp10_labeling.joblib
        artifact_base = 'lgbm_muscular_best_iteration'
        if exp11_data:
            deployment_dir = MODEL_OUTPUT_DIR.parent / 'model_msu_lgbm'
            deploy_model_arg = 'msu_lgbm'
        elif algorithm == 'gb':
            deployment_dir = MODEL_OUTPUT_DIR.parent / 'model_muscular_gb'
            deploy_model_arg = 'muscular_gb'
        else:
            deployment_dir = MODEL_OUTPUT_DIR.parent / 'model_muscular_lgbm'
            deploy_model_arg = 'muscular_lgbm'
    model_path = MODEL_OUTPUT_DIR / f'{artifact_base}{suffix}.joblib'
    features_path = MODEL_OUTPUT_DIR / f'{artifact_base}_features{suffix}.json'
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
        'features': training_results['feature_names_used'],
        # Full metric snapshots for this exported model
        'train_metrics': training_results['train_metrics'],
        'val_metrics': training_results['val_metrics'] if not train_on_full_data else None,
        'test_metrics': training_results['test_metrics'],
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
    # Export test predictions, deployment guidelines, and selected-model report to deployment dir (all model types)
    if deployment_dir is not None and return_datasets_for_export:
        deployment_dir.mkdir(parents=True, exist_ok=True)
        label_col = 'target2' if deploy_model_arg == 'skeletal' else ('target_msu' if deploy_model_arg == 'msu_lgbm' else 'target1')
        if 'X_test' in training_results and 'df_test_export' in training_results:
            proba_test = model.predict_proba(training_results['X_test'])[:, 1]
            export_df = training_results['df_test_export'].copy()
            export_df['predicted_probability'] = proba_test
            export_df[label_col] = training_results['y_test']
            export_path = deployment_dir / "test_predictions_from_training_pipeline.csv"
            export_df.to_csv(export_path, index=False)
            log_message(f"Exported test predictions to: {export_path}")
            log_message(f"   Rows: {len(export_df):,} (player_id, reference_date, predicted_probability, {label_col})")
        if deploy_model_arg == 'msu_lgbm' and 'categorical_base_names' in training_results:
            feature_names_used = training_results['feature_names_used']
            encoding_schema = {}
            for base in training_results['categorical_base_names']:
                cols = [c for c in feature_names_used if c.startswith(base + "_")]
                if cols:
                    encoding_schema[base] = sorted(cols)
            if encoding_schema:
                schema_path = deployment_dir / "encoding_schema.json"
                with open(schema_path, 'w', encoding='utf-8') as f:
                    json.dump(encoding_schema, f, indent=2)
                log_message(f"Exported encoding_schema.json to: {schema_path} ({len(encoding_schema)} categoricals)")
        suffix_display = suffix if suffix else ""
        if deploy_model_arg == 'skeletal':
            step2 = f"- `{artifact_base}{suffix_display}.joblib` → `{artifact_base}.joblib`\n- `{artifact_base}_features{suffix_display}.json` → `{artifact_base}_features.json`"
            artifact_desc = f"  - `{artifact_base}{suffix_display}.joblib`\n  - `{artifact_base}_features{suffix_display}.json`"
        elif deploy_model_arg == 'msu_lgbm':
            step2 = "Artifacts are `lgbm_muscular_best_iteration_exp11.joblib` and `lgbm_muscular_best_iteration_features_exp11.json`. The deploy script reads these directly; no copy needed. Proceed to step 3."
            artifact_desc = f"  - `{artifact_base}{suffix_display}.joblib`\n  - `{artifact_base}_features{suffix_display}.json`"
        elif suffix_display:
            step2 = f"""- `lgbm_muscular_best_iteration{suffix_display}.joblib` → `lgbm_muscular_best_iteration.joblib`
- `lgbm_muscular_best_iteration_features{suffix_display}.json` → `lgbm_muscular_best_iteration_features.json`

(Replace suffix with the actual one used: `{suffix_display}`.)"""
            artifact_desc = f"  - `lgbm_muscular_best_iteration{suffix_display}.joblib`\n  - `lgbm_muscular_best_iteration_features{suffix_display}.json`"
        else:
            step2 = "Artifacts already have the expected non-suffixed names; no copy needed. Proceed to step 3."
            artifact_desc = f"  - `lgbm_muscular_best_iteration.joblib`\n  - `lgbm_muscular_best_iteration_features.json`"
        guidelines_path = deployment_dir / "DEPLOYMENT_GUIDELINES.md"
        guidelines_content = f"""# Deployment guidelines – {deployment_dir.name} (export-best)

Use these steps when deploying the model so production predictions match the training pipeline.

## 1. Training (already done)

- Model was exported with `--export-best` (best iteration from iterative results, model type: {deploy_model_arg}).
- Artifacts produced in `models_production/lgbm_muscular_v4/models/`:
{artifact_desc}
- Test predictions have been exported to:
  - `models_production/lgbm_muscular_v4/{deployment_dir.name}/test_predictions_from_training_pipeline.csv`
  - Columns: `player_id`, `reference_date`, `predicted_probability`, `{label_col}`

## 2. Prepare artifacts for the deploy script

In `models_production/lgbm_muscular_v4/models/`:

{step2}

## 3. Deploy to production

From the repository root:

```
python models_production/lgbm_muscular_v4/code/modeling/deploy_gb_to_production.py --model {deploy_model_arg}
```

This refreshes `models_production/lgbm_muscular_v4/{deployment_dir.name}/` (e.g. `model.joblib`, `columns.json`, `MODEL_METADATA.json`).
"""
        guidelines_path.write_text(guidelines_content, encoding="utf-8")
        log_message(f"Exported deployment guidelines to: {guidelines_path}")
        # Selected model report for export-best
        report_path = deployment_dir / "SELECTED_MODEL_REPORT.md"
        report_lines = [
            "# Selected model report",
            "",
            f"**Model type:** {deploy_model_arg}",
            "**Source:** `--export-best`",
            "",
            "## Selected iteration",
            f"- Iteration: {best_it['iteration']}",
            f"- Features used: {n_features_used}",
            f"- Optimize on: {optimize_on}",
            "",
            "## Performance (this run)",
            f"- Train Gini: {training_results['train_metrics']['gini']:.4f}",
            f"- Train F1: {training_results['train_metrics']['f1']:.4f}",
        ]
        if combined_score_val is not None:
            report_lines.append(f"- Validation combined score: {combined_score_val:.4f}")
        report_lines.extend([
            f"- Test Gini: {training_results['test_metrics']['gini']:.4f}",
            f"- Test F1: {training_results['test_metrics']['f1']:.4f}",
            f"- Test combined score: {combined_score_test:.4f}",
            "",
            "## Artifacts",
            f"- Model: `models_production/lgbm_muscular_v4/models/{artifact_base}{suffix_display}.joblib`",
            f"- Features: `models_production/lgbm_muscular_v4/models/{artifact_base}_features{suffix_display}.json`",
            f"- Test predictions: `models_production/lgbm_muscular_v4/{deployment_dir.name}/test_predictions_from_training_pipeline.csv` (label column: `{label_col}`)",
            "",
            "## Deploy",
            "```",
            f"python models_production/lgbm_muscular_v4/code/modeling/deploy_gb_to_production.py --model {deploy_model_arg}",
            "```",
            ""
        ])
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        log_message(f"Exported selected model report to: {report_path}")
    return 0


# Best model artifacts (for evaluate-best-on-test)
BEST_MODEL_PATH = MODEL_OUTPUT_DIR / 'lgbm_muscular_best_iteration.joblib'
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


def train_500_on_labeled_datasets(use_cache=None, save_model=True):
    """
    Retrain the 500-feature LGBM model on the new labeled timeline datasets
    (train: *_v4_labeled.csv, test: timelines_35day_season_2025_2026_v4_labeled.csv).
    Report test Gini and combined score; optionally save the new model under a
    distinct name (lgbm_muscular_500_labeled.*) so the previous best is preserved.
    """
    global USE_LABELED_TIMELINES
    if use_cache is None:
        use_cache = USE_CACHE

    if not BEST_FEATURES_PATH.exists():
        log_error(f"Best features JSON not found: {BEST_FEATURES_PATH}. Export the 500-feature list first.")
        return 1

    with open(BEST_FEATURES_PATH, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    feature_list = meta['features']
    prev_gini = meta.get('test_metrics', {}).get('gini') or meta.get('combined_score_test')
    log_message("\n" + "=" * 80)
    log_message("RETRAIN 500-FEATURE LGBM ON LABELED TIMELINES (train + test)")
    log_message("=" * 80)
    log_message(f"   Features: {len(feature_list)} (from {BEST_FEATURES_PATH.name})")
    log_message(f"   Train: *_v4_labeled.csv (excl. 2025_2026)  |  Test: 2025_2026_v4_labeled.csv")
    if prev_gini is not None:
        log_message(f"   Previous test Gini (old data): {prev_gini:.4f}")
    log_message("=" * 80 + "\n")

    USE_LABELED_TIMELINES = True
    try:
        result = train_model_with_feature_subset(
            feature_list,
            verbose=True,
            use_cache=use_cache,
            algorithm='lgbm',
            hyperparameter_set='standard',
            use_full_train=True,
        )
    finally:
        USE_LABELED_TIMELINES = False

    test_metrics = result['test_metrics']
    gini = test_metrics.get('gini')
    combined = calculate_combined_score(
        test_metrics,
        gini_weight=GINI_WEIGHT,
        f1_weight=F1_WEIGHT
    )

    log_message("\n" + "=" * 80)
    log_message("RETRAIN RESULTS (LABELED DATA)")
    log_message("=" * 80)
    log_message(f"   Test Gini:        {gini:.4f}" + (f"  (previous: {prev_gini:.4f})" if prev_gini is not None else ""))
    log_message(f"   Combined score:   {combined:.4f}")
    log_message("=" * 80 + "\n")

    if save_model and result.get('model') is not None:
        labeled_model_path = MODEL_OUTPUT_DIR / 'lgbm_muscular_500_labeled.joblib'
        labeled_features_path = MODEL_OUTPUT_DIR / 'lgbm_muscular_500_labeled_features.json'
        MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(result['model'], labeled_model_path)
        meta_labeled = {
            'n_features': len(feature_list),
            'algorithm': 'lgbm',
            'data': 'labeled_timelines',
            'test_gini': gini,
            'combined_score_test': combined,
            'features': result['feature_names_used'],
        }
        with open(labeled_features_path, 'w', encoding='utf-8') as f:
            json.dump(meta_labeled, f, indent=2)
        log_message(f"   Saved: {labeled_model_path.name}, {labeled_features_path.name}\n")
    return 0


def run_hyperparameter_test(
    algorithm='lgbm',
    use_cache=None,
    presets_list=None,
    output_suffix=None,
    exp10_data=False,
    test_negatives_before=None,
):
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
        # For GB, allow an Exp 10-specific HP test on the 340-feature Exp 10 set
        # when exp10_data is enabled; otherwise fall back to the legacy 300-feature GB set.
        if exp10_data:
            features_path = MODEL_OUTPUT_DIR / 'lgbm_muscular_best_iteration_features.json'
            out_base = 'hyperparameter_test_gb_exp10'
            n_feat_label = None  # will be set from feature_list length
        else:
            features_path = MODEL_OUTPUT_DIR / 'lgbm_muscular_best_iteration_features_gb.json'
            out_base = 'hyperparameter_test_gb_300'
            n_feat_label = '300'
    else:
        features_path = MODEL_OUTPUT_DIR / 'lgbm_muscular_best_iteration_features.json'
        # Use the actual feature count from meta for logging/labels (e.g. 340 for iteration 17)
        out_base = 'hyperparameter_test_lgbm'
        n_feat_label = None
    out_json = MODEL_OUTPUT_DIR / f"{out_base}{output_suffix or ''}.json"
    if not features_path.exists():
        log_error(f"Features file not found: {features_path}. Run --algorithm {algorithm} --export-best first.")
        return 1
    with open(features_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    feature_list = meta['features']
    if n_feat_label is None:
        n_feat_label = str(len(feature_list))
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
                feature_list,
                verbose=True,
                use_cache=use_cache,
                algorithm=algorithm,
                hyperparameter_set=preset,
                exp10_data=exp10_data,
                test_negatives_before=test_negatives_before,
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
                        help="Use 100%% of training pool (no 80/20 split). For iterative run: forces optimize_on=test. For --export-best: train final model on full data.")
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
    parser.add_argument("--train-500-on-labeled", action="store_true",
                        help="Retrain the 500-feature LGBM on labeled timelines (train + test *_v4_labeled.csv), report test Gini, save as lgbm_muscular_500_labeled.*")
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
    parser.add_argument("--exp10-data", action="store_true",
                        help="Use Exp 10 setup: D-7 labeled data, train seasons >= 2020/21 (excl. 2018/19, 2019/20), negatives only if no muscular/skeletal/unknown onset in [ref, ref+35]")
    parser.add_argument("--test-negatives-before", type=str, default=None,
                        help="With --exp10-data: drop test negatives with reference_date >= this date (YYYY-MM-DD) for eval (e.g. 2025-11-01)")
    parser.add_argument("--exp11-data", action="store_true",
                        help="Use MSU labeled data: _v4_labeled_msu_d7.csv, target_msu, same Exp10 negative filter, all seasons. Results/ranking/log use exp11 names.")
    parser.add_argument("--exp12-data", action="store_true",
                        help="Exp 12: same as Exp 10 (D-7 muscular labeled data, 2020/21+, Exp10 negative filter) plus exclude all timelines when reference_date in [D, D+5] for muscular onset D. Results/ranking/log use exp12 names.")
    parser.add_argument("--min-season", type=str, default=None,
                        help="Override minimum train season (e.g. 2020_2021). Only seasons >= this are loaded; earlier seasons excluded. With --exp11-data use 2020_2021 to exclude 2018/19 and 2019/20.")
    parser.add_argument("--deploy-dir", type=str, default=None,
                        help="Directory for deployment artifacts. With --only-iteration or --export-best, save model and feature list here (e.g. model_msu_lgbm).")
    parser.add_argument("--only-iteration", type=int, default=None,
                        help="Run only this iteration number then exit (e.g. 20 for 400 features with default step). Uses same logic as the iterative loop; if results already have that iteration, re-runs and replaces it.")
    parser.add_argument("--chosen", action="store_true",
                        help="With --only-iteration: treat this run as the selected model and write all artifacts (model, features, deployment folder, reports). If omitted with --only-iteration, only run the iteration and print results without saving anything.")
    parser.add_argument("--run-feature-counts", type=str, default=None,
                        help="Comma-separated feature counts to run (e.g. 270,290,310). Uses same ranking; trains once per count and saves to neighbourhood results file. Implies algorithm=gb.")
    parser.add_argument("--model", choices=["muscular", "skeletal", "msu"], default="muscular",
                        help="Model to train: muscular (target1), skeletal (target2), or msu (target_msu; implies --exp11-data). Default: muscular.")
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    # --model msu implies MSU (Exp 11) data
    if args.model == 'msu':
        args.exp11_data = True
    # Skeletal: LGBM only, 100% train (no val split)
    if args.model == 'skeletal':
        args.algorithm = 'lgbm'
        args.train_on_full_data = True
    if args.chosen and args.only_iteration is None:
        print("--chosen requires --only-iteration. Specify an iteration number (e.g. --only-iteration 21 --chosen).", file=sys.stderr)
        sys.exit(1)
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
        initial_features = INITIAL_FEATURES  # 200 for both LGBM and GB; step remains 20/50

    # When using GB, write to separate results/log so LGBM and GB runs don't overwrite each other
    if algorithm == 'gb':
        RESULTS_FILE = MODEL_OUTPUT_DIR / 'iterative_feature_selection_results_muscular_gb.json'
        LOG_FILE = MODEL_OUTPUT_DIR / 'iterative_training_muscular_gb.log'
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            f.write(f"=== Iterative Training Log (Model 1 - Muscular, algorithm=gb) Started at {datetime.now().isoformat()} ===\n")

    # Configure Exp 10 data mode before any training/export/HP tests
    if args.exp10_data:
        USE_LABELED_TIMELINES = True
        LABELED_SUFFIX = '_v4_labeled_muscle_skeletal_only_d7.csv'
        MIN_SEASON = '2020_2021'
        RANKING_FILE = MODEL_OUTPUT_DIR / 'feature_ranking_exp10.json'
        RESULTS_FILE = MODEL_OUTPUT_DIR / 'iterative_feature_selection_results_muscular_exp10.json'
        LOG_FILE = MODEL_OUTPUT_DIR / 'iterative_training_muscular_exp10.log'
        if algorithm == 'gb':
            RESULTS_FILE = MODEL_OUTPUT_DIR / 'iterative_feature_selection_results_muscular_gb_exp10.json'
            LOG_FILE = MODEL_OUTPUT_DIR / 'iterative_training_muscular_gb_exp10.log'
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            f.write(f"=== Iterative Training Log (Model 1 - Muscular, {'GB, ' if algorithm == 'gb' else ''}Exp10 data) Started at {datetime.now().isoformat()} ===\n")
        log_message("Exp 10 data mode: USE_LABELED_TIMELINES=True, LABELED_SUFFIX=D-7, MIN_SEASON=2020_2021")
        if algorithm == 'gb':
            log_message("   GB on Exp10: using dedicated results/log (gb_exp10).")
        log_message(f"   Ranking: {RANKING_FILE.name}, Results: {RESULTS_FILE.name}, Log: {LOG_FILE.name}")

    # Configure Exp 11 (MSU) data mode: MSU labeled timelines, target_msu, same Exp10 negative filter, all seasons
    if args.exp11_data:
        USE_LABELED_TIMELINES = True
        LABELED_SUFFIX = '_v4_labeled_msu_d7.csv'
        MIN_SEASON = None  # all seasons
        TARGET_COLUMN = 'target_msu'
        RANKING_FILE = MODEL_OUTPUT_DIR / 'feature_ranking_exp11.json'
        RESULTS_FILE = MODEL_OUTPUT_DIR / 'iterative_feature_selection_results_muscular_exp11.json'
        LOG_FILE = MODEL_OUTPUT_DIR / 'iterative_training_muscular_exp11.log'
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            f.write(f"=== Iterative Training Log (MSU target, Exp11 data) Started at {datetime.now().isoformat()} ===\n")
        log_message("Exp 11 data mode: USE_LABELED_TIMELINES=True, LABELED_SUFFIX=_v4_labeled_msu_d7.csv, target_msu, MIN_SEASON=all")
        log_message(f"   Ranking: {RANKING_FILE.name}, Results: {RESULTS_FILE.name}, Log: {LOG_FILE.name}")
        # Ensure Exp 11 ranking exists (copy from Exp 10 if missing)
        exp10_ranking = MODEL_OUTPUT_DIR / 'feature_ranking_exp10.json'
        if not RANKING_FILE.exists() and exp10_ranking.exists():
            import shutil
            shutil.copy(exp10_ranking, RANKING_FILE)
            log_message(f"   Created {RANKING_FILE.name} from {exp10_ranking.name}")

    # Configure Exp 12: Exp 10 data + exclude timelines with reference_date in [D, D+5] for muscular onset D
    if args.exp12_data:
        USE_LABELED_TIMELINES = True
        LABELED_SUFFIX = '_v4_labeled_muscle_skeletal_only_d7.csv'
        MIN_SEASON = '2020_2021'
        TARGET_COLUMN = 'target1'
        RANKING_FILE = MODEL_OUTPUT_DIR / 'feature_ranking_exp12.json'
        RESULTS_FILE = MODEL_OUTPUT_DIR / 'iterative_feature_selection_results_muscular_exp12.json'
        LOG_FILE = MODEL_OUTPUT_DIR / 'iterative_training_muscular_exp12.log'
        if algorithm == 'gb':
            RESULTS_FILE = MODEL_OUTPUT_DIR / 'iterative_feature_selection_results_muscular_gb_exp12.json'
            LOG_FILE = MODEL_OUTPUT_DIR / 'iterative_training_muscular_gb_exp12.log'
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            f.write(f"=== Iterative Training Log (Exp 12: Exp10 + no ref in [D,D+5]{', algorithm=gb' if algorithm == 'gb' else ''}) Started at {datetime.now().isoformat()} ===\n")
        log_message("Exp 12 data mode: same as Exp 10 + exclude timelines with reference_date in [D, D+5] for muscular onset D")
        if algorithm == 'gb':
            log_message("   GB on Exp12: using dedicated results/log (gb_exp12).")
        log_message(f"   Ranking: {RANKING_FILE.name}, Results: {RESULTS_FILE.name}, Log: {LOG_FILE.name}")
        exp10_ranking = MODEL_OUTPUT_DIR / 'feature_ranking_exp10.json'
        if not RANKING_FILE.exists() and exp10_ranking.exists():
            import shutil
            shutil.copy(exp10_ranking, RANKING_FILE)
            log_message(f"   Created {RANKING_FILE.name} from {exp10_ranking.name}")

    # Configure skeletal model (target2): own ranking/results/log; supports exp12 data
    if args.model == 'skeletal':
        TARGET_COLUMN = 'target2'
        if args.exp12_data:
            USE_LABELED_TIMELINES = True
            LABELED_SUFFIX = '_v4_labeled_muscle_skeletal_only_d7.csv'
            MIN_SEASON = '2020_2021'
            RANKING_FILE = MODEL_OUTPUT_DIR / 'feature_ranking_skeletal_exp12.json'
            RESULTS_FILE = MODEL_OUTPUT_DIR / 'iterative_feature_selection_results_skeletal_exp12.json'
            LOG_FILE = MODEL_OUTPUT_DIR / 'iterative_training_skeletal_exp12.log'
        else:
            RANKING_FILE = MODEL_OUTPUT_DIR / 'feature_ranking_skeletal.json'
            RESULTS_FILE = MODEL_OUTPUT_DIR / 'iterative_feature_selection_results_skeletal.json'
            LOG_FILE = MODEL_OUTPUT_DIR / 'iterative_training_skeletal.log'
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            f.write(f"=== Iterative Training Log (Model 2 - Skeletal{' Exp12' if args.exp12_data else ''}) Started at {datetime.now().isoformat()} ===\n")
        log_message("Model: skeletal (target2). Algorithm: LGBM only. Train: 100% of pool.")
        log_message(f"   Ranking: {RANKING_FILE.name}, Results: {RESULTS_FILE.name}, Log: {LOG_FILE.name}")
        if not RANKING_FILE.exists():
            default_ranking = MODEL_OUTPUT_DIR / 'feature_ranking.json'
            if default_ranking.exists():
                import shutil
                shutil.copy(default_ranking, RANKING_FILE)
                log_message(f"   Created {RANKING_FILE.name} from {default_ranking.name}")

    # Override MIN_SEASON when --min-season is set (e.g. 2020_2021 to exclude 2018/19 and 2019/20)
    if args.min_season is not None:
        MIN_SEASON = args.min_season
        log_message(f"   Override: MIN_SEASON={MIN_SEASON} (only seasons >= {MIN_SEASON})")

    try:
        if run_feature_counts is not None:
            exit_code = run_neighbourhood_feature_counts(
                run_feature_counts, use_cache=use_cache, algorithm=algorithm
            )
            sys.exit(exit_code)
        if args.evaluate_best_on_test:
            exit_code = evaluate_best_model_on_test()
            sys.exit(exit_code)
        if args.train_500_on_labeled:
            exit_code = train_500_on_labeled_datasets(use_cache=use_cache, save_model=True)
            sys.exit(exit_code)
        if args.export_best:
            exit_code = export_best_model(
                use_cache=use_cache, algorithm=algorithm,
                export_iteration=args.export_iteration, hp_preset=args.export_hp_preset,
                train_on_full_data=args.train_on_full_data,
                exp10_data=args.exp10_data,
                test_negatives_before=args.test_negatives_before,
                exp11_data=args.exp11_data,
                exp12_data=args.exp12_data,
                model_type=args.model,
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
                algorithm=algorithm,
                use_cache=use_cache,
                presets_list=presets_list,
                output_suffix=output_suffix,
                exp10_data=args.exp10_data,
                test_negatives_before=args.test_negatives_before,
            )
            sys.exit(exit_code)
        if args.threshold_sweep:
            exit_code = run_threshold_sweep(algorithm=algorithm, hp_preset=args.hp_preset, use_cache=use_cache)
            sys.exit(exit_code)
        deploy_dir = Path(args.deploy_dir) if args.deploy_dir else None
        exit_code = run_iterative_training(
            resume=args.resume,
            optimize_on=args.optimize_on,
            use_cache=use_cache,
            algorithm=algorithm,
            features_per_iteration=features_per_iteration,
            initial_features=initial_features,
            hp_preset=args.iterative_hp_preset,
            use_full_train=args.train_on_full_data,
            exp10_data=args.exp10_data,
            test_negatives_before=args.test_negatives_before,
            only_iteration=args.only_iteration,
            exp11_data=args.exp11_data,
            exp12_data=args.exp12_data,
            deploy_dir=deploy_dir,
            model_type=args.model,
            only_iteration_chosen=args.chosen,
        )
        sys.exit(exit_code)
    except KeyboardInterrupt:
        log_message("\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        log_error("Fatal error in main execution", e)
        sys.exit(1)
