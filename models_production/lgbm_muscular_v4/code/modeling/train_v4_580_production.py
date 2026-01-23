#!/usr/bin/env python3
"""
Train and Save V4 580-Feature Production Model

This script trains the final V4 580 model (With Test, Excl 2021/22-2022/23)
and saves it in production format with all necessary files.

Configuration:
- Training: 2018/19-2025/26, excluding 2021/22 and 2022/23 seasons
- Features: 580 optimal features from iterative selection
- Test: 2025/26 (in-sample)
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

# Import helper functions from comparison script
# We'll copy the essential functions here to avoid import issues
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent.parent.parent
MODEL_OUTPUT_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'models'
PRODUCTION_MODEL_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'model'
CACHE_DIR = ROOT_DIR / 'cache'

# ========== CONFIGURATION ==========
ITERATIVE_RESULTS_FILE = MODEL_OUTPUT_DIR / 'iterative_feature_selection_results.json'
MIN_SEASON = '2018_2019'
TRAIN_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'data' / 'timelines' / 'train'
TEST_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'data' / 'timelines' / 'test'
USE_CACHE = True

# Production model files
PRODUCTION_MODEL_FILE = PRODUCTION_MODEL_DIR / 'model.joblib'
PRODUCTION_COLUMNS_FILE = PRODUCTION_MODEL_DIR / 'columns.json'
PRODUCTION_METADATA_FILE = PRODUCTION_MODEL_DIR / 'MODEL_METADATA.json'
PRODUCTION_TRAIN_METRICS_FILE = PRODUCTION_MODEL_DIR / 'lgbm_v4_580_metrics_train.json'
PRODUCTION_TEST_METRICS_FILE = PRODUCTION_MODEL_DIR / 'lgbm_v4_580_metrics_test.json'
LOG_FILE = PRODUCTION_MODEL_DIR / 'train_v4_580_production.log'
# ===================================

# Ensure production model directory exists
PRODUCTION_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Initialize log file
try:
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(f"=== V4 580 Production Model Training Started at {datetime.now().isoformat()} ===\n")
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
# HELPER FUNCTIONS (copied from train_v4_620_features_fair_comparison.py)
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
    """Load the 580 optimal features from iteration 31"""
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
        combined_score = iteration_31.get('combined_score', None)
        log_message(f"✅ Loaded {len(features)} optimal features from iteration 31")
        if combined_score is not None:
            log_message(f"   Combined score: {combined_score:.4f}")
        
        return features, combined_score
        
    except Exception as e:
        log_error("Failed to load optimal features", e)
        raise

def calculate_model_hash(model, features):
    """Calculate hash of model and features for reproducibility"""
    import pickle
    model_bytes = pickle.dumps((model, features))
    return hashlib.sha256(model_bytes).hexdigest()

def get_season_breakdown(df):
    """Get breakdown of records by season"""
    df['_season'] = df['reference_date'].apply(extract_season_from_date)
    season_counts = df.groupby('_season').agg({
        'target1': ['count', 'sum']
    }).reset_index()
    season_counts.columns = ['season', 'records', 'positives']
    season_breakdown = {}
    for _, row in season_counts.iterrows():
        season_breakdown[row['season']] = {
            'records': int(row['records']),
            'positives': int(row['positives'])
        }
    df = df.drop(columns=['_season'])
    return season_breakdown

def main():
    """Main function to train and save production model"""
    log_message("="*80)
    log_message("V4 580 PRODUCTION MODEL TRAINING")
    log_message("="*80)
    
    start_time = datetime.now()
    log_message(f"\n⏰ Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Load optimal features
        log_message("\n" + "="*80)
        log_message("STEP 1: LOADING OPTIMAL FEATURES")
        log_message("="*80)
        optimal_features, combined_score = load_optimal_features()
        
        # Load training data
        log_message("\n" + "="*80)
        log_message("STEP 2: LOADING TRAINING DATA")
        log_message("="*80)
        log_message("Configuration: Training 2018/19-2025/26, excluding 2021/22 and 2022/23 seasons")
        
        # Load all data including test
        df_train_all = load_combined_seasonal_datasets_natural(
            min_season=MIN_SEASON,
            exclude_seasons=[],
            include_test=True
        )
        
        # Filter out 2021/22 and 2022/23 seasons
        df_train_all = filter_by_seasons(df_train_all, ['2021_2022', '2022_2023'])
        
        # Load test dataset (for evaluation)
        log_message("\n" + "="*80)
        log_message("STEP 3: LOADING TEST DATASET")
        log_message("="*80)
        df_test_all = load_test_dataset()
        
        # Filter for Model 1 (Muscular)
        log_message("\n" + "="*80)
        log_message("STEP 4: PREPARING DATA FOR MODEL 1 (MUSCULAR)")
        log_message("="*80)
        df_train_muscular = filter_timelines_for_model(df_train_all, 'target1')
        df_test_muscular = filter_timelines_for_model(df_test_all, 'target1')
        
        # Prepare features
        cache_file_train = str(CACHE_DIR / 'preprocessed_muscular_train_v4_580_production.csv')
        cache_file_test = str(CACHE_DIR / 'preprocessed_muscular_test_v4_580_production.csv')
        
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
        log_message("\n" + "="*80)
        log_message("STEP 5: TRAINING MODEL")
        log_message("="*80)
        log_message("🚀 Training Model 1 (Muscular Injuries)...")
        
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
        log_message("\n" + "="*80)
        log_message("STEP 6: EVALUATING MODEL")
        log_message("="*80)
        train_metrics = evaluate_model(lgbm_model, X_train, y_train, "Training")
        test_metrics = evaluate_model(lgbm_model, X_test, y_test, "Test")
        
        # Get season breakdown
        season_breakdown = get_season_breakdown(df_train_muscular.copy())
        
        # Calculate model hash
        model_hash = calculate_model_hash(lgbm_model, list(X_train.columns))
        
        # Save model files
        log_message("\n" + "="*80)
        log_message("STEP 7: SAVING PRODUCTION MODEL FILES")
        log_message("="*80)
        
        # Save model
        log_message(f"   Saving model to: {PRODUCTION_MODEL_FILE}")
        joblib.dump(lgbm_model, PRODUCTION_MODEL_FILE)
        log_message("   ✅ Model saved")
        
        # Save columns
        log_message(f"   Saving feature columns to: {PRODUCTION_COLUMNS_FILE}")
        with open(PRODUCTION_COLUMNS_FILE, 'w', encoding='utf-8') as f:
            json.dump(list(X_train.columns), f, indent=2)
        log_message("   ✅ Columns saved")
        
        # Save metrics
        log_message(f"   Saving training metrics to: {PRODUCTION_TRAIN_METRICS_FILE}")
        with open(PRODUCTION_TRAIN_METRICS_FILE, 'w', encoding='utf-8') as f:
            json.dump(train_metrics, f, indent=2)
        log_message("   ✅ Training metrics saved")
        
        log_message(f"   Saving test metrics to: {PRODUCTION_TEST_METRICS_FILE}")
        with open(PRODUCTION_TEST_METRICS_FILE, 'w', encoding='utf-8') as f:
            json.dump(test_metrics, f, indent=2)
        log_message("   ✅ Test metrics saved")
        
        # Create and save metadata
        creation_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        metadata = {
            "model_version": "V4_580_with_test_excl_2021_2022_2022_2023",
            "model_name": "LGBM Muscular Injury Prediction V4 (580 Features, With Test, Excl 2021/22-2022/23)",
            "creation_date": creation_date,
            "training_configuration": {
                "target_ratio": None,
                "target_ratio_display": "natural (unbalanced)",
                "excluded_seasons": ["2021_2022", "2022_2023"],
                "excluded_seasons_reason": {
                    "2021_2022": "Low injury rate - not representative of normal seasons",
                    "2022_2023": "Low injury rate - not representative of normal seasons"
                },
                "min_season": "2018_2019",
                "max_season": "2025_2026",
                "seasons_used": ["2018_2019", "2019_2020", "2020_2021", "2023_2024", "2024_2025", "2025_2026"],
                "filter_type": "PL-only (only days when players were at PL clubs)",
                "random_state": 42,
                "hyperparameters": {
                    "n_estimators": 200,
                    "max_depth": 10,
                    "learning_rate": 0.1,
                    "min_child_samples": 20,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_alpha": 0.1,
                    "reg_lambda": 1.0,
                    "class_weight": "balanced"
                },
                "feature_selection": {
                    "method": "iterative_feature_selection",
                    "iteration": 31,
                    "n_features": 580,
                    "combined_score": combined_score
                }
            },
            "training_dataset": {
                "total_records": len(df_train_muscular),
                "total_positives": int(y_train.sum()),
                "total_negatives": int(len(y_train) - y_train.sum()),
                "injury_rate": float(y_train.sum() / len(y_train)),
                "season_breakdown": season_breakdown
            },
            "performance_metrics": {
                "train": train_metrics,
                "test": test_metrics
            },
            "model_hash": model_hash,
            "deployment_info": {
                "model_file": "model.joblib",
                "columns_file": "columns.json",
                "feature_count": len(X_train.columns),
                "deployment_date": creation_date
            }
        }
        
        log_message(f"   Saving metadata to: {PRODUCTION_METADATA_FILE}")
        with open(PRODUCTION_METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        log_message("   ✅ Metadata saved")
        
        # Summary
        log_message("\n" + "="*80)
        log_message("SUMMARY")
        log_message("="*80)
        log_message(f"✅ Production model saved successfully!")
        log_message(f"   Model file: {PRODUCTION_MODEL_FILE}")
        log_message(f"   Columns file: {PRODUCTION_COLUMNS_FILE}")
        log_message(f"   Metadata file: {PRODUCTION_METADATA_FILE}")
        log_message(f"   Features: {len(X_train.columns)}")
        log_message(f"   Training samples: {len(X_train):,}")
        log_message(f"   Test samples: {len(X_test):,}")
        log_message(f"\n   Test Performance:")
        log_message(f"      Gini: {test_metrics['gini']:.4f}")
        log_message(f"      F1-Score: {test_metrics['f1']:.4f}")
        log_message(f"      Precision: {test_metrics['precision']:.4f}")
        log_message(f"      Recall: {test_metrics['recall']:.4f}")
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds() / 60
        log_message(f"\n⏰ Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        log_message(f"⏱️  Total time: {total_time:.1f} minutes")
        
        return 0
        
    except Exception as e:
        log_error("Fatal error in production model training", e)
        return 1

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
