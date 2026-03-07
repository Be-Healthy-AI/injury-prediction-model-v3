#!/usr/bin/env python3
"""
Train LGBM models for V4 Enhanced with dual targets (muscular and skeletal injuries)
- Training: Seasons 2018/19 onwards (natural target ratio)
- Test: Season 2025/26 (natural target ratio)
- Models: Two separate LGBM models (one for muscular, one for skeletal)
- Approach: Natural target ratios, filtered to exclude other injury types from negatives
"""

import sys
import io
# NOTE: stdout/stderr wrapping is handled by the main script (train_iterative_feature_selection.py)
# We skip wrapping here to avoid double-wrapping issues

import os
import json
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
    roc_auc_score, confusion_matrix, classification_report
)
from tqdm import tqdm

# Add paths for imports
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent.parent.parent
TIMELINES_DIR = SCRIPT_DIR.parent / 'timelines'  # code/timelines
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
MODEL_OUTPUT_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'models'
CACHE_DIR = ROOT_DIR / 'cache'
USE_CACHE = True
# ===================================

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
            # Check for target1 and target2 columns
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
    
    # Combine all dataframes
    print(f"\n📊 Combining {len(dfs)} season datasets...")
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
    
    target1_count = (df_test['target1'] == 1).sum()
    target2_count = (df_test['target2'] == 1).sum()
    
    print(f"✅ Test dataset: {len(df_test):,} records")
    print(f"   target1=1 (Muscular): {target1_count:,} ({target1_count/len(df_test)*100:.4f}%)")
    print(f"   target2=1 (Skeletal): {target2_count:,} ({target2_count/len(df_test)*100:.4f}%)")
    
    return df_test

def prepare_data(df, cache_file=None, use_cache=True):
    """Prepare data with basic preprocessing (no feature selection) and optional caching"""
    
    # Check cache - but verify it matches the current dataframe length
    if use_cache and cache_file and os.path.exists(cache_file):
        print(f"   📦 Loading preprocessed data from cache: {cache_file}...")
        try:
            df_preprocessed = pd.read_csv(cache_file, encoding='utf-8-sig', low_memory=False)
            # Verify cache matches current dataframe length
            if len(df_preprocessed) != len(df):
                print(f"   ⚠️  Cache length ({len(df_preprocessed)}) doesn't match data length ({len(df)}), preprocessing fresh...")
                use_cache = False
            else:
                # Drop target columns if they exist (we'll add them back)
                target_cols = ['target1', 'target2', 'target']
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
        if col not in ['player_id', 'reference_date', 'date', 'player_name', 'target1', 'target2', 'target', 'has_minimum_activity']
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
            # Fill NaN first
            X_encoded[feature] = X_encoded[feature].fillna('Unknown')
            
            # Check for problematic values before cleaning
            problematic_mask = X_encoded[feature].astype(str).str.contains(r'[:,\'"\\/;|&?!*+=@#$%^\s]', regex=True, na=False)
            problematic_count = problematic_mask.sum()
            if problematic_count > 0:
                problematic_values = X_encoded[feature][problematic_mask].unique()[:10]
                print(f"\n⚠️  Found {problematic_count} problematic values in '{feature}': {list(problematic_values)}")
            
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
            df_preprocessed = X_encoded.copy()
            df_preprocessed.to_csv(cache_file, index=False, encoding='utf-8-sig')
            print(f"   ✅ Preprocessed data cached")
        except Exception as e:
            print(f"   ⚠️  Failed to cache ({e})")
    
    return X_encoded

def align_features(X_train, X_test):
    """Ensure all datasets have the same features"""
    # Get common features across both datasets
    common_features = list(set(X_train.columns) & set(X_test.columns))
    
    # Sort for consistency
    common_features = sorted(common_features)
    
    print(f"   Aligning features: {len(common_features)} common features")
    print(f"   Training: {X_train.shape[1]} -> {len(common_features)}")
    print(f"   Test: {X_test.shape[1]} -> {len(common_features)}")
    
    return X_train[common_features], X_test[common_features]

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
    
    # Model configuration
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
        verbose=1
    )
    
    print(f"\n🔧 Model hyperparameters:")
    print(f"   n_estimators: {lgbm_model.n_estimators}")
    print(f"   max_depth: {lgbm_model.max_depth}")
    print(f"   learning_rate: {lgbm_model.learning_rate}")
    print(f"   class_weight: {lgbm_model.class_weight}")
    
    # Training data stats
    print(f"\n📊 Training data:")
    print(f"   Total samples: {len(X_train):,}")
    print(f"   Positives: {y_train.sum():,} ({y_train.mean()*100:.4f}%)")
    print(f"   Negatives: {(y_train == 0).sum():,}")
    
    # Train model
    print(f"\n⏳ Training model...")
    start_time = datetime.now()
    lgbm_model.fit(X_train, y_train)
    training_time = (datetime.now() - start_time).total_seconds()
    print(f"✅ Training completed in {training_time:.1f} seconds")
    
    # Evaluate on training set
    train_metrics = evaluate_model(lgbm_model, X_train, y_train, "Training")
    
    # Evaluate on test set
    test_metrics = evaluate_model(lgbm_model, X_test, y_test, "Test")
    
    return lgbm_model, train_metrics, test_metrics

def main():
    print("="*80)
    print("TRAINING LGBM MODELS - V4 ENHANCED DUAL TARGETS (NATURAL RATIO)")
    print("="*80)
    print(f"\n📋 Configuration:")
    print(f"   Training: Seasons {MIN_SEASON} onwards (natural target ratio)")
    print(f"   Test: Season {EXCLUDE_SEASON} (natural target ratio)")
    print(f"   Models: Two separate LGBM models (muscular and skeletal)")
    print(f"   Filtering: Exclude other injury types from negatives")
    print("="*80)
    
    start_time = datetime.now()
    print(f"\n⏰ Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directories
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load training data
    print("\n" + "="*80)
    print("STEP 1: LOADING TRAINING DATA")
    print("="*80)
    df_train_all = load_combined_seasonal_datasets_natural(
        min_season=MIN_SEASON,
        exclude_season=EXCLUDE_SEASON
    )
    
    # Load test data
    print("\n" + "="*80)
    print("STEP 2: LOADING TEST DATA")
    print("="*80)
    df_test_all = load_test_dataset()
    
    # Filter for Model 1 (Muscular)
    print("\n" + "="*80)
    print("STEP 3: FILTERING FOR MODEL 1 (MUSCULAR INJURIES)")
    print("="*80)
    df_train_muscular = filter_timelines_for_model(df_train_all, 'target1')
    df_test_muscular = filter_timelines_for_model(df_test_all, 'target1')
    
    # Filter for Model 2 (Skeletal)
    print("\n" + "="*80)
    print("STEP 4: FILTERING FOR MODEL 2 (SKELETAL INJURIES)")
    print("="*80)
    df_train_skeletal = filter_timelines_for_model(df_train_all, 'target2')
    df_test_skeletal = filter_timelines_for_model(df_test_all, 'target2')
    
    # Prepare features
    print("\n" + "="*80)
    print("STEP 5: PREPARING FEATURES")
    print("="*80)
    
    # Model 1 features
    print("Preparing features for Model 1 (Muscular)...")
    cache_file_muscular_train = str(CACHE_DIR / 'preprocessed_muscular_train_natural.csv')
    cache_file_muscular_test = str(CACHE_DIR / 'preprocessed_muscular_test_natural.csv')
    
    # Reset indices to ensure alignment
    df_train_muscular = df_train_muscular.reset_index(drop=True)
    df_test_muscular = df_test_muscular.reset_index(drop=True)
    df_train_skeletal = df_train_skeletal.reset_index(drop=True)
    df_test_skeletal = df_test_skeletal.reset_index(drop=True)
    
    X_train_muscular = prepare_data(df_train_muscular, cache_file=cache_file_muscular_train, use_cache=USE_CACHE)
    y_train_muscular = df_train_muscular['target1'].values
    X_test_muscular = prepare_data(df_test_muscular, cache_file=cache_file_muscular_test, use_cache=USE_CACHE)
    y_test_muscular = df_test_muscular['target1'].values
    
    # Verify lengths match
    assert len(X_train_muscular) == len(y_train_muscular), f"X_train_muscular length ({len(X_train_muscular)}) != y_train_muscular length ({len(y_train_muscular)})"
    assert len(X_test_muscular) == len(y_test_muscular), f"X_test_muscular length ({len(X_test_muscular)}) != y_test_muscular length ({len(y_test_muscular)})"
    
    # Model 2 features
    print("\nPreparing features for Model 2 (Skeletal)...")
    cache_file_skeletal_train = str(CACHE_DIR / 'preprocessed_skeletal_train_natural.csv')
    cache_file_skeletal_test = str(CACHE_DIR / 'preprocessed_skeletal_test_natural.csv')
    
    X_train_skeletal = prepare_data(df_train_skeletal, cache_file=cache_file_skeletal_train, use_cache=USE_CACHE)
    y_train_skeletal = df_train_skeletal['target2'].values
    X_test_skeletal = prepare_data(df_test_skeletal, cache_file=cache_file_skeletal_test, use_cache=USE_CACHE)
    y_test_skeletal = df_test_skeletal['target2'].values
    
    # Verify lengths match
    assert len(X_train_skeletal) == len(y_train_skeletal), f"X_train_skeletal length ({len(X_train_skeletal)}) != y_train_skeletal length ({len(y_train_skeletal)})"
    assert len(X_test_skeletal) == len(y_test_skeletal), f"X_test_skeletal length ({len(X_test_skeletal)}) != y_test_skeletal length ({len(y_test_skeletal)})"
    
    # Align features between train and test
    print("\nAligning features between train and test sets...")
    X_train_muscular, X_test_muscular = align_features(X_train_muscular, X_test_muscular)
    X_train_skeletal, X_test_skeletal = align_features(X_train_skeletal, X_test_skeletal)
    
    print(f"   Model 1 features: {len(X_train_muscular.columns)}")
    print(f"   Model 2 features: {len(X_train_skeletal.columns)}")
    
    # Train Model 1 (Muscular)
    print("\n" + "="*80)
    print("STEP 6: TRAINING MODEL 1 (MUSCULAR INJURIES)")
    print("="*80)
    model1, train_metrics1, test_metrics1 = train_lgbm_model(
        X_train_muscular, y_train_muscular,
        X_test_muscular, y_test_muscular,
        "Model 1: Muscular Injuries",
        "target1"
    )
    
    # Train Model 2 (Skeletal)
    print("\n" + "="*80)
    print("STEP 7: TRAINING MODEL 2 (SKELETAL INJURIES)")
    print("="*80)
    model2, train_metrics2, test_metrics2 = train_lgbm_model(
        X_train_skeletal, y_train_skeletal,
        X_test_skeletal, y_test_skeletal,
        "Model 2: Skeletal Injuries",
        "target2"
    )
    
    # Save models
    print("\n" + "="*80)
    print("STEP 8: SAVING MODELS")
    print("="*80)
    
    model1_path = MODEL_OUTPUT_DIR / 'lgbm_muscular_v4_natural_model1.joblib'
    model1_cols_path = MODEL_OUTPUT_DIR / 'lgbm_muscular_v4_natural_model1_columns.json'
    model2_path = MODEL_OUTPUT_DIR / 'lgbm_muscular_v4_natural_model2.joblib'
    model2_cols_path = MODEL_OUTPUT_DIR / 'lgbm_muscular_v4_natural_model2_columns.json'
    
    joblib.dump(model1, model1_path)
    with open(model1_cols_path, 'w') as f:
        json.dump(list(X_train_muscular.columns), f, indent=2)
    print(f"✅ Saved Model 1: {model1_path}")
    print(f"✅ Saved Model 1 columns: {model1_cols_path}")
    
    joblib.dump(model2, model2_path)
    with open(model2_cols_path, 'w') as f:
        json.dump(list(X_train_skeletal.columns), f, indent=2)
    print(f"✅ Saved Model 2: {model2_path}")
    print(f"✅ Saved Model 2 columns: {model2_cols_path}")
    
    # Save metrics
    metrics_path = MODEL_OUTPUT_DIR / 'lgbm_muscular_v4_natural_metrics.json'
    metrics = {
        'model1_muscular': {
            'train': train_metrics1,
            'test': test_metrics1
        },
        'model2_skeletal': {
            'train': train_metrics2,
            'test': test_metrics2
        },
        'configuration': {
            'min_season': MIN_SEASON,
            'exclude_season': EXCLUDE_SEASON,
            'training_date': start_time.isoformat()
        }
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Saved metrics: {metrics_path}")
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    print("\n📊 MODEL 1 (MUSCULAR INJURIES):")
    print(f"   Training - Accuracy: {train_metrics1['accuracy']:.4f}, F1: {train_metrics1['f1']:.4f}, ROC-AUC: {train_metrics1['roc_auc']:.4f}, Gini: {train_metrics1['gini']:.4f}")
    print(f"   Test     - Accuracy: {test_metrics1['accuracy']:.4f}, F1: {test_metrics1['f1']:.4f}, ROC-AUC: {test_metrics1['roc_auc']:.4f}, Gini: {test_metrics1['gini']:.4f}")
    
    print("\n📊 MODEL 2 (SKELETAL INJURIES):")
    print(f"   Training - Accuracy: {train_metrics2['accuracy']:.4f}, F1: {train_metrics2['f1']:.4f}, ROC-AUC: {train_metrics2['roc_auc']:.4f}, Gini: {train_metrics2['gini']:.4f}")
    print(f"   Test     - Accuracy: {test_metrics2['accuracy']:.4f}, F1: {test_metrics2['f1']:.4f}, ROC-AUC: {test_metrics2['roc_auc']:.4f}, Gini: {test_metrics2['gini']:.4f}")
    
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds() / 60
    print(f"\n⏰ Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Total time: {total_time:.1f} minutes")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
