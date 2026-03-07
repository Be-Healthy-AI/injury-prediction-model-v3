#!/usr/bin/env python3
"""
Train LGBM Target1 model using BOTH training and test datasets
This allows direct comparison with V3 models on the test dataset
"""

import sys
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
TRAIN_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'data' / 'timelines' / 'train'
TEST_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'data' / 'timelines' / 'test'
MODEL_OUTPUT_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'models' / 'comparison'
CACHE_DIR = ROOT_DIR / 'cache'
USE_CACHE = True
# ===================================

# Copy helper functions from train_lgbm_v4_dual_targets_natural.py
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

def load_combined_seasonal_datasets_natural(min_season=None):
    """Load and combine all seasonal datasets with natural target ratio (including test season)"""
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
                if min_season is None or season_part >= min_season:
                    season_files.append((season_part, filepath))
    
    # Also load test dataset
    test_file = TEST_DIR / 'timelines_35day_season_2025_2026_v4_muscular_test.csv'
    if test_file.exists():
        season_files.append(('2025_2026', test_file))
    
    season_files.sort(key=lambda x: x[0])
    
    print(f"\n📂 Loading {len(season_files)} season files (including test) with natural target ratio...")
    if min_season:
        print(f"   (Filtering: Only seasons >= {min_season})")
    
    dfs = []
    total_records = 0
    total_target1 = 0
    
    for season_id, filepath in season_files:
        try:
            df = pd.read_csv(filepath, encoding='utf-8-sig', low_memory=False)
            if 'target1' not in df.columns:
                print(f"   ⚠️  {season_id}: Missing target1 column - skipping")
                continue
            
            target1_count = (df['target1'] == 1).sum()
            
            if len(df) > 0:
                dfs.append(df)
                total_records += len(df)
                total_target1 += target1_count
                print(f"   ✅ {season_id}: {len(df):,} records (target1: {target1_count:,})")
        except Exception as e:
            print(f"   ⚠️  Error loading {season_id}: {e}")
            continue
    
    if not dfs:
        raise ValueError("No valid season files found!")
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    print(f"✅ Combined dataset: {len(combined_df):,} records")
    print(f"   Total target1=1 (Muscular): {total_target1:,} ({total_target1/len(combined_df)*100:.4f}%)")
    
    return combined_df

def load_test_dataset():
    """Load test dataset (2025/26 season) for evaluation only"""
    test_file = TEST_DIR / 'timelines_35day_season_2025_2026_v4_muscular_test.csv'
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")
    print(f"\n📂 Loading test dataset for evaluation: {test_file.name}")
    df_test = pd.read_csv(test_file, encoding='utf-8-sig', low_memory=False)
    if 'target1' not in df_test.columns:
        raise ValueError("Test dataset missing target1 column")
    target1_count = (df_test['target1'] == 1).sum()
    print(f"✅ Test dataset: {len(df_test):,} records")
    print(f"   target1=1 (Muscular): {target1_count:,} ({target1_count/len(df_test)*100:.4f}%)")
    return df_test

def prepare_data(df, cache_file=None, use_cache=True):
    """Prepare data with basic preprocessing"""
    if use_cache and cache_file and os.path.exists(cache_file):
        print(f"   📦 Loading preprocessed data from cache: {cache_file}...")
        try:
            df_preprocessed = pd.read_csv(cache_file, encoding='utf-8-sig', low_memory=False)
            if len(df_preprocessed) != len(df):
                print(f"   ⚠️  Cache length mismatch, preprocessing fresh...")
                use_cache = False
            else:
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
    
    feature_columns = [
        col for col in df.columns
        if col not in ['player_id', 'reference_date', 'date', 'player_name', 'target1', 'target2', 'target', 'has_minimum_activity']
    ]
    X = df[feature_columns].copy()
    
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    X_encoded = X.copy()
    
    if len(categorical_features) > 0:
        print(f"   Processing {len(categorical_features)} categorical features...")
        for feature in tqdm(categorical_features, desc="   Encoding categorical features", unit="feature", leave=False):
            X_encoded[feature] = X_encoded[feature].fillna('Unknown')
            X_encoded[feature] = X_encoded[feature].apply(clean_categorical_value)
            dummies = pd.get_dummies(X_encoded[feature], prefix=feature, drop_first=True)
            dummies.columns = [sanitize_feature_name(col) for col in dummies.columns]
            X_encoded = pd.concat([X_encoded, dummies], axis=1)
            X_encoded = X_encoded.drop(columns=[feature])
    
    if len(numeric_features) > 0:
        print(f"   Processing {len(numeric_features)} numeric features...")
        for feature in tqdm(numeric_features, desc="   Filling numeric missing values", unit="feature", leave=False):
            X_encoded[feature] = X_encoded[feature].fillna(0)
    
    X_encoded.columns = [sanitize_feature_name(col) for col in X_encoded.columns]
    
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

def evaluate_model(model, X, y, dataset_name):
    """Evaluate model and return metrics"""
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    
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

def main():
    print("="*80)
    print("TRAINING LGBM TARGET1 MODEL - USING BOTH TRAIN AND TEST DATASETS")
    print("="*80)
    print(f"\n📋 Configuration:")
    print(f"   Training: Seasons {MIN_SEASON} onwards + Test (2025/26) - ALL DATA")
    print(f"   Evaluation: Test dataset (2025/26) only")
    print(f"   Model: LGBM for Target1 (Muscular Injuries)")
    print("="*80)
    
    start_time = datetime.now()
    print(f"\n⏰ Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load ALL data (training + test) for training
    print("\n" + "="*80)
    print("STEP 1: LOADING ALL DATA FOR TRAINING (TRAIN + TEST)")
    print("="*80)
    df_train_all = load_combined_seasonal_datasets_natural(min_season=MIN_SEASON)
    
    # Load test data separately for evaluation
    print("\n" + "="*80)
    print("STEP 2: LOADING TEST DATA FOR EVALUATION")
    print("="*80)
    df_test_all = load_test_dataset()
    
    # Filter for Target1 (Muscular)
    print("\n" + "="*80)
    print("STEP 3: FILTERING FOR TARGET1 (MUSCULAR INJURIES)")
    print("="*80)
    df_train_target1 = filter_timelines_for_model(df_train_all, 'target1')
    df_test_target1 = filter_timelines_for_model(df_test_all, 'target1')
    
    # Prepare features
    print("\n" + "="*80)
    print("STEP 4: PREPARING FEATURES")
    print("="*80)
    
    print("Preparing features for training data...")
    cache_file_train = str(CACHE_DIR / 'preprocessed_target1_train_with_test.csv')
    X_train = prepare_data(df_train_target1, cache_file=cache_file_train, use_cache=USE_CACHE)
    y_train = df_train_target1['target1'].values
    
    print("\nPreparing features for test data...")
    cache_file_test = str(CACHE_DIR / 'preprocessed_target1_test_eval.csv')
    X_test = prepare_data(df_test_target1, cache_file=cache_file_test, use_cache=USE_CACHE)
    y_test = df_test_target1['target1'].values
    
    # Align features
    print("\nAligning features between train and test sets...")
    X_train, X_test = align_features(X_train, X_test)
    
    print(f"   Final features: {len(X_train.columns)}")
    
    # Train model
    print("\n" + "="*80)
    print("STEP 5: TRAINING LGBM MODEL")
    print("="*80)
    
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
    
    print(f"\n📊 Training data:")
    print(f"   Total samples: {len(X_train):,}")
    print(f"   Positives: {y_train.sum():,} ({y_train.mean()*100:.4f}%)")
    print(f"   Negatives: {(y_train == 0).sum():,}")
    
    print(f"\n⏳ Training model...")
    train_start = datetime.now()
    lgbm_model.fit(X_train, y_train)
    training_time = (datetime.now() - train_start).total_seconds()
    print(f"✅ Training completed in {training_time:.1f} seconds")
    
    # Evaluate on training set
    train_metrics = evaluate_model(lgbm_model, X_train, y_train, "Training")
    
    # Evaluate on test set
    test_metrics = evaluate_model(lgbm_model, X_test, y_test, "Test")
    
    # Save model
    print("\n" + "="*80)
    print("STEP 6: SAVING MODEL")
    print("="*80)
    
    model_path = MODEL_OUTPUT_DIR / 'lgbm_target1_train_with_test.joblib'
    model_cols_path = MODEL_OUTPUT_DIR / 'lgbm_target1_train_with_test_columns.json'
    
    joblib.dump(lgbm_model, model_path)
    with open(model_cols_path, 'w') as f:
        json.dump(list(X_train.columns), f, indent=2)
    print(f"✅ Saved model: {model_path}")
    print(f"✅ Saved columns: {model_cols_path}")
    
    # Save metrics
    metrics_path = MODEL_OUTPUT_DIR / 'lgbm_target1_train_with_test_metrics.json'
    metrics = {
        'train': train_metrics,
        'test': test_metrics,
        'configuration': {
            'min_season': MIN_SEASON,
            'training_includes_test': True,
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
    
    print("\n📊 V4 TARGET1 MODEL (Trained on Train + Test, Evaluated on Test):")
    print(f"   Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"   Test Precision: {test_metrics['precision']:.4f}")
    print(f"   Test Recall: {test_metrics['recall']:.4f}")
    print(f"   Test F1-Score: {test_metrics['f1']:.4f}")
    print(f"   Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"   Test Gini: {test_metrics['gini']:.4f}")
    
    print("\n💡 Note: This model was trained on BOTH training and test datasets.")
    print("   This allows direct comparison with V3 models that may have been")
    print("   trained on similar data configurations.")
    
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds() / 60
    print(f"\n⏰ Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Total time: {total_time:.1f} minutes")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
