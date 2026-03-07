#!/usr/bin/env python3
"""
Utility functions for training models with a specific feature subset.

This module provides functions that can be used by the iterative training script
to train models on a subset of features.
"""

import sys
import io
# NOTE: stdout/stderr wrapping is handled by the main script (train_iterative_feature_selection.py)
# We skip wrapping here to avoid double-wrapping issues

import os
import json
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

# Add paths for imports
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent.parent.parent
TIMELINES_DIR = SCRIPT_DIR.parent / 'timelines'
sys.path.insert(0, str(TIMELINES_DIR.resolve()))

# Import functions from main training script
import importlib.util
training_script_path = SCRIPT_DIR / 'train_lgbm_v4_dual_targets_natural.py'
spec = importlib.util.spec_from_file_location("train_lgbm_v4_dual_targets_natural", training_script_path)
training_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training_module)

# Import filter function from timeline generation script
timeline_script_path = TIMELINES_DIR / 'create_35day_timelines_v4_enhanced.py'
spec_timeline = importlib.util.spec_from_file_location("create_35day_timelines_v4_enhanced", timeline_script_path)
timeline_module = importlib.util.module_from_spec(spec_timeline)
spec_timeline.loader.exec_module(timeline_module)
filter_timelines_for_model = timeline_module.filter_timelines_for_model

# Reuse configuration and functions
MIN_SEASON = training_module.MIN_SEASON
EXCLUDE_SEASON = training_module.EXCLUDE_SEASON
TRAIN_DIR = training_module.TRAIN_DIR
TEST_DIR = training_module.TEST_DIR
MODEL_OUTPUT_DIR = training_module.MODEL_OUTPUT_DIR
CACHE_DIR = training_module.CACHE_DIR
USE_CACHE = training_module.USE_CACHE

# Import helper functions
load_combined_seasonal_datasets_natural = training_module.load_combined_seasonal_datasets_natural
load_test_dataset = training_module.load_test_dataset
prepare_data = training_module.prepare_data
align_features = training_module.align_features
evaluate_model = training_module.evaluate_model
convert_numpy_types = training_module.convert_numpy_types

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
        print(f"   ⚠️  Warning: {len(missing_features)} requested features not found in datasets")
        if len(missing_features) <= 10:
            print(f"      Missing: {missing_features}")
    
    if len(requested_features) == 0:
        raise ValueError("No requested features found in datasets!")
    
    print(f"   Using {len(requested_features)}/{len(feature_subset)} requested features")
    
    return X_train[requested_features], X_test[requested_features]

def train_models_with_feature_subset(feature_subset, verbose=True):
    """
    Train both models (muscular and skeletal) using a specific feature subset.
    
    Args:
        feature_subset: List of feature names to use for training
        verbose: Whether to print progress messages
        
    Returns:
        Dictionary containing:
        - model1: Trained muscular model
        - model2: Trained skeletal model
        - train_metrics1: Training metrics for model 1
        - test_metrics1: Test metrics for model 1
        - train_metrics2: Training metrics for model 2
        - test_metrics2: Test metrics for model 2
        - feature_names_used: List of features actually used
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"TRAINING WITH {len(feature_subset)} FEATURES")
        print(f"{'='*80}")
    
    # Load and prepare data (reuse cached preprocessed data if available)
    if verbose:
        print("\n📂 Loading datasets...")
    
    df_train_all = load_combined_seasonal_datasets_natural(
        min_season=MIN_SEASON,
        exclude_season=EXCLUDE_SEASON
    )
    df_test_all = load_test_dataset()
    
    # Filter for each model
    df_train_muscular = filter_timelines_for_model(df_train_all, 'target1')
    df_test_muscular = filter_timelines_for_model(df_test_all, 'target1')
    df_train_skeletal = filter_timelines_for_model(df_train_all, 'target2')
    df_test_skeletal = filter_timelines_for_model(df_test_all, 'target2')
    
    # Prepare features (use cache with unique names to avoid conflicts)
    cache_suffix = hashlib.md5(str(sorted(feature_subset)).encode()).hexdigest()[:8]
    
    cache_file_muscular_train = str(CACHE_DIR / f'preprocessed_muscular_train_subset_{cache_suffix}.csv')
    cache_file_muscular_test = str(CACHE_DIR / f'preprocessed_muscular_test_subset_{cache_suffix}.csv')
    cache_file_skeletal_train = str(CACHE_DIR / f'preprocessed_skeletal_train_subset_{cache_suffix}.csv')
    cache_file_skeletal_test = str(CACHE_DIR / f'preprocessed_skeletal_test_subset_{cache_suffix}.csv')
    
    df_train_muscular = df_train_muscular.reset_index(drop=True)
    df_test_muscular = df_test_muscular.reset_index(drop=True)
    df_train_skeletal = df_train_skeletal.reset_index(drop=True)
    df_test_skeletal = df_test_skeletal.reset_index(drop=True)
    
    if verbose:
        print("   Preparing features for Model 1 (Muscular)...")
    X_train_muscular = prepare_data(df_train_muscular, cache_file=cache_file_muscular_train, use_cache=USE_CACHE)
    y_train_muscular = df_train_muscular['target1'].values
    X_test_muscular = prepare_data(df_test_muscular, cache_file=cache_file_muscular_test, use_cache=USE_CACHE)
    y_test_muscular = df_test_muscular['target1'].values
    
    if verbose:
        print("   Preparing features for Model 2 (Skeletal)...")
    X_train_skeletal = prepare_data(df_train_skeletal, cache_file=cache_file_skeletal_train, use_cache=USE_CACHE)
    y_train_skeletal = df_train_skeletal['target2'].values
    X_test_skeletal = prepare_data(df_test_skeletal, cache_file=cache_file_skeletal_test, use_cache=USE_CACHE)
    y_test_skeletal = df_test_skeletal['target2'].values
    
    # Align features
    X_train_muscular, X_test_muscular = align_features(X_train_muscular, X_test_muscular)
    X_train_skeletal, X_test_skeletal = align_features(X_train_skeletal, X_test_skeletal)
    
    # Filter to feature subset
    if verbose:
        print(f"\n   Filtering to {len(feature_subset)} requested features...")
    X_train_muscular, X_test_muscular = filter_features(X_train_muscular, X_test_muscular, feature_subset)
    X_train_skeletal, X_test_skeletal = filter_features(X_train_skeletal, X_test_skeletal, feature_subset)
    
    # Get actual features used (intersection of both models)
    features_used = sorted(list(set(X_train_muscular.columns) & set(X_train_skeletal.columns)))
    
    if verbose:
        print(f"   ✅ Using {len(features_used)} features")
    
    # Train Model 1 (Muscular)
    if verbose:
        print(f"\n🚀 Training Model 1 (Muscular Injuries)...")
    
    lgbm_model1 = LGBMClassifier(
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
    
    lgbm_model1.fit(X_train_muscular, y_train_muscular)
    train_metrics1 = evaluate_model(lgbm_model1, X_train_muscular, y_train_muscular, "Training")
    test_metrics1 = evaluate_model(lgbm_model1, X_test_muscular, y_test_muscular, "Test")
    
    # Train Model 2 (Skeletal)
    if verbose:
        print(f"\n🚀 Training Model 2 (Skeletal Injuries)...")
    
    lgbm_model2 = LGBMClassifier(
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
    
    lgbm_model2.fit(X_train_skeletal, y_train_skeletal)
    train_metrics2 = evaluate_model(lgbm_model2, X_train_skeletal, y_train_skeletal, "Training")
    test_metrics2 = evaluate_model(lgbm_model2, X_test_skeletal, y_test_skeletal, "Test")
    
    # Convert numpy types for JSON serialization
    train_metrics1 = convert_numpy_types(train_metrics1)
    test_metrics1 = convert_numpy_types(test_metrics1)
    train_metrics2 = convert_numpy_types(train_metrics2)
    test_metrics2 = convert_numpy_types(test_metrics2)
    
    return {
        'model1': lgbm_model1,
        'model2': lgbm_model2,
        'train_metrics1': train_metrics1,
        'test_metrics1': test_metrics1,
        'train_metrics2': train_metrics2,
        'test_metrics2': test_metrics2,
        'feature_names_used': features_used
    }
