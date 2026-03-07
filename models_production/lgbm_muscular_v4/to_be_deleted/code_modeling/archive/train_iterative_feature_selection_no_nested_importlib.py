#!/usr/bin/env python3
"""
Iterative Feature Selection Training (No Nested Importlib)

This script avoids nested importlib calls by:
1. Using regular Python imports where possible
2. Including the training function directly
3. Only using importlib for the timeline module (single level)

This script:
1. Loads ranked features from feature_ranking.json
2. Trains models iteratively with increasing feature sets (20, 40, 60, ...)
3. Tracks performance metrics (Gini and F1-Score on test set)
4. Stops when 3 consecutive drops in performance are detected
5. Identifies the optimal number of features

Performance Metric: weighted combination of Gini coefficient and F1-Score
    combined_score = gini_weight * gini + f1_weight * f1_score
"""

import sys
# NOTE: stdout/stderr wrapping removed to avoid conflicts with importlib and Cursor's terminal
# The script will work without explicit encoding wrapping

import os
import json
import traceback
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
MODEL_OUTPUT_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'models'
TIMELINES_DIR = SCRIPT_DIR.parent / 'timelines'
CACHE_DIR = ROOT_DIR / 'cache'

# Add to sys.path for regular imports
sys.path.insert(0, str(SCRIPT_DIR.resolve()))
sys.path.insert(0, str(TIMELINES_DIR.resolve()))

# ========== CONFIGURATION ==========
FEATURES_PER_ITERATION = 20
INITIAL_FEATURES = 20
CONSECUTIVE_DROPS_THRESHOLD = 3
PERFORMANCE_DROP_THRESHOLD = 0.001  # Minimum drop to count as a drop (0.1%)
GINI_WEIGHT = 0.6
F1_WEIGHT = 0.4
RANKING_FILE = MODEL_OUTPUT_DIR / 'feature_ranking.json'
RESULTS_FILE = MODEL_OUTPUT_DIR / 'iterative_feature_selection_results.json'
LOG_FILE = MODEL_OUTPUT_DIR / 'iterative_training.log'

# Training configuration (from train_lgbm_v4_dual_targets_natural.py)
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
        f.write(f"=== Iterative Training Log Started at {datetime.now().isoformat()} ===\n")
except Exception as e:
    print(f"Warning: Could not initialize log file: {e}")

def log_message(message, level="INFO"):
    """Log a message to both console and log file"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] [{level}] {message}"
    print(log_entry)
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')
    except Exception as e:
        # If logging fails, at least print to console
        print(f"Warning: Could not write to log file: {e}")

def log_error(message, exception=None):
    """Log an error message with optional exception details"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] [ERROR] {message}"
    if exception:
        log_entry += f"\nException: {str(exception)}"
        log_entry += f"\nTraceback:\n{traceback.format_exc()}"
    print(log_entry)
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')
    except Exception:
        pass

# Import timeline filter function (single importlib call, not nested)
log_message("Importing timeline filter function")
try:
    import importlib.util
    timeline_script_path = TIMELINES_DIR / 'create_35day_timelines_v4_enhanced.py'
    if not timeline_script_path.exists():
        raise FileNotFoundError(f"Timeline script not found: {timeline_script_path}")
    
    # Use a consistent module name that train_lgbm_v4_dual_targets_natural will also use
    module_name = "create_35day_timelines_v4_enhanced"
    
    # Check if already imported
    if module_name in sys.modules:
        log_message("Timeline module already in sys.modules, reusing")
        timeline_module = sys.modules[module_name]
    else:
        spec = importlib.util.spec_from_file_location(module_name, timeline_script_path)
        timeline_module = importlib.util.module_from_spec(spec)
        # Store in sys.modules so train_lgbm_v4_dual_targets_natural can reuse it
        sys.modules[module_name] = timeline_module
        spec.loader.exec_module(timeline_module)
        log_message("Timeline module executed and stored in sys.modules")
    
    filter_timelines_for_model = timeline_module.filter_timelines_for_model
    log_message("Successfully imported filter_timelines_for_model")
except Exception as e:
    log_error("Failed to import timeline filter function", e)
    raise

# Import functions from train_lgbm_v4_dual_targets_natural using importlib (but only one level)
log_message("Importing training functions from train_lgbm_v4_dual_targets_natural")
try:
    # Use importlib to import the training module (this avoids execution conflicts)
    training_script_path = SCRIPT_DIR / 'train_lgbm_v4_dual_targets_natural.py'
    if not training_script_path.exists():
        raise FileNotFoundError(f"Training script not found: {training_script_path}")
    
    log_message(f"Found training script at: {training_script_path}")
    spec_training = importlib.util.spec_from_file_location("train_lgbm_v4_dual_targets_natural", training_script_path)
    
    if spec_training is None or spec_training.loader is None:
        raise ImportError(f"Could not create spec for training module: {training_script_path}")
    
    training_module = importlib.util.module_from_spec(spec_training)
    log_message("Executing training module to load functions")
    
    # Execute the module - this will trigger its importlib call for timeline module
    # but we've already imported that, so it should reuse it
    spec_training.loader.exec_module(training_module)
    log_message("Training module executed successfully")
    
    # Get the functions we need
    load_combined_seasonal_datasets_natural = training_module.load_combined_seasonal_datasets_natural
    load_test_dataset = training_module.load_test_dataset
    prepare_data = training_module.prepare_data
    align_features = training_module.align_features
    evaluate_model = training_module.evaluate_model
    convert_numpy_types = training_module.convert_numpy_types
    
    log_message("Successfully imported training functions")
except Exception as e:
    log_error("Failed to import training functions", e)
    raise

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
        log_message(f"\n{'='*80}")
        log_message(f"TRAINING WITH {len(feature_subset)} FEATURES")
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
    
    # Filter for each model
    try:
        df_train_muscular = filter_timelines_for_model(df_train_all, 'target1')
        df_test_muscular = filter_timelines_for_model(df_test_all, 'target1')
        df_train_skeletal = filter_timelines_for_model(df_train_all, 'target2')
        df_test_skeletal = filter_timelines_for_model(df_test_all, 'target2')
    except Exception as e:
        log_error("Failed to filter timelines", e)
        raise
    
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
        log_message("   Preparing features for Model 1 (Muscular)...")
    try:
        X_train_muscular = prepare_data(df_train_muscular, cache_file=cache_file_muscular_train, use_cache=USE_CACHE)
        y_train_muscular = df_train_muscular['target1'].values
        X_test_muscular = prepare_data(df_test_muscular, cache_file=cache_file_muscular_test, use_cache=USE_CACHE)
        y_test_muscular = df_test_muscular['target1'].values
    except Exception as e:
        log_error("Failed to prepare muscular features", e)
        raise
    
    if verbose:
        log_message("   Preparing features for Model 2 (Skeletal)...")
    try:
        X_train_skeletal = prepare_data(df_train_skeletal, cache_file=cache_file_skeletal_train, use_cache=USE_CACHE)
        y_train_skeletal = df_train_skeletal['target2'].values
        X_test_skeletal = prepare_data(df_test_skeletal, cache_file=cache_file_skeletal_test, use_cache=USE_CACHE)
        y_test_skeletal = df_test_skeletal['target2'].values
    except Exception as e:
        log_error("Failed to prepare skeletal features", e)
        raise
    
    # Align features
    try:
        X_train_muscular, X_test_muscular = align_features(X_train_muscular, X_test_muscular)
        X_train_skeletal, X_test_skeletal = align_features(X_train_skeletal, X_test_skeletal)
    except Exception as e:
        log_error("Failed to align features", e)
        raise
    
    # Filter to feature subset
    if verbose:
        log_message(f"\n   Filtering to {len(feature_subset)} requested features...")
    try:
        X_train_muscular, X_test_muscular = filter_features(X_train_muscular, X_test_muscular, feature_subset)
        X_train_skeletal, X_test_skeletal = filter_features(X_train_skeletal, X_test_skeletal, feature_subset)
    except Exception as e:
        log_error("Failed to filter features", e)
        raise
    
    # Get actual features used (intersection of both models)
    features_used = sorted(list(set(X_train_muscular.columns) & set(X_train_skeletal.columns)))
    
    if verbose:
        log_message(f"   ✅ Using {len(features_used)} features")
    
    # Train Model 1 (Muscular)
    if verbose:
        log_message(f"\n🚀 Training Model 1 (Muscular Injuries)...")
    
    try:
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
    except Exception as e:
        log_error("Failed to train Model 1 (Muscular)", e)
        raise
    
    # Train Model 2 (Skeletal)
    if verbose:
        log_message(f"\n🚀 Training Model 2 (Skeletal Injuries)...")
    
    try:
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
    except Exception as e:
        log_error("Failed to train Model 2 (Skeletal)", e)
        raise
    
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

def calculate_combined_score(test_metrics_muscular, test_metrics_skeletal, 
                            gini_weight=GINI_WEIGHT, f1_weight=F1_WEIGHT):
    """
    Calculate weighted combination of Gini and F1-Score.
    Average across both models (muscular and skeletal).
    
    Args:
        test_metrics_muscular: Test metrics for muscular model
        test_metrics_skeletal: Test metrics for skeletal model
        gini_weight: Weight for Gini coefficient (default 0.6)
        f1_weight: Weight for F1-Score (default 0.4)
    
    Returns:
        Combined performance score
    """
    try:
        muscular_score = (
            gini_weight * test_metrics_muscular['gini'] + 
            f1_weight * test_metrics_muscular['f1']
        )
        skeletal_score = (
            gini_weight * test_metrics_skeletal['gini'] + 
            f1_weight * test_metrics_skeletal['f1']
        )
        
        # Average across both models
        combined_score = (muscular_score + skeletal_score) / 2.0
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
    """Main function to run iterative feature selection training"""
    log_message("="*80)
    log_message("ITERATIVE FEATURE SELECTION TRAINING")
    log_message("="*80)
    log_message(f"\n📋 Configuration:")
    log_message(f"   Features per iteration: {FEATURES_PER_ITERATION}")
    log_message(f"   Initial features: {INITIAL_FEATURES}")
    log_message(f"   Stop after: {CONSECUTIVE_DROPS_THRESHOLD} consecutive drops")
    log_message(f"   Performance metric: {GINI_WEIGHT} * Gini + {F1_WEIGHT} * F1-Score")
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
        log_message(f"ITERATION {iteration}: Training with {n_features} features")
        log_message(f"{'='*80}")
        log_message(f"   Features: Top {n_features} from ranked list")
        
        iteration_start = datetime.now()
        
        try:
            log_message(f"Starting training for iteration {iteration} with {n_features} features")
            # Train models with this feature subset
            training_results = train_models_with_feature_subset(
                feature_subset, 
                verbose=True
            )
            log_message(f"Training completed for iteration {iteration}")
            
            # Validate training results
            required_keys = ['test_metrics1', 'test_metrics2', 'feature_names_used', 
                           'train_metrics1', 'train_metrics2']
            for key in required_keys:
                if key not in training_results:
                    raise KeyError(f"Missing key in training results: {key}")
            
            log_message("Calculating combined performance score")
            # Calculate combined performance score
            combined_score = calculate_combined_score(
                training_results['test_metrics1'],
                training_results['test_metrics2'],
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
                    'train': training_results['train_metrics1'],
                    'test': training_results['test_metrics1']
                },
                'model2_skeletal': {
                    'train': training_results['train_metrics2'],
                    'test': training_results['test_metrics2']
                },
                'combined_score': float(combined_score),
                'timestamp': iteration_start.isoformat(),
                'training_time_seconds': (datetime.now() - iteration_start).total_seconds()
            }
            
            results['iterations'].append(iteration_data)
            
            # Print iteration summary
            log_message(f"\n📊 Iteration {iteration} Results:")
            log_message(f"   Features: {n_features}")
            log_message(f"   Model 1 (Muscular) - Test: Gini={training_results['test_metrics1']['gini']:.4f}, "
                      f"F1={training_results['test_metrics1']['f1']:.4f}")
            log_message(f"   Model 2 (Skeletal) - Test: Gini={training_results['test_metrics2']['gini']:.4f}, "
                      f"F1={training_results['test_metrics2']['f1']:.4f}")
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
        log_message(f"   Model 2 (Skeletal) - Test:")
        log_message(f"      Gini: {best_iter_data['model2_skeletal']['test']['gini']:.4f}")
        log_message(f"      F1-Score: {best_iter_data['model2_skeletal']['test']['f1']:.4f}")
        log_message(f"      ROC AUC: {best_iter_data['model2_skeletal']['test']['roc_auc']:.4f}")
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
            plt.title('Performance vs Number of Features', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            plot_file = MODEL_OUTPUT_DIR / 'iterative_feature_selection_plot.png'
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
