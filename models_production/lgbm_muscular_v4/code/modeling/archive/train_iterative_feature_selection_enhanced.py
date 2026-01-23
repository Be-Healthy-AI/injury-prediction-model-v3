#!/usr/bin/env python3
"""
Iterative Feature Selection Training (Enhanced with Logging)

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
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# Add paths for imports
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent.parent.parent
MODEL_OUTPUT_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'models'

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

# Import training function with error handling
log_message("Starting script initialization")
log_message(f"Script directory: {SCRIPT_DIR}")
log_message(f"Model output directory: {MODEL_OUTPUT_DIR}")

try:
    log_message("Attempting to import train_with_feature_subset module")
    import importlib.util
    training_subset_path = SCRIPT_DIR / 'train_with_feature_subset.py'
    
    if not training_subset_path.exists():
        raise FileNotFoundError(f"Training module not found: {training_subset_path}")
    
    log_message(f"Found training module at: {training_subset_path}")
    spec = importlib.util.spec_from_file_location("train_with_feature_subset", training_subset_path)
    
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create spec for module: {training_subset_path}")
    
    training_subset_module = importlib.util.module_from_spec(spec)
    log_message("Executing module to load training function")
    
    spec.loader.exec_module(training_subset_module)
    log_message("Module executed successfully")
    
    if not hasattr(training_subset_module, 'train_models_with_feature_subset'):
        raise AttributeError("Module does not have 'train_models_with_feature_subset' function")
    
    train_models_with_feature_subset = training_subset_module.train_models_with_feature_subset
    log_message("Successfully imported train_models_with_feature_subset function")
    
except Exception as e:
    log_error("Failed to import training module", e)
    raise

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
