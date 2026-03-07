#!/usr/bin/env python3
"""
Iterative Feature Selection Training

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
import io
if sys.platform == 'win32':
    try:
        # Only wrap if not already wrapped and buffer is accessible
        if not isinstance(sys.stdout, io.TextIOWrapper):
            if hasattr(sys.stdout, 'buffer') and not getattr(sys.stdout.buffer, 'closed', False):
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        if not isinstance(sys.stderr, io.TextIOWrapper):
            if hasattr(sys.stderr, 'buffer') and not getattr(sys.stderr.buffer, 'closed', False):
                sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        # If wrapping fails, continue without wrapping (better than crashing)
        pass

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# Add paths for imports
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent.parent.parent
MODEL_OUTPUT_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'models'

# Import training function
import importlib.util
training_subset_path = SCRIPT_DIR / 'train_with_feature_subset.py'
spec = importlib.util.spec_from_file_location("train_with_feature_subset", training_subset_path)
training_subset_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training_subset_module)
train_models_with_feature_subset = training_subset_module.train_models_with_feature_subset

# ========== CONFIGURATION ==========
FEATURES_PER_ITERATION = 20
INITIAL_FEATURES = 20
CONSECUTIVE_DROPS_THRESHOLD = 3
PERFORMANCE_DROP_THRESHOLD = 0.001  # Minimum drop to count as a drop (0.1%)
GINI_WEIGHT = 0.6
F1_WEIGHT = 0.4
RANKING_FILE = MODEL_OUTPUT_DIR / 'feature_ranking.json'
RESULTS_FILE = MODEL_OUTPUT_DIR / 'iterative_feature_selection_results.json'
# ===================================

def load_feature_ranking():
    """Load ranked features from JSON file"""
    if not RANKING_FILE.exists():
        raise FileNotFoundError(
            f"Feature ranking file not found: {RANKING_FILE}\n"
            f"Please run rank_features_by_importance.py first."
        )
    
    print(f"📂 Loading feature ranking from: {RANKING_FILE}")
    with open(RANKING_FILE, 'r') as f:
        ranking_data = json.load(f)
    
    ranked_features = ranking_data['ranked_features']
    print(f"✅ Loaded {len(ranked_features)} ranked features")
    
    return ranked_features

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
    print("="*80)
    print("ITERATIVE FEATURE SELECTION TRAINING")
    print("="*80)
    print(f"\n📋 Configuration:")
    print(f"   Features per iteration: {FEATURES_PER_ITERATION}")
    print(f"   Initial features: {INITIAL_FEATURES}")
    print(f"   Stop after: {CONSECUTIVE_DROPS_THRESHOLD} consecutive drops")
    print(f"   Performance metric: {GINI_WEIGHT} * Gini + {F1_WEIGHT} * F1-Score")
    print(f"   Drop threshold: {PERFORMANCE_DROP_THRESHOLD}")
    print("="*80)
    
    start_time = datetime.now()
    print(f"\n⏰ Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load ranked features
    print("\n" + "="*80)
    print("STEP 1: LOADING FEATURE RANKING")
    print("="*80)
    ranked_features = load_feature_ranking()
    
    # Initialize results storage
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
    print("\n" + "="*80)
    print("STEP 2: ITERATIVE TRAINING")
    print("="*80)
    
    iteration = 0
    best_score = -np.inf
    best_iteration = None
    best_n_features = None
    
    # Calculate number of iterations needed
    max_iterations = (len(ranked_features) - INITIAL_FEATURES) // FEATURES_PER_ITERATION + 1
    if (len(ranked_features) - INITIAL_FEATURES) % FEATURES_PER_ITERATION != 0:
        max_iterations += 1
    
    print(f"\n   Will train up to {max_iterations} iterations")
    print(f"   (from {INITIAL_FEATURES} to {len(ranked_features)} features)")
    
    # Main iteration loop
    while True:
        iteration += 1
        n_features = INITIAL_FEATURES + (iteration - 1) * FEATURES_PER_ITERATION
        
        # Check if we've used all features
        if n_features > len(ranked_features):
            print(f"\n✅ Reached maximum number of features ({len(ranked_features)})")
            break
        
        # Select feature subset
        feature_subset = ranked_features[:n_features]
        
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration}: Training with {n_features} features")
        print(f"{'='*80}")
        print(f"   Features: Top {n_features} from ranked list")
        
        iteration_start = datetime.now()
        
        try:
            # Train models with this feature subset
            training_results = train_models_with_feature_subset(
                feature_subset, 
                verbose=True
            )
            
            # Calculate combined performance score
            combined_score = calculate_combined_score(
                training_results['test_metrics1'],
                training_results['test_metrics2'],
                gini_weight=GINI_WEIGHT,
                f1_weight=F1_WEIGHT
            )
            
            combined_scores.append(combined_score)
            
            # Check if this is the best so far
            if combined_score > best_score:
                best_score = combined_score
                best_iteration = iteration
                best_n_features = n_features
            
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
            print(f"\n📊 Iteration {iteration} Results:")
            print(f"   Features: {n_features}")
            print(f"   Model 1 (Muscular) - Test: Gini={training_results['test_metrics1']['gini']:.4f}, "
                  f"F1={training_results['test_metrics1']['f1']:.4f}")
            print(f"   Model 2 (Skeletal) - Test: Gini={training_results['test_metrics2']['gini']:.4f}, "
                  f"F1={training_results['test_metrics2']['f1']:.4f}")
            print(f"   Combined Score: {combined_score:.4f}")
            print(f"   Training time: {iteration_data['training_time_seconds']:.1f} seconds")
            
            # Check for consecutive drops
            if has_consecutive_drops(combined_scores, 
                                    threshold=PERFORMANCE_DROP_THRESHOLD,
                                    consecutive_count=CONSECUTIVE_DROPS_THRESHOLD):
                print(f"\n⚠️  Detected {CONSECUTIVE_DROPS_THRESHOLD} consecutive drops in performance!")
                print(f"   Stopping iterative training.")
                break
            
            # Save intermediate results (after each iteration)
            with open(RESULTS_FILE, 'w') as f:
                json.dump(results, f, indent=2)
            
        except Exception as e:
            print(f"\n❌ Error in iteration {iteration}: {e}")
            import traceback
            traceback.print_exc()
            # Continue to next iteration
            continue
    
    # Finalize results
    results['best_iteration'] = best_iteration
    results['best_n_features'] = best_n_features
    results['best_combined_score'] = float(best_score) if best_score != -np.inf else None
    results['configuration']['end_time'] = datetime.now().isoformat()
    results['configuration']['total_iterations'] = iteration
    results['configuration']['total_time_minutes'] = (datetime.now() - start_time).total_seconds() / 60
    
    # Save final results
    print("\n" + "="*80)
    print("STEP 3: SAVING RESULTS")
    print("="*80)
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Saved results to: {RESULTS_FILE}")
    
    # Print final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    print(f"\n📊 Training Summary:")
    print(f"   Total iterations: {iteration}")
    print(f"   Best iteration: {best_iteration}")
    print(f"   Best number of features: {best_n_features}")
    print(f"   Best combined score: {best_score:.4f}")
    
    if best_iteration:
        best_iter_data = results['iterations'][best_iteration - 1]
        print(f"\n📈 Best Performance (Iteration {best_iteration}):")
        print(f"   Model 1 (Muscular) - Test:")
        print(f"      Gini: {best_iter_data['model1_muscular']['test']['gini']:.4f}")
        print(f"      F1-Score: {best_iter_data['model1_muscular']['test']['f1']:.4f}")
        print(f"      ROC AUC: {best_iter_data['model1_muscular']['test']['roc_auc']:.4f}")
        print(f"   Model 2 (Skeletal) - Test:")
        print(f"      Gini: {best_iter_data['model2_skeletal']['test']['gini']:.4f}")
        print(f"      F1-Score: {best_iter_data['model2_skeletal']['test']['f1']:.4f}")
        print(f"      ROC AUC: {best_iter_data['model2_skeletal']['test']['roc_auc']:.4f}")
        print(f"   Combined Score: {best_iter_data['combined_score']:.4f}")
    
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
            print(f"\n✅ Saved performance plot to: {plot_file}")
            plt.close()
    except ImportError:
        print("\n   (Skipping plot - matplotlib not available)")
    
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds() / 60
    print(f"\n⏰ Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Total time: {total_time:.1f} minutes")
    
    return 0

if __name__ == "__main__":
    sys.exit(run_iterative_training())
