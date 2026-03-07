#!/usr/bin/env python3
"""
Experiment 4: Multi-algorithm comparison on Exp 3's 760-feature setup.

Uses the same data and 760-feature set as Exp 3 (14-day, optimize on test, best iteration 38).
Trains multiple algorithms with standard hyperparameters and compares train/val/test metrics.

Algorithms: Logistic Regression, Random Forest, Linear SVM (calibrated for proba),
Gradient Boosting (sklearn), LightGBM.

Output: experiment_4_multi_algorithm_results.json and console summary.
"""

import sys
import json
import hashlib
import argparse
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from lightgbm import LGBMClassifier

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent.parent.parent
MODEL_OUTPUT_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'models'
CACHE_DIR = ROOT_DIR / 'cache'
TRAIN_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'data' / 'timelines' / 'train'
TEST_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'data' / 'timelines' / 'test'

RESULTS_FILE = MODEL_OUTPUT_DIR / 'experiment_4_multi_algorithm_results.json'
LOG_FILE = MODEL_OUTPUT_DIR / 'experiment_4_multi_algorithm.log'
ITERATIVE_RESULTS_FILE = MODEL_OUTPUT_DIR / 'iterative_feature_selection_results_muscular.json'

MIN_SEASON = '2018_2019'
EXCLUDE_SEASON = '2025_2026'
TRAIN_VAL_RATIO = 0.8
SPLIT_RANDOM_STATE = 42
USE_CACHE = True
GINI_WEIGHT = 0.6
F1_WEIGHT = 0.4

# Import data loading and preprocessing from iterative script
sys.path.insert(0, str(SCRIPT_DIR))
import train_iterative_feature_selection_muscular_standalone as iterative

# Redirect iterative script's log to our log file
iterative.LOG_FILE = LOG_FILE
iterative.MODEL_OUTPUT_DIR = MODEL_OUTPUT_DIR
iterative.TRAIN_DIR = TRAIN_DIR
iterative.TEST_DIR = TEST_DIR
iterative.CACHE_DIR = CACHE_DIR
iterative.MIN_SEASON = MIN_SEASON
iterative.EXCLUDE_SEASON = EXCLUDE_SEASON
iterative.TRAIN_VAL_RATIO = TRAIN_VAL_RATIO
iterative.SPLIT_RANDOM_STATE = SPLIT_RANDOM_STATE
iterative.USE_CACHE = USE_CACHE


def get_760_feature_list():
    """Load the 760-feature list from Exp 3 (best iteration 38 from test-optimized results)."""
    if not ITERATIVE_RESULTS_FILE.exists():
        raise FileNotFoundError(f"Results file not found: {ITERATIVE_RESULTS_FILE}. Run Exp 3 first.")
    with open(ITERATIVE_RESULTS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    best_it = data.get('best_iteration')
    if best_it is None:
        # Fallback: use iteration 38 if best_* not set
        for it in data.get('iterations', []):
            if it.get('iteration') == 38:
                return it['features'], it.get('n_features', 760)
        raise ValueError("Could not find best iteration or iteration 38 in results.")
    for it in data['iterations']:
        if it['iteration'] == best_it:
            return it['features'], it.get('n_features', len(it['features']))
    raise ValueError(f"Best iteration {best_it} not found in results.")


def load_data_and_prepare(feature_list):
    """Load train/val/test and prepare 760-feature matrices (same as Exp 3)."""
    iterative.log_message("\n📂 Loading datasets...")
    df_pool = iterative.load_combined_seasonal_datasets_natural(
        min_season=MIN_SEASON, exclude_season=EXCLUDE_SEASON
    )
    df_test_all = iterative.load_test_dataset()
    df_pool_muscular = iterative.filter_timelines_for_model(df_pool, 'target1')
    df_test_muscular = iterative.filter_timelines_for_model(df_test_all, 'target1')
    iterative.log_message("   Splitting pool into train (80%) and validation (20%) by timeline...")
    df_train_80, df_val_20 = iterative.split_train_val_by_timeline(
        df_pool_muscular, ratio=TRAIN_VAL_RATIO, random_state=SPLIT_RANDOM_STATE
    )
    df_test_muscular = df_test_muscular.reset_index(drop=True)

    cache_suffix = hashlib.md5(str(sorted(feature_list)).encode()).hexdigest()[:8]
    cache_train = str(CACHE_DIR / f'preprocessed_muscular_train80_subset_{cache_suffix}.csv')
    cache_val = str(CACHE_DIR / f'preprocessed_muscular_val20_subset_{cache_suffix}.csv')
    cache_test = str(CACHE_DIR / f'preprocessed_muscular_test_subset_{cache_suffix}.csv')

    iterative.log_message("   Preparing features...")
    X_train = iterative.prepare_data(df_train_80, cache_file=cache_train, use_cache=USE_CACHE)
    y_train = df_train_80['target1'].values
    X_val = iterative.prepare_data(df_val_20, cache_file=cache_val, use_cache=USE_CACHE)
    y_val = df_val_20['target1'].values
    X_test = iterative.prepare_data(df_test_muscular, cache_file=cache_test, use_cache=USE_CACHE)
    y_test = df_test_muscular['target1'].values

    X_train, X_val, X_test = iterative.align_features_three(X_train, X_val, X_test)
    X_train, X_val, X_test = iterative.filter_features_three(X_train, X_val, X_test, feature_list)
    return X_train, y_train, X_val, y_val, X_test, y_test


def metrics_dict(y_true, y_pred, y_proba=None):
    """Compute metrics and confusion matrix; y_proba optional for ROC/Gini."""
    m = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        m['roc_auc'] = float(roc_auc_score(y_true, y_proba))
        m['gini'] = float(2 * m['roc_auc'] - 1)
    else:
        m['roc_auc'] = 0.0
        m['gini'] = 0.0
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        m['confusion_matrix'] = {'tn': int(cm[0, 0]), 'fp': int(cm[0, 1]), 'fn': int(cm[1, 0]), 'tp': int(cm[1, 1])}
    else:
        m['confusion_matrix'] = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
    m['combined_score'] = float(GINI_WEIGHT * m['gini'] + F1_WEIGHT * m['f1'])
    return m


def predict_proba_safe(model, X):
    """Get P(y=1); use predict_proba if available else decision_function scaled to [0,1]."""
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, 'decision_function'):
        d = model.decision_function(X)
        # Simple min-max scale to [0,1] for ROC
        d = np.clip(d, -10, 10)
        return (d - d.min()) / (d.max() - d.min() + 1e-9)
    return np.zeros(len(X)) + 0.5


def evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test, name):
    """Fit and evaluate one model; return dict of train/val/test metrics and training_time_seconds."""
    iterative.log_message(f"\n{'='*60}")
    iterative.log_message(f"  Training: {name}")
    iterative.log_message(f"{'='*60}")
    start_time = time.perf_counter()
    start_iso = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    iterative.log_message(f"   Started at {start_iso}")
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        iterative.log_error(f"Fit failed for {name}", e)
        return None
    elapsed = time.perf_counter() - start_time
    elapsed_min = elapsed / 60.0
    iterative.log_message(f"   Training completed in {elapsed:.1f}s ({elapsed_min:.1f} min)")

    def eval_set(X, y, label):
        y_pred = model.predict(X)
        y_proba = predict_proba_safe(model, X)
        return metrics_dict(y, y_pred, y_proba)

    train_m = eval_set(X_train, y_train, 'train')
    val_m = eval_set(X_val, y_val, 'val')
    test_m = eval_set(X_test, y_test, 'test')
    iterative.log_message(f"   Train: Gini={train_m['gini']:.4f} F1={train_m['f1']:.4f} combined={train_m['combined_score']:.4f}")
    iterative.log_message(f"   Val:   Gini={val_m['gini']:.4f} F1={val_m['f1']:.4f} combined={val_m['combined_score']:.4f}")
    iterative.log_message(f"   Test:  Gini={test_m['gini']:.4f} F1={test_m['f1']:.4f} combined={test_m['combined_score']:.4f}")
    return {
        'train': train_m,
        'validation': val_m,
        'test_2025_26': test_m,
        'training_time_seconds': round(elapsed, 1),
    }


def get_models(skip_linearsvc=False):
    """Return dict of name -> (model, description). Standard hyperparameters.
    If skip_linearsvc=True, LinearSVC_calibrated is omitted (saves hours on large data).
    Order: LR, RF, GB, LightGBM, then LinearSVC last so other results are available even if stopped during LinearSVC."""
    models = {
        'LogisticRegression': (
            LogisticRegression(
                max_iter=2000,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                solver='lbfgs',
            ),
            'sklearn LogisticRegression, balanced, lbfgs, max_iter=2000',
        ),
        'RandomForest': (
            RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=20,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
            ),
            'sklearn RandomForest, 100 trees, max_depth=10, balanced',
        ),
        'GradientBoosting': (
            GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                min_samples_leaf=20,
                random_state=42,
            ),
            'sklearn GradientBoosting, 100 estimators, max_depth=5',
        ),
        'LightGBM': (
            LGBMClassifier(
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
                verbose=-1,
            ),
            'LightGBM, same standard params as iterative script (Exp 3 baseline)',
        ),
    }
    if not skip_linearsvc:
        models['LinearSVC_calibrated'] = (
            CalibratedClassifierCV(
                LinearSVC(max_iter=3000, class_weight='balanced', random_state=42, dual=False),
                cv=3,
                method='isotonic',
            ),
            'LinearSVC + CalibratedClassifierCV(3-fold) for predict_proba',
        )
    # LinearSVC last so we have all other results if we stop during it
    order = ['LogisticRegression', 'RandomForest', 'GradientBoosting', 'LightGBM']
    if not skip_linearsvc:
        order.append('LinearSVC_calibrated')
    return {k: models[k] for k in order}


def main():
    parser = argparse.ArgumentParser(description='Experiment 4: Multi-algorithm comparison (760 features)')
    parser.add_argument('--skip-linearsvc', action='store_true',
                        help='Skip LinearSVC_calibrated (very slow on large data)')
    args = parser.parse_args()
    skip_linearsvc = args.skip_linearsvc

    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            f.write(f"=== Experiment 4 Multi-Algorithm (760 features) Started at {datetime.now().isoformat()} ===\n")
    except Exception as e:
        print(f"Warning: Could not init log: {e}")

    iterative.log_message("="*80)
    iterative.log_message("EXPERIMENT 4: Multi-algorithm comparison (760 features from Exp 3)")
    iterative.log_message("="*80)
    algo_list = "LogisticRegression, RandomForest, GradientBoosting, LightGBM"
    if not skip_linearsvc:
        algo_list += ", LinearSVC (calibrated) [last]"
    iterative.log_message(f"Algorithms (order): {algo_list}")
    iterative.log_message("Same data/split as Exp 3: 80/20 train/val by timeline, test 2025/26")
    if skip_linearsvc:
        iterative.log_message("Option: --skip-linearsvc set; LinearSVC_calibrated will be skipped.")

    feature_list, n_features = get_760_feature_list()
    iterative.log_message(f"\nLoaded {n_features} features from Exp 3 (iterative results).")

    X_train, y_train, X_val, y_val, X_test, y_test = load_data_and_prepare(feature_list)
    iterative.log_message(f"\n✅ Data ready: train {X_train.shape[0]:,} x {X_train.shape[1]}, val {X_val.shape[0]:,}, test {X_test.shape[0]:,}")

    models = get_models(skip_linearsvc=skip_linearsvc)
    config_start = datetime.now().isoformat()
    results = {
        'description': 'Experiment 4: Multi-algorithm comparison on Exp 3 760-feature setup (14-day labels, same train/val/test split).',
        'feature_source': 'iterative_feature_selection_results_muscular.json (best iteration, 760 features)',
        'n_features': n_features,
        'train_val_split': '80/20 by timeline (random rows), same as Exp 3',
        'holdout_season': '2025_2026',
        'gini_weight': GINI_WEIGHT,
        'f1_weight': F1_WEIGHT,
        'models': {},
        'configuration': {
            'start_time': config_start,
            'use_cache': USE_CACHE,
            'skip_linearsvc': skip_linearsvc,
        },
    }

    for name, (model, desc) in models.items():
        results['models'][name] = {'description': desc, 'metrics': None}
        metrics = evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test, name)
        if metrics is not None:
            results['models'][name]['metrics'] = metrics
        results['configuration']['end_time'] = datetime.now().isoformat()
        # Save after each model so partial results exist if we stop during LinearSVC
        with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

    results['configuration']['end_time'] = datetime.now().isoformat()

    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    iterative.log_message("\n" + "="*80)
    iterative.log_message("TRAINING TIMES (per algorithm)")
    iterative.log_message("="*80)
    total_sec = 0
    for name in results['models']:
        m = results['models'][name].get('metrics')
        if m and 'training_time_seconds' in m:
            sec = m['training_time_seconds']
            total_sec += sec
            mins = sec / 60.0
            iterative.log_message(f"   {name}: {sec:.1f}s ({mins:.1f} min)")
    iterative.log_message(f"   Total: {total_sec:.1f}s ({total_sec/60:.1f} min)")

    iterative.log_message("\n" + "="*80)
    iterative.log_message("SUMMARY: Test combined score (0.6*Gini + 0.4*F1)")
    iterative.log_message("="*80)
    for name in results['models']:
        m = results['models'][name].get('metrics')
        if m:
            sc = m['test_2025_26']['combined_score']
            g = m['test_2025_26']['gini']
            f1 = m['test_2025_26']['f1']
            iterative.log_message(f"   {name}: combined={sc:.4f} (Gini={g:.4f}, F1={f1:.4f})")
    iterative.log_message(f"\n✅ Results saved to: {RESULTS_FILE}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
