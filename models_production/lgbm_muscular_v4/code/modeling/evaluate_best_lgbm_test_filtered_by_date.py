#!/usr/bin/env python3
"""
Evaluate the 500-feature best-iteration LGBM on test with negatives-only date filter.

Loads the model from models/lgbm_muscular_best_iteration.joblib (and feature list from
lgbm_muscular_best_iteration_features.json), loads the test CSV unchanged, filters
only negatives in memory to reference_date <= 2025-11-01 (all positives kept), then
prepares features and evaluates. Reports test Gini and compares to the full-test Gini (0.476).

Usage:
  python evaluate_best_lgbm_test_filtered_by_date.py

The test dataset file is not modified; filtering is applied only at evaluation time.
"""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import joblib
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
ROOT_DIR = SCRIPT_DIR.parent.parent.parent.parent
MODEL_OUTPUT_DIR = ROOT_DIR / "models_production" / "lgbm_muscular_v4" / "models"
CACHE_DIR = ROOT_DIR / "cache"

# Full-test Gini from iterative run (best iteration 25, 500 features)
FULL_TEST_GINI_REF = 0.476
MAX_REFERENCE_DATE = "2025-11-01"


def main():
    from train_iterative_feature_selection_muscular_standalone import (
        CACHE_DIR as _CACHE_DIR,
        load_test_dataset,
        filter_timelines_for_model,
        prepare_data,
        evaluate_model,
        calculate_combined_score,
        GINI_WEIGHT,
        F1_WEIGHT,
        USE_CACHE,
        log_message,
        log_error,
    )

    cache_dir = _CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_OUTPUT_DIR / "lgbm_muscular_best_iteration.joblib"
    features_path = MODEL_OUTPUT_DIR / "lgbm_muscular_best_iteration_features.json"

    if not model_path.exists():
        log_error(f"Best-iteration model not found: {model_path}. Run --export-best first.")
        return 1
    if not features_path.exists():
        log_error(f"Best-iteration features not found: {features_path}. Run --export-best first.")
        return 1

    log_message("\n" + "=" * 80)
    log_message("BEST 500-FEATURE LGBM ON TEST (negatives: ref_date <= 2025-11-01)")
    log_message("=" * 80)

    log_message(f"\nLoading best-iteration model: {model_path}")
    model = joblib.load(model_path)
    with open(features_path, "r", encoding="utf-8") as f:
        columns_data = json.load(f)
    if isinstance(columns_data, list):
        feature_list = columns_data
    else:
        feature_list = columns_data.get("features", columns_data)
    model_features = list(getattr(model, "feature_name_", None) or feature_list)
    log_message(f"Model has {len(model_features)} features")

    log_message("\nLoading test dataset (file unchanged)...")
    df_test_all = load_test_dataset()
    df_test_muscular = filter_timelines_for_model(df_test_all, "target1")
    df_test_muscular = df_test_muscular.reset_index(drop=True)
    n_full = len(df_test_muscular)

    if "reference_date" not in df_test_muscular.columns:
        log_error("Test dataset has no 'reference_date' column.")
        return 1

    df_test_muscular["reference_date"] = pd.to_datetime(
        df_test_muscular["reference_date"], errors="coerce"
    )
    max_ref = pd.Timestamp(MAX_REFERENCE_DATE)
    positives = df_test_muscular[df_test_muscular["target1"] == 1].copy()
    negatives = df_test_muscular[df_test_muscular["target1"] == 0].copy()
    negatives_filtered = negatives[negatives["reference_date"] <= max_ref]
    df_filtered = pd.concat([positives, negatives_filtered], ignore_index=True)
    df_filtered = df_filtered.reset_index(drop=True)
    n_filtered = len(df_filtered)
    n_pos = len(positives)
    n_neg_before = len(negatives)
    n_neg_after = len(negatives_filtered)
    log_message(f"   Full test rows: {n_full:,}")
    log_message(f"   Positives: all kept ({n_pos:,})")
    log_message(f"   Negatives: filtered to ref_date <= {MAX_REFERENCE_DATE} ({n_neg_after:,} of {n_neg_before:,})")
    log_message(f"   Filtered test total: {n_filtered:,} rows")

    cache_suffix = hashlib.md5(
        (str(sorted(model_features)) + "_neg_only_" + MAX_REFERENCE_DATE.replace("-", "")
    ).encode()
    ).hexdigest()[:12]
    cache_file_test = str(
        cache_dir / f"preprocessed_best_lgbm_test_neg_filtered_{cache_suffix}.csv"
    )
    log_message("Preparing features on filtered test...")
    X_test_full = prepare_data(
        df_filtered, cache_file=cache_file_test, use_cache=USE_CACHE
    )
    y_test = df_filtered["target1"].values

    missing = [f for f in model_features if f not in X_test_full.columns]
    if missing:
        log_message(f"   Adding {len(missing)} missing feature(s) as zeros.")
        for f in missing:
            X_test_full[f] = 0
    X_test = X_test_full[model_features]

    log_message("\nEvaluating on filtered test set...")
    test_metrics = evaluate_model(
        model, X_test, y_test, "Test (negatives: ref_date <= 2025-11-01)"
    )
    combined_score = calculate_combined_score(
        test_metrics, gini_weight=GINI_WEIGHT, f1_weight=F1_WEIGHT
    )

    gini_filtered = test_metrics["gini"]
    log_message("\n" + "=" * 80)
    log_message("GINI COMPARISON")
    log_message("=" * 80)
    log_message(f"   Full test (iter 25, 500 feat):  Gini = {FULL_TEST_GINI_REF:.4f}")
    log_message(f"   Negatives-only filtered:       Gini = {gini_filtered:.4f}")
    log_message(f"   Combined (0.6*Gini+0.4*F1): {combined_score:.4f}")
    log_message("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
