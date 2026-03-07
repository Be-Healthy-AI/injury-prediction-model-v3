#!/usr/bin/env python3
"""
Evaluate the production LGBM (model_muscular_lgbm) on the new test dataset.

Loads model.joblib and columns.json from model_muscular_lgbm/, loads the current
test CSV (data/timelines/test/), prepares features with the same pipeline as training,
and reports test Gini (and combined score) for direct comparison with the previous
~60% Gini on the old test set.

Usage:
  python evaluate_production_lgbm_on_new_test.py

Once the new test Gini is recorded, this script can be deprecated.
"""

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Paths (same layout as deploy and iterative script)
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
ROOT_DIR = SCRIPT_DIR.parent.parent.parent.parent
PRODUCTION_DIR = ROOT_DIR / "models_production" / "lgbm_muscular_v4" / "model_muscular_lgbm"
TEST_DIR = ROOT_DIR / "models_production" / "lgbm_muscular_v4" / "data" / "timelines" / "test"
CACHE_DIR = ROOT_DIR / "cache"

# Previous reported test Gini on old test (from MODEL_METADATA / deployment)
PREVIOUS_TEST_GINI_OLD_TEST = 0.598


def main():
    # Import after path setup so that the training module sees correct ROOT_DIR when it runs its own
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

    model_path = PRODUCTION_DIR / "model.joblib"
    columns_path = PRODUCTION_DIR / "columns.json"

    if not model_path.exists():
        log_error(f"Production model not found: {model_path}")
        return 1
    if not columns_path.exists():
        log_error(f"Production columns not found: {columns_path}")
        return 1

    log_message("\n" + "=" * 80)
    log_message("PRODUCTION LGBM ON NEW TEST DATASET")
    log_message("=" * 80)

    # Load production model and feature list (columns.json is a JSON array)
    log_message(f"\nLoading production model: {model_path}")
    model = joblib.load(model_path)
    with open(columns_path, "r", encoding="utf-8") as f:
        columns_data = json.load(f)
    if isinstance(columns_data, list):
        feature_list = columns_data
    else:
        feature_list = columns_data.get("features", columns_data)
    model_features = list(getattr(model, "feature_name_", None) or feature_list)
    log_message(f"Model has {len(model_features)} features")

    # Load new test dataset (use same labeled test file as Exp 12 training)
    import train_iterative_feature_selection_muscular_standalone as _tmod
    _tmod.USE_LABELED_TIMELINES = True
    _tmod.LABELED_SUFFIX = "_v4_labeled_muscle_skeletal_only_d7.csv"
    _tmod.TARGET_COLUMN = "target1"
    log_message("\nLoading new test dataset...")
    df_test_all = load_test_dataset()
    df_test_muscular = filter_timelines_for_model(df_test_all, "target1")
    df_test_muscular = df_test_muscular.reset_index(drop=True)

    # Prepare features (same pipeline as training but drop_first=False so encoding matches production model columns.json)
    cache_suffix = hashlib.md5(str(sorted(model_features)).encode()).hexdigest()[:8]
    cache_file_test = str(cache_dir / f"preprocessed_production_lgbm_test_{cache_suffix}_drop_first_false.csv")
    log_message("Preparing test features (drop_first=False to match production model)...")
    X_test_full = prepare_data(df_test_muscular, cache_file=cache_file_test, use_cache=USE_CACHE, drop_first=False)
    y_test = df_test_muscular["target1"].values

    # Align to model: missing features -> zeros
    missing = [f for f in model_features if f not in X_test_full.columns]
    if missing:
        log_message(f"   Adding {len(missing)} missing feature(s) as zeros (categories absent in test).")
        for f in missing:
            X_test_full[f] = 0
    X_test = X_test_full[model_features].copy()

    # Convert to numeric (match production: booleans/objects -> float for LightGBM)
    for col in X_test.columns:
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0.0)

    # Evaluate
    log_message("\nEvaluating production LGBM on new test set...")
    test_metrics = evaluate_model(model, X_test, y_test, "Test (new)")
    combined_score = calculate_combined_score(
        test_metrics, gini_weight=GINI_WEIGHT, f1_weight=F1_WEIGHT
    )

    # Export test predictions for comparison with deployment (all players)
    proba_test = model.predict_proba(X_test)[:, 1]
    export_df = df_test_muscular[["player_id", "reference_date"]].copy()
    export_df["predicted_probability"] = proba_test
    export_df["target1"] = y_test
    export_path = ROOT_DIR / "models_production" / "lgbm_muscular_v4" / "test_predictions_from_training_pipeline.csv"
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(export_path, index=False)
    log_message(f"\nExported test predictions to: {export_path}")
    log_message(f"   Rows: {len(export_df):,} (player_id, reference_date, predicted_probability, target1)")

    # Summary and comparison
    gini_new = test_metrics["gini"]
    log_message("\n" + "=" * 80)
    log_message("TEST GINI COMPARISON")
    log_message("=" * 80)
    log_message(f"   Previous (old test set):  Gini = {PREVIOUS_TEST_GINI_OLD_TEST:.4f}")
    log_message(f"   Current  (new test set):  Gini = {gini_new:.4f}")
    log_message(f"   Combined (0.6*Gini+0.4*F1): {combined_score:.4f}")
    log_message("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
