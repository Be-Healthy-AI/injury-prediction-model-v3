#!/usr/bin/env python3
"""
Train LGBM winner model (v2) on ALL seasons including 2025-2026
using internal cross-validation (no external hold-out test set).

- Target: muscular injuries only, 35-day horizon
- Data: 10% target ratio, seasonal combined datasets, ALL seasons
- Correlation filtering: threshold = 0.8 (same as v1)
- Model: LightGBM with same hyperparameters as audited v1

This script is intended to produce a production-focused model (v2)
trained on all available history up to 2025-12-05, with validation
coming from internal cross-validation only.
"""

import io
import sys

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold

from train_models_seasonal_combined import (  # type: ignore
    apply_correlation_filter,
    evaluate_model,
    load_combined_seasonal_datasets,
    prepare_data,
    sanitize_feature_name,
    serialize_column_names,
)


def main() -> None:
    # ========= CONFIGURATION =========
    TARGET_RATIO = 0.10  # 10% target ratio for training data
    CORR_THRESHOLD = 0.8
    EXCLUDE_SEASON = None  # Include ALL seasons (no explicit hold-out)
    MIN_SEASON = None  # Include all from earliest available

    N_FOLDS = 5  # kept for reference; can be reused if we re-enable CV
    RANDOM_STATE = 42

    USE_CACHE = True
    CACHE_DIR = "cache"
    PREPROCESS_CACHE = True
    # ================================

    ratio_display = f"{TARGET_RATIO:.0%} (balanced)"
    ratio_title = (
        f"{TARGET_RATIO:.0%} TARGET RATIO - LGBM ALL SEASONS (INCLUDING 2025-2026, CV ONLY)"
    )

    print("=" * 80)
    print(f"LGBM V2 TRAINING - V4 MUSCULAR INJURIES ONLY ({ratio_title})")
    print("=" * 80)
    print("\n📋 Dataset Configuration:")
    print(
        f"   Training: ALL seasons (including 2025-2026) with {TARGET_RATIO:.0%} target ratio"
    )
    print("   Validation: Internal StratifiedKFold cross-validation (no external hold-out)")
    print("   Test: None (previous 2025-2026 test season now part of training history)")
    print("   Target: Muscular injuries only")
    print(f"   Target ratio: {ratio_display}")
    print(f"   Approach: Baseline LGBM with correlation filtering (threshold={CORR_THRESHOLD})")
    print("=" * 80)

    start_time = datetime.now()
    print(f"\n⏰ Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Estimated total time: ~10-20 minutes (depending on hardware)\n")

    # ------------------------------------------------------------------
    # 1) LOAD ALL SEASONAL DATASETS (10% RATIO, INCLUDING 2025-2026)
    # ------------------------------------------------------------------
    print("\n📂 Loading timeline data (ALL seasons, 10% ratio)...")
    # IMPORTANT: we archived the original timelines; the canonical copies we want
    # to use for training now live under the v1 model bundle. To reuse the existing
    # loader (which expects files in the current working directory), temporarily
    # change into that folder while loading, then restore the original cwd.
    original_cwd = os.getcwd()
    timelines_dir = os.path.join(
        original_cwd,
        "models_production",
        "lgbm_muscular_v1",
        "data",
        "timelines",
        "train",
    )
    os.chdir(timelines_dir)
    try:
        df_all = load_combined_seasonal_datasets(
            target_ratio=TARGET_RATIO,
            exclude_season=EXCLUDE_SEASON,  # None => include everything
            min_season=MIN_SEASON,
        )
    finally:
        os.chdir(original_cwd)

    print(f"\n✅ Combined dataset (ALL seasons): {len(df_all):,} records")
    print(f"   Injury ratio: {df_all['target'].mean():.2%}")

    # ------------------------------------------------------------------
    # 2) PREPARE DATA (SAME PIPELINE AS v1)
    # ------------------------------------------------------------------
    print("\n📊 Preparing data (single combined dataset)...")
    prep_start = datetime.now()

    # Create a cache filename for the combined dataset size if desired
    if PREPROCESS_CACHE:
        import hashlib
        hash_key = hashlib.md5(str(len(df_all)).encode()).hexdigest()[:8]
        cache_file = f"{CACHE_DIR}/preprocessed_all_seasonal_{hash_key}.csv"
    else:
        cache_file = None

    X_all, y_all = prepare_data(df_all, cache_file=cache_file, use_cache=USE_CACHE)
    prep_time = datetime.now() - prep_start

    print(f"✅ Data preparation completed in {prep_time}")
    print(f"   Features: {X_all.shape[1]}")
    print(f"   Samples:  {X_all.shape[0]:,}")

    # ------------------------------------------------------------------
    # 3) SANITIZE FEATURE NAMES + CORRELATION FILTER
    # ------------------------------------------------------------------
    print("\n🔧 Sanitizing feature names for LightGBM compatibility...")
    sanitize_start = datetime.now()
    X_all.columns = [sanitize_feature_name(col) for col in X_all.columns]
    sanitize_time = datetime.now() - sanitize_start
    print(f"✅ Sanitized {len(X_all.columns)} feature names in {sanitize_time}")

    initial_feature_count = X_all.shape[1]

    print("\n🔎 Applying correlation filter on ALL data (training features only)...")
    corr_start = datetime.now()
    selected_features = apply_correlation_filter(
        X_all, CORR_THRESHOLD, cache_dir=CACHE_DIR, use_cache=USE_CACHE
    )
    X_all = X_all[selected_features]
    corr_time = datetime.now() - corr_start

    print(
        f"\n✅ After correlation filtering: {len(selected_features)} features "
        f"(removed {initial_feature_count - len(selected_features)})"
    )
    print(f"   Total correlation filtering time: {corr_time}")

    # ------------------------------------------------------------------
    # 4) TRAIN FINAL MODEL ON ALL DATA AND SAVE AS v2 (NO CV)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TRAIN FINAL LGBM MODEL ON ALL DATA (v2)")
    print("=" * 80)

    final_model = LGBMClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("\n🚀 Training final LGBM v2 model on ALL data...")
    final_start = datetime.now()
    final_model.fit(X_all, y_all)
    final_time = datetime.now() - final_start
    print(f"✅ Final training completed in {final_time}")

    # Classic metrics on full training data
    print("\n📊 Evaluating classic metrics on ALL training data...")
    train_metrics = evaluate_model(final_model, X_all, y_all, "All Seasons (Train)")

    # Save training metrics for auditability
    os.makedirs("models", exist_ok=True)
    train_metrics_path = (
        "models/lgbm_model_seasonal_10pc_v4_muscular_corr08_v2_allseasons_metrics_train.json"
    )
    with open(train_metrics_path, "w", encoding="utf-8") as f:
        json.dump(train_metrics, f, indent=2)
    print(f"✅ Saved training metrics to {train_metrics_path}")

    # Serialize feature names
    column_names = serialize_column_names(X_all.columns)

    # Save model + columns as v2 artifacts
    model_path = "models/lgbm_model_seasonal_10pc_v4_muscular_corr08_v2_allseasons.joblib"
    columns_path = (
        "models/lgbm_model_seasonal_10pc_v4_muscular_corr08_v2_allseasons_columns.json"
    )

    joblib.dump(final_model, model_path)
    with open(columns_path, "w", encoding="utf-8") as f:
        json.dump(column_names, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved final LGBM v2 model to {model_path}")
    print(f"✅ Saved feature columns to {columns_path}")

    total_time = datetime.now() - start_time
    print(f"\n✅ Total execution time: {total_time}")
    print("\n🎯 NOTE:")
    print(
        "   - This v2 model is trained on ALL seasons including 2025-2026 "
        "(no external hold-out test).\n"
        "   - Use CV metrics above to compare with v1, and then monitor "
        "v2 in production over time."
    )


if __name__ == "__main__":
    main()


