#!/usr/bin/env python3
"""
Train LGBM V3_natural model on PL-only filtered timelines using ONLY recent seasons (2018-2026).

This is an experiment to see if excluding older seasons improves precision while maintaining performance.

- Target: muscular injuries only, 35-day horizon
- Data: Natural ratio, recent seasons only (2018-2026), PL-only timelines
- Correlation filtering: threshold = 0.8 (same as v1/v2)
- Model: LightGBM with same hyperparameters as v1/v2
- Key difference: Only recent seasons (2018-2026) instead of all seasons
"""

import io
import sys

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import json
import os
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

# Add project root to path to import from scripts
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_models_seasonal_combined import (
    apply_correlation_filter,
    evaluate_model,
    load_combined_seasonal_datasets,
    prepare_data,
    sanitize_feature_name,
    serialize_column_names,
)


def main() -> None:
    # ========= CONFIGURATION =========
    TARGET_RATIO = None  # Natural ratio
    CORR_THRESHOLD = 0.8
    EXCLUDE_SEASON = None  # Include ALL seasons (no explicit hold-out)
    MIN_SEASON = '2018_2019'  # Only include seasons from 2018-2019 onwards

    RANDOM_STATE = 42

    USE_CACHE = True
    CACHE_DIR = "cache"
    PREPROCESS_CACHE = True
    # ================================

    ratio_display = "natural (unbalanced)"
    ratio_title = "NATURAL TARGET RATIO - LGBM V3 PL-ONLY (RECENT SEASONS 2018-2026 ONLY)"

    print("=" * 80)
    print(f"LGBM V3 TRAINING - PL-ONLY TIMELINES ({ratio_title})")
    print("=" * 80)
    print("\n📋 Dataset Configuration:")
    print("   Training: Recent seasons only (2018-2019 to 2025-2026) with natural target ratio")
    print("   Filter: PL-only timelines (only days when players were at PL clubs)")
    print("   Validation: Internal metrics on full training data")
    print("   Target: Muscular injuries only")
    print(f"   Target ratio: {ratio_display}")
    print(f"   Approach: Baseline LGBM with correlation filtering (threshold={CORR_THRESHOLD})")
    print("=" * 80)

    start_time = datetime.now()
    print(f"\n⏰ Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Estimated total time: ~10-20 minutes (depending on hardware)\n")

    # ------------------------------------------------------------------
    # 1) LOAD RECENT SEASONAL DATASETS (NATURAL RATIO, 2018-2026)
    # ------------------------------------------------------------------
    print("\n📂 Loading PL-only timeline data (Recent seasons 2018-2026, natural ratio)...")
    
    # Change to V3 timelines directory to load filtered files
    original_cwd = os.getcwd()
    v3_root = PROJECT_ROOT / "models_production" / "lgbm_muscular_v3"
    timelines_dir = v3_root / "data" / "timelines" / "train"
    
    if not timelines_dir.exists():
        print(f"ERROR: V3 timelines directory not found: {timelines_dir}")
        return
    
    os.chdir(str(timelines_dir))
    try:
        df_all = load_combined_seasonal_datasets(
            target_ratio=TARGET_RATIO,
            exclude_season=EXCLUDE_SEASON,  # None => include everything
            min_season=MIN_SEASON,  # Only seasons >= 2018_2019
        )
    finally:
        os.chdir(original_cwd)

    print(f"\n✅ Combined PL-only dataset (Recent seasons 2018-2026): {len(df_all):,} records")
    print(f"   Injury ratio: {df_all['target'].mean():.2%}")

    # ------------------------------------------------------------------
    # 2) PREPARE DATA (SAME PIPELINE AS v1/v2)
    # ------------------------------------------------------------------
    print("\n📊 Preparing data (single combined dataset)...")
    prep_start = datetime.now()

    # Create a cache filename for the combined dataset size if desired
    if PREPROCESS_CACHE:
        import hashlib
        hash_key = hashlib.md5(str(len(df_all)).encode()).hexdigest()[:8]
        cache_file = f"{CACHE_DIR}/preprocessed_v3_pl_only_natural_recent_{hash_key}.csv"
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
    # 4) TRAIN FINAL MODEL ON ALL DATA AND SAVE
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TRAIN FINAL LGBM MODEL ON RECENT SEASONS PL-ONLY DATA (v3_natural_recent)")
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

    print(f"\n🚀 Training final LGBM v3_natural_recent model on recent seasons PL-only data...")
    final_start = datetime.now()
    final_model.fit(X_all, y_all)
    final_time = datetime.now() - final_start
    print(f"✅ Final training completed in {final_time}")

    # Classic metrics on full training data
    print("\n📊 Evaluating classic metrics on ALL training data...")
    train_metrics = evaluate_model(final_model, X_all, y_all, "Recent Seasons PL-Only (Train, natural_recent)")

    # Save training metrics for auditability
    v3_model_dir = v3_root / "model_natural_recent"
    v3_model_dir.mkdir(parents=True, exist_ok=True)
    
    train_metrics_path = v3_model_dir / "lgbm_v3_natural_recent_pl_only_metrics_train.json"
    with open(train_metrics_path, "w", encoding="utf-8") as f:
        json.dump(train_metrics, f, indent=2)
    print(f"✅ Saved training metrics to {train_metrics_path}")

    # Serialize feature names
    column_names = serialize_column_names(X_all.columns)

    # Save model + columns
    model_path = v3_model_dir / "model.joblib"
    columns_path = v3_model_dir / "columns.json"

    joblib.dump(final_model, model_path)
    with open(columns_path, "w", encoding="utf-8") as f:
        json.dump(column_names, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved final LGBM v3_natural_recent model to {model_path}")
    print(f"✅ Saved feature columns to {columns_path}")

    total_time = datetime.now() - start_time
    print(f"\n✅ Total execution time: {total_time}")
    print(f"\n🎯 V3_natural_recent Model Summary:")
    print(
        "   - Trained on PL-only timelines (only days when players were at PL clubs)\n"
        "   - Recent seasons only (2018-2019 to 2025-2026)\n"
        "   - Same hyperparameters as v1/v2\n"
        f"   - Target ratio: {ratio_display}"
    )


if __name__ == "__main__":
    main()

