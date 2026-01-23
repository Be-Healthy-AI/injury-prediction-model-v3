#!/usr/bin/env python3
"""
Train LGBM V3_natural model on PL-only filtered timelines excluding low-injury-rate seasons.

This version excludes:
- 2021-2022 (low injury rate: 0.32%)
- 2022-2023 (low injury rate: 0.34%)

Training includes: 2018-2019, 2019-2020, 2020-2021, 2023-2024, 2024-2025, 2025-2026

- Target: muscular injuries only, 35-day horizon
- Data: Natural ratio, recent seasons (2018-2026) excluding 2021-2022 and 2022-2023, PL-only timelines
- Correlation filtering: threshold = 0.8 (same as v1/v2)
- Model: LightGBM with same hyperparameters as v1/v2
"""

import io
import sys
import glob
import os

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import json
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
    prepare_data,
    sanitize_feature_name,
    serialize_column_names,
)


def load_filtered_seasonal_datasets(exclude_seasons=None):
    """
    Load natural ratio timeline files, excluding specified seasons.
    
    Args:
        exclude_seasons: List of seasons to exclude (e.g., ['2021_2022', '2022_2023'])
    
    Returns:
        Combined DataFrame
    """
    if exclude_seasons is None:
        exclude_seasons = []
    
    # Natural ratio: files without _XXpc suffix
    pattern = 'timelines_35day_season_*_v4_muscular.csv'
    files = glob.glob(pattern)
    season_files = []
    
    for filepath in files:
        filename = os.path.basename(filepath)
        # Exclude files with ratio suffixes (10pc, 25pc, 50pc)
        if '_10pc_' in filename or '_25pc_' in filename or '_50pc_' in filename:
            continue
        if 'season_' in filename:
            parts = filename.split('season_')
            if len(parts) > 1:
                # Extract season from pattern: timelines_35day_season_YYYY_YYYY_v4_muscular.csv
                season_part = parts[1].split('_v4_muscular')[0]
                # Exclude specified seasons
                if season_part not in exclude_seasons:
                    # Only include seasons >= 2018_2019
                    if season_part >= '2018_2019':
                        season_files.append((season_part, filepath))
    
    # Sort chronologically
    season_files.sort(key=lambda x: x[0])
    
    print(f"\n📂 Loading {len(season_files)} season files with natural (unbalanced) target ratio...")
    if exclude_seasons:
        print(f"   (Excluding seasons: {', '.join(exclude_seasons)})")
    print(f"   (Filtering: Only seasons >= 2018_2019)")
    
    dfs = []
    total_records = 0
    total_positives = 0
    
    for season_id, filepath in season_files:
        try:
            df = pd.read_csv(filepath, encoding='utf-8-sig', low_memory=False)
            positives = df['target'].sum()
            if len(df) > 0 and positives > 0:  # Skip empty files AND files with 0 positives
                dfs.append(df)
                total_records += len(df)
                total_positives += positives
                print(f"   ✅ {season_id}: {len(df):,} records ({positives:,} positives)")
            elif len(df) > 0 and positives == 0:
                print(f"   ⚠️  {season_id}: {len(df):,} records (0 positives - skipping)")
        except Exception as e:
            print(f"   ⚠️  Error loading {season_id}: {e}")
            continue
    
    if not dfs:
        raise ValueError("No valid season files found!")
    
    # Combine all dataframes
    print(f"\n📊 Combining {len(dfs)} season datasets...")
    combined_df = pd.concat(dfs, ignore_index=True)
    
    print(f"✅ Combined dataset: {len(combined_df):,} records")
    print(f"   Total positives: {total_positives:,} ({combined_df['target'].mean():.2%})")
    print(f"   Total negatives: {len(combined_df) - total_positives:,}")
    
    return combined_df


def main() -> None:
    # ========= CONFIGURATION =========
    TARGET_RATIO = None  # Natural ratio
    CORR_THRESHOLD = 0.8
    EXCLUDE_SEASONS = ['2021_2022', '2022_2023']  # Exclude low-injury-rate seasons

    RANDOM_STATE = 42

    USE_CACHE = True
    CACHE_DIR = "cache"
    PREPROCESS_CACHE = True
    # ================================

    ratio_display = "natural (unbalanced)"
    ratio_title = "NATURAL TARGET RATIO - LGBM V3 PL-ONLY (2018-2026 EXCLUDING 2021-2022 & 2022-2023)"

    print("=" * 80)
    print(f"LGBM V3 TRAINING - PL-ONLY TIMELINES ({ratio_title})")
    print("=" * 80)
    print("\n📋 Dataset Configuration:")
    print("   Training: Recent seasons (2018-2026) excluding 2021-2022 and 2022-2023")
    print("   Filter: PL-only timelines (only days when players were at PL clubs)")
    print("   Validation: Internal metrics on full training data")
    print("   Target: Muscular injuries only")
    print(f"   Target ratio: {ratio_display}")
    print(f"   Excluded seasons: {', '.join(EXCLUDE_SEASONS)} (low injury rates)")
    print(f"   Approach: Baseline LGBM with correlation filtering (threshold={CORR_THRESHOLD})")
    print("=" * 80)

    start_time = datetime.now()
    print(f"\n⏰ Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Estimated total time: ~10-20 minutes (depending on hardware)\n")

    # ------------------------------------------------------------------
    # 1) LOAD FILTERED SEASONAL DATASETS
    # ------------------------------------------------------------------
    print("\n📂 Loading PL-only timeline data (Recent seasons excluding low-injury-rate seasons)...")
    
    # Change to V3 timelines directory to load filtered files
    original_cwd = os.getcwd()
    v3_root = PROJECT_ROOT / "models_production" / "lgbm_muscular_v3"
    timelines_dir = v3_root / "data" / "timelines" / "train"
    
    if not timelines_dir.exists():
        print(f"ERROR: V3 timelines directory not found: {timelines_dir}")
        return
    
    os.chdir(str(timelines_dir))
    try:
        df_all = load_filtered_seasonal_datasets(exclude_seasons=EXCLUDE_SEASONS)
    finally:
        os.chdir(original_cwd)

    print(f"\n✅ Combined PL-only dataset (Filtered recent seasons): {len(df_all):,} records")
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
        cache_file = f"{CACHE_DIR}/preprocessed_v3_pl_only_natural_filtered_{hash_key}.csv"
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
    print("TRAIN FINAL LGBM MODEL ON FILTERED PL-ONLY DATA (v3_natural_filtered)")
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

    print(f"\n🚀 Training final LGBM v3_natural_filtered model on filtered PL-only data...")
    final_start = datetime.now()
    final_model.fit(X_all, y_all)
    final_time = datetime.now() - final_start
    print(f"✅ Final training completed in {final_time}")

    # Classic metrics on full training data
    print("\n📊 Evaluating classic metrics on ALL training data...")
    train_metrics = evaluate_model(final_model, X_all, y_all, "Filtered Seasons PL-Only (Train, natural_filtered)")

    # Save training metrics for auditability
    v3_model_dir = v3_root / "model_natural_filtered"
    v3_model_dir.mkdir(parents=True, exist_ok=True)
    
    train_metrics_path = v3_model_dir / "lgbm_v3_natural_filtered_pl_only_metrics_train.json"
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

    print(f"\n✅ Saved final LGBM v3_natural_filtered model to {model_path}")
    print(f"✅ Saved feature columns to {columns_path}")

    total_time = datetime.now() - start_time
    print(f"\n✅ Total execution time: {total_time}")
    print(f"\n🎯 V3_natural_filtered Model Summary:")
    print(
        "   - Trained on PL-only timelines (only days when players were at PL clubs)\n"
        "   - Recent seasons (2018-2026) excluding 2021-2022 and 2022-2023\n"
        "   - Same hyperparameters as v1/v2\n"
        f"   - Target ratio: {ratio_display}"
    )


if __name__ == "__main__":
    main()

