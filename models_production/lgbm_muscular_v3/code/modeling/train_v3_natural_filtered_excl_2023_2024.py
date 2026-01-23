#!/usr/bin/env python3
"""
Train LGBM V3 model on PL-only filtered timelines, excluding 2021-2022, 2022-2023, and 2023-2024 seasons.

This script trains a model using:
- Recent seasons (2018-2026) 
- Excluding 2021-2022, 2022-2023, and 2023-2024 (low injury rates)
- Natural target ratio (unbalanced)
- PL-only timelines (only days when players were at PL clubs)

- Target: muscular injuries only, 35-day horizon
- Correlation filtering: threshold = 0.8 (same as v1/v2)
- Model: LightGBM with same hyperparameters as v1/v2
"""

import io
import sys

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import json
import os
import glob
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
        exclude_seasons: List of seasons to exclude (e.g., ['2021_2022', '2022_2023', '2023_2024'])
    
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
    EXCLUDE_SEASONS = ['2021_2022', '2022_2023', '2023_2024']  # Exclude low-injury-rate seasons

    RANDOM_STATE = 42

    USE_CACHE = True
    CACHE_DIR = "cache"
    PREPROCESS_CACHE = True
    # ================================

    ratio_display = "natural (unbalanced)"
    ratio_title = "NATURAL TARGET RATIO - LGBM V3 PL-ONLY (2018-2026 EXCLUDING 2021-2022, 2022-2023 & 2023-2024)"

    print("=" * 80)
    print(f"LGBM V3 TRAINING - PL-ONLY TIMELINES ({ratio_title})")
    print("=" * 80)
    print("\n📋 Dataset Configuration:")
    print("   Training: Recent seasons (2018-2026) excluding 2021-2022, 2022-2023, and 2023-2024")
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
    # 2) PREPARE DATA
    # ------------------------------------------------------------------
    print("\n📊 Preparing data (single combined dataset)...")
    prep_start = datetime.now()
    
    # Use cache file name based on dataset hash
    cache_file = None
    if USE_CACHE and CACHE_DIR:
        import hashlib
        hash_key = hashlib.md5(str(len(df_all)).encode()).hexdigest()[:8]
        cache_file = f"{CACHE_DIR}/preprocessed_v3_pl_only_natural_filtered_excl_2023_2024_{hash_key}.csv"
    
    X, y = prepare_data(df_all, cache_file=cache_file, use_cache=USE_CACHE)
    
    prep_time = datetime.now() - prep_start
    print(f"✅ Data preparation completed in {prep_time}")
    print(f"   Features: {len(X.columns)}")
    print(f"   Samples:  {len(X):,}")

    # ------------------------------------------------------------------
    # 3) SANITIZE FEATURE NAMES
    # ------------------------------------------------------------------
    print("\n🔧 Sanitizing feature names for LightGBM compatibility...")
    sanitize_start = datetime.now()
    X.columns = [sanitize_feature_name(col) for col in X.columns]
    sanitize_time = datetime.now() - sanitize_start
    print(f"✅ Sanitized {len(X.columns)} feature names in {sanitize_time}")

    # ------------------------------------------------------------------
    # 4) APPLY CORRELATION FILTER
    # ------------------------------------------------------------------
    print("\n🔎 Applying correlation filter on ALL data (training features only)...")
    initial_feature_count = X.shape[1]
    selected_features = apply_correlation_filter(
        X, 
        threshold=CORR_THRESHOLD,
        cache_dir=CACHE_DIR,
        use_cache=USE_CACHE
    )
    X_filtered = X[selected_features]
    print(f"✅ After correlation filtering: {len(selected_features)} features (removed {initial_feature_count - len(selected_features)})")

    # ------------------------------------------------------------------
    # 5) TRAIN FINAL MODEL
    # ------------------------------------------------------------------
    model_suffix = "natural_filtered_excl_2023_2024"
    
    print("\n" + "=" * 80)
    print("TRAIN FINAL LGBM MODEL ON FILTERED PL-ONLY DATA (v3_natural_filtered_excl_2023_2024)")
    print("=" * 80)
    v3_model_dir = v3_root / f"model_{model_suffix}"
    v3_model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🚀 Training final LGBM v3_{model_suffix} model on filtered PL-only data...")
    train_start = datetime.now()
    
    model = LGBMClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        verbose=-1,
        n_jobs=-1
    )
    
    model.fit(X_filtered, y)
    
    train_time = datetime.now() - train_start
    print(f"✅ Final training completed in {train_time}")

    # ------------------------------------------------------------------
    # 6) EVALUATE ON TRAINING DATA
    # ------------------------------------------------------------------
    print("\n📊 Evaluating classic metrics on ALL training data...")
    train_metrics = evaluate_model(model, X_filtered, y, f"Filtered Seasons PL-ONLY (Train, {model_suffix})")
    
    # Save training metrics
    train_metrics_path = v3_model_dir / f"lgbm_v3_{model_suffix}_pl_only_metrics_train.json"
    with open(train_metrics_path, "w", encoding="utf-8") as f:
        json.dump(train_metrics, f, indent=2)
    print(f"✅ Saved training metrics to {train_metrics_path}")

    # ------------------------------------------------------------------
    # 7) SAVE MODEL
    # ------------------------------------------------------------------
    model_path = v3_model_dir / "model.joblib"
    joblib.dump(model, model_path)
    print(f"✅ Saved final LGBM v3_{model_suffix} model to {model_path}")
    
    # Save feature columns
    columns_path = v3_model_dir / "columns.json"
    with open(columns_path, "w", encoding="utf-8") as f:
        json.dump(X_filtered.columns.tolist(), f, indent=2)
    print(f"✅ Saved feature columns to {columns_path}")

    # ------------------------------------------------------------------
    # 8) SUMMARY
    # ------------------------------------------------------------------
    total_time = datetime.now() - start_time
    print(f"\n✅ Total execution time: {total_time}")
    
    print(f"\n🎯 V3_{model_suffix} Model Summary:")
    print(f"   - Trained on PL-only timelines (only days when players were at PL clubs)")
    print(f"   - Recent seasons (2018-2026) excluding {', '.join(EXCLUDE_SEASONS)}")
    print(f"   - Same hyperparameters as v1/v2")
    print(f"   - Target ratio: {ratio_display}")


if __name__ == "__main__":
    main()

