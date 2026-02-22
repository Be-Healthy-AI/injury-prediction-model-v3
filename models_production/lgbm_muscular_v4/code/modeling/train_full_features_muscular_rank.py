#!/usr/bin/env python3
"""
First training: LGBM on all features (muscular target only) and regenerate feature_ranking.json.

This script:
1. Loads new timeline train/test data (muscular only, target1)
2. Trains one LGBM on ALL available features
3. Extracts gain-based feature importances
4. Saves feature_ranking.json with ranked_features for use by iterative feature selection

Run this once after regenerating timelines, then run train_iterative_feature_selection_muscular_standalone.py.
"""

import sys
import os
import json
import glob
import argparse
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

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent.parent.parent
MODEL_OUTPUT_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'models'
CACHE_DIR = ROOT_DIR / 'cache'

# Config (match iterative script)
MIN_SEASON = '2018_2019'
EXCLUDE_SEASON = '2025_2026'
TRAIN_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'data' / 'timelines' / 'train'
TEST_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'data' / 'timelines' / 'test'
USE_CACHE = True

RANKING_OUTPUT_FILE = MODEL_OUTPUT_DIR / 'feature_ranking.json'
FULL_MODEL_PATH = MODEL_OUTPUT_DIR / 'lgbm_muscular_full_features_model.joblib'
FULL_MODEL_COLUMNS_PATH = MODEL_OUTPUT_DIR / 'lgbm_muscular_full_features_columns.json'

# Exclude granular club/country features (same as iterative script)
EXCLUDED_RAW_FEATURES = ('current_club', 'current_club_country', 'previous_club', 'previous_club_country')

# ---------------------------------------------------------------------------
# Helpers (from train_iterative_feature_selection_muscular_standalone.py)
# ---------------------------------------------------------------------------

def clean_categorical_value(value):
    if pd.isna(value) or value is None:
        return 'Unknown'
    value_str = str(value).strip()
    problematic = ['tel:', 'tel', 'phone', 'n/a', 'na', 'null', 'none', '', 'nan', 'address:', 'website:']
    if value_str.lower() in problematic:
        return 'Unknown'
    if "'" in value_str:
        value_str = value_str.replace("'", '_')
    replacements = {
        ':': '_', ',': '_', '"': '_', ';': '_', '/': '_', '\\': '_',
        '{': '_', '}': '_', '[': '_', ']': '_', '(': '_', ')': '_', '|': '_',
        '&': '_', '?': '_', '!': '_', '*': '_', '+': '_', '=': '_', '@': '_',
        '#': '_', '$': '_', '%': '_', '^': '_', ' ': '_',
    }
    for old, new in replacements.items():
        value_str = value_str.replace(old, new)
    value_str = ''.join(c for c in value_str if ord(c) >= 32 or c in '\n\r\t')
    while '__' in value_str:
        value_str = value_str.replace('__', '_')
    value_str = value_str.strip('_')
    return value_str if value_str else 'Unknown'

def sanitize_feature_name(name):
    name_str = str(name)
    if "'" in name_str:
        name_str = name_str.replace("'", '_apostrophe_')
    replacements = {
        '"': '_quote_', '\\': '_backslash_', '/': '_slash_',
        '\b': '_bs_', '\f': '_ff_', '\n': '_nl_', '\r': '_cr_', '\t': '_tab_',
        ' ': '_', ':': '_colon_', ';': '_semicolon_',
        ',': '_comma_', '&': '_amp_', '?': '_qmark_', '!': '_excl_',
        '*': '_star_', '+': '_plus_', '=': '_eq_', '@': '_at_',
        '#': '_hash_', '$': '_dollar_', '%': '_pct_', '^': '_caret_',
    }
    for old, new in replacements.items():
        name_str = name_str.replace(old, new)
    name_str = ''.join(c for c in name_str if ord(c) >= 32 or c in '\n\r\t')
    while '__' in name_str:
        name_str = name_str.replace('__', '_')
    name_str = name_str.strip('_')
    return name_str if name_str else 'Unknown'

# ---------------------------------------------------------------------------
# Data loading (muscular-only; target1 only required)
# ---------------------------------------------------------------------------

def load_train_data(min_season=None, exclude_season='2025_2026'):
    """Load and combine train season CSVs. Requires only target1 column."""
    pattern = str(TRAIN_DIR / 'timelines_35day_season_*_v4_muscular_train.csv')
    files = glob.glob(pattern)
    season_files = []
    for filepath in files:
        filename = os.path.basename(filepath)
        if '_10pc_' in filename or '_25pc_' in filename or '_50pc_' in filename:
            continue
        if 'season_' in filename:
            parts = filename.split('season_')
            if len(parts) > 1:
                season_part = parts[1].split('_v4_muscular_train')[0]
                if season_part != exclude_season:
                    if min_season is None or season_part >= min_season:
                        season_files.append((season_part, filepath))
    season_files.sort(key=lambda x: x[0])
    if not season_files:
        raise FileNotFoundError(f"No train season files found in {TRAIN_DIR}")
    print(f"Loading {len(season_files)} train seasons (>= {min_season}, exclude {exclude_season})...")
    dfs = []
    for season_id, filepath in season_files:
        df = pd.read_csv(filepath, encoding='utf-8-sig', low_memory=False)
        if 'target1' not in df.columns:
            print(f"  Skip {season_id}: no target1 column")
            continue
        dfs.append(df)
        n_pos = (df['target1'] == 1).sum()
        print(f"  {season_id}: {len(df):,} rows, target1=1: {n_pos:,}")
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Combined train: {len(combined):,} rows, target1=1: {(combined['target1']==1).sum():,}")
    return combined

def load_test_data():
    """Load test dataset (2025/26)."""
    test_file = TEST_DIR / 'timelines_35day_season_2025_2026_v4_muscular_test.csv'
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")
    df = pd.read_csv(test_file, encoding='utf-8-sig', low_memory=False)
    if 'target1' not in df.columns:
        raise ValueError("Test dataset missing target1 column")
    print(f"Test: {len(df):,} rows, target1=1: {(df['target1']==1).sum():,}")
    return df

# ---------------------------------------------------------------------------
# Prepare features (same logic as iterative script)
# ---------------------------------------------------------------------------

def prepare_data(df, cache_file=None, use_cache=True):
    """Prepare features: drop non-feature cols, encode categoricals, fillna."""
    if use_cache and cache_file and os.path.exists(cache_file):
        try:
            X = pd.read_csv(cache_file, encoding='utf-8-sig', low_memory=False)
            if len(X) != len(df):
                use_cache = False
            else:
                for col in ['target1', 'target2', 'target']:
                    if col in X.columns:
                        X = X.drop(columns=[col])
                print(f"  Loaded preprocessed data from cache ({X.shape[1]} cols)")
                return X
        except Exception:
            use_cache = False
    exclude = [
        'player_id', 'reference_date', 'date', 'player_name',
        'target1', 'target2', 'target', 'has_minimum_activity'
    ] + list(EXCLUDED_RAW_FEATURES)
    feature_columns = [c for c in df.columns if c not in exclude]
    X = df[feature_columns].copy()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    X_encoded = X.copy()
    if categorical_features:
        for feature in tqdm(categorical_features, desc="  Encode categoricals", leave=False):
            X_encoded[feature] = X_encoded[feature].fillna('Unknown').apply(clean_categorical_value)
            dummies = pd.get_dummies(X_encoded[feature], prefix=feature, drop_first=True)
            dummies.columns = [sanitize_feature_name(c) for c in dummies.columns]
            X_encoded = pd.concat([X_encoded, dummies], axis=1)
            X_encoded = X_encoded.drop(columns=[feature])
    for feature in numeric_features:
        X_encoded[feature] = X_encoded[feature].fillna(0)
    X_encoded.columns = [sanitize_feature_name(c) for c in X_encoded.columns]
    if use_cache and cache_file:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        X_encoded.to_csv(cache_file, index=False, encoding='utf-8-sig')
    return X_encoded

def align_features(X_train, X_test):
    common = sorted(set(X_train.columns) & set(X_test.columns))
    print(f"  Aligned: {len(common)} common features (train {X_train.shape[1]} -> {len(common)}, test {X_test.shape[1]} -> {len(common)})")
    return X_train[common], X_test[common]

# ---------------------------------------------------------------------------
# Train and rank
# ---------------------------------------------------------------------------

def extract_gain_importance(model):
    if hasattr(model, 'booster_'):
        return model.booster_.feature_importance(importance_type='gain')
    return model.feature_importances_

def main(use_cache=None):
    if use_cache is None:
        use_cache = USE_CACHE
    print("=" * 80)
    print("FIRST TRAINING: LGBM ON ALL FEATURES (MUSCULAR ONLY) -> feature_ranking.json")
    print("=" * 80)
    if not use_cache:
        print("   Cache disabled (--no-cache): preprocessing from CSV.")
    start_time = datetime.now()
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n1. Loading data...")
    df_train = load_train_data(min_season=MIN_SEASON, exclude_season=EXCLUDE_SEASON)
    df_test = load_test_data()

    # Prepare features (all)
    print("\n2. Preparing features...")
    cache_train = str(CACHE_DIR / 'preprocessed_muscular_full_train.csv')
    cache_test = str(CACHE_DIR / 'preprocessed_muscular_full_test.csv')
    X_train = prepare_data(df_train, cache_file=cache_train, use_cache=use_cache)
    y_train = df_train['target1'].values
    X_test = prepare_data(df_test, cache_file=cache_test, use_cache=use_cache)
    y_test = df_test['target1'].values
    X_train, X_test = align_features(X_train, X_test)
    feature_names = list(X_train.columns)
    print(f"  Total features for training: {len(feature_names)}")

    # Train LGBM (same hyperparams as iterative script)
    print("\n3. Training LGBM on all features...")
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
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print("\n4. Test set metrics:")
    print(f"   Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"   Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"   Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"   F1:        {f1_score(y_test, y_pred, zero_division=0):.4f}")
    roc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.0
    print(f"   ROC AUC:   {roc:.4f}")
    print(f"   Gini:      {(2*roc - 1):.4f}")

    # Extract importances and rank
    print("\n5. Extracting feature importances (gain)...")
    importances = extract_gain_importance(model)
    if len(importances) != len(feature_names):
        raise ValueError(f"Importances length {len(importances)} != features {len(feature_names)}")
    df_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
    df_imp = df_imp.sort_values('importance', ascending=False).reset_index(drop=True)
    ranked_features = df_imp['feature'].tolist()
    print(f"   Top 10: {ranked_features[:10]}")

    # Save feature_ranking.json (same structure as before for iterative script)
    print("\n6. Saving feature_ranking.json...")
    ranking_data = {
        'ranked_features': ranked_features,
        'feature_importances': {
            'muscular': {f: float(df_imp.loc[df_imp['feature'] == f, 'importance'].iloc[0]) for f in ranked_features},
            'average': {f: float(df_imp.loc[df_imp['feature'] == f, 'importance'].iloc[0]) for f in ranked_features},
            'normalized': {}
        },
        'ranking_metadata': {
            'total_features': len(ranked_features),
            'ranking_method': 'LightGBM gain importance (single muscular model, all features)',
            'source': 'train_full_features_muscular_rank.py',
            'ranking_date': start_time.isoformat(),
            'train_seasons_min': MIN_SEASON,
            'test_season': EXCLUDE_SEASON
        }
    }
    max_imp = df_imp['importance'].max()
    if max_imp > 0:
        ranking_data['feature_importances']['normalized'] = {
            row['feature']: float(row['importance'] / max_imp) for _, row in df_imp.iterrows()
        }
    with open(RANKING_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(ranking_data, f, indent=2)
    print(f"   Saved: {RANKING_OUTPUT_FILE}")

    # Optionally save model and column list
    joblib.dump(model, FULL_MODEL_PATH)
    with open(FULL_MODEL_COLUMNS_PATH, 'w', encoding='utf-8') as f:
        json.dump(feature_names, f, indent=2)
    print(f"   Saved model: {FULL_MODEL_PATH}")
    print(f"   Saved columns: {FULL_MODEL_COLUMNS_PATH}")

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\nDone in {elapsed:.1f}s. Next: run train_iterative_feature_selection_muscular_standalone.py")
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train on all features (muscular only) and save feature_ranking.json")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache; preprocess from timeline CSVs")
    args = parser.parse_args()
    sys.exit(main(use_cache=not args.no_cache))
