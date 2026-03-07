#!/usr/bin/env python3
"""
Generate daily injury predictions for production using V4 (4 models).

Reads timelines from production/deployments/{country}/challenger/{club}/timelines/ and generates
predictions using four V4 models:
- Muscular LGBM
- Muscular GB
- Skeletal LGBM
- MSU LGBM (Muscular/Skeletal/Unknown)

For the Muscular LGBM model, encoding and options follow the deployment guidelines so that
production predictions match the training pipeline. See:
  models_production/lgbm_muscular_v4/model_muscular_lgbm/DEPLOYMENT_GUIDELINES.md
Reference predictions from the test set (for validation) are in:
  models_production/lgbm_muscular_v4/model_muscular_lgbm/test_predictions_from_training_pipeline.csv

Output: one CSV with injury_probability_muscular_lgbm, injury_probability_muscular_gb,
injury_probability_skeletal, injury_probability_msu_lgbm, plus backward-compat injury_probability (= muscular LGBM).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import shap

# Calculate paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent
ROOT_DIR = PRODUCTION_ROOT.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# Import insight utilities (shared with V3)
from production.scripts.insight_utils import (
    load_pipeline,
    predict_insights,
    classify_risk_4level,
)

# Skeletal model: use encoding_schema.json from model_skeletal for fixed one-hot universe
# (same pattern as muscular: standalone encoding to match training, no call to training script)

# V4: four models (model_muscular_lgbm encoding follows DEPLOYMENT_GUIDELINES.md)
V4_BASE = ROOT_DIR / "models_production" / "lgbm_muscular_v4"
MODELS_V4 = {
    "muscular_lgbm": V4_BASE / "model_muscular_lgbm",
    "muscular_gb": V4_BASE / "model_muscular_gb",
    "skeletal": V4_BASE / "model_skeletal",
    "msu_lgbm": V4_BASE / "model_msu_lgbm",   # Muscular/Skeletal/Unknown LGBM
}
PRIMARY_MODEL_KEY = "muscular_lgbm"  # Used for SHAP, body part/severity, top_feature_*, risk_level


def load_models() -> Dict[str, Tuple[object, List[str]]]:
    """Load all four V4 models and their feature columns. Returns dict model_key -> (model, columns).
    For muscular_lgbm, model and columns come from model_muscular_lgbm/ per DEPLOYMENT_GUIDELINES.md.
    """
    result = {}
    for key, model_dir in MODELS_V4.items():
        model_path = model_dir / "model.joblib"
        cols_path = model_dir / "columns.json"
        if not model_path.exists():
            raise FileNotFoundError(f"V4 model {key}: model file not found: {model_path}")
        if not cols_path.exists():
            raise FileNotFoundError(f"V4 model {key}: columns file not found: {cols_path}")
        model = joblib.load(model_path)
        # Use the model's own feature order so prediction column order matches training.
        # Priority:
        # 1) LightGBM: feature_name_
        # 2) Sklearn GB: feature_names_in_
        # 3) Fallback to columns.json (legacy)
        columns = getattr(model, "feature_name_", None)
        if columns is None or len(columns) == 0:
            # For sklearn GradientBoosting (muscular_gb), feature_names_in_ stores the training order
            columns = getattr(model, "feature_names_in_", None)
        if columns is None or len(columns) == 0:
            with open(cols_path, 'r', encoding='utf-8', errors='replace') as f:
                columns_data = json.load(f)
            # Support both JSON array of names and object with "features" key (same as evaluate_production script)
            if isinstance(columns_data, list):
                columns = columns_data
            else:
                columns = columns_data.get("features", columns_data)
        result[key] = (model, list(columns))
        print(f"[LOAD] V4 {key}: {len(columns)} features")
    return result


def clean_categorical_value(value):
    """Clean categorical values to remove special characters that cause issues in feature names.
    Must match training pipeline (train_iterative_feature_selection_muscular_standalone.py).
    """
    if pd.isna(value) or value is None:
        return 'Unknown'

    value_str = str(value).strip()

    problematic_values = ['tel:', 'tel', 'phone', 'n/a', 'na', 'null', 'none', '', 'nan', 'address:', 'website:']
    if value_str.lower() in problematic_values:
        return 'Unknown'

    replacements = {
        ':': '_', "'": '_', ',': '_', '"': '_', ';': '_', '/': '_', '\\': '_',
        '{': '_', '}': '_', '[': '_', ']': '_', '(': '_', ')': '_', '|': '_',
        '&': '_', '?': '_', '!': '_', '*': '_', '+': '_', '=': '_', '@': '_',
        '#': '_', '$': '_', '%': '_', '^': '_', ' ': '_',
    }
    for old_char, new_char in replacements.items():
        value_str = value_str.replace(old_char, new_char)

    value_str = ''.join(char for char in value_str if ord(char) >= 32 or char in '\n\r\t')
    while '__' in value_str:
        value_str = value_str.replace('__', '_')
    value_str = value_str.strip('_')
    if not value_str:
        return 'Unknown'
    return value_str


# Map nationality variants (e.g. from production timelines) to exact strings used in training
# so one-hot column names match model columns (e.g. nationality1_Türkiye).
NATIONALITY_NORMALIZE = {
    'turkey': 'Türkiye',  # ASCII variant -> Unicode as in model columns.json
}


def normalize_nationality(value):
    """Normalize nationality string to match training pipeline column names."""
    if pd.isna(value) or value is None:
        return value
    s = str(value).strip()
    if not s:
        return value
    key = s.lower()
    return NATIONALITY_NORMALIZE.get(key, value)


def sanitize_feature_name(name):
    """Sanitize feature names to be JSON-safe for LightGBM.
    Must match training pipeline (train_iterative_feature_selection_muscular_standalone.py).
    """
    name_str = str(name)
    replacements = {
        '"': '_quote_', '\\': '_backslash_', '/': '_slash_',
        '\b': '_bs_', '\f': '_ff_', '\n': '_nl_', '\r': '_cr_', '\t': '_tab_',
        ' ': '_', "'": '_apostrophe_', ':': '_colon_', ';': '_semicolon_',
        ',': '_comma_', '&': '_amp_', '?': '_qmark_', '!': '_excl_',
        '*': '_star_', '+': '_plus_', '=': '_eq_', '@': '_at_',
        '#': '_hash_', '$': '_dollar_', '%': '_pct_', '^': '_caret_',
    }
    for old_char, new_char in replacements.items():
        name_str = name_str.replace(old_char, new_char)
    name_str = ''.join(char for char in name_str if ord(char) >= 32 or char in '\n\r\t')
    while '__' in name_str:
        name_str = name_str.replace('__', '_')
    name_str = name_str.strip('_')
    if not name_str:
        return 'Unknown'
    return name_str


def encode_categorical_features(timelines_df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode categorical features to match the training pipeline exactly.
    Encoding follows model_muscular_lgbm/DEPLOYMENT_GUIDELINES.md and
    train_iterative_feature_selection_muscular_standalone.prepare_data():
    - EXCLUDED_RAW_FEATURES dropped; drop_first=True; clean_categorical_value; sanitize_feature_name.
    """
    df_encoded = timelines_df.copy()

    # Same meta and raw exclusions as training prepare_data (production timelines have no target cols)
    META_COLUMNS = ['player_id', 'reference_date', 'player_name', 'date']
    EXCLUDED_RAW_FEATURES = ('current_club', 'current_club_country', 'previous_club', 'previous_club_country')
    feature_columns = [
        c for c in df_encoded.columns
        if c not in META_COLUMNS and c not in EXCLUDED_RAW_FEATURES
    ]
    if not feature_columns:
        return df_encoded

    X = df_encoded[feature_columns].copy()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    for feature in categorical_features:
        if feature not in X.columns:
            continue
        X[feature] = X[feature].fillna('Unknown')
        if feature in ('nationality1', 'nationality2'):
            X[feature] = X[feature].apply(normalize_nationality)
        X[feature] = X[feature].apply(clean_categorical_value)
        dummies = pd.get_dummies(X[feature], prefix=feature, drop_first=True)
        dummies.columns = [sanitize_feature_name(col) for col in dummies.columns]
        X = pd.concat([X.drop(columns=[feature]), dummies], axis=1)

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0)

    X.columns = [sanitize_feature_name(col) for col in X.columns]

    meta_present = [c for c in META_COLUMNS if c in df_encoded.columns]
    if meta_present:
        df_encoded = pd.concat([df_encoded[meta_present].reset_index(drop=True), X], axis=1)
    else:
        df_encoded = X

    if categorical_features:
        print(f"[ENCODE] One-hot encoded {len(categorical_features)} categorical features (drop_first=True): {categorical_features}")
    return df_encoded


# Skeletal: same meta exclusions as skeletal prepare_data (no EXCLUDED_RAW_FEATURES)
SKELETAL_META_COLUMNS = ['player_id', 'reference_date', 'player_name', 'date']


def encode_skeletal_features(
    timelines_df: pd.DataFrame,
    model_columns: List[str],
    encoding_schema: Dict[str, List[str]],
) -> pd.DataFrame:
    """
    Encode timelines for skeletal model using a fixed one-hot schema (same as muscular pattern).
    Uses encoding_schema.json from model_skeletal so production matches training pipeline.
    - Only meta columns dropped (no current_club / previous_club exclusion).
    - Categoricals: 1 for matching dummy column, 0 for others in schema.
    - Numerics: value from timeline, 0 if missing.
    """
    # Map each dummy column name -> its categorical base
    dummy_to_base: Dict[str, str] = {}
    for base, dummies in encoding_schema.items():
        for d in dummies:
            dummy_to_base[d] = base

    feature_columns = [
        c for c in timelines_df.columns
        if c not in SKELETAL_META_COLUMNS
        and c not in ('target1', 'target2', 'target', 'has_minimum_activity')
    ]
    X = timelines_df[feature_columns].copy() if feature_columns else pd.DataFrame(index=timelines_df.index)

    aligned = pd.DataFrame(index=timelines_df.index)
    for col in model_columns:
        if col in dummy_to_base:
            base = dummy_to_base[col]
            if base not in X.columns:
                aligned[col] = 0.0
                continue
            # Same as skeletal prepare_data: fillna Unknown, clean_categorical_value, then dummy = base + "_" + sanitize(value)
            raw = X[base].fillna('Unknown').astype(str)
            cleaned = raw.apply(clean_categorical_value)
            dummy_names = cleaned.apply(lambda v: f"{base}_{sanitize_feature_name(v)}")
            aligned[col] = (dummy_names == col).astype(float)
        else:
            # Numeric: take from timeline, 0 if missing
            if col in X.columns:
                aligned[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0)
            else:
                aligned[col] = 0.0

    return aligned


# MSU: same exclusions as muscular prepare_data (EXCLUDED_RAW_FEATURES)
META_COLUMNS_MSU = ['player_id', 'reference_date', 'player_name', 'date']
EXCLUDED_RAW_FEATURES_MSU = ('current_club', 'current_club_country', 'previous_club', 'previous_club_country')


def encode_msu_features(
    timelines_df: pd.DataFrame,
    model_columns: List[str],
    encoding_schema: Dict[str, List[str]],
) -> pd.DataFrame:
    """
    Encode timelines for MSU model using a fixed one-hot schema (same pattern as muscular/skeletal).
    Uses encoding_schema.json from model_msu_lgbm so production matches training pipeline.
    - Same exclusions as muscular: META + EXCLUDED_RAW_FEATURES; nationality normalized.
    - Categoricals: 1 for matching dummy column, 0 for others in schema.
    - Numerics: value from timeline, 0 if missing.
    """
    dummy_to_base: Dict[str, str] = {}
    for base, dummies in encoding_schema.items():
        for d in dummies:
            dummy_to_base[d] = base

    feature_columns = [
        c for c in timelines_df.columns
        if c not in META_COLUMNS_MSU and c not in EXCLUDED_RAW_FEATURES_MSU
        and c not in ('target1', 'target2', 'target', 'target_msu', 'has_minimum_activity')
    ]
    X = timelines_df[feature_columns].copy() if feature_columns else pd.DataFrame(index=timelines_df.index)

    aligned = pd.DataFrame(index=timelines_df.index)
    for col in model_columns:
        if col in dummy_to_base:
            base = dummy_to_base[col]
            if base not in X.columns:
                aligned[col] = 0.0
                continue
            raw = X[base].fillna('Unknown').astype(str)
            if base in ('nationality1', 'nationality2'):
                raw = raw.apply(normalize_nationality)
            cleaned = raw.apply(clean_categorical_value)
            dummy_names = cleaned.apply(lambda v: f"{base}_{sanitize_feature_name(v)}")
            aligned[col] = (dummy_names == col).astype(float)
        else:
            if col in X.columns:
                aligned[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0)
            else:
                aligned[col] = 0.0

    return aligned


def align_features_to_model(timelines_df: pd.DataFrame, model_columns: list) -> pd.DataFrame:
    """
    Align timeline features to match model column order and handle missing features.

    V4 timelines should already have the correct features, but we need to:
    1. One-hot encode categorical features
    2. Ensure column order matches model_columns
    3. Fill missing features with defaults (0 for numeric features)
    """
    print(f"[ALIGN] Aligning {len(timelines_df)} timelines to {len(model_columns)} model features...")
    
    # First, one-hot encode categorical features
    print(f"[ENCODE] Encoding categorical features...")
    timelines_encoded = encode_categorical_features(timelines_df)
    
    # Create aligned DataFrame using concat for better performance
    # First, collect all columns that exist
    existing_cols = []
    missing_features = []
    for col in model_columns:
        if col in timelines_encoded.columns:
            existing_cols.append(col)
        else:
            missing_features.append(col)
    
    # Create DataFrame with existing columns
    aligned_df = timelines_encoded[existing_cols].copy() if existing_cols else pd.DataFrame(index=timelines_encoded.index)
    
    # Add missing columns with default values (0.0 for one-hot encoded features)
    if missing_features:
        missing_df = pd.DataFrame(0.0, index=timelines_encoded.index, columns=missing_features)
        aligned_df = pd.concat([aligned_df, missing_df], axis=1)
    
    # Ensure columns are in the exact order expected by model
    aligned_df = aligned_df[model_columns]
    
    if missing_features:
        print(f"[WARN] {len(missing_features)} features missing from timelines, filled with defaults")
        if len(missing_features) <= 10:
            print(f"       Missing: {missing_features}")
        else:
            print(f"       Missing: {missing_features[:10]} ... and {len(missing_features) - 10} more")
    
    # Ensure correct data types (numeric)
    for col in aligned_df.columns:
        if aligned_df[col].dtype == 'object':
            # Try to convert to numeric
            aligned_df[col] = pd.to_numeric(aligned_df[col], errors='coerce').fillna(0.0)
        aligned_df[col] = aligned_df[col].fillna(0.0).astype(float)
    
    print(f"[ALIGN] Aligned features: {len(aligned_df.columns)} columns, {len(aligned_df)} rows")
    return aligned_df


def align_skeletal_features_to_model(
    timelines_df: pd.DataFrame,
    model_columns: list,
    model_skeletal_dir: Path,
) -> pd.DataFrame:
    """
    Align features for skeletal LGBM using encoding_schema.json (same pattern as muscular).
    Encodes with fixed one-hot universe so production matches test_predictions_from_training_pipeline.csv.
    """
    schema_path = model_skeletal_dir / "encoding_schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(
            f"Skeletal model requires encoding_schema.json in {model_skeletal_dir}. "
            "Re-export the model with --export-best from the skeletal training script, or run "
            "models_production/lgbm_muscular_v4/code/modeling/generate_skeletal_encoding_schema.py to generate it."
        )
    with open(schema_path, "r", encoding="utf-8") as f:
        encoding_schema = json.load(f)

    print(f"[ALIGN] Aligning {len(timelines_df)} timelines to skeletal model ({len(model_columns)} features)...")
    print(f"[ENCODE] Skeletal encoding using encoding_schema.json ({len(encoding_schema)} categoricals)...")
    X_encoded = encode_skeletal_features(timelines_df, model_columns, encoding_schema)

    # Ensure exact column order and fill any missing with 0
    existing = [c for c in model_columns if c in X_encoded.columns]
    missing = [c for c in model_columns if c not in X_encoded.columns]
    if existing:
        aligned_df = X_encoded[existing].copy()
    else:
        aligned_df = pd.DataFrame(index=X_encoded.index)
    if missing:
        aligned_df = pd.concat([
            aligned_df,
            pd.DataFrame(0.0, index=X_encoded.index, columns=missing),
        ], axis=1)
    aligned_df = aligned_df[model_columns]

    for col in aligned_df.columns:
        aligned_df[col] = pd.to_numeric(aligned_df[col], errors="coerce").fillna(0.0).astype(float)

    print(f"[ALIGN] Skeletal aligned features: {len(aligned_df.columns)} columns, {len(aligned_df)} rows")
    return aligned_df


def align_msu_features_to_model(
    timelines_df: pd.DataFrame,
    model_columns: list,
    model_msu_dir: Path,
) -> pd.DataFrame:
    """
    Align features for MSU LGBM using encoding_schema.json (same pattern as muscular/skeletal).
    """
    schema_path = model_msu_dir / "encoding_schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(
            f"MSU model requires encoding_schema.json in {model_msu_dir}. "
            "Re-export the model with --only-iteration or --export-best (MSU), or run "
            "models_production/lgbm_muscular_v4/code/modeling/generate_msu_encoding_schema.py to generate it."
        )
    with open(schema_path, "r", encoding="utf-8") as f:
        encoding_schema = json.load(f)

    print(f"[ALIGN] Aligning {len(timelines_df)} timelines to MSU model ({len(model_columns)} features)...")
    print(f"[ENCODE] MSU encoding using encoding_schema.json ({len(encoding_schema)} categoricals)...")
    X_encoded = encode_msu_features(timelines_df, model_columns, encoding_schema)

    existing = [c for c in model_columns if c in X_encoded.columns]
    missing = [c for c in model_columns if c not in X_encoded.columns]
    if existing:
        aligned_df = X_encoded[existing].copy()
    else:
        aligned_df = pd.DataFrame(index=X_encoded.index)
    if missing:
        aligned_df = pd.concat([
            aligned_df,
            pd.DataFrame(0.0, index=X_encoded.index, columns=missing),
        ], axis=1)
    aligned_df = aligned_df[model_columns]

    for col in aligned_df.columns:
        aligned_df[col] = pd.to_numeric(aligned_df[col], errors="coerce").fillna(0.0).astype(float)

    print(f"[ALIGN] MSU aligned features: {len(aligned_df.columns)} columns, {len(aligned_df)} rows")
    return aligned_df


def validate_muscular_lgbm_against_training(
    predictions_df: pd.DataFrame,
    reference_csv_path: Path,
    tolerance: float = 1e-5,
) -> Tuple[bool, float, int]:
    """
    Compare production muscular LGBM predictions to test_predictions_from_training_pipeline.csv
    for overlapping (player_id, reference_date). Returns (all_within_tolerance, max_abs_diff, n_over_tolerance).
    """
    if not reference_csv_path.exists():
        print(f"[VALIDATE] Reference file not found: {reference_csv_path}")
        return True, 0.0, 0
    if 'injury_probability_muscular_lgbm' not in predictions_df.columns:
        print(f"[VALIDATE] No injury_probability_muscular_lgbm column in predictions")
        return True, 0.0, 0
    ref = pd.read_csv(reference_csv_path, encoding='utf-8-sig')
    ref['reference_date'] = pd.to_datetime(ref['reference_date'], errors='coerce')
    pred = predictions_df[['player_id', 'reference_date', 'injury_probability_muscular_lgbm']].copy()
    pred['reference_date'] = pd.to_datetime(pred['reference_date'], errors='coerce')
    merged = pred.merge(
        ref[['player_id', 'reference_date', 'predicted_probability']],
        on=['player_id', 'reference_date'],
        how='inner',
        suffixes=('_prod', '_train'),
    )
    if len(merged) == 0:
        print(f"[VALIDATE] No overlapping (player_id, reference_date) between production and reference CSV")
        return True, 0.0, 0
    merged['diff'] = (merged['injury_probability_muscular_lgbm'] - merged['predicted_probability']).abs()
    max_diff = float(merged['diff'].max())
    over = (merged['diff'] > tolerance).sum()
    mean_diff = float(merged['diff'].mean())
    print(f"[VALIDATE] Muscular LGBM vs test_predictions_from_training_pipeline.csv:")
    print(f"   Rows compared: {len(merged):,}")
    print(f"   Max abs diff:  {max_diff:.6e}")
    print(f"   Mean abs diff: {mean_diff:.6e}")
    print(f"   Over tolerance ({tolerance:.0e}): {over:,}")
    if over > 0:
        worst = merged.nlargest(5, 'diff')[['player_id', 'reference_date', 'injury_probability_muscular_lgbm', 'predicted_probability', 'diff']]
        print(f"   Worst 5: player_id, reference_date, prod, train, diff")
        for _, r in worst.iterrows():
            print(f"      {int(r['player_id'])}, {r['reference_date'].strftime('%Y-%m-%d')}, {r['injury_probability_muscular_lgbm']:.6f}, {r['predicted_probability']:.6f}, {r['diff']:.6e}")
    return over == 0, max_diff, int(over)


def validate_muscular_gb_against_training(
    predictions_df: pd.DataFrame,
    reference_csv_path: Path,
    tolerance: float = 1e-5,
) -> Tuple[bool, float, int]:
    """
    Compare production muscular GB predictions to test_predictions_from_training_pipeline.csv
    for overlapping (player_id, reference_date). Returns (all_within_tolerance, max_abs_diff, n_over_tolerance).
    """
    if not reference_csv_path.exists():
        print(f"[VALIDATE] Reference file not found: {reference_csv_path}")
        return True, 0.0, 0
    if 'injury_probability_muscular_gb' not in predictions_df.columns:
        print(f"[VALIDATE] No injury_probability_muscular_gb column in predictions")
        return True, 0.0, 0
    ref = pd.read_csv(reference_csv_path, encoding='utf-8-sig')
    ref['reference_date'] = pd.to_datetime(ref['reference_date'], errors='coerce')
    pred = predictions_df[['player_id', 'reference_date', 'injury_probability_muscular_gb']].copy()
    pred['reference_date'] = pd.to_datetime(pred['reference_date'], errors='coerce')
    merged = pred.merge(
        ref[['player_id', 'reference_date', 'predicted_probability']],
        on=['player_id', 'reference_date'],
        how='inner',
        suffixes=('_prod', '_train'),
    )
    if len(merged) == 0:
        print("[VALIDATE] No overlapping (player_id, reference_date) between production and reference CSV (muscular GB)")
        return True, 0.0, 0
    merged['diff'] = (merged['injury_probability_muscular_gb'] - merged['predicted_probability']).abs()
    max_diff = float(merged['diff'].max())
    over = (merged['diff'] > tolerance).sum()
    mean_diff = float(merged['diff'].mean())
    print("[VALIDATE] Muscular GB vs test_predictions_from_training_pipeline.csv:")
    print(f"   Rows compared: {len(merged):,}")
    print(f"   Max abs diff:  {max_diff:.6e}")
    print(f"   Mean abs diff: {mean_diff:.6e}")
    print(f"   Over tolerance ({tolerance:.0e}): {over:,}")
    if over > 0:
        worst = merged.nlargest(5, 'diff')[['player_id', 'reference_date', 'injury_probability_muscular_gb', 'predicted_probability', 'diff']]
        print("   Worst 5: player_id, reference_date, prod, train, diff")
        for _, r in worst.iterrows():
            print(
                f"      {int(r['player_id'])}, "
                f"{r['reference_date'].strftime('%Y-%m-%d')}, "
                f"{r['injury_probability_muscular_gb']:.6f}, "
                f"{r['predicted_probability']:.6f}, "
                f"{r['diff']:.6e}"
            )
    return over == 0, max_diff, int(over)


def get_latest_raw_data_folder(country: str = "england") -> Path:
    """Get the latest raw data folder."""
    base_dir = PRODUCTION_ROOT / "raw_data" / country.lower().replace(" ", "_")
    if not base_dir.exists():
        return None
    
    date_folders = [d for d in base_dir.iterdir() if d.is_dir() and d.name.isdigit() and len(d.name) == 8]
    if not date_folders:
        return None
    
    return max(date_folders, key=lambda x: x.name)


def load_player_profile(player_id: int, raw_data_dir: Path) -> pd.Series:
    """Load player profile from latest raw data folder."""
    if raw_data_dir is None or not raw_data_dir.exists():
        return pd.Series()
    
    profile_file = raw_data_dir / "players_profile.csv"
    if not profile_file.exists():
        return pd.Series()
    
    try:
        df = pd.read_csv(profile_file, sep=';', encoding='utf-8-sig')
        # Try 'player_id' first, then fallback to 'id'
        if 'player_id' in df.columns:
            player_row = df[df['player_id'] == player_id]
        elif 'id' in df.columns:
            player_row = df[df['id'] == player_id]
        else:
            return pd.Series()
        
        if player_row.empty:
            return pd.Series()
        return player_row.iloc[0]
    except Exception as e:
        print(f"[WARN] Could not load profile for player {player_id}: {e}")
        return pd.Series()


def generate_predictions(
    timelines_df: pd.DataFrame,
    models: Dict[str, Tuple[object, List[str]]],
    primary_key: str = PRIMARY_MODEL_KEY,
    bodypart_pipeline=None,
    severity_pipeline=None,
    daily_features_dir: Optional[Path] = None,
    timelines_file_mtime: Optional[float] = None,
) -> pd.DataFrame:
    """
    Generate predictions for V4 timelines using all four models.
    Returns one DataFrame with injury_probability_muscular_lgbm, injury_probability_muscular_gb,
    injury_probability_skeletal, injury_probability_msu_lgbm, plus injury_probability (= primary) and risk_level.
    SHAP and body part/severity are computed for the primary model only.
    """
    print(f"\n[PREPROCESS] Preparing data for {len(timelines_df):,} timelines...")
    timelines_df = timelines_df.reset_index(drop=True)

    # Run each model and collect probability columns
    prob_columns = {}
    aligned_indices = None

    for model_key, (model, model_columns) in models.items():
        print(f"\n[ALIGN] Aligning features to V4 model: {model_key} ({len(model_columns)} features)...")
        if model_key == "skeletal":
            X_aligned = align_skeletal_features_to_model(timelines_df, model_columns, MODELS_V4["skeletal"])
        elif model_key == "msu_lgbm":
            X_aligned = align_msu_features_to_model(timelines_df, model_columns, MODELS_V4["msu_lgbm"])
        else:
            X_aligned = align_features_to_model(timelines_df, model_columns)
        if aligned_indices is None:
            aligned_indices = X_aligned.index
        # With model_columns coming from the model itself (feature_name_ / feature_names_in_),
        # X_aligned has the exact same column order as during training, so we can rely on
        # sklearn/LightGBM's feature name checks without bypassing them.
        proba = model.predict_proba(X_aligned)[:, 1]
        col_name = f"injury_probability_{model_key}"
        prob_columns[col_name] = proba
        print(f"[PREDICT] {model_key}: {len(proba):,} predictions")

    # SHAP and top features for primary model only
    primary_model, primary_columns = models[primary_key]
    print(f"\n[ALIGN] Re-aligning for SHAP (primary: {primary_key})...")
    X_primary = align_features_to_model(timelines_df, primary_columns)
    print(f"\n[SHAP] Computing feature importance ({primary_key})...")
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, module='shap')
        shap_explainer = shap.TreeExplainer(primary_model)
        shap_values = shap_explainer.shap_values(X_primary)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    top_features_list = []
    feature_names = X_primary.columns.tolist()
    for row_idx, row_shap in enumerate(shap_values):
        indices = np.argsort(np.abs(row_shap))[::-1][:10]
        top_features_list.append([
            {"name": feature_names[idx], "value": float(X_primary.iloc[row_idx, idx]),
             "shap": float(row_shap[idx]), "abs_shap": float(abs(row_shap[idx]))}
            for idx in indices
        ])

    # Body part and severity (primary model context only)
    body_part_predictions = []
    severity_predictions = []
    body_part_probs = []
    severity_probs = []
    if bodypart_pipeline is not None and severity_pipeline is not None and daily_features_dir is not None:
        print(f"\n[INSIGHTS] Predicting body part and severity...")
        daily_features_cache = {}
        for idx in aligned_indices:
            row = timelines_df.loc[idx]
            player_id = row['player_id']
            ref_date = pd.to_datetime(row['reference_date'])
            if player_id not in daily_features_cache:
                daily_features_file = daily_features_dir / f"player_{player_id}_daily_features.csv"
                if daily_features_file.exists():
                    try:
                        df_features = pd.read_csv(daily_features_file, parse_dates=['date'], low_memory=False)
                        daily_features_cache[player_id] = df_features
                    except Exception as e:
                        print(f"[WARN] Could not load daily features for player {player_id}: {e}")
                        daily_features_cache[player_id] = None
                else:
                    daily_features_cache[player_id] = None
            df_features = daily_features_cache.get(player_id)
            if df_features is not None:
                feature_row = df_features[df_features['date'] == ref_date]
                if not feature_row.empty:
                    feature_series = feature_row.iloc[0].drop(['date'])
                    insights = predict_insights(feature_series, bodypart_pipeline, severity_pipeline)
                    if insights.get("bodypart_rank"):
                        top_body = insights["bodypart_rank"][0]
                        body_part_predictions.append(top_body[0])
                        body_part_probs.append(top_body[1])
                    else:
                        body_part_predictions.append("unknown")
                        body_part_probs.append(0.0)
                    if insights.get("severity_label"):
                        severity_predictions.append(insights["severity_label"])
                        severity_probs.append(insights.get("severity_probs", {}).get(insights["severity_label"], 0.0))
                    else:
                        severity_predictions.append("unknown")
                        severity_probs.append(0.0)
                else:
                    body_part_predictions.append("unknown")
                    body_part_probs.append(0.0)
                    severity_predictions.append("unknown")
                    severity_probs.append(0.0)
            else:
                body_part_predictions.append("unknown")
                body_part_probs.append(0.0)
                severity_predictions.append("unknown")
                severity_probs.append(0.0)
    else:
        n = len(aligned_indices)
        body_part_predictions = ["unknown"] * n
        body_part_probs = [0.0] * n
        severity_predictions = ["unknown"] * n
        severity_probs = [0.0] * n

    # Build predictions DataFrame: three prob columns + backward-compat primary
    predictions = pd.DataFrame({
        'player_id': timelines_df.loc[aligned_indices, 'player_id'].values,
        'reference_date': timelines_df.loc[aligned_indices, 'reference_date'].values,
        **prob_columns,
        'injury_probability': prob_columns[f"injury_probability_{primary_key}"],
    })
    predictions['risk_level'] = predictions['injury_probability'].apply(
        lambda p: classify_risk_4level(float(p))['label'] if pd.notna(p) else 'Unknown'
    )
    predictions['predicted_body_part'] = body_part_predictions
    predictions['body_part_probability'] = body_part_probs
    predictions['predicted_severity'] = severity_predictions
    predictions['severity_probability'] = severity_probs
    for i in range(10):
        predictions[f'top_feature_{i+1}_name'] = [feat[i]['name'] if i < len(feat) else '' for feat in top_features_list]
        predictions[f'top_feature_{i+1}_value'] = [feat[i]['value'] if i < len(feat) else 0.0 for feat in top_features_list]
        predictions[f'top_feature_{i+1}_shap'] = [feat[i]['shap'] if i < len(feat) else 0.0 for feat in top_features_list]
    if 'player_name' in timelines_df.columns:
        predictions['player_name'] = timelines_df.loc[aligned_indices, 'player_name'].values
    else:
        predictions['player_name'] = predictions['player_id'].astype(str)
    return predictions


def main():
    parser = argparse.ArgumentParser(
        description='Generate daily injury predictions using lgbm_muscular_v4 model'
    )
    parser.add_argument(
        '--country',
        type=str,
        default='England',
        help='Country name (default: England)'
    )
    parser.add_argument(
        '--club',
        type=str,
        required=True,
        help='Club name (required)'
    )
    parser.add_argument(
        '--timelines-file',
        type=str,
        default=None,
        help='Path to timelines CSV file (default: auto-detect from club deployment)'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Output file path (default: predictions_lgbm_v4_YYYYMMDD.csv in predictions folder)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force regeneration even if predictions already exist'
    )
    parser.add_argument(
        '--data-date',
        type=str,
        default=None,
        help='Data date for filename (YYYYMMDD). If provided, overrides max date from timelines'
    )
    parser.add_argument(
        '--min-date',
        type=str,
        default=None,
        help='Minimum reference date (YYYY-MM-DD). Only generate predictions from this date onwards'
    )
    parser.add_argument(
        '--debug-row',
        type=str,
        default=None,
        metavar='PLAYER_ID,REF_DATE',
        help='Debug: dump aligned muscular_lgbm feature row for (player_id, reference_date) to CSV. Example: 144028,2025-12-05'
    )
    parser.add_argument(
        '--validate-muscular-lgbm',
        action='store_true',
        help='After saving, compare muscular LGBM predictions to test_predictions_from_training_pipeline.csv for overlapping (player_id, reference_date)'
    )
    parser.add_argument(
        '--validate-tolerance',
        type=float,
        default=1e-5,
        help='Max allowed absolute difference in probability for validation (default: 1e-5). Used with --validate-muscular-lgbm'
    )
    parser.add_argument(
        '--validate-muscular-gb',
        action='store_true',
        help='After saving, compare muscular GB predictions to model_muscular_gb/test_predictions_from_training_pipeline.csv for overlapping (player_id, reference_date)'
    )
    parser.add_argument(
        '--validate-gb-tolerance',
        type=float,
        default=1e-5,
        help='Max allowed absolute difference in probability for GB validation (default: 1e-5). Used with --validate-muscular-gb'
    )

    args = parser.parse_args()
    
    # Get challenger club paths
    challenger_path = PRODUCTION_ROOT / "deployments" / args.country / "challenger" / args.club
    timelines_dir = challenger_path / "timelines"
    predictions_dir = challenger_path / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine timelines file
    if args.timelines_file:
        timelines_file = Path(args.timelines_file)
    else:
        # Auto-detect: look for the main timelines file
        timelines_file = timelines_dir / "timelines_35day_season_2025_2026_v4_muscular.csv"
    
    if not timelines_file.exists():
        raise FileNotFoundError(f"Timelines file not found: {timelines_file}")
    
    # Determine output file
    if args.output_file:
        output_file = Path(args.output_file)
    else:
        if args.data_date:
            date_str = args.data_date
            print(f"[INFO] Using data_date parameter for filename: {date_str}")
        else:
            # Get max reference_date from timelines
            print(f"\n[LOAD] Loading timelines to determine date range...")
            timelines_df_temp = pd.read_csv(timelines_file, encoding='utf-8-sig', low_memory=False)
            if 'reference_date' in timelines_df_temp.columns:
                timelines_df_temp['reference_date'] = pd.to_datetime(timelines_df_temp['reference_date'], errors='coerce')
                max_date = timelines_df_temp['reference_date'].max()
                if pd.notna(max_date):
                    date_str = max_date.strftime('%Y%m%d')
                    print(f"[INFO] Using max reference_date for filename: {max_date.date()} ({date_str})")
                else:
                    from datetime import datetime
                    date_str = datetime.now().strftime('%Y%m%d')
                    print(f"[WARN] Could not determine max date, using today: {date_str}")
            else:
                from datetime import datetime
                date_str = datetime.now().strftime('%Y%m%d')
                print(f"[WARN] No reference_date column found, using today: {date_str}")
        
        output_file = predictions_dir / f"predictions_lgbm_v4_{date_str}.csv"
    
    # Check if output exists
    if output_file.exists() and not args.force:
        print(f"[SKIP] Predictions file already exists: {output_file}")
        print(f"       Use --force to regenerate")
        return 0
    
    print("=" * 80)
    print("GENERATE PREDICTIONS - V4 (4 models: muscular_lgbm, muscular_gb, skeletal, msu_lgbm)")
    print("=" * 80)
    print(f"Country: {args.country}")
    print(f"Club: {args.club}")
    print(f"Timelines file: {timelines_file}")
    print(f"Output file: {output_file}")
    print("=" * 80)
    
    # Load all four models
    models = load_models()
    
    # Load insight models (optional, shared with V3)
    insight_models_dir = PRODUCTION_ROOT / "models" / "insights"
    bodypart_pipeline = None
    severity_pipeline = None
    if (insight_models_dir / "bodypart_classifier.pkl").exists() and (insight_models_dir / "severity_classifier.pkl").exists():
        print(f"\n[LOAD] Loading insight models from {insight_models_dir}...")
        try:
            bodypart_pipeline, _ = load_pipeline(insight_models_dir / "bodypart_classifier")
            severity_pipeline, _ = load_pipeline(insight_models_dir / "severity_classifier")
            print(f"[LOAD] Insight models loaded successfully")
        except Exception as e:
            print(f"[WARN] Could not load insight models: {e}")
            print(f"[WARN] Continuing without body part and severity predictions")
    else:
        print(f"\n[INFO] Insight models not found at {insight_models_dir}")
        print(f"[INFO] Continuing without body part and severity predictions")
    
    # Load timelines
    print(f"\n[LOAD] Loading timelines from {timelines_file}...")
    timelines_df = pd.read_csv(timelines_file, encoding='utf-8-sig', low_memory=False)
    print(f"[LOAD] Loaded {len(timelines_df):,} timelines")
    
    if 'reference_date' in timelines_df.columns:
        timelines_df['reference_date'] = pd.to_datetime(timelines_df['reference_date'], errors='coerce')
        print(f"[LOAD] Date range: {timelines_df['reference_date'].min().date()} to {timelines_df['reference_date'].max().date()}")
        
        # Filter by min_date if provided
        if args.min_date:
            min_date_ts = pd.to_datetime(args.min_date).normalize()
            before_count = len(timelines_df)
            timelines_df = timelines_df[timelines_df['reference_date'] >= min_date_ts].copy()
            after_count = len(timelines_df)
            if before_count > after_count:
                print(f"[FILTER] Filtered timelines: removed {before_count - after_count} rows before {min_date_ts.date()}")
                print(f"[FILTER] Remaining timelines: {after_count:,} rows")
                if after_count > 0:
                    print(f"[FILTER] Filtered date range: {timelines_df['reference_date'].min().date()} to {timelines_df['reference_date'].max().date()}")
    
    # Get timelines file modification time
    timelines_file_mtime = os.path.getmtime(timelines_file) if timelines_file.exists() else None
    
    # Get daily features directory
    daily_features_dir = challenger_path / "daily_features"

    # Debug: dump one row's aligned features and prediction for comparison with training pipeline
    if getattr(args, 'debug_row', None):
        parts = [p.strip() for p in args.debug_row.split(',')]
        if len(parts) != 2:
            print(f"[DEBUG] Invalid --debug-row: use PLAYER_ID,REF_DATE (e.g. 144028,2025-12-05)")
        else:
            try:
                debug_player_id = int(parts[0])
                debug_ref_date = pd.to_datetime(parts[1]).normalize()
            except Exception as e:
                print(f"[DEBUG] Invalid --debug-row: {e}")
            else:
                mask = (timelines_df['player_id'] == debug_player_id) & (
                    pd.to_datetime(timelines_df['reference_date']).dt.normalize() == debug_ref_date
                )
                if not mask.any():
                    print(f"[DEBUG] No row found for player_id={debug_player_id}, reference_date={debug_ref_date}")
                else:
                    idx = timelines_df.index[mask][0]
                    model_columns = models['muscular_lgbm'][1]
                    X_aligned = align_features_to_model(timelines_df, model_columns)
                    aligned_row = X_aligned.loc[[idx]]
                    debug_csv = predictions_dir / f"debug_row_{debug_player_id}_{parts[1].replace('-', '')}_muscular_lgbm_features.csv"
                    aligned_row.to_csv(debug_csv, index=False, encoding='utf-8-sig')
                    print(f"[DEBUG] Wrote aligned muscular_lgbm feature row to: {debug_csv}")
                    prob = models['muscular_lgbm'][0].predict_proba(aligned_row)[:, 1][0]
                    print(f"[DEBUG] Muscular LGBM prediction for (player_id={debug_player_id}, reference_date={parts[1]}): {prob}")

    # Generate predictions (all four models, one CSV)
    predictions = generate_predictions(
        timelines_df=timelines_df,
        models=models,
        primary_key=PRIMARY_MODEL_KEY,
        bodypart_pipeline=bodypart_pipeline,
        severity_pipeline=severity_pipeline,
        daily_features_dir=daily_features_dir,
        timelines_file_mtime=timelines_file_mtime,
    )
    
    # Filter predictions by data_date if provided
    # Note: data_date is the reference date of the calculation process, so we include all predictions up to that date
    if args.data_date:
        data_date_ts = pd.to_datetime(args.data_date, format='%Y%m%d').normalize()
        print(f"\n[INFO] Using calculation reference date: {data_date_ts.date()}")
        before_count = len(predictions)
        predictions = predictions[predictions['reference_date'] <= data_date_ts].copy()
        after_count = len(predictions)
        if before_count > after_count:
            print(f"[FILTER] Filtered predictions: removed {before_count - after_count} rows beyond {data_date_ts.date()}")
            print(f"[FILTER] Remaining predictions: {after_count:,} rows")
        else:
            print(f"[INFO] All {after_count:,} predictions are within the calculation reference date range")
    
    # Save predictions
    print(f"\n[SAVE] Saving predictions to {output_file}...")
    predictions.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"[SAVE] Saved {len(predictions):,} predictions")
    
    # Optional: validate muscular LGBM against training test predictions (DEPLOYMENT_GUIDELINES.md)
    if getattr(args, 'validate_muscular_lgbm', False):
        ref_csv = MODELS_V4["muscular_lgbm"] / "test_predictions_from_training_pipeline.csv"
        tol = getattr(args, 'validate_tolerance', 1e-5)
        all_ok, max_diff, n_over = validate_muscular_lgbm_against_training(predictions, ref_csv, tolerance=tol)
        if not all_ok:
            print(f"[VALIDATE] FAILED: {n_over} row(s) exceed tolerance {tol:.0e}. Production should match training pipeline.")
            return 1
        print(f"[VALIDATE] OK: all overlapping rows within tolerance {tol:.0e}")
    
    # Optional: validate muscular GB against its training test predictions
    if getattr(args, 'validate_muscular_gb', False):
        ref_csv_gb = MODELS_V4["muscular_gb"] / "test_predictions_from_training_pipeline.csv"
        tol_gb = getattr(args, 'validate_gb_tolerance', 1e-5)
        all_ok_gb, max_diff_gb, n_over_gb = validate_muscular_gb_against_training(
            predictions,
            ref_csv_gb,
            tolerance=tol_gb,
        )
        if not all_ok_gb:
            print(
                f"[VALIDATE] FAILED (GB): {n_over_gb} row(s) exceed tolerance {tol_gb:.0e}. "
                "Production should match training pipeline for muscular GB."
            )
            return 1
        print(f"[VALIDATE] OK (GB): all overlapping rows within tolerance {tol_gb:.0e}")
    
    # Summary
    print(f"\n[SUMMARY] Prediction Summary:")
    print(f"   Total predictions: {len(predictions):,}")
    for col in ['injury_probability_muscular_lgbm', 'injury_probability_muscular_gb', 'injury_probability_skeletal', 'injury_probability_msu_lgbm']:
        if col in predictions.columns:
            print(f"   {col}: mean={predictions[col].mean():.4f}, max={predictions[col].max():.4f}")
    if 'injury_probability' in predictions.columns:
        print(f"   Primary (muscular_lgbm): High risk (>=0.3): {(predictions['injury_probability'] >= 0.3).sum():,}, "
              f"Medium (0.1-0.3): {((predictions['injury_probability'] >= 0.1) & (predictions['injury_probability'] < 0.3)).sum():,}, "
              f"Low (<0.1): {(predictions['injury_probability'] < 0.1).sum():,}")
    
    print("=" * 80)
    print("[COMPLETE] Prediction generation complete!")
    print("=" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
