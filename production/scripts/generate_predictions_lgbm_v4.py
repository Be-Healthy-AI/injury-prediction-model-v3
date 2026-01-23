#!/usr/bin/env python3
"""
Generate daily injury predictions for production using lgbm_muscular_v4 model.

Reads timelines from production/deployments/{country}/challenger/{club}/timelines/ and generates
predictions using the lgbm_muscular_v4 model (580 features).

The model predicts the probability of muscular injury within 35 days.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

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

# Model paths - V4
V4_MODEL_DIR = ROOT_DIR / "models_production" / "lgbm_muscular_v4" / "model"
V4_MODEL_PATH = V4_MODEL_DIR / "model.joblib"
V4_COLUMNS_PATH = V4_MODEL_DIR / "columns.json"


def load_model():
    """Load the lgbm_muscular_v4 model and feature columns."""
    if not V4_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {V4_MODEL_PATH}")
    if not V4_COLUMNS_PATH.exists():
        raise FileNotFoundError(f"Columns file not found: {V4_COLUMNS_PATH}")
    
    print(f"[LOAD] Loading V4 model from {V4_MODEL_PATH}...")
    model = joblib.load(V4_MODEL_PATH)
    
    print(f"[LOAD] Loading feature columns from {V4_COLUMNS_PATH}...")
    with open(V4_COLUMNS_PATH, 'r', encoding='utf-8', errors='replace') as f:
        model_columns = json.load(f)
    
    print(f"[LOAD] V4 model loaded: {len(model_columns)} features expected")
    return model, model_columns


def encode_categorical_features(timelines_df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode categorical features that the V4 model expects.
    
    The model expects one-hot encoded versions of:
    - current_club_country -> current_club_country_England, current_club_country_Northern_Ireland, etc.
    - dominant_foot -> dominant_foot_left, dominant_foot_right
    - last_match_position_week_X -> last_match_position_week_X_* (for each position)
    """
    df_encoded = timelines_df.copy()
    
    # List of categorical features that need one-hot encoding
    categorical_features = []
    
    # Check for current_club_country (categorical)
    if 'current_club_country' in df_encoded.columns:
        if df_encoded['current_club_country'].dtype == 'object' or df_encoded['current_club_country'].nunique() < 50:
            categorical_features.append('current_club_country')
    
    # Check for dominant_foot (categorical)
    if 'dominant_foot' in df_encoded.columns:
        if df_encoded['dominant_foot'].dtype == 'object' or df_encoded['dominant_foot'].nunique() < 10:
            categorical_features.append('dominant_foot')
    
    # Check for nationality1 and nationality2 (categorical)
    for nat_col in ['nationality1', 'nationality2']:
        if nat_col in df_encoded.columns:
            if df_encoded[nat_col].dtype == 'object' or df_encoded[nat_col].nunique() < 200:
                if nat_col not in categorical_features:
                    categorical_features.append(nat_col)
    
    # Check for last_match_position_week_X columns
    # These are categorical columns like 'last_match_position_week_1' that need to be one-hot encoded
    # into columns like 'last_match_position_week_1_Central_Midfielder', etc.
    for col in df_encoded.columns:
        # Match pattern: last_match_position_week_X (where X is 1-5)
        if col.startswith('last_match_position_week_') and col.replace('last_match_position_week_', '').isdigit():
            # This is a categorical column that needs encoding
            # Check if it's object type or has reasonable number of unique non-null values
            non_null_count = df_encoded[col].notna().sum()
            unique_count = df_encoded[col].nunique()
            if df_encoded[col].dtype == 'object' or (non_null_count > 0 and unique_count < 20):
                if col not in categorical_features:
                    categorical_features.append(col)
    
    # One-hot encode each categorical feature
    for feature in categorical_features:
        if feature not in df_encoded.columns:
            continue
        
        # Clean the values (handle NaN, empty strings, etc.)
        feature_values = df_encoded[feature].copy()
        
        # Replace empty strings with NaN
        if feature_values.dtype == 'object':
            feature_values = feature_values.replace('', pd.NA)
        
        # Normalize values to match model expectations
        # For positions: "Defensive Midfielder" -> "Defensive_Midfielder" (replace spaces with underscores)
        # For dominant_foot: "right" -> "right", "left" -> "left" (already correct)
        # For current_club_country: keep as is
        # For nationality: normalize country names (replace spaces with underscores, handle special chars)
        if 'last_match_position' in feature:
            # Normalize position names: replace spaces with underscores
            feature_values = feature_values.astype(str).str.replace(' ', '_', regex=False)
            feature_values = feature_values.replace('nan', pd.NA)
        elif 'nationality' in feature:
            # Normalize nationality names: replace spaces with underscores, handle special characters
            feature_values = feature_values.astype(str).str.replace(' ', '_', regex=False)
            feature_values = feature_values.str.replace("'", '', regex=False)  # Remove apostrophes
            feature_values = feature_values.str.replace('-', '_', regex=False)  # Replace hyphens with underscores
            feature_values = feature_values.replace('nan', pd.NA)
        
        # For one-hot encoding, we need to handle NaN specially
        # pd.get_dummies will create columns for all non-null values
        # For null values, all dummy columns will be 0 (which is what we want)
        feature_values_clean = feature_values.fillna('__MISSING__').astype(str)
        feature_values_clean = feature_values_clean.replace('', '__MISSING__')
        feature_values_clean = feature_values_clean.str.strip()
        
        # Create one-hot encoded columns
        # Use prefix matching the model's expected format: feature_value
        dummies = pd.get_dummies(feature_values_clean, prefix=feature, drop_first=False)
        
        # Remove the __MISSING__ column if it exists (we don't want it)
        if f'{feature}__MISSING__' in dummies.columns:
            dummies = dummies.drop(columns=[f'{feature}__MISSING__'])
        
        # For rows where the original value was NaN, set all dummy columns to 0
        nan_mask = pd.isna(feature_values)
        if nan_mask.any():
            # Convert to int to avoid dtype warnings
            dummies = dummies.astype(int)
            dummies.loc[nan_mask, :] = 0
        
        # Clean column names to match model format exactly
        # Remove any numeric suffixes that pd.get_dummies might add
        dummies.columns = [col.replace('_1.0', '').replace('_0.0', '') for col in dummies.columns]
        
        # Add dummy columns to dataframe
        df_encoded = pd.concat([df_encoded, dummies], axis=1)
        
        # Drop original categorical column
        df_encoded = df_encoded.drop(columns=[feature])
    
    if categorical_features:
        print(f"[ENCODE] One-hot encoded {len(categorical_features)} categorical features: {categorical_features}")
    
    return df_encoded


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
    model,
    model_columns: list,
    bodypart_pipeline=None,
    severity_pipeline=None,
    daily_features_dir: Optional[Path] = None,
    timelines_file_mtime: Optional[float] = None
) -> pd.DataFrame:
    """
    Generate predictions for V4 timelines data.
    
    V4 timelines should already have the correct features, so we just need to:
    1. Align features to model column order
    2. Generate predictions
    3. Add SHAP values and insights
    """
    print(f"\n[PREPROCESS] Preparing data for {len(timelines_df):,} timelines...")
    
    # Reset index to ensure alignment
    timelines_df = timelines_df.reset_index(drop=True)
    
    # Align features to match model columns
    print(f"\n[ALIGN] Aligning features to V4 model...")
    X_aligned = align_features_to_model(timelines_df, model_columns)
    
    # Store the index of rows that were successfully aligned
    aligned_indices = X_aligned.index
    
    # Generate injury probabilities
    print(f"\n[PREDICT] Generating risk scores...")
    risk_scores = model.predict_proba(X_aligned)[:, 1]
    
    # Compute SHAP values for feature importance
    print(f"\n[SHAP] Computing feature importance...")
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, module='shap')
        shap_explainer = shap.TreeExplainer(model)
        shap_values = shap_explainer.shap_values(X_aligned)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Get positive class SHAP values
    
    # Get top 10 features for each prediction
    top_features_list = []
    feature_names = X_aligned.columns.tolist()
    for row_idx, row_shap in enumerate(shap_values):
        indices = np.argsort(np.abs(row_shap))[::-1][:10]
        top_features = [
            {
                "name": feature_names[idx],
                "value": float(X_aligned.iloc[row_idx, idx]),
                "shap": float(row_shap[idx]),
                "abs_shap": float(abs(row_shap[idx]))
            }
            for idx in indices
        ]
        top_features_list.append(top_features)
    
    # Predict body part and severity for each date (only for aligned rows)
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
            
            # Load daily features for this player and date
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
                    
                    # Get top body part
                    if insights.get("bodypart_rank"):
                        top_body = insights["bodypart_rank"][0]
                        body_part_predictions.append(top_body[0])
                        body_part_probs.append(top_body[1])
                    else:
                        body_part_predictions.append("unknown")
                        body_part_probs.append(0.0)
                    
                    # Get severity
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
        # Fill with defaults if insight models not available
        body_part_predictions = ["unknown"] * len(aligned_indices)
        body_part_probs = [0.0] * len(aligned_indices)
        severity_predictions = ["unknown"] * len(aligned_indices)
        severity_probs = [0.0] * len(aligned_indices)
    
    # Build predictions DataFrame
    predictions = pd.DataFrame({
        'player_id': timelines_df.loc[aligned_indices, 'player_id'].values,
        'reference_date': timelines_df.loc[aligned_indices, 'reference_date'].values,
        'injury_probability': risk_scores,
    })
    
    # Add risk classification
    predictions['risk_level'] = predictions['injury_probability'].apply(classify_risk_4level)
    
    # Add body part and severity predictions
    predictions['predicted_body_part'] = body_part_predictions
    predictions['body_part_probability'] = body_part_probs
    predictions['predicted_severity'] = severity_predictions
    predictions['severity_probability'] = severity_probs
    
    # Add top 10 features
    for i in range(10):
        predictions[f'top_feature_{i+1}_name'] = [feat[i]['name'] if i < len(feat) else '' for feat in top_features_list]
        predictions[f'top_feature_{i+1}_value'] = [feat[i]['value'] if i < len(feat) else 0.0 for feat in top_features_list]
        predictions[f'top_feature_{i+1}_shap'] = [feat[i]['shap'] if i < len(feat) else 0.0 for feat in top_features_list]
    
    # Add player name if available
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
    print("GENERATE PREDICTIONS - lgbm_muscular_v4")
    print("=" * 80)
    print(f"Country: {args.country}")
    print(f"Club: {args.club}")
    print(f"Timelines file: {timelines_file}")
    print(f"Output file: {output_file}")
    print("=" * 80)
    
    # Load model
    model, model_columns = load_model()
    
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
    
    # Get timelines file modification time
    timelines_file_mtime = os.path.getmtime(timelines_file) if timelines_file.exists() else None
    
    # Get daily features directory
    daily_features_dir = challenger_path / "daily_features"
    
    # Generate predictions
    predictions = generate_predictions(
        timelines_df=timelines_df,
        model=model,
        model_columns=model_columns,
        bodypart_pipeline=bodypart_pipeline,
        severity_pipeline=severity_pipeline,
        daily_features_dir=daily_features_dir,
        timelines_file_mtime=timelines_file_mtime
    )
    
    # Filter predictions by data_date if provided
    if args.data_date:
        data_date_ts = pd.to_datetime(args.data_date, format='%Y%m%d').normalize()
        before_count = len(predictions)
        predictions = predictions[predictions['reference_date'] <= data_date_ts].copy()
        after_count = len(predictions)
        if before_count > after_count:
            print(f"\n[FILTER] Filtered predictions: removed {before_count - after_count} rows beyond {data_date_ts.date()}")
            print(f"[FILTER] Remaining predictions: {after_count:,} rows")
    
    # Save predictions
    print(f"\n[SAVE] Saving predictions to {output_file}...")
    predictions.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"[SAVE] Saved {len(predictions):,} predictions")
    
    # Summary
    print(f"\n[SUMMARY] Prediction Summary:")
    print(f"   Total predictions: {len(predictions):,}")
    if 'injury_probability' in predictions.columns:
        print(f"   Average probability: {predictions['injury_probability'].mean():.4f}")
        print(f"   Max probability: {predictions['injury_probability'].max():.4f}")
        print(f"   High risk (>=0.3): {(predictions['injury_probability'] >= 0.3).sum():,}")
        print(f"   Medium risk (0.1-0.3): {((predictions['injury_probability'] >= 0.1) & (predictions['injury_probability'] < 0.3)).sum():,}")
        print(f"   Low risk (<0.1): {(predictions['injury_probability'] < 0.1).sum():,}")
    
    print("=" * 80)
    print("[COMPLETE] Prediction generation complete!")
    print("=" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
