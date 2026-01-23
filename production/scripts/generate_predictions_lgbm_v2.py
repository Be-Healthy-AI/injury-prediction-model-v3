#!/usr/bin/env python3
"""
Generate daily injury predictions for production using lgbm_muscular_v2 model.

Reads timelines from production/deployments/{country}/{club}/timelines/ and generates
predictions using the lgbm_muscular_v2 winner model.

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

# Import preprocessing functions
from production.scripts.preprocessing_lgbm_v2 import (
    prepare_data,
    align_features_to_model
)

# Import insight utilities
from production.scripts.insight_utils import (
    load_pipeline,
    predict_insights,
    classify_risk_4level,
)

# Model paths
MODEL_DIR = PRODUCTION_ROOT / "models" / "lgbm_muscular_v2"
MODEL_PATH = MODEL_DIR / "model.joblib"
COLUMNS_PATH = MODEL_DIR / "columns.json"


def load_model():
    """Load the lgbm_muscular_v2 model and feature columns."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not COLUMNS_PATH.exists():
        raise FileNotFoundError(f"Columns file not found: {COLUMNS_PATH}")
    
    print(f"[LOAD] Loading model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)
    
    print(f"[LOAD] Loading feature columns from {COLUMNS_PATH}...")
    with open(COLUMNS_PATH, 'r', encoding='utf-8', errors='replace') as f:
        model_columns = json.load(f)
    
    print(f"[LOAD] Model loaded: {len(model_columns)} features expected")
    return model, model_columns


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
    cache_file: Optional[str] = None,
    timelines_file_mtime: Optional[float] = None
) -> pd.DataFrame:
    """
    Generate predictions for timelines data with body part, severity, SHAP, and risk classification.
    
    Args:
        timelines_df: DataFrame with timelines data
        model: Trained LightGBM model
        model_columns: List of feature column names expected by model
        bodypart_pipeline: Optional body part prediction pipeline
        severity_pipeline: Optional severity prediction pipeline
        daily_features_dir: Directory containing daily features files
        cache_file: Optional path to cache preprocessed data
        
    Returns:
        DataFrame with predictions including risk level, body part, severity, and top features
    """
    print(f"\n[PREPROCESS] Preparing data for {len(timelines_df):,} timelines...")
    
    # Reset index to ensure alignment between timelines_df and processed data
    # This ensures that indices match even when loading from cache
    original_index = timelines_df.index.copy()
    timelines_df = timelines_df.reset_index(drop=True)
    
    # Prepare data (preprocessing + encoding)
    # Pass timelines_file_mtime to invalidate cache if timelines file was modified
    X, y = prepare_data(timelines_df, cache_file=cache_file, use_cache=True, timelines_file_mtime=timelines_file_mtime)
    
    # Ensure X has sequential index matching timelines_df (cache might reset index)
    if len(X) != len(timelines_df):
        raise ValueError(f"Data length mismatch: timelines_df has {len(timelines_df)} rows but X has {len(X)} rows")
    # Reset X index to match timelines_df (0, 1, 2, ...)
    X = X.reset_index(drop=True)
    X.index = timelines_df.index
    
    # Align features to match model columns
    print(f"\n[ALIGN] Aligning features to model...")
    X_aligned = align_features_to_model(X, model_columns)
    
    # Store the index of rows that were successfully aligned (for matching with timelines_df)
    aligned_indices = X_aligned.index
    
    # Generate injury probabilities
    print(f"\n[PREDICT] Generating risk scores...")
    risk_scores = model.predict_proba(X_aligned)[:, 1]
    
    # Compute SHAP values for feature importance
    print(f"\n[SHAP] Computing feature importance...")
    # Suppress SHAP UserWarning about LightGBM binary classifier output format
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
        # Cache daily features per player
        daily_features_cache = {}
        
        # Iterate only through the rows that were successfully aligned
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
        # No insight models available
        body_part_predictions = ["unknown"] * len(aligned_indices)
        body_part_probs = [0.0] * len(aligned_indices)
        severity_predictions = ["unknown"] * len(aligned_indices)
        severity_probs = [0.0] * len(aligned_indices)
    
    # Classify risk levels (4-level)
    risk_levels = [classify_risk_4level(prob)["label"] for prob in risk_scores]
    
    # Build predictions DataFrame (only for aligned rows)
    predictions = pd.DataFrame({
        'player_id': timelines_df.loc[aligned_indices, 'player_id'].values,
        'reference_date': timelines_df.loc[aligned_indices, 'reference_date'].values,
        'injury_probability': risk_scores,
        'risk_level': risk_levels,
        'predicted_body_part': body_part_predictions,
        'body_part_probability': body_part_probs,
        'predicted_severity': severity_predictions,
        'severity_probability': severity_probs,
    })
    
    # Add top 10 features as columns
    for feat_idx in range(10):
        predictions[f'top_feature_{feat_idx+1}_name'] = [feat[feat_idx]['name'] if len(feat) > feat_idx else '' for feat in top_features_list]
        predictions[f'top_feature_{feat_idx+1}_value'] = [feat[feat_idx]['value'] if len(feat) > feat_idx else 0.0 for feat in top_features_list]
        predictions[f'top_feature_{feat_idx+1}_shap'] = [feat[feat_idx]['shap'] if len(feat) > feat_idx else 0.0 for feat in top_features_list]
    
    # Add player_name if available
    if 'player_name' in timelines_df.columns:
        predictions['player_name'] = timelines_df.loc[aligned_indices, 'player_name'].values
    
    print(f"[PREDICT] Generated {len(predictions):,} predictions")
    print(f"[PREDICT] Risk score range: {risk_scores.min():.4f} - {risk_scores.max():.4f}")
    print(f"[PREDICT] Mean risk score: {risk_scores.mean():.4f}")
    
    return predictions


def main():
    parser = argparse.ArgumentParser(
        description='Generate daily injury predictions using lgbm_muscular_v2 model'
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
        default='Chelsea FC',
        help='Club name (default: Chelsea FC)'
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
        help='Output file path (default: predictions_lgbm_v2_YYYYMMDD.csv in predictions folder)'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default=None,
        help='Directory for caching preprocessed data (default: production/cache)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force regeneration even if predictions already exist'
    )
    
    args = parser.parse_args()
    
    # Get club paths
    club_path = PRODUCTION_ROOT / "deployments" / args.country / args.club
    timelines_dir = club_path / "timelines"
    predictions_dir = club_path / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine timelines file
    if args.timelines_file:
        timelines_file = Path(args.timelines_file)
    else:
        # Auto-detect: look for the main timelines file
        timelines_file = timelines_dir / "timelines_35day_season_2025_2026_v4_muscular.csv"
    
    if not timelines_file.exists():
        raise FileNotFoundError(f"Timelines file not found: {timelines_file}")
    
    # Determine output file - use max reference_date from timelines
    if args.output_file:
        output_file = Path(args.output_file)
    else:
        # Get max reference_date from timelines to use for filename
        # Load timelines first to get the date range
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
        output_file = predictions_dir / f"predictions_lgbm_v2_{date_str}.csv"
    
    # Check if output exists
    if output_file.exists() and not args.force:
        print(f"[SKIP] Predictions file already exists: {output_file}")
        print(f"       Use --force to regenerate")
        return 0
    
    # Determine cache file
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
    else:
        cache_dir = PRODUCTION_ROOT / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"preprocessed_{timelines_file.stem}.csv"
    
    print("=" * 80)
    print("GENERATE PREDICTIONS - lgbm_muscular_v2")
    print("=" * 80)
    print(f"Country: {args.country}")
    print(f"Club: {args.club}")
    print(f"Timelines file: {timelines_file}")
    print(f"Output file: {output_file}")
    print(f"Cache file: {cache_file}")
    print("=" * 80)
    
    # Load model
    model, model_columns = load_model()
    
    # Load insight models (optional)
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
    
    # Get timelines file modification time for cache validation
    timelines_file_mtime = os.path.getmtime(timelines_file) if timelines_file.exists() else None
    
    # Get daily features directory
    daily_features_dir = club_path / "daily_features"
    
    # Generate predictions
    predictions = generate_predictions(
        timelines_df=timelines_df,
        model=model,
        model_columns=model_columns,
        bodypart_pipeline=bodypart_pipeline,
        severity_pipeline=severity_pipeline,
        daily_features_dir=daily_features_dir,
        cache_file=str(cache_file),
        timelines_file_mtime=timelines_file_mtime
    )
    
    # Get max reference_date from predictions for per-player filenames
    if 'reference_date' in predictions.columns:
        max_ref_date = predictions['reference_date'].max()
        if pd.notna(max_ref_date):
            date_str = pd.to_datetime(max_ref_date).strftime('%Y%m%d')
            print(f"[INFO] Using max reference_date for per-player files: {pd.to_datetime(max_ref_date).date()} ({date_str})")
        else:
            from datetime import datetime
            date_str = datetime.now().strftime('%Y%m%d')
            print(f"[WARN] Could not determine max date, using today: {date_str}")
    else:
        from datetime import datetime
        date_str = datetime.now().strftime('%Y%m%d')
        print(f"[WARN] No reference_date column in predictions, using today: {date_str}")
    
    # Save aggregated predictions
    print(f"\n[SAVE] Saving aggregated predictions to {output_file}...")
    predictions.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"[SAVE] Saved {len(predictions):,} predictions")
    
    # Generate per-player CSVs with profile data
    print(f"\n[PLAYERS] Generating per-player prediction files...")
    players_dir = predictions_dir / "players"
    players_dir.mkdir(parents=True, exist_ok=True)
    
    # Get latest raw data folder for profiles
    latest_raw_data = get_latest_raw_data_folder(args.country.lower().replace(" ", "_"))
    if latest_raw_data:
        print(f"[PROFILES] Using profiles from {latest_raw_data}")
    else:
        print(f"[WARN] Could not find latest raw data folder for profiles")
    
    # date_str is now determined from predictions above, not from datetime.now()
    
    players_processed = 0
    for player_id in predictions['player_id'].unique():
        player_predictions = predictions[predictions['player_id'] == player_id].copy()
        
        # Load player profile
        if latest_raw_data:
            player_profile = load_player_profile(player_id, latest_raw_data)
            if not player_profile.empty:
                # Merge profile columns
                profile_cols = player_profile.to_dict()
                for col, val in profile_cols.items():
                    if col != 'player_id':  # Don't duplicate player_id
                        player_predictions[col] = val
        
        # Save per-player CSV
        player_csv = players_dir / f"player_{player_id}_predictions_{date_str}.csv"
        player_predictions.to_csv(player_csv, index=False, encoding='utf-8-sig')
        players_processed += 1
    
    print(f"[PLAYERS] Generated {players_processed} per-player prediction files")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("PREDICTION SUMMARY")
    print("=" * 80)
    print(f"Total predictions: {len(predictions):,}")
    print(f"Risk score statistics:")
    print(f"  Min: {predictions['injury_probability'].min():.4f}")
    print(f"  Max: {predictions['injury_probability'].max():.4f}")
    print(f"  Mean: {predictions['injury_probability'].mean():.4f}")
    print(f"  Median: {predictions['injury_probability'].median():.4f}")
    print(f"  Std: {predictions['injury_probability'].std():.4f}")
    
    # Top 10 highest risk predictions
    print(f"\nTop 10 Highest Risk Predictions:")
    top_risk = predictions.nlargest(10, 'injury_probability')
    for idx, row in top_risk.iterrows():
        player_info = f"Player {row['player_id']}"
        if 'player_name' in row and pd.notna(row['player_name']):
            player_info += f" ({row['player_name']})"
        print(f"  {player_info} on {row['reference_date']}: {row['injury_probability']:.4f}")
    
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    exit(main())

