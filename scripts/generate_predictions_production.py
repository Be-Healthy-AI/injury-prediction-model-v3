#!/usr/bin/env python3
"""
Generate daily injury predictions for production.

Reads timelines from production_predictions/timelines/ and generates predictions
using both Random Forest and Gradient Boosting models, then creates ensemble predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd

import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.backtests.run_predictions import encode_features

PRODUCTION_TIMELINES_DIR = ROOT_DIR / "production_predictions" / "timelines"
PRODUCTION_PREDICTIONS_DIR = ROOT_DIR / "production_predictions" / "predictions"
MODELS_DIR = ROOT_DIR / "models"

MODEL_CONFIG: Dict[str, Dict[str, Path]] = {
    "random_forest": {
        "model": MODELS_DIR / "model_v3_random_forest_final_100percent.pkl",
        "columns": MODELS_DIR / "model_v3_rf_final_columns.json",
    },
    "gradient_boosting": {
        "model": MODELS_DIR / "model_v3_gradient_boosting_final_100percent.pkl",
        "columns": MODELS_DIR / "model_v3_gb_final_columns.json",
    },
}

RISK_THRESHOLDS = {
    'Low': 0.3,
    'Medium': 0.5,
    'High': 1.0
}


def classify_risk_level(probability: float) -> str:
    """Classify injury risk level based on probability."""
    if probability < RISK_THRESHOLDS['Low']:
        return 'Low'
    elif probability < RISK_THRESHOLDS['Medium']:
        return 'Medium'
    else:
        return 'High'


def load_model(model_path: Path, columns_path: Path):
    """Load model and training columns."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not columns_path.exists():
        raise FileNotFoundError(f"Columns file not found: {columns_path}")
    
    model = joblib.load(model_path)
    with open(columns_path, 'r') as f:
        training_columns = json.load(f)
    
    return model, training_columns


def generate_predictions_for_timeline_file(
    timeline_file: Path,
    output_dir: Path,
    date_suffix: str
) -> pd.DataFrame:
    """Generate predictions for a single timeline file."""
    if not timeline_file.exists():
        raise FileNotFoundError(f"Timeline file not found: {timeline_file}")
    
    df = pd.read_csv(timeline_file)
    if df.empty:
        raise ValueError(f"Timeline file {timeline_file} is empty")
    
    # Extract player info
    if 'player_id' not in df.columns or 'reference_date' not in df.columns:
        raise ValueError(f"Timeline file {timeline_file} missing required columns")
    
    player_id = df['player_id'].iloc[0]
    player_name = df.get('player_name', [f'Player_{player_id}']).iloc[0]
    
    # Store probabilities from both models
    rf_proba = None
    gb_proba = None
    meta_cols = ["player_id", "reference_date"]
    if "player_name" in df.columns:
        meta_cols.append("player_name")
    
    # Generate predictions with each model
    for model_name, paths in MODEL_CONFIG.items():
        model, training_columns = load_model(paths["model"], paths["columns"])
        encoded = encode_features(df, training_columns)
        
        meta = encoded[meta_cols]
        features = encoded.drop(columns=meta_cols, errors='ignore')
        
        probabilities = model.predict_proba(features)[:, 1]
        predictions = meta.copy()
        predictions["injury_probability"] = probabilities
        predictions["model"] = model_name
        predictions["risk_level"] = [classify_risk_level(p) for p in probabilities]
        
        # Store for ensemble
        if model_name == "random_forest":
            rf_proba = probabilities
        elif model_name == "gradient_boosting":
            gb_proba = probabilities
        
        # Save individual model predictions
        model_output_dir = output_dir / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        output_file = model_output_dir / f"player_{player_id}_predictions_{date_suffix}.csv"
        predictions.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"   ✅ {model_name}: {output_file.name}")
    
    # Generate ensemble predictions
    if rf_proba is not None and gb_proba is not None:
        ensemble_proba = (rf_proba + gb_proba) / 2.0
        ensemble_predictions = meta.copy()
        ensemble_predictions["rf_probability"] = rf_proba
        ensemble_predictions["gb_probability"] = gb_proba
        ensemble_predictions["ensemble_probability"] = ensemble_proba
        ensemble_predictions["risk_level"] = [classify_risk_level(p) for p in ensemble_proba]
        
        ensemble_output_dir = output_dir / "ensemble"
        ensemble_output_dir.mkdir(parents=True, exist_ok=True)
        ensemble_file = ensemble_output_dir / f"player_{player_id}_predictions_{date_suffix}.csv"
        ensemble_predictions.to_csv(ensemble_file, index=False, encoding='utf-8-sig')
        print(f"   ✅ ensemble: {ensemble_file.name}")
        
        return ensemble_predictions
    
    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--timelines-dir',
        type=str,
        default=str(PRODUCTION_TIMELINES_DIR),
        help='Directory containing timeline files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(PRODUCTION_PREDICTIONS_DIR),
        help='Directory for output prediction files'
    )
    parser.add_argument(
        '--date',
        type=str,
        default=None,
        help='Target date for predictions (YYYY-MM-DD). Default: today'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Optional explicit start date (YYYY-MM-DD) for the prediction window'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='Optional explicit end date (YYYY-MM-DD) for the prediction window'
    )
    parser.add_argument(
        '--players',
        type=int,
        nargs='*',
        help='Specific player IDs to include (matches timeline files)'
    )
    args = parser.parse_args()
    
    timelines_dir = Path(args.timelines_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine date range
    if args.start_date:
        start_date = pd.to_datetime(args.start_date).normalize()
        if args.end_date:
            end_date = pd.to_datetime(args.end_date).normalize()
        elif args.date:
            end_date = pd.to_datetime(args.date).normalize()
        else:
            end_date = pd.Timestamp.today().normalize()
    else:
        if args.date:
            end_date = pd.to_datetime(args.date).normalize()
        else:
            end_date = pd.Timestamp.today().normalize()
        start_date = end_date
    
    if end_date < start_date:
        raise ValueError("End date cannot be earlier than start date.")
    
    start_label = start_date.strftime('%Y%m%d')
    end_label = end_date.strftime('%Y%m%d')
    date_str = end_label if start_label == end_label else f"{start_label}_{end_label}"
    
    print("=" * 70)
    print("GENERATE PREDICTIONS FOR PRODUCTION")
    print("=" * 70)
    print(f"📂 Timelines directory: {timelines_dir}")
    print(f"📂 Output directory: {output_dir}")
    print(f"📅 Date range: {start_date.date()} to {end_date.date()}")
    print()
    
    # Find all timeline files for the date range
    if args.players:
        timeline_files = []
        for pid in args.players:
            file_path = timelines_dir / f"player_{pid}_timelines_{date_str}.csv"
            if file_path.exists():
                timeline_files.append(file_path)
            else:
                print(f"⚠️  Timeline file not found for player {pid}: {file_path}")
    else:
        timeline_files = list(timelines_dir.glob(f"*_timelines_{date_str}.csv"))
    
    if not timeline_files:
        print(f"⚠️  No timeline files found for date {date_str}")
        print(f"   Looking for pattern: *_timelines_{date_str}.csv")
        return 1
    
    print(f"🎯 Processing {len(timeline_files)} timeline files...")
    print("-" * 70)
    
    all_predictions = []
    successful = 0
    failed = 0
    
    for timeline_file in timeline_files:
        try:
            player_id = int(timeline_file.stem.split('_')[1])
            print(f"\n[{successful + failed + 1}/{len(timeline_files)}] Player {player_id}")
            
            predictions = generate_predictions_for_timeline_file(
                timeline_file=timeline_file,
                output_dir=output_dir,
                date_suffix=date_str
            )
            
            if not predictions.empty:
                all_predictions.append(predictions)
            
            successful += 1
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            failed += 1
    
    # Create combined predictions file
    if all_predictions:
        combined_df = pd.concat(all_predictions, ignore_index=True)
        combined_file = output_dir / f"predictions_{date_str}.csv"
        combined_df.to_csv(combined_file, index=False, encoding='utf-8-sig')
        print(f"\n✅ Combined predictions: {combined_file.name}")
        print(f"   Total predictions: {len(combined_df)}")
        
        # Print summary statistics
        print("\n📊 Risk Level Distribution:")
        risk_counts = combined_df['risk_level'].value_counts()
        for level, count in risk_counts.items():
            print(f"   {level}: {count} ({count/len(combined_df)*100:.1f}%)")
        
        print("\n📊 Top 10 Highest Risk Players:")
        top_risk = combined_df.nlargest(10, 'ensemble_probability')[
            ['player_id', 'player_name', 'ensemble_probability', 'risk_level']
        ]
        for _, row in top_risk.iterrows():
            print(f"   Player {row['player_id']} ({row['player_name']}): "
                  f"{row['ensemble_probability']:.3f} ({row['risk_level']})")
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())

