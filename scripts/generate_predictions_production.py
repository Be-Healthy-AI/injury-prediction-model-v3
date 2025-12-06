#!/usr/bin/env python3
"""
Generate daily injury predictions for production.

Reads timelines from production_predictions/timelines/ and generates predictions
using Random Forest and Gradient Boosting models trained on combined train+val data,
then creates ensemble predictions using Weighted Average (70% GB + 30% RF) - the
best performing configuration (F1 >30%, AUC ~76%).
"""

from __future__ import annotations

import argparse
import json
import io
import time
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

import sys

# Try to import tqdm for progress bars, fallback to simple counter if not available
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    class tqdm:
        def __init__(self, iterable, desc="", unit="", disable=False, bar_format=None):
            self.iterable = iterable
            self.desc = desc
            self.unit = unit
            self.disable = disable
            self.bar_format = bar_format
            self.n = 0
            self.total = len(iterable) if hasattr(iterable, '__len__') else None
        
        def __iter__(self):
            return iter(self.iterable)
        
        def __enter__(self):
            if not self.disable:
                print(f"{self.desc}: Starting...")
                sys.stdout.flush()
            return self
        
        def __exit__(self, *args):
            if not self.disable:
                print(f"{self.desc}: Complete!")
                sys.stdout.flush()
        
        def set_description(self, desc):
            """Update the description."""
            self.desc = desc
        
        def update(self, n=1):
            self.n += n
            if not self.disable and self.total:
                pct = (self.n / self.total) * 100
                print(f"\r{self.desc}: {self.n}/{self.total} ({pct:.1f}%)", end="", flush=True)
        
        @staticmethod
        def write(msg):
            """Write a message (for compatibility with real tqdm)."""
            print(msg)
            sys.stdout.flush()

# Fix Unicode encoding for Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.backtests.run_predictions import encode_features

PRODUCTION_TIMELINES_DIR = ROOT_DIR / "production_predictions" / "timelines"
PRODUCTION_PREDICTIONS_DIR = ROOT_DIR / "production_predictions" / "predictions"
MODELS_DIR = ROOT_DIR / "models"

# Best performing models: Combined Train+Val configuration
# Using Weighted Average (70% GB + 30% RF) ensemble
MODEL_CONFIG: Dict[str, Dict[str, Path]] = {
    "random_forest": {
        "model": MODELS_DIR / "rf_model_combined_trainval.joblib",
        "columns": MODELS_DIR / "rf_model_combined_trainval_columns.json",
    },
    "gradient_boosting": {
        "model": MODELS_DIR / "gb_model_combined_trainval.joblib",
        "columns": MODELS_DIR / "gb_model_combined_trainval_columns.json",
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


def load_all_models():
    """Load all models once at startup for efficiency."""
    print("📦 Loading Combined Train+Val models...")
    sys.stdout.flush()
    models = {}
    start_load = time.time()
    
    for model_name, paths in MODEL_CONFIG.items():
        load_start = time.time()
        print(f"   Loading {model_name}...", end=" ", flush=True)
        try:
            model, training_columns = load_model(paths["model"], paths["columns"])
            models[model_name] = {
                "model": model,
                "columns": training_columns
            }
            load_time = time.time() - load_start
            print(f"✅ ({load_time:.1f}s)", flush=True)
        except Exception as e:
            print(f"❌ Error: {e}", flush=True)
            raise
    
    total_load_time = time.time() - start_load
    print(f"✅ All {len(models)} models loaded in {total_load_time:.1f}s!\n")
    sys.stdout.flush()
    return models


def generate_predictions_for_timeline_file(
    timeline_file: Path,
    output_dir: Path,
    date_suffix: str,
    models: Dict,  # Pre-loaded models
    force: bool = False
) -> pd.DataFrame:
    """Generate predictions for a single timeline file using Phase 1 & Phase 2 models."""
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
    
    # Store probabilities from all models
    rf_phase1_proba = None
    gb_phase1_proba = None
    rf_phase2_proba = None
    gb_phase2_proba = None
    meta_cols = ["player_id", "reference_date"]
    if "player_name" in df.columns:
        meta_cols.append("player_name")
    
    meta = None
    rf_proba = None
    gb_proba = None
    
    # Generate predictions with each model (using pre-loaded models)
    for model_name, model_data in models.items():
        model = model_data["model"]  # Use pre-loaded model
        training_columns = model_data["columns"]  # Use pre-loaded columns
        encoded = encode_features(df, training_columns)
        
        if meta is None:
            meta = encoded[meta_cols]
        features = encoded.drop(columns=meta_cols, errors='ignore')
        
        probabilities = model.predict_proba(features)[:, 1]
        
        # Store probabilities for ensemble
        if model_name == "random_forest":
            rf_proba = probabilities
        elif model_name == "gradient_boosting":
            gb_proba = probabilities
        
        # Save individual model predictions (optional, for debugging)
        model_output_dir = output_dir / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        predictions = meta.copy()
        predictions["injury_probability"] = probabilities
        predictions["model"] = model_name
        predictions["risk_level"] = [classify_risk_level(p) for p in probabilities]
        output_file = model_output_dir / f"player_{player_id}_predictions_{date_suffix}.csv"
        predictions.to_csv(output_file, index=False, encoding='utf-8-sig')
        if not HAS_TQDM:  # Only print detailed info if no progress bar
            print(f"   ✅ {model_name}: {output_file.name}")
            sys.stdout.flush()
    
    # Generate ensemble predictions using Weighted Average (70% GB + 30% RF)
    if rf_proba is not None and gb_proba is not None and meta is not None:
        # Weighted Average: 70% Gradient Boosting + 30% Random Forest
        # This is the best performing configuration (F1 >30%, AUC ~76%)
        ensemble_proba = 0.7 * gb_proba + 0.3 * rf_proba
        
        ensemble_predictions = meta.copy()
        ensemble_predictions["rf_probability"] = rf_proba
        ensemble_predictions["gb_probability"] = gb_proba
        ensemble_predictions["ensemble_probability"] = ensemble_proba
        ensemble_predictions["risk_level"] = [classify_risk_level(p) for p in ensemble_proba]
        
        ensemble_output_dir = output_dir / "ensemble"
        ensemble_output_dir.mkdir(parents=True, exist_ok=True)
        ensemble_file = ensemble_output_dir / f"player_{player_id}_predictions_{date_suffix}.csv"
        
        # Check if file exists and force flag
        if ensemble_file.exists() and not force:
            print(f"   ⏭️  Skipping (file exists, use --force to regenerate): {ensemble_file.name}")
            sys.stdout.flush()
            return ensemble_predictions
        
        ensemble_predictions.to_csv(ensemble_file, index=False, encoding='utf-8-sig')
        print(f"   ✅ ensemble (Weighted Avg: 70% GB + 30% RF): {ensemble_file.name}")
        sys.stdout.flush()
        
        return ensemble_predictions
    
    raise ValueError("Failed to generate predictions from all required models")


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
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force regeneration even if predictions already exist'
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
    print("Using Combined Train+Val Models with Weighted Average Ensemble")
    print("=" * 70)
    print(f"📂 Timelines directory: {timelines_dir}")
    print(f"📂 Output directory: {output_dir}")
    print(f"📅 Date range: {start_date.date()} to {end_date.date()}")
    print(f"🤖 Models: Combined Train+Val (RF + GB) → Weighted Avg (70% GB + 30% RF)")
    print(f"📊 Performance: F1 >30%, AUC ~76% (best configuration)")
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
    if args.force:
        print("🔄 Force mode: Regenerating all predictions")
    print("-" * 70)
    sys.stdout.flush()
    
    # Load all models ONCE at the start (major performance improvement)
    all_models = load_all_models()
    
    all_predictions = []
    successful = 0
    failed = 0
    start_time = time.time()
    
    # Enhanced progress bar with percentage and ETA
    if HAS_TQDM:
        progress_bar = tqdm(
            timeline_files, 
            desc="Generating predictions", 
            unit="player",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {percentage:3.0f}%',
            disable=False
        )
    else:
        progress_bar = timeline_files
    
    for timeline_file in progress_bar:
        try:
            player_id = int(timeline_file.stem.split('_')[1])
            
            # Update progress bar description with current player
            if HAS_TQDM:
                progress_bar.set_description(f"Player {player_id}")
            else:
                current = successful + failed + 1
                pct = (current / len(timeline_files)) * 100
                elapsed = time.time() - start_time
                if current > 1:
                    avg_time = elapsed / (current - 1)
                    remaining = avg_time * (len(timeline_files) - current)
                    eta_str = f", ETA: {remaining:.0f}s"
                else:
                    eta_str = ""
                print(f"\n[{current}/{len(timeline_files)}] ({pct:.1f}%) Player {player_id}{eta_str}")
                sys.stdout.flush()
            
            predictions = generate_predictions_for_timeline_file(
                timeline_file=timeline_file,
                output_dir=output_dir,
                date_suffix=date_str,
                models=all_models,  # Pass pre-loaded models
                force=args.force
            )
            
            if not predictions.empty:
                all_predictions.append(predictions)
            
            successful += 1
            if HAS_TQDM:
                progress_bar.update(1)
            else:
                print(f"   ✅ Success")
                sys.stdout.flush()
            
        except Exception as e:
            error_msg = f"   ❌ Error: {e}"
            if HAS_TQDM:
                tqdm.write(error_msg)
            else:
                print(error_msg)
            sys.stdout.flush()
            failed += 1
            if HAS_TQDM:
                progress_bar.update(1)
    
    # Create combined predictions file
    if all_predictions:
        combined_df = pd.concat(all_predictions, ignore_index=True)
        combined_file = output_dir / f"predictions_{date_str}.csv"
        combined_df.to_csv(combined_file, index=False, encoding='utf-8-sig')
        print(f"\n✅ Combined predictions: {combined_file.name}")
        print(f"   Total predictions: {len(combined_df)}")
        sys.stdout.flush()
        
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
        sys.stdout.flush()
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    sys.stdout.flush()
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())

