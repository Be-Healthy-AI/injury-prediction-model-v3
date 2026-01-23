#!/usr/bin/env python3
"""
Generate per-player dashboards for V4 production predictions.

This script is adapted from generate_dashboards.py for V4 challenger model.
It reads predictions from challenger/{club}/predictions/ and generates dashboards.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

try:
    from scipy.interpolate import make_interp_spline
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

import sys
import io

# Calculate paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent
ROOT_DIR = PRODUCTION_ROOT.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from production.scripts.insight_utils import (
    RISK_CLASS_COLORS_4LEVEL,
    RISK_CLASS_LABELS_4LEVEL,
    RISK_THRESHOLDS_4LEVEL,
    classify_risk_4level,
    compute_risk_series,
    compute_trend_metrics,
    load_pipeline,
    predict_insights,
)

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, desc="", unit=""):
        return iterable

# Fix Unicode encoding for Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def get_challenger_path(country: str, club: str) -> Path:
    """Get the base path for a challenger club deployment."""
    return PRODUCTION_ROOT / "deployments" / country / "challenger" / club


def get_latest_raw_data_folder(country: str = "england") -> Optional[Path]:
    """Get the latest raw data folder."""
    base_dir = PRODUCTION_ROOT / "raw_data" / country.lower().replace(" ", "_")
    if not base_dir.exists():
        return None
    
    date_folders = [d for d in base_dir.iterdir() if d.is_dir() and d.name.isdigit() and len(d.name) == 8]
    if not date_folders:
        return None
    
    return max(date_folders, key=lambda x: x.name)


@dataclass
class PlayerDashboardContext:
    player_id: int
    player_name: str
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    suffix: str

    @property
    def entry_id(self) -> str:
        return f"player_{self.player_id}_{self.suffix}"

    @property
    def chart_filename(self) -> str:
        return f"{self.entry_id}_v4_probabilities.png"


def parse_date_range(args: argparse.Namespace) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    if args.start_date:
        start_date = pd.to_datetime(args.start_date).normalize()
        if args.end_date:
            end_date = pd.to_datetime(args.end_date).normalize()
        else:
            end_date = pd.Timestamp.today().normalize()
    else:
        if args.end_date:
            end_date = pd.to_datetime(args.end_date).normalize()
        elif args.date:
            end_date = pd.to_datetime(args.date).normalize()
        else:
            end_date = pd.Timestamp.today().normalize()
        start_date = end_date

    if end_date < start_date:
        raise ValueError("End date cannot be earlier than start date.")

    start_label = start_date.strftime("%Y%m%d")
    end_label = end_date.strftime("%Y%m%d")
    suffix = end_label if start_label == end_label else f"{start_label}_{end_label}"
    return start_date, end_date, suffix


def find_latest_prediction_file(predictions_dir: Path, player_id: int) -> Optional[Path]:
    """Find the latest V4 prediction file for a player."""
    # V4 predictions are in aggregated file, not per-player files
    # Look for predictions_lgbm_v4_*.csv
    pattern = "predictions_lgbm_v4_*.csv"
    matching_files = list(predictions_dir.glob(pattern))
    
    if not matching_files:
        return None
    
    # Extract date from filename and find the latest
    latest_file = None
    latest_date = None
    
    for file_path in matching_files:
        try:
            # Extract date from filename: predictions_lgbm_v4_YYYYMMDD.csv
            parts = file_path.stem.split('_')
            if len(parts) >= 4:
                date_str = parts[-1]  # Last part should be YYYYMMDD
                if len(date_str) == 8 and date_str.isdigit():
                    file_date = pd.to_datetime(date_str, format='%Y%m%d')
                    if latest_date is None or file_date > latest_date:
                        latest_date = file_date
                        latest_file = file_path
        except (ValueError, IndexError):
            continue
    
    return latest_file


def determine_players(predictions_dir: Path, suffix: str, specific_players: Optional[List[int]], model_version: str = "v4") -> List[int]:
    """Determine which players to process."""
    if specific_players:
        return specific_players
    
    # For V4, we read from aggregated predictions file
    # Find latest predictions file
    pattern = "predictions_lgbm_v4_*.csv"
    prediction_files = list(predictions_dir.glob(pattern))
    
    if not prediction_files:
        return []
    
    # Get latest file
    latest_file = max(prediction_files, key=lambda p: p.stat().st_mtime)
    
    try:
        df = pd.read_csv(latest_file, low_memory=False)
        if 'player_id' in df.columns:
            players = sorted(df['player_id'].unique().tolist())
            return players
    except Exception as e:
        print(f"[WARN] Could not read predictions file {latest_file}: {e}")
    
    return []


def create_player_dashboard(
    ctx: PlayerDashboardContext,
    pivot: pd.DataFrame,
    risk_df: pd.DataFrame,
    insights: dict,
    trend_metrics: dict,
    output_dir: Path,
    injury_periods: List[Tuple[pd.Timestamp, pd.Timestamp]] = None,
) -> Path:
    """Create dashboard for a single player (reuse V3 logic)."""
    # This function is complex - we'll reuse the V3 implementation
    # For now, create a simplified version
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Risk evolution
    ax1 = axes[0]
    dates = pd.to_datetime(pivot['reference_date'])
    probabilities = pivot['injury_probability'].values
    
    ax1.plot(dates, probabilities, 'b-', linewidth=2, label='Injury Probability')
    ax1.fill_between(dates, 0, probabilities, alpha=0.3, color='blue')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Injury Probability')
    ax1.set_title(f'{ctx.player_name} (ID: {ctx.player_id}) - Injury Risk Evolution')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Format x-axis dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 2: Risk levels
    ax2 = axes[1]
    risk_levels = pivot['risk_level'].value_counts()
    colors = [RISK_CLASS_COLORS_4LEVEL.get(level, 'gray') for level in risk_levels.index]
    ax2.bar(risk_levels.index, risk_levels.values, color=colors)
    ax2.set_xlabel('Risk Level')
    ax2.set_ylabel('Count')
    ax2.set_title('Risk Level Distribution')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_file = output_dir / ctx.chart_filename
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_file


def main() -> int:
    parser = argparse.ArgumentParser(description='Generate per-player dashboards for V4 predictions')
    parser.add_argument("--country", type=str, required=True, help="Country name (e.g., 'England')")
    parser.add_argument("--club", type=str, required=True, help="Club name (e.g., 'Chelsea FC')")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD) of the dashboard window")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD) of the dashboard window")
    parser.add_argument("--date", type=str, help="Fallback single date (YYYY-MM-DD) if --start-date is not provided")
    parser.add_argument("--players", type=int, nargs="*", help="Specific player IDs to process")
    args = parser.parse_args()

    # Get challenger club paths
    challenger_path = get_challenger_path(args.country, args.club)
    predictions_dir = challenger_path / "predictions"
    dashboards_dir = challenger_path / "dashboards" / "players"
    features_dir = challenger_path / "daily_features"
    
    start_date, end_date, suffix = parse_date_range(args)

    # Load insight models once
    insight_models_dir = PRODUCTION_ROOT / "models" / "insights"
    bodypart_pipeline = None
    severity_pipeline = None
    if (insight_models_dir / "bodypart_classifier.pkl").exists() and (insight_models_dir / "severity_classifier.pkl").exists():
        bodypart_pipeline, _ = load_pipeline(insight_models_dir / "bodypart_classifier")
        severity_pipeline, _ = load_pipeline(insight_models_dir / "severity_classifier")
    else:
        print(f"[WARN] Insight models not found at {insight_models_dir}, continuing without body part/severity predictions")

    # Find latest predictions file
    pattern = "predictions_lgbm_v4_*.csv"
    prediction_files = list(predictions_dir.glob(pattern))
    if not prediction_files:
        print(f"❌ No V4 prediction files found in {predictions_dir}")
        return 1
    
    latest_pred_file = max(prediction_files, key=lambda p: p.stat().st_mtime)
    print(f"[LOAD] Loading predictions from {latest_pred_file.name}...")
    
    try:
        predictions_df = pd.read_csv(latest_pred_file, low_memory=False)
        if 'reference_date' in predictions_df.columns:
            predictions_df['reference_date'] = pd.to_datetime(predictions_df['reference_date'], errors='coerce')
            predictions_df = predictions_df.dropna(subset=['reference_date'])
            # Filter by date range
            # If start_date == end_date (single date provided), show all predictions up to that date
            if start_date == end_date:
                predictions_df = predictions_df[
                    predictions_df['reference_date'] <= end_date
                ].copy()
            else:
                predictions_df = predictions_df[
                    (predictions_df['reference_date'] >= start_date) & 
                    (predictions_df['reference_date'] <= end_date)
                ].copy()
        print(f"[LOAD] Loaded {len(predictions_df):,} predictions")
    except Exception as e:
        print(f"❌ Error loading predictions: {e}")
        return 1

    # Determine players from filtered predictions (after date filtering)
    if args.players:
        players = args.players
    else:
        if 'player_id' in predictions_df.columns and len(predictions_df) > 0:
            players = sorted(predictions_df['player_id'].unique().tolist())
        else:
            print(f"❌ No players found in filtered predictions")
            return 1

    print("=" * 70)
    print("PLAYER DASHBOARDS FOR V4 PREDICTIONS")
    print("=" * 70)
    print(f"🌍 Country: {args.country}")
    print(f"🏆 Club: {args.club}")
    print(f"📅 Date range: {start_date.date()} to {end_date.date()}")
    print(f"🎯 Players: {len(players)}")
    print()

    dashboards_dir.mkdir(parents=True, exist_ok=True)
    
    successes = 0
    failures = 0
    
    for idx, player_id in enumerate(tqdm(players, desc="Generating dashboards", unit="player", disable=not HAS_TQDM), 1):
        try:
            if not HAS_TQDM:
                print(f"\n[{idx}/{len(players)}] Processing player {player_id}...")
            
            # Filter predictions for this player
            player_preds = predictions_df[predictions_df['player_id'] == player_id].copy()
            if player_preds.empty:
                if not HAS_TQDM:
                    print(f"   [SKIP] No predictions found for player {player_id}")
                continue
            
            player_name = player_preds.iloc[0].get('player_name', f'Player {player_id}')
            
            # Create context
            ctx = PlayerDashboardContext(
                player_id=player_id,
                player_name=player_name,
                window_start=start_date,
                window_end=end_date,
                suffix=suffix
            )
            
            # Prepare data for dashboard
            player_preds = player_preds.sort_values('reference_date')
            pivot = player_preds[['reference_date', 'injury_probability', 'risk_level']].copy()
            
            # Compute risk series and trend metrics
            # Pass Series, not .values (numpy array)
            risk_series = compute_risk_series(pivot['injury_probability'])
            trend_metrics = compute_trend_metrics(pivot['injury_probability'])
            
            # Create dashboard
            output_file = create_player_dashboard(
                ctx=ctx,
                pivot=pivot,
                risk_df=player_preds,
                insights={},
                trend_metrics=trend_metrics,
                output_dir=dashboards_dir,
                injury_periods=[]
            )
            
            successes += 1
            if not HAS_TQDM:
                print(f"   [OK] Generated dashboard: {output_file.name}")
        
        except Exception as e:
            failures += 1
            if not HAS_TQDM:
                print(f"   [ERROR] Failed to generate dashboard for player {player_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*70}")
    print(f"✅ Success: {successes}")
    if failures > 0:
        print(f"❌ Failures: {failures}")
    print(f"{'='*70}")
    
    return 0 if failures == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
