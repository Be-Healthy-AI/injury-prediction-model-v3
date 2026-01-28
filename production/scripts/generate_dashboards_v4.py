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
    reference_date: pd.Timestamp  # The calculation reference date (YYYYMMDD format for filename)

    @property
    def chart_filename(self) -> str:
        """Generate filename: player_{player_id}_{reference_date_YYYYMMDD}_v4_probabilities.png"""
        ref_date_str = self.reference_date.strftime("%Y%m%d")
        return f"player_{self.player_id}_{ref_date_str}_v4_probabilities.png"


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


def load_latest_feature_row(player_id: int, features_dir: Path, end_date: pd.Timestamp) -> pd.Series:
    """Load the latest feature row for a player up to end_date."""
    file_path = features_dir / f"player_{player_id}_daily_features.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Daily features not found: {file_path}")
    df = pd.read_csv(file_path, parse_dates=["date"])
    df = df[df["date"] <= end_date].sort_values("date")
    if df.empty:
        raise ValueError(f"No feature rows available up to {end_date.date()} for player {player_id}")
    return df.iloc[-1]


def load_injury_periods(
    player_id: int, 
    features_dir: Path, 
    start_date: pd.Timestamp, 
    end_date: pd.Timestamp,
    country: str = "england"
) -> List[Tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Load injury periods for a player from injuries_data.csv, including injury_class."""
    # Get latest raw data folder
    latest_raw_data = get_latest_raw_data_folder(country.lower().replace(" ", "_"))
    if latest_raw_data is None:
        return []
    
    injuries_file = latest_raw_data / "injuries_data.csv"
    if not injuries_file.exists():
        return []
    
    try:
        # Load injuries data
        df = pd.read_csv(injuries_file, sep=';', encoding='utf-8-sig', low_memory=False)
        
        # Parse dates explicitly to handle various formats
        df['fromDate'] = pd.to_datetime(df['fromDate'], errors='coerce')
        df['untilDate'] = pd.to_datetime(df['untilDate'], errors='coerce')
        
        # Filter for this player
        player_injuries = df[df['player_id'] == player_id].copy()
        if player_injuries.empty:
            return []
        
        # Filter injuries that overlap with the date range
        injury_periods = []
        for _, row in player_injuries.iterrows():
            injury_start = pd.to_datetime(row['fromDate'])
            if pd.isna(injury_start):
                continue  # Skip if start date is invalid
            
            injury_start = injury_start.normalize()
            injury_end = row.get('untilDate')
            
            if pd.notna(injury_end):
                injury_end = pd.to_datetime(injury_end).normalize()
            else:
                # If no end date, use the end_date as the injury end (ongoing injury)
                injury_end = end_date
            
            # Get injury_class (default to 'unknown' if missing)
            injury_class = str(row.get('injury_class', 'unknown')).lower().strip()
            
            # Check if injury overlaps with the date range
            if injury_start <= end_date and injury_end >= start_date:
                # Clip to the actual date range
                period_start = max(injury_start, start_date)
                period_end = min(injury_end, end_date)
                injury_periods.append((period_start, period_end, injury_class))
        
        return sorted(injury_periods, key=lambda x: x[0])
    except Exception as e:
        print(f"[WARN] Could not load injury periods for player {player_id}: {e}")
        return []


def load_player_profile(player_id: int, country: str = "england") -> pd.Series:
    """Load player profile from latest raw data folder."""
    latest_raw_data = get_latest_raw_data_folder(country.lower().replace(" ", "_"))
    if latest_raw_data is None or not latest_raw_data.exists():
        return pd.Series()
    
    profile_file = latest_raw_data / "players_profile.csv"
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


def create_player_dashboard(
    ctx: PlayerDashboardContext,
    pivot: pd.DataFrame,
    risk_df: pd.DataFrame,
    insights: dict,
    trend_metrics: dict,
    output_dir: Path,
    injury_periods: List[Tuple[pd.Timestamp, pd.Timestamp, str]] = None,
    player_profile: Optional[pd.Series] = None,
) -> Path:
    """Create dashboard PNG showing predictions with 4-level risk classification for V4."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.0, 1.0], hspace=0.5, wspace=0.45)

    # Top panel – risk evolution (full width)
    ax_main = fig.add_subplot(gs[0, :])
    dates = pivot["reference_date"]
    
    # Get actual date range from predictions
    actual_start = dates.min()
    actual_end = dates.max()
    
    # Add background shading for injury periods with different colors for muscular vs non-muscular
    if injury_periods:
        first_muscular_label = True
        first_non_muscular_label = True
        for period_start, period_end, injury_class in injury_periods:
            chart_start = dates.min()
            chart_end = dates.max()
            if period_start <= chart_end and period_end >= chart_start:
                shade_start = max(period_start, chart_start)
                shade_end = min(period_end, chart_end)
                
                # Determine color based on injury_class
                is_muscular = injury_class == 'muscular'
                if is_muscular:
                    color = '#AA5555'  # Darker red for muscular injuries
                    label = 'Muscular Injury' if first_muscular_label else ''
                    first_muscular_label = False
                else:
                    color = '#FF9999'  # Light red for non-muscular injuries
                    label = 'Non-Muscular Injury' if first_non_muscular_label else ''
                    first_non_muscular_label = False
                
                ax_main.axvspan(shade_start, shade_end, alpha=0.15, color=color, zorder=0, label=label)
    
    # Compute probabilities as percentages (keep risk labels for color coding only)
    probabilities = pivot["injury_probability"]
    probabilities_percent = probabilities * 100  # Convert to percentages
    risk_labels_4level = []
    for prob in probabilities:
        risk_info = classify_risk_4level(prob)
        risk_labels_4level.append(risk_info["label"])

    if HAS_SCIPY and len(dates) > 3:
        # Smooth curves using spline interpolation
        dates_numeric = np.arange(len(dates))
        x_smooth = np.linspace(dates_numeric.min(), dates_numeric.max(), 500)
        dates_smooth = pd.date_range(dates.min(), dates.max(), periods=500)
        
        # Plot with risk-based coloring (using probabilities as percentages)
        spl = make_interp_spline(dates_numeric, probabilities_percent, k=min(3, len(dates) - 1))
        y_smooth = spl(x_smooth)
        
        # Determine colors based on risk classification of smoothed values
        colors_smooth = []
        for val in y_smooth:
            # Convert percentage back to probability for classification
            prob_val = val / 100.0
            risk_info = classify_risk_4level(prob_val)
            label = risk_info["label"]
            colors_smooth.append(RISK_CLASS_COLORS_4LEVEL[label])
        
        for i in range(len(dates_smooth) - 1):
            ax_main.plot(
                dates_smooth[i : i + 2],
                y_smooth[i : i + 2],
                color=colors_smooth[i],
                linewidth=2.5,
                alpha=0.9,
                zorder=3,
                label="Injury Risk" if i == 0 else "",
            )
        
        ax_main.fill_between(dates_smooth, y_smooth, alpha=0.15, color="gray", zorder=1)
    else:
        # For very few points, plot simple lines
        colors = [RISK_CLASS_COLORS_4LEVEL[label] for label in risk_labels_4level]
        for i in range(len(dates) - 1):
            ax_main.plot(
                dates.iloc[i : i + 2],
                probabilities_percent.iloc[i : i + 2],
                color=colors[i],
                linewidth=2.5,
                alpha=0.9,
                zorder=3,
                label="Injury Risk" if i == 0 else "",
            )
        ax_main.fill_between(dates, probabilities_percent, alpha=0.15, color="gray", zorder=1)

    ax_main.set_title(f"Injury Risk Evolution - {ctx.player_name}", fontsize=14, fontweight="bold", pad=15)
    ax_main.set_ylabel("Injury Probability (%)", fontweight="bold")
    ax_main.set_xlabel("Date", fontweight="bold")
    # Use percentage ticks instead of risk labels
    ax_main.set_yticks([0, 25, 50, 75, 100])
    ax_main.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax_main.set_ylim(0, 100)
    ax_main.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax_main.grid(axis="y", alpha=0.3, linestyle="--")
    
    # Make time axis more granular - YYYY-MM-DD format, only show days
    ax_main.xaxis.set_major_locator(mdates.DayLocator(interval=7))  # Show every week
    ax_main.xaxis.set_minor_locator(mdates.DayLocator(interval=1))  # Minor ticks every day
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # YYYY-MM-DD format
    ax_main.xaxis.set_minor_formatter(mdates.DateFormatter(''))  # No labels on minor ticks
    plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=6)

    # Bottom left – body parts
    ax_body = fig.add_subplot(gs[1, 0])
    if insights and insights.get("bodypart_rank"):
        body_rank = insights.get("bodypart_rank", [])[:3]
        labels = [item[0].replace("_", " ").title() for item in body_rank]
        probs = [item[1] * 100 for item in body_rank]
        colors_body = ["#2E86AB", "#A23B72", "#F18F01"][: len(labels)]
        bars = ax_body.barh(labels, probs, color=colors_body, alpha=0.8, edgecolor="white", linewidth=0.5, height=0.6)
        ax_body.set_xlabel("Probability (%)", fontweight="bold", fontsize=8)
        ax_body.set_title("Body Parts at Risk", fontweight="bold", pad=5, fontsize=9)
        ax_body.set_xlim(0, 100)
        ax_body.tick_params(labelsize=7)
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            ax_body.text(prob + 2, i, f"{prob:.1f}%", va="center", fontweight="bold", fontsize=7)
        ax_body.grid(axis="x", alpha=0.3, linestyle="--")
    else:
        ax_body.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_body.transAxes, fontsize=8)
        ax_body.set_title("Body Parts at Risk", fontweight="bold", fontsize=9)

    # Bottom center – severity
    ax_sev = fig.add_subplot(gs[1, 1])
    if insights and insights.get("severity_probs"):
        severity_probs = insights.get("severity_probs", {})
        severity_label = insights.get("severity_label", "Unknown")
        labels_sev = list(severity_probs.keys())
        sizes = [severity_probs[k] * 100 for k in labels_sev]
        colors_sev = ["#06A77D", "#F4A261", "#E76F51", "#264653"][: len(labels_sev)]
        bars_sev = ax_sev.barh(
            range(len(labels_sev)), sizes, color=colors_sev, alpha=0.8, edgecolor="white", linewidth=0.5, height=0.6
        )
        ax_sev.set_yticks(range(len(labels_sev)))
        formatted_labels = []
        for l in labels_sev:
            formatted = l.replace("_", " ").title()
            if formatted == "Long Term":
                formatted = "Long term"
            formatted_labels.append(formatted)
        ax_sev.set_yticklabels(formatted_labels, fontsize=7)
        ax_sev.set_xlabel("Probability (%)", fontweight="bold", fontsize=8)
        formatted_severity_label = severity_label.replace("_", " ").title()
        if formatted_severity_label == "Long Term":
            formatted_severity_label = "Long term"
        ax_sev.set_title(f"Severity: {formatted_severity_label}", fontweight="bold", pad=5, fontsize=9)
        ax_sev.set_xlim(0, 100)
        ax_sev.tick_params(labelsize=7)
        for i, (bar, size) in enumerate(zip(bars_sev, sizes)):
            ax_sev.text(size + 2, i, f"{size:.1f}%", va="center", fontweight="bold", fontsize=7)
        ax_sev.grid(axis="x", alpha=0.3, linestyle="--")
    else:
        ax_sev.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_sev.transAxes, fontsize=8)
        ax_sev.set_title("Severity", fontweight="bold", fontsize=9)
    
    # Add footnote at the bottom of the page
    fig.text(0.5, 0.02, "Note: Body part and severity probabilities refer to the last day shown in the main chart.", 
             ha='center', fontsize=7, style='italic', color='gray')

    # Bottom right – player profile and risk summary
    ax_alert = fig.add_subplot(gs[1, 2])
    ax_alert.axis("off")
    
    # Player profile section (top part)
    profile_text = "PLAYER PROFILE\n\n"
    if player_profile is not None and not player_profile.empty:
        position = player_profile.get('position', 'N/A')
        date_of_birth = player_profile.get('date_of_birth', '')
        nationality1 = player_profile.get('nationality1', '')
        nationality2 = player_profile.get('nationality2', '')
        
        # Calculate age if date_of_birth is available
        age = "N/A"
        if date_of_birth and pd.notna(date_of_birth):
            try:
                dob = pd.to_datetime(date_of_birth)
                age = str((actual_end - dob).days // 365)
            except:
                pass
        
        # Format nationality
        nationality = nationality1 if pd.notna(nationality1) and str(nationality1).strip() else 'N/A'
        if nationality2 and pd.notna(nationality2) and str(nationality2).strip():
            nationality += f" / {nationality2}"
        
        profile_text += f"Position: {position}\n"
        profile_text += f"Age: {age}\n"
        profile_text += f"Nationality: {nationality}\n"
    else:
        profile_text += "Position: N/A\n"
        profile_text += "Age: N/A\n"
        profile_text += "Nationality: N/A\n"
    
    profile_text += "\n" + "=" * 20 + "\n\n"
    
    # Risk summary section (bottom part)
    if risk_labels_4level:
        peak_risk = max(risk_labels_4level, key=lambda x: RISK_CLASS_LABELS_4LEVEL.index(x))
        final_risk = risk_labels_4level[-1]
    else:
        peak_risk = "N/A"
        final_risk = "N/A"

    sustained = trend_metrics.get("sustained_elevated_days", 0) if trend_metrics else 0
    max_jump = trend_metrics.get("max_jump", 0.0) if trend_metrics else 0.0
    slope = trend_metrics.get("slope", 0.0) if trend_metrics else 0.0

    profile_text += f"""RISK SUMMARY

Peak Risk Level: {peak_risk}
Final Risk Level: {final_risk}
Sustained Elevated: {sustained} days
Max Daily Jump: {max_jump:.3f}
Trend Slope: {slope:.4f}"""
    
    ax_alert.text(
        0.1,
        0.5,
        profile_text,
        transform=ax_alert.transAxes,
        fontsize=9,
        verticalalignment="center",
        family="monospace",
        fontweight="bold",
    )

    # Use actual date range from predictions instead of window dates
    plt.suptitle(
        f"Observation Period: {actual_start.strftime('%Y-%m-%d')} to {actual_end.strftime('%Y-%m-%d')}",
        fontsize=11,
        y=0.98,
    )

    # Use V4 filename format
    chart_path = output_dir / ctx.chart_filename
    plt.savefig(chart_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return chart_path


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
    
    start_date, end_date, _ = parse_date_range(args)  # suffix not used for V4, we use reference_date from predictions

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
            
            # Get the maximum reference date from predictions (the calculation reference date)
            max_ref_date = predictions_df['reference_date'].max()
            print(f"[INFO] Predictions file contains data up to: {max_ref_date.date()}")
            
            # If start_date == end_date (single date provided), use all predictions up to the calculation reference date
            # This ensures dashboards show all available predictions, not just up to the provided date
            if start_date == end_date:
                # Use the maximum reference date from predictions (calculation reference date)
                actual_end_date = max_ref_date.normalize()
                print(f"[INFO] Using calculation reference date as end date: {actual_end_date.date()}")
                predictions_df = predictions_df[
                    predictions_df['reference_date'] <= actual_end_date
                ].copy()
                # Update end_date for display purposes
                end_date = actual_end_date
            else:
                # For date ranges, still respect the provided end_date, but don't go beyond max_ref_date
                actual_end_date = min(end_date, max_ref_date.normalize())
                if actual_end_date < end_date:
                    print(f"[INFO] Clamping end_date from {end_date.date()} to calculation reference date: {actual_end_date.date()}")
                predictions_df = predictions_df[
                    (predictions_df['reference_date'] >= start_date) & 
                    (predictions_df['reference_date'] <= actual_end_date)
                ].copy()
                end_date = actual_end_date
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
            
            # Sort predictions by reference_date
            player_preds = player_preds.sort_values('reference_date')
            
            # Get the maximum reference_date (calculation reference date) for filename
            max_ref_date = player_preds['reference_date'].max()
            
            # Use actual prediction date range for injury periods
            valid_dates = player_preds["reference_date"].dropna()
            if len(valid_dates) == 0:
                if not HAS_TQDM:
                    print(f"   [WARN] No valid dates found for player {player_id}, skipping...")
                failures += 1
                continue
            
            actual_start = valid_dates.min()
            actual_end = valid_dates.max()
            
            # Ensure actual_start and actual_end are not NaT
            if pd.isna(actual_start) or pd.isna(actual_end):
                if not HAS_TQDM:
                    print(f"   [WARN] Invalid date range for player {player_id}, skipping...")
                failures += 1
                continue
            
            # Load injury periods
            injury_periods = load_injury_periods(player_id, features_dir, actual_start, actual_end, country=args.country)
            
            # Load latest feature row for insights
            try:
                feature_row = load_latest_feature_row(player_id, features_dir, end_date)
                insights = {}
                if bodypart_pipeline is not None and severity_pipeline is not None:
                    insights = predict_insights(feature_row, bodypart_pipeline, severity_pipeline)
            except FileNotFoundError:
                if not HAS_TQDM:
                    print(f"   [WARN] Daily features not found for player {player_id}, continuing without insights")
                insights = {}
            except Exception as e:
                if not HAS_TQDM:
                    print(f"   [WARN] Could not load features for player {player_id}: {e}")
                insights = {}
            
            # Load player profile
            player_profile = load_player_profile(player_id, country=args.country)
            
            # Compute trend metrics
            trend_metrics = compute_trend_metrics(player_preds['injury_probability'])
            
            # Create context with reference_date for filename
            ctx = PlayerDashboardContext(
                player_id=player_id,
                player_name=player_name,
                window_start=start_date,
                window_end=end_date,
                reference_date=max_ref_date  # Use max reference_date for filename
            )
            
            # Prepare pivot data for dashboard
            pivot = player_preds[['reference_date', 'injury_probability', 'risk_level']].copy()
            
            # Create dummy risk_df for compatibility (not used in 4-level dashboard)
            risk_df_dummy = pd.DataFrame({
                "risk_index": [classify_risk_4level(p)["index"] for p in player_preds["injury_probability"]],
                "risk_label": [classify_risk_4level(p)["label"] for p in player_preds["injury_probability"]],
            })
            
            # Create dashboard
            output_file = create_player_dashboard(
                ctx=ctx,
                pivot=pivot,
                risk_df=risk_df_dummy,
                insights=insights,
                trend_metrics=trend_metrics,
                output_dir=dashboards_dir,
                injury_periods=injury_periods,
                player_profile=player_profile
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
