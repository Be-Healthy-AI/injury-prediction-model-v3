#!/usr/bin/env python3
"""
Generate per-player dashboards for production predictions.

Reuses the retrospective dashboard visual style (risk evolution + body parts +
severity) to create one PNG per player over a specified date range.

Adapted for production structure with country/club organization.
"""

from __future__ import annotations

import argparse
import glob
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


def get_club_path(country: str, club: str) -> Path:
    """Get the base path for a specific club deployment."""
    return PRODUCTION_ROOT / "deployments" / country / club

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
        return f"{self.entry_id}_probabilities.png"
    
    def chart_filename_with_version(self, model_version: str = "v2") -> str:
        """Get chart filename with model version suffix."""
        if model_version == "v3":
            return f"{self.entry_id}_v3_probabilities.png"
        return f"{self.entry_id}_probabilities.png"


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


def find_latest_prediction_file(predictions_dir: Path, player_id: int, model_version: str = "v2") -> Optional[Path]:
    """Find the latest prediction file for a player by date suffix."""
    players_dir = predictions_dir / "players"
    if not players_dir.exists():
        return None
    
    # Find all prediction files for this player matching the model version
    if model_version == "v3":
        pattern = f"player_{player_id}_predictions_v3_*.csv"
    else:
        pattern = f"player_{player_id}_predictions_*.csv"
        # Exclude V3 files when looking for V2
        exclude_v3_pattern = f"player_{player_id}_predictions_v3_*.csv"
    
    matching_files = list(players_dir.glob(pattern))
    
    # If looking for V2, exclude V3 files
    if model_version == "v2":
        v3_files = set(players_dir.glob(exclude_v3_pattern))
        matching_files = [f for f in matching_files if f not in v3_files]
    
    if not matching_files:
        return None
    
    # Extract date from filename and find the latest
    latest_file = None
    latest_date = None
    
    for file_path in matching_files:
        # Extract date suffix from filename
        # V2: player_ID_predictions_YYYYMMDD.csv
        # V3: player_ID_predictions_v3_YYYYMMDD.csv
        try:
            parts = file_path.stem.split('_')
            # For V3, date is in parts[-1], for V2 it's also in parts[-1]
            if model_version == "v3":
                # V3 format: player_ID_predictions_v3_YYYYMMDD
                if len(parts) >= 5 and parts[-2] == "v3" and parts[-1].isdigit() and len(parts[-1]) == 8:
                    date_str = parts[-1]
                    file_date = pd.to_datetime(date_str, format='%Y%m%d')
                    if latest_date is None or file_date > latest_date:
                        latest_date = file_date
                        latest_file = file_path
            else:
                # V2 format: player_ID_predictions_YYYYMMDD
                if len(parts) >= 4 and parts[-1].isdigit() and len(parts[-1]) == 8:
                    date_str = parts[-1]
                    file_date = pd.to_datetime(date_str, format='%Y%m%d')
                    if latest_date is None or file_date > latest_date:
                        latest_date = file_date
                        latest_file = file_path
        except (ValueError, IndexError):
            continue
    
    return latest_file


def load_prediction_file(predictions_dir: Path, player_id: int, suffix: str, model_version: str = "v2") -> pd.DataFrame:
    """Load predictions file from players directory.
    
    First tries to load with the exact suffix, then falls back to finding the latest file.
    Supports both V2 and V3 model versions.
    """
    # Try exact match first
    if model_version == "v3":
        file_path = predictions_dir / "players" / f"player_{player_id}_predictions_v3_{suffix}.csv"
    else:
        file_path = predictions_dir / "players" / f"player_{player_id}_predictions_{suffix}.csv"
    
    if not file_path.exists():
        # Fallback: find the latest prediction file for this player
        print(f"[WARN] Prediction file with suffix '{suffix}' not found for player {player_id}, trying to find latest file...")
        file_path = find_latest_prediction_file(predictions_dir, player_id, model_version)
        if file_path is None:
            raise FileNotFoundError(f"Predictions file not found for player {player_id} (tried suffix '{suffix}' and latest file)")
        print(f"[INFO] Using latest prediction file: {file_path.name}")
    
    df = pd.read_csv(file_path, parse_dates=["reference_date"])
    if df.empty:
        raise ValueError(f"Predictions file {file_path} is empty")
    # Ensure we have the required probability column
    if "injury_probability" not in df.columns:
        raise ValueError(f"Missing required probability column: injury_probability")
    return df.sort_values("reference_date")


def load_latest_feature_row(player_id: int, features_dir: Path, end_date: pd.Timestamp) -> pd.Series:
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
        # An injury overlaps if: fromDate <= end_date AND (untilDate >= start_date OR untilDate is NaN)
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
        import traceback
        traceback.print_exc()
        return []


def create_player_dashboard(
    ctx: PlayerDashboardContext,
    pivot: pd.DataFrame,
    risk_df: pd.DataFrame,
    insights: dict,
    trend_metrics: dict,
    output_dir: Path,
    injury_periods: List[Tuple[pd.Timestamp, pd.Timestamp, str]] = None,
    player_profile: Optional[pd.Series] = None,
    model_version: str = "v2",
) -> Path:
    """Create dashboard PNG showing predictions with 4-level risk classification.
    
    Supports both V2 and V3 models. Model version is included in the filename.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.0, 1.0], hspace=0.5, wspace=0.45)

    # Top panel – risk evolution (full width)
    model_label = "lgbm_muscular_v3" if model_version == "v3" else "lgbm_muscular_v2"
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
    if player_profile is not None:
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

    # Use model version in filename
    chart_filename = ctx.chart_filename_with_version(model_version)
    chart_path = output_dir / chart_filename
    plt.savefig(chart_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return chart_path


def determine_players(predictions_dir: Path, suffix: str, explicit_players: Iterable[int] | None, model_version: str = "v2") -> List[int]:
    """Determine which players to process.
    
    If explicit_players is provided, use those. Otherwise, try to find players
    with the exact suffix, and if none found, find all players with any prediction files.
    """
    if explicit_players:
        return list(explicit_players)
    
    # First try exact suffix match based on model version
    if model_version == "v3":
        pattern = predictions_dir / "players" / f"player_*_predictions_v3_{suffix}.csv"
    else:
        pattern = predictions_dir / "players" / f"player_*_predictions_{suffix}.csv"
    
    players = []
    for path in pattern.parent.glob(pattern.name):
        try:
            pid = int(path.stem.split("_")[1])
            players.append(pid)
        except ValueError:
            continue
    
    # If no exact matches, find all players with any prediction files for this model version
    if not players:
        print(f"[WARN] No prediction files found with suffix '{suffix}', searching for all prediction files...")
        if model_version == "v3":
            all_pattern = predictions_dir / "players" / "player_*_predictions_v3_*.csv"
            v3_files = set()  # Not needed for V3
        else:
            all_pattern = predictions_dir / "players" / "player_*_predictions_*.csv"
            # Exclude V3 files when looking for V2
            exclude_v3_pattern = predictions_dir / "players" / "player_*_predictions_v3_*.csv"
            v3_files = set(exclude_v3_pattern.parent.glob(exclude_v3_pattern.name))
        
        for path in all_pattern.parent.glob(all_pattern.name):
            # Skip V3 files when looking for V2
            if model_version == "v2" and path in v3_files:
                continue
            try:
                pid = int(path.stem.split("_")[1])
                players.append(pid)
            except ValueError:
                continue
        if players:
            print(f"[INFO] Found {len(set(players))} players with prediction files")
    
    return sorted(set(players))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", type=str, required=True, help="Country name (e.g., 'England')")
    parser.add_argument("--club", type=str, required=True, help="Club name (e.g., 'Chelsea FC')")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD) of the dashboard window")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD) of the dashboard window")
    parser.add_argument(
        "--date",
        type=str,
        help="Fallback single date (YYYY-MM-DD) if --start-date is not provided",
    )
    parser.add_argument("--players", type=int, nargs="*", help="Specific player IDs to process")
    parser.add_argument(
        "--model-version",
        type=str,
        choices=["v2", "v3"],
        default="v2",
        help="Model version to use for predictions (default: v2)"
    )
    args = parser.parse_args()

    # Get club paths
    club_path = get_club_path(args.country, args.club)
    predictions_dir = club_path / "predictions"
    dashboards_dir = club_path / "dashboards" / "players"
    features_dir = club_path / "daily_features"
    
    start_date, end_date, suffix = parse_date_range(args)

    players = determine_players(predictions_dir, suffix, args.players, args.model_version)
    if not players:
        print(f"❌ No prediction files found for suffix '{suffix}'.")
        return 1

    print("=" * 70)
    print("PLAYER DASHBOARDS FOR PRODUCTION PREDICTIONS")
    print("=" * 70)
    print(f"🌍 Country: {args.country}")
    print(f" club: {args.club}")
    print(f"📅 Date range: {start_date.date()} to {end_date.date()}")
    print(f"🎯 Players: {len(players)}")
    print()

    # Load insight models once
    insight_models_dir = PRODUCTION_ROOT / "models" / "insights"
    bodypart_pipeline = None
    severity_pipeline = None
    if (insight_models_dir / "bodypart_classifier.pkl").exists() and (insight_models_dir / "severity_classifier.pkl").exists():
        bodypart_pipeline, _ = load_pipeline(insight_models_dir / "bodypart_classifier")
        severity_pipeline, _ = load_pipeline(insight_models_dir / "severity_classifier")
    else:
        print(f"[WARN] Insight models not found at {insight_models_dir}, continuing without body part/severity predictions")

    successes = 0
    failures = 0
    
    for idx, player_id in enumerate(tqdm(players, desc="Generating dashboards", unit="player", disable=not HAS_TQDM), 1):
        try:
            if not HAS_TQDM:
                print(f"\n[{idx}/{len(players)}] Processing player {player_id}...")
                sys.stdout.flush()
            
            predictions = load_prediction_file(predictions_dir, player_id, suffix, args.model_version)
            player_name = predictions["player_name"].iloc[0] if "player_name" in predictions.columns else f"Player {player_id}"
            
            # If only --date was provided (start_date == end_date), use all predictions up to that date
            # This ensures the dashboard shows the full risk evolution over time, not just one day
            if start_date == end_date and args.date:
                # Use all predictions from the beginning up to the specified date
                predictions_window = predictions[
                    predictions["reference_date"] <= end_date
                ].copy()
            else:
                predictions_window = predictions[
                    (predictions["reference_date"] >= start_date)
                    & (predictions["reference_date"] <= end_date)
                ].copy()
            
            if predictions_window.empty:
                predictions_window = predictions.copy()
            
            # Extract player profile from first row (all rows have same profile data)
            player_profile = predictions.iloc[0] if len(predictions) > 0 else None
            
            # Use actual prediction date range for injury periods, not window dates
            # Filter out NaT dates before calculating min/max
            valid_dates = predictions["reference_date"].dropna()
            if len(valid_dates) == 0:
                print(f"[WARN] No valid dates found for player {player_id}, skipping...")
                failures += 1
                continue
            
            actual_start = valid_dates.min()
            actual_end = valid_dates.max()
            
            # Ensure actual_start and actual_end are not NaT
            if pd.isna(actual_start) or pd.isna(actual_end):
                print(f"[WARN] Invalid date range for player {player_id}, skipping...")
                failures += 1
                continue
            
            injury_periods = load_injury_periods(player_id, features_dir, actual_start, actual_end, country=args.country)
            
            feature_row = load_latest_feature_row(player_id, features_dir, end_date)
            insights = {}
            if bodypart_pipeline is not None and severity_pipeline is not None:
                insights = predict_insights(feature_row, bodypart_pipeline, severity_pipeline)
            
            # Use 4-level risk classification
            risk_df = None  # Not used for 4-level, but kept for compatibility
            trend_metrics = compute_trend_metrics(predictions_window["injury_probability"])
            ctx = PlayerDashboardContext(
                player_id=player_id,
                player_name=player_name,
                window_start=start_date,
                window_end=end_date,
                suffix=suffix,
            )
            
            # Create a dummy risk_df for compatibility (not used in 4-level dashboard)
            risk_df_dummy = pd.DataFrame({
                "risk_index": [classify_risk_4level(p)["index"] for p in predictions_window["injury_probability"]],
                "risk_label": [classify_risk_4level(p)["label"] for p in predictions_window["injury_probability"]],
            })
            create_player_dashboard(ctx, predictions_window, risk_df_dummy, insights, trend_metrics, dashboards_dir, injury_periods, player_profile=player_profile, model_version=args.model_version)
            chart_filename = ctx.chart_filename_with_version(args.model_version)
            if not HAS_TQDM:
                print(f"   ✅ Dashboard generated: {chart_filename}")
            else:
                tqdm.write(f"✅ Player {player_id}: {chart_filename}")
            sys.stdout.flush()
            successes += 1
        except Exception as exc:
            error_msg = f"❌ Player {player_id}: {exc}"
            if HAS_TQDM:
                tqdm.write(error_msg)
            else:
                print(error_msg)
            sys.stdout.flush()
            failures += 1

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✅ Successful dashboards: {successes}")
    print(f"❌ Failed dashboards: {failures}")
    sys.stdout.flush()

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())



