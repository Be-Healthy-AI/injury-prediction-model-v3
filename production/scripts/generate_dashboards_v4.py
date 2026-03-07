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
from typing import Dict, Iterable, List, Optional, Tuple

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


# V4 4-model: column names and display labels (fallback to single injury_probability for old CSV)
V4_PROB_COLUMNS = [
    "injury_probability_muscular_lgbm",
    "injury_probability_muscular_gb",
    "injury_probability_skeletal",
    "injury_probability_msu_lgbm",
]
V4_PROB_LABELS = {
    "injury_probability_muscular_lgbm": "Muscular (LGBM)",
    "injury_probability_muscular_gb": "Muscular (GB)",
    "injury_probability_skeletal": "Skeletal",
    "injury_probability_msu_lgbm": "MSU (LGBM)",
}
PRIMARY_PROB_COL = "injury_probability_muscular_lgbm"

# Combined probability weights for muscular and skeletal dashboards
# Default formulas:
#   - Muscular combined = 0.2 * MSU LGBM + 0.8 * muscular LGBM + 0.0 * muscular GB
#   - Skeletal combined = 0.2 * MSU LGBM + 0.8 * skeletal LGBM
MUSCULAR_WEIGHTS = {
    "injury_probability_msu_lgbm": 0.2,
    "injury_probability_muscular_lgbm": 0.8,
    "injury_probability_muscular_gb": 0.0,
}
SKELETAL_WEIGHTS = {
    "injury_probability_msu_lgbm": 0.2,
    "injury_probability_skeletal": 0.8,
}


def compute_combined_probability(pivot: pd.DataFrame, weights: dict) -> pd.Series:
    """Compute weighted sum of probability columns. Missing columns treated as 0."""
    out = pd.Series(0.0, index=pivot.index)
    for col, w in weights.items():
        if col in pivot.columns:
            out = out + pivot[col].fillna(0).astype(float) * w
    return out


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
        """Legacy: player_{player_id}_{reference_date_YYYYMMDD}_v4_probabilities.png"""
        ref_date_str = self.reference_date.strftime("%Y%m%d")
        return f"player_{self.player_id}_{ref_date_str}_v4_probabilities.png"

    @property
    def chart_filename_index(self) -> str:
        """Legacy: player_{player_id}_{reference_date_YYYYMMDD}_v4_index.png"""
        ref_date_str = self.reference_date.strftime("%Y%m%d")
        return f"player_{self.player_id}_{ref_date_str}_v4_index.png"

    def chart_filename_variant(self, variant: str, chart_type: str) -> str:
        """Variant = muscular|skeletal, chart_type = prob|index."""
        ref_date_str = self.reference_date.strftime("%Y%m%d")
        return f"player_{self.player_id}_{ref_date_str}_v4_{variant}_{chart_type}.png"


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


def _injury_period_color_and_label(injury_class: str) -> Tuple[str, str]:
    """Return (color_hex, legend_label) for injury period shading.
    4-level: a) skeletal (dark red), b) muscular (red), c) unknown (light red), d) other (light yellow).
    """
    ic = str(injury_class).lower().strip()
    if ic == "skeletal":
        return "#6B0000", "Skeletal Injury"
    if ic == "muscular":
        return "#AA3333", "Muscular Injury"
    if ic == "unknown":
        return "#FF9999", "Unknown Injury"
    return "#FFF9C4", "Other Injury"


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
    
    # Add background shading for injury periods: skeletal (dark red), muscular (red), unknown (light red), other (light yellow)
    if injury_periods:
        shown_labels = set()
        for period_start, period_end, injury_class in injury_periods:
            chart_start = dates.min()
            chart_end = dates.max()
            if period_start <= chart_end and period_end >= chart_start:
                shade_start = max(period_start, chart_start)
                shade_end = min(period_end, chart_end)
                color, label = _injury_period_color_and_label(injury_class)
                show_label = label if label not in shown_labels else ""
                if label not in shown_labels:
                    shown_labels.add(label)
                ax_main.axvspan(shade_start, shade_end, alpha=0.15, color=color, zorder=0, label=show_label)
    
    # Which probability column(s) to plot: 3-model or legacy single
    prob_cols = [c for c in V4_PROB_COLUMNS if c in pivot.columns]
    if not prob_cols and "injury_probability" in pivot.columns:
        prob_cols = ["injury_probability"]
    if not prob_cols and len(pivot.columns) >= 2:
        prob_cols = [pivot.columns[1]]
    primary_col = PRIMARY_PROB_COL if PRIMARY_PROB_COL in pivot.columns else (prob_cols[0] if prob_cols else "injury_probability")
    probabilities = pivot[primary_col] if primary_col in pivot.columns else (pivot[prob_cols[0]] if prob_cols else pd.Series(dtype=float))
    risk_labels_4level = [classify_risk_4level(p)["label"] for p in probabilities]

    # Colors for 3-model lines (distinct)
    SERIES_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]

    if HAS_SCIPY and len(dates) > 3:
        dates_numeric = np.arange(len(dates))
        x_smooth = np.linspace(dates_numeric.min(), dates_numeric.max(), 500)
        dates_smooth = pd.date_range(dates.min(), dates.max(), periods=500)
        for series_idx, col in enumerate(prob_cols):
            prob_series = pivot[col] * 100
            spl = make_interp_spline(dates_numeric, prob_series.values, k=min(3, len(dates) - 1))
            y_smooth = spl(x_smooth)
            color = SERIES_COLORS[series_idx % len(SERIES_COLORS)]
            label = V4_PROB_LABELS.get(col, col)
            ax_main.plot(dates_smooth, y_smooth, color=color, linewidth=2.0, alpha=0.9, zorder=3, label=label)
        ax_main.fill_between(dates_smooth, 0, 100, alpha=0.05, color="gray", zorder=1)
    else:
        for series_idx, col in enumerate(prob_cols):
            prob_series = pivot[col] * 100
            color = SERIES_COLORS[series_idx % len(SERIES_COLORS)]
            label = V4_PROB_LABELS.get(col, col)
            ax_main.plot(dates, prob_series, color=color, linewidth=2.0, alpha=0.9, zorder=3, label=label)
        ax_main.fill_between(dates, 0, 100, alpha=0.05, color="gray", zorder=1)

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


def create_player_dashboard_prob_variant(
    ctx: PlayerDashboardContext,
    pivot_single: pd.DataFrame,
    variant: str,
    risk_df: pd.DataFrame,
    insights: dict,
    trend_metrics: dict,
    output_dir: Path,
    injury_periods: List[Tuple[pd.Timestamp, pd.Timestamp, str]] = None,
    player_profile: Optional[pd.Series] = None,
) -> Path:
    """Create probability dashboard with a single combined curve (muscular or skeletal formula)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.0, 1.0], hspace=0.5, wspace=0.45)

    ax_main = fig.add_subplot(gs[0, :])
    dates = pivot_single["reference_date"]
    actual_start = dates.min()
    actual_end = dates.max()

    # Injury period shading: skeletal (dark red), muscular (red), unknown (light red), other (light yellow)
    if injury_periods:
        shown_labels = set()
        for period_start, period_end, injury_class in injury_periods:
            chart_start = dates.min()
            chart_end = dates.max()
            if period_start <= chart_end and period_end >= chart_start:
                shade_start = max(period_start, chart_start)
                shade_end = min(period_end, chart_end)
                color, label = _injury_period_color_and_label(injury_class)
                show_label = label if label not in shown_labels else ""
                if label not in shown_labels:
                    shown_labels.add(label)
                ax_main.axvspan(shade_start, shade_end, alpha=0.15, color=color, zorder=0, label=show_label)

    prob_series = pivot_single["combined_prob"]
    risk_labels_4level = [classify_risk_4level(p)["label"] for p in prob_series]
    prob_pct = prob_series * 100

    if HAS_SCIPY and len(dates) > 3:
        dates_numeric = np.arange(len(dates))
        x_smooth = np.linspace(dates_numeric.min(), dates_numeric.max(), 500)
        dates_smooth = pd.date_range(dates.min(), dates.max(), periods=500)
        spl = make_interp_spline(dates_numeric, prob_pct.values, k=min(3, len(dates) - 1))
        y_smooth = spl(x_smooth)
        ax_main.plot(dates_smooth, y_smooth, color="#1f77b4", linewidth=2.0, alpha=0.9, zorder=3, label=f"Combined ({variant.title()})")
    else:
        ax_main.plot(dates, prob_pct, color="#1f77b4", linewidth=2.0, alpha=0.9, zorder=3, label=f"Combined ({variant.title()})")
    ax_main.fill_between(dates, 0, 100, alpha=0.05, color="gray", zorder=1)

    variant_title = variant.title()
    ax_main.set_title(f"Injury Risk Evolution ({variant_title}) - {ctx.player_name}", fontsize=14, fontweight="bold", pad=15)
    ax_main.set_ylabel("Injury Probability (%)", fontweight="bold")
    ax_main.set_xlabel("Date", fontweight="bold")
    ax_main.set_yticks([0, 25, 50, 75, 100])
    ax_main.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax_main.set_ylim(0, 100)
    ax_main.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax_main.grid(axis="y", alpha=0.3, linestyle="--")
    ax_main.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax_main.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax_main.xaxis.set_minor_formatter(mdates.DateFormatter(''))
    plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=6)

    # Bottom panels (same as create_player_dashboard)
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
        formatted_labels = [l.replace("_", " ").title() for l in labels_sev]
        formatted_labels = ["Long term" if x == "Long Term" else x for x in formatted_labels]
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

    fig.text(0.5, 0.02, "Note: Body part and severity probabilities refer to the last day shown in the main chart.",
             ha='center', fontsize=7, style='italic', color='gray')

    ax_alert = fig.add_subplot(gs[1, 2])
    ax_alert.axis("off")
    profile_text = "PLAYER PROFILE\n\n"
    if player_profile is not None and len(player_profile) > 0:
        position = player_profile.get('position', 'N/A')
        date_of_birth = player_profile.get('date_of_birth', '')
        nationality1 = player_profile.get('nationality1', '')
        nationality2 = player_profile.get('nationality2', '')
        age = "N/A"
        if date_of_birth and pd.notna(date_of_birth):
            try:
                dob = pd.to_datetime(date_of_birth)
                age = str((actual_end - dob).days // 365)
            except Exception:
                pass
        nationality = str(nationality1) if pd.notna(nationality1) and str(nationality1).strip() else 'N/A'
        if nationality2 and pd.notna(nationality2) and str(nationality2).strip():
            nationality += f" / {nationality2}"
        profile_text += f"Position: {position}\nAge: {age}\nNationality: {nationality}\n"
    else:
        profile_text += "Position: N/A\nAge: N/A\nNationality: N/A\n"
    profile_text += "\n" + "=" * 20 + "\n\n"
    if risk_labels_4level:
        peak_risk = max(risk_labels_4level, key=lambda x: RISK_CLASS_LABELS_4LEVEL.index(x))
        final_risk = risk_labels_4level[-1]
    else:
        peak_risk = "N/A"
        final_risk = "N/A"
    sustained = trend_metrics.get("sustained_elevated_days", 0) if trend_metrics else 0
    max_jump = trend_metrics.get("max_jump", 0.0) if trend_metrics else 0.0
    slope = trend_metrics.get("slope", 0.0) if trend_metrics else 0.0
    profile_text += f"RISK SUMMARY\n\nPeak Risk Level: {peak_risk}\nFinal Risk Level: {final_risk}\nSustained Elevated: {sustained} days\nMax Daily Jump: {max_jump:.3f}\nTrend Slope: {slope:.4f}"
    ax_alert.text(0.1, 0.5, profile_text, transform=ax_alert.transAxes, fontsize=9, verticalalignment="center", family="monospace", fontweight="bold")

    plt.suptitle(f"Observation Period: {actual_start.strftime('%Y-%m-%d')} to {actual_end.strftime('%Y-%m-%d')}", fontsize=11, y=0.98)
    chart_path = output_dir / ctx.chart_filename_variant(variant, "probabilities")
    plt.savefig(chart_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return chart_path


def _get_msu_injury_end_dates(
    injury_periods: List[Tuple[pd.Timestamp, pd.Timestamp, str]],
    chart_start: pd.Timestamp,
    chart_end: pd.Timestamp,
) -> List[pd.Timestamp]:
    """Return sorted list of end dates of MSU (muscular/skeletal/unknown) injury periods within the chart range."""
    msu_classes = {"muscular", "skeletal", "unknown"}
    ends = []
    for period_start, period_end, injury_class in injury_periods:
        ic = str(injury_class).lower().strip()
        if ic not in msu_classes:
            continue
        if period_start <= chart_end and period_end >= chart_start:
            end_n = pd.Timestamp(period_end).normalize()
            if chart_start < end_n < chart_end:
                ends.append(end_n)
    return sorted(set(ends))


def create_player_dashboard_index_variant(
    ctx: PlayerDashboardContext,
    pivot_single: pd.DataFrame,
    variant: str,
    risk_df: pd.DataFrame,
    insights: dict,
    trend_metrics: dict,
    output_dir: Path,
    injury_periods: List[Tuple[pd.Timestamp, pd.Timestamp, str]] = None,
    player_profile: Optional[pd.Series] = None,
) -> Path:
    """Create index dashboard with one combined curve; 95% CI resets after each MSU injury end."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.0, 1.0], hspace=0.5, wspace=0.45)

    ax_main = fig.add_subplot(gs[0, :])
    dates = pivot_single["reference_date"]
    actual_start = dates.min()
    actual_end = dates.max()
    probs = pivot_single["combined_prob"]

    # Injury period shading (same 4-level)
    if injury_periods:
        shown_labels = set()
        for period_start, period_end, injury_class in injury_periods:
            chart_start = dates.min()
            chart_end = dates.max()
            if period_start <= chart_end and period_end >= chart_start:
                shade_start = max(period_start, chart_start)
                shade_end = min(period_end, chart_end)
                color, label = _injury_period_color_and_label(injury_class)
                show_label = label if label not in shown_labels else ""
                if label not in shown_labels:
                    shown_labels.add(label)
                ax_main.axvspan(shade_start, shade_end, alpha=0.15, color=color, zorder=0, label=show_label)

    # Segment boundaries: chart_start, then each MSU injury end date in range, then chart_end
    chart_start = dates.min()
    chart_end = dates.max()
    msu_ends = _get_msu_injury_end_dates(injury_periods or [], chart_start, chart_end)
    boundaries = [chart_start] + msu_ends + [chart_end]
    boundaries = sorted(set(boundaries))

    # Per-segment baseline, CI, and index curve
    first_ci_label = True
    first_curve_label = True
    y_max_plot = 2.5
    for i in range(len(boundaries) - 1):
        seg_start, seg_end = boundaries[i], boundaries[i + 1]
        mask = (dates >= seg_start) & (dates <= seg_end)
        if not mask.any():
            continue
        dates_seg = dates[mask]
        probs_seg = probs[mask]
        baseline_seg = float(probs_seg.mean())
        if baseline_seg < 1e-9:
            baseline_seg = 1e-9
        std_seg = float(probs_seg.std())
        if pd.isna(std_seg) or std_seg < 1e-12:
            std_seg = 0.0
        cv_seg = std_seg / baseline_seg
        index_low_seg = max(0.0, 1.0 - 1.96 * cv_seg)
        index_high_seg = 1.0 + 1.96 * cv_seg
        index_series_seg = probs_seg / baseline_seg

        label_ci = "95% CI (baseline)" if first_ci_label else ""
        if first_ci_label:
            first_ci_label = False
        ax_main.fill_between(dates_seg, index_low_seg, index_high_seg, alpha=0.25, color="#FFF9C4", zorder=1, label=label_ci)
        y_max_plot = max(y_max_plot, float(index_series_seg.max()) * 1.1 if len(index_series_seg) else 2.5)

        curve_label = f"Combined ({variant.title()})" if first_curve_label else ""
        if first_curve_label:
            first_curve_label = False
        if HAS_SCIPY and len(dates_seg) > 3:
            dates_numeric = np.arange(len(dates_seg))
            x_smooth = np.linspace(dates_numeric.min(), dates_numeric.max(), min(500, len(dates_seg) * 10))
            dates_smooth = pd.date_range(dates_seg.min(), dates_seg.max(), periods=len(x_smooth))
            spl = make_interp_spline(dates_numeric, index_series_seg.values, k=min(3, len(dates_seg) - 1))
            y_smooth = np.maximum(spl(x_smooth), 0.0)
            ax_main.plot(dates_smooth, y_smooth, color="#1f77b4", linewidth=2.0, alpha=0.9, zorder=3, label=curve_label)
        else:
            ax_main.plot(dates_seg, index_series_seg, color="#1f77b4", linewidth=2.0, alpha=0.9, zorder=3, label=curve_label)

    ax_main.axhline(1.0, color="gray", linestyle="--", linewidth=1.5, zorder=2, label="Baseline")
    ax_main.legend(loc="upper left", fontsize=9, framealpha=0.9)

    variant_title = variant.title()
    ax_main.set_title(f"Injury Risk Index vs Baseline ({variant_title}) - {ctx.player_name}", fontsize=14, fontweight="bold", pad=15)
    ax_main.set_ylabel("Injury Risk Index (vs baseline)", fontweight="bold")
    ax_main.set_xlabel("Date", fontweight="bold")
    ax_main.axhline(1.0, color="gray", linestyle="--", linewidth=1, zorder=0)
    tick_max = float(np.ceil(y_max_plot * 2) / 2)
    ax_main.set_ylim(0, tick_max)
    ax_main.set_yticks(np.arange(0, tick_max + 0.5, 0.5))
    plt.setp(ax_main.yaxis.get_majorticklabels(), fontsize=6)
    ax_main.grid(axis="y", alpha=0.3, linestyle="--")
    ax_main.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax_main.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax_main.xaxis.set_minor_formatter(mdates.DateFormatter(""))
    plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=6)

    # Bottom panels (same as index dashboard)
    ax_body = fig.add_subplot(gs[1, 0])
    if insights and insights.get("bodypart_rank"):
        body_rank = insights.get("bodypart_rank", [])[:3]
        labels = [item[0].replace("_", " ").title() for item in body_rank]
        probs_b = [item[1] * 100 for item in body_rank]
        colors_body = ["#2E86AB", "#A23B72", "#F18F01"][: len(labels)]
        bars = ax_body.barh(labels, probs_b, color=colors_body, alpha=0.8, edgecolor="white", linewidth=0.5, height=0.6)
        ax_body.set_xlabel("Probability (%)", fontweight="bold", fontsize=8)
        ax_body.set_title("Body Parts at Risk", fontweight="bold", pad=5, fontsize=9)
        ax_body.set_xlim(0, 100)
        ax_body.tick_params(labelsize=7)
        for i, (bar, prob) in enumerate(zip(bars, probs_b)):
            ax_body.text(prob + 2, i, f"{prob:.1f}%", va="center", fontweight="bold", fontsize=7)
        ax_body.grid(axis="x", alpha=0.3, linestyle="--")
    else:
        ax_body.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_body.transAxes, fontsize=8)
        ax_body.set_title("Body Parts at Risk", fontweight="bold", fontsize=9)

    ax_sev = fig.add_subplot(gs[1, 1])
    if insights and insights.get("severity_probs"):
        severity_probs = insights.get("severity_probs", {})
        labels_sev = list(severity_probs.keys())
        sizes = [severity_probs[k] * 100 for k in labels_sev]
        colors_sev = ["#06A77D", "#F4A261", "#E76F51", "#264653"][: len(labels_sev)]
        bars_sev = ax_sev.barh(
            range(len(labels_sev)), sizes, color=colors_sev, alpha=0.8, edgecolor="white", linewidth=0.5, height=0.6
        )
        ax_sev.set_yticks(range(len(labels_sev)))
        formatted_labels = [l.replace("_", " ").title() for l in labels_sev]
        formatted_labels = ["Long term" if x == "Long Term" else x for x in formatted_labels]
        ax_sev.set_yticklabels(formatted_labels, fontsize=7)
        ax_sev.set_xlabel("Probability (%)", fontweight="bold", fontsize=8)
        sev_title = (insights.get("severity_label", "Unknown") or "Unknown").replace("_", " ").title()
        if sev_title == "Long Term":
            sev_title = "Long term"
        ax_sev.set_title(f"Severity: {sev_title}", fontweight="bold", pad=5, fontsize=9)
        ax_sev.set_xlim(0, 100)
        ax_sev.tick_params(labelsize=7)
        for i, (bar, size) in enumerate(zip(bars_sev, sizes)):
            ax_sev.text(size + 2, i, f"{size:.1f}%", va="center", fontweight="bold", fontsize=7)
        ax_sev.grid(axis="x", alpha=0.3, linestyle="--")
    else:
        ax_sev.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_sev.transAxes, fontsize=8)
        ax_sev.set_title("Severity", fontweight="bold", fontsize=9)

    fig.text(0.5, 0.02, "Note: Body part and severity probabilities refer to the last day shown in the main chart.",
             ha="center", fontsize=7, style="italic", color="gray")

    ax_alert = fig.add_subplot(gs[1, 2])
    ax_alert.axis("off")
    profile_text = "PLAYER PROFILE\n\n"
    if player_profile is not None and len(player_profile) > 0:
        position = player_profile.get("position", "N/A")
        date_of_birth = player_profile.get("date_of_birth", "")
        nationality1 = player_profile.get("nationality1", "")
        nationality2 = player_profile.get("nationality2", "")
        age = "N/A"
        if date_of_birth and pd.notna(date_of_birth):
            try:
                dob = pd.to_datetime(date_of_birth)
                age = str((actual_end - dob).days // 365)
            except Exception:
                pass
        nationality = str(nationality1) if pd.notna(nationality1) and str(nationality1).strip() else "N/A"
        if nationality2 and pd.notna(nationality2) and str(nationality2).strip():
            nationality += f" / {nationality2}"
        profile_text += f"Position: {position}\nAge: {age}\nNationality: {nationality}\n"
    else:
        profile_text += "Position: N/A\nAge: N/A\nNationality: N/A\n"
    profile_text += "\n" + "=" * 20 + "\n\n"
    risk_labels_4level = [classify_risk_4level(p)["label"] for p in probs]
    peak_risk = max(risk_labels_4level, key=lambda x: RISK_CLASS_LABELS_4LEVEL.index(x)) if risk_labels_4level else "N/A"
    final_risk = risk_labels_4level[-1] if risk_labels_4level else "N/A"
    sustained = trend_metrics.get("sustained_elevated_days", 0) if trend_metrics else 0
    max_jump = trend_metrics.get("max_jump", 0.0) if trend_metrics else 0.0
    slope = trend_metrics.get("slope", 0.0) if trend_metrics else 0.0
    profile_text += f"RISK SUMMARY\n\nPeak Risk Level: {peak_risk}\nFinal Risk Level: {final_risk}\nSustained Elevated: {sustained} days\nMax Daily Jump: {max_jump:.3f}\nTrend Slope: {slope:.4f}"
    ax_alert.text(0.1, 0.5, profile_text, transform=ax_alert.transAxes, fontsize=9, verticalalignment="center", family="monospace", fontweight="bold")

    plt.suptitle(f"Observation Period: {actual_start.strftime('%Y-%m-%d')} to {actual_end.strftime('%Y-%m-%d')}", fontsize=11, y=0.98)
    chart_path = output_dir / ctx.chart_filename_variant(variant, "index")
    plt.savefig(chart_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return chart_path


def create_player_dashboard_index_multicurve(
    ctx: PlayerDashboardContext,
    pivot_multi: pd.DataFrame,
    curves: Dict[str, Tuple[str, str]],
    variant: str,
    risk_df: pd.DataFrame,
    insights: dict,
    trend_metrics: dict,
    output_dir: Path,
    injury_periods: List[Tuple[pd.Timestamp, pd.Timestamp, str]] = None,
    player_profile: Optional[pd.Series] = None,
    kind: str = "index_components",
) -> Path:
    """
    Create an index-based dashboard showing multiple component curves
    (e.g. LGBM, GB, MSU) as index vs baseline, with MSU-based segmentation
    and 95% CI band (similar to create_player_dashboard_index_variant).

    - pivot_multi: columns = ["reference_date", <probability columns...>]
    - curves: mapping from probability column name to (display label, color)
    - variant: "muscular" or "skeletal"
    - kind: used in the filename, e.g. "index_components"
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.0, 1.0], hspace=0.5, wspace=0.45)

    ax_main = fig.add_subplot(gs[0, :])
    dates = pivot_multi["reference_date"]
    actual_start = dates.min()
    actual_end = dates.max()

    # Injury period shading (same 4-level scheme)
    if injury_periods:
        shown_labels = set()
        for period_start, period_end, injury_class in injury_periods:
            chart_start = dates.min()
            chart_end = dates.max()
            if period_start <= chart_end and period_end >= chart_start:
                shade_start = max(period_start, chart_start)
                shade_end = min(period_end, chart_end)
                color, label = _injury_period_color_and_label(injury_class)
                show_label = label if label not in shown_labels else ""
                if label not in shown_labels:
                    shown_labels.add(label)
                ax_main.axvspan(
                    shade_start,
                    shade_end,
                    alpha=0.15,
                    color=color,
                    zorder=0,
                    label=show_label,
                )

    chart_start = dates.min()
    chart_end = dates.max()
    msu_ends = _get_msu_injury_end_dates(injury_periods or [], chart_start, chart_end)
    boundaries = [chart_start] + msu_ends + [chart_end]
    boundaries = sorted(set(boundaries))

    prob_cols = [c for c in pivot_multi.columns if c != "reference_date"]
    if not prob_cols:
        ax_main.text(
            0.5,
            0.5,
            "No probability data available",
            ha="center",
            va="center",
            transform=ax_main.transAxes,
            fontsize=9,
        )
        chart_path = output_dir / ctx.chart_filename_variant(variant, kind)
        plt.savefig(chart_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return chart_path

    primary_col = prob_cols[0]
    first_ci_label = True
    y_max_plot = 2.5

    for i in range(len(boundaries) - 1):
        seg_start, seg_end = boundaries[i], boundaries[i + 1]
        mask = (dates >= seg_start) & (dates <= seg_end)
        if not mask.any():
            continue

        dates_seg = dates[mask]

        # CI band based on primary curve in this segment
        primary_probs_seg = pivot_multi.loc[mask, primary_col]
        baseline_seg = float(primary_probs_seg.mean())
        if baseline_seg < 1e-9:
            baseline_seg = 1e-9
        std_seg = float(primary_probs_seg.std())
        if pd.isna(std_seg) or std_seg < 1e-12:
            std_seg = 0.0
        cv_seg = std_seg / baseline_seg
        index_low_seg = max(0.0, 1.0 - 1.96 * cv_seg)
        index_high_seg = 1.0 + 1.96 * cv_seg

        ax_main.fill_between(
            dates_seg,
            index_low_seg,
            index_high_seg,
            alpha=0.25,
            color="#FFF9C4",
            zorder=1,
            label="95% CI (baseline)" if first_ci_label else "",
        )
        if first_ci_label:
            first_ci_label = False

        # Plot each component curve as its own index vs its segment baseline
        for col in prob_cols:
            probs_seg = pivot_multi.loc[mask, col]
            base_col = float(probs_seg.mean())
            if base_col < 1e-9:
                base_col = 1e-9
            index_series_seg = probs_seg / base_col
            y_max_plot = max(
                y_max_plot,
                float(index_series_seg.max()) * 1.1 if len(index_series_seg) else 2.5,
            )

            label, color = curves.get(col, (col, "#1f77b4"))
            if HAS_SCIPY and len(dates_seg) > 3:
                dates_numeric = np.arange(len(dates_seg))
                x_smooth = np.linspace(
                    dates_numeric.min(),
                    dates_numeric.max(),
                    min(500, len(dates_seg) * 10),
                )
                dates_smooth = pd.date_range(
                    dates_seg.min(), dates_seg.max(), periods=len(x_smooth)
                )
                spl = make_interp_spline(
                    dates_numeric,
                    index_series_seg.values,
                    k=min(3, len(dates_seg) - 1),
                )
                y_smooth = np.maximum(spl(x_smooth), 0.0)
                ax_main.plot(
                    dates_smooth,
                    y_smooth,
                    color=color,
                    linewidth=2.0,
                    alpha=0.9,
                    zorder=3,
                    label=label,
                )
            else:
                ax_main.plot(
                    dates_seg,
                    index_series_seg,
                    color=color,
                    linewidth=2.0,
                    alpha=0.9,
                    zorder=3,
                    label=label,
                )

    ax_main.axhline(
        1.0,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        zorder=2,
        label="Baseline",
    )
    ax_main.legend(loc="upper left", fontsize=9, framealpha=0.9)

    variant_title = variant.title()
    ax_main.set_title(
        f"Injury Risk Index Components vs Baseline ({variant_title}) - {ctx.player_name}",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax_main.set_ylabel("Injury Risk Index (vs baseline)", fontweight="bold")
    ax_main.set_xlabel("Date", fontweight="bold")
    ax_main.axhline(1.0, color="gray", linestyle="--", linewidth=1, zorder=0)
    tick_max = float(np.ceil(y_max_plot * 2) / 2)
    ax_main.set_ylim(0, tick_max)
    ax_main.set_yticks(np.arange(0, tick_max + 0.5, 0.5))
    plt.setp(ax_main.yaxis.get_majorticklabels(), fontsize=6)
    ax_main.grid(axis="y", alpha=0.3, linestyle="--")
    ax_main.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax_main.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax_main.xaxis.set_minor_formatter(mdates.DateFormatter(""))
    plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=6)

    # Bottom panels: body parts, severity, risk summary
    ax_body = fig.add_subplot(gs[1, 0])
    if insights and insights.get("bodypart_rank"):
        body_rank = insights.get("bodypart_rank", [])[:3]
        labels = [item[0].replace("_", " ").title() for item in body_rank]
        probs_b = [item[1] * 100 for item in body_rank]
        colors_body = ["#2E86AB", "#A23B72", "#F18F01"][: len(labels)]
        bars = ax_body.barh(
            labels,
            probs_b,
            color=colors_body,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.5,
            height=0.6,
        )
        ax_body.set_xlabel("Probability (%)", fontweight="bold", fontsize=8)
        ax_body.set_title("Body Parts at Risk", fontweight="bold", pad=5, fontsize=9)
        ax_body.set_xlim(0, 100)
        ax_body.tick_params(labelsize=7)
        for i, (bar, prob) in enumerate(zip(bars, probs_b)):
            ax_body.text(
                prob + 2,
                i,
                f"{prob:.1f}%",
                va="center",
                fontweight="bold",
                fontsize=7,
            )
        ax_body.grid(axis="x", alpha=0.3, linestyle="--")
    else:
        ax_body.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            transform=ax_body.transAxes,
            fontsize=8,
        )
        ax_body.set_title("Body Parts at Risk", fontweight="bold", fontsize=9)

    ax_sev = fig.add_subplot(gs[1, 1])
    if insights and insights.get("severity_probs"):
        severity_probs = insights.get("severity_probs", {})
        labels_sev = list(severity_probs.keys())
        sizes = [severity_probs[k] * 100 for k in labels_sev]
        colors_sev = ["#06A77D", "#F4A261", "#E76F51", "#264653"][: len(labels_sev)]
        bars_sev = ax_sev.barh(
            range(len(labels_sev)),
            sizes,
            color=colors_sev,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.5,
            height=0.6,
        )
        ax_sev.set_yticks(range(len(labels_sev)))
        formatted_labels = [l.replace("_", " ").title() for l in labels_sev]
        formatted_labels = ["Long term" if x == "Long Term" else x for x in formatted_labels]
        ax_sev.set_yticklabels(formatted_labels, fontsize=7)
        ax_sev.set_xlabel("Probability (%)", fontweight="bold", fontsize=8)
        sev_title = (insights.get("severity_label", "Unknown") or "Unknown").replace("_", " ").title()
        if sev_title == "Long Term":
            sev_title = "Long term"
        ax_sev.set_title(f"Severity: {sev_title}", fontweight="bold", pad=5, fontsize=9)
        ax_sev.set_xlim(0, 100)
        ax_sev.tick_params(labelsize=7)
        for i, (bar, size) in enumerate(zip(bars_sev, sizes)):
            ax_sev.text(size + 2, i, f"{size:.1f}%", va="center", fontweight="bold", fontsize=7)
        ax_sev.grid(axis="x", alpha=0.3, linestyle="--")
    else:
        ax_sev.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_sev.transAxes, fontsize=8)
        ax_sev.set_title("Severity", fontweight="bold", fontsize=9)

    fig.text(0.5, 0.02, "Note: Body part and severity probabilities refer to the last day shown in the main chart.",
             ha="center", fontsize=7, style="italic", color="gray")

    ax_alert = fig.add_subplot(gs[1, 2])
    ax_alert.axis("off")
    profile_text = "PLAYER PROFILE\n\n"
    if player_profile is not None and len(player_profile) > 0:
        position = player_profile.get("position", "N/A")
        date_of_birth = player_profile.get("date_of_birth", "")
        nationality1 = player_profile.get("nationality1", "")
        nationality2 = player_profile.get("nationality2", "")
        age = "N/A"
        if date_of_birth and pd.notna(date_of_birth):
            try:
                dob = pd.to_datetime(date_of_birth)
                age = str((actual_end - dob).days // 365)
            except Exception:
                pass
        nationality = str(nationality1) if pd.notna(nationality1) and str(nationality1).strip() else "N/A"
        if nationality2 and pd.notna(nationality2) and str(nationality2).strip():
            nationality += f" / {nationality2}"
        profile_text += f"Position: {position}\nAge: {age}\nNationality: {nationality}\n"
    else:
        profile_text += "Position: N/A\nAge: N/A\nNationality: N/A\n"
    profile_text += "\n" + "=" * 20 + "\n\n"
    risk_labels_4level = [classify_risk_4level(p)["label"] for p in pivot_multi[primary_col]]
    peak_risk = max(risk_labels_4level, key=lambda x: RISK_CLASS_LABELS_4LEVEL.index(x)) if risk_labels_4level else "N/A"
    final_risk = risk_labels_4level[-1] if risk_labels_4level else "N/A"
    sustained = trend_metrics.get("sustained_elevated_days", 0) if trend_metrics else 0
    max_jump = trend_metrics.get("max_jump", 0.0) if trend_metrics else 0.0
    slope = trend_metrics.get("slope", 0.0) if trend_metrics else 0.0
    profile_text += f"RISK SUMMARY\n\nPeak Risk Level: {peak_risk}\nFinal Risk Level: {final_risk}\nSustained Elevated: {sustained} days\nMax Daily Jump: {max_jump:.3f}\nTrend Slope: {slope:.4f}"
    ax_alert.text(0.1, 0.5, profile_text, transform=ax_alert.transAxes, fontsize=9, verticalalignment="center", family="monospace", fontweight="bold")

    plt.suptitle(f"Observation Period: {actual_start.strftime('%Y-%m-%d')} to {actual_end.strftime('%Y-%m-%d')}", fontsize=11, y=0.98)
    chart_path = output_dir / ctx.chart_filename_variant(variant, kind)
    plt.savefig(chart_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return chart_path

def create_player_dashboard_index(
    ctx: PlayerDashboardContext,
    pivot: pd.DataFrame,
    risk_df: pd.DataFrame,
    insights: dict,
    trend_metrics: dict,
    output_dir: Path,
    injury_periods: List[Tuple[pd.Timestamp, pd.Timestamp, str]] = None,
    player_profile: Optional[pd.Series] = None,
) -> Path:
    """Create dashboard PNG with index-based main chart (current prediction / baseline) for V4.
    Baseline = mean(injury_probability) over the chart period. Index = prob / baseline.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.0, 1.0], hspace=0.5, wspace=0.45)

    ax_main = fig.add_subplot(gs[0, :])
    dates = pivot["reference_date"]
    probabilities = pivot["injury_probability"]
    actual_start = dates.min()
    actual_end = dates.max()

    if injury_periods:
        shown_labels = set()
        for period_start, period_end, injury_class in injury_periods:
            chart_start = dates.min()
            chart_end = dates.max()
            if period_start <= chart_end and period_end >= chart_start:
                shade_start = max(period_start, chart_start)
                shade_end = min(period_end, chart_end)
                color, label = _injury_period_color_and_label(injury_class)
                show_label = label if label not in shown_labels else ""
                if label not in shown_labels:
                    shown_labels.add(label)
                ax_main.axvspan(shade_start, shade_end, alpha=0.15, color=color, zorder=0, label=show_label)

    # Which probability column(s): 3-model or legacy
    prob_cols_idx = [c for c in V4_PROB_COLUMNS if c in pivot.columns]
    if not prob_cols_idx and "injury_probability" in pivot.columns:
        prob_cols_idx = ["injury_probability"]
    if not prob_cols_idx:
        prob_cols_idx = [c for c in pivot.columns if "injury_probability" in c]
    if not prob_cols_idx:
        prob_cols_idx = [pivot.columns[1]] if len(pivot.columns) > 1 else []

    # Use first series for baseline band (95% CI)
    probabilities = pivot[prob_cols_idx[0]]
    baseline = float(probabilities.mean())
    if baseline < 1e-9:
        baseline = 1e-9
    std_prob = float(probabilities.std())
    if pd.isna(std_prob) or std_prob < 1e-12:
        std_prob = 0.0
    cv = std_prob / baseline
    index_low = max(0.0, 1.0 - 1.96 * cv)
    index_high = 1.0 + 1.96 * cv

    ax_main.fill_between(dates, index_low, index_high, alpha=0.25, color="#FFF9C4", zorder=1, label="95% CI (baseline)")
    ax_main.axhline(1.0, color="gray", linestyle="--", linewidth=1.5, zorder=2, label="Baseline")

    SERIES_COLORS_INDEX = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    y_max_plot = 2.5
    for series_idx, col in enumerate(prob_cols_idx):
        probs = pivot[col]
        base = float(probs.mean())
        if base < 1e-9:
            base = 1e-9
        index_series = probs / base
        y_max_plot = max(y_max_plot, float(index_series.max()) * 1.1 if len(index_series) else 2.5)
        color = SERIES_COLORS_INDEX[series_idx % len(SERIES_COLORS_INDEX)]
        label = V4_PROB_LABELS.get(col, col)
        if HAS_SCIPY and len(dates) > 3:
            dates_numeric = np.arange(len(dates))
            x_smooth = np.linspace(dates_numeric.min(), dates_numeric.max(), 500)
            dates_smooth = pd.date_range(dates.min(), dates.max(), periods=500)
            spl = make_interp_spline(dates_numeric, index_series.values, k=min(3, len(dates) - 1))
            y_smooth = np.maximum(spl(x_smooth), 0.0)
            ax_main.plot(dates_smooth, y_smooth, color=color, linewidth=2.0, alpha=0.9, zorder=3, label=label)
        else:
            ax_main.plot(dates, index_series, color=color, linewidth=2.0, alpha=0.9, zorder=3, label=label)

    ax_main.set_title(f"Injury Risk Index vs Baseline - {ctx.player_name}", fontsize=14, fontweight="bold", pad=15)
    ax_main.set_ylabel("Injury Risk Index (vs baseline)", fontweight="bold")
    ax_main.set_xlabel("Date", fontweight="bold")
    ax_main.axhline(1.0, color="gray", linestyle="--", linewidth=1, zorder=0)
    tick_max = float(np.ceil(y_max_plot * 2) / 2)
    ax_main.set_ylim(0, tick_max)
    ax_main.set_yticks(np.arange(0, tick_max + 0.5, 0.5))
    plt.setp(ax_main.yaxis.get_majorticklabels(), fontsize=6)
    ax_main.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax_main.grid(axis="y", alpha=0.3, linestyle="--")
    ax_main.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax_main.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax_main.xaxis.set_minor_formatter(mdates.DateFormatter(""))
    plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=6)

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

    ax_sev = fig.add_subplot(gs[1, 1])
    if insights and insights.get("severity_probs"):
        severity_probs = insights.get("severity_probs", {})
        labels_sev = list(severity_probs.keys())
        sizes = [severity_probs[k] * 100 for k in labels_sev]
        colors_sev = ["#06A77D", "#F4A261", "#E76F51", "#264653"][: len(labels_sev)]
        bars_sev = ax_sev.barh(
            range(len(labels_sev)), sizes, color=colors_sev, alpha=0.8, edgecolor="white", linewidth=0.5, height=0.6
        )
        ax_sev.set_yticks(range(len(labels_sev)))
        formatted_labels = [l.replace("_", " ").title() for l in labels_sev]
        formatted_labels = ["Long term" if x == "Long Term" else x for x in formatted_labels]
        ax_sev.set_yticklabels(formatted_labels, fontsize=7)
        ax_sev.set_xlabel("Probability (%)", fontweight="bold", fontsize=8)
        sev_title = (insights.get("severity_label", "Unknown") or "Unknown").replace("_", " ").title()
        if sev_title == "Long Term":
            sev_title = "Long term"
        ax_sev.set_title(f"Severity: {sev_title}", fontweight="bold", pad=5, fontsize=9)
        ax_sev.set_xlim(0, 100)
        ax_sev.tick_params(labelsize=7)
        for i, (bar, size) in enumerate(zip(bars_sev, sizes)):
            ax_sev.text(size + 2, i, f"{size:.1f}%", va="center", fontweight="bold", fontsize=7)
        ax_sev.grid(axis="x", alpha=0.3, linestyle="--")
    else:
        ax_sev.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_sev.transAxes, fontsize=8)
        ax_sev.set_title("Severity", fontweight="bold", fontsize=9)

    fig.text(0.5, 0.02, "Note: Body part and severity probabilities refer to the last day shown in the main chart.",
             ha="center", fontsize=7, style="italic", color="gray")

    # Primary series for risk summary
    primary_col_alert = PRIMARY_PROB_COL if PRIMARY_PROB_COL in pivot.columns else (prob_cols_idx[0] if prob_cols_idx else None)
    probabilities = pivot[primary_col_alert] if primary_col_alert else pd.Series(dtype=float)

    ax_alert = fig.add_subplot(gs[1, 2])
    ax_alert.axis("off")
    profile_text = "PLAYER PROFILE\n\n"
    if player_profile is not None and len(player_profile) > 0:
        position = player_profile.get("position", "N/A")
        date_of_birth = player_profile.get("date_of_birth", "")
        nationality1 = player_profile.get("nationality1", "")
        nationality2 = player_profile.get("nationality2", "")
        age = "N/A"
        if date_of_birth and pd.notna(date_of_birth):
            try:
                dob = pd.to_datetime(date_of_birth)
                age = str((actual_end - dob).days // 365)
            except Exception:
                pass
        nationality = str(nationality1) if pd.notna(nationality1) and str(nationality1).strip() else "N/A"
        if nationality2 and pd.notna(nationality2) and str(nationality2).strip():
            nationality += f" / {nationality2}"
        profile_text += f"Position: {position}\nAge: {age}\nNationality: {nationality}\n"
    else:
        profile_text += "Position: N/A\nAge: N/A\nNationality: N/A\n"
    profile_text += "\n" + "=" * 20 + "\n\n"
    risk_labels_4level = [classify_risk_4level(p)["label"] for p in probabilities]
    peak_risk = max(risk_labels_4level, key=lambda x: RISK_CLASS_LABELS_4LEVEL.index(x)) if risk_labels_4level else "N/A"
    final_risk = risk_labels_4level[-1] if risk_labels_4level else "N/A"
    sustained = trend_metrics.get("sustained_elevated_days", 0) if trend_metrics else 0
    max_jump = trend_metrics.get("max_jump", 0.0) if trend_metrics else 0.0
    slope = trend_metrics.get("slope", 0.0) if trend_metrics else 0.0
    profile_text += f"RISK SUMMARY\n\nPeak Risk Level: {peak_risk}\nFinal Risk Level: {final_risk}\nSustained Elevated: {sustained} days\nMax Daily Jump: {max_jump:.3f}\nTrend Slope: {slope:.4f}"
    ax_alert.text(0.1, 0.5, profile_text, transform=ax_alert.transAxes, fontsize=9, verticalalignment="center", family="monospace", fontweight="bold")

    plt.suptitle(
        f"Observation Period: {actual_start.strftime('%Y-%m-%d')} to {actual_end.strftime('%Y-%m-%d')}",
        fontsize=11, y=0.98,
    )
    chart_path = output_dir / ctx.chart_filename_index
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
    parser.add_argument("--dashboard-type", type=str, choices=["prob", "index", "both"], default="prob",
                        help="Dashboard type: prob (probability evolution), index (index vs baseline), both (default: prob)")
    # Muscular combined formula (default): 0.2 * MSU + 0.8 * muscular LGBM + 0.0 * muscular GB
    parser.add_argument("--muscular-w-msu", type=float, default=0.2,
                        help="Weight for MSU model in muscular combined probability (default: 0.2)")
    parser.add_argument("--muscular-w-lgbm", type=float, default=0.8,
                        help="Weight for muscular LGBM in muscular combined probability (default: 0.8)")
    parser.add_argument("--muscular-w-gb", type=float, default=0.0,
                        help="Weight for muscular GB in muscular combined probability (default: 0.0)")
    # Skeletal combined formula (default): 0.2 * MSU + 0.8 * skeletal LGBM
    parser.add_argument("--skeletal-w-msu", type=float, default=0.2,
                        help="Weight for MSU model in skeletal combined probability (default: 0.2)")
    parser.add_argument("--skeletal-w-skeletal", type=float, default=0.8,
                        help="Weight for skeletal model in skeletal combined probability (default: 0.8)")
    args = parser.parse_args()

    # Build weight dicts from CLI (defaults match current MUSCULAR_WEIGHTS / SKELETAL_WEIGHTS)
    muscular_weights = {
        "injury_probability_msu_lgbm": args.muscular_w_msu,
        "injury_probability_muscular_lgbm": args.muscular_w_lgbm,
        "injury_probability_muscular_gb": args.muscular_w_gb,
    }
    skeletal_weights = {
        "injury_probability_msu_lgbm": args.skeletal_w_msu,
        "injury_probability_skeletal": args.skeletal_w_skeletal,
    }

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
    print(f"📐 Muscular formula weights: MSU={muscular_weights['injury_probability_msu_lgbm']}, "
          f"LGBM={muscular_weights['injury_probability_muscular_lgbm']}, "
          f"GB={muscular_weights['injury_probability_muscular_gb']}")
    print(f"📐 Skeletal formula weights: MSU={skeletal_weights['injury_probability_msu_lgbm']}, "
          f"Skeletal={skeletal_weights['injury_probability_skeletal']}")
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

            # Pivot with all V4 prob columns for combined formulas
            pivot_cols = ["reference_date"]
            for c in V4_PROB_COLUMNS:
                if c in player_preds.columns:
                    pivot_cols.append(c)
            if "injury_probability" not in pivot_cols and "injury_probability" in player_preds.columns:
                pivot_cols.append("injury_probability")
            if "risk_level" in player_preds.columns:
                pivot_cols.append("risk_level")
            pivot = player_preds[[c for c in pivot_cols if c in player_preds.columns]].copy()

            # Combined probabilities using CLI weights (muscular: MSU + muscular_lgbm + muscular_gb; skeletal: MSU + skeletal)
            prob_muscular = compute_combined_probability(pivot, muscular_weights)
            prob_skeletal = compute_combined_probability(pivot, skeletal_weights)
            pivot_muscular = pivot[["reference_date"]].copy()
            pivot_muscular["combined_prob"] = prob_muscular
            pivot_skeletal = pivot[["reference_date"]].copy()
            pivot_skeletal["combined_prob"] = prob_skeletal

            trend_metrics_muscular = compute_trend_metrics(prob_muscular)
            trend_metrics_skeletal = compute_trend_metrics(prob_skeletal)
            risk_df_muscular = pd.DataFrame({
                "risk_index": [classify_risk_4level(p)["index"] for p in prob_muscular],
                "risk_label": [classify_risk_4level(p)["label"] for p in prob_muscular],
            })
            risk_df_skeletal = pd.DataFrame({
                "risk_index": [classify_risk_4level(p)["index"] for p in prob_skeletal],
                "risk_label": [classify_risk_4level(p)["label"] for p in prob_skeletal],
            })

            # Create context with reference_date for filename
            ctx = PlayerDashboardContext(
                player_id=player_id,
                player_name=player_name,
                window_start=start_date,
                window_end=end_date,
                reference_date=max_ref_date
            )

            # Create 6 dashboards: muscular/skeletal prob + index + index_components
            dashboard_type = getattr(args, "dashboard_type", "prob")
            if dashboard_type in ("prob", "both"):
                out_musc = create_player_dashboard_prob_variant(
                ctx=ctx,
                    pivot_single=pivot_muscular,
                    variant="muscular",
                    risk_df=risk_df_muscular,
                    insights=insights,
                    trend_metrics=trend_metrics_muscular,
                output_dir=dashboards_dir,
                    injury_periods=injury_periods,
                    player_profile=player_profile,
                )
                out_skel = create_player_dashboard_prob_variant(
                    ctx=ctx,
                    pivot_single=pivot_skeletal,
                    variant="skeletal",
                    risk_df=risk_df_skeletal,
                    insights=insights,
                    trend_metrics=trend_metrics_skeletal,
                    output_dir=dashboards_dir,
                    injury_periods=injury_periods,
                    player_profile=player_profile,
                )
                if not HAS_TQDM:
                    print(f"   [OK] Generated muscular prob: {out_musc.name}, skeletal prob: {out_skel.name}")
            if dashboard_type in ("index", "both"):
                out_musc_idx = create_player_dashboard_index_variant(
                    ctx=ctx,
                    pivot_single=pivot_muscular,
                    variant="muscular",
                    risk_df=risk_df_muscular,
                    insights=insights,
                    trend_metrics=trend_metrics_muscular,
                    output_dir=dashboards_dir,
                    injury_periods=injury_periods,
                    player_profile=player_profile,
                )
                out_skel_idx = create_player_dashboard_index_variant(
                    ctx=ctx,
                    pivot_single=pivot_skeletal,
                    variant="skeletal",
                    risk_df=risk_df_skeletal,
                    insights=insights,
                    trend_metrics=trend_metrics_skeletal,
                    output_dir=dashboards_dir,
                    injury_periods=injury_periods,
                    player_profile=player_profile,
                )

                # Additional index dashboards: multi-curve components (LGBM, GB, MSU)
                muscular_curve_cols = [
                    "injury_probability_muscular_lgbm",
                    "injury_probability_muscular_gb",
                    "injury_probability_msu_lgbm",
                ]
                musc_available = [c for c in muscular_curve_cols if c in pivot.columns]
                pivot_muscular_multi = pivot[["reference_date"] + musc_available].copy()
                muscular_curves: Dict[str, Tuple[str, str]] = {}
                for c in musc_available:
                    if c == "injury_probability_muscular_lgbm":
                        muscular_curves[c] = ("LGBM", "#1f77b4")
                    elif c == "injury_probability_muscular_gb":
                        muscular_curves[c] = ("GB", "#ff7f0e")
                    elif c == "injury_probability_msu_lgbm":
                        muscular_curves[c] = ("MSU LGBM", "#2ca02c")
                out_musc_idx_components = create_player_dashboard_index_multicurve(
                    ctx=ctx,
                    pivot_multi=pivot_muscular_multi,
                    curves=muscular_curves,
                    variant="muscular",
                    risk_df=risk_df_muscular,
                    insights=insights,
                    trend_metrics=trend_metrics_muscular,
                    output_dir=dashboards_dir,
                    injury_periods=injury_periods,
                    player_profile=player_profile,
                    kind="index_components",
                )

                skeletal_curve_cols = [
                    "injury_probability_skeletal",
                    "injury_probability_msu_lgbm",
                ]
                skel_available = [c for c in skeletal_curve_cols if c in pivot.columns]
                pivot_skeletal_multi = pivot[["reference_date"] + skel_available].copy()
                skeletal_curves: Dict[str, Tuple[str, str]] = {}
                for c in skel_available:
                    if c == "injury_probability_skeletal":
                        skeletal_curves[c] = ("Skeletal LGBM", "#1f77b4")
                    elif c == "injury_probability_msu_lgbm":
                        skeletal_curves[c] = ("MSU LGBM", "#2ca02c")
                out_skel_idx_components = create_player_dashboard_index_multicurve(
                    ctx=ctx,
                    pivot_multi=pivot_skeletal_multi,
                    curves=skeletal_curves,
                    variant="skeletal",
                    risk_df=risk_df_skeletal,
                    insights=insights,
                    trend_metrics=trend_metrics_skeletal,
                    output_dir=dashboards_dir,
                    injury_periods=injury_periods,
                    player_profile=player_profile,
                    kind="index_components",
                )

                if not HAS_TQDM:
                    msg = f"   [OK] Generated muscular index: {out_musc_idx.name}, skeletal index: {out_skel_idx.name}"
                    msg += (
                        f"; muscular index components: {out_musc_idx_components.name}, "
                        f"skeletal index components: {out_skel_idx_components.name}"
                    )
                    print(msg)
            
            successes += 1
        
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
