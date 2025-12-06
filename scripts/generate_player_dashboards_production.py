#!/usr/bin/env python3
"""
Generate per-player dashboards for production predictions.

Reuses the retrospective dashboard visual style (risk evolution + body parts +
severity) to create one PNG per player over a specified date range.
"""
from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.interpolate import make_interp_spline

    HAS_SCIPY = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_SCIPY = False

import sys
import io

# Try to import tqdm for progress bars, fallback to simple counter if not available
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

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.backtests.insight_utils import (  # pylint: disable=wrong-import-position
    RISK_CLASS_COLORS,
    RISK_CLASS_LABELS,
    compute_risk_series,
    compute_trend_metrics,
    load_pipeline,
)
from scripts.backtests.summarize_results import predict_insights

PRODUCTION_PREDICTIONS_DIR = ROOT_DIR / "production_predictions" / "predictions"
PRODUCTION_DASHBOARD_DIR = ROOT_DIR / "production_predictions" / "dashboards" / "players"
PRODUCTION_FEATURES_DIR = ROOT_DIR / "production_predictions" / "daily_features"


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


def load_prediction_file(predictions_dir: Path, player_id: int, suffix: str) -> pd.DataFrame:
    """Load ensemble predictions file which contains RF, GB, and Ensemble probabilities."""
    file_path = predictions_dir / "ensemble" / f"player_{player_id}_predictions_{suffix}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {file_path}")
    df = pd.read_csv(file_path, parse_dates=["reference_date"])
    if df.empty:
        raise ValueError(f"Predictions file {file_path} is empty")
    # Ensure we have all three probability columns
    required_cols = ["rf_probability", "gb_probability", "ensemble_probability"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required probability columns: {missing}")
    return df.sort_values("reference_date")


def load_latest_feature_row(player_id: int, end_date: pd.Timestamp) -> pd.Series:
    file_path = PRODUCTION_FEATURES_DIR / f"player_{player_id}_daily_features.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Daily features not found: {file_path}")
    df = pd.read_csv(file_path, parse_dates=["date"])
    df = df[df["date"] <= end_date].sort_values("date")
    if df.empty:
        raise ValueError(f"No feature rows available up to {end_date.date()} for player {player_id}")
    return df.iloc[-1]


def load_injury_periods(player_id: int, start_date: pd.Timestamp, end_date: pd.Timestamp) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Load injury periods for a player within the date range.
    
    Returns a list of (start_date, end_date) tuples for each injury period.
    """
    # Try to load from daily features first (if inj_days column exists)
    file_path = PRODUCTION_FEATURES_DIR / f"player_{player_id}_daily_features.csv"
    if file_path.exists():
        df = pd.read_csv(file_path, parse_dates=["date"])
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].sort_values("date")
        
        # Check if inj_days column exists
        if "inj_days" in df.columns:
            injury_periods = []
            in_injury = False
            period_start = None
            
            for _, row in df.iterrows():
                if row["inj_days"] == 1 and not in_injury:
                    # Start of injury period
                    in_injury = True
                    period_start = row["date"]
                elif row["inj_days"] == 0 and in_injury:
                    # End of injury period
                    injury_periods.append((period_start, row["date"] - pd.Timedelta(days=1)))
                    in_injury = False
                    period_start = None
            
            # Handle case where injury extends to end of date range
            if in_injury and period_start is not None:
                injury_periods.append((period_start, end_date))
            
            return injury_periods
    
    # Fallback: try to load from raw injury data
    try:
        data_dir = ROOT_DIR / "original_data"
        injury_files = glob.glob(str(data_dir / "*injuries*.xlsx"))
        if injury_files:
            injuries_df = pd.read_excel(injury_files[0], engine='openpyxl')
            
            # Try different possible column names for player_id
            player_id_col = None
            for col in ['player_id', 'Player ID', 'playerId', 'id']:
                if col in injuries_df.columns:
                    player_id_col = col
                    break
            
            if player_id_col is None:
                return []
            
            player_injuries = injuries_df[injuries_df[player_id_col] == player_id]
            
            if not player_injuries.empty:
                injury_periods = []
                for _, injury in player_injuries.iterrows():
                    # Try different possible column names for dates
                    from_date_col = None
                    until_date_col = None
                    
                    for col in ['fromDate', 'From', 'from', 'From Date', 'start_date']:
                        if col in injury.index:
                            from_date_col = col
                            break
                    
                    for col in ['untilDate', 'Until', 'until', 'Until Date', 'end_date']:
                        if col in injury.index:
                            until_date_col = col
                            break
                    
                    if from_date_col is None:
                        continue
                    
                    from_date = pd.to_datetime(injury[from_date_col], errors='coerce')
                    until_date = pd.to_datetime(injury[until_date_col], errors='coerce') if until_date_col else None
                    
                    if pd.notna(from_date):
                        # Handle missing until_date
                        if pd.isna(until_date):
                            # Use default recovery period (30 days for standard, 90 for severe)
                            severity_col = None
                            for col in ['injury_severity', 'Severity', 'severity']:
                                if col in injury.index:
                                    severity_col = col
                                    break
                            
                            severity = injury[severity_col] if severity_col and pd.notna(injury[severity_col]) else 1
                            if severity >= 4:
                                until_date = from_date + pd.Timedelta(days=90)
                            else:
                                until_date = from_date + pd.Timedelta(days=30)
                        
                        # Only include injuries that overlap with our date range
                        if from_date <= end_date and until_date >= start_date:
                            period_start = max(from_date, start_date)
                            period_end = min(until_date, end_date)
                            injury_periods.append((period_start, period_end))
                
                return injury_periods
    except Exception:
        pass  # If loading fails, return empty list
    
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
    """Create dashboard PNG showing RF, GB, and Ensemble predictions."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 2], hspace=0.35, wspace=0.45)

    # Top panel – risk evolution (full width) with all three models
    ax_main = fig.add_subplot(gs[0, :])
    dates = pivot["reference_date"]
    
    # Add background shading for injury periods
    if injury_periods:
        first_label = True
        for period_start, period_end in injury_periods:
            # Only shade if the period overlaps with our date range
            chart_start = dates.min()
            chart_end = dates.max()
            if period_start <= chart_end and period_end >= chart_start:
                shade_start = max(period_start, chart_start)
                shade_end = min(period_end, chart_end)
                label = 'Injury Period' if first_label else ''
                ax_main.axvspan(shade_start, shade_end, alpha=0.15, color='#FFB6C1', zorder=0, label=label)
                first_label = False
    
    # Compute risk series for all three models
    risk_df_rf = compute_risk_series(pivot["rf_probability"])
    risk_df_gb = compute_risk_series(pivot["gb_probability"])
    risk_df_ensemble = compute_risk_series(pivot["ensemble_probability"])
    
    risk_indices_rf = risk_df_rf["risk_index"] + 1
    risk_indices_gb = risk_df_gb["risk_index"] + 1
    risk_indices_ensemble = risk_df_ensemble["risk_index"] + 1

    # Model colors and styles
    model_configs = [
        ("Random Forest", risk_indices_rf, "#2E86AB", "-", 1.5),
        ("Gradient Boosting", risk_indices_gb, "#A23B72", "--", 1.5),
        ("Ensemble", risk_indices_ensemble, "#F18F01", "-", 3.0),
    ]

    if HAS_SCIPY and len(dates) > 3:
        # Smooth curves using spline interpolation
        dates_numeric = np.arange(len(dates))
        x_smooth = np.linspace(dates_numeric.min(), dates_numeric.max(), 500)
        dates_smooth = pd.date_range(dates.min(), dates.max(), periods=500)
        
        # Plot RF and GB as simple lines
        for model_name, risk_indices, color, linestyle, linewidth in model_configs[:2]:  # RF and GB only
            spl = make_interp_spline(dates_numeric, risk_indices, k=min(3, len(dates) - 1))
            y_smooth = spl(x_smooth)
            
            ax_main.plot(
                dates_smooth,
                y_smooth,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=0.7,
                zorder=2,
                label=model_name,
            )
        
        # Plot Ensemble with risk-based coloring (highlighted)
        spl_ensemble = make_interp_spline(dates_numeric, risk_indices_ensemble, k=min(3, len(dates) - 1))
        y_smooth_ensemble = spl_ensemble(x_smooth)
        
        # Map interpolated values back to risk classes for coloring
        colors_smooth = []
        for val in y_smooth_ensemble:
            idx = int(np.clip(np.round(val - 1), 0, len(RISK_CLASS_LABELS) - 1))
            label = RISK_CLASS_LABELS[idx]
            colors_smooth.append(RISK_CLASS_COLORS[label])
        
        # Plot smooth line with gradient coloring using segments
        for i in range(len(dates_smooth) - 1):
            ax_main.plot(
                dates_smooth[i : i + 2],
                y_smooth_ensemble[i : i + 2],
                color=colors_smooth[i],
                linewidth=3.5,
                alpha=0.9,
                zorder=3,
                label="Ensemble" if i == 0 else "",
            )
        
        # Add area fill under ensemble curve
        ax_main.fill_between(dates_smooth, y_smooth_ensemble, alpha=0.15, color="gray", zorder=1)
    else:
        # For very few points, plot simple lines
        for model_name, risk_indices, color, linestyle, linewidth in model_configs[:2]:  # RF and GB only
            ax_main.plot(
                dates,
                risk_indices,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=0.7,
                zorder=2,
                label=model_name,
            )
        
        # Plot Ensemble with risk-based coloring
        colors = [RISK_CLASS_COLORS[label] for label in risk_df_ensemble["risk_label"]]
        for i in range(len(dates) - 1):
            ax_main.plot(
                dates.iloc[i : i + 2],
                risk_indices_ensemble.iloc[i : i + 2],
                color=colors[i],
                linewidth=3.5,
                alpha=0.9,
                zorder=3,
                label="Ensemble" if i == 0 else "",
            )
        ax_main.fill_between(dates, risk_indices_ensemble, alpha=0.15, color="gray", zorder=1)

    ax_main.set_title(f"Injury Risk Evolution - {ctx.player_name}", fontsize=14, fontweight="bold", pad=15)
    ax_main.set_ylabel("Risk Class", fontweight="bold")
    ax_main.set_xlabel("Date", fontweight="bold")
    ax_main.set_yticks(range(1, len(RISK_CLASS_LABELS) + 1))
    ax_main.set_yticklabels(RISK_CLASS_LABELS)
    ax_main.set_ylim(0.5, len(RISK_CLASS_LABELS) + 0.5)
    ax_main.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax_main.grid(axis="y", alpha=0.3, linestyle="--")
    plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=7)

    # Bottom left – body parts
    ax_body = fig.add_subplot(gs[1, 0])
    if insights and insights.get("bodypart_rank"):
        body_rank = insights.get("bodypart_rank", [])[:3]
        labels = [item[0].replace("_", " ").title() for item in body_rank]
        probs = [item[1] * 100 for item in body_rank]
        colors_body = ["#2E86AB", "#A23B72", "#F18F01"][: len(labels)]
        bars = ax_body.barh(labels, probs, color=colors_body, alpha=0.8, edgecolor="white", linewidth=1)
        ax_body.set_xlabel("Probability (%)", fontweight="bold")
        ax_body.set_title("Body Parts at Risk", fontweight="bold", pad=10)
        ax_body.set_xlim(0, 100)
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            ax_body.text(prob + 2, i, f"{prob:.1f}%", va="center", fontweight="bold")
        ax_body.grid(axis="x", alpha=0.3, linestyle="--")
    else:
        ax_body.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_body.transAxes)
        ax_body.set_title("Body Parts at Risk", fontweight="bold")

    # Bottom center – severity (horizontal bar chart, not pie)
    ax_sev = fig.add_subplot(gs[1, 1])
    if insights and insights.get("severity_probs"):
        severity_probs = insights.get("severity_probs", {})
        severity_label = insights.get("severity_label", "Unknown")
        labels_sev = list(severity_probs.keys())
        sizes = [severity_probs[k] * 100 for k in labels_sev]
        colors_sev = ["#06A77D", "#F4A261", "#E76F51", "#264653"][: len(labels_sev)]
        bars_sev = ax_sev.barh(
            range(len(labels_sev)), sizes, color=colors_sev, alpha=0.8, edgecolor="white", linewidth=1
        )
        ax_sev.set_yticks(range(len(labels_sev)))
        # Replace underscores with spaces and format labels
        formatted_labels = []
        for l in labels_sev:
            formatted = l.replace("_", " ").title()
            # Special case: "Long Term" -> "Long term"
            if formatted == "Long Term":
                formatted = "Long term"
            formatted_labels.append(formatted)
        ax_sev.set_yticklabels(formatted_labels)
        ax_sev.set_xlabel("Probability (%)", fontweight="bold")
        formatted_severity_label = severity_label.replace("_", " ").title()
        if formatted_severity_label == "Long Term":
            formatted_severity_label = "Long term"
        ax_sev.set_title(f"Severity: {formatted_severity_label}", fontweight="bold", pad=10)
        ax_sev.set_xlim(0, 100)
        for i, (bar, size) in enumerate(zip(bars_sev, sizes)):
            ax_sev.text(size + 2, i, f"{size:.1f}%", va="center", fontweight="bold")
        ax_sev.grid(axis="x", alpha=0.3, linestyle="--")
    else:
        ax_sev.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_sev.transAxes)
        ax_sev.set_title("Severity", fontweight="bold")

    # Bottom right – alert summary
    ax_alert = fig.add_subplot(gs[1, 2])
    ax_alert.axis("off")
    if risk_df is not None and not risk_df.empty:
        peak_risk = RISK_CLASS_LABELS[int(risk_df["risk_index"].max())]
        final_risk = risk_df["risk_label"].iloc[-1]
    else:
        peak_risk = "N/A"
        final_risk = "N/A"

    sustained = trend_metrics.get("sustained_elevated_days", 0) if trend_metrics else 0
    max_jump = trend_metrics.get("max_jump", 0.0) if trend_metrics else 0.0
    slope = trend_metrics.get("slope", 0.0) if trend_metrics else 0.0

    alert_text = f"""RISK SUMMARY

Peak Risk Level: {peak_risk}
Final Risk Level: {final_risk}
Sustained Elevated: {sustained} days
Max Daily Jump: {max_jump:.3f}
Trend Slope: {slope:.4f}"""
    ax_alert.text(
        0.1,
        0.5,
        alert_text,
        transform=ax_alert.transAxes,
        fontsize=9,
        verticalalignment="center",
        family="monospace",
        fontweight="bold",
    )

    plt.suptitle(
        f"Observation Period: {ctx.window_start.strftime('%Y-%m-%d')} to {ctx.window_end.strftime('%Y-%m-%d')}",
        fontsize=11,
        y=0.98,
    )

    chart_path = output_dir / ctx.chart_filename
    plt.savefig(chart_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return chart_path


def determine_players(predictions_dir: Path, suffix: str, explicit_players: Iterable[int] | None) -> List[int]:
    if explicit_players:
        return list(explicit_players)
    pattern = predictions_dir / "ensemble" / f"player_*_predictions_{suffix}.csv"
    players = []
    for path in pattern.parent.glob(pattern.name):
        try:
            pid = int(path.stem.split("_")[1])
            players.append(pid)
        except ValueError:
            continue
    return sorted(set(players))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD) of the dashboard window")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD) of the dashboard window")
    parser.add_argument(
        "--date",
        type=str,
        help="Fallback single date (YYYY-MM-DD) if --start-date is not provided",
    )
    parser.add_argument("--players", type=int, nargs="*", help="Specific player IDs to process")
    parser.add_argument(
        "--predictions-dir",
        type=str,
        default=str(PRODUCTION_PREDICTIONS_DIR),
        help="Directory with production predictions",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PRODUCTION_DASHBOARD_DIR),
        help="Directory for per-player dashboard PNGs",
    )
    args = parser.parse_args()

    start_date, end_date, suffix = parse_date_range(args)
    predictions_dir = Path(args.predictions_dir)
    output_dir = Path(args.output_dir)

    players = determine_players(predictions_dir, suffix, args.players)
    if not players:
        print(f"❌ No prediction files found for suffix '{suffix}'.")
        return 1

    print("=" * 70)
    print("PLAYER DASHBOARDS FOR PRODUCTION PREDICTIONS")
    print("=" * 70)
    print(f"📅 Date range: {start_date.date()} to {end_date.date()}")
    print(f"🎯 Players: {len(players)}")
    print()

    # Load insight models once
    bodypart_pipeline, _ = load_pipeline(Path("models") / "insights" / "bodypart_classifier")
    severity_pipeline, _ = load_pipeline(Path("models") / "insights" / "severity_classifier")

    successes = 0
    failures = 0
    
    # Use progress bar if available
    for idx, player_id in enumerate(tqdm(players, desc="Generating dashboards", unit="player", disable=not HAS_TQDM), 1):
        try:
            if not HAS_TQDM:  # Only print if no progress bar
                print(f"\n[{idx}/{len(players)}] Processing player {player_id}...")
                sys.stdout.flush()
            
            predictions = load_prediction_file(predictions_dir, player_id, suffix)
            player_name = predictions["player_name"].iloc[0] if "player_name" in predictions.columns else f"Player {player_id}"
            
            # Use full predictions range (no clipping - show all data from start_date)
            predictions_window = predictions[
                (predictions["reference_date"] >= start_date)
                & (predictions["reference_date"] <= end_date)
            ].copy()
            if predictions_window.empty:
                # Fallback: if no data in the range, use all predictions
                predictions_window = predictions.copy()
            
            # Load injury periods for background shading
            injury_periods = load_injury_periods(player_id, start_date, end_date)
            
            feature_row = load_latest_feature_row(player_id, end_date)
            insights = predict_insights(feature_row, bodypart_pipeline, severity_pipeline)
            
            # Use ensemble for risk_df and trend_metrics
            risk_df = compute_risk_series(predictions_window["ensemble_probability"])
            trend_metrics = compute_trend_metrics(predictions_window["ensemble_probability"])
            ctx = PlayerDashboardContext(
                player_id=player_id,
                player_name=player_name,
                window_start=start_date,
                window_end=end_date,
                suffix=suffix,
            )
            
            create_player_dashboard(ctx, predictions_window, risk_df, insights, trend_metrics, output_dir, injury_periods)
            if not HAS_TQDM:
                print(f"   ✅ Dashboard generated: {ctx.chart_filename}")
            else:
                # For tqdm, update description
                tqdm.write(f"✅ Player {player_id}: {ctx.chart_filename}")
            sys.stdout.flush()
            successes += 1
        except Exception as exc:  # pylint: disable=broad-except
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

