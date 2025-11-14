#!/usr/bin/env python3
"""Generate 4 different dashboard design options for customer presentations."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.backtests.config_utils import BacktestEntry, load_backtest_config
from scripts.backtests.insight_utils import (
    load_pipeline,
    RISK_CLASS_LABELS,
    RISK_CLASS_COLORS,
    compute_risk_series,
    compute_trend_metrics,
)
from scripts.backtests.summarize_results import (
    predict_insights,
    load_daily_feature_vector,
)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9


def load_player_data(entry_id: str, config_path: Path, predictions_dir: Path, daily_features_dir: Path):
    """Load all data needed for dashboard generation."""
    entries = load_backtest_config(config_path)
    entry = next((e for e in entries if e.entry_id == entry_id), None)
    if not entry:
        raise ValueError(f"Entry {entry_id} not found in config")

    # Load predictions
    pivot_path = predictions_dir / "gradient_boosting" / f"{entry_id}_gradient_boosting_predictions.csv"
    if not pivot_path.exists():
        raise FileNotFoundError(f"Predictions not found: {pivot_path}")
    pivot = pd.read_csv(pivot_path)
    pivot["reference_date"] = pd.to_datetime(pivot["reference_date"])
    pivot.sort_values("reference_date", inplace=True)

    # Load insights
    bodypart_pipeline, _ = load_pipeline(Path("models") / "insights" / "bodypart_classifier")
    severity_pipeline, _ = load_pipeline(Path("models") / "insights" / "severity_classifier")
    feature_row = load_daily_feature_vector(entry, daily_features_dir)
    insights = predict_insights(feature_row, bodypart_pipeline, severity_pipeline)

    # Compute risk series
    gb_series = pivot["injury_probability"]
    risk_df = compute_risk_series(gb_series)
    trend_metrics = compute_trend_metrics(gb_series)

    return entry, pivot, risk_df, insights, trend_metrics


def create_option_1_horizontal_split(
    entry: BacktestEntry,
    pivot: pd.DataFrame,
    risk_df: pd.DataFrame,
    insights: Dict,
    trend_metrics: Dict,
    output_path: Path,
):
    """Option 1: Horizontal Split Layout"""
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 2], hspace=0.3, wspace=0.3)

    # Top: Risk evolution chart (full width)
    ax_main = fig.add_subplot(gs[0, :])
    dates = pivot["reference_date"]
    risk_indices = risk_df["risk_index"] + 1
    colors = [RISK_CLASS_COLORS[label] for label in risk_df["risk_label"]]

    ax_main.bar(dates, risk_indices, color=colors, width=0.8, alpha=0.7, edgecolor="white", linewidth=0.5)
    injury_date = entry.injury_date
    has_injury_line = False
    if dates.min() <= injury_date <= dates.max():
        ax_main.axvline(injury_date, color="black", linestyle="--", linewidth=2, label="Injury Date", zorder=10)
        has_injury_line = True

    ax_main.set_title(f"Injury Risk Evolution - {pivot['player_name'].iloc[0]}", fontsize=14, fontweight="bold", pad=15)
    ax_main.set_xlabel("Date", fontweight="bold")
    ax_main.set_ylabel("Risk Class", fontweight="bold")
    ax_main.set_yticks(range(1, len(RISK_CLASS_LABELS) + 1))
    ax_main.set_yticklabels(RISK_CLASS_LABELS)
    ax_main.set_ylim(0.5, len(RISK_CLASS_LABELS) + 0.5)
    if has_injury_line:
        ax_main.legend(loc="upper left")
    ax_main.grid(axis="y", alpha=0.3, linestyle="--")
    plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Bottom Left: Body Parts
    ax_body = fig.add_subplot(gs[1, 0])
    body_rank = insights.get("bodypart_rank", [])[:3]
    if body_rank:
        labels = [item[0].replace("_", " ").title() for item in body_rank]
        probs = [item[1] * 100 for item in body_rank]
        colors_body = ["#2E86AB", "#A23B72", "#F18F01"][:len(labels)]
        bars = ax_body.barh(labels, probs, color=colors_body, alpha=0.8, edgecolor="white", linewidth=1)
        ax_body.set_xlabel("Probability (%)", fontweight="bold")
        ax_body.set_title("Most Likely Affected Body Parts", fontweight="bold", pad=10)
        ax_body.set_xlim(0, 100)
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            ax_body.text(prob + 2, i, f"{prob:.1f}%", va="center", fontweight="bold")
        ax_body.grid(axis="x", alpha=0.3, linestyle="--")
    else:
        ax_body.text(0.5, 0.5, "No body part data", ha="center", va="center", transform=ax_body.transAxes)
        ax_body.set_title("Most Likely Affected Body Parts", fontweight="bold")

    # Bottom Right: Severity
    ax_sev = fig.add_subplot(gs[1, 1])
    severity_probs = insights.get("severity_probs", {})
    severity_label = insights.get("severity_label", "Unknown")
    if severity_probs:
        labels_sev = list(severity_probs.keys())
        sizes = [severity_probs[k] * 100 for k in labels_sev]
        colors_sev = ["#06A77D", "#F4A261", "#E76F51", "#264653"][:len(labels_sev)]
        wedges, texts, autotexts = ax_sev.pie(
            sizes, labels=labels_sev, colors=colors_sev, autopct="%1.1f%%", startangle=90, textprops={"fontweight": "bold"}
        )
        ax_sev.set_title(f"Predicted Severity: {severity_label.title()}", fontweight="bold", pad=10)
    else:
        ax_sev.text(0.5, 0.5, "No severity data", ha="center", va="center", transform=ax_sev.transAxes)
        ax_sev.set_title("Predicted Severity", fontweight="bold")

    plt.suptitle(
        f"Observation Period: {entry.window_start.strftime('%Y-%m-%d')} to {entry.window_end.strftime('%Y-%m-%d')}",
        fontsize=11,
        y=0.98,
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def create_option_2_vertical_split(
    entry: BacktestEntry,
    pivot: pd.DataFrame,
    risk_df: pd.DataFrame,
    insights: Dict,
    trend_metrics: Dict,
    output_path: Path,
):
    """Option 2: Vertical Split Layout"""
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(3, 2, width_ratios=[2, 1], hspace=0.4, wspace=0.3)

    # Left: Risk evolution chart
    ax_main = fig.add_subplot(gs[:, 0])
    dates = pivot["reference_date"]
    risk_indices = risk_df["risk_index"] + 1
    colors = [RISK_CLASS_COLORS[label] for label in risk_df["risk_label"]]

    ax_main.bar(dates, risk_indices, color=colors, width=0.8, alpha=0.7, edgecolor="white", linewidth=0.5)
    injury_date = entry.injury_date
    has_injury_line = False
    if dates.min() <= injury_date <= dates.max():
        ax_main.axvline(injury_date, color="black", linestyle="--", linewidth=2, label="Injury Date", zorder=10)
        has_injury_line = True

    ax_main.set_title(f"Injury Risk Evolution - {pivot['player_name'].iloc[0]}", fontsize=14, fontweight="bold", pad=15)
    ax_main.set_xlabel("Date", fontweight="bold")
    ax_main.set_ylabel("Risk Class", fontweight="bold")
    ax_main.set_yticks(range(1, len(RISK_CLASS_LABELS) + 1))
    ax_main.set_yticklabels(RISK_CLASS_LABELS)
    ax_main.set_ylim(0.5, len(RISK_CLASS_LABELS) + 0.5)
    if has_injury_line:
        ax_main.legend(loc="upper left")
    ax_main.grid(axis="y", alpha=0.3, linestyle="--")
    plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Right Top: Body Parts
    ax_body = fig.add_subplot(gs[0, 1])
    body_rank = insights.get("bodypart_rank", [])[:3]
    if body_rank:
        labels = [item[0].replace("_", " ").title() for item in body_rank]
        probs = [item[1] * 100 for item in body_rank]
        colors_body = ["#2E86AB", "#A23B72", "#F18F01"][:len(labels)]
        bars = ax_body.barh(labels, probs, color=colors_body, alpha=0.8, edgecolor="white", linewidth=1)
        ax_body.set_xlabel("Probability (%)", fontweight="bold", fontsize=9)
        ax_body.set_title("Body Parts at Risk", fontweight="bold", pad=8, fontsize=10)
        ax_body.set_xlim(0, 100)
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            ax_body.text(prob + 2, i, f"{prob:.1f}%", va="center", fontweight="bold", fontsize=8)
        ax_body.grid(axis="x", alpha=0.3, linestyle="--")
    else:
        ax_body.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_body.transAxes, fontsize=9)
        ax_body.set_title("Body Parts at Risk", fontweight="bold", fontsize=10)

    # Right Middle: Severity Gauge
    ax_sev = fig.add_subplot(gs[1, 1])
    severity_probs = insights.get("severity_probs", {})
    severity_label = insights.get("severity_label", "Unknown")
    if severity_probs:
        labels_sev = list(severity_probs.keys())
        sizes = [severity_probs[k] * 100 for k in labels_sev]
        colors_sev = ["#06A77D", "#F4A261", "#E76F51", "#264653"][:len(labels_sev)]
        bars_sev = ax_sev.barh(range(len(labels_sev)), sizes, color=colors_sev, alpha=0.8, edgecolor="white", linewidth=1)
        ax_sev.set_yticks(range(len(labels_sev)))
        ax_sev.set_yticklabels([l.title() for l in labels_sev])
        ax_sev.set_xlabel("Probability (%)", fontweight="bold", fontsize=9)
        ax_sev.set_title(f"Severity: {severity_label.title()}", fontweight="bold", pad=8, fontsize=10)
        ax_sev.set_xlim(0, 100)
        for i, (bar, size) in enumerate(zip(bars_sev, sizes)):
            ax_sev.text(size + 2, i, f"{size:.1f}%", va="center", fontweight="bold", fontsize=8)
        ax_sev.grid(axis="x", alpha=0.3, linestyle="--")
    else:
        ax_sev.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_sev.transAxes, fontsize=9)
        ax_sev.set_title("Severity", fontweight="bold", fontsize=10)

    # Right Bottom: Key Metrics
    ax_metrics = fig.add_subplot(gs[2, 1])
    ax_metrics.axis("off")
    peak_risk = RISK_CLASS_LABELS[int(risk_df["risk_index"].max())] if not risk_df.empty else "N/A"
    final_risk = risk_df["risk_label"].iloc[-1] if not risk_df.empty else "N/A"
    sustained = trend_metrics.get("sustained_elevated_days", 0)
    max_jump = trend_metrics.get("max_jump", 0.0)

    metrics_text = f"""KEY METRICS

Peak Risk: {peak_risk}
Final Risk: {final_risk}
Sustained Elevated: {sustained} days
Max Daily Jump: {max_jump:.3f}"""
    ax_metrics.text(0.1, 0.5, metrics_text, transform=ax_metrics.transAxes, fontsize=9, verticalalignment="center", family="monospace", fontweight="bold")

    plt.suptitle(
        f"Observation Period: {entry.window_start.strftime('%Y-%m-%d')} to {entry.window_end.strftime('%Y-%m-%d')}",
        fontsize=11,
        y=0.98,
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def create_option_3_grid_layout(
    entry: BacktestEntry,
    pivot: pd.DataFrame,
    risk_df: pd.DataFrame,
    insights: Dict,
    trend_metrics: Dict,
    output_path: Path,
):
    """Option 3: Dashboard Grid Layout - Original Design with Smooth Curves"""
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 2], hspace=0.35, wspace=0.3)

    # Top: Risk evolution chart (full width) - Smooth curve
    ax_main = fig.add_subplot(gs[0, :])
    dates = pivot["reference_date"]
    risk_indices = risk_df["risk_index"] + 1
    
    # Create smooth curve using interpolation
    from scipy.interpolate import make_interp_spline
    import numpy as np
    
    # Convert dates to numeric for interpolation
    dates_numeric = np.arange(len(dates))
    # Create smooth curve
    if len(dates) > 3:
        x_smooth = np.linspace(dates_numeric.min(), dates_numeric.max(), 500)
        spl = make_interp_spline(dates_numeric, risk_indices, k=min(3, len(dates)-1))
        y_smooth = spl(x_smooth)
        dates_smooth = pd.date_range(dates.min(), dates.max(), periods=500)
        
        # Interpolate risk labels for coloring
        risk_indices_smooth = y_smooth
        # Map interpolated values back to risk classes for coloring
        colors_smooth = []
        for val in risk_indices_smooth:
            idx = int(np.clip(np.round(val - 1), 0, len(RISK_CLASS_LABELS) - 1))
            label = RISK_CLASS_LABELS[idx]
            colors_smooth.append(RISK_CLASS_COLORS[label])
        
        # Plot smooth line with gradient coloring using segments
        for i in range(len(dates_smooth) - 1):
            ax_main.plot(dates_smooth[i:i+2], risk_indices_smooth[i:i+2], 
                        color=colors_smooth[i], linewidth=3.5, alpha=0.85, zorder=3)
        
        # Add area fill under the curve
        ax_main.fill_between(dates_smooth, risk_indices_smooth, alpha=0.15, 
                            color='gray', zorder=1)
    else:
        # For very few points, just plot a simple line
        colors = [RISK_CLASS_COLORS[label] for label in risk_df["risk_label"]]
        for i in range(len(dates) - 1):
            ax_main.plot(dates[i:i+2], risk_indices[i:i+2], 
                        color=colors[i], linewidth=3.5, alpha=0.85, zorder=3)
        ax_main.fill_between(dates, risk_indices, alpha=0.15, color='gray', zorder=1)
    
    injury_date = entry.injury_date
    has_injury_line = False
    if dates.min() <= injury_date <= dates.max():
        ax_main.axvline(injury_date, color="black", linestyle="--", linewidth=2, label="Injury Date", zorder=10)
        has_injury_line = True

    ax_main.set_title(f"Injury Risk Evolution - {pivot['player_name'].iloc[0]}", fontsize=14, fontweight="bold", pad=15)
    ax_main.set_xlabel("Date", fontweight="bold")
    ax_main.set_ylabel("Risk Class", fontweight="bold")
    ax_main.set_yticks(range(1, len(RISK_CLASS_LABELS) + 1))
    ax_main.set_yticklabels(RISK_CLASS_LABELS)
    ax_main.set_ylim(0.5, len(RISK_CLASS_LABELS) + 0.5)
    if has_injury_line:
        ax_main.legend(loc="upper left")
    ax_main.grid(axis="y", alpha=0.3, linestyle="--")
    plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Bottom Left: Body Parts
    ax_body = fig.add_subplot(gs[1, 0])
    body_rank = insights.get("bodypart_rank", [])[:3]
    if body_rank:
        labels = [item[0].replace("_", " ").title() for item in body_rank]
        probs = [item[1] * 100 for item in body_rank]
        colors_body = ["#2E86AB", "#A23B72", "#F18F01"][:len(labels)]
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

    # Bottom Center: Severity
    ax_sev = fig.add_subplot(gs[1, 1])
    severity_probs = insights.get("severity_probs", {})
    severity_label = insights.get("severity_label", "Unknown")
    if severity_probs:
        labels_sev = list(severity_probs.keys())
        sizes = [severity_probs[k] * 100 for k in labels_sev]
        colors_sev = ["#06A77D", "#F4A261", "#E76F51", "#264653"][:len(labels_sev)]
        bars_sev = ax_sev.barh(range(len(labels_sev)), sizes, color=colors_sev, alpha=0.8, edgecolor="white", linewidth=1)
        ax_sev.set_yticks(range(len(labels_sev)))
        ax_sev.set_yticklabels([l.title() for l in labels_sev])
        ax_sev.set_xlabel("Probability (%)", fontweight="bold")
        ax_sev.set_title(f"Severity: {severity_label.title()}", fontweight="bold", pad=10)
        ax_sev.set_xlim(0, 100)
        for i, (bar, size) in enumerate(zip(bars_sev, sizes)):
            ax_sev.text(size + 2, i, f"{size:.1f}%", va="center", fontweight="bold")
        ax_sev.grid(axis="x", alpha=0.3, linestyle="--")
    else:
        ax_sev.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_sev.transAxes)
        ax_sev.set_title("Severity", fontweight="bold")

    # Bottom Right: Alert Summary
    ax_alert = fig.add_subplot(gs[1, 2])
    ax_alert.axis("off")
    peak_risk = RISK_CLASS_LABELS[int(risk_df["risk_index"].max())] if not risk_df.empty else "N/A"
    final_risk = risk_df["risk_label"].iloc[-1] if not risk_df.empty else "N/A"
    sustained = trend_metrics.get("sustained_elevated_days", 0)
    max_jump = trend_metrics.get("max_jump", 0.0)
    slope = trend_metrics.get("slope", 0.0)

    alert_text = f"""RISK SUMMARY

Peak Risk Level: {peak_risk}
Final Risk Level: {final_risk}
Sustained Elevated: {sustained} days
Max Daily Jump: {max_jump:.3f}
Trend Slope: {slope:.4f}"""
    ax_alert.text(0.1, 0.5, alert_text, transform=ax_alert.transAxes, fontsize=9, verticalalignment="center", family="monospace", fontweight="bold")

    plt.suptitle(
        f"Observation Period: {entry.window_start.strftime('%Y-%m-%d')} to {entry.window_end.strftime('%Y-%m-%d')}",
        fontsize=11,
        y=0.98,
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def create_option_4_card_design(
    entry: BacktestEntry,
    pivot: pd.DataFrame,
    risk_df: pd.DataFrame,
    insights: Dict,
    trend_metrics: Dict,
    output_path: Path,
):
    """Option 4: Modern Card Design"""
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(1, 3, width_ratios=[7, 2.5, 2.5], wspace=0.4)

    # Left: Main chart
    ax_main = fig.add_subplot(gs[0, 0])
    dates = pivot["reference_date"]
    risk_indices = risk_df["risk_index"] + 1
    colors = [RISK_CLASS_COLORS[label] for label in risk_df["risk_label"]]

    ax_main.bar(dates, risk_indices, color=colors, width=0.8, alpha=0.7, edgecolor="white", linewidth=0.5)
    injury_date = entry.injury_date
    has_injury_line = False
    if dates.min() <= injury_date <= dates.max():
        ax_main.axvline(injury_date, color="black", linestyle="--", linewidth=2, label="Injury Date", zorder=10)
        has_injury_line = True

    ax_main.set_title(f"Injury Risk Evolution - {pivot['player_name'].iloc[0]}", fontsize=14, fontweight="bold", pad=15)
    ax_main.set_xlabel("Date", fontweight="bold")
    ax_main.set_ylabel("Risk Class", fontweight="bold")
    ax_main.set_yticks(range(1, len(RISK_CLASS_LABELS) + 1))
    ax_main.set_yticklabels(RISK_CLASS_LABELS)
    ax_main.set_ylim(0.5, len(RISK_CLASS_LABELS) + 0.5)
    if has_injury_line:
        ax_main.legend(loc="upper left")
    ax_main.grid(axis="y", alpha=0.3, linestyle="--")
    plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Right: Card stack
    # Card 1: Body Parts
    ax_card1 = fig.add_subplot(gs[0, 1])
    ax_card1.set_facecolor("#F8F9FA")
    body_rank = insights.get("bodypart_rank", [])[:3]
    if body_rank:
        labels = [item[0].replace("_", " ").title() for item in body_rank]
        probs = [item[1] * 100 for item in body_rank]
        colors_body = ["#2E86AB", "#A23B72", "#F18F01"][:len(labels)]
        bars = ax_card1.barh(labels, probs, color=colors_body, alpha=0.8, edgecolor="white", linewidth=1.5)
        ax_card1.set_xlabel("Probability (%)", fontweight="bold", fontsize=9)
        ax_card1.set_title("Body Parts at Risk", fontweight="bold", pad=12, fontsize=11, color="#2C3E50")
        ax_card1.set_xlim(0, 100)
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            ax_card1.text(prob + 3, i, f"{prob:.1f}%", va="center", fontweight="bold", fontsize=9)
        ax_card1.grid(axis="x", alpha=0.2, linestyle="--")
    else:
        ax_card1.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_card1.transAxes, fontsize=9)
        ax_card1.set_title("Body Parts at Risk", fontweight="bold", fontsize=11)
    ax_card1.spines["top"].set_visible(False)
    ax_card1.spines["right"].set_visible(False)
    ax_card1.spines["left"].set_color("#BDC3C7")
    ax_card1.spines["bottom"].set_color("#BDC3C7")

    # Card 2: Severity
    ax_card2 = fig.add_subplot(gs[0, 2])
    ax_card2.set_facecolor("#F8F9FA")
    severity_probs = insights.get("severity_probs", {})
    severity_label = insights.get("severity_label", "Unknown")
    if severity_probs:
        labels_sev = list(severity_probs.keys())
        sizes = [severity_probs[k] * 100 for k in labels_sev]
        colors_sev = ["#06A77D", "#F4A261", "#E76F51", "#264653"][:len(labels_sev)]
        bars_sev = ax_card2.barh(range(len(labels_sev)), sizes, color=colors_sev, alpha=0.8, edgecolor="white", linewidth=1.5)
        ax_card2.set_yticks(range(len(labels_sev)))
        ax_card2.set_yticklabels([l.title() for l in labels_sev])
        ax_card2.set_xlabel("Probability (%)", fontweight="bold", fontsize=9)
        ax_card2.set_title(f"Severity: {severity_label.title()}", fontweight="bold", pad=12, fontsize=11, color="#2C3E50")
        ax_card2.set_xlim(0, 100)
        for i, (bar, size) in enumerate(zip(bars_sev, sizes)):
            ax_card2.text(size + 3, i, f"{size:.1f}%", va="center", fontweight="bold", fontsize=9)
        ax_card2.grid(axis="x", alpha=0.2, linestyle="--")
    else:
        ax_card2.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_card2.transAxes, fontsize=9)
        ax_card2.set_title("Severity", fontweight="bold", fontsize=11)
    ax_card2.spines["top"].set_visible(False)
    ax_card2.spines["right"].set_visible(False)
    ax_card2.spines["left"].set_color("#BDC3C7")
    ax_card2.spines["bottom"].set_color("#BDC3C7")

    plt.suptitle(
        f"Observation Period: {entry.window_start.strftime('%Y-%m-%d')} to {entry.window_end.strftime('%Y-%m-%d')}",
        fontsize=11,
        y=0.98,
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def main():
    # Use Alexander Bah as example
    entry_id = "player_452607_20250209"
    config_path = Path("backtests/config/players_2025_45d.json")
    predictions_dir = Path("backtests/predictions/2025_45d")
    daily_features_dir = Path("backtests/daily_features/2025_45d")
    output_dir = Path("backtests/visualizations/dashboard_options")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data for {entry_id}...")
    entry, pivot, risk_df, insights, trend_metrics = load_player_data(
        entry_id, config_path, predictions_dir, daily_features_dir
    )

    print("Generating Option 1: Horizontal Split Layout...")
    create_option_1_horizontal_split(entry, pivot, risk_df, insights, trend_metrics, output_dir / "option_1_horizontal_split.png")

    print("Generating Option 2: Vertical Split Layout...")
    create_option_2_vertical_split(entry, pivot, risk_df, insights, trend_metrics, output_dir / "option_2_vertical_split.png")

    print("Generating Option 3: Dashboard Grid Layout...")
    create_option_3_grid_layout(entry, pivot, risk_df, insights, trend_metrics, output_dir / "option_3_grid_layout.png")

    print("Generating Option 4: Modern Card Design...")
    create_option_4_card_design(entry, pivot, risk_df, insights, trend_metrics, output_dir / "option_4_card_design.png")

    print(f"\n[OK] All 4 dashboard options generated in {output_dir}")
    print("Files created:")
    print("  - option_1_horizontal_split.png")
    print("  - option_2_vertical_split.png")
    print("  - option_3_grid_layout.png")
    print("  - option_4_card_design.png")


if __name__ == "__main__":
    main()

