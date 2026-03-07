#!/usr/bin/env python3
"""
Generate calibration chart(s) for V3 and/or V4 model predictions.

This script can update calibration for V3 only, V4 only, or both in one run.
Charts show how well predicted probabilities match actual injury rates by
binning predictions and comparing to observed muscular injuries in a time window.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Calculate paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent
ROOT_DIR = PRODUCTION_ROOT.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# Version identifier for paths and labels
V3 = "v3"
V4 = "v4"


def get_all_clubs(country: str = "England", version: str = V3) -> List[str]:
    """Get all club folders for the given model version."""
    if version == V3:
        deployments_dir = PRODUCTION_ROOT / "deployments" / country
    else:
        deployments_dir = PRODUCTION_ROOT / "deployments" / country / "challenger"
    if not deployments_dir.exists():
        return []
    clubs = []
    for item in deployments_dir.iterdir():
        if item.is_dir() and (item / "config.json").exists():
            clubs.append(item.name)
    return sorted(clubs)


def _extract_date_from_filename(file_path: Path) -> str:
    """Extract YYYYMMDD date from prediction filename."""
    parts = file_path.stem.split("_")
    return parts[-1] if len(parts) >= 4 else ""


def load_all_predictions(
    clubs: List[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    version: str = V3,
) -> pd.DataFrame:
    """
    Load predictions from all clubs within date range for the given version.
    Uses the most recent prediction file per club (each file has all reference_date values).
    """
    all_predictions = []
    if version == V3:
        base_dir = PRODUCTION_ROOT / "deployments" / "England"
        pattern = "predictions_lgbm_v3_*.csv"
    else:
        base_dir = PRODUCTION_ROOT / "deployments" / "England" / "challenger"
        pattern = "predictions_lgbm_v4_*.csv"

    for club in clubs:
        predictions_dir = base_dir / club / "predictions"
        prediction_files = list(predictions_dir.glob(pattern))

        if not prediction_files:
            print(f"  [WARN] No {version.upper()} prediction files found for {club}")
            continue

        files_with_dates = []
        for pred_file in prediction_files:
            date_str = _extract_date_from_filename(pred_file)
            if date_str and len(date_str) == 8 and date_str.isdigit():
                try:
                    file_date = datetime.strptime(date_str, "%Y%m%d")
                    files_with_dates.append((file_date, pred_file))
                except ValueError:
                    continue

        if not files_with_dates:
            print(f"  [WARN] Could not extract valid dates from prediction files for {club}")
            continue

        files_with_dates.sort(key=lambda x: x[0], reverse=True)
        most_recent_date, most_recent_file = files_with_dates[0]

        try:
            df = pd.read_csv(most_recent_file, parse_dates=["reference_date"], low_memory=False)
            df_filtered = df[
                (df["reference_date"] >= start_date) & (df["reference_date"] <= end_date)
            ].copy()
            if not df_filtered.empty:
                df_filtered["club"] = club
                all_predictions.append(df_filtered)
                print(f"  Loaded {len(df_filtered)} predictions from {club} ({most_recent_file.name})")
            else:
                print(f"  [WARN] No predictions in date range for {club} ({most_recent_file.name})")
        except Exception as e:
            print(f"  [WARN] Could not load {most_recent_file}: {e}")

    if not all_predictions:
        raise ValueError(f"No {version.upper()} predictions found for the specified date range")

    combined = pd.concat(all_predictions, ignore_index=True)
    print(f"\n[INFO] Total {version.upper()} predictions loaded: {len(combined):,}")
    print(f"       Date range: {combined['reference_date'].min().date()} to {combined['reference_date'].max().date()}")
    print(f"       Unique players: {combined['player_id'].nunique()}")
    print(f"       Unique clubs: {combined['club'].nunique()}")
    return combined


def load_all_injuries(data_date: str = None) -> Dict[int, set]:
    """
    Load all muscular injury dates for all players.
    Returns dict mapping player_id -> set of muscular injury dates.
    """
    if data_date is None:
        # Find latest raw data folder
        raw_data_dir = PRODUCTION_ROOT / "raw_data" / "england"
        date_folders = [d for d in raw_data_dir.iterdir() 
                       if d.is_dir() and d.name.isdigit() and len(d.name) == 8]
        if not date_folders:
            raise FileNotFoundError("No raw data folders found")
        latest_folder = max(date_folders, key=lambda x: x.name)
        injuries_file = latest_folder / "injuries_data.csv"
    else:
        injuries_file = PRODUCTION_ROOT / "raw_data" / "england" / data_date / "injuries_data.csv"
    
    if not injuries_file.exists():
        raise FileNotFoundError(f"Injuries file not found: {injuries_file}")
    
    print(f"\n[INFO] Loading muscular injuries from: {injuries_file}")
    
    # Load injuries
    injuries_df = pd.read_csv(injuries_file, sep=';', encoding='utf-8-sig', low_memory=False)
    
    # Parse dates
    if 'fromDate' in injuries_df.columns:
        # Try DD/MM/YYYY format first
        injuries_df['fromDate_parsed'] = pd.to_datetime(
            injuries_df['fromDate'], 
            format='%d/%m/%Y', 
            errors='coerce'
        )
        valid_count = injuries_df['fromDate_parsed'].notna().sum()
        
        # If that didn't work well, try auto-detect
        if valid_count < len(injuries_df) * 0.9:
            injuries_df['fromDate_parsed2'] = pd.to_datetime(
                injuries_df['fromDate'], 
                errors='coerce'
            )
            valid_count2 = injuries_df['fromDate_parsed2'].notna().sum()
            if valid_count2 > valid_count:
                injuries_df['fromDate'] = injuries_df['fromDate_parsed2']
            else:
                injuries_df['fromDate'] = injuries_df['fromDate_parsed']
        else:
            injuries_df['fromDate'] = injuries_df['fromDate_parsed']
    
    # Derive injury_class if it doesn't exist
    if 'injury_class' not in injuries_df.columns:
        print("   [WARNING] injury_class column not found, deriving from injury_type and no_physio_injury")
        # Import the derive_injury_class function from update_timelines
        from production.scripts.update_timelines import derive_injury_class
        injuries_df['injury_class'] = injuries_df.apply(
            lambda row: derive_injury_class(
                row.get('injury_type', ''),
                row.get('no_physio_injury', None)
            ),
            axis=1
        )
    
    # Filter to only muscular injuries
    injuries_df['injury_class_lower'] = injuries_df['injury_class'].astype(str).str.lower()
    muscular_injuries = injuries_df[injuries_df['injury_class_lower'] == 'muscular'].copy()
    
    print(f"   Total injuries in file: {len(injuries_df):,}")
    print(f"   Muscular injuries: {len(muscular_injuries):,}")
    
    # Show injury class distribution
    if 'injury_class' in injuries_df.columns:
        class_counts = injuries_df['injury_class'].value_counts()
        print(f"   Injury class distribution:")
        for class_name, count in class_counts.items():
            print(f"      {class_name}: {count:,}")
    
    # Create mapping: player_id -> set of muscular injury dates
    injury_dates = defaultdict(set)
    
    # Handle both 'player_id' and 'id' column names
    player_id_col = None
    if 'player_id' in muscular_injuries.columns:
        player_id_col = 'player_id'
    elif 'id' in muscular_injuries.columns:
        player_id_col = 'id'
    else:
        print(f"   [ERROR] Neither 'player_id' nor 'id' column found in injuries data")
        print(f"   Available columns: {list(muscular_injuries.columns)}")
        return {}
    
    skipped_count = 0
    all_injury_dates_list = []
    
    for _, row in muscular_injuries.iterrows():
        player_id = row.get(player_id_col)
        from_date = row.get('fromDate')
        
        if pd.notna(player_id) and pd.notna(from_date):
            try:
                injury_date = pd.Timestamp(from_date).normalize()
                injury_dates[int(player_id)].add(injury_date)
                all_injury_dates_list.append(injury_date)
            except Exception as e:
                skipped_count += 1
                continue
        else:
            skipped_count += 1
    
    print(f"   Loaded muscular injury dates for {len(injury_dates)} players")
    total_injuries = sum(len(dates) for dates in injury_dates.values())
    print(f"   Total muscular injury dates: {total_injuries:,}")
    
    if all_injury_dates_list:
        min_date = min(all_injury_dates_list)
        max_date = max(all_injury_dates_list)
        print(f"   Injury date range: {min_date.date()} to {max_date.date()}")
    
    if skipped_count > 0:
        print(f"   [WARNING] Skipped {skipped_count} injuries due to missing data")
    
    return dict(injury_dates)


def check_injury_in_window(
    player_id: int,
    prediction_date: pd.Timestamp,
    injury_dates: Dict[int, set],
    window_days: int = 10
) -> bool:
    """
    Check if player was injured within the next window_days after prediction_date.
    """
    if player_id not in injury_dates:
        return False
    
    window_end = prediction_date + timedelta(days=window_days)
    
    for injury_date in injury_dates[player_id]:
        # Check if injury occurred within the window
        if prediction_date < injury_date <= window_end:
            return True
    
    return False


def bin_probability(prob: float) -> int:
    """Bin probability into 0-9 (representing 0.0-0.1, 0.1-0.2, ..., 0.9-1.0)."""
    # Clamp to [0, 1] range
    prob = max(0.0, min(1.0, prob))
    # Bin into 0-9
    return int(prob * 10) if prob < 1.0 else 9


def calculate_calibration_data(
    predictions_df: pd.DataFrame,
    injury_dates: Dict[int, set],
    window_days: int = 10
) -> pd.DataFrame:
    """
    Calculate calibration data by binning predictions and checking injuries.
    
    Returns DataFrame with columns:
    - bin: bin index (0-9)
    - bin_label: label like "0.0-0.1"
    - bin_center: center of bin (for plotting)
    - total_observations: number of predictions in this bin
    - injuries: number of injuries in this bin
    - injury_rate: percentage of injuries
    - avg_predicted_prob: average predicted probability in this bin
    """
    print(f"\n[INFO] Calculating calibration data (checking injuries in next {window_days} days)...")
    
    # Add injury flag for each prediction
    predictions_df['injured_in_window'] = predictions_df.apply(
        lambda row: check_injury_in_window(
            row['player_id'],
            row['reference_date'],
            injury_dates,
            window_days
        ),
        axis=1
    )
    
    # Bin probabilities
    predictions_df['bin'] = predictions_df['injury_probability'].apply(bin_probability)
    
    # Calculate statistics per bin
    bin_stats = []
    for bin_idx in range(10):
        bin_data = predictions_df[predictions_df['bin'] == bin_idx]
        
        if len(bin_data) == 0:
            continue
        
        bin_start = bin_idx * 0.1
        bin_end = (bin_idx + 1) * 0.1
        bin_center = (bin_start + bin_end) / 2
        
        total_obs = len(bin_data)
        injuries = bin_data['injured_in_window'].sum()
        injury_rate = (injuries / total_obs) * 100 if total_obs > 0 else 0.0
        avg_predicted = bin_data['injury_probability'].mean()
        
        bin_stats.append({
            'bin': bin_idx,
            'bin_label': f"{bin_start:.1f}-{bin_end:.1f}",
            'bin_center': bin_center,
            'total_observations': total_obs,
            'injuries': injuries,
            'injury_rate': injury_rate,
            'avg_predicted_prob': avg_predicted
        })
    
    calibration_df = pd.DataFrame(bin_stats)
    
    print(f"\n[INFO] Calibration summary:")
    print(calibration_df.to_string(index=False))
    
    return calibration_df


def create_calibration_chart(
    calibration_df: pd.DataFrame,
    output_path: Path,
    window_days: int = 10,
    start_date: pd.Timestamp = None,
    end_date: pd.Timestamp = None,
    version_label: str = "",
):
    """Create and save the calibration chart."""
    print(f"\n[INFO] Creating calibration chart...")
    title_suffix = f" - {version_label}" if version_label else ""
    fig, ax = plt.subplots(figsize=(12, 8))

    bin_centers = calibration_df["bin_center"].values
    injury_rates = calibration_df["injury_rate"].values
    avg_predicted = calibration_df["avg_predicted_prob"].values * 100
    bin_labels = calibration_df["bin_label"].values
    observations = calibration_df["total_observations"].values

    bars = ax.bar(
        bin_centers,
        injury_rates,
        width=0.08,
        alpha=0.7,
        color="steelblue",
        edgecolor="navy",
        linewidth=1.5,
        label="Actual Injury Rate (%)",
    )
    ax.plot(
        bin_centers,
        avg_predicted,
        "ro-",
        linewidth=2.5,
        markersize=10,
        label="Average Predicted Probability (%)",
        zorder=5,
    )
    perfect_line = np.linspace(0, 100, 100)
    ax.plot(
        np.linspace(0, 1, 100),
        perfect_line,
        "k--",
        linewidth=2,
        alpha=0.5,
        label="Perfect Calibration",
        zorder=1,
    )
    for bar, obs in zip(bars, observations):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"n={obs:,}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xlabel("Predicted Injury Probability", fontsize=14, fontweight="bold")
    ax.set_ylabel("Actual Injury Rate (%)", fontsize=14, fontweight="bold")
    ax.set_title(
        f"Model Calibration Chart (Muscular Injuries Only){title_suffix}\n"
        f"Injury Window: {window_days} days | "
        f"Observations: {start_date.date() if start_date else 'N/A'} to {end_date.date() if end_date else 'N/A'}",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    
    # Set x-axis ticks and labels
    ax.set_xticks(bin_centers)
    ax.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax.set_xlim(-0.05, 1.05)
    
    # Set y-axis
    ax.set_ylim(0, max(100, max(injury_rates) * 1.15))
    ax.set_ylabel('Injury Rate (%)', fontsize=14, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_axisbelow(True)
    
    # Add legend
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    
    # Add text box with summary stats
    total_obs = observations.sum()
    total_injuries = calibration_df['injuries'].sum()
    overall_rate = (total_injuries / total_obs * 100) if total_obs > 0 else 0
    
    textstr = f'Total Observations: {total_obs:,}\n'
    textstr += f'Total Injuries: {total_injuries:,}\n'
    textstr += f'Overall Injury Rate: {overall_rate:.2f}%'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(
        0.02, 0.98,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=props
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"       Saved chart to: {output_path}")
    plt.close()


def create_dashboard_calibration_chart(
    calibration_df: pd.DataFrame,
    output_path: Path,
    window_days: int = 5,
    start_date: pd.Timestamp = None,
    end_date: pd.Timestamp = None,
    version_label: str = "",
):
    """Create a professional dashboard-style calibration chart for customer presentations."""
    print(f"\n[INFO] Creating dashboard-style calibration chart...")
    plt.style.use("default")
    fig = plt.figure(figsize=(18, 11), facecolor="none")
    ax = fig.add_subplot(111, facecolor="none")
    ax.patch.set_facecolor("none")
    ax.patch.set_alpha(0.0)

    bin_centers = calibration_df["bin_center"].values
    injury_rates = calibration_df["injury_rate"].values
    bin_labels = calibration_df["bin_label"].values
    observations = calibration_df["total_observations"].values
    num_bars = len(bin_centers)

    color_stops = [
        np.array([34, 197, 94]) / 255.0,
        np.array([132, 204, 22]) / 255.0,
        np.array([234, 179, 8]) / 255.0,
        np.array([251, 146, 60]) / 255.0,
        np.array([239, 68, 68]) / 255.0,
        np.array([185, 28, 28]) / 255.0,
    ]
    bar_colors = []
    for i in range(num_bars):
        t = i / (num_bars - 1) if num_bars > 1 else 0
        eased_t = t * t * (3.0 - 2.0 * t)
        color_idx = eased_t * (len(color_stops) - 1)
        idx_low = int(color_idx)
        idx_high = min(idx_low + 1, len(color_stops) - 1)
        blend = color_idx - idx_low
        color_rgb = color_stops[idx_low] * (1 - blend) + color_stops[idx_high] * blend
        bar_colors.append(
            "#{:02X}{:02X}{:02X}".format(
                int(color_rgb[0] * 255), int(color_rgb[1] * 255), int(color_rgb[2] * 255)
            )
        )

    shadow_offset = 0.005
    ax.bar(
        [x + shadow_offset for x in bin_centers],
        injury_rates,
        width=0.06,
        alpha=0.1,
        color="#000000",
        edgecolor="none",
        zorder=1,
    )
    bars = ax.bar(
        bin_centers,
        injury_rates,
        width=0.06,
        alpha=1.0,
        color=bar_colors,
        edgecolor="white",
        linewidth=3.5,
        zorder=3,
    )
    for bar in bars:
        bar.set_alpha(0.95)
        bar.set_capstyle("round")

    x = np.asarray(bin_centers, dtype=float)
    y = np.asarray(injury_rates, dtype=float)
    if len(x) >= 2:

        def _pchip_monotone(xi: np.ndarray, yi: np.ndarray, xq: np.ndarray) -> np.ndarray:
            xi = np.asarray(xi, dtype=float)
            yi = np.asarray(yi, dtype=float)
            xq = np.asarray(xq, dtype=float)
            n = len(xi)
            if n == 2:
                return np.interp(xq, xi, yi)
            h = np.diff(xi)
            d = np.diff(yi) / h
            m = np.zeros(n, dtype=float)
            m[0] = d[0]
            m[-1] = d[-1]
            for k in range(1, n - 1):
                if d[k - 1] == 0.0 or d[k] == 0.0 or np.sign(d[k - 1]) != np.sign(d[k]):
                    m[k] = 0.0
                else:
                    w1 = 2 * h[k] + h[k - 1]
                    w2 = h[k] + 2 * h[k - 1]
                    m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])
            yq = np.empty_like(xq, dtype=float)
            idx = np.searchsorted(xi, xq) - 1
            idx = np.clip(idx, 0, n - 2)
            x0, x1 = xi[idx], xi[idx + 1]
            y0, y1 = yi[idx], yi[idx + 1]
            m0, m1 = m[idx], m[idx + 1]
            t = (xq - x0) / (x1 - x0)
            t2, t3 = t * t, t * t * t
            h00 = 2 * t3 - 3 * t2 + 1
            h10 = t3 - 2 * t2 + t
            h01 = -2 * t3 + 3 * t2
            h11 = t3 - t2
            yq = h00 * y0 + h10 * (x1 - x0) * m0 + h01 * y1 + h11 * (x1 - x0) * m1
            return yq

        x_smooth = np.linspace(x.min(), x.max(), 400)
        y_smooth = _pchip_monotone(x, y, x_smooth)
        y_smooth = np.clip(y_smooth, 0, None)
        ax.plot(x_smooth, y_smooth, color="#38BDF8", linewidth=10, alpha=0.18, solid_capstyle="round", zorder=4)
        ax.plot(x_smooth, y_smooth, color="#F8FAFC", linewidth=3.5, alpha=0.9, solid_capstyle="round", zorder=5)
        ax.plot(x, y, linestyle="none", marker="o", markersize=5.5, markerfacecolor="#F8FAFC", markeredgecolor="#0EA5E9", markeredgewidth=1.2, alpha=0.9, zorder=6)

    ax.set_xlabel("Predicted Risk Level", fontsize=20, fontweight="600", color="#E2E8F0", labelpad=20, family="sans-serif")
    ax.set_ylabel("Actual Injury Rate (%)", fontsize=20, fontweight="600", color="#E2E8F0", labelpad=20, family="sans-serif")
    title_text = "Injury Prediction Model Performance"
    subtitle_text = (
        f"Actual Injury Rates by Predicted Risk Level | "
        f"{start_date.date() if start_date else 'N/A'} to {end_date.date() if end_date else 'N/A'}"
    )
    if version_label:
        subtitle_text += f" | {version_label}"
    ax.set_title(title_text, fontsize=24, fontweight="700", color="#F1F5F9", pad=30, family="sans-serif")
    ax.text(0.5, 0.96, subtitle_text, transform=ax.transAxes, fontsize=13, fontweight="400", color="#CBD5E1", ha="center", va="top", family="sans-serif")
    ax.set_xlim(-0.02, 1.02)
    y_max = max(injury_rates) * 1.3 if len(injury_rates) > 0 else 20
    ax.set_ylim(0, y_max)
    ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.8, color="#64748B", zorder=0, which="major")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="major", labelsize=12, width=1.5, length=5, color="#94A3B8", labelcolor="#CBD5E1", pad=8)
    ax.set_xticks(bin_centers)
    ax.set_xticklabels(bin_labels, rotation=0, ha="center", fontsize=14, fontweight="600", color="#CBD5E1", family="sans-serif")
    ax.set_yticklabels([f"{int(y)}%" for y in ax.get_yticks()], fontsize=14, fontweight="500", color="#CBD5E1", family="sans-serif")
    for spine in ax.spines.values():
        spine.set_edgecolor("#64748B")
        spine.set_linewidth(1.5)
        spine.set_visible(True)
    fig.patch.set_facecolor("none")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="none", edgecolor="none", pad_inches=0.2, transparent=True)
    plt.close()
    print(f"[INFO] Saved dashboard chart to: {output_path}")


def _run_one_version(
    version: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    window_days: int,
    injury_dates: Dict[int, set],
    data_date: str,
    output_override: str,
    dashboard: bool,
) -> None:
    """Generate calibration data and chart(s) for one model version (v3 or v4)."""
    country = "England"
    version_upper = version.upper()
    if version == V3:
        out_dir = PRODUCTION_ROOT / "deployments" / country
    else:
        out_dir = PRODUCTION_ROOT / "deployments" / country / "challenger"

    clubs = get_all_clubs(country, version=version)
    if not clubs:
        print(f"[WARN] No clubs found for {version_upper}; skipping.")
        return
    print(f"[INFO] Found {len(clubs)} clubs for {version_upper}")
    print(f"       Clubs: {', '.join(clubs)}")

    print(f"\n[INFO] Loading {version_upper} predictions from all clubs...")
    predictions_df = load_all_predictions(clubs, start_date, end_date, version=version)

    calibration_df = calculate_calibration_data(
        predictions_df, injury_dates, window_days=window_days
    )

    date_str = end_date.strftime("%Y%m%d")
    if output_override and not output_override.endswith(".png"):
        output_override = str(Path(output_override).with_suffix(".png"))
    if output_override:
        output_path = Path(output_override)
        output_csv = output_path.with_suffix(".csv")
    else:
        output_csv = out_dir / f"calibration_data_{version}_{date_str}.csv"
        output_path = out_dir / f"calibration_chart_{version}_{date_str}.png"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_df.to_csv(output_csv, index=False)
    print(f"\n[INFO] Saved calibration data to: {output_csv}")

    create_calibration_chart(
        calibration_df,
        output_path,
        window_days=window_days,
        start_date=start_date,
        end_date=end_date,
        version_label=version_upper,
    )

    if version == V4 and dashboard:
        dash_path = out_dir / f"calibration_chart_{version}_dashboard_{date_str}.png"
        create_dashboard_calibration_chart(
            calibration_df,
            dash_path,
            window_days=window_days,
            start_date=start_date,
            end_date=end_date,
            version_label=version_upper,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate calibration chart(s) for V3 and/or V4 model predictions"
    )
    parser.add_argument(
        "--version",
        type=str,
        choices=[V3, V4, "both"],
        default=V3,
        help="Which model version(s) to run: v3, v4, or both (default: v3)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2025-12-06",
        help="Start date for observations (YYYY-MM-DD, default: 2025-12-06)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for observations (YYYY-MM-DD, default: today - 11 days)",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=5,
        help="Observation window in days (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (only used when --version is v3 or v4; ignored when both)",
    )
    parser.add_argument(
        "--data-date",
        type=str,
        default=None,
        help="Raw data date to use for injuries (YYYYMMDD, default: latest)",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Also generate dashboard-style chart (only applies to V4)",
    )
    args = parser.parse_args()

    start_date = pd.to_datetime(args.start_date)
    if args.end_date:
        end_date = pd.to_datetime(args.end_date)
    else:
        end_date = pd.Timestamp.now().normalize() - timedelta(days=11)

    versions_to_run = [args.version] if args.version != "both" else [V3, V4]

    print("=" * 80)
    print("MODEL CALIBRATION CHART GENERATOR")
    print("=" * 80)
    print(f"Start date: {start_date.date()}")
    print(f"End date: {end_date.date()}")
    print(f"Observation window: {args.window_days} days")
    print(f"Version(s): {', '.join(versions_to_run)}")
    if args.dashboard and V4 in versions_to_run:
        print("Dashboard chart: yes (V4)")
    print()

    injury_dates = load_all_injuries(data_date=args.data_date)

    for ver in versions_to_run:
        print("\n" + "=" * 80)
        print(f"CALIBRATION: {ver.upper()}")
        print("=" * 80)
        use_output = args.output if len(versions_to_run) == 1 else None
        try:
            _run_one_version(
                version=ver,
                start_date=start_date,
                end_date=end_date,
                window_days=args.window_days,
                injury_dates=injury_dates,
                data_date=args.data_date or "",
                output_override=use_output or "",
                dashboard=args.dashboard,
            )
        except ValueError as e:
            print(f"[ERROR] {ver.upper()}: {e}")
            if args.version == "both":
                print(f"       Continuing with other version(s).")
            else:
                raise

    print("\n" + "=" * 80)
    print("CALIBRATION CHART GENERATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

