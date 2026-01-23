#!/usr/bin/env python3
"""
Generate calibration chart showing model predictive power.

This chart demonstrates how well the model's predicted probabilities
match actual injury rates by binning predictions and comparing to
observed injury rates in the next 10 days.
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
import matplotlib.dates as mdates

# Calculate paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent
ROOT_DIR = PRODUCTION_ROOT.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


def get_all_clubs(country: str = "England") -> List[str]:
    """Get all club folders in the deployments directory."""
    deployments_dir = PRODUCTION_ROOT / "deployments" / country
    if not deployments_dir.exists():
        return []
    
    clubs = []
    for item in deployments_dir.iterdir():
        if item.is_dir() and (item / "config.json").exists():
            clubs.append(item.name)
    
    return sorted(clubs)


def load_all_predictions(
    clubs: List[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp
) -> pd.DataFrame:
    """
    Load all V3 predictions from all clubs within date range.
    Optimized to only load the most recent prediction file per club,
    since each file contains all historical reference_date values.
    """
    all_predictions = []
    
    for club in clubs:
        predictions_dir = PRODUCTION_ROOT / "deployments" / "England" / club / "predictions"
        
        # Find all V3 prediction files (e.g., predictions_lgbm_v3_20260106.csv)
        pattern = "predictions_lgbm_v3_*.csv"
        prediction_files = list(predictions_dir.glob(pattern))
        
        if not prediction_files:
            print(f"  [WARN] No V3 prediction files found for {club}")
            continue
        
        # Extract date from filename and find the most recent file
        # Filename format: predictions_lgbm_v3_YYYYMMDD.csv
        def extract_date_from_filename(file_path: Path) -> str:
            """Extract YYYYMMDD date from filename."""
            filename = file_path.stem  # e.g., "predictions_lgbm_v3_20260122"
            parts = filename.split('_')
            if len(parts) >= 4:
                return parts[-1]  # Last part should be the date
            return ""
        
        # Sort files by date (most recent first)
        files_with_dates = []
        for pred_file in prediction_files:
            date_str = extract_date_from_filename(pred_file)
            if date_str and len(date_str) == 8 and date_str.isdigit():
                try:
                    file_date = datetime.strptime(date_str, "%Y%m%d")
                    files_with_dates.append((file_date, pred_file))
                except ValueError:
                    continue
        
        if not files_with_dates:
            print(f"  [WARN] Could not extract valid dates from prediction files for {club}")
            continue
        
        # Get the most recent file
        files_with_dates.sort(key=lambda x: x[0], reverse=True)
        most_recent_date, most_recent_file = files_with_dates[0]
        
        try:
            df = pd.read_csv(most_recent_file, parse_dates=['reference_date'], low_memory=False)
            # Filter by date range
            df_filtered = df[
                (df['reference_date'] >= start_date) & 
                (df['reference_date'] <= end_date)
            ].copy()
            
            if not df_filtered.empty:
                df_filtered['club'] = club
                all_predictions.append(df_filtered)
                print(f"  Loaded {len(df_filtered)} predictions from {club} ({most_recent_file.name})")
            else:
                print(f"  [WARN] No predictions in date range for {club} ({most_recent_file.name})")
        except Exception as e:
            print(f"  [WARN] Could not load {most_recent_file}: {e}")
    
    if not all_predictions:
        raise ValueError("No predictions found for the specified date range")
    
    combined = pd.concat(all_predictions, ignore_index=True)
    print(f"\n[INFO] Total predictions loaded: {len(combined):,}")
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
    end_date: pd.Timestamp = None
):
    """Create and save the calibration chart."""
    print(f"\n[INFO] Creating calibration chart...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Extract data
    bin_centers = calibration_df['bin_center'].values
    injury_rates = calibration_df['injury_rate'].values
    avg_predicted = calibration_df['avg_predicted_prob'].values * 100  # Convert to percentage
    bin_labels = calibration_df['bin_label'].values
    observations = calibration_df['total_observations'].values
    
    # Plot actual injury rate (bars)
    bars = ax.bar(
        bin_centers,
        injury_rates,
        width=0.08,
        alpha=0.7,
        color='steelblue',
        edgecolor='navy',
        linewidth=1.5,
        label='Actual Injury Rate (%)'
    )
    
    # Plot average predicted probability (line)
    ax.plot(
        bin_centers,
        avg_predicted,
        'ro-',
        linewidth=2.5,
        markersize=10,
        label='Average Predicted Probability (%)',
        zorder=5
    )
    
    # Plot perfect calibration line (diagonal)
    perfect_line = np.linspace(0, 100, 100)
    ax.plot(
        np.linspace(0, 1, 100),
        perfect_line,
        'k--',
        linewidth=2,
        alpha=0.5,
        label='Perfect Calibration',
        zorder=1
    )
    
    # Add observation counts on bars
    for i, (bar, obs) in enumerate(zip(bars, observations)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f'n={obs:,}',
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold'
        )
    
    # Customize axes
    ax.set_xlabel('Predicted Injury Probability', fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual Injury Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title(
        f'Model Calibration Chart (Muscular Injuries Only)\n'
        f'Injury Window: {window_days} days | '
        f'Observations: {start_date.date() if start_date else "N/A"} to {end_date.date() if end_date else "N/A"}',
        fontsize=16,
        fontweight='bold',
        pad=20
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


def main():
    parser = argparse.ArgumentParser(
        description='Generate calibration chart showing model predictive power'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default='2025-12-06',
        help='Start date for observations (YYYY-MM-DD, default: 2025-12-06)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date for observations (YYYY-MM-DD, default: today - 6 days)'
    )
    parser.add_argument(
        '--window-days',
        type=int,
        default=5,
        help='Observation window in days (default: 5)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: production/deployments/England/calibration_chart_v3_YYYYMMDD.png)'
    )
    parser.add_argument(
        '--data-date',
        type=str,
        default=None,
        help='Raw data date to use for injuries (YYYYMMDD, default: latest)'
    )
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = pd.to_datetime(args.start_date)
    
    if args.end_date:
        end_date = pd.to_datetime(args.end_date)
    else:
        # Default: today - 11 days (to have full 10-day observation window)
        end_date = pd.Timestamp.now().normalize() - timedelta(days=11)
    
    print("=" * 80)
    print("MODEL CALIBRATION CHART GENERATOR (V3)")
    print("=" * 80)
    print(f"Start date: {start_date.date()}")
    print(f"End date: {end_date.date()}")
    print(f"Observation window: {args.window_days} days")
    print(f"Model version: V3 (lgbm_muscular_v3)")
    print()
    
    # Get all clubs
    clubs = get_all_clubs("England")
    print(f"[INFO] Found {len(clubs)} Premier League clubs")
    print(f"       Clubs: {', '.join(clubs)}")
    
    # Load predictions
    print(f"\n[INFO] Loading V3 predictions from all clubs...")
    predictions_df = load_all_predictions(
        clubs,
        start_date,
        end_date
    )
    
    # Load injuries
    injury_dates = load_all_injuries(data_date=args.data_date)
    
    # Calculate calibration data
    calibration_df = calculate_calibration_data(
        predictions_df,
        injury_dates,
        window_days=args.window_days
    )
    
    # Save calibration data to CSV
    output_csv = args.output.replace('.png', '.csv') if args.output else None
    if output_csv is None:
        output_csv = PRODUCTION_ROOT / "deployments" / "England" / f"calibration_data_v3_{end_date.strftime('%Y%m%d')}.csv"
    else:
        output_csv = Path(output_csv).with_suffix('.csv')
    
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    calibration_df.to_csv(output_csv, index=False)
    print(f"\n[INFO] Saved calibration data to: {output_csv}")
    
    # Create chart
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = PRODUCTION_ROOT / "deployments" / "England" / f"calibration_chart_v3_{end_date.strftime('%Y%m%d')}.png"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    create_calibration_chart(
        calibration_df,
        output_path,
        window_days=args.window_days,
        start_date=start_date,
        end_date=end_date
    )
    
    print("\n" + "=" * 80)
    print("CALIBRATION CHART GENERATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

