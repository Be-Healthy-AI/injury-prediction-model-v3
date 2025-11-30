#!/usr/bin/env python3
"""
Generate 35-day timelines for production predictions.

Reads daily features from production_predictions/daily_features/ and generates
timelines for the latest dates (today and recent past) for prediction.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import pandas as pd

import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.create_35day_timelines_v3 import (
    create_windowed_features_vectorized,
    build_timeline,
    get_player_name,
    get_static_features,
)

PRODUCTION_FEATURES_DIR = ROOT_DIR / "production_predictions" / "daily_features"
PRODUCTION_TIMELINES_DIR = ROOT_DIR / "production_predictions" / "timelines"
WINDOW_SIZE_DAYS = 35
BUFFER_DAYS = WINDOW_SIZE_DAYS - 1  # 34 days


def generate_timelines_for_player(
    player_id: int,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    daily_features_file: Path,
    output_dir: Path,
) -> Path:
    """Generate timelines for a player for a date range."""
    if not daily_features_file.exists():
        raise FileNotFoundError(f"Daily features file not found: {daily_features_file}")
    
    df = pd.read_csv(daily_features_file, parse_dates=['date'])
    if 'date' not in df.columns:
        raise KeyError(f"Column 'date' not found in {daily_features_file}")
    
    df = df.sort_values('date').reset_index(drop=True)
    available_dates = set(df['date'].dropna())
    
    player_name = get_player_name(player_id)
    timelines = []
    
    # Generate timelines for each date in range
    current_date = start_date
    while current_date <= end_date:
        if current_date not in available_dates:
            current_date += pd.Timedelta(days=1)
            continue
        
        # Need 35 days of history (34 days before + current day)
        window_start = current_date - pd.Timedelta(days=BUFFER_DAYS)
        
        if window_start not in available_dates:
            # Not enough history, skip
            current_date += pd.Timedelta(days=1)
            continue
        
        # Create windowed features
        window_features = create_windowed_features_vectorized(df, window_start, current_date)
        if window_features is None:
            current_date += pd.Timedelta(days=1)
            continue
        
        # Get reference row
        ref_row = df[df['date'] == current_date].iloc[0]
        
        # Build timeline (without target column for production)
        timeline = build_timeline(
            player_id=player_id,
            player_name=player_name,
            reference_date=current_date,
            ref_row=ref_row,
            windowed_features=window_features,
            # target=None by default - not included in production timelines
        )
        timelines.append(timeline)
        
        current_date += pd.Timedelta(days=1)
    
    if not timelines:
        raise ValueError(f"No valid timelines generated for player {player_id} between {start_date.date()} and {end_date.date()}")
    
    # Save timelines
    output_dir.mkdir(parents=True, exist_ok=True)
    start_label = start_date.strftime('%Y%m%d')
    end_label = end_date.strftime('%Y%m%d')
    suffix = end_label if start_label == end_label else f"{start_label}_{end_label}"
    output_file = output_dir / f"player_{player_id}_timelines_{suffix}.csv"
    pd.DataFrame(timelines).to_csv(output_file, index=False, encoding='utf-8-sig')
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--features-dir',
        type=str,
        default=str(PRODUCTION_FEATURES_DIR),
        help='Directory containing daily features files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(PRODUCTION_TIMELINES_DIR),
        help='Directory for output timeline files'
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
        '--days-back',
        type=int,
        default=7,
        help='When --start-date is not provided, number of days back from target date (default: 7)'
    )
    parser.add_argument(
        '--players',
        type=int,
        nargs='*',
        help='Specific player IDs to process (if not provided, processes all players with daily features)'
    )
    args = parser.parse_args()
    
    features_dir = Path(args.features_dir)
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
        start_date = end_date - pd.Timedelta(days=max(args.days_back, 1) - 1)
    
    if end_date < start_date:
        raise ValueError("End date cannot be earlier than start date.")
    
    print("=" * 70)
    print("GENERATE TIMELINES FOR PRODUCTION PREDICTIONS")
    print("=" * 70)
    print(f"📂 Features directory: {features_dir}")
    print(f"📂 Output directory: {output_dir}")
    print(f"📅 Date range: {start_date.date()} to {end_date.date()}")
    print()
    
    # Find all daily features files
    if args.players:
        feature_files = []
        for pid in args.players:
            file_path = features_dir / f"player_{pid}_daily_features.csv"
            if file_path.exists():
                feature_files.append(file_path)
            else:
                print(f"⚠️  Daily features file not found for player {pid}: {file_path}")
    else:
        feature_files = list(features_dir.glob("player_*_daily_features.csv"))
    
    print(f"🎯 Processing {len(feature_files)} players...")
    print("-" * 70)
    
    successful = 0
    failed = 0
    
    for feature_file in feature_files:
        try:
            # Extract player ID from filename
            player_id = int(feature_file.stem.replace('player_', '').replace('_daily_features', ''))
            
            print(f"\n[{successful + failed + 1}/{len(feature_files)}] Player {player_id}")
            
            output_file = generate_timelines_for_player(
                player_id=player_id,
                start_date=start_date,
                end_date=end_date,
                daily_features_file=feature_file,
                output_dir=output_dir,
            )
            
            print(f"   ✅ Generated: {output_file.name}")
            successful += 1
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            failed += 1
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())

