#!/usr/bin/env python3
"""
Master deployment script for club predictions.

Orchestrates the complete pipeline:
1. Update daily features (incremental, unless --skip-features)
2. Generate timelines
3. Generate predictions
4. Generate dashboards

All steps use the production structure with country/club organization.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Calculate paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent


def run_script(script_name: str, args: list[str] = None) -> bool:
    """Run a production script and return True if successful."""
    script_path = SCRIPT_DIR / script_name
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return False
    
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    
    print(f"\n{'=' * 70}")
    print(f"Running: {script_name}")
    print(f"{'=' * 70}")
    
    result = subprocess.run(cmd, cwd=PRODUCTION_ROOT.parent)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--country',
        type=str,
        required=True,
        help='Country name (e.g., "England")'
    )
    parser.add_argument(
        '--club',
        type=str,
        required=True,
        help='Club name (e.g., "Chelsea FC")'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for prediction window (YYYY-MM-DD). Default: today'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date for prediction window (YYYY-MM-DD). Default: today'
    )
    parser.add_argument(
        '--data-date',
        type=str,
        default=None,
        help='Date folder to use for raw data (YYYYMMDD format). Defaults to latest available.'
    )
    parser.add_argument(
        '--skip-features',
        action='store_true',
        help='Skip daily features update step'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force regeneration of predictions even if they exist'
    )
    args = parser.parse_args()
    
    # Build common arguments
    common_args = [
        '--country', args.country,
        '--club', args.club,
    ]
    
    date_args = []
    if args.start_date:
        date_args.extend(['--start-date', args.start_date])
    if args.end_date:
        date_args.extend(['--end-date', args.end_date])
    
    print("=" * 70)
    print("CLUB DEPLOYMENT PIPELINE")
    print("=" * 70)
    print(f"🌍 Country: {args.country}")
    print(f" club: {args.club}")
    if args.start_date:
        print(f"📅 Start date: {args.start_date}")
    if args.end_date:
        print(f"📅 End date: {args.end_date}")
    if args.data_date:
        print(f"📂 Data date: {args.data_date}")
    print()
    
    success = True
    
    # Step 1: Update daily features (unless skipped)
    if not args.skip_features:
        print("\n" + "=" * 70)
        print("STEP 1: Update Daily Features")
        print("=" * 70)
        update_args = common_args.copy()
        if args.data_date:
            update_args.extend(['--data-date', args.data_date])
        
        if not run_script('update_daily_features.py', update_args):
            print("❌ Daily features update failed")
            success = False
            # Continue anyway - might have existing features
    else:
        print("\n⏭️  Skipping daily features update (--skip-features)")
    
    # Step 2: Generate timelines
    print("\n" + "=" * 70)
    print("STEP 2: Generate Timelines")
    print("=" * 70)
    timeline_args = common_args + date_args.copy()
    
    if not run_script('generate_timelines.py', timeline_args):
        print("❌ Timeline generation failed")
        return 1
    
    # Step 3: Generate predictions
    print("\n" + "=" * 70)
    print("STEP 3: Generate Predictions")
    print("=" * 70)
    prediction_args = common_args + date_args.copy()
    if args.force:
        prediction_args.append('--force')
    
    if not run_script('generate_predictions.py', prediction_args):
        print("❌ Prediction generation failed")
        return 1
    
    # Step 4: Generate dashboards
    print("\n" + "=" * 70)
    print("STEP 4: Generate Dashboards")
    print("=" * 70)
    dashboard_args = common_args + date_args.copy()
    
    if not run_script('generate_dashboards.py', dashboard_args):
        print("❌ Dashboard generation failed")
        return 1
    
    print("\n" + "=" * 70)
    print("✅ DEPLOYMENT COMPLETE")
    print("=" * 70)
    print(f"🌍 Country: {args.country}")
    print(f" club: {args.club}")
    print(f"\n📊 Results available in:")
    club_path = PRODUCTION_ROOT / "deployments" / args.country / args.club
    print(f"   - Predictions: {club_path / 'predictions'}")
    print(f"   - Dashboards: {club_path / 'dashboards' / 'players'}")
    print()
    
    return 0


if __name__ == "__main__":
    exit(main())



