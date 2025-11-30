#!/usr/bin/env python3
"""
Master pipeline script for production predictions.

Orchestrates the complete pipeline:
1. Update daily features (incremental)
2. Generate timelines
3. Generate predictions
4. Generate report
5. Generate dashboard
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"


def run_command(script_name: str, args: list[str] = None) -> bool:
    """Run a script and return True if successful."""
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return False
    
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    
    print(f"\n{'=' * 70}")
    print(f"Running: {script_name}")
    print(f"{'=' * 70}")
    
    result = subprocess.run(cmd, cwd=ROOT_DIR)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--date',
        type=str,
        default=None,
        help='Target date for predictions (YYYY-MM-DD). Default: today'
    )
    parser.add_argument(
        '--skip-features',
        action='store_true',
        help='Skip daily features update step'
    )
    parser.add_argument(
        '--skip-timelines',
        action='store_true',
        help='Skip timeline generation step'
    )
    parser.add_argument(
        '--skip-predictions',
        action='store_true',
        help='Skip prediction generation step'
    )
    parser.add_argument(
        '--skip-report',
        action='store_true',
        help='Skip report generation step'
    )
    parser.add_argument(
        '--skip-dashboard',
        action='store_true',
        help='Skip dashboard generation step'
    )
    args = parser.parse_args()
    
    # Determine date
    if args.date:
        date_str = args.date.replace('-', '')
        date_arg = ['--date', args.date]
    else:
        date_str = datetime.now().strftime('%Y%m%d')
        date_arg = []
    
    print("=" * 70)
    print("PRODUCTION PREDICTIONS PIPELINE")
    print("=" * 70)
    print(f"📅 Target date: {args.date or 'today'}")
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    success = True
    
    # Step 1: Update daily features
    if not args.skip_features:
        if not run_command('update_daily_features_incremental.py'):
            print("\n❌ Daily features update failed")
            success = False
            if input("\nContinue with remaining steps? (y/n): ").lower() != 'y':
                return 1
    else:
        print("\n⏭️  Skipping daily features update")
    
    # Step 2: Generate timelines
    if not args.skip_timelines:
        timeline_args = date_arg.copy()
        if not run_command('generate_timelines_production.py', timeline_args):
            print("\n❌ Timeline generation failed")
            success = False
            if input("\nContinue with remaining steps? (y/n): ").lower() != 'y':
                return 1
    else:
        print("\n⏭️  Skipping timeline generation")
    
    # Step 3: Generate predictions
    if not args.skip_predictions:
        prediction_args = date_arg.copy()
        if not run_command('generate_predictions_production.py', prediction_args):
            print("\n❌ Prediction generation failed")
            success = False
            if input("\nContinue with remaining steps? (y/n): ").lower() != 'y':
                return 1
    else:
        print("\n⏭️  Skipping prediction generation")
    
    # Step 4: Generate report
    if not args.skip_report:
        report_args = ['--date', date_str]
        if not run_command('generate_report_production.py', report_args):
            print("\n❌ Report generation failed")
            success = False
    else:
        print("\n⏭️  Skipping report generation")
    
    # Step 5: Generate dashboard
    if not args.skip_dashboard:
        dashboard_args = ['--date', date_str]
        if not run_command('generate_dashboard_production.py', dashboard_args):
            print("\n❌ Dashboard generation failed")
            success = False
    else:
        print("\n⏭️  Skipping dashboard generation")
    
    print()
    print("=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"⏰ Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success:
        print("✅ All steps completed successfully")
        return 0
    else:
        print("⚠️  Some steps failed (see above)")
        return 1


if __name__ == "__main__":
    exit(main())

