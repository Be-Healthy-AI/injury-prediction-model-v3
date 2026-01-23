#!/usr/bin/env python3
"""
Multi-club deployment orchestrator for Premier League.

Runs the complete 6-step pipeline for all clubs in a country (or selected clubs).
This script runs in parallel to the existing Chelsea FC process and does not interfere with it.

Pipeline steps:
1. Fetch raw data (once for all clubs, or skip if already done)
2. Update daily features (incremental)
3. Update timelines (incremental)
4. Generate predictions
5. Generate dashboards
6. Generate predictions table CSV

Usage for one-by-one testing:
    python deploy_all_clubs.py --clubs "Arsenal FC" --data-date 20251229 --stop-on-error

Usage for all clubs:
    python deploy_all_clubs.py --data-date 20251229
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Set

# Calculate paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent
ROOT_DIR = PRODUCTION_ROOT.parent


def get_all_clubs(country: str, exclude_clubs: Set[str] = None) -> List[str]:
    """
    Get all club folders in the deployments directory that have config.json files.
    Excludes clubs in exclude_clubs set.
    """
    if exclude_clubs is None:
        exclude_clubs = set()
    
    deployments_dir = PRODUCTION_ROOT / "deployments" / country
    if not deployments_dir.exists():
        return []
    
    clubs = []
    for item in deployments_dir.iterdir():
        if item.is_dir() and (item / "config.json").exists():
            club_name = item.name
            if club_name not in exclude_clubs:
                clubs.append(club_name)
    
    return sorted(clubs)


def run_script(script_name: str, args: List[str] = None, cwd: Path = None) -> bool:
    """Run a production script and return True if successful."""
    script_path = SCRIPT_DIR / script_name
    if not script_path.exists():
        print(f"[ERROR] Script not found: {script_path}")
        return False
    
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    
    if cwd is None:
        cwd = PRODUCTION_ROOT.parent
    
    result = subprocess.run(cmd, cwd=cwd, capture_output=False)
    return result.returncode == 0


def run_club_pipeline(
    country: str,
    club: str,
    data_date: str = None,
    skip_fetch: bool = True,
    skip_daily_features: bool = False,
    skip_timelines: bool = False,
    skip_predictions: bool = False,
    skip_dashboards: bool = False,
    skip_table: bool = False
) -> bool:
    """
    Run the complete pipeline for a single club.
    Returns True if all steps succeed, False otherwise.
    """
    print(f"\n{'='*80}")
    print(f"Processing: {club}")
    print(f"{'='*80}")
    
    success = True
    
    # Step 1: Fetch raw data (typically done once for all clubs)
    if not skip_fetch:
        print(f"\n[STEP 1] Fetching raw data for {club}...")
        # Note: fetch_raw_data.py can handle multiple clubs, so this step
        # is usually done once before processing all clubs
        print(f"[SKIP] Raw data fetch - should be done once for all clubs")
    else:
        print(f"[SKIP] Raw data fetch (--skip-fetch)")
    
    # Step 2: Update daily features
    if not skip_daily_features:
        print(f"\n[STEP 2] Updating daily features for {club}...")
        cmd_args = [
            "--country", country,
            "--club", club
        ]
        if data_date:
            cmd_args.extend(["--data-date", data_date])
        
        if not run_script('update_daily_features.py', cmd_args):
            print(f"[FAILED] {club} - Daily features")
            success = False
        else:
            print(f"[OK] {club} - Daily features completed")
    else:
        print(f"[SKIP] Daily features update")
    
    # Step 3: Update timelines
    if not skip_timelines:
        print(f"\n[STEP 3] Updating timelines for {club}...")
        config_path = PRODUCTION_ROOT / "deployments" / country / club / "config.json"
        daily_features_dir = PRODUCTION_ROOT / "deployments" / country / club / "daily_features"
        output_dir = PRODUCTION_ROOT / "deployments" / country / club / "timelines"
        
        if not config_path.exists():
            print(f"[ERROR] Config file not found: {config_path}")
            success = False
        else:
            cmd_args = [
                "--config", str(config_path),
                "--daily-features-dir", str(daily_features_dir),
                "--output-dir", str(output_dir)
            ]
            if data_date:
                cmd_args.extend(["--data-date", data_date])
            else:
                # Use today's date
                cmd_args.extend(["--data-date", datetime.now().strftime("%Y%m%d")])
            
            # Determine regenerate-from-date (day before data_date)
            if data_date:
                try:
                    data_dt = datetime.strptime(data_date, "%Y%m%d")
                    # Regenerate from day before data_date
                    prev_day = data_dt - timedelta(days=1)
                    regenerate_date = prev_day.strftime("%Y-%m-%d")
                    cmd_args.extend(["--regenerate-from-date", regenerate_date])
                except ValueError:
                    pass
            
            if not run_script('update_timelines.py', cmd_args):
                print(f"[FAILED] {club} - Timelines")
                success = False
            else:
                print(f"[OK] {club} - Timelines completed")
    else:
        print(f"[SKIP] Timelines update")
    
    # Step 4: Generate predictions
    if not skip_predictions:
        print(f"\n[STEP 4] Generating predictions for {club}...")
        cmd_args = [
            "--country", country,
            "--club", club
        ]
        
        if not run_script('generate_predictions_lgbm_v2.py', cmd_args):
            print(f"[FAILED] {club} - Predictions")
            success = False
        else:
            print(f"[OK] {club} - Predictions completed")
    else:
        print(f"[SKIP] Predictions generation")
    
    # Step 5: Generate dashboards
    if not skip_dashboards:
        print(f"\n[STEP 5] Generating dashboards for {club}...")
        cmd_args = [
            "--country", country,
            "--club", club,
            "--date", data_date[:4] + "-" + data_date[4:6] + "-" + data_date[6:8] if data_date else datetime.now().strftime("%Y-%m-%d")
        ]
        
        if not run_script('generate_dashboards.py', cmd_args):
            print(f"[FAILED] {club} - Dashboards")
            success = False
        else:
            print(f"[OK] {club} - Dashboards completed")
    else:
        print(f"[SKIP] Dashboards generation")
    
    # Step 6: Generate predictions table (optional, per-club)
    if not skip_table:
        print(f"\n[STEP 6] Generating predictions table for {club}...")
        cmd_args = [
            "--country", country,
            "--club", club
        ]
        if data_date:
            # Convert YYYYMMDD to YYYY-MM-DD
            date_formatted = data_date[:4] + "-" + data_date[4:6] + "-" + data_date[6:8]
            cmd_args.extend(["--date", date_formatted])
        
        if not run_script('generate_predictions_table.py', cmd_args):
            print(f"[FAILED] {club} - Predictions table")
            success = False
        else:
            print(f"[OK] {club} - Predictions table completed")
    
    return success


def main():
    parser = argparse.ArgumentParser(
        description='Run pipeline for all Premier League clubs (excludes Chelsea FC)'
    )
    parser.add_argument(
        "--country",
        type=str,
        default="England",
        help='Country name (default: "England")'
    )
    parser.add_argument(
        "--clubs",
        type=str,
        default=None,
        help='Comma-separated list of clubs to process. Use for one-by-one testing (e.g., "Arsenal FC"). Default: all clubs except excluded ones'
    )
    parser.add_argument(
        "--exclude-clubs",
        type=str,
        default="Chelsea FC",
        help='Comma-separated list of clubs to exclude (default: "Chelsea FC")'
    )
    parser.add_argument(
        "--data-date",
        type=str,
        default=None,
        help='Data date (YYYYMMDD format, defaults to today)'
    )
    parser.add_argument(
        "--skip-fetch",
        action='store_true',
        default=True,
        help='Skip raw data fetch (default: True, should be done once for all clubs)'
    )
    parser.add_argument(
        "--skip-daily-features",
        action='store_true',
        help='Skip daily features update'
    )
    parser.add_argument(
        "--skip-timelines",
        action='store_true',
        help='Skip timelines update'
    )
    parser.add_argument(
        "--skip-predictions",
        action='store_true',
        help='Skip predictions generation'
    )
    parser.add_argument(
        "--skip-dashboards",
        action='store_true',
        help='Skip dashboards generation'
    )
    parser.add_argument(
        "--skip-table",
        action='store_true',
        help='Skip predictions table generation'
    )
    parser.add_argument(
        "--stop-on-error",
        action='store_true',
        help='Stop processing if a club fails (recommended for one-by-one testing). Default: continue with other clubs'
    )
    
    args = parser.parse_args()
    
    # Parse exclude clubs
    exclude_clubs = {c.strip() for c in args.exclude_clubs.split(',') if c.strip()}
    
    # Get clubs to process
    if args.clubs:
        clubs = [c.strip() for c in args.clubs.split(",")]
        # Remove excluded clubs
        clubs = [c for c in clubs if c not in exclude_clubs]
    else:
        clubs = get_all_clubs(args.country, exclude_clubs)
    
    if not clubs:
        print("ERROR: No clubs found to process")
        return 1
    
    # Determine data date
    data_date = args.data_date or datetime.now().strftime("%Y%m%d")
    
    print("=" * 80)
    if len(clubs) == 1:
        print("SINGLE-CLUB DEPLOYMENT PIPELINE (Testing Mode)")
    else:
        print("MULTI-CLUB DEPLOYMENT PIPELINE")
    print("=" * 80)
    print(f"Country: {args.country}")
    print(f"Data date: {data_date}")
    print(f"Clubs to process: {len(clubs)}")
    print(f"   {', '.join(clubs)}")
    print(f"Excluded clubs: {', '.join(sorted(exclude_clubs))}")
    if "Chelsea FC" in exclude_clubs:
        print("\n[NOTE] Chelsea FC is excluded - using independent process")
    print()
    
    # Step 0: Fetch raw data for all clubs (once, if not skipped)
    if not args.skip_fetch:
        print("=" * 80)
        print("STEP 0: Fetching raw data for all clubs")
        print("=" * 80)
        cmd_args = [
            "--country", args.country,
            "--league", "Premier League",
            "--competition-id", "GB1",
            "--competition-slug", "premier-league",
            "--as-of-date", data_date
        ]
        if args.clubs:
            # If specific clubs requested, filter to those
            cmd_args.extend(["--clubs", ",".join(clubs)])
        
        if not run_script('fetch_raw_data.py', cmd_args):
            print("[FAILED] Raw data fetch failed")
            if args.stop_on_error:
                return 1
        else:
            print("[OK] Raw data fetch completed")
    
    # Process each club
    successful = []
    failed = []
    
    for idx, club in enumerate(clubs, 1):
        print(f"\n{'='*80}")
        if len(clubs) == 1:
            print(f"PROCESSING: {club}")
        else:
            print(f"CLUB {idx}/{len(clubs)}: {club}")
        print(f"{'='*80}")
        
        if run_club_pipeline(
            country=args.country,
            club=club,
            data_date=data_date,
            skip_fetch=True,  # Already done above
            skip_daily_features=args.skip_daily_features,
            skip_timelines=args.skip_timelines,
            skip_predictions=args.skip_predictions,
            skip_dashboards=args.skip_dashboards,
            skip_table=args.skip_table
        ):
            successful.append(club)
        else:
            failed.append(club)
            if args.stop_on_error:
                print(f"\n[STOPPED] Stopping due to error in {club} (--stop-on-error)")
                break
    
    # Summary
    print("\n" + "=" * 80)
    print("DEPLOYMENT SUMMARY")
    print("=" * 80)
    print(f"[OK] Successful: {len(successful)}/{len(clubs)}")
    if successful:
        print(f"   {', '.join(successful)}")
    print(f"[FAILED] Failed: {len(failed)}/{len(clubs)}")
    if failed:
        print(f"   {', '.join(failed)}")
    print()
    
    if failed:
        return 1
    return 0


if __name__ == "__main__":
    exit(main())

