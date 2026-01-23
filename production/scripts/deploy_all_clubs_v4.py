#!/usr/bin/env python3
"""
Multi-club deployment orchestrator for Premier League using lgbm_muscular_v4 model.

Runs the complete 7-step pipeline for all clubs in a country (or selected clubs).
This script uses the V4 model and runs in parallel to the V3 deployment process.

Pipeline steps:
1. Fetch raw data (once for all clubs, or skip if already done)
2. Validate and sync club configs (write to challenger/{club}/)
3. Update daily features - Layer 1 (V4-specific)
4. Enrich daily features - Layer 2 (V4-specific)
5. Update timelines (V4-specific)
6. Generate predictions using V4 model
7. Generate dashboards
8. Generate predictions table CSV (optional)

Usage for one-by-one testing:
    python deploy_all_clubs_v4.py --clubs "Arsenal FC" --data-date 20260122 --stop-on-error

Usage for all clubs:
    python deploy_all_clubs_v4.py --data-date 20260122
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
    Get all club folders in the challenger directory that have config.json files.
    Excludes clubs in exclude_clubs set.
    """
    if exclude_clubs is None:
        exclude_clubs = set()
    
    challenger_dir = PRODUCTION_ROOT / "deployments" / country / "challenger"
    if not challenger_dir.exists():
        return []
    
    clubs = []
    for item in challenger_dir.iterdir():
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
    
    # Capture output for better error reporting
    result = subprocess.run(
        cmd, 
        cwd=cwd, 
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    
    if result.returncode != 0:
        print(f"[ERROR] Script {script_name} failed with return code {result.returncode}")
        if result.stderr:
            # Print last 500 chars of stderr for context
            try:
                error_output = result.stderr[-500:] if len(result.stderr) > 500 else result.stderr
                print(f"[ERROR] Error output: {error_output}")
            except (UnicodeEncodeError, UnicodeDecodeError):
                error_output = result.stderr[-500:] if len(result.stderr) > 500 else result.stderr
                error_output_safe = error_output.encode('ascii', errors='replace').decode('ascii')
                print(f"[ERROR] Error output (some characters replaced): {error_output_safe}")
        return False
    
    return True


def run_club_pipeline(
    country: str,
    club: str,
    data_date: str = None,
    skip_fetch: bool = True,
    skip_config_sync: bool = False,
    auto_fix_configs: bool = False,
    sync_transfermarkt: bool = False,
    skip_daily_features_layer1: bool = False,
    skip_daily_features_layer2: bool = False,
    skip_timelines: bool = False,
    skip_predictions: bool = False,
    skip_dashboards: bool = False,
    skip_table: bool = False
) -> bool:
    """
    Run the complete pipeline for a single club using V4 model.
    Returns True if all steps succeed, False otherwise.
    """
    print(f"\n{'='*80}")
    print(f"Processing: {club} (V4 Model)")
    print(f"{'='*80}")
    
    success = True
    
    # Step 1: Fetch raw data (typically done once for all clubs)
    if not skip_fetch:
        print(f"\n[STEP 1] Fetching raw data for {club}...")
        print(f"[SKIP] Raw data fetch - should be done once for all clubs")
    else:
        print(f"[SKIP] Raw data fetch (--skip-fetch)")
    
    # Step 1.5: Validate and sync club config (write to challenger/{club}/)
    if not skip_config_sync:
        print(f"\n[STEP 1.5] Validating and syncing config for {club} (challenger)...")
        # Note: We need to modify validate_and_sync_club_config.py to support challenger path
        # For now, we'll call it and it should handle challenger path via --challenger flag
        # If that flag doesn't exist, we'll need to adapt the script
        cmd_args = [
            "--country", country,
            "--club", club,
            "--challenger"  # New flag to write to challenger path
        ]
        if data_date:
            cmd_args.extend(["--data-date", data_date])
        
        if auto_fix_configs:
            cmd_args.append("--auto-fix")
        
        if sync_transfermarkt:
            cmd_args.append("--sync-transfermarkt")
        
        # TODO: Update validate_and_sync_club_config.py to support --challenger flag
        # For now, we'll skip this step or create a wrapper
        print(f"[NOTE] Config sync for challenger - may need script update")
        # if not run_script('validate_and_sync_club_config.py', cmd_args):
        #     print(f"[WARNING] {club} - Config validation failed (continuing anyway)")
        # else:
        #     print(f"[OK] {club} - Config validated/synced")
    else:
        print(f"[SKIP] Config validation/sync")
    
    # Step 2: Update daily features - Layer 1
    if not skip_daily_features_layer1:
        print(f"\n[STEP 2] Updating daily features - Layer 1 for {club}...")
        cmd_args = [
            "--country", country,
            "--club", club
        ]
        if data_date:
            cmd_args.extend(["--data-date", data_date])
            # Add max-date to cap generation
            try:
                data_dt = datetime.strptime(data_date, "%Y%m%d")
                max_date_str = data_dt.strftime("%Y-%m-%d")
                cmd_args.extend(["--max-date", max_date_str])
            except ValueError:
                pass
        
        if not run_script('update_daily_features_v4.py', cmd_args):
            print(f"[FAILED] {club} - Daily features Layer 1")
            success = False
        else:
            print(f"[OK] {club} - Daily features Layer 1 completed")
    else:
        print(f"[SKIP] Daily features Layer 1 update")
    
    # Step 3: Enrich daily features - Layer 2
    if not skip_daily_features_layer2:
        print(f"\n[STEP 3] Enriching daily features - Layer 2 for {club}...")
        cmd_args = [
            "--country", country,
            "--club", club
        ]
        
        if not run_script('enrich_daily_features_v4.py', cmd_args):
            print(f"[FAILED] {club} - Daily features Layer 2")
            success = False
        else:
            print(f"[OK] {club} - Daily features Layer 2 completed")
    else:
        print(f"[SKIP] Daily features Layer 2 enrichment")
    
    # Step 4: Update timelines
    if not skip_timelines:
        print(f"\n[STEP 4] Updating timelines for {club}...")
        challenger_path = PRODUCTION_ROOT / "deployments" / country / "challenger" / club
        config_path = challenger_path / "config.json"
        daily_features_dir = challenger_path / "daily_features"
        output_dir = challenger_path / "timelines"
        
        if not config_path.exists():
            print(f"[ERROR] Config file not found: {config_path}")
            success = False
        else:
            cmd_args = [
                "--country", country,
                "--club", club
            ]
            if data_date:
                cmd_args.extend(["--data-date", data_date])
                # Add max-date to cap generation
                try:
                    data_dt = datetime.strptime(data_date, "%Y%m%d")
                    max_date_str = data_dt.strftime("%Y-%m-%d")
                    cmd_args.extend(["--max-date", max_date_str])
                    # Regenerate from day before data_date
                    prev_day = data_dt - timedelta(days=1)
                    regenerate_date = prev_day.strftime("%Y-%m-%d")
                    cmd_args.extend(["--regenerate-from-date", regenerate_date])
                except ValueError:
                    pass
            
            if not run_script('update_timelines_v4.py', cmd_args):
                print(f"[FAILED] {club} - Timelines")
                success = False
            else:
                print(f"[OK] {club} - Timelines completed")
    else:
        print(f"[SKIP] Timelines update")
    
    # Step 5: Generate predictions using V4 model
    if not skip_predictions:
        print(f"\n[STEP 5] Generating predictions for {club} (V4 model)...")
        cmd_args = [
            "--country", country,
            "--club", club
        ]
        
        # Add --force flag and --data-date to regenerate predictions if data_date is specified
        if data_date:
            cmd_args.append("--force")
            cmd_args.extend(["--data-date", data_date])
        
        if not run_script('generate_predictions_lgbm_v4.py', cmd_args):
            print(f"[FAILED] {club} - Predictions (V4)")
            success = False
        else:
            print(f"[OK] {club} - Predictions completed (V4)")
    else:
        print(f"[SKIP] Predictions generation")
    
    # Step 6: Generate dashboards
    if not skip_dashboards:
        print(f"\n[STEP 6] Generating dashboards for {club} (V4)...")
        cmd_args = [
            "--country", country,
            "--club", club,
            "--date", data_date[:4] + "-" + data_date[4:6] + "-" + data_date[6:8] if data_date else datetime.now().strftime("%Y-%m-%d"),
        ]
        
        if not run_script('generate_dashboards_v4.py', cmd_args):
            print(f"[FAILED] {club} - Dashboards (V4)")
            success = False
        else:
            print(f"[OK] {club} - Dashboards completed (V4)")
    else:
        print(f"[SKIP] Dashboards generation")
    
    # Step 7: Generate predictions table (optional, per-club)
    if not skip_table:
        print(f"\n[STEP 7] Generating predictions table for {club} (V4)...")
        # Note: generate_predictions_table.py may need V4 adaptation
        # For now, skip or adapt as needed
        print(f"[SKIP] Predictions table generation (V4 version not yet implemented)")
    
    return success


def main():
    parser = argparse.ArgumentParser(
        description='Run pipeline for all Premier League clubs using lgbm_muscular_v4 model'
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
        default="",
        help='Comma-separated list of clubs to exclude (default: none - all clubs processed)'
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
        default=False,
        help='Skip raw data fetch (default: False, will fetch if not specified)'
    )
    parser.add_argument(
        "--skip-config-sync",
        action='store_true',
        default=False,
        help='Skip config validation and sync (default: False, will validate/sync configs)'
    )
    parser.add_argument(
        "--auto-fix-configs",
        action='store_true',
        default=False,
        help='Automatically fix config.json files by adding/removing players (default: False, report only)'
    )
    parser.add_argument(
        "--sync-transfermarkt",
        action='store_true',
        default=False,
        help='Sync with Transfermarkt to detect transfers (slower but more accurate, default: False)'
    )
    parser.add_argument(
        "--skip-daily-features-layer1",
        action='store_true',
        help='Skip daily features Layer 1 update'
    )
    parser.add_argument(
        "--skip-daily-features-layer2",
        action='store_true',
        help='Skip daily features Layer 2 enrichment'
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
    
    # Get clubs to process (from challenger directory)
    if args.clubs:
        clubs = [c.strip() for c in args.clubs.split(",")]
        # Remove excluded clubs
        clubs = [c for c in clubs if c not in exclude_clubs]
    else:
        clubs = get_all_clubs(args.country, exclude_clubs)
    
    if not clubs:
        print("ERROR: No clubs found to process in challenger directory")
        print("       Run initialize_challenger_structure.py first to set up challenger structure")
        return 1
    
    # Determine data date
    data_date = args.data_date or datetime.now().strftime("%Y%m%d")
    
    print("=" * 80)
    if len(clubs) == 1:
        print("SINGLE-CLUB DEPLOYMENT PIPELINE (V4 Model - Testing Mode)")
    else:
        print("MULTI-CLUB DEPLOYMENT PIPELINE (V4 Model)")
    print("=" * 80)
    print(f"Model: lgbm_muscular_v4 (580 features)")
    print(f"Country: {args.country}")
    print(f"Data date: {data_date}")
    print(f"Clubs to process: {len(clubs)}")
    print(f"   {', '.join(clubs)}")
    if exclude_clubs:
        print(f"Excluded clubs: {', '.join(sorted(exclude_clubs))}")
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
            "--as-of-date", data_date,
            "--resume"  # Skip already-fetched match files and continue with missing ones
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
            print(f"PROCESSING: {club} (V4)")
        else:
            print(f"CLUB {idx}/{len(clubs)}: {club} (V4)")
        print(f"{'='*80}")
        
        if run_club_pipeline(
            country=args.country,
            club=club,
            data_date=data_date,
            skip_fetch=True,  # Already done above
            skip_config_sync=args.skip_config_sync,
            auto_fix_configs=args.auto_fix_configs,
            sync_transfermarkt=args.sync_transfermarkt,
            skip_daily_features_layer1=args.skip_daily_features_layer1,
            skip_daily_features_layer2=args.skip_daily_features_layer2,
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
    print("DEPLOYMENT SUMMARY (V4 Model)")
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
