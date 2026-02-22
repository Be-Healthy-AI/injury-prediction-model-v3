#!/usr/bin/env python3
"""
Multi-club deployment orchestrator for Premier League using lgbm_muscular_v3 model.

Runs the complete 6-step pipeline for all clubs in a country (or selected clubs).
This script uses the V3 model and runs in parallel to the V2 deployment process.

Pipeline steps:
1. Fetch raw data (once for all clubs, or skip if already done)
2. Update daily features (incremental)
3. Update timelines (incremental)
4. Generate predictions using V3 model
5. Generate dashboards
6. Generate predictions table CSV

Usage for one-by-one testing:
    python deploy_all_clubs_v3.py --clubs "Arsenal FC" --data-date 20251229 --stop-on-error

Usage for all clubs:
    python deploy_all_clubs_v3.py --data-date 20251229
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Set, Tuple

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
            # Handle Unicode encoding issues safely
            try:
                error_output = result.stderr[-500:] if len(result.stderr) > 500 else result.stderr
                print(f"[ERROR] Error output: {error_output}")
            except (UnicodeEncodeError, UnicodeDecodeError):
                # Fallback: encode to ASCII with error handling
                error_output = result.stderr[-500:] if len(result.stderr) > 500 else result.stderr
                error_output_safe = error_output.encode('ascii', errors='replace').decode('ascii')
                print(f"[ERROR] Error output (some characters replaced): {error_output_safe}")
        return False
    
    return True


def move_daily_features_pl_to_pl(
    country: str,
    player_id: int,
    from_club: str,
    to_club: str,
) -> bool:
    """Move daily_features file from from_club to to_club (PL->PL transfer). Returns True if moved or already at destination."""
    deployments_dir = PRODUCTION_ROOT / "deployments" / country
    src = deployments_dir / from_club / "daily_features" / f"player_{player_id}_daily_features.csv"
    dst_dir = deployments_dir / to_club / "daily_features"
    dst = dst_dir / f"player_{player_id}_daily_features.csv"
    if not src.exists():
        return True  # Already removed (e.g. by config sync cleanup)
    dst_dir.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(src, dst)
        src.unlink()
        print(f"  [PL->PL] Moved daily_features player_{player_id} from {from_club} to {to_club}")
        return True
    except OSError as e:
        print(f"  [ERROR] PL->PL move failed for player {player_id}: {e}")
        return False


def run_config_sync_all_clubs(
    country: str,
    clubs: List[str],
    data_date: str,
    auto_fix: bool,
    sync_transfermarkt: bool,
) -> bool:
    """Run validate_and_sync_club_config for each club. Returns True if all succeeded."""
    for club in clubs:
        cmd_args = ["--country", country, "--club", club]
        if data_date:
            cmd_args.extend(["--data-date", data_date])
        if auto_fix:
            cmd_args.append("--auto-fix")
        if sync_transfermarkt:
            cmd_args.append("--sync-transfermarkt")
        if not run_script('validate_and_sync_club_config.py', cmd_args):
            print(f"[WARNING] Config sync failed for {club} (continuing)")
    return True


def collect_pl_to_pl_moves(country: str, data_date: str) -> List[Tuple[int, str, str]]:
    """Read .sync_result_{data_date}.json and return [(player_id, from_club, to_club), ...]."""
    sync_result_path = PRODUCTION_ROOT / "deployments" / country / f".sync_result_{data_date}.json"
    if not sync_result_path.exists():
        return []
    try:
        with open(sync_result_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    clubs_data = data.get('clubs', {})
    moves = []
    for from_club, club_result in clubs_data.items():
        for p in club_result.get('pl_to_pl', []):
            player_id = p.get('player_id')
            to_club = p.get('to_club')
            if player_id is not None and to_club:
                moves.append((int(player_id), from_club, to_club))
    return moves


def run_club_pipeline(
    country: str,
    club: str,
    data_date: str = None,
    skip_fetch: bool = True,
    skip_config_sync: bool = False,
    auto_fix_configs: bool = False,
    sync_transfermarkt: bool = False,
    skip_daily_features: bool = False,
    skip_timelines: bool = False,
    skip_predictions: bool = False,
    skip_dashboards: bool = False,
    skip_table: bool = False,
    dashboard_type: str = "prob",
) -> bool:
    """
    Run the complete pipeline for a single club using V3 model.
    Returns True if all steps succeed, False otherwise.
    """
    print(f"\n{'='*80}")
    print(f"Processing: {club} (V3 Model)")
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
    
    # Step 1.5: Validate and sync club config
    if not skip_config_sync:
        print(f"\n[STEP 1.5] Validating and syncing config for {club}...")
        cmd_args = [
            "--country", country,
            "--club", club
        ]
        if data_date:
            cmd_args.extend(["--data-date", data_date])
        
        # Add auto-fix flag if enabled
        if auto_fix_configs:
            cmd_args.append("--auto-fix")
        
        # Add Transfermarkt sync if enabled
        if sync_transfermarkt:
            cmd_args.append("--sync-transfermarkt")
        
        if not run_script('validate_and_sync_club_config.py', cmd_args):
            print(f"[WARNING] {club} - Config validation failed (continuing anyway)")
        else:
            print(f"[OK] {club} - Config validated/synced")
    else:
        print(f"[SKIP] Config validation/sync")
    
    # Step 2: Update daily features
    if not skip_daily_features:
        print(f"\n[STEP 2] Updating daily features for {club}...")
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
                # Add max-date to cap generation
                try:
                    data_dt = datetime.strptime(data_date, "%Y%m%d")
                    max_date_str = data_dt.strftime("%Y-%m-%d")
                    cmd_args.extend(["--max-date", max_date_str])
                    # Full regeneration from 2025-07-01 so timelines only include players with current daily features (no legacy rows)
                    cmd_args.append("--full-regeneration")
                except ValueError:
                    pass
            else:
                # Use today's date
                cmd_args.extend(["--data-date", datetime.now().strftime("%Y%m%d")])
            
            if not run_script('update_timelines.py', cmd_args):
                print(f"[FAILED] {club} - Timelines")
                success = False
            else:
                print(f"[OK] {club} - Timelines completed")
    else:
        print(f"[SKIP] Timelines update")
    
    # Step 4: Generate predictions using V3 model
    if not skip_predictions:
        print(f"\n[STEP 4] Generating predictions for {club} (V3 model)...")
        cmd_args = [
            "--country", country,
            "--club", club
        ]
        
        # Add --force flag and --data-date to regenerate predictions if data_date is specified
        # This ensures predictions are generated for the specific date with correct filename
        if data_date:
            cmd_args.append("--force")
            cmd_args.extend(["--data-date", data_date])
        
        if not run_script('generate_predictions_lgbm_v3.py', cmd_args):
            print(f"[FAILED] {club} - Predictions (V3)")
            success = False
        else:
            print(f"[OK] {club} - Predictions completed (V3)")
    else:
        print(f"[SKIP] Predictions generation")
    
    # Step 5: Generate dashboards
    if not skip_dashboards:
        print(f"\n[STEP 5] Generating dashboards for {club} (V3)...")
        cmd_args = [
            "--country", country,
            "--club", club,
            "--date", data_date[:4] + "-" + data_date[4:6] + "-" + data_date[6:8] if data_date else datetime.now().strftime("%Y-%m-%d"),
            "--model-version", "v3"
        ]
        if dashboard_type:
            cmd_args.extend(["--dashboard-type", dashboard_type])
        if not run_script('generate_dashboards.py', cmd_args):
            print(f"[FAILED] {club} - Dashboards (V3)")
            success = False
        else:
            print(f"[OK] {club} - Dashboards completed (V3)")
    else:
        print(f"[SKIP] Dashboards generation")
    
    # Step 6: Generate predictions table (optional, per-club)
    if not skip_table:
        print(f"\n[STEP 6] Generating predictions table for {club} (V3)...")
        cmd_args = [
            "--country", country,
            "--club", club,
            "--model-version", "v3"
        ]
        if data_date:
            # Convert YYYYMMDD to YYYY-MM-DD
            date_formatted = data_date[:4] + "-" + data_date[4:6] + "-" + data_date[6:8]
            cmd_args.extend(["--date", date_formatted])
        
        if not run_script('generate_predictions_table.py', cmd_args):
            print(f"[FAILED] {club} - Predictions table (V3)")
            success = False
        else:
            print(f"[OK] {club} - Predictions table completed (V3)")
    
    return success


def main():
    parser = argparse.ArgumentParser(
        description='Run pipeline for all Premier League clubs using lgbm_muscular_v3 model'
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
    parser.add_argument(
        "--dashboard-type",
        type=str,
        choices=["prob", "index", "both"],
        default="prob",
        help='Dashboard type: prob (default), index, or both'
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
        print("SINGLE-CLUB DEPLOYMENT PIPELINE (V3 Model - Testing Mode)")
    else:
        print("MULTI-CLUB DEPLOYMENT PIPELINE (V3 Model)")
    print("=" * 80)
    print(f"Model: lgbm_muscular_v3")
    print(f"Country: {args.country}")
    print(f"Data date: {data_date}")
    print(f"Clubs to process: {len(clubs)}")
    print(f"   {', '.join(clubs)}")
    print(f"Excluded clubs: {', '.join(sorted(exclude_clubs))}")
    if "Chelsea FC" in exclude_clubs:
        print("\n[NOTE] Chelsea FC is excluded - using independent process")
    print()
    
    # Pre-step: Config sync for all clubs (before fetch so new players are in configs for raw data)
    if not args.skip_config_sync:
        print("=" * 80)
        print("PRE-STEP: Config sync for all clubs (before raw data fetch)")
        print("=" * 80)
        run_config_sync_all_clubs(
            country=args.country,
            clubs=clubs,
            data_date=data_date,
            auto_fix=args.auto_fix_configs,
            sync_transfermarkt=args.sync_transfermarkt,
        )
        # PL->PL: move daily_features files from old club to new club
        pl_to_pl_moves = collect_pl_to_pl_moves(args.country, data_date)
        if pl_to_pl_moves:
            print(f"\n[PL->PL] Moving daily_features for {len(pl_to_pl_moves)} player(s)...")
            for player_id, from_club, to_club in pl_to_pl_moves:
                move_daily_features_pl_to_pl(args.country, player_id, from_club, to_club)
        print("[OK] Pre-step config sync completed\n")
    
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
            print(f"PROCESSING: {club} (V3)")
        else:
            print(f"CLUB {idx}/{len(clubs)}: {club} (V3)")
        print(f"{'='*80}")
        
        if run_club_pipeline(
            country=args.country,
            club=club,
            data_date=data_date,
            skip_fetch=True,  # Already done above
            skip_config_sync=args.skip_config_sync,
            auto_fix_configs=args.auto_fix_configs,
            sync_transfermarkt=args.sync_transfermarkt,
            skip_daily_features=args.skip_daily_features,
            skip_timelines=args.skip_timelines,
            skip_predictions=args.skip_predictions,
            skip_dashboards=args.skip_dashboards,
            skip_table=args.skip_table,
            dashboard_type=args.dashboard_type,
        ):
            successful.append(club)
        else:
            failed.append(club)
            if args.stop_on_error:
                print(f"\n[STOPPED] Stopping due to error in {club} (--stop-on-error)")
                break
    
    # Summary
    print("\n" + "=" * 80)
    print("DEPLOYMENT SUMMARY (V3 Model)")
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

