#!/usr/bin/env python3
"""
35-Day Timeline Generator - Version 4 - Production Wrapper

This script wraps the V4 timeline generator for production use.
It adapts the V4 training script to work with the production directory structure
and generates a single timeline file per club (combining all seasons).

Key features:
- Reads from production/deployments/England/challenger/{club}/daily_features/
- Writes to production/deployments/England/challenger/{club}/timelines/
- Prefers enriched features (Layer 2), falls back to Layer 1
- Supports incremental updates
- Handles date capping (max-date parameter)
- Processes one club at a time
- Uses config.json to get player list
- Generates single timeline file per club (combines all seasons)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Set, Tuple

import pandas as pd

# Fix encoding issues on Windows
if sys.platform == 'win32':
    import io
    if not isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Calculate paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent
ROOT_DIR = PRODUCTION_ROOT.parent

# Add paths for V4 imports
V4_CODE_DIR = ROOT_DIR / "models_production" / "lgbm_muscular_v4" / "code" / "timelines"
if str(V4_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(V4_CODE_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import V4 timeline functions
try:
    import importlib.util
    v4_timeline_module_path = V4_CODE_DIR / "create_35day_timelines_v4_enhanced.py"
    spec = importlib.util.spec_from_file_location("create_35day_timelines_v4_enhanced", v4_timeline_module_path)
    v4_timeline_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v4_timeline_module)
    
    # Get functions we need
    process_season_v4 = v4_timeline_module.process_season
    get_all_seasons_from_daily_features_v4 = v4_timeline_module.get_all_seasons_from_daily_features
    build_pl_clubs_per_season_v4 = v4_timeline_module.build_pl_clubs_per_season
    build_player_pl_membership_periods_v4 = v4_timeline_module.build_player_pl_membership_periods
    load_injuries_data_v4 = v4_timeline_module.load_injuries_data
    load_all_injury_dates_v4 = v4_timeline_module.load_all_injury_dates
    load_player_names_mapping_v4 = v4_timeline_module.load_player_names_mapping
    get_season_date_range_v4 = v4_timeline_module.get_season_date_range
    filter_timelines_pl_only_v4 = v4_timeline_module.filter_timelines_pl_only
    
except Exception as e:
    print(f"[ERROR] Failed to import V4 timeline module: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    format_str = '%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s' if verbose else '%(asctime)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )
    return logging.getLogger(__name__)


def get_challenger_path(country: str, club: str) -> Path:
    """Get base path for challenger club."""
    return PRODUCTION_ROOT / "deployments" / country / "challenger" / club


def get_raw_data_path(country: str, data_date: str) -> Path:
    """Get raw data path (shared with V3)."""
    country_lower = country.lower().replace(" ", "_")
    return PRODUCTION_ROOT / "raw_data" / country_lower / data_date


def load_player_ids_from_config(config_path: Path) -> List[int]:
    """Load player IDs from deployment config.json."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        player_ids = config.get('player_ids', [])
        return player_ids
    except Exception as e:
        logging.getLogger(__name__).error(f"Error loading config from {config_path}: {e}")
        return []


def main():
    """Main entry point - Production version."""
    parser = argparse.ArgumentParser(description='Update timelines (V4) - Production')
    parser.add_argument('--country', type=str, default='England', help='Country name (default: England)')
    parser.add_argument('--club', type=str, required=True, help='Club name (required)')
    parser.add_argument('--data-date', type=str, default=None, help='Raw data date (YYYYMMDD), auto-detects latest if not provided')
    parser.add_argument('--daily-features-dir', type=str, default=None, help='Daily features directory (overrides country/club)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (overrides country/club)')
    parser.add_argument('--config', type=str, default=None, help='Config file path (overrides country/club)')
    parser.add_argument('--min-date', type=str, default=None, help='Minimum reference date (YYYY-MM-DD)')
    parser.add_argument('--max-date', type=str, default=None, help='Maximum reference date (YYYY-MM-DD). Truncates existing timelines beyond this date.')
    parser.add_argument('--regenerate-from-date', type=str, default=None, help='Regenerate timelines from this date onwards (YYYY-MM-DD)')
    parser.add_argument('--max-players', type=int, default=None, help='Limit number of players (for testing)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging (DEBUG level)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(verbose=args.verbose)
    
    # Resolve paths
    challenger_path = get_challenger_path(args.country, args.club)
    
    if args.daily_features_dir:
        daily_features_dir = Path(args.daily_features_dir)
    else:
        daily_features_dir = challenger_path / "daily_features"
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = challenger_path / "timelines"
    
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = challenger_path / "config.json"
    
    # Resolve raw data path
    if args.data_date:
        raw_data_dir = get_raw_data_path(args.country, args.data_date)
    else:
        # Auto-detect latest
        country_lower = args.country.lower().replace(" ", "_")
        raw_data_base = PRODUCTION_ROOT / "raw_data" / country_lower
        if raw_data_base.exists():
            date_folders = [d for d in raw_data_base.iterdir() if d.is_dir() and len(d.name) == 8 and d.name.isdigit()]
            if date_folders:
                raw_data_dir = max(date_folders, key=lambda x: x.name)
                args.data_date = raw_data_dir.name
            else:
                logger.error(f"Could not find raw data folder for {args.country}")
                return 1
        else:
            logger.error(f"Raw data base directory not found: {raw_data_base}")
            return 1
    
    if not raw_data_dir.exists():
        logger.error(f"Raw data directory not found: {raw_data_dir}")
        return 1
    
    # Check for enriched features first, fallback to Layer 1
    daily_features_enriched_dir = daily_features_dir  # Same directory (Layer 2 overwrites Layer 1)
    if not daily_features_dir.exists():
        logger.error(f"Daily features directory not found: {daily_features_dir}")
        return 1
    
    # Resolve dates
    min_date_ts = None
    max_date_ts = None
    if args.max_date:
        max_date_ts = pd.Timestamp(args.max_date).normalize()
    elif args.data_date:
        max_date_str = f"{args.data_date[:4]}-{args.data_date[4:6]}-{args.data_date[6:8]}"
        max_date_ts = pd.Timestamp(max_date_str)
    
    if args.min_date:
        min_date_ts = pd.Timestamp(args.min_date).normalize()
    
    regenerate_from_date_ts = None
    if args.regenerate_from_date:
        regenerate_from_date_ts = pd.Timestamp(args.regenerate_from_date).normalize()
    
    logger.info("=" * 70)
    logger.info("TIMELINE GENERATOR - V4 (PRODUCTION)")
    logger.info("=" * 70)
    logger.info(f"Country: {args.country}")
    logger.info(f"Club: {args.club}")
    logger.info(f"Daily features directory: {daily_features_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Raw data directory: {raw_data_dir}")
    logger.info(f"Data date: {args.data_date}")
    if min_date_ts:
        logger.info(f"Min date: {min_date_ts.date()}")
    if max_date_ts:
        logger.info(f"Max date: {max_date_ts.date()}")
    if regenerate_from_date_ts:
        logger.info(f"Regenerate from date: {regenerate_from_date_ts.date()}")
    logger.info("=" * 70)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load player IDs from config
    player_ids = load_player_ids_from_config(config_path)
    if not player_ids:
        logger.error(f"No player IDs found. Check config.json at: {config_path}")
        return 1
    
    if args.max_players:
        player_ids = player_ids[:args.max_players]
        logger.info(f"[TEST] Processing {len(player_ids)} players (limited from config)")
    
    logger.info(f"Processing {len(player_ids)} players from config")
    
    # Check for existing timelines file
    existing_timelines_file = output_dir / "timelines_35day_season_2025_2026_v4_muscular.csv"
    existing_timelines = None
    max_existing_date = None
    
    if existing_timelines_file.exists():
        logger.info(f"[LOAD] Loading existing timelines from: {existing_timelines_file}")
        try:
            existing_timelines = pd.read_csv(existing_timelines_file, low_memory=False)
            if 'reference_date' in existing_timelines.columns:
                existing_timelines['reference_date'] = pd.to_datetime(existing_timelines['reference_date'], errors='coerce', format='mixed')
                existing_timelines = existing_timelines.dropna(subset=['reference_date'])
                if len(existing_timelines) > 0:
                    max_existing_date = existing_timelines['reference_date'].max()
                    logger.info(f"[LOAD] Existing timelines have {len(existing_timelines)} rows, latest date: {max_existing_date.date()}")
                    
                    # Handle regenerate_from_date
                    if regenerate_from_date_ts:
                        before_count = len(existing_timelines)
                        existing_timelines = existing_timelines[existing_timelines['reference_date'] <= regenerate_from_date_ts].copy()
                        after_count = len(existing_timelines)
                        logger.info(f"[REGENERATE] Removed {before_count - after_count} timelines after {regenerate_from_date_ts.date()}")
                        if len(existing_timelines) > 0:
                            max_existing_date = existing_timelines['reference_date'].max()
                        else:
                            max_existing_date = None
                            existing_timelines = None
                    
                    # Handle max_date truncation
                    if max_date_ts:
                        before_count = len(existing_timelines)
                        existing_timelines = existing_timelines[existing_timelines['reference_date'] <= max_date_ts].copy()
                        after_count = len(existing_timelines)
                        if before_count > after_count:
                            logger.info(f"[TRUNCATE] Truncated {before_count - after_count} timelines beyond {max_date_ts.date()}")
                            existing_timelines.to_csv(existing_timelines_file, index=False, encoding='utf-8-sig')
                            if len(existing_timelines) > 0:
                                max_existing_date = existing_timelines['reference_date'].max()
                            else:
                                max_existing_date = None
                                existing_timelines = None
                    
                    # Set min_date if not provided
                    if max_existing_date and min_date_ts is None:
                        min_date_ts = max_existing_date + pd.Timedelta(days=1)
                        logger.info(f"[INCREMENTAL] Will generate timelines from {min_date_ts.date()} onwards")
        except Exception as e:
            logger.warning(f"[WARNING] Error loading existing timelines: {e}, will generate from scratch")
            existing_timelines = None
    
    # Get available seasons from daily features
    logger.info("[SCAN] Scanning daily features to determine available seasons...")
    try:
        available_seasons = get_all_seasons_from_daily_features_v4(str(daily_features_dir))
        logger.info(f"[SCAN] Found seasons: {available_seasons}")
    except Exception as e:
        logger.error(f"[ERROR] Failed to scan seasons: {e}")
        return 1
    
    # Setup V4 paths (temporarily modify V4 module paths)
    match_data_dir = raw_data_dir / "match_data"
    career_file = raw_data_dir / "players_career.csv"
    injuries_file = raw_data_dir / "injuries_data.csv"
    players_profile_file = raw_data_dir / "players_profile.csv"
    
    # Validate paths
    if not match_data_dir.exists():
        logger.error(f"Match data directory not found: {match_data_dir}")
        return 1
    if not career_file.exists():
        logger.error(f"Career file not found: {career_file}")
        return 1
    if not injuries_file.exists():
        logger.error(f"Injuries file not found: {injuries_file}")
        return 1
    if not players_profile_file.exists():
        logger.error(f"Players profile file not found: {players_profile_file}")
        return 1
    
    # Build PL clubs per season mapping
    logger.info("[BUILD] Building PL clubs per season mapping...")
    try:
        pl_clubs_by_season = build_pl_clubs_per_season_v4(str(match_data_dir))
        logger.info(f"[BUILD] Found PL clubs for {len(pl_clubs_by_season)} seasons")
    except Exception as e:
        logger.error(f"[ERROR] Failed to build PL clubs mapping: {e}")
        return 1
    
    # Build player PL membership periods
    logger.info("[BUILD] Building player PL membership periods...")
    try:
        player_pl_periods = build_player_pl_membership_periods_v4(str(career_file), pl_clubs_by_season)
        logger.info(f"[BUILD] Found PL membership periods for {len(player_pl_periods)} players")
    except Exception as e:
        logger.error(f"[ERROR] Failed to build PL membership periods: {e}")
        return 1
    
    # Load injuries data
    logger.info("[LOAD] Loading injuries data...")
    try:
        injury_class_map = load_injuries_data_v4(str(injuries_file))
        all_injury_dates_by_player = load_all_injury_dates_v4(str(injuries_file))
        logger.info(f"[LOAD] Loaded injury data for {len(all_injury_dates_by_player)} players")
    except Exception as e:
        logger.error(f"[ERROR] Failed to load injuries: {e}")
        return 1
    
    # Load player names
    logger.info("[LOAD] Loading player names...")
    try:
        player_names_map = load_player_names_mapping_v4(str(players_profile_file))
        logger.info(f"[LOAD] Loaded {len(player_names_map)} player names")
    except Exception as e:
        logger.warning(f"[WARNING] Failed to load player names: {e}")
        player_names_map = {}
    
    # Process each season and combine into single file
    logger.info("=" * 70)
    logger.info(f"[START] Processing {len(available_seasons)} seasons for {len(player_ids)} players")
    logger.info("=" * 70)
    
    all_season_timelines = []
    total_target1 = 0
    total_target2 = 0
    total_non_injuries = 0
    
    start_time = time.time()
    
    for season_start_year in available_seasons:
        season_start, season_end = get_season_date_range_v4(season_start_year)
        
        # Skip if min_date is after season end
        if min_date_ts and min_date_ts > season_end:
            logger.info(f"[SKIP] Season {season_start_year}_{season_start_year+1}: min_date ({min_date_ts.date()}) is after season end")
            continue
        
        # Skip if max_date is before season start
        if max_date_ts and max_date_ts < season_start:
            logger.info(f"[SKIP] Season {season_start_year}_{season_start_year+1}: max_date ({max_date_ts.date()}) is before season start")
            continue
        
        logger.info(f"\n[SEASON] Processing season {season_start_year}_{season_start_year+1}...")
        logger.info(f"   Date range: {season_start.date()} to {season_end.date()}")
        
        try:
            # Call V4 process_season function
            # Note: V4 function expects output_dir to be V4_TIMELINES_TRAIN or V4_TIMELINES_TEST
            # We'll use a temp directory and then combine
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                output_file, target1_count, target2_count, non_injury_count = process_season_v4(
                    season_start_year=season_start_year,
                    daily_features_dir=str(daily_features_dir),
                    all_injury_dates_by_player=all_injury_dates_by_player,
                    injury_class_map=injury_class_map,
                    player_names_map=player_names_map,
                    player_ids=player_ids,  # Filter by club's players
                    output_dir=temp_dir,
                    max_players=args.max_players
                )
                
                if output_file and Path(output_file).exists():
                    # Load the season timeline file
                    season_df = pd.read_csv(output_file, low_memory=False)
                    
                    # Filter by date range if needed
                    if 'reference_date' in season_df.columns:
                        season_df['reference_date'] = pd.to_datetime(season_df['reference_date'], errors='coerce')
                        season_df = season_df.dropna(subset=['reference_date'])
                        
                        if min_date_ts:
                            before_count = len(season_df)
                            season_df = season_df[season_df['reference_date'] >= min_date_ts].copy()
                            after_count = len(season_df)
                            if before_count > after_count:
                                logger.info(f"   [FILTER] Filtered {before_count - after_count} rows before min_date")
                        
                        if max_date_ts:
                            before_count = len(season_df)
                            season_df = season_df[season_df['reference_date'] <= max_date_ts].copy()
                            after_count = len(season_df)
                            if before_count > after_count:
                                logger.info(f"   [FILTER] Filtered {before_count - after_count} rows after max_date")
                    
                    if len(season_df) > 0:
                        all_season_timelines.append(season_df)
                        total_target1 += target1_count
                        total_target2 += target2_count
                        total_non_injuries += non_injury_count
                        logger.info(f"   [OK] Season {season_start_year}_{season_start_year+1}: {len(season_df)} timelines")
                    else:
                        logger.info(f"   [SKIP] Season {season_start_year}_{season_start_year+1}: No timelines after filtering")
                else:
                    logger.warning(f"   [WARN] Season {season_start_year}_{season_start_year+1}: No output file generated")
        
        except Exception as e:
            logger.error(f"   [ERROR] Season {season_start_year}_{season_start_year+1}: {e}")
            if args.verbose:
                logger.error(traceback.format_exc())
            continue
    
    # Combine all season timelines
    if all_season_timelines:
        logger.info(f"\n[COMBINE] Combining {len(all_season_timelines)} season files...")
        combined_timelines = pd.concat(all_season_timelines, ignore_index=True)
        
        # Deduplicate by player_id and reference_date
        if 'player_id' in combined_timelines.columns and 'reference_date' in combined_timelines.columns:
            before_dedup = len(combined_timelines)
            combined_timelines = combined_timelines.sort_values(['player_id', 'reference_date'])
            combined_timelines = combined_timelines.drop_duplicates(subset=['player_id', 'reference_date'], keep='first')
            after_dedup = len(combined_timelines)
            if before_dedup != after_dedup:
                logger.info(f"[DEDUP] Removed {before_dedup - after_dedup} duplicate timelines")
        
        # Apply PL filtering (post-processing, like V3)
        logger.info(f"[FILTER] Applying PL-only filter to {len(combined_timelines)} timelines...")
        before_filter = len(combined_timelines)
        combined_timelines = filter_timelines_pl_only_v4(combined_timelines, player_pl_periods)
        after_filter = len(combined_timelines)
        if before_filter > 0:
            logger.info(f"[FILTER] Filtered from {before_filter:,} to {after_filter:,} timelines ({after_filter/before_filter*100:.1f}% kept)")
        else:
            logger.info(f"[FILTER] No timelines to filter")
        
        # Merge with existing timelines if provided
        if existing_timelines is not None and len(existing_timelines) > 0:
            logger.info(f"[MERGE] Merging {len(combined_timelines)} new timelines with {len(existing_timelines)} existing timelines...")
            
            # Ensure column alignment
            existing_cols = set(existing_timelines.columns)
            new_cols = set(combined_timelines.columns)
            
            # Add missing columns
            for col in new_cols - existing_cols:
                existing_timelines[col] = pd.NA
            for col in existing_cols - new_cols:
                combined_timelines[col] = pd.NA
            
            # Reorder to match existing
            combined_timelines = combined_timelines[existing_timelines.columns]
            
            # Concatenate and deduplicate (keep new ones)
            final_timelines = pd.concat([existing_timelines, combined_timelines], ignore_index=True)
            final_timelines = final_timelines.sort_values(['player_id', 'reference_date'])
            final_timelines = final_timelines.drop_duplicates(subset=['player_id', 'reference_date'], keep='last')
            
            logger.info(f"[MERGE] Final timelines: {len(final_timelines)} rows")
        else:
            final_timelines = combined_timelines
        
        # Format reference_date consistently
        if 'reference_date' in final_timelines.columns:
            final_timelines['reference_date'] = pd.to_datetime(final_timelines['reference_date'], errors='coerce').dt.date.astype(str)
            final_timelines = final_timelines[final_timelines['reference_date'] != 'NaT'].copy()
        
        # Save to single file per club
        output_file = output_dir / "timelines_35day_season_2025_2026_v4_muscular.csv"
        final_timelines.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        logger.info(f"[SAVE] Saved timelines to: {output_file}")
        logger.info(f"   Total timelines: {len(final_timelines):,}")
        if 'reference_date' in final_timelines.columns:
            valid_dates = pd.to_datetime(final_timelines['reference_date'], errors='coerce').dropna()
            if len(valid_dates) > 0:
                logger.info(f"   Date range: {valid_dates.min().date()} to {valid_dates.max().date()}")
    else:
        logger.warning("[WARN] No timelines generated")
        if existing_timelines is not None:
            logger.info(f"[KEEP] Keeping existing timelines: {len(existing_timelines)} rows")
            output_file = output_dir / "timelines_35day_season_2025_2026_v4_muscular.csv"
            existing_timelines.to_csv(output_file, index=False, encoding='utf-8-sig')
        else:
            return 1
    
    # Summary
    total_elapsed = time.time() - start_time
    logger.info("")
    logger.info("=" * 70)
    logger.info("[COMPLETE] TIMELINE GENERATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"[STATS] Processed {len(available_seasons)} seasons")
    logger.info(f"   [TARGET1] Muscular injury timelines: {total_target1:,}")
    logger.info(f"   [TARGET2] Skeletal injury timelines: {total_target2:,}")
    logger.info(f"   [NON-INJ] Non-injury timelines: {total_non_injuries:,}")
    logger.info(f"   [TOTAL] Total timelines: {total_target1 + total_target2 + total_non_injuries:,}")
    logger.info(f"[TIME] Total time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")
    logger.info("=" * 70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
