#!/usr/bin/env python3
"""
Daily Features Generator - Version 4 (Layer 1) - Production Wrapper

This script wraps the V4 enhanced daily features generator for production use.
It adapts the V4 training script to work with the production directory structure
and adds incremental update support.

Key features:
- Reads from production/raw_data/england/{YYYYMMDD}/
- Writes to production/deployments/England/challenger/{club}/daily_features/
- Supports incremental updates
- Handles date capping (max-date parameter)
- Processes one club at a time
- Uses config.json to get player list
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
from typing import List, Optional

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
V4_CODE_DIR = ROOT_DIR / "models_production" / "lgbm_muscular_v4" / "code" / "daily_features"
if str(V4_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(V4_CODE_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import V4 daily features generator
try:
    # Import the V4 function - we'll call it directly
    import importlib.util
    v4_module_path = V4_CODE_DIR / "create_daily_features_v4_enhanced.py"
    spec = importlib.util.spec_from_file_location("create_daily_features_v4_enhanced", v4_module_path)
    v4_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v4_module)
    
    # Get the function we need
    generate_daily_features_for_player_v4 = v4_module.generate_daily_features_for_player
    load_player_data_v4 = v4_module.load_player_data
    setup_logging_v4 = v4_module.setup_logging
    
    # Get logger from V4 module
    v4_logger = v4_module.logger
except Exception as e:
    print(f"[ERROR] Failed to import V4 daily features module: {e}")
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


def get_latest_raw_data_folder(country: str) -> Optional[Path]:
    """Find the latest date folder in production/raw_data/{country}/."""
    country_lower = country.lower().replace(" ", "_")
    raw_data_path = PRODUCTION_ROOT / "raw_data" / country_lower
    
    if not raw_data_path.exists():
        return None
    
    # List all subdirectories matching YYYYMMDD pattern
    date_folders = []
    for item in raw_data_path.iterdir():
        if item.is_dir() and len(item.name) == 8 and item.name.isdigit():
            try:
                # Validate it's a valid date
                year = int(item.name[:4])
                month = int(item.name[4:6])
                day = int(item.name[6:8])
                if 1 <= month <= 12 and 1 <= day <= 31:
                    date_folders.append((item.name, year * 10000 + month * 100 + day))
            except (ValueError, IndexError):
                continue
    
    if not date_folders:
        return None
    
    # Sort by date value and return latest
    date_folders.sort(key=lambda x: x[1], reverse=True)
    latest_folder = date_folders[0][0]
    return raw_data_path / latest_folder


def get_challenger_path(country: str, club: str) -> Path:
    """Get base path for challenger club."""
    return PRODUCTION_ROOT / "deployments" / country / "challenger" / club


def load_player_ids_from_config(config_path: Path) -> List[int]:
    """Load player IDs from deployment config.json."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        player_ids = config.get('player_ids', [])
        return player_ids
    except Exception as e:
        print(f"[ERROR] Error loading config from {config_path}: {e}")
        return []


def generate_daily_features_v4_wrapper(
    player_id: int,
    data_dir: Path,
    reference_date: pd.Timestamp,
    output_dir: Path,
    existing_file_path: Optional[Path] = None,
    incremental: bool = True,
    max_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """
    Wrapper around V4 generate_daily_features_for_player that adds:
    - Incremental update support
    - Date capping (max_date)
    - Path adaptation
    """
    logger = logging.getLogger(__name__)
    
    # Load existing data if incremental
    existing_df = None
    start_date_override = None
    
    if incremental and existing_file_path and existing_file_path.exists():
        try:
            logger.info(f"[INCREMENTAL] Loading existing file: {existing_file_path}")
            existing_df = pd.read_csv(existing_file_path, parse_dates=['date'], encoding='utf-8-sig', low_memory=False)
            
            if not existing_df.empty and 'date' in existing_df.columns:
                # Truncate existing file if it exceeds max_date
                if max_date is not None:
                    before_count = len(existing_df)
                    existing_df = existing_df[existing_df['date'] <= max_date].copy()
                    after_count = len(existing_df)
                    if before_count > after_count:
                        logger.info(f"[TRUNCATE] Truncated existing file: removed {before_count - after_count} rows beyond {max_date.date()}")
                        existing_df.to_csv(existing_file_path, index=False, encoding='utf-8-sig')
                
                if not existing_df.empty:
                    max_existing_date = pd.to_datetime(existing_df['date'].max()).normalize()
                    start_date_override = max_existing_date + pd.Timedelta(days=1)
                    logger.info(f"[INCREMENTAL] Existing file has data up to {max_existing_date.date()}, will generate from {start_date_override.date()}")
                else:
                    logger.warning("[INCREMENTAL] Existing file is now empty after truncation, generating full history")
                    existing_df = None
            else:
                logger.warning("[INCREMENTAL] Existing file is empty or missing date column, generating full history")
                existing_df = None
        except Exception as e:
            logger.warning(f"[INCREMENTAL] Error loading existing file: {e}, generating full history")
            existing_df = None
    
    # Call V4 function to generate features
    # Note: V4 function doesn't support incremental or max_date, so we'll handle that ourselves
    try:
        # Convert Path to string for V4 function
        data_dir_str = str(data_dir)
        output_dir_str = str(output_dir)
        
        # Generate features using V4 function
        # The V4 function generates full history, so we'll filter afterwards
        daily_features = generate_daily_features_for_player_v4(
            player_id=player_id,
            data_dir=data_dir_str,
            reference_date=reference_date,
            output_dir=output_dir_str
        )
        
        if daily_features.empty:
            logger.warning(f"[WARN] No features generated for player {player_id}")
            return existing_df if existing_df is not None else pd.DataFrame()
        
        # Ensure date column exists and is datetime
        if 'date' not in daily_features.columns:
            if isinstance(daily_features.index, pd.DatetimeIndex):
                daily_features = daily_features.reset_index()
                # Rename index column to 'date' if needed
                if daily_features.columns[0] not in ['date', 'player_id']:
                    daily_features = daily_features.rename(columns={daily_features.columns[0]: 'date'})
            else:
                logger.error(f"[ERROR] No date column found in generated features for player {player_id}")
                return existing_df if existing_df is not None else pd.DataFrame()
        
        # Convert date to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(daily_features['date']):
            daily_features['date'] = pd.to_datetime(daily_features['date'], errors='coerce')
        
        # Filter by max_date if provided
        if max_date is not None:
            before_count = len(daily_features)
            daily_features = daily_features[daily_features['date'] <= max_date].copy()
            after_count = len(daily_features)
            if before_count > after_count:
                logger.info(f"[FILTER] Filtered {before_count - after_count} rows beyond {max_date.date()}")
        
        # Filter for incremental updates (only new dates)
        if start_date_override is not None:
            before_count = len(daily_features)
            daily_features = daily_features[daily_features['date'] >= start_date_override].copy()
            after_count = len(daily_features)
            if before_count > after_count:
                logger.info(f"[INCREMENTAL] Filtered to {after_count} new rows (from {start_date_override.date()})")
            
            if daily_features.empty:
                logger.info("[INCREMENTAL] No new dates to generate, returning existing data")
                return existing_df if existing_df is not None else pd.DataFrame()
        
        # Merge with existing data if incremental
        if incremental and existing_df is not None and not existing_df.empty:
            logger.info(f"[INCREMENTAL] Merging {len(daily_features)} new rows with {len(existing_df)} existing rows")
            
            # Ensure column alignment
            existing_cols = set(existing_df.columns)
            new_cols = set(daily_features.columns)
            
            # Add missing columns to existing_df
            for col in new_cols - existing_cols:
                existing_df[col] = pd.NA
            
            # Add missing columns to daily_features
            for col in existing_cols - new_cols:
                daily_features[col] = pd.NA
            
            # Reorder new columns to match existing
            daily_features = daily_features[existing_df.columns]
            
            # Concatenate and deduplicate
            combined = pd.concat([existing_df, daily_features], ignore_index=True)
            combined = combined.sort_values('date').drop_duplicates(subset=['date'], keep='first').reset_index(drop=True)
            
            logger.info(f"[INCREMENTAL] Combined result: {len(combined)} rows (from {combined['date'].min().date()} to {combined['date'].max().date()})")
            return combined
        
        return daily_features
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to generate features for player {player_id}: {e}")
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            traceback.print_exc()
        return existing_df if existing_df is not None else pd.DataFrame()


def main():
    """Main entry point - Production version with incremental updates."""
    parser = argparse.ArgumentParser(description='Update daily features for football players (V4 Layer 1 - Production)')
    parser.add_argument('--country', type=str, default='England', help='Country name (default: England)')
    parser.add_argument('--club', type=str, required=True, help='Club name (required)')
    parser.add_argument('--data-date', type=str, default=None, help='Raw data date (YYYYMMDD), auto-detects latest if not provided')
    parser.add_argument('--reference-date', type=str, default=None, help='Reference date (YYYY-MM-DD), defaults to data-date')
    parser.add_argument('--max-date', type=str, default=None, help='Maximum date to generate features (YYYY-MM-DD). Truncates existing files beyond this date.')
    parser.add_argument('--player-id', type=int, default=None, help='Player ID to process (if not provided, processes all players from config)')
    parser.add_argument('--data-dir', type=str, default=None, help='Explicit data directory (overrides country/data-date)')
    parser.add_argument('--output-dir', type=str, default=None, help='Explicit output directory (overrides country/club)')
    parser.add_argument('--incremental', action='store_true', default=True, help='Append new rows to existing files (default: True)')
    parser.add_argument('--force', action='store_true', help='Force full regeneration (ignore existing files)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging (DEBUG level)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(verbose=args.verbose)
    
    # Resolve data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    elif args.data_date:
        country_lower = args.country.lower().replace(" ", "_")
        data_dir = PRODUCTION_ROOT / "raw_data" / country_lower / args.data_date
    else:
        # Auto-detect latest
        data_dir = get_latest_raw_data_folder(args.country)
        if not data_dir:
            logger.error(f"Could not find raw data folder for {args.country}")
            return 1
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return 1
    
    # Resolve output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        challenger_path = get_challenger_path(args.country, args.club)
        output_dir = challenger_path / "daily_features"
    
    # Resolve reference date
    if args.reference_date:
        reference_date = pd.Timestamp(args.reference_date)
    elif args.data_date:
        # Convert YYYYMMDD to YYYY-MM-DD
        ref_date_str = f"{args.data_date[:4]}-{args.data_date[4:6]}-{args.data_date[6:8]}"
        reference_date = pd.Timestamp(ref_date_str)
    else:
        # Extract from data_dir folder name (YYYYMMDD)
        folder_name = data_dir.name
        if len(folder_name) == 8 and folder_name.isdigit():
            ref_date_str = f"{folder_name[:4]}-{folder_name[4:6]}-{folder_name[6:8]}"
            reference_date = pd.Timestamp(ref_date_str)
        else:
            reference_date = pd.Timestamp.today()
    
    # Resolve max_date
    max_date = None
    if args.max_date:
        max_date = pd.Timestamp(args.max_date)
    elif args.data_date:
        # Use data_date as max_date if not explicitly provided
        max_date_str = f"{args.data_date[:4]}-{args.data_date[4:6]}-{args.data_date[6:8]}"
        max_date = pd.Timestamp(max_date_str)
    
    logger.info("=" * 70)
    logger.info("DAILY FEATURES UPDATER - V4 LAYER 1 (PRODUCTION)")
    logger.info("=" * 70)
    logger.info(f"Country: {args.country}")
    logger.info(f"Club: {args.club}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Reference date: {reference_date.date()}")
    if max_date:
        logger.info(f"Max date (cap): {max_date.date()}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Incremental mode: {args.incremental}")
    logger.info(f"Force regeneration: {args.force}")
    logger.info(f"Verbose mode: {args.verbose}")
    logger.info("=" * 70)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which players to process
    if args.player_id:
        player_ids = [args.player_id]
        logger.info(f"Processing single player: {args.player_id}")
    else:
        # Load from config.json (challenger path)
        challenger_path = get_challenger_path(args.country, args.club)
        config_path = challenger_path / "config.json"
        player_ids = load_player_ids_from_config(config_path)
        if not player_ids:
            logger.error(f"No player IDs found. Check config.json at: {config_path}")
            return 1
        logger.info(f"Processing {len(player_ids)} players from config.json")
    
    # Process all players
    total_players = len(player_ids)
    successful = 0
    failed = 0
    skipped = 0
    total_time = 0
    total_rows = 0
    total_size_mb = 0
    
    logger.info("=" * 70)
    logger.info(f"[START] Starting batch processing for {total_players} players")
    logger.info("=" * 70)
    
    overall_start_time = time.time()
    last_progress_log = overall_start_time
    
    for idx, player_id in enumerate(player_ids, 1):
        player_start_time = time.time()
        
        # Output file path
        output_file = output_dir / f'player_{player_id}_daily_features.csv'
        
        # Determine if we should use incremental mode
        use_incremental = args.incremental and not args.force
        existing_file_path = output_file if (use_incremental and output_file.exists()) else None
        
        if existing_file_path:
            logger.info(f"[{idx}/{total_players}] [INCREMENTAL] Player {player_id}: Appending to existing file")
        elif output_file.exists() and args.force:
            logger.info(f"[{idx}/{total_players}] [FORCE] Player {player_id}: Regenerating from scratch")
            existing_file_path = None
            use_incremental = False
        else:
            logger.info(f"[{idx}/{total_players}] [NEW] Player {player_id}: Generating full history")
        
        # Progress visualization
        current_time = time.time()
        if (current_time - last_progress_log >= 5) or (idx % 10 == 0):
            elapsed_total = current_time - overall_start_time
            progress_pct = (idx * 100) // total_players if total_players > 0 else 0
            avg_time_per_player = elapsed_total / idx if idx > 0 else 0
            remaining_players = total_players - idx
            eta_seconds = remaining_players * avg_time_per_player
            eta_str = str(timedelta(seconds=int(eta_seconds)))
            elapsed_str = str(timedelta(seconds=int(elapsed_total)))
            
            logger.info("")
            logger.info("=" * 70)
            logger.info("[PROGRESS] OVERALL PROGRESS")
            logger.info("=" * 70)
            logger.info(f"   Progress: {idx}/{total_players} ({progress_pct}%)")
            logger.info(f"   [TIME] Elapsed: {elapsed_str} | ETA: {eta_str}")
            logger.info(f"   [STATS] Success: {successful} | Failed: {failed} | Skipped: {skipped}")
            if successful > 0:
                logger.info(f"   [RATE] Avg time/player: {total_time/successful:.1f}s | Rate: {successful/(elapsed_total/3600):.1f} players/hour")
            logger.info("=" * 70)
            logger.info("")
            last_progress_log = current_time
        
        # Per-player progress
        progress_pct = (idx * 100) // total_players if total_players > 0 else 0
        logger.info(f"[{idx}/{total_players}] ({progress_pct}%) [PROCESSING] Player {player_id}...")
        
        try:
            daily_features = generate_daily_features_v4_wrapper(
                player_id=player_id,
                data_dir=data_dir,
                reference_date=reference_date,
                output_dir=output_dir,
                existing_file_path=existing_file_path,
                incremental=use_incremental,
                max_date=max_date
            )
            
            if daily_features.empty:
                logger.warning(f"[{idx}/{total_players}] [WARN] No features generated for player {player_id}")
                failed += 1
                continue
            
            # Save to CSV
            daily_features.to_csv(output_file, index=False, encoding='utf-8-sig')
            file_size = output_file.stat().st_size / (1024 * 1024)  # Size in MB
            
            player_elapsed = time.time() - player_start_time
            total_time += player_elapsed
            total_rows += len(daily_features)
            total_size_mb += file_size
            
            successful += 1
            logger.info(f"[{idx}/{total_players}] [OK] SUCCESS: Player {player_id}")
            logger.info(f"   [FILE] Output: {output_file.name}")
            logger.info(f"   [DATA] Rows: {len(daily_features):,}, Columns: {len(daily_features.columns)}")
            logger.info(f"   [SIZE] File size: {file_size:.2f} MB")
            logger.info(f"   [TIME] Time: {player_elapsed:.2f}s ({player_elapsed/60:.2f} min)")
            if player_elapsed > 0:
                logger.info(f"   [RATE] Rate: {len(daily_features)/player_elapsed:.0f} rows/sec")
            
        except Exception as e:
            failed += 1
            logger.error(f"[{idx}/{total_players}] [ERROR] ERROR: Player {player_id}")
            logger.error(f"   Error: {str(e)}")
            if args.verbose:
                logger.error(traceback.format_exc())
            continue
    
    # Final summary
    total_elapsed = time.time() - overall_start_time
    logger.info("")
    logger.info("=" * 70)
    logger.info("[COMPLETE] BATCH PROCESSING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"[STATS] Total players: {total_players}")
    logger.info(f"   [OK] Successful: {successful} ({successful*100//total_players if total_players > 0 else 0}%)")
    logger.info(f"   [FAIL] Failed: {failed} ({failed*100//total_players if total_players > 0 else 0}%)")
    logger.info(f"   [SKIP] Skipped: {skipped} ({skipped*100//total_players if total_players > 0 else 0}%)")
    logger.info(f"")
    logger.info(f"[TIME] Total time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes / {total_elapsed/3600:.2f} hours)")
    if successful > 0:
        logger.info(f"[STATS] Average time per player: {total_time/successful:.2f} seconds")
        logger.info(f"[RATE] Processing rate: {successful/(total_elapsed/3600):.2f} players/hour")
        logger.info(f"[DATA] Total rows generated: {total_rows:,}")
        logger.info(f"[SIZE] Total size: {total_size_mb:.2f} MB ({total_size_mb/1024:.2f} GB)")
        logger.info(f"[DATA] Average rows per player: {total_rows//successful if successful > 0 else 0:,}")
        logger.info(f"[SIZE] Average size per player: {total_size_mb/successful:.2f} MB")
    logger.info("=" * 70)
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
