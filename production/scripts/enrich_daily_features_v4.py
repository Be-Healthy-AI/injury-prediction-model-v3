#!/usr/bin/env python3
"""
Layer 2 Enrichment for Daily Features (V4) - Production Wrapper

This script wraps the V4 Layer 2 enrichment script for production use.
It enriches Layer 1 daily features with advanced workload, recovery, and
injury-history features.

Key features:
- Reads from production/deployments/England/challenger/{club}/daily_features/
- Writes enriched features to daily_features_enhanced/ folder (preserves Layer 1)
- Supports incremental updates (only enriches new files)
- Processes one club at a time
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

# Import V4 enrichment function
try:
    import importlib.util
    v4_enrich_module_path = V4_CODE_DIR / "enrich_daily_features_v4_layer2.py"
    spec = importlib.util.spec_from_file_location("enrich_daily_features_v4_layer2", v4_enrich_module_path)
    v4_enrich_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v4_enrich_module)
    
    # Get the enrichment function
    enrich_one_file_v4 = v4_enrich_module.enrich_one_file
    
except Exception as e:
    print(f"[ERROR] Failed to import V4 enrichment module: {e}")
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
    parser = argparse.ArgumentParser(description='Enrich daily features (V4 Layer 2) - Production')
    parser.add_argument('--country', type=str, default='England', help='Country name (default: England)')
    parser.add_argument('--club', type=str, required=True, help='Club name (required)')
    parser.add_argument('--input-dir', type=str, default=None, help='Input directory (overrides country/club)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (overrides country/club, defaults to input-dir)')
    parser.add_argument('--player-id', type=int, default=None, help='Player ID to process (if not provided, processes all players from config)')
    parser.add_argument('--force', action='store_true', help='Force regeneration of existing enriched files')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging (DEBUG level)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(verbose=args.verbose)
    
    # Resolve input directory
    if args.input_dir:
        input_dir = Path(args.input_dir)
    else:
        challenger_path = get_challenger_path(args.country, args.club)
        input_dir = challenger_path / "daily_features"
    
    # Resolve output directory (defaults to daily_features_enhanced folder)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Write to separate daily_features_enhanced folder to preserve Layer 1
        challenger_path = get_challenger_path(args.country, args.club)
        output_dir = challenger_path / "daily_features_enhanced"
    
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return 1
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("DAILY FEATURES ENRICHMENT - V4 LAYER 2 (PRODUCTION)")
    logger.info("=" * 70)
    logger.info(f"Country: {args.country}")
    logger.info(f"Club: {args.club}")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Force regeneration: {args.force}")
    logger.info(f"Verbose mode: {args.verbose}")
    logger.info("=" * 70)
    
    # Determine which players to process
    if args.player_id:
        # Process single player
        input_file = input_dir / f'player_{args.player_id}_daily_features.csv'
        if not input_file.exists():
            logger.error(f"Input file not found: {input_file}")
            return 1
        input_files = [input_file]
        logger.info(f"Processing single player: {args.player_id}")
    else:
        # Load from config.json to get player list
        challenger_path = get_challenger_path(args.country, args.club)
        config_path = challenger_path / "config.json"
        player_ids = load_player_ids_from_config(config_path)
        
        if not player_ids:
            logger.error(f"No player IDs found. Check config.json at: {config_path}")
            return 1
        
        # Find all input files for players in config
        input_files = []
        for player_id in player_ids:
            input_file = input_dir / f'player_{player_id}_daily_features.csv'
            if input_file.exists():
                input_files.append(input_file)
            else:
                logger.warning(f"[WARN] Layer 1 file not found for player {player_id}: {input_file}")
        
        logger.info(f"Found {len(input_files)} Layer 1 files to enrich (out of {len(player_ids)} players in config)")
    
    if not input_files:
        logger.error("No input files found to enrich")
        return 1
    
    # Process all files
    total_files = len(input_files)
    successful = 0
    failed = 0
    skipped = 0
    
    logger.info("=" * 70)
    logger.info(f"[START] Starting enrichment for {total_files} files")
    logger.info("=" * 70)
    
    overall_start_time = time.time()
    last_progress_log = overall_start_time
    
    for idx, input_file in enumerate(input_files, 1):
        file_start_time = time.time()
        
        # Output file (same name, overwrites Layer 1)
        output_file = output_dir / input_file.name
        
        # Check if already enriched
        if output_file.exists() and not args.force:
            # Check if output is newer than input (already enriched)
            if output_file.stat().st_mtime >= input_file.stat().st_mtime:
                logger.info(f"[{idx}/{total_files}] [SKIP] {input_file.name}: Already enriched")
                skipped += 1
                continue
        
        # Progress visualization
        current_time = time.time()
        if (current_time - last_progress_log >= 5) or (idx % 10 == 0):
            elapsed_total = current_time - overall_start_time
            progress_pct = (idx * 100) // total_files if total_files > 0 else 0
            avg_time_per_file = elapsed_total / idx if idx > 0 else 0
            remaining_files = total_files - idx
            eta_seconds = remaining_files * avg_time_per_file
            eta_str = str(timedelta(seconds=int(eta_seconds)))
            elapsed_str = str(timedelta(seconds=int(elapsed_total)))
            
            logger.info("")
            logger.info("=" * 70)
            logger.info("[PROGRESS] OVERALL PROGRESS")
            logger.info("=" * 70)
            logger.info(f"   Progress: {idx}/{total_files} ({progress_pct}%)")
            logger.info(f"   [TIME] Elapsed: {elapsed_str} | ETA: {eta_str}")
            logger.info(f"   [STATS] Success: {successful} | Failed: {failed} | Skipped: {skipped}")
            logger.info("=" * 70)
            logger.info("")
            last_progress_log = current_time
        
        logger.info(f"[{idx}/{total_files}] [PROCESSING] {input_file.name}...")
        
        try:
            # Call V4 enrichment function
            enrich_one_file_v4(
                input_path=input_file,
                output_path=output_file,
                verbose=args.verbose,
                force=args.force
            )
            
            file_elapsed = time.time() - file_start_time
            successful += 1
            
            # Check file size
            if output_file.exists():
                file_size = output_file.stat().st_size / (1024 * 1024)  # MB
                logger.info(f"[{idx}/{total_files}] [OK] SUCCESS: {input_file.name}")
                logger.info(f"   [SIZE] File size: {file_size:.2f} MB")
                logger.info(f"   [TIME] Time: {file_elapsed:.2f}s")
            else:
                logger.warning(f"[{idx}/{total_files}] [WARN] Output file not created: {output_file}")
                failed += 1
            
        except Exception as e:
            failed += 1
            logger.error(f"[{idx}/{total_files}] [ERROR] ERROR: {input_file.name}")
            logger.error(f"   Error: {str(e)}")
            if args.verbose:
                logger.error(traceback.format_exc())
            continue
    
    # Final summary
    total_elapsed = time.time() - overall_start_time
    logger.info("")
    logger.info("=" * 70)
    logger.info("[COMPLETE] ENRICHMENT COMPLETE")
    logger.info("=" * 70)
    logger.info(f"[STATS] Total files: {total_files}")
    logger.info(f"   [OK] Successful: {successful} ({successful*100//total_files if total_files > 0 else 0}%)")
    logger.info(f"   [FAIL] Failed: {failed} ({failed*100//total_files if total_files > 0 else 0}%)")
    logger.info(f"   [SKIP] Skipped: {skipped} ({skipped*100//total_files if total_files > 0 else 0}%)")
    logger.info(f"")
    logger.info(f"[TIME] Total time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")
    if successful > 0:
        logger.info(f"[STATS] Average time per file: {total_elapsed/successful:.2f} seconds")
    logger.info("=" * 70)
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
