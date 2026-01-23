#!/usr/bin/env python3
"""
One-time migration script to copy original daily features files for Chelsea players
from daily_features_output/ to production/deployments/England/Chelsea FC/daily_features/

This script should only be run once during initial setup.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List

# Calculate paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent
ROOT_DIR = PRODUCTION_ROOT.parent

def load_chelsea_player_ids(config_path: Path) -> List[int]:
    """Load Chelsea player IDs from config.json."""
    import json
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config.get('player_ids', [])

def migrate_chelsea_files(
    source_dir: Path,
    dest_dir: Path,
    player_ids: List[int],
    dry_run: bool = False
) -> None:
    """Copy daily features files for Chelsea players."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    copied = 0
    skipped = 0
    errors = 0
    
    print(f"Source directory: {source_dir}")
    print(f"Destination directory: {dest_dir}")
    print(f"Player IDs to migrate: {len(player_ids)}")
    print(f"Dry run: {dry_run}")
    print("=" * 70)
    
    for player_id in player_ids:
        source_file = source_dir / f"player_{player_id}_daily_features.csv"
        dest_file = dest_dir / f"player_{player_id}_daily_features.csv"
        
        if not source_file.exists():
            print(f"[SKIP] Player {player_id}: Source file not found: {source_file.name}")
            skipped += 1
            continue
        
        if dest_file.exists():
            print(f"[SKIP] Player {player_id}: Destination file already exists: {dest_file.name}")
            skipped += 1
            continue
        
        if dry_run:
            print(f"[DRY RUN] Would copy: {source_file.name} -> {dest_file}")
            copied += 1
        else:
            try:
                shutil.copy2(source_file, dest_file)
                file_size = dest_file.stat().st_size / (1024 * 1024)  # MB
                print(f"[OK] Player {player_id}: Copied {source_file.name} ({file_size:.2f} MB)")
                copied += 1
            except Exception as e:
                print(f"[ERROR] Player {player_id}: Failed to copy {source_file.name}: {e}")
                errors += 1
    
    print("=" * 70)
    print(f"Migration complete:")
    print(f"  Copied: {copied}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {errors}")

def main():
    parser = argparse.ArgumentParser(description='Migrate Chelsea daily features files')
    parser.add_argument(
        '--source-dir',
        type=Path,
        default=ROOT_DIR / 'daily_features_output',
        help='Source directory (default: daily_features_output)'
    )
    parser.add_argument(
        '--dest-dir',
        type=Path,
        default=PRODUCTION_ROOT / 'deployments' / 'England' / 'Chelsea FC' / 'daily_features',
        help='Destination directory'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=PRODUCTION_ROOT / 'deployments' / 'England' / 'Chelsea FC' / 'config.json',
        help='Path to config.json with player IDs'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be copied without actually copying'
    )
    
    args = parser.parse_args()
    
    if not args.config.exists():
        print(f"ERROR: Config file not found: {args.config}")
        return
    
    if not args.source_dir.exists():
        print(f"ERROR: Source directory not found: {args.source_dir}")
        return
    
    player_ids = load_chelsea_player_ids(args.config)
    
    if not player_ids:
        print("ERROR: No player IDs found in config.json")
        return
    
    migrate_chelsea_files(
        source_dir=args.source_dir,
        dest_dir=args.dest_dir,
        player_ids=player_ids,
        dry_run=args.dry_run
    )

if __name__ == '__main__':
    main()










