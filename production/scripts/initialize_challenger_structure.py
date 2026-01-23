#!/usr/bin/env python3
"""
Initialize challenger directory structure for V4 model deployment.

This script:
1. Creates production/deployments/England/challenger/ directory
2. Copies all club configs from V3 to challenger
3. Creates subdirectories for each club (daily_features, timelines, predictions, dashboards)
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import List

# Calculate paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent


def get_all_v3_clubs(country: str = "England") -> List[str]:
    """Get all club folders in V3 deployments directory."""
    deployments_dir = PRODUCTION_ROOT / "deployments" / country
    if not deployments_dir.exists():
        return []
    
    clubs = []
    for item in deployments_dir.iterdir():
        if item.is_dir() and (item / "config.json").exists():
            # Skip challenger folder
            if item.name == "challenger":
                continue
            clubs.append(item.name)
    
    return sorted(clubs)


def copy_config_to_challenger(country: str, club: str) -> bool:
    """Copy config.json from V3 to challenger."""
    v3_config = PRODUCTION_ROOT / "deployments" / country / club / "config.json"
    challenger_path = PRODUCTION_ROOT / "deployments" / country / "challenger" / club
    challenger_config = challenger_path / "config.json"
    
    if not v3_config.exists():
        print(f"  [WARN] V3 config not found for {club}: {v3_config}")
        return False
    
    challenger_path.mkdir(parents=True, exist_ok=True)
    
    try:
        shutil.copy2(v3_config, challenger_config)
        print(f"  [OK] Copied config for {club}")
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to copy config for {club}: {e}")
        return False


def create_club_directories(country: str, club: str):
    """Create subdirectories for a challenger club."""
    challenger_path = PRODUCTION_ROOT / "deployments" / country / "challenger" / club
    
    directories = [
        "daily_features",
        "timelines",
        "predictions",
        "dashboards",
        "dashboards/players"
    ]
    
    for dir_name in directories:
        dir_path = challenger_path / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description='Initialize challenger directory structure for V4')
    parser.add_argument('--country', type=str, default='England', help='Country name (default: England)')
    parser.add_argument('--clubs', type=str, default=None, help='Comma-separated list of clubs (default: all clubs)')
    parser.add_argument('--force', action='store_true', help='Overwrite existing configs')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("INITIALIZE CHALLENGER STRUCTURE FOR V4")
    print("=" * 70)
    print(f"Country: {args.country}")
    print()
    
    # Create challenger root directory
    challenger_root = PRODUCTION_ROOT / "deployments" / args.country / "challenger"
    challenger_root.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Created challenger root: {challenger_root}")
    print()
    
    # Get clubs to process
    if args.clubs:
        clubs = [c.strip() for c in args.clubs.split(',')]
    else:
        clubs = get_all_v3_clubs(args.country)
    
    print(f"[INFO] Processing {len(clubs)} clubs")
    print()
    
    # Process each club
    successful = 0
    failed = 0
    
    for club in clubs:
        print(f"[PROCESSING] {club}...")
        
        # Copy config
        if not copy_config_to_challenger(args.country, club):
            failed += 1
            continue
        
        # Create directories
        create_club_directories(args.country, club)
        print(f"  [OK] Created directories for {club}")
        
        successful += 1
        print()
    
    # Summary
    print("=" * 70)
    print("INITIALIZATION COMPLETE")
    print("=" * 70)
    print(f"[OK] Successful: {successful}/{len(clubs)}")
    print(f"[FAIL] Failed: {failed}/{len(clubs)}")
    print()
    print(f"[INFO] Challenger structure ready at: {challenger_root}")
    print("=" * 70)
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
