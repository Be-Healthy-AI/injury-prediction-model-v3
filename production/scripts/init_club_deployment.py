#!/usr/bin/env python3
"""
Initialize a new club deployment.

Creates folder structure and default configuration file for a new country/club.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Calculate paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent


def get_club_path(country: str, club: str) -> Path:
    """Get the base path for a specific club deployment."""
    return PRODUCTION_ROOT / "deployments" / country / club


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
        '--player-ids',
        type=int,
        nargs='*',
        help='Initial list of player IDs for this club'
    )
    args = parser.parse_args()
    
    club_path = get_club_path(args.country, args.club)
    
    # Create folder structure
    directories = [
        club_path / "daily_features",
        club_path / "timelines",
        club_path / "predictions" / "ensemble",
        club_path / "dashboards" / "players",
        club_path / "reports",
    ]
    
    print("=" * 70)
    print("INITIALIZE CLUB DEPLOYMENT")
    print("=" * 70)
    print(f"🌍 Country: {args.country}")
    print(f" club: {args.club}")
    print(f"📂 Base path: {club_path}")
    print()
    
    print("📁 Creating folder structure...")
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory.relative_to(PRODUCTION_ROOT)}")
    
    # Create default config file
    config_file = club_path / "config.json"
    if config_file.exists():
        print(f"\n⚠️  Config file already exists: {config_file}")
        response = input("   Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("   Skipping config file creation")
            return 0
    
    config = {
        "club_name": args.club,
        "country": args.country,
        "player_ids": args.player_ids if args.player_ids else [],
        "models": {
            "random_forest": "../../../models/rf_model_combined_trainval.joblib",
            "gradient_boosting": "../../../models/gb_model_combined_trainval.joblib",
            "rf_columns": "../../../models/rf_model_combined_trainval_columns.json",
            "gb_columns": "../../../models/gb_model_combined_trainval_columns.json"
        },
        "default_date_range_days": 7
    }
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Created config file: {config_file.relative_to(PRODUCTION_ROOT)}")
    print("\n✅ Club deployment initialized successfully!")
    print(f"\n📝 Next steps:")
    print(f"   1. Copy daily features files to: {club_path / 'daily_features'}")
    print(f"   2. Update player_ids in: {config_file}")
    print(f"   3. Run: python production/scripts/deploy_club.py --country {args.country} --club \"{args.club}\"")
    
    return 0


if __name__ == "__main__":
    exit(main())



