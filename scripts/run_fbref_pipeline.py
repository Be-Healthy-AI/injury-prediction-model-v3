#!/usr/bin/env python3
"""
Scrape FBRef data for players and export match statistics.

This script:
1. Loads TransferMarkt players from existing data
2. Maps players to FBRef IDs
3. Scrapes FBRef match statistics for all seasons
4. Transforms and exports to data_exports/fbref/
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_collection.fbref_scraper import FBRefConfig, FBRefScraper
from scripts.data_collection.fbref_transformers import transform_fbref_match_log
from scripts.data_collection.player_mapper import PlayerMapper


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--country",
        required=True,
        help="Country name (e.g., 'england')",
    )
    parser.add_argument(
        "--as-of-date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="ISO date for output folder naming (default: today)",
    )
    parser.add_argument(
        "--tm-data-dir",
        required=True,
        help="Path to TransferMarkt data directory (e.g., data_exports/transfermarkt/england/20251205)",
    )
    parser.add_argument(
        "--min-season",
        type=int,
        default=2009,
        help="Minimum season year to scrape (default: 2009)",
    )
    parser.add_argument(
        "--max-players",
        type=int,
        help="Maximum number of players to process (for testing)",
    )
    parser.add_argument(
        "--output-root",
        default="data_exports/fbref",
        help="Root directory for FBRef data exports",
    )
    parser.add_argument(
        "--force-remap",
        action="store_true",
        help="Force re-mapping even if mapping exists",
    )
    return parser.parse_args(argv)


def load_transfermarkt_players(data_dir: Path) -> pd.DataFrame:
    """Load TransferMarkt players profile data."""
    profile_file = data_dir / "players_profile.csv"
    if not profile_file.exists():
        # Try alternative naming
        profile_file = data_dir / f"{data_dir.name}_players_profile.csv"
    
    if not profile_file.exists():
        raise FileNotFoundError(f"Could not find players_profile.csv in {data_dir}")
    
    # Try to read with different encodings and separators
    # FBRef CSV files may use semicolon or comma separators
    try:
        # Try semicolon first (common in European CSV files)
        df = pd.read_csv(profile_file, encoding='utf-8-sig', sep=';', on_bad_lines='skip')
    except:
        try:
            df = pd.read_csv(profile_file, encoding='utf-8-sig', sep=',', on_bad_lines='skip')
        except:
            try:
                df = pd.read_csv(profile_file, encoding='latin-1', sep=';', on_bad_lines='skip')
            except:
                try:
                    df = pd.read_csv(profile_file, encoding='latin-1', sep=',', on_bad_lines='skip')
                except:
                    # Last resort: try with error_bad_lines=False (pandas < 2.0)
                    try:
                        df = pd.read_csv(profile_file, sep=';', error_bad_lines=False, warn_bad_lines=False)
                    except:
                        df = pd.read_csv(profile_file, sep=',', error_bad_lines=False, warn_bad_lines=False)
    
    # Parse date_of_birth if column exists
    if 'date_of_birth' in df.columns:
        df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce')
    
    print(f"Loaded {len(df)} players from TransferMarkt data")
    return df


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    
    # Setup output directory
    as_of = datetime.fromisoformat(args.as_of_date)
    output_dir = Path(args.output_root) / args.country / as_of.strftime("%Y%m%d")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup match stats directory
    match_stats_dir = output_dir / "match_stats"
    match_stats_dir.mkdir(exist_ok=True)
    
    # Initialize components
    fbref_scraper = FBRefScraper(FBRefConfig())
    mapping_file = output_dir / "players_mapping.csv"
    mapper = PlayerMapper(mapping_file=mapping_file, fbref_scraper=fbref_scraper)
    
    # Load TransferMarkt players
    tm_data_dir = Path(args.tm_data_dir)
    if not tm_data_dir.exists():
        raise FileNotFoundError(f"TransferMarkt data directory not found: {tm_data_dir}")
    
    tm_players = load_transfermarkt_players(tm_data_dir)
    
    # Limit players if specified
    if args.max_players:
        tm_players = tm_players.head(args.max_players)
        print(f"Limited to {args.max_players} players for testing")
    
    print(f"\n{'='*70}")
    print(f"FBRef Data Extraction Pipeline")
    print(f"{'='*70}")
    print(f"Country: {args.country}")
    print(f"Output directory: {output_dir}")
    print(f"Total players to process: {len(tm_players)}")
    print(f"{'='*70}\n")
    
    # Process each player
    mapped_count = 0
    scraped_count = 0
    failed_count = 0
    
    for idx, (_, player) in enumerate(tm_players.iterrows(), 1):
        tm_id = int(player['id'])
        tm_name = player.get('name', 'Unknown')
        tm_dob = player.get('date_of_birth')
        
        print(f"[{idx}/{len(tm_players)}] Processing: {tm_name} (TM ID: {tm_id})")
        
        # Map to FBRef
        if args.force_remap:
            # Clear existing mapping
            if not mapper.mappings.empty:
                mapper.mappings = mapper.mappings[mapper.mappings['transfermarkt_id'] != tm_id]
        
        fbref_match = mapper.find_fbref_player(
            tm_id=tm_id,
            tm_name=tm_name,
            tm_dob=tm_dob,
            tm_clubs=None,  # Could extract from career data if needed
            tm_seasons=None,
        )
        
        if not fbref_match:
            print(f"  [SKIP] No FBRef mapping found")
            failed_count += 1
            continue
        
        mapped_count += 1
        fbref_id = fbref_match['fbref_id']
        confidence = fbref_match['confidence']
        
        print(f"  [MAP] FBRef ID: {fbref_id} (confidence: {confidence:.2f}, method: {fbref_match['method']})")
        
        # Check if match stats already exist
        match_stats_file = match_stats_dir / f"player_{fbref_id}_matches.csv"
        if match_stats_file.exists() and not args.force_remap:
            print(f"  [SKIP] Match stats already exist: {match_stats_file.name}")
            scraped_count += 1
            continue
        
        # Scrape match logs
        try:
            print(f"  [SCRAPE] Fetching match logs...", end=" ", flush=True)
            raw_matches = fbref_scraper.fetch_player_match_logs(fbref_id=fbref_id)
            
            if raw_matches.empty:
                print("[WARNING] No matches found")
                failed_count += 1
                continue
            
            print(f"[OK] Found {len(raw_matches)} raw match rows")
            
            # Transform match data
            print(f"  [TRANSFORM] Transforming match data...", end=" ", flush=True)
            transformed_matches = transform_fbref_match_log(raw_matches, fbref_id)
            
            if transformed_matches.empty:
                print("[WARNING] No valid matches after transformation")
                failed_count += 1
                continue
            
            print(f"[OK] {len(transformed_matches)} valid matches")
            
            # Save to CSV
            transformed_matches.to_csv(
                match_stats_file,
                index=False,
                encoding='utf-8-sig'
            )
            print(f"  [SAVE] Saved to {match_stats_file.name}")
            scraped_count += 1
            
        except Exception as e:
            print(f"  [ERROR] Failed to scrape/transform: {e}")
            failed_count += 1
            import traceback
            traceback.print_exc()
            continue
        
        print()  # Blank line between players
    
    # Final summary
    print(f"\n{'='*70}")
    print("Extraction Summary")
    print(f"{'='*70}")
    print(f"Total players processed: {len(tm_players)}")
    print(f"Successfully mapped: {mapped_count}")
    print(f"Successfully scraped: {scraped_count}")
    print(f"Failed/Skipped: {failed_count}")
    print(f"Mapping coverage: {mapped_count/len(tm_players)*100:.1f}%")
    print(f"Scraping coverage: {scraped_count/len(tm_players)*100:.1f}%")
    print(f"{'='*70}\n")
    
    # Save final mapping file
    mapper._save_mappings()
    print(f"Saved player mappings to: {mapping_file}")
    print(f"\n{'='*70}")
    print("Output Files Location")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print(f"Player mappings: {mapping_file}")
    print(f"Match statistics: {match_stats_dir}")
    print(f"  - Files: player_<fbref_id>_matches.csv")
    print(f"  - Example: {match_stats_dir / 'player_dc7f8a28_matches.csv'}")
    print(f"{'='*70}\n")
    
    # Cleanup
    mapper.close()
    fbref_scraper.close()


if __name__ == "__main__":
    main()

