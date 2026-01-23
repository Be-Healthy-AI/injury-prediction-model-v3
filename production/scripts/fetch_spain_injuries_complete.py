#!/usr/bin/env python3
"""
Fetch injury data for all players in LaLiga 2024/25 and LaLiga2 2025/26.
This script:
1. Loads player IDs from the players listing
2. Checks existing injury data
3. Fetches missing injury data from Transfermarkt
4. Consolidates all injury data
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

from pathlib import Path
import pandas as pd
from datetime import datetime
from typing import Set, Dict, List
from collections import defaultdict

# Add root directory to path
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent
ROOT_DIR = PRODUCTION_ROOT.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.data_collection.transfermarkt_scraper import ScraperConfig, TransfermarktScraper
from scripts.data_collection.transformers import transform_injuries

def get_latest_players_file() -> Path:
    """Get the latest players listing file."""
    spain_data_dir = PRODUCTION_ROOT / "raw_data" / "spain"
    if not spain_data_dir.exists():
        raise FileNotFoundError(f"Spain data directory not found: {spain_data_dir}")
    
    # Find all players_listing folders
    listing_folders = []
    for folder in spain_data_dir.iterdir():
        if folder.is_dir() and folder.name.endswith("_players_listing"):
            listing_folders.append(folder)
    
    if not listing_folders:
        raise FileNotFoundError("No players listing folder found. Please run fetch_spain_players.py first.")
    
    # Get the most recent one
    latest_folder = max(listing_folders, key=lambda x: x.name)
    players_file = latest_folder / "spain_players_2024_25_2025_26.csv"
    
    if not players_file.exists():
        raise FileNotFoundError(f"Players file not found: {players_file}")
    
    return players_file

def find_existing_injury_data() -> Dict[str, pd.DataFrame]:
    """Find all existing injury data files."""
    spain_data_dir = PRODUCTION_ROOT / "raw_data" / "spain"
    injury_files = {}
    
    # Look for injuries_data.csv in date folders
    for folder in spain_data_dir.iterdir():
        if folder.is_dir() and folder.name.isdigit() and len(folder.name) == 8:
            injury_file = folder / "injuries_data.csv"
            if injury_file.exists():
                try:
                    df = pd.read_csv(injury_file, sep=';', encoding='utf-8-sig')
                    if 'player_id' in df.columns or 'id' in df.columns:
                        injury_files[folder.name] = df
                except Exception as e:
                    print(f"  Warning: Could not read {injury_file}: {e}")
    
    return injury_files

def get_existing_player_ids(injury_files: Dict[str, pd.DataFrame]) -> Set[int]:
    """Extract all player IDs from existing injury data."""
    existing_ids = set()
    
    for date_folder, df in injury_files.items():
        # Check for 'player_id' or 'id' column
        if 'player_id' in df.columns:
            existing_ids.update(df['player_id'].dropna().astype(int).unique())
        elif 'id' in df.columns:
            existing_ids.update(df['id'].dropna().astype(int).unique())
    
    return existing_ids

def get_player_slug_from_players_data(players_df: pd.DataFrame, player_id: int) -> str:
    """Get player slug from players data if available."""
    player_row = players_df[players_df['player_id'] == player_id]
    if not player_row.empty and 'player_slug' in players_df.columns:
        return player_row.iloc[0].get('player_slug', '')
    return ''

def main():
    print("=" * 80)
    print("Fetching Injury Data for Spain Players")
    print("=" * 80)
    
    # Step 1: Load player IDs
    print("\nStep 1: Loading player information...")
    players_file = get_latest_players_file()
    print(f"  Loading from: {players_file}")
    players_df = pd.read_csv(players_file, sep=';', encoding='utf-8-sig')
    print(f"  Loaded {len(players_df)} player entries")
    
    # Get unique player IDs
    target_player_ids = set(players_df['player_id'].unique())
    print(f"  Unique player IDs: {len(target_player_ids)}")
    
    # Step 2: Check existing injury data
    print("\nStep 2: Checking existing injury data...")
    existing_injury_files = find_existing_injury_data()
    print(f"  Found {len(existing_injury_files)} existing injury data files")
    
    existing_player_ids = get_existing_player_ids(existing_injury_files)
    print(f"  Players with existing injury data: {len(existing_player_ids)}")
    
    # Step 3: Identify missing players
    missing_player_ids = target_player_ids - existing_player_ids
    print(f"\nStep 3: Identifying missing players...")
    print(f"  Players needing injury data: {len(missing_player_ids)}")
    
    if missing_player_ids:
        print(f"  Fetching injury data for {len(missing_player_ids)} players from Transfermarkt...")
        
        # Initialize scraper
        scraper = TransfermarktScraper(ScraperConfig())
        
        new_injuries = []
        fetched_count = 0
        error_count = 0
        
        try:
            for idx, player_id in enumerate(sorted(missing_player_ids), 1):
                # Get player info
                player_info = players_df[players_df['player_id'] == player_id].iloc[0]
                player_name = player_info.get('player_name', f'Player {player_id}')
                club_name = player_info.get('club_name', 'Unknown')
                season = player_info.get('season', 'Unknown')
                
                # Try to get player slug from the players data
                # If not available, we'll use None and the scraper will use default
                player_slug = None
                if 'player_slug' in players_df.columns:
                    slug_row = players_df[players_df['player_id'] == player_id]
                    if not slug_row.empty:
                        player_slug = slug_row.iloc[0].get('player_slug', None)
                
                print(f"\n  [{idx}/{len(missing_player_ids)}] {player_name} (ID: {player_id}, {club_name}, {season})")
                print(f"    Fetching injuries...", end=" ", flush=True)
                
                try:
                    injuries_df = scraper.fetch_player_injuries(player_slug, player_id)
                    
                    if not injuries_df.empty:
                        # Transform injuries
                        transformed_injuries = transform_injuries(player_id, injuries_df)
                        if not transformed_injuries.empty:
                            new_injuries.append(transformed_injuries)
                            print(f"[OK] {len(injuries_df)} injuries")
                            fetched_count += 1
                        else:
                            print(f"[OK] No injuries (after transformation)")
                    else:
                        print(f"[OK] No injuries")
                    
                except Exception as e:
                    print(f"[ERROR] {e}")
                    error_count += 1
                    continue
        
        finally:
            scraper.close()
        
        print(f"\n  Summary:")
        print(f"    Successfully fetched: {fetched_count} players")
        print(f"    Errors: {error_count} players")
        
        # Consolidate new injuries
        if new_injuries:
            new_injuries_df = pd.concat(new_injuries, ignore_index=True)
            print(f"    Total new injury records: {len(new_injuries_df)}")
        else:
            new_injuries_df = pd.DataFrame()
    else:
        print("  All players already have injury data!")
        new_injuries_df = pd.DataFrame()
    
    # Step 4: Consolidate all injury data
    print("\nStep 4: Consolidating all injury data...")
    
    all_injuries = []
    
    # Add existing injuries
    for date_folder, df in existing_injury_files.items():
        print(f"  Loading from {date_folder}... ({len(df)} records)")
        all_injuries.append(df)
    
    # Add new injuries
    if not new_injuries_df.empty:
        print(f"  Adding new injuries... ({len(new_injuries_df)} records)")
        all_injuries.append(new_injuries_df)
    
    if all_injuries:
        # Consolidate
        consolidated_injuries = pd.concat(all_injuries, ignore_index=True)
        
        # Remove duplicates (same player_id, fromDate, untilDate, injury_type)
        initial_count = len(consolidated_injuries)
        consolidated_injuries = consolidated_injuries.drop_duplicates(
            subset=['player_id' if 'player_id' in consolidated_injuries.columns else 'id', 
                   'fromDate', 'untilDate', 'injury_type'],
            keep='first'
        )
        duplicates_removed = initial_count - len(consolidated_injuries)
        if duplicates_removed > 0:
            print(f"  Removed {duplicates_removed} duplicate injury records")
        
        # Ensure player_id column exists (rename 'id' to 'player_id' if needed)
        if 'id' in consolidated_injuries.columns and 'player_id' not in consolidated_injuries.columns:
            consolidated_injuries['player_id'] = consolidated_injuries['id']
        
        # Save consolidated data
        spain_data_dir = PRODUCTION_ROOT / "raw_data" / "spain"
        today = datetime.now().strftime("%Y%m%d")
        output_dir = spain_data_dir / today
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "injuries_data.csv"
        consolidated_injuries.to_csv(output_file, index=False, sep=';', encoding='utf-8-sig')
        
        print(f"\n{'=' * 80}")
        print("Summary")
        print(f"{'=' * 80}")
        print(f"  Total injury records: {len(consolidated_injuries)}")
        print(f"  Unique players with injuries: {consolidated_injuries['player_id'].nunique()}")
        print(f"  Saved to: {output_file}")
        
        # Filter to only target players and show coverage
        target_injuries = consolidated_injuries[consolidated_injuries['player_id'].isin(target_player_ids)]
        print(f"\n  Coverage:")
        print(f"    Target players: {len(target_player_ids)}")
        print(f"    Players with injury data: {target_injuries['player_id'].nunique()}")
        print(f"    Coverage: {target_injuries['player_id'].nunique() / len(target_player_ids) * 100:.1f}%")
        
    else:
        print("  No injury data to consolidate!")

if __name__ == "__main__":
    main()

