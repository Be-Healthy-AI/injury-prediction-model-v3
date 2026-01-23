#!/usr/bin/env python3
"""Fetch all players from all clubs in LaLiga 2024/25 and LaLiga2 2025/26"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

from pathlib import Path
import pandas as pd
from datetime import datetime
from typing import List, Dict

# Add root directory to path
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent
ROOT_DIR = PRODUCTION_ROOT.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.data_collection.transfermarkt_scraper import ScraperConfig, TransfermarktScraper

def get_latest_clubs_file() -> Path:
    """Get the latest clubs listing file."""
    spain_data_dir = PRODUCTION_ROOT / "raw_data" / "spain"
    if not spain_data_dir.exists():
        raise FileNotFoundError(f"Spain data directory not found: {spain_data_dir}")
    
    # Find all clubs_listing folders
    listing_folders = []
    for folder in spain_data_dir.iterdir():
        if folder.is_dir() and folder.name.endswith("_clubs_listing"):
            listing_folders.append(folder)
    
    if not listing_folders:
        raise FileNotFoundError("No clubs listing folder found. Please run list_spain_clubs_complete.py first.")
    
    # Get the most recent one
    latest_folder = max(listing_folders, key=lambda x: x.name)
    clubs_file = latest_folder / "spain_clubs_included_only.csv"
    
    if not clubs_file.exists():
        raise FileNotFoundError(f"Clubs file not found: {clubs_file}")
    
    return clubs_file

def fetch_players_for_club(
    scraper: TransfermarktScraper,
    club_name: str,
    club_id: int,
    club_slug: str,
    season: str,
    season_year: int,
    league: str
) -> List[Dict]:
    """Fetch all players for a club."""
    players_data = []
    
    try:
        print(f"  Fetching players from {club_name}...", end=" ", flush=True)
        players = scraper.get_squad_players(club_slug, club_id, "kader", season_year)
        print(f"[OK] {len(players)} players")
        
        for player in players:
            players_data.append({
                'player_id': player['player_id'],
                'player_name': player.get('player_name', ''),
                'club_name': club_name,
                'club_id': club_id,
                'season': season,
                'league': league
            })
    except Exception as e:
        print(f"[ERROR] {e}")
    
    return players_data

def main():
    print("=" * 80)
    print("Fetching Players from LaLiga 2024/25 and LaLiga2 2025/26")
    print("=" * 80)
    
    # Load clubs information
    clubs_file = get_latest_clubs_file()
    print(f"\nLoading clubs from: {clubs_file}")
    clubs_df = pd.read_csv(clubs_file, sep=';', encoding='utf-8-sig')
    print(f"  Loaded {len(clubs_df)} clubs")
    
    # Filter to only included clubs
    clubs_df = clubs_df[clubs_df['included'] == 'Yes'].copy()
    print(f"  Processing {len(clubs_df)} included clubs\n")
    
    # Initialize scraper
    scraper = TransfermarktScraper(ScraperConfig())
    
    all_players = []
    
    try:
        # Group by season for better organization
        for season in ['2024/25', '2025/26']:
            season_clubs = clubs_df[clubs_df['season'] == season].copy()
            
            if season_clubs.empty:
                continue
            
            league_name = season_clubs.iloc[0]['league']
            season_year = 2024 if season == '2024/25' else 2025
            
            print(f"\n{'=' * 80}")
            print(f"Processing {league_name} {season} ({len(season_clubs)} clubs)")
            print(f"{'=' * 80}\n")
            
            for idx, club_row in season_clubs.iterrows():
                club_name = club_row['club_name']
                club_id = int(club_row['club_id'])
                club_slug = club_row['club_slug']
                league = club_row['league']
                
                players = fetch_players_for_club(
                    scraper, club_name, club_id, club_slug,
                    season, season_year, league
                )
                all_players.extend(players)
        
        # Create DataFrame
        if all_players:
            players_df = pd.DataFrame(all_players)
            
            # Remove duplicates (same player_id, club_name, season)
            initial_count = len(players_df)
            players_df = players_df.drop_duplicates(subset=['player_id', 'club_name', 'season'], keep='first')
            duplicates_removed = initial_count - len(players_df)
            if duplicates_removed > 0:
                print(f"\n  Removed {duplicates_removed} duplicate player entries")
            
            # Save to CSV
            spain_data_dir = PRODUCTION_ROOT / "raw_data" / "spain"
            today = datetime.now().strftime("%Y%m%d")
            output_dir = spain_data_dir / f"{today}_players_listing"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_dir / "spain_players_2024_25_2025_26.csv"
            players_df.to_csv(output_file, index=False, sep=';', encoding='utf-8-sig')
            
            print(f"\n{'=' * 80}")
            print("Summary")
            print(f"{'=' * 80}")
            print(f"  Total players fetched: {len(players_df)}")
            print(f"  Unique player IDs: {players_df['player_id'].nunique()}")
            
            # Summary by season
            for season in ['2024/25', '2025/26']:
                season_players = players_df[players_df['season'] == season]
                if not season_players.empty:
                    league_name = season_players.iloc[0]['league']
                    print(f"  {league_name} {season}: {len(season_players)} player entries ({season_players['player_id'].nunique()} unique players)")
            
            # Summary by club
            print(f"\n  Players per club:")
            club_counts = players_df.groupby(['club_name', 'season']).agg({
                'player_id': ['count', 'nunique']
            }).reset_index()
            club_counts.columns = ['club_name', 'season', 'total_entries', 'unique_players']
            club_counts = club_counts.sort_values(['season', 'club_name'])
            
            for _, row in club_counts.iterrows():
                print(f"    {row['club_name']} ({row['season']}): {row['unique_players']} players")
            
            print(f"\n  Saved to: {output_file}")
            
            # Also save a simplified version with just player IDs and clubs
            simplified_df = players_df[['player_id', 'club_name', 'season', 'league']].copy()
            simplified_file = output_dir / "spain_players_ids_only.csv"
            simplified_df.to_csv(simplified_file, index=False, sep=';', encoding='utf-8-sig')
            print(f"  Simplified version (IDs only): {simplified_file}")
        else:
            print("\n  No players were fetched!")
    
    finally:
        scraper.close()

if __name__ == "__main__":
    main()

