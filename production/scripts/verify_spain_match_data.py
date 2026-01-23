#!/usr/bin/env python3
"""
Verify match data completeness for players.
Checks that we have match files for each player for all seasons available on Transfermarkt.
Uses the actual season dropdown from Transfermarkt performance details page.
"""

import argparse
import sys
import io
# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Optional
import sys

# Add scripts directory to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent
ROOT_DIR = PRODUCTION_ROOT.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.data_collection.transfermarkt_scraper import (
    ScraperConfig,
    TransfermarktScraper,
)

def get_available_seasons_from_transfermarkt(
    scraper: TransfermarktScraper,
    player_slug: Optional[str],
    player_id: int,
    current_season_key: int
) -> List[int]:
    """
    Extract available seasons from Transfermarkt performance details page dropdown.
    This is the most accurate way to determine which seasons have match data.
    """
    slug = player_slug or "spieler"
    url = (
        f"{scraper.config.base_url}/{slug}/leistungsdatendetails/spieler/{player_id}"
        "/verein/0/liga/0/wettbewerb//pos/0/trainer_id/0/plus/1"
    )
    
    try:
        soup = scraper._fetch_soup(url)
        
        # Find the season dropdown - try multiple selectors
        season_select = (
            soup.find('select', {'name': 'saison_id'}) or 
            soup.find('select', id='saison_id') or
            soup.find('select', class_=lambda x: x and 'saison' in x.lower()) or
            soup.find('select', attrs={'data-placeholder': lambda x: x and 'season' in x.lower() if x else False})
        )
        
        seasons = []
        if season_select:
            for option in season_select.find_all('option'):
                value = option.get('value')
                if value and value.isdigit():
                    season_year = int(value)
                    # Only include seasons up to current season
                    if season_year <= current_season_key:
                        seasons.append(season_year)
        
        # If dropdown not found, try alternative approach: look for season links in the page
        if not seasons:
            # Look for season links in the page (e.g., links with /saison/ in href)
            season_links = soup.find_all('a', href=lambda x: x and '/saison/' in str(x))
            for link in season_links:
                href = link.get('href', '')
                # Extract season from href like /saison/2025
                import re
                match = re.search(r'/saison/(\d+)', href)
                if match:
                    season_year = int(match.group(1))
                    if season_year <= current_season_key:
                        seasons.append(season_year)
        
        return sorted(list(set(seasons)), reverse=True)
    except Exception as e:
        print(f"      Warning: Could not fetch seasons for player {player_id}: {e}")
        return []

def check_match_file_exists(player_id: int, season: int, current_season_key: int, 
                            current_season_dir: Path, previous_seasons_dir: Path) -> bool:
    """Check if match file exists for player and season."""
    season_label = f"{season}_{season + 1}"
    filename = f"match_{player_id}_{season_label}.csv"
    
    # Check current season folder
    if season == current_season_key:
        return (current_season_dir / filename).exists()
    else:
        # Check previous seasons folder
        return (previous_seasons_dir / filename).exists()

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Verify match data completeness for players in a league"
    )
    parser.add_argument(
        "--country",
        required=True,
        help="Country name (e.g., 'Spain', 'Saudi Arabia') - used to locate data folder"
    )
    parser.add_argument(
        "--as-of-date",
        required=True,
        help="Data extraction date in YYYYMMDD format (e.g., '20260106')"
    )
    parser.add_argument(
        "--league",
        default="",
        help="League name for display purposes (optional)"
    )
    return parser.parse_args()

def _get_current_season_key(as_of_date: datetime) -> int:
    """Determine current season key from date."""
    # Season starts in summer, so if month >= 7, we're in the new season
    if as_of_date.month >= 7:
        return as_of_date.year
    else:
        return as_of_date.year - 1

def main():
    args = parse_args()
    
    # Parse as-of-date
    try:
        if len(args.as_of_date) == 8 and args.as_of_date.isdigit():
            as_of = datetime.strptime(args.as_of_date, "%Y%m%d")
        else:
            as_of = datetime.fromisoformat(args.as_of_date)
    except ValueError:
        print(f"ERROR: Invalid date format: {args.as_of_date}. Use YYYYMMDD format.")
        return 1
    
    # Setup paths
    country_folder = args.country.lower().replace(" ", "_")
    DATA_DIR = PRODUCTION_ROOT / "raw_data" / country_folder / args.as_of_date
    CURRENT_SEASON_DIR = DATA_DIR / "match_data"
    PREVIOUS_SEASONS_DIR = PRODUCTION_ROOT / "raw_data" / country_folder / "previous_seasons"
    
    # Determine current season
    CURRENT_SEASON_KEY = _get_current_season_key(as_of)
    CURRENT_YEAR = as_of.year
    
    # League name for display
    league_name = args.league or args.country
    
    print("=" * 80)
    print(f"VERIFYING MATCH DATA COMPLETENESS FOR {league_name.upper()} PLAYERS")
    print("Using Transfermarkt season dropdown as source of truth")
    print("=" * 80)
    print(f"Country: {args.country}")
    print(f"Data folder: {DATA_DIR}")
    print(f"Current season: {CURRENT_SEASON_KEY}/{CURRENT_SEASON_KEY + 1}")
    print()
    
    # Check if data directory exists
    if not DATA_DIR.exists():
        print(f"ERROR: Data directory not found: {DATA_DIR}")
        return 1
    
    # Initialize scraper
    scraper = TransfermarktScraper(ScraperConfig())
    
    try:
        # Load data
        profiles_path = DATA_DIR / "players_profile.csv"
        if not profiles_path.exists():
            print(f"ERROR: Profiles file not found: {profiles_path}")
            return 1
        
        print("Loading player data...")
        profiles_df = pd.read_csv(profiles_path, sep=';', encoding='utf-8-sig')
        
        print(f"  Loaded {len(profiles_df)} players")
        print()
        
        # Track statistics
        total_players = 0
        players_with_missing_seasons = []
        players_with_errors = []
        total_expected_files = 0
        total_found_files = 0
        
        print("Fetching available seasons from Transfermarkt and verifying match files...")
        print("(This may take a while as we need to fetch each player's page)")
        print()
        
        for idx, player in profiles_df.iterrows():
            player_id = int(player['id'])
            player_name = player.get('name', f'Player {player_id}')
            player_slug = player.get('slug', None)  # May not be in profile
            
            total_players += 1
            
            # Get available seasons from Transfermarkt
            print(f"  [{total_players}/{len(profiles_df)}] {player_name} (ID: {player_id})...", end=" ", flush=True)
            expected_seasons = get_available_seasons_from_transfermarkt(scraper, player_slug, player_id, CURRENT_SEASON_KEY)
            
            if not expected_seasons:
                print("No seasons found on Transfermarkt")
                players_with_errors.append({
                    'player_id': player_id,
                    'player_name': player_name,
                    'error': 'No seasons found on Transfermarkt'
                })
                continue
            
            print(f"Found {len(expected_seasons)} seasons: {expected_seasons[:5]}{'...' if len(expected_seasons) > 5 else ''}")
            
            # Check each expected season
            missing_seasons = []
            found_count = 0
            
            for season in expected_seasons:
                total_expected_files += 1
                if check_match_file_exists(player_id, season, CURRENT_SEASON_KEY, 
                                         CURRENT_SEASON_DIR, PREVIOUS_SEASONS_DIR):
                    total_found_files += 1
                    found_count += 1
                else:
                    missing_seasons.append(season)
            
            if missing_seasons:
                players_with_missing_seasons.append({
                    'player_id': player_id,
                    'player_name': player_name,
                    'expected_seasons': len(expected_seasons),
                    'found_seasons': found_count,
                    'missing_seasons': missing_seasons
                })
            
            # Progress indicator
            if total_players % 10 == 0:
                print(f"    Progress: {total_players}/{len(profiles_df)} players processed...")
    finally:
        scraper.close()
    
        print()
        print("=" * 80)
        print("VERIFICATION RESULTS")
        print("=" * 80)
        print(f"Total players checked: {total_players}")
        print(f"Total expected match files (from Transfermarkt): {total_expected_files}")
        print(f"Total found match files: {total_found_files}")
        print(f"Missing match files: {total_expected_files - total_found_files}")
        if total_expected_files > 0:
            print(f"Completeness: {total_found_files / total_expected_files * 100:.2f}%")
        if players_with_errors:
            print(f"Players with errors: {len(players_with_errors)}")
        print()
        
        if players_with_missing_seasons:
            print(f"WARNING: Players with missing seasons: {len(players_with_missing_seasons)}")
            print()
            print("Top 20 players with most missing seasons:")
            print("-" * 80)
            
            sorted_players = sorted(players_with_missing_seasons, 
                                  key=lambda x: len(x['missing_seasons']), 
                                  reverse=True)[:20]
            
            for player_info in sorted_players:
                print(f"  {player_info['player_name']} (ID: {player_info['player_id']})")
                print(f"    Expected: {player_info['expected_seasons']} seasons")
                print(f"    Found: {player_info['found_seasons']} seasons")
                print(f"    Missing: {len(player_info['missing_seasons'])} seasons")
                if len(player_info['missing_seasons']) <= 10:
                    print(f"    Missing seasons: {player_info['missing_seasons']}")
                else:
                    print(f"    Missing seasons: {player_info['missing_seasons'][:10]} ... ({len(player_info['missing_seasons'])} total)")
                print()
            
            # Save detailed report - convert missing_seasons list to string for CSV
            report_data = []
            for player_info in players_with_missing_seasons:
                report_data.append({
                    'player_id': player_info['player_id'],
                    'player_name': player_info['player_name'],
                    'expected_seasons': player_info['expected_seasons'],
                    'found_seasons': player_info['found_seasons'],
                    'missing_count': len(player_info['missing_seasons']),
                    'missing_seasons': ','.join(map(str, player_info['missing_seasons']))
                })
            
            report_df = pd.DataFrame(report_data)
            report_path = DATA_DIR / "match_data_verification_report.csv"
            report_df.to_csv(report_path, index=False, encoding='utf-8-sig', sep=';')
            print(f"Detailed report saved to: {report_path}")
        
        return 0
        else:
            print("SUCCESS: All players have complete match data!")
        
        if players_with_errors:
            print()
            print(f"Players with errors ({len(players_with_errors)}):")
            for error_info in players_with_errors[:10]:
                print(f"  {error_info['player_name']} (ID: {error_info['player_id']}): {error_info['error']}")
            if len(players_with_errors) > 10:
                print(f"  ... and {len(players_with_errors) - 10} more")
        
        print("=" * 80)
        return 0

if __name__ == "__main__":
    exit(main())

