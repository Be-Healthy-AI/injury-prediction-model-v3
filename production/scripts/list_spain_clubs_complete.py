#!/usr/bin/env python3
"""List all clubs in LaLiga 2024/25 and LaLiga2 2025/26, including excluded ones"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

from pathlib import Path
import pandas as pd
from datetime import datetime

# Add root directory to path
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent
ROOT_DIR = PRODUCTION_ROOT.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.data_collection.transfermarkt_scraper import ScraperConfig, TransfermarktScraper

def check_if_excluded(club_name: str, league: str = "") -> tuple[bool, str]:
    """Check if a club should be excluded and return reason."""
    # Note: In LaLiga2, some clubs with "B" in the name might actually be first teams
    # For now, we'll be conservative and only exclude obvious B teams
    
    # Specific B team names that should always be excluded
    specific_b_teams = [
        'Real Madrid Castilla',
        'Real Sociedad SS B'  # This is definitely a B team
    ]
    for b_team in specific_b_teams:
        if b_team in club_name:
            return True, 'B team'
    
    # For LaLiga2, "Real Sociedad B" might actually be a first team club
    # Let's check if it ends with just " B" (not " SS B")
    if club_name == 'Real Sociedad B':
        # In LaLiga2 context, this might be the actual first team
        # We'll include it for now and let the user verify
        return False, ""
    
    # Check for B teams (must end with " B" or be specific B team names)
    if club_name.endswith(' B') or club_name.endswith(' B.'):
        return True, 'B team'
    
    # Youth/U19/U23 teams
    if ' U19' in club_name or ' U23' in club_name or ' U21' in club_name:
        return True, 'Youth team'
    
    # Other reserve/youth team indicators
    exclude_patterns = [
        ('Promesas', 'Promesas team'),
        ('Vetusta', 'Vetusta team'),
        ('Mirandilla', 'Mirandilla team'),
        ('Castilla', 'Castilla team'),
        ('Juvenil', 'Juvenil team'),
        ('Youth', 'Youth team'),
        ('Reserve', 'Reserve team')
    ]
    
    for pattern, reason in exclude_patterns:
        if pattern in club_name:
            return True, reason
    return False, ""

def main():
    scraper = TransfermarktScraper(ScraperConfig())
    
    try:
        print("=" * 80)
        print("LaLiga 2024/25 (First Division)")
        print("=" * 80)
        clubs_2024_25 = scraper.fetch_league_clubs("laliga", "ES1", 2024)
        print(f"\nTotal clubs fetched from Transfermarkt: {len(clubs_2024_25)}\n")
        
        included = []
        excluded = []
        
        for club in clubs_2024_25:
            club_name = club['club_name']
            is_excluded, reason = check_if_excluded(club_name)
            
            if is_excluded:
                excluded.append((club_name, reason))
            else:
                included.append(club_name)
        
        print("INCLUDED CLUBS (First Team):")
        print("-" * 80)
        for i, club_name in enumerate(sorted(included), 1):
            print(f"{i:2d}. {club_name}")
        
        print(f"\nTotal included: {len(included)} clubs\n")
        
        if excluded:
            print("EXCLUDED CLUBS (B Teams / Youth Teams):")
            print("-" * 80)
            for i, (club_name, reason) in enumerate(sorted(excluded), 1):
                print(f"{i:2d}. {club_name} (Reason: {reason})")
            print(f"\nTotal excluded: {len(excluded)} clubs")
        
        print("\n" + "=" * 80)
        print("LaLiga2 2025/26 (Second Division)")
        print("=" * 80)
        clubs_2025_26 = scraper.fetch_league_clubs("segunda-division", "ES2", 2025)
        print(f"\nTotal clubs fetched from Transfermarkt: {len(clubs_2025_26)}\n")
        
        included_2 = []
        excluded_2 = []
        
        for club in clubs_2025_26:
            club_name = club['club_name']
            is_excluded, reason = check_if_excluded(club_name)
            
            if is_excluded:
                excluded_2.append((club_name, reason))
            else:
                included_2.append(club_name)
        
        print("ALL CLUBS FETCHED (in order from Transfermarkt):")
        print("-" * 80)
        for i, club in enumerate(clubs_2025_26, 1):
            club_name = club['club_name']
            is_excluded, reason = check_if_excluded(club_name, "LaLiga2")
            status = f"[EXCLUDED: {reason}]" if is_excluded else "[INCLUDED]"
            print(f"{i:2d}. {club_name} {status}")
        
        print("\n" + "=" * 80)
        print("INCLUDED CLUBS (First Team):")
        print("-" * 80)
        for i, club_name in enumerate(sorted(included_2), 1):
            print(f"{i:2d}. {club_name}")
        
        print(f"\nTotal included: {len(included_2)} clubs\n")
        
        if excluded_2:
            print("EXCLUDED CLUBS (B Teams / Youth Teams):")
            print("-" * 80)
            for i, (club_name, reason) in enumerate(sorted(excluded_2), 1):
                print(f"{i:2d}. {club_name} (Reason: {reason})")
            print(f"\nTotal excluded: {len(excluded_2)} clubs")
        
        # Save club information to CSV
        print("\n" + "=" * 80)
        print("Saving club information to CSV...")
        save_clubs_to_csv(clubs_2024_25, clubs_2025_26, included, included_2)
        
    finally:
        scraper.close()

def save_clubs_to_csv(clubs_2024_25, clubs_2025_26, included_2024_25, included_2025_26):
    """Save club information to CSV file in the Spain data folder."""
    # Get the Spain data folder
    spain_data_dir = PRODUCTION_ROOT / "raw_data" / "spain"
    spain_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a timestamped folder for this listing
    today = datetime.now().strftime("%Y%m%d")
    output_dir = spain_data_dir / f"{today}_clubs_listing"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for CSV
    clubs_data = []
    
    # Add LaLiga 2024/25 clubs
    for club in clubs_2024_25:
        club_name = club['club_name']
        is_included = club_name in included_2024_25
        clubs_data.append({
            'club_name': club_name,
            'club_id': club['club_id'],
            'club_slug': club['club_slug'],
            'season': '2024/25',
            'league': 'LaLiga',
            'division': 'First Division',
            'included': 'Yes' if is_included else 'No',
            'notes': '' if is_included else 'Excluded (B team/Youth team)'
        })
    
    # Add LaLiga2 2025/26 clubs
    for club in clubs_2025_26:
        club_name = club['club_name']
        is_included = club_name in included_2025_26
        clubs_data.append({
            'club_name': club_name,
            'club_id': club['club_id'],
            'club_slug': club['club_slug'],
            'season': '2025/26',
            'league': 'LaLiga2',
            'division': 'Second Division',
            'included': 'Yes' if is_included else 'No',
            'notes': '' if is_included else 'Excluded (B team/Youth team)'
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(clubs_data)
    output_file = output_dir / "spain_clubs_2024_25_2025_26.csv"
    df.to_csv(output_file, index=False, sep=';', encoding='utf-8-sig')
    
    print(f"  Saved to: {output_file}")
    print(f"  Total clubs: {len(clubs_data)}")
    print(f"  - LaLiga 2024/25: {len(clubs_2024_25)} clubs ({len(included_2024_25)} included)")
    print(f"  - LaLiga2 2025/26: {len(clubs_2025_26)} clubs ({len(included_2025_26)} included)")
    
    # Also save a summary file with just included clubs
    included_clubs = df[df['included'] == 'Yes'].copy()
    summary_file = output_dir / "spain_clubs_included_only.csv"
    included_clubs.to_csv(summary_file, index=False, sep=';', encoding='utf-8-sig')
    print(f"  Summary (included only): {summary_file}")
    print(f"  Total included clubs: {len(included_clubs)}")

if __name__ == "__main__":
    main()

