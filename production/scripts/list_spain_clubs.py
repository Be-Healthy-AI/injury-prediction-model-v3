#!/usr/bin/env python3
"""Quick script to list all clubs in LaLiga2"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent
ROOT_DIR = PRODUCTION_ROOT.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.data_collection.transfermarkt_scraper import (
    ScraperConfig,
    TransfermarktScraper,
)

def main():
    scraper = TransfermarktScraper(ScraperConfig())
    
    print("Fetching clubs from LaLiga2...")
    clubs = scraper.fetch_league_clubs('segunda-division', 'ES2', 2024)
    
    print(f"\nFound {len(clubs)} clubs:\n")
    for i, club in enumerate(clubs, 1):
        print(f"{i:2d}. {club['club_name']:30s} (ID: {club['club_id']:6d}, slug: {club['club_slug']})")
        if 'Leganés' in club['club_name'] or 'Leganes' in club['club_name']:
            print("     ^^^ CD LEGANÉS FOUND HERE!")
    
    scraper.close()

if __name__ == "__main__":
    main()




