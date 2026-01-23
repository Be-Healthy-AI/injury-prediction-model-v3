#!/usr/bin/env python3
"""
Test script for FBRef scraper using Cole Palmer as test case.
"""

import sys
from pathlib import Path

# Add scripts directory to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_collection.fbref_scraper import FBRefConfig, FBRefScraper
from scripts.data_collection.fbref_transformers import transform_fbref_match_log


def test_cole_palmer():
    """Test FBRef scraper with Cole Palmer (FBRef ID: dc7f8a28)."""
    print("=" * 70)
    print("FBRef Scraper Test - Cole Palmer")
    print("=" * 70)
    
    # Initialize scraper
    config = FBRefConfig()
    scraper = FBRefScraper(config)
    
    try:
        # Test 1: Fetch player profile
        print("\n[Test 1] Fetching player profile...")
        fbref_id = "dc7f8a28"
        profile = scraper.fetch_player_profile(fbref_id)
        print(f"Profile data:")
        for key, value in profile.items():
            print(f"  {key}: {value}")
        
        # Test 2: Search for player
        print("\n[Test 2] Searching for 'Cole Palmer'...")
        search_results = scraper.search_player("Cole Palmer")
        print(f"Found {len(search_results)} results:")
        for i, result in enumerate(search_results[:5], 1):  # Show first 5
            print(f"  {i}. {result['name']} (ID: {result['fbref_id']}, Club: {result.get('club', 'N/A')})")
        
        # Test 3: Fetch match logs for 2024-25 season
        print("\n[Test 3] Fetching match logs for 2024-25 season (Premier League)...")
        matches_2024 = scraper._fetch_season_match_logs(
            fbref_id=fbref_id,
            season=2024,
            competition="Premier-League"
        )
        if not matches_2024.empty:
            print(f"[OK] Successfully fetched {len(matches_2024)} raw matches")
            print(f"Raw columns: {list(matches_2024.columns)[:10]}...")  # Show first 10 columns
            
            # Test transformer
            print("\n[Test 3b] Transforming match data...")
            transformed = transform_fbref_match_log(matches_2024, fbref_id)
            if not transformed.empty:
                print(f"[OK] Successfully transformed {len(transformed)} matches")
                print(f"Transformed columns: {list(transformed.columns)[:15]}...")
                print(f"\nFirst few transformed matches:")
                # Select only columns that exist
                display_cols = ['match_date', 'team', 'opponent', 'minutes']
                for col in ['goals', 'assists', 'pass_accuracy_pct', 'xG']:
                    if col in transformed.columns:
                        display_cols.append(col)
                print(transformed[display_cols].head(3).to_string())
            else:
                print("[WARNING] Transformer returned empty DataFrame")
        else:
            print("[WARNING] No matches found for 2024-25 season")
        
        # Test 4: Fetch all available match logs
        print("\n[Test 4] Fetching all available match logs (all seasons)...")
        all_matches = scraper.fetch_player_match_logs(fbref_id=fbref_id)
        if not all_matches.empty:
            print(f"[OK] Successfully fetched {len(all_matches)} total matches")
            print(f"Seasons covered: {sorted(all_matches['season'].unique()) if 'season' in all_matches.columns else 'N/A'}")
            print(f"Competitions: {sorted(all_matches['competition'].unique()) if 'competition' in all_matches.columns else 'N/A'}")
        else:
            print("[WARNING] No matches found")
        
        print("\n" + "=" * 70)
        print("All tests completed!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n[ERROR] Error during testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        scraper.close()


if __name__ == "__main__":
    test_cole_palmer()

