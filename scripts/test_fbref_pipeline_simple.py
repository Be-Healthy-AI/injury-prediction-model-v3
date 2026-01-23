#!/usr/bin/env python3
"""
Simple test of FBRef pipeline with a known player (Cole Palmer).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_collection.fbref_scraper import FBRefConfig, FBRefScraper
from scripts.data_collection.fbref_transformers import transform_fbref_match_log

def test_with_cole_palmer():
    """Test pipeline with Cole Palmer (known FBRef ID)."""
    print("=" * 70)
    print("FBRef Pipeline Test - Cole Palmer")
    print("=" * 70)
    
    # Known values
    fbref_id = "dc7f8a28"
    tm_id = 123456  # Dummy TM ID for testing
    
    # Initialize scraper
    scraper = FBRefScraper(FBRefConfig())
    
    try:
        # Scrape match logs
        print(f"\n[1] Fetching match logs for Cole Palmer (FBRef ID: {fbref_id})...")
        raw_matches = scraper.fetch_player_match_logs(fbref_id=fbref_id)
        
        if raw_matches.empty:
            print("[ERROR] No matches found")
            return
        
        print(f"[OK] Found {len(raw_matches)} raw match rows")
        
        # Transform
        print(f"\n[2] Transforming match data...")
        transformed = transform_fbref_match_log(raw_matches, fbref_id)
        
        if transformed.empty:
            print("[ERROR] No valid matches after transformation")
            return
        
        print(f"[OK] Transformed {len(transformed)} matches")
        print(f"\nColumns: {list(transformed.columns)}")
        print(f"\nSample data:")
        print(transformed[['match_date', 'team', 'opponent', 'goals', 'assists', 'minutes', 'pass_accuracy_pct', 'xG']].head(5).to_string())
        
        # Save to test output
        output_dir = Path("data_exports/fbref/test")
        output_dir.mkdir(parents=True, exist_ok=True)
        match_stats_dir = output_dir / "match_stats"
        match_stats_dir.mkdir(exist_ok=True)
        
        output_file = match_stats_dir / f"player_{fbref_id}_matches.csv"
        transformed.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n[3] Saved to: {output_file}")
        print(f"[OK] Test completed successfully!")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        scraper.close()

if __name__ == "__main__":
    test_with_cole_palmer()









