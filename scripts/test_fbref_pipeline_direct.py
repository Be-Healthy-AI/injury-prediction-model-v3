#!/usr/bin/env python3
"""
Test FBRef pipeline with a known player (direct FBRef ID).
This bypasses the search issue and tests the full pipeline.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_collection.fbref_scraper import FBRefConfig, FBRefScraper
from scripts.data_collection.fbref_transformers import transform_fbref_match_log

def test_full_pipeline():
    """Test full pipeline with Cole Palmer (known FBRef ID)."""
    print("=" * 70)
    print("FBRef Full Pipeline Test - Cole Palmer")
    print("=" * 70)
    
    # Known values
    fbref_id = "dc7f8a28"
    tm_id = 123456  # Dummy TM ID for testing
    
    # Setup output directory
    output_dir = Path("data_exports/fbref/test_direct")
    output_dir.mkdir(parents=True, exist_ok=True)
    match_stats_dir = output_dir / "match_stats"
    match_stats_dir.mkdir(exist_ok=True)
    
    # Initialize scraper
    scraper = FBRefScraper(FBRefConfig())
    
    try:
        print(f"\n[1] Fetching all match logs for Cole Palmer (FBRef ID: {fbref_id})...")
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
        print(f"\nColumns ({len(transformed.columns)}): {list(transformed.columns)[:20]}...")
        
        # Show sample data
        print(f"\n[3] Sample data:")
        display_cols = ['match_date', 'team', 'opponent', 'minutes']
        for col in ['goals', 'assists', 'pass_accuracy_pct', 'xG', 'touches']:
            if col in transformed.columns:
                display_cols.append(col)
        print(transformed[display_cols].head(5).to_string())
        
        # Save to file
        output_file = match_stats_dir / f"player_{fbref_id}_matches.csv"
        transformed.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"\n[4] Saved to: {output_file}")
        print(f"    File size: {output_file.stat().st_size / 1024:.2f} KB")
        print(f"    Rows: {len(transformed)}")
        print(f"    Columns: {len(transformed.columns)}")
        print(f"    Date range: {transformed['match_date'].min()} to {transformed['match_date'].max()}")
        
        # Show statistics
        print(f"\n[5] Data Statistics:")
        numeric_cols = transformed.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            print(f"    Numeric columns: {len(numeric_cols)}")
            print(f"    Sample stats:")
            for col in ['goals', 'assists', 'minutes', 'pass_accuracy_pct', 'xG']:
                if col in transformed.columns:
                    non_null = transformed[col].notna().sum()
                    if non_null > 0:
                        mean_val = transformed[col].mean()
                        print(f"      {col}: {non_null} non-null values, mean: {mean_val:.2f}")
        
        print(f"\n{'='*70}")
        print("[SUCCESS] Full pipeline test completed!")
        print(f"{'='*70}")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        scraper.close()

if __name__ == "__main__":
    test_full_pipeline()









