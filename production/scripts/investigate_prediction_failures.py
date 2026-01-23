#!/usr/bin/env python3
"""
Investigate prediction generation failures for specific clubs.
Checks for common issues like missing files, data format problems, etc.
"""

import sys
import io
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent

def check_club_prediction_readiness(country: str, club: str) -> Dict:
    """
    Check if a club is ready for prediction generation.
    Returns a dictionary with check results.
    """
    results = {
        'club': club,
        'status': 'OK',
        'issues': [],
        'warnings': [],
        'info': []
    }
    
    club_path = PRODUCTION_ROOT / "deployments" / country / club
    
    # Check 1: Config file exists
    config_path = club_path / "config.json"
    if not config_path.exists():
        results['status'] = 'FAILED'
        results['issues'].append(f"Config file not found: {config_path}")
        return results
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        player_ids = config.get('player_ids', [])
        results['info'].append(f"Found {len(player_ids)} players in config")
    except Exception as e:
        results['status'] = 'FAILED'
        results['issues'].append(f"Error reading config file: {e}")
        return results
    
    # Check 2: Timelines file exists
    timelines_file = club_path / "timelines" / "timelines_35day_season_2025_2026_v4_muscular.csv"
    if not timelines_file.exists():
        results['status'] = 'FAILED'
        results['issues'].append(f"Timelines file not found: {timelines_file}")
        return results
    
    # Check 3: Timelines file is readable and has data
    try:
        df_timelines = pd.read_csv(timelines_file, nrows=100, low_memory=False)
        if len(df_timelines) == 0:
            results['status'] = 'FAILED'
            results['issues'].append("Timelines file is empty")
            return results
        
        # Check for reference_date column
        if 'reference_date' not in df_timelines.columns:
            results['status'] = 'FAILED'
            results['issues'].append("Timelines file missing 'reference_date' column")
            return results
        
        # Check date format
        try:
            dates = pd.to_datetime(df_timelines['reference_date'].head(10), errors='coerce')
            na_count = dates.isna().sum()
            if na_count > 0:
                results['warnings'].append(f"Found {na_count} invalid dates in sample")
        except Exception as e:
            results['warnings'].append(f"Date parsing issue: {e}")
        
        # Check total rows
        total_rows = sum(1 for _ in open(timelines_file, 'r', encoding='utf-8-sig')) - 1
        results['info'].append(f"Timelines file has approximately {total_rows:,} rows")
        
    except Exception as e:
        results['status'] = 'FAILED'
        results['issues'].append(f"Error reading timelines file: {e}")
        return results
    
    # Check 4: Daily features directory exists
    daily_features_dir = club_path / "daily_features"
    if not daily_features_dir.exists():
        results['status'] = 'FAILED'
        results['issues'].append(f"Daily features directory not found: {daily_features_dir}")
        return results
    
    # Check 5: Daily features files exist for players
    missing_daily_features = []
    for player_id in player_ids[:10]:  # Check first 10 players
        daily_features_file = daily_features_dir / f"player_{player_id}_daily_features.csv"
        if not daily_features_file.exists():
            missing_daily_features.append(player_id)
    
    if missing_daily_features:
        results['warnings'].append(f"Missing daily features for {len(missing_daily_features)} players (sample): {missing_daily_features[:5]}")
    
    # Check 6: Model files exist
    model_dir = PRODUCTION_ROOT / "models" / "lgbm_muscular_v2"
    model_file = model_dir / "model.joblib"
    columns_file = model_dir / "columns.json"
    
    if not model_file.exists():
        results['status'] = 'FAILED'
        results['issues'].append(f"Model file not found: {model_file}")
    
    if not columns_file.exists():
        results['status'] = 'FAILED'
        results['issues'].append(f"Columns file not found: {columns_file}")
    
    # Check 7: Predictions directory exists (or can be created)
    predictions_dir = club_path / "predictions"
    if not predictions_dir.exists():
        try:
            predictions_dir.mkdir(parents=True, exist_ok=True)
            results['info'].append("Created predictions directory")
        except Exception as e:
            results['warnings'].append(f"Cannot create predictions directory: {e}")
    
    # Check 8: Cache directory exists
    cache_dir = PRODUCTION_ROOT / "cache"
    if not cache_dir.exists():
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            results['info'].append("Created cache directory")
        except Exception as e:
            results['warnings'].append(f"Cannot create cache directory: {e}")
    
    return results

def main():
    # Clubs that failed predictions
    failed_clubs = [
        "Aston Villa",
        "Everton FC",
        "Fulham FC",
        "Manchester City"
    ]
    
    country = "England"
    
    print("=" * 80)
    print("INVESTIGATING PREDICTION FAILURES")
    print("=" * 80)
    print()
    
    all_results = []
    
    for club in failed_clubs:
        print(f"\n{'=' * 80}")
        print(f"Checking: {club}")
        print(f"{'=' * 80}")
        
        results = check_club_prediction_readiness(country, club)
        all_results.append(results)
        
        print(f"\nStatus: {results['status']}")
        
        if results['issues']:
            print(f"\nIssues ({len(results['issues'])}):")
            for issue in results['issues']:
                print(f"  - {issue}")
        
        if results['warnings']:
            print(f"\nWarnings ({len(results['warnings'])}):")
            for warning in results['warnings']:
                print(f"  - {warning}")
        
        if results['info']:
            print(f"\nInfo:")
            for info in results['info']:
                print(f"  - {info}")
    
    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    
    failed_count = sum(1 for r in all_results if r['status'] == 'FAILED')
    ok_count = len(all_results) - failed_count
    
    print(f"\nClubs checked: {len(all_results)}")
    print(f"  OK: {ok_count}")
    print(f"  FAILED: {failed_count}")
    
    if failed_count > 0:
        print(f"\nFailed clubs:")
        for result in all_results:
            if result['status'] == 'FAILED':
                print(f"  - {result['club']}: {len(result['issues'])} issue(s)")
                for issue in result['issues']:
                    print(f"      * {issue}")

if __name__ == "__main__":
    main()







