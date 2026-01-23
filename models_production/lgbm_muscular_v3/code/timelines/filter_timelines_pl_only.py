#!/usr/bin/env python3
"""
Filter existing V1 timelines to only include timelines where players were at PL clubs.

This script:
1. Identifies PL clubs per season from raw match data
2. Determines PL membership periods for each player from career data
3. Filters existing timeline files to only include PL-only timelines
4. Saves filtered timelines to V3 location with same filenames
"""

import sys
import os
import re
import glob
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import pandas as pd
from datetime import timedelta

# Add project root to path
# Script is at: models_production/lgbm_muscular_v3/code/timelines/filter_timelines_pl_only.py
# So parents[4] gets us to project root
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import normalize_team_name and is_national_team from daily features
try:
    from models_production.lgbm_muscular_v1.code.daily_features.create_daily_features import (
        normalize_team_name,
        is_national_team
    )
except ImportError:
    # Fallback: define locally if import fails
    def normalize_team_name(team_name: str) -> str:
        """Normalize team name for comparison."""
        if pd.isna(team_name) or not team_name:
            return ''
        name = str(team_name).strip().lower()
        name = re.sub(r'\s+', ' ', name)
        name = re.sub(r'\b(fc|cf|sc|ac|bc|bk)\b', '', name)
        name = re.sub(r'\b(u\d+|u-\d+|youth|junior|reserve|b team)\b', '', name)
        name = re.sub(r'\s+', ' ', name).strip()
        return name
    
    def is_national_team(team_name: str) -> bool:
        """Check if team name indicates a national team."""
        if pd.isna(team_name):
            return False
        team_lower = str(team_name).lower()
        national_indicators = [
            'national', 'nacional', 'seleção', 'selecao', 'seleccion', 'seleccion',
            'national team', 'team', 'portugal', 'spain', 'france', 'germany',
            'italy', 'england', 'brazil', 'argentina', 'netherlands', 'belgium'
        ]
        return any(indicator in team_lower for indicator in national_indicators)


def build_pl_clubs_per_season(raw_match_dir: str) -> Dict[int, Set[str]]:
    """
    Build mapping of season_year -> set of PL club names from raw match data.
    
    Args:
        raw_match_dir: Directory containing raw match CSV files
        
    Returns:
        Dictionary mapping season start year to set of normalized PL club names
    """
    print("Building PL clubs per season mapping from raw match data...")
    pl_clubs_by_season = defaultdict(set)
    
    # Find all match CSV files
    match_files = glob.glob(os.path.join(raw_match_dir, "**", "*.csv"), recursive=True)
    
    if not match_files:
        print(f"  WARNING: No match files found in {raw_match_dir}")
        return {}
    
    print(f"  Found {len(match_files)} match files")
    
    for match_file in match_files:
        try:
            df = pd.read_csv(match_file, low_memory=False, encoding='utf-8-sig')
            
            if 'competition' not in df.columns or 'date' not in df.columns:
                continue
            
            # Filter Premier League matches
            pl_mask = df['competition'].str.contains('Premier League', case=False, na=False)
            pl_matches = df[pl_mask].copy()
            
            if pl_matches.empty:
                continue
            
            # Parse dates
            pl_matches['date'] = pd.to_datetime(pl_matches['date'], errors='coerce')
            pl_matches = pl_matches[pl_matches['date'].notna()]
            
            if pl_matches.empty:
                continue
            
            # Determine season year (season starts in July, so July-Dec is current year, Jan-Jun is previous year)
            pl_matches['season_year'] = pl_matches['date'].apply(
                lambda d: d.year if d.month >= 7 else d.year - 1
            )
            
            # Extract club names from home_team, away_team, and team columns
            for _, match in pl_matches.iterrows():
                season = match['season_year']
                
                for team_col in ['home_team', 'away_team', 'team']:
                    if team_col in match and pd.notna(match[team_col]):
                        club_name = str(match[team_col]).strip()
                        if club_name and not is_national_team(club_name):
                            # Store both original and normalized for matching
                            pl_clubs_by_season[season].add(club_name)
                            pl_clubs_by_season[season].add(normalize_team_name(club_name))
        
        except Exception as e:
            print(f"  WARNING: Error processing {match_file}: {e}")
            continue
    
    # Convert to regular dict and report
    result = dict(pl_clubs_by_season)
    print(f"  Found PL clubs for {len(result)} seasons")
    for season in sorted(result.keys()):
        print(f"    Season {season}: {len(result[season])} unique club names")
    
    return result


def is_club_pl_club(club_name: str, season_year: int, pl_clubs_by_season: Dict[int, Set[str]]) -> bool:
    """
    Check if a club is a PL club for a given season.
    
    Args:
        club_name: Club name to check
        season_year: Season start year
        pl_clubs_by_season: Mapping of season -> set of PL club names
        
    Returns:
        True if club is PL club in this season (or adjacent seasons for robustness)
    """
    if not club_name or pd.isna(club_name) or is_national_team(club_name):
        return False
    
    club_normalized = normalize_team_name(club_name)
    club_original = str(club_name).strip()
    
    # Check current season and adjacent seasons (for robustness)
    for check_season in [season_year - 1, season_year, season_year + 1]:
        if check_season in pl_clubs_by_season:
            pl_clubs = pl_clubs_by_season[check_season]
            # Check both original and normalized names
            if club_original in pl_clubs or club_normalized in pl_clubs:
                # Also check normalized match
                for pl_club in pl_clubs:
                    if normalize_team_name(pl_club) == club_normalized:
                        return True
    
    return False


def build_player_pl_membership_periods(
    career_file: str,
    pl_clubs_by_season: Dict[int, Set[str]]
) -> Dict[int, List[Tuple[pd.Timestamp, pd.Timestamp]]]:
    """
    Build mapping of player_id -> list of (start_date, end_date) periods when player was at PL club.
    
    The career data records transfers chronologically. Each row represents:
    - Date: Transfer date
    - To: Destination club (where player goes)
    - From: Origin club (where player came from)
    
    Args:
        career_file: Path to players_career.csv file
        pl_clubs_by_season: Mapping of season -> set of PL club names
        
    Returns:
        Dictionary mapping player_id to list of (start, end) date tuples
    """
    print("\nBuilding player PL membership periods from career data...")
    
    # Load career data
    try:
        career_df = pd.read_csv(career_file, sep=';', encoding='utf-8-sig', low_memory=False)
    except Exception as e:
        print(f"  ERROR: Failed to load career file: {e}")
        return {}
    
    # Parse dates (DD/MM/YYYY format)
    career_df['Date'] = pd.to_datetime(career_df['Date'], format='%d/%m/%Y', errors='coerce')
    
    # Filter out rows with invalid dates or missing club info
    career_df = career_df[
        career_df['Date'].notna() & 
        career_df['To'].notna() &
        (career_df['To'].astype(str).str.strip() != '')
    ].copy()
    
    if career_df.empty:
        print("  WARNING: No valid career data found")
        return {}
    
    player_pl_periods = defaultdict(list)
    
    # Group by player
    for player_id, player_career in career_df.groupby('id'):
        # Sort by date ascending (chronological order)
        player_career = player_career.sort_values('Date').reset_index(drop=True)
        
        current_pl_start = None
        current_pl_club = None
        
        for idx, row in player_career.iterrows():
            transfer_date = row['Date']
            to_club = str(row['To']).strip()
            
            if not to_club or pd.isna(transfer_date):
                continue
            
            # Determine season for this transfer
            season_year = transfer_date.year if transfer_date.month >= 7 else transfer_date.year - 1
            
            # Check if destination club is PL club
            is_pl_destination = is_club_pl_club(to_club, season_year, pl_clubs_by_season)
            
            if is_pl_destination:
                # Starting or continuing a PL period
                if current_pl_start is None:
                    # Starting a new PL period
                    current_pl_start = transfer_date
                    current_pl_club = to_club
            else:
                # Ending a PL period (transferring to non-PL club)
                if current_pl_start is not None:
                    # Close the PL period at this transfer date
                    player_pl_periods[player_id].append((current_pl_start, transfer_date))
                    current_pl_start = None
                    current_pl_club = None
        
        # If still in PL at end of career, close the period
        if current_pl_start is not None:
            # Use last transfer date + 1 year as end (or could use a far future date)
            last_date = player_career['Date'].max()
            # Extend to end of that season + buffer
            end_season_year = last_date.year if last_date.month >= 7 else last_date.year - 1
            end_date = pd.Timestamp(f'{end_season_year + 1}-06-30')  # End of season
            # For players still at PL clubs, extend to latest date in raw data (2025-12-05)
            max_future = pd.Timestamp('2025-12-05')
            end_date = max(end_date, max_future)
            player_pl_periods[player_id].append((current_pl_start, end_date))
    
    result = dict(player_pl_periods)
    print(f"  Found PL periods for {len(result)} players")
    
    # Report some statistics
    total_periods = sum(len(periods) for periods in result.values())
    print(f"  Total PL periods: {total_periods}")
    
    return result


def is_player_in_pl_on_date(
    player_id: int,
    reference_date: pd.Timestamp,
    player_pl_periods: Dict[int, List[Tuple[pd.Timestamp, pd.Timestamp]]]
) -> bool:
    """
    Check if player was at a PL club on the given reference date.
    
    Args:
        player_id: Player ID
        reference_date: Reference date to check
        player_pl_periods: Mapping of player_id -> list of (start, end) date tuples
        
    Returns:
        True if player was at PL club on reference_date
    """
    if player_id not in player_pl_periods:
        return False
    
    periods = player_pl_periods[player_id]
    reference_date_norm = reference_date.normalize()
    
    for start_date, end_date in periods:
        start_norm = pd.Timestamp(start_date).normalize()
        end_norm = pd.Timestamp(end_date).normalize()
        if start_norm <= reference_date_norm <= end_norm:
            return True
    
    return False


def filter_timeline_file(
    input_file: str,
    output_file: str,
    player_pl_periods: Dict[int, List[Tuple[pd.Timestamp, pd.Timestamp]]]
) -> Tuple[int, int]:
    """
    Filter a timeline CSV file to only include PL-only timelines.
    
    Args:
        input_file: Path to input timeline CSV file
        output_file: Path to output filtered timeline CSV file
        player_pl_periods: Mapping of player_id -> list of PL membership periods
        
    Returns:
        (original_count, filtered_count)
    """
    filename = os.path.basename(input_file)
    print(f"\n  Filtering {filename}...")
    
    # Read in chunks to handle large files
    chunk_size = 10000
    filtered_chunks = []
    original_count = 0
    
    try:
        for chunk in pd.read_csv(input_file, chunksize=chunk_size, low_memory=False, encoding='utf-8-sig'):
            original_count += len(chunk)
            
            # Convert reference_date to datetime
            chunk['reference_date'] = pd.to_datetime(chunk['reference_date'], errors='coerce')
            
            # Filter rows where player was in PL on reference_date
            mask = chunk.apply(
                lambda row: is_player_in_pl_on_date(
                    row['player_id'],
                    row['reference_date'],
                    player_pl_periods
                ) if pd.notna(row['reference_date']) else False,
                axis=1
            )
            
            filtered_chunk = chunk[mask]
            if len(filtered_chunk) > 0:
                filtered_chunks.append(filtered_chunk)
    
    except Exception as e:
        print(f"    ERROR: Failed to process {filename}: {e}")
        return original_count, 0
    
    # Combine and save
    if filtered_chunks:
        filtered_df = pd.concat(filtered_chunks, ignore_index=True)
        filtered_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        filtered_count = len(filtered_df)
        reduction_pct = (1 - filtered_count / original_count) * 100 if original_count > 0 else 0
        print(f"    Original: {original_count:,} rows")
        print(f"    Filtered: {filtered_count:,} rows ({filtered_count/original_count*100:.1f}% kept, {reduction_pct:.1f}% removed)")
    else:
        filtered_count = 0
        # Create empty file with same columns
        try:
            sample_df = pd.read_csv(input_file, nrows=1)
            empty_df = pd.DataFrame(columns=sample_df.columns)
            empty_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"    Original: {original_count:,} rows")
            print(f"    Filtered: 0 rows (100% removed - no PL timelines found)")
        except Exception as e:
            print(f"    ERROR: Failed to create empty file: {e}")
    
    return original_count, filtered_count


def main():
    """Main function to filter timelines."""
    print("=" * 80)
    print("LGBM Muscular V3: Filter Timelines to PL-Only")
    print("=" * 80)
    
    # Paths
    v1_root = PROJECT_ROOT / "models_production" / "lgbm_muscular_v1"
    v3_root = PROJECT_ROOT / "models_production" / "lgbm_muscular_v3"
    
    v1_timelines_train = v1_root / "data" / "timelines" / "train"
    v1_timelines_test = v1_root / "data" / "timelines" / "test"
    v3_timelines_train = v3_root / "data" / "timelines" / "train"
    v3_timelines_test = v3_root / "data" / "timelines" / "test"
    
    raw_match_dir = v1_root / "data" / "raw" / "match_data"
    career_file = v1_root / "data" / "raw" / "players_career.csv"
    
    # Validate paths
    if not v1_timelines_train.exists():
        print(f"ERROR: V1 train timelines directory not found: {v1_timelines_train}")
        return
    
    if not career_file.exists():
        print(f"ERROR: Career file not found: {career_file}")
        return
    
    # Step 1: Build PL clubs per season
    print("\n" + "=" * 80)
    print("Step 1: Building PL clubs per season mapping")
    print("=" * 80)
    pl_clubs_by_season = build_pl_clubs_per_season(str(raw_match_dir))
    
    if not pl_clubs_by_season:
        print("ERROR: No PL clubs found. Cannot proceed.")
        return
    
    # Step 2: Build player PL membership periods
    print("\n" + "=" * 80)
    print("Step 2: Building player PL membership periods")
    print("=" * 80)
    player_pl_periods = build_player_pl_membership_periods(str(career_file), pl_clubs_by_season)
    
    if not player_pl_periods:
        print("ERROR: No PL membership periods found. Cannot proceed.")
        return
    
    # Step 3: Filter train timelines
    print("\n" + "=" * 80)
    print("Step 3: Filtering train timelines")
    print("=" * 80)
    v3_timelines_train.mkdir(parents=True, exist_ok=True)
    
    train_files = sorted(glob.glob(str(v1_timelines_train / "*.csv")))
    
    if not train_files:
        print(f"WARNING: No train timeline files found in {v1_timelines_train}")
    else:
        print(f"Found {len(train_files)} train timeline files")
        
        total_original = 0
        total_filtered = 0
        
        for train_file in train_files:
            filename = os.path.basename(train_file)
            output_file = v3_timelines_train / filename
            
            orig, filt = filter_timeline_file(str(train_file), str(output_file), player_pl_periods)
            total_original += orig
            total_filtered += filt
        
        print(f"\n  Train Summary:")
        print(f"    Total original: {total_original:,} rows")
        print(f"    Total filtered: {total_filtered:,} rows")
        if total_original > 0:
            print(f"    Retention: {total_filtered/total_original*100:.1f}%")
    
    # Step 4: Filter test timeline
    print("\n" + "=" * 80)
    print("Step 4: Filtering test timeline")
    print("=" * 80)
    v3_timelines_test.mkdir(parents=True, exist_ok=True)
    
    test_file = v1_timelines_test / "timelines_35day_season_2025_2026_v4_muscular.csv"
    output_test = v3_timelines_test / "timelines_35day_season_2025_2026_v4_muscular.csv"
    
    if not test_file.exists():
        print(f"WARNING: Test timeline file not found: {test_file}")
    else:
        orig_test, filt_test = filter_timeline_file(str(test_file), str(output_test), player_pl_periods)
        
        print(f"\n  Test Summary:")
        print(f"    Original: {orig_test:,} rows")
        print(f"    Filtered: {filt_test:,} rows")
        if orig_test > 0:
            print(f"    Retention: {filt_test/orig_test*100:.1f}%")
    
    # Final summary
    print("\n" + "=" * 80)
    print("FILTERING COMPLETE")
    print("=" * 80)
    print(f"Filtered timelines saved to: {v3_root / 'data' / 'timelines'}")
    print("\nNext steps:")
    print("1. Review filtered timeline counts")
    print("2. Train V3 model using filtered timelines")
    print("3. Evaluate V3 model performance")


if __name__ == "__main__":
    main()

