import pandas as pd
import numpy as np
import os
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import re
import glob

# Paths
V4_ROOT = Path(r"C:\Users\joao.henriques\IPM V3\models_production\lgbm_muscular_v4")
V4_DATA_DIR = V4_ROOT / "data"
V4_RAW_DATA = V4_DATA_DIR / "raw_data"
V4_TIMELINES_TRAIN = V4_DATA_DIR / "timelines" / "train"
V4_TIMELINES_TEST = V4_DATA_DIR / "timelines" / "test"

# Helper functions (copied from create_35day_timelines_v4_enhanced.py)
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
    """Build mapping of season_year -> set of PL club names from raw match data."""
    print("Building PL clubs per season mapping...")
    pl_clubs_by_season = defaultdict(set)
    
    match_files = glob.glob(os.path.join(raw_match_dir, "**", "*.csv"), recursive=True)
    
    for match_file in match_files:
        try:
            df = pd.read_csv(match_file, low_memory=False, encoding='utf-8-sig')
            
            if 'competition' not in df.columns or 'date' not in df.columns:
                continue
            
            pl_mask = df['competition'].str.contains('Premier League', case=False, na=False)
            pl_matches = df[pl_mask].copy()
            
            if pl_matches.empty:
                continue
            
            pl_matches['date'] = pd.to_datetime(pl_matches['date'], errors='coerce')
            pl_matches = pl_matches[pl_matches['date'].notna()]
            
            if pl_matches.empty:
                continue
            
            pl_matches['season_year'] = pl_matches['date'].apply(
                lambda d: d.year if d.month >= 7 else d.year - 1
            )
            
            for _, match in pl_matches.iterrows():
                season = match['season_year']
                
                for team_col in ['home_team', 'away_team', 'team']:
                    if team_col in match and pd.notna(match[team_col]):
                        club_name = str(match[team_col]).strip()
                        if club_name and not is_national_team(club_name):
                            pl_clubs_by_season[season].add(club_name)
                            pl_clubs_by_season[season].add(normalize_team_name(club_name))
        
        except Exception as e:
            continue
    
    return dict(pl_clubs_by_season)

def is_club_pl_club(club_name: str, season_year: int, pl_clubs_by_season: Dict[int, Set[str]]) -> bool:
    """Check if a club is a PL club for a given season."""
    if not club_name or pd.isna(club_name) or is_national_team(club_name):
        return False
    
    club_normalized = normalize_team_name(club_name)
    club_original = str(club_name).strip()
    
    for check_season in [season_year - 1, season_year, season_year + 1]:
        if check_season in pl_clubs_by_season:
            pl_clubs = pl_clubs_by_season[check_season]
            if club_original in pl_clubs or club_normalized in pl_clubs:
                for pl_club in pl_clubs:
                    if normalize_team_name(pl_club) == club_normalized:
                        return True
    
    return False

def build_player_pl_membership_periods(career_file: str, pl_clubs_by_season: Dict[int, Set[str]]) -> Dict[int, List[Tuple[pd.Timestamp, pd.Timestamp]]]:
    """Build mapping of player_id -> list of (start_date, end_date) periods when player was at PL club."""
    print("Building player PL membership periods...")
    
    try:
        career_df = pd.read_csv(career_file, sep=';', encoding='utf-8-sig', low_memory=False)
    except Exception as e:
        print(f"ERROR: Failed to load career file: {e}")
        return {}
    
    career_df['Date'] = pd.to_datetime(career_df['Date'], errors='coerce')
    if career_df['Date'].isna().sum() > len(career_df) * 0.5:
        career_df['Date'] = pd.to_datetime(career_df['Date'], format='%d/%m/%Y', errors='coerce')
    
    career_df = career_df[
        career_df['Date'].notna() & 
        career_df['To'].notna() &
        (career_df['To'].astype(str).str.strip() != '')
    ].copy()
    
    if career_df.empty:
        return {}
    
    player_pl_periods = defaultdict(list)
    
    for player_id, player_career in career_df.groupby('id'):
        player_career = player_career.sort_values('Date').reset_index(drop=True)
        
        current_pl_start = None
        
        for idx, row in player_career.iterrows():
            transfer_date = row['Date']
            to_club = str(row['To']).strip()
            
            if not to_club or pd.isna(transfer_date):
                continue
            
            season_year = transfer_date.year if transfer_date.month >= 7 else transfer_date.year - 1
            is_pl_destination = is_club_pl_club(to_club, season_year, pl_clubs_by_season)
            
            if is_pl_destination:
                if current_pl_start is None:
                    current_pl_start = transfer_date
            else:
                if current_pl_start is not None:
                    player_pl_periods[player_id].append((current_pl_start, transfer_date))
                    current_pl_start = None
        
        if current_pl_start is not None:
            last_date = player_career['Date'].max()
            end_season_year = last_date.year if last_date.month >= 7 else last_date.year - 1
            end_date = pd.Timestamp(f'{end_season_year + 1}-06-30')
            max_future = pd.Timestamp('2025-12-05')
            end_date = max(end_date, max_future)
            player_pl_periods[player_id].append((current_pl_start, end_date))
    
    return dict(player_pl_periods)

def is_player_in_pl_on_date(player_id: int, reference_date: pd.Timestamp, 
                            player_pl_periods: Dict[int, List[Tuple[pd.Timestamp, pd.Timestamp]]]) -> bool:
    """Check if player was at a PL club on the given reference date."""
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

def derive_injury_class(injury_type: str, no_physio_injury) -> str:
    """Derive injury_class from injury_type and no_physio_injury"""
    if pd.notna(no_physio_injury) and no_physio_injury == 1.0:
        return 'other'
    
    if pd.isna(injury_type) or injury_type == '':
        return 'unknown'
    
    injury_lower = str(injury_type).lower()
    
    skeletal_keywords = ['fracture', 'bone', 'ligament', 'tendon', 'cartilage', 'meniscus', 
                         'acl', 'cruciate', 'dislocation', 'subluxation', 'sprain', 'joint']
    if any(keyword in injury_lower for keyword in skeletal_keywords):
        return 'skeletal'
    
    muscular_keywords = ['strain', 'muscle', 'hamstring', 'quadriceps', 'calf', 'groin', 
                        'adductor', 'abductor', 'thigh', 'tear', 'rupture', 'pull']
    if any(keyword in injury_lower for keyword in muscular_keywords):
        return 'muscular'
    
    return 'unknown'

def get_season_from_date(date: pd.Timestamp) -> int:
    """Get season start year from a date."""
    return date.year if date.month >= 7 else date.year - 1

def main():
    print("="*80)
    print("INJURY VALIDATION CHECK")
    print("="*80)
    print("\nThis script will:")
    print("1. Pick 10 random players from profile data")
    print("2. Check when they played in PL clubs (from career data)")
    print("3. Identify all injuries they had while in PL (from injuries data)")
    print("4. Verify all injuries are captured in timeline datasets (x5 timelines per injury)")
    print("="*80)
    
    # Step 1: Load profile data and pick 10 random players
    print("\nStep 1: Loading player profiles and selecting 10 random players...")
    profile_file = V4_RAW_DATA / "players_profile.csv"
    
    if not profile_file.exists():
        print(f"ERROR: Profile file not found: {profile_file}")
        return
    
    profiles_df = pd.read_csv(profile_file, sep=';', encoding='utf-8-sig')
    all_player_ids = profiles_df['id'].dropna().unique().tolist()
    
    # Filter to players who are not goalkeepers (if position column exists)
    if 'position' in profiles_df.columns:
        non_gk_players = profiles_df[~profiles_df['position'].str.contains('Goalkeeper', case=False, na=False)]['id'].unique().tolist()
        all_player_ids = [pid for pid in all_player_ids if pid in non_gk_players]
    
    selected_player_ids = random.sample(all_player_ids, min(10, len(all_player_ids)))
    print(f"[OK] Selected {len(selected_player_ids)} random players: {selected_player_ids}")
    
    # Step 2: Build PL clubs and player PL membership periods
    print("\nStep 2: Building PL clubs per season and player PL membership periods...")
    match_data_dir = V4_RAW_DATA / "match_data"
    career_file = V4_RAW_DATA / "players_career.csv"
    
    pl_clubs_by_season = build_pl_clubs_per_season(str(match_data_dir))
    player_pl_periods = build_player_pl_membership_periods(str(career_file), pl_clubs_by_season)
    
    print(f"[OK] Found PL periods for {len(player_pl_periods)} players")
    
    # Step 3: Load injuries data and identify PL injuries for selected players
    print("\nStep 3: Loading injuries data and identifying PL injuries...")
    injuries_file = V4_RAW_DATA / "injuries_data.csv"
    
    if not injuries_file.exists():
        print(f"ERROR: Injuries file not found: {injuries_file}")
        return
    
    injuries_df = pd.read_csv(injuries_file, sep=';', encoding='utf-8-sig')
    
    # Parse dates - try DD-MM-YYYY and DD/MM/YYYY formats explicitly to avoid MM-DD-YYYY misinterpretation
    if 'fromDate' in injuries_df.columns:
        # Store original column for retry
        original_dates = injuries_df['fromDate'].copy()
        
        # First, try DD-MM-YYYY format (European format with dashes)
        injuries_df['fromDate'] = pd.to_datetime(injuries_df['fromDate'], format='%d-%m-%Y', errors='coerce')
        
        # If that fails for many rows, try DD/MM/YYYY format (European format with slashes)
        if injuries_df['fromDate'].isna().sum() > len(injuries_df) * 0.5:
            injuries_df['fromDate'] = pd.to_datetime(original_dates, format='%d/%m/%Y', errors='coerce')
        
        # If that also fails, try auto-detect with dayfirst=True to prefer DD/MM/YYYY over MM/DD/YYYY
        if injuries_df['fromDate'].isna().sum() > len(injuries_df) * 0.5:
            injuries_df['fromDate'] = pd.to_datetime(original_dates, dayfirst=True, errors='coerce')
    
    # Derive injury class if needed
    if 'injury_class' not in injuries_df.columns:
        injuries_df['injury_class'] = injuries_df.apply(
            lambda row: derive_injury_class(
                row.get('injury_type', ''),
                row.get('no_physio_injury', None)
            ),
            axis=1
        )
    
    # Filter to selected players and valid injuries
    selected_injuries = injuries_df[
        (injuries_df['player_id'].isin(selected_player_ids)) &
        (injuries_df['fromDate'].notna())
    ].copy()
    
    print(f"[OK] Found {len(selected_injuries)} total injuries for selected players")
    
    # Identify PL injuries (injuries that occurred while player was in PL)
    pl_injuries = []
    player_names = dict(zip(profiles_df['id'], profiles_df['name']))
    
    for _, injury_row in selected_injuries.iterrows():
        player_id = int(injury_row['player_id'])
        injury_date = pd.Timestamp(injury_row['fromDate']).normalize()
        
        # Check if player was in PL on injury date
        if is_player_in_pl_on_date(player_id, injury_date, player_pl_periods):
            injury_class = str(injury_row.get('injury_class', 'unknown')).lower()
            injury_type = injury_row.get('injury_type', '')
            season = get_season_from_date(injury_date)
            
            pl_injuries.append({
                'player_id': player_id,
                'player_name': player_names.get(player_id, f'Player_{player_id}'),
                'injury_date': injury_date,
                'injury_class': injury_class,
                'injury_type': injury_type,
                'season': season
            })
    
    print(f"[OK] Found {len(pl_injuries)} injuries that occurred while players were in PL")
    
    if len(pl_injuries) == 0:
        print("\n[WARNING] No PL injuries found for selected players. Try selecting more players or different players.")
        return
    
    # Step 4: Verify injuries are in timeline datasets
    print("\nStep 4: Verifying injuries are captured in timeline datasets...")
    print("="*80)
    
    validation_results = []
    
    for injury in pl_injuries:
        player_id = injury['player_id']
        injury_date = injury['injury_date']
        injury_class = injury['injury_class']
        season = injury['season']
        
        # Determine if train or test
        is_test = season >= 2025
        timeline_dir = V4_TIMELINES_TEST if is_test else V4_TIMELINES_TRAIN
        suffix = 'test' if is_test else 'train'
        
        timeline_file = timeline_dir / f"timelines_35day_season_{season}_{season+1}_v4_muscular_{suffix}.csv"
        
        if not timeline_file.exists():
            validation_results.append({
                'player_id': player_id,
                'player_name': injury['player_name'],
                'injury_date': injury_date,
                'injury_class': injury_class,
                'injury_type': injury['injury_type'],
                'season': season,
                'status': 'FILE_NOT_FOUND',
                'expected_timelines': 5 if injury_class in ['muscular', 'skeletal'] else 0,
                'found_timelines': 0,
                'reference_dates_found': []
            })
            continue
        
        # Load timeline file and check for this injury
        try:
            # Read only relevant columns for efficiency
            timeline_df = pd.read_csv(timeline_file, usecols=['player_id', 'reference_date', 'target1', 'target2'], low_memory=False)
            timeline_df['reference_date'] = pd.to_datetime(timeline_df['reference_date'])
            
            # Filter to this player
            player_timelines = timeline_df[timeline_df['player_id'] == player_id].copy()
            
            # Expected reference dates: D-1, D-2, D-3, D-4, D-5 (5 days before injury)
            expected_ref_dates = [injury_date - timedelta(days=d) for d in range(1, 6)]
            expected_ref_dates_normalized = [d.normalize() for d in expected_ref_dates]
            
            # Check which reference dates are found
            found_ref_dates = []
            for ref_date in expected_ref_dates_normalized:
                matching = player_timelines[player_timelines['reference_date'].dt.normalize() == ref_date]
                if len(matching) > 0:
                    # Check if target is correct
                    if injury_class == 'muscular':
                        if (matching['target1'] == 1).any():
                            found_ref_dates.append(ref_date)
                    elif injury_class == 'skeletal':
                        if (matching['target2'] == 1).any():
                            found_ref_dates.append(ref_date)
                    else:
                        # For other/unknown, check if it's NOT in timelines (should be excluded)
                        found_ref_dates.append(ref_date)  # Will mark as excluded
            
            # Determine status
            if injury_class in ['muscular', 'skeletal']:
                if len(found_ref_dates) == 5:
                    status = '[OK] ALL_FOUND'
                elif len(found_ref_dates) > 0:
                    status = f'[PARTIAL] PARTIAL ({len(found_ref_dates)}/5)'
                else:
                    status = '[ERROR] NOT_FOUND'
            else:
                # Other/unknown injuries should NOT be in timelines
                if len(found_ref_dates) == 0:
                    status = '[OK] CORRECTLY_EXCLUDED'
                else:
                    status = '[WARNING] INCORRECTLY_INCLUDED'
            
            validation_results.append({
                'player_id': player_id,
                'player_name': injury['player_name'],
                'injury_date': injury_date,
                'injury_class': injury_class,
                'injury_type': injury['injury_type'],
                'season': season,
                'status': status,
                'expected_timelines': 5 if injury_class in ['muscular', 'skeletal'] else 0,
                'found_timelines': len(found_ref_dates),
                'reference_dates_found': found_ref_dates,
                'reference_dates_expected': expected_ref_dates_normalized
            })
            
        except Exception as e:
            validation_results.append({
                'player_id': player_id,
                'player_name': injury['player_name'],
                'injury_date': injury_date,
                'injury_class': injury_class,
                'injury_type': injury['injury_type'],
                'season': season,
                'status': f'ERROR: {str(e)}',
                'expected_timelines': 5,
                'found_timelines': 0,
                'reference_dates_found': []
            })
    
    # Print results
    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)
    
    for result in validation_results:
        print(f"\nPlayer: {result['player_name']} (ID: {result['player_id']})")
        print(f"  Injury Date: {result['injury_date'].strftime('%Y-%m-%d')}")
        print(f"  Injury Class: {result['injury_class']}")
        print(f"  Injury Type: {result['injury_type']}")
        print(f"  Season: {result['season']}_{result['season']+1}")
        print(f"  Status: {result['status']}")
        print(f"  Expected Timelines: {result['expected_timelines']}")
        print(f"  Found Timelines: {result['found_timelines']}")
        
        if result['expected_timelines'] > 0:
            missing = set(result.get('reference_dates_expected', [])) - set(result['reference_dates_found'])
            if missing:
                print(f"  Missing Reference Dates: {sorted(missing)}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    total_injuries = len(validation_results)
    all_found = sum(1 for r in validation_results if r['status'] == '[OK] ALL_FOUND')
    partial = sum(1 for r in validation_results if 'PARTIAL' in r['status'])
    not_found = sum(1 for r in validation_results if r['status'] == '[ERROR] NOT_FOUND')
    correctly_excluded = sum(1 for r in validation_results if r['status'] == '[OK] CORRECTLY_EXCLUDED')
    errors = sum(1 for r in validation_results if 'ERROR' in r['status'] or 'FILE_NOT_FOUND' in r['status'])
    
    print(f"Total PL injuries checked: {total_injuries}")
    print(f"[OK] All timelines found: {all_found}")
    print(f"[PARTIAL] Partial timelines found: {partial}")
    print(f"[ERROR] Not found: {not_found}")
    print(f"[OK] Correctly excluded (other/unknown): {correctly_excluded}")
    print(f"[ERROR] Errors/File not found: {errors}")
    
    # Calculate validation success (excluding file errors)
    validation_issues = not_found + sum(1 for r in validation_results if 'INCORRECTLY_INCLUDED' in r['status'])
    validation_success = all_found + correctly_excluded
    
    if validation_issues == 0 and errors == 0:
        print("\n[SUCCESS] VALIDATION PASSED: All injuries are correctly captured!")
    elif validation_issues > 0:
        print(f"\n[WARNING] VALIDATION ISSUES FOUND: {validation_issues} injuries are missing or incorrectly handled.")
        print("  This may indicate:")
        print("  - Injury date mismatch between injuries_data.csv and daily features")
        print("  - Injury not recorded as new start in daily features (cum_inj_starts didn't increment)")
        print("  - Player not in PL on reference dates (D-1 through D-5)")
        print("  - Incomplete 35-day window for reference dates")
    else:
        print(f"\n[ERROR] File/processing errors: {errors} files not found or processing errors.")

if __name__ == "__main__":
    main()
