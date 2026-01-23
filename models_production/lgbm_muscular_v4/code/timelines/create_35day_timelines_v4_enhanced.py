#!/usr/bin/env python3
"""
V4 Enhanced 35-Day Timeline Generator - Consolidated Version
Generates timelines from V4 daily features with PL-only filtering.

Key features:
- Generates timelines season by season (YYYY_YYYY+1 format)
- Dual targets: target1 (muscular injuries) and target2 (skeletal injuries)
- Natural target ratios (all available positives and negatives)
- Each injury generates 5 timelines (D-1, D-2, D-3, D-4, D-5)
- Non-injury validation checks for ANY injury (any class) in 35 days after reference
- PL-only filtering using career data
- Season segmentation: train (≤2024/25) and test (2025/26)
- Activity flag: has_minimum_activity (≥90 minutes in 35-day window)

IMPORTANT: Model Training Filtering
-------------------------------------
When training models, use filter_timelines_for_model() to exclude other injury types:
- Model 1 (muscular): Use only target1=1 (positives) and target1=0, target2=0 (negatives)
- Model 2 (skeletal): Use only target2=1 (positives) and target1=0, target2=0 (negatives)

This ensures each model learns to distinguish its specific injury type from non-injuries,
not from other injury types.
"""

import sys
import io
# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import os
import random
import csv
import re
import glob
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
from pathlib import Path
import warnings
from tqdm import tqdm
try:
    import openpyxl  # For reading Excel files
except ImportError:
    openpyxl = None
warnings.filterwarnings('ignore')

# Configuration
WINDOW_SIZE = 35  # 5 weeks
USE_NATURAL_RATIO = True  # Flag to use all available positives and negatives

# Season configuration
SEASON_START_MONTH = 7  # July
SEASON_START_DAY = 1

# Allowed injury classes for injury timelines (dual targets)
ALLOWED_INJURY_CLASSES_MUSCULAR = {'muscular'}  # For target1
ALLOWED_INJURY_CLASSES_SKELETAL = {'skeletal'}  # For target2

# Activity requirement configuration
MIN_ACTIVITY_MINUTES = 90  # Minimum minutes in 35-day window for activity flag

# Train/test split
TRAIN_SEASON_CUTOFF = 2024  # Train on seasons ≤ 2024/25
TEST_SEASON_START = 2025    # Test on season 2025/26

# Script directory and paths
SCRIPT_DIR = Path(__file__).resolve().parent
V4_ROOT = SCRIPT_DIR.parent.parent
V4_DATA_DIR = V4_ROOT / "data"
V4_RAW_DATA = V4_DATA_DIR / "raw_data"
V4_DAILY_FEATURES = V4_DATA_DIR / "daily_features"
# Layer 2 enriched daily features directory
V4_DAILY_FEATURES_ENRICHED = V4_DATA_DIR / "daily_features_enriched"
V4_TIMELINES = V4_DATA_DIR / "timelines"
V4_TIMELINES_TRAIN = V4_TIMELINES / "train"
V4_TIMELINES_TEST = V4_TIMELINES / "test"

# Helper functions for team name normalization (from V3/V4)
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

def get_season_date_range(season_start_year: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Get start and end dates for a season"""
    start = pd.Timestamp(f'{season_start_year}-07-01')
    end = pd.Timestamp(f'{season_start_year + 1}-06-30')
    return start, end

def get_all_seasons_from_daily_features(daily_features_dir: str) -> List[int]:
    """Scan daily features to determine available seasons"""
    print(f"\n📅 Scanning daily features to determine available seasons...")
    all_dates = set()
    
    daily_files = [f for f in os.listdir(daily_features_dir) if f.endswith('_daily_features.csv')]
    print(f"   Found {len(daily_files)} daily feature files")
    
    for i, filename in enumerate(tqdm(daily_files, desc="   Scanning files", unit="file")):
        try:
            filepath = os.path.join(daily_features_dir, filename)
            # Read just the date column to get date range
            df = pd.read_csv(filepath, usecols=['date'], low_memory=False)
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            valid_dates = df['date'].dropna()
            if len(valid_dates) > 0:
                all_dates.update(valid_dates.dt.date)
        except Exception as e:
            continue
    
    if not all_dates:
        raise ValueError("No valid dates found in daily features files!")
    
    min_date = min(all_dates)
    max_date = max(all_dates)
    min_year = min_date.year
    max_year = max_date.year
    
    # Determine seasons: from earliest July 1st to latest June 30th
    seasons = []
    # Start from the year of the earliest date, but only if it's July or later
    start_year = min_year if min_date >= pd.Timestamp(f'{min_year}-07-01').date() else min_year - 1
    
    # End at the year of the latest date
    end_year = max_year
    
    for year in range(start_year, end_year + 1):
        season_start = pd.Timestamp(f'{year}-07-01').date()
        season_end = pd.Timestamp(f'{year + 1}-06-30').date()
        # Only include season if we have data in that range
        if any(season_start <= d <= season_end for d in all_dates):
            seasons.append(year)
    
    print(f"   Date range: {min_date} to {max_date}")
    print(f"   Available seasons: {seasons[0]}_{seasons[0]+1} to {seasons[-1]}_{seasons[-1]+1} ({len(seasons)} seasons)")
    return seasons

def get_static_features() -> List[str]:
    """Get features that remain static (not windowed) - adapted for V4"""
    return [
        'player_id', 'reference_date', 'player_name',  # Technical features
        'position', 'nationality1', 'nationality2', 'height_cm', 'dominant_foot',
        'previous_club', 'previous_club_country', 'current_club_country',  # Removed 'current_club'
        'age', 'seniority_days',
        # Career cumulative features
        'cum_minutes_played_numeric', 'cum_goals_numeric', 'cum_assists_numeric',
        'cum_yellow_cards_numeric', 'cum_red_cards_numeric', 'cum_matches_played',
        'cum_matches_bench', 'cum_matches_not_selected', 'cum_competitions',
        'matches', 'career_matches', 'career_goals', 'career_assists', 'career_minutes',
        # Career injury features
        'cum_inj_starts', 'cum_inj_days', 'avg_injury_severity', 'max_injury_severity',
        'lower_leg_injuries', 'knee_injuries', 'upper_leg_injuries', 'hip_injuries',
        'upper_body_injuries', 'head_injuries', 'illness_count', 'other_injuries', 'cum_matches_injured',
        # Career injury features by class
        'muscular_injury_count', 'skeletal_injury_count', 'unknown_injury_count',
        'muscular_injury_days', 'skeletal_injury_days', 'unknown_injury_days', 'other_injury_days',
        # Career injury features by severity
        'mild_injury_count', 'moderate_injury_count', 'severe_injury_count', 'critical_injury_count',
        # Career injury features by body part (additional)
        'unknown_body_part_count',
        # Combined injury features (class + body part)
        'muscular_lower_leg_count', 'muscular_knee_count', 'skeletal_lower_leg_count', 'skeletal_knee_count',
        # Combined injury features (class + severity)
        'muscular_mild_count', 'muscular_moderate_count', 'muscular_severe_count', 'muscular_critical_count',
        'skeletal_mild_count', 'skeletal_moderate_count', 'skeletal_severe_count', 'skeletal_critical_count',
        # Combined injury features (body part + severity)
        'mild_lower_leg_count', 'moderate_lower_leg_count', 'severe_lower_leg_count', 'critical_lower_leg_count',
        'mild_knee_count', 'moderate_knee_count', 'severe_knee_count', 'critical_knee_count',
        # Injury class ratios
        'muscular_to_total_ratio', 'skeletal_to_total_ratio', 'unknown_to_total_ratio', 'other_to_total_ratio',
        # Career competition features
        'avg_competition_importance', 'cum_disciplinary_actions', 'teams_last_season',
        'national_team_appearances', 'national_team_minutes', 'national_team_last_season',
        'national_team_frequency', 'senior_national_team', 'competition_level',
        'international_competitions', 'cup_competitions',
        'competition_experience', 'competition_pressure', 'teams_today', 'cum_teams', 'seasons_count',
        # Career club features
        'club_cum_goals', 'club_cum_assists', 'club_cum_minutes',
        'club_cum_matches_played', 'club_cum_yellow_cards', 'club_cum_red_cards',
        # Interaction features
        'age_x_career_matches', 'age_x_career_goals', 'seniority_x_goals_per_match',
        'club_seniority_x_goals_per_match',
        # V4 Enhanced features - injury category
        'injury_free_period_category', 'injuries_last_2_years', 'muscular_injuries_last_2_years',
        'skeletal_injuries_last_2_years', 'last_injury_severity',
        # V4 Enhanced features - composite indicators
        'fatigue_indicator', 'workload_x_injury_history', 'age_workload_risk',
        # V4 Enhanced features - temporal
        # REMOVED: 'season_year',  # Removed - temporal leak, should not be used in model
        'season_month', 'is_pre_season', 'is_mid_season', 'is_end_season', 'days_into_season', 'is_early_season',  # NEW: is_early_season (first 30 days)
        # Layer 2 enriched daily features (per-date historical summaries)
        # Workload windows (minutes)
        'minutes_last_3d',
        'minutes_last_7d',
        'minutes_last_14d',
        'minutes_last_28d',
        'minutes_last_35d',
        # Workload windows (matches)
        'matches_last_7d',
        'matches_last_14d',
        'matches_last_28d',
        'matches_last_35d',
        # Intensity ratios
        'acwr_min_7_28',
        # Season-normalized workload
        'minutes_season_to_date',
        'minutes_last_7d_pct_season',
        'minutes_last_28d_pct_season',
        # Injury history (recent counts)
        'injuries_last_90d',
        'injuries_last_365d',
        'injuries_season_to_date',
        # Recovery / rest flags
        'is_back_to_back',
        'short_rest_3_4d',
        'long_rest_7d_plus',
        # Simple activity flags
        'has_played_last_7d',
        'has_played_last_28d',
        'no_recent_activity_28d',
        # Layer 2 interaction features (NEW)
        'inactivity_risk',
        'early_season_low_activity',
        'preseason_long_rest',
        'low_activity_with_history',
    ]

def get_windowed_features() -> List[str]:
    """Get features that will be windowed (35 days) - adapted for V4"""
    return [
        # Daily performance metrics
        'matches_played', 'minutes_played_numeric', 'goals_numeric', 'assists_numeric',
        'yellow_cards_numeric', 'red_cards_numeric', 'matches_bench_unused', 
        'matches_not_selected', 'matches_injured',
        # Recent patterns
        'days_since_last_match', 'last_match_position', 'position_match_default',
        'disciplinary_action', 'goals_per_match', 'assists_per_match', 'minutes_per_match',
        # Recent injury indicators (V4 has log-transformed core features)
        'days_since_last_injury', 'days_since_last_injury_ended', 'days_since_last_muscular',
        'days_since_last_skeletal', 'avg_injury_duration', 'injury_frequency',
        # Recent competition context
        'competition_importance', 'month', 'teams_this_season',
        'teams_season_today',
        # Recent national team activity
        'days_since_last_national_match', 'national_team_this_season',
        # Recent club performance
        'club_goals_per_match', 'club_assists_per_match', 'club_minutes_per_match',
        'club_seniority_x_goals_per_match',
        # Match location
        'home_matches', 'away_matches',
        # Team result features
        'team_win', 'team_draw', 'team_loss', 'team_points',
        'cum_team_wins', 'cum_team_draws', 'cum_team_losses',
        'team_win_rate', 'cum_team_points',
        # Substitution patterns
        'substitution_on_count', 'substitution_off_count', 'late_substitution_on_count',
        'early_substitution_off_count', 'impact_substitution_count', 'tactical_substitution_count',
        'substitution_minutes_played', 'consecutive_substitutions',
        # V4 Enhanced features - workload acceleration
        'workload_acceleration_7d', 'workload_acceleration_14d', 'workload_spike_7d', 'workload_spike_14d',
        # V4 Enhanced features - recent match frequency
        'matches_in_last_7_days', 'matches_in_last_14_days', 'matches_in_last_21_days', 'matches_in_last_30_days',
        # V4 Enhanced features - recovery time
        'recovery_time_avg_last_3', 'recovery_time_min_last_3', 'short_recovery_matches_14d',
        # V4 Enhanced features - recent patterns
        'early_substitutions_last_7d', 'early_substitutions_last_14d', 'consecutive_early_substitutions',
        'matches_bench_last_7d', 'matches_bench_last_14d', 'bench_ratio_last_14d',
        'high_intensity_matches_14d', 'consecutive_high_intensity'
    ]

# ===== PL FILTERING FUNCTIONS (from V3) =====

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
    
    # Parse dates - try ISO format first, then DD/MM/YYYY
    career_df['Date'] = pd.to_datetime(career_df['Date'], errors='coerce')
    if career_df['Date'].isna().sum() > len(career_df) * 0.5:
        # Try DD/MM/YYYY format
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

def filter_timelines_pl_only(timelines_df: pd.DataFrame,
                              player_pl_periods: Dict[int, List[Tuple[pd.Timestamp, pd.Timestamp]]]) -> pd.DataFrame:
    """
    Filter timelines DataFrame to only include rows where player was in PL on reference_date.
    This is called after timeline generation (post-processing filter).
    
    Args:
        timelines_df: DataFrame with timelines (must have 'player_id' and 'reference_date' columns)
        player_pl_periods: Mapping of player_id -> list of PL membership periods
        
    Returns:
        Filtered DataFrame with only PL timelines
    """
    if 'player_id' not in timelines_df.columns or 'reference_date' not in timelines_df.columns:
        return timelines_df
    
    # Convert reference_date to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(timelines_df['reference_date']):
        timelines_df['reference_date'] = pd.to_datetime(timelines_df['reference_date'], errors='coerce')
    
    # Filter rows where player was in PL on reference_date
    mask = timelines_df.apply(
        lambda row: is_player_in_pl_on_date(
            row['player_id'],
            row['reference_date'],
            player_pl_periods
        ) if pd.notna(row['reference_date']) else False,
        axis=1
    )
    
    filtered_df = timelines_df[mask].copy()
    return filtered_df

# ===== INJURY DATA LOADING FUNCTIONS (from V1) =====

def derive_injury_class(injury_type: str, no_physio_injury: Optional[float]) -> str:
    """
    Derive injury_class from injury_type and no_physio_injury
    
    Args:
        injury_type: Injury type description
        no_physio_injury: 1.0 if non-physio injury, NaN otherwise
    
    Returns:
        injury_class: 'muscular', 'skeletal', 'unknown', or 'other'
    """
    if pd.notna(no_physio_injury) and no_physio_injury == 1.0:
        # Non-physio injuries are typically skeletal or other
        return 'other'
    
    if pd.isna(injury_type) or injury_type == '':
        return 'unknown'
    
    injury_lower = str(injury_type).lower()
    
    # Skeletal injuries (bones, joints, ligaments)
    skeletal_keywords = ['fracture', 'bone', 'ligament', 'tendon', 'cartilage', 'meniscus', 
                         'acl', 'cruciate', 'dislocation', 'subluxation', 'sprain', 'joint']
    if any(keyword in injury_lower for keyword in skeletal_keywords):
        return 'skeletal'
    
    # Muscular injuries (muscles, strains)
    muscular_keywords = ['strain', 'muscle', 'hamstring', 'quadriceps', 'calf', 'groin', 
                        'adductor', 'abductor', 'thigh', 'tear', 'rupture', 'pull']
    if any(keyword in injury_lower for keyword in muscular_keywords):
        return 'muscular'
    
    # If we can't determine, return 'unknown'
    return 'unknown'

def load_injuries_data(injuries_file: str) -> Dict[Tuple[int, pd.Timestamp], str]:
    """
    Load injuries data file and create a mapping of (player_id, injury_date) -> injury_class
    
    Args:
        injuries_file: Path to injuries_data.csv file
    
    Returns:
        Dictionary mapping (player_id, fromDate) to injury_class
    """
    print("📂 Loading injuries data file...")
    print(f"   Loading from: {injuries_file}")
    
    if not os.path.exists(injuries_file):
        raise FileNotFoundError(f"Injuries data file not found: {injuries_file}")
    
    # Load injuries data
    injuries_df = pd.read_csv(injuries_file, sep=';', encoding='utf-8-sig')
    
    # Convert dates - try DD-MM-YYYY and DD/MM/YYYY formats explicitly to avoid MM-DD-YYYY misinterpretation
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
    
    # Derive injury_class if it doesn't exist
    if 'injury_class' not in injuries_df.columns:
        print("   ⚠️  injury_class column not found, deriving from injury_type and no_physio_injury")
        injuries_df['injury_class'] = injuries_df.apply(
            lambda row: derive_injury_class(
                row.get('injury_type', ''),
                row.get('no_physio_injury', None)
            ),
            axis=1
        )
    
    # Create mapping: (player_id, fromDate) -> injury_class
    injury_class_map = {}
    for _, row in injuries_df.iterrows():
        player_id = row.get('player_id')
        from_date = row.get('fromDate')
        injury_class = row.get('injury_class', '').lower() if pd.notna(row.get('injury_class')) else ''
        
        if pd.notna(player_id) and pd.notna(from_date):
            injury_class_map[(int(player_id), pd.Timestamp(from_date).normalize())] = injury_class
    
    print(f"   Loaded {len(injury_class_map)} injury records with injury_class")
    print(f"   Injury class distribution:")
    class_counts = injuries_df['injury_class'].value_counts()
    for class_name, count in class_counts.items():
        print(f"      {class_name}: {count}")
    
    return injury_class_map

def load_all_injury_dates(injuries_file: str) -> Dict[int, Set[pd.Timestamp]]:
    """
    Load all injury dates (any class) for each player
    Used for non-injury validation (checking if ANY injury occurs in 35 days after)
    
    Args:
        injuries_file: Path to injuries_data.csv file
    
    Returns:
        Dictionary mapping player_id to set of injury dates
    """
    print("📂 Loading all injury dates (any class) for non-injury validation...")
    
    if not os.path.exists(injuries_file):
        raise FileNotFoundError(f"Injuries data file not found: {injuries_file}")
    
    # Load injuries data
    injuries_df = pd.read_csv(injuries_file, sep=';', encoding='utf-8-sig')
    
    # Convert dates - try DD-MM-YYYY and DD/MM/YYYY formats explicitly to avoid MM-DD-YYYY misinterpretation
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
    
    # Create mapping: player_id -> set of all injury dates (any class)
    all_injury_dates = defaultdict(set)
    for _, row in injuries_df.iterrows():
        player_id = row.get('player_id')
        from_date = row.get('fromDate')
        
        if pd.notna(player_id) and pd.notna(from_date):
            all_injury_dates[int(player_id)].add(pd.Timestamp(from_date).normalize())
    
    print(f"   Loaded injury dates for {len(all_injury_dates)} players")
    total_injuries = sum(len(dates) for dates in all_injury_dates.values())
    print(f"   Total injury dates: {total_injuries}")
    
    return dict(all_injury_dates)

# ===== WINDOWED FEATURES CREATION (from V1, adapted for V4) =====

def create_windowed_features_vectorized(df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> Optional[Dict]:
    """Create windowed features using vectorized operations"""
    # Filter data for the 35-day window
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    window_data = df[mask].copy()
    
    if len(window_data) != 35:
        return None  # Incomplete window
    
    # Pre-calculate week boundaries
    week_starts = [start_date + timedelta(days=i * 7) for i in range(5)]
    week_ends = [start_date + timedelta(days=i * 7 + 6) for i in range(5)]
    
    weekly_features = {}
    
    # Vectorized weekly aggregation
    for week in range(5):
        week_mask = (window_data['date'] >= week_starts[week]) & (window_data['date'] <= week_ends[week])
        week_data = window_data[week_mask]
        
        if len(week_data) != 7:
            return None  # Incomplete week
        
        # Define aggregation strategies for enhanced features
        last_value_features = ['last_match_position', 'position_match_default', 'disciplinary_action',
                              'competition_importance', 'month', 'teams_this_season',
                              'teams_season_today', 'national_team_this_season']
        
        # Season-specific features that should be excluded from week 5
        season_specific_features = ['teams_this_season', 'national_team_this_season']
        
        sum_features = ['matches_played', 'minutes_played_numeric', 'goals_numeric', 'assists_numeric',
                       'yellow_cards_numeric', 'red_cards_numeric', 'matches_bench_unused',
                       'matches_not_selected', 'matches_injured', 'substitution_on_count',
                       'substitution_off_count', 'late_substitution_on_count', 'early_substitution_off_count',
                       'impact_substitution_count', 'tactical_substitution_count', 'substitution_minutes_played',
                       'consecutive_substitutions', 'matches_in_last_7_days', 'matches_in_last_14_days',
                       'matches_in_last_21_days', 'matches_in_last_30_days', 'short_recovery_matches_14d',
                       'early_substitutions_last_7d', 'early_substitutions_last_14d', 'consecutive_early_substitutions',
                       'matches_bench_last_7d', 'matches_bench_last_14d', 'high_intensity_matches_14d',
                       'consecutive_high_intensity', 'home_matches', 'away_matches', 'team_win', 'team_draw',
                       'team_loss', 'team_points', 'cum_team_wins', 'cum_team_draws', 'cum_team_losses']
        
        mean_features = ['goals_per_match', 'assists_per_match', 'minutes_per_match',
                        'avg_injury_duration', 'injury_frequency', 'recovery_time_avg_last_3',
                        'recovery_time_min_last_3', 'bench_ratio_last_14d', 'workload_acceleration_7d',
                        'workload_acceleration_14d', 'workload_spike_7d', 'workload_spike_14d',
                        'club_goals_per_match', 'club_assists_per_match', 'club_minutes_per_match',
                        'club_seniority_x_goals_per_match', 'team_win_rate', 'cum_team_points']
        
        min_features = ['days_since_last_match', 'days_since_last_injury', 'days_since_last_injury_ended',
                       'days_since_last_muscular', 'days_since_last_skeletal', 'days_since_last_national_match']
        
        # Apply aggregations
        for feature in get_windowed_features():
            # Skip season-specific features in week 5
            if feature in season_specific_features and week == 4:
                continue
            
            if feature in week_data.columns:
                if feature in last_value_features:
                    weekly_features[f'{feature}_week_{week+1}'] = week_data[feature].iloc[-1]
                elif feature in sum_features:
                    weekly_features[f'{feature}_week_{week+1}'] = week_data[feature].sum()
                elif feature in mean_features:
                    weekly_features[f'{feature}_week_{week+1}'] = week_data[feature].mean()
                elif feature in min_features:
                    weekly_features[f'{feature}_week_{week+1}'] = week_data[feature].min()
                else:
                    weekly_features[f'{feature}_week_{week+1}'] = week_data[feature].mean()
    
    return weekly_features

# ===== TIMELINE GENERATION FUNCTIONS (from V1, adapted for V4) =====

def generate_injury_timelines_enhanced(player_id: int, player_name: str, df: pd.DataFrame,
                                       injury_class_map: Dict[Tuple[int, pd.Timestamp], str]) -> List[Dict]:
    """
    Generate injury timelines for BOTH muscular and skeletal injuries
    Each injury gets 5 timelines (D-1, D-2, D-3, D-4, D-5)
    (PL filtering is done in post-processing, not during generation)
    Returns timelines with target1 (muscular) and target2 (skeletal)
    """
    timelines = []
    
    # Vectorized injury start detection
    injury_starts = df[df['cum_inj_starts'] > df['cum_inj_starts'].shift(1)]
    
    for _, injury_row in injury_starts.iterrows():
        injury_date = injury_row['date']
        injury_date_normalized = pd.Timestamp(injury_date).normalize()
        
        # Get injury class
        injury_class = injury_class_map.get((player_id, injury_date_normalized), '').lower()
        
        # Skip if not muscular or skeletal
        if injury_class not in ALLOWED_INJURY_CLASSES_MUSCULAR and injury_class not in ALLOWED_INJURY_CLASSES_SKELETAL:
            continue
        
        # Determine targets
        target1 = 1 if injury_class in ALLOWED_INJURY_CLASSES_MUSCULAR else 0
        target2 = 1 if injury_class in ALLOWED_INJURY_CLASSES_SKELETAL else 0
        
        # Generate 5 timelines (D-1, D-2, D-3, D-4, D-5)
        for days_before in range(1, 6):
            reference_date = injury_date - timedelta(days=days_before)
            start_date = reference_date - timedelta(days=34)  # 35 days before reference
            
            if start_date < df['date'].min():
                continue
            
            # Create windowed features
            windowed_features = create_windowed_features_vectorized(df, start_date, reference_date)
            if windowed_features is None:
                continue
            
            # Get static features from reference date
            ref_mask = df['date'] == reference_date
            if not ref_mask.any():
                continue
                
            ref_row = df[ref_mask].iloc[0]
            
            # Build timeline with dual targets
            timeline = build_timeline(player_id, player_name, reference_date, ref_row, windowed_features, 
                                     target1=target1, target2=target2, player_df=df, window_start_date=start_date)
            timelines.append(timeline)
    
    return timelines

def get_valid_non_injury_dates(df: pd.DataFrame,
                                season_start: pd.Timestamp,
                                season_end: pd.Timestamp,
                                all_injury_dates: Optional[Set[pd.Timestamp]] = None) -> List[pd.Timestamp]:
    """
    Get all valid non-injury reference dates for a specific season
    (PL filtering is done in post-processing, not during generation)
    
    Eligibility rules:
    - Reference date must be within season date range
    - No injury of ANY class in the 35 days after reference date
    - Complete 35-day window must be available
    """
    valid_dates = []
    
    # Get all injury dates for this player (any class) - used for validation
    player_injury_dates = all_injury_dates if all_injury_dates is not None else set()
    
    max_date = df['date'].max()
    min_date = df['date'].min()
    max_reference_date = max_date - timedelta(days=34)
    
    # Create date range for potential reference dates
    potential_dates = pd.date_range(min_date, max_reference_date, freq='D')
    
    # Process valid dates
    for reference_date in potential_dates:
        # Check if reference date exists in data
        if reference_date not in df['date'].values:
            continue
        
        # Season filtering
        if not (season_start <= reference_date <= season_end):
            continue
        
        # CRITICAL: Check if there's an injury of ANY class in the next 35 days
        future_end = reference_date + timedelta(days=34)
        if future_end > max_date:
            continue
        
        # Check if any injury (any class) occurs in this 35-day window
        future_dates = pd.date_range(reference_date, future_end, freq='D')
        injury_in_window = any(pd.Timestamp(date).normalize() in player_injury_dates for date in future_dates)
        if injury_in_window:
            continue  # Skip this date - injury (any class) in next 35 days
        
        # Check if we can create a complete 35-day window
        start_date = reference_date - timedelta(days=34)
        if start_date < min_date:
            continue
        
        # All checks passed
        valid_dates.append(reference_date)
    
    return valid_dates

def generate_non_injury_timelines_for_dates(player_id: int, player_name: str, df: pd.DataFrame, 
                                            reference_dates: List[pd.Timestamp]) -> List[Dict]:
    """Generate non-injury timelines for specific reference dates (both targets = 0)"""
    timelines = []
    
    for reference_date in reference_dates:
        # Create windowed features
        start_date = reference_date - timedelta(days=34)
        windowed_features = create_windowed_features_vectorized(df, start_date, reference_date)
        if windowed_features is None:
            continue
        
        # Get static features from reference date
        ref_mask = df['date'] == reference_date
        if not ref_mask.any():
            continue
        ref_row = df[ref_mask].iloc[0]
        
        # Build timeline with both targets = 0 (no injury)
        timeline = build_timeline(player_id, player_name, reference_date, ref_row, windowed_features, 
                                 target1=0, target2=0, player_df=df, window_start_date=start_date)
        timelines.append(timeline)
    
    return timelines

def build_timeline(player_id: int, player_name: str, reference_date: pd.Timestamp, 
                  ref_row: pd.Series, windowed_features: Dict, 
                  target1: int = None, target2: int = None,
                  player_df: Optional[pd.DataFrame] = None,
                  window_start_date: Optional[pd.Timestamp] = None) -> Dict:
    """Build a complete timeline with normalized cumulative features (simplified version for V4)
    
    Args:
        player_id: Player ID
        player_name: Player name
        reference_date: Reference date for the timeline
        ref_row: Reference row from daily features
        windowed_features: Windowed features dictionary
        target1: Optional target1 value (muscular injuries). If None, target1 column is not included.
        target2: Optional target2 value (skeletal injuries). If None, target2 column is not included.
        player_df: Optional full player daily features dataframe to calculate career start and activity
        window_start_date: Optional start date of the 35-day window (for activity calculation)
    """
    # Build static features
    static_features = {
        'player_id': player_id,
        'reference_date': reference_date.strftime('%Y-%m-%d'),
        'player_name': player_name
    }
    
    # Add other static features
    for feature in get_static_features()[3:]:  # Skip technical features
        if feature in ref_row.index:
            static_features[feature] = ref_row[feature]
        else:
            static_features[feature] = None
    
    # Normalize cumulative features to rates/percentiles
    normalized_features = {}
    
    # Calculate years_active from career start to reference date
    years_active = 1.0  # Default to 1 year to avoid division by zero
    if player_df is not None and not player_df.empty and 'date' in player_df.columns:
        # Get first match date (career start)
        player_df['date'] = pd.to_datetime(player_df['date'])
        first_match_date = player_df['date'].min()
        if pd.notna(first_match_date):
            days_active = (reference_date - first_match_date).days
            years_active = max(1.0, days_active / 365.25)  # At least 1 year to avoid division issues
    else:
        # Fallback: use seasons_count if available
        seasons_count = ref_row.get('seasons_count', 1) if ref_row.get('seasons_count') is not None else 1
        years_active = max(1.0, seasons_count)
    
    # Get career duration in days and seasons for normalization
    seasons_count = ref_row.get('seasons_count', 1) if ref_row.get('seasons_count') is not None else 1
    career_matches = ref_row.get('career_matches', 0) if ref_row.get('career_matches') is not None else 0
    
    # Normalize cumulative features by years_active
    cum_features_to_normalize = [
        ('cum_minutes_played_numeric', 'minutes_per_year'),
        ('cum_goals_numeric', 'goals_per_year'),
        ('cum_assists_numeric', 'assists_per_year'),
        ('cum_yellow_cards_numeric', 'yellow_cards_per_year'),
        ('cum_red_cards_numeric', 'red_cards_per_year'),
        ('cum_matches_played', 'matches_played_per_year'),
        ('cum_matches_bench', 'matches_bench_per_year'),
        ('cum_matches_not_selected', 'matches_not_selected_per_year'),
        ('cum_matches_injured', 'matches_injured_per_year'),
        ('cum_competitions', 'competitions_per_year'),
    ]
    
    # Always set all normalized features, even if 0.0
    for cum_feature, normalized_name in cum_features_to_normalize:
        cum_value = ref_row.get(cum_feature, 0)
        if cum_value is not None and pd.notna(cum_value) and cum_value > 0:
            normalized_features[normalized_name] = cum_value / years_active
        else:
            normalized_features[normalized_name] = 0.0
    
    # Injury cumulative features - always set
    cum_inj_starts = ref_row.get('cum_inj_starts', 0) if ref_row.get('cum_inj_starts') is not None else 0
    if cum_inj_starts is not None and pd.notna(cum_inj_starts) and cum_inj_starts > 0:
        normalized_features['injuries_per_year'] = cum_inj_starts / years_active
    else:
        normalized_features['injuries_per_year'] = 0.0
    
    cum_inj_days = ref_row.get('cum_inj_days', 0) if ref_row.get('cum_inj_days') is not None else 0
    if cum_inj_days is not None and pd.notna(cum_inj_days) and cum_inj_days > 0:
        normalized_features['injury_days_per_year'] = cum_inj_days / years_active
    else:
        normalized_features['injury_days_per_year'] = 0.0
    
    # Career totals normalization
    career_goals = ref_row.get('career_goals', 0) if ref_row.get('career_goals') is not None else 0
    career_assists = ref_row.get('career_assists', 0) if ref_row.get('career_assists') is not None else 0
    career_minutes = ref_row.get('career_minutes', 0) if ref_row.get('career_minutes') is not None else 0
    
    if career_goals > 0:
        normalized_features['career_goals_per_year'] = career_goals / years_active
    else:
        normalized_features['career_goals_per_year'] = 0.0
    
    if career_assists > 0:
        normalized_features['career_assists_per_year'] = career_assists / years_active
    else:
        normalized_features['career_assists_per_year'] = 0.0
    
    if career_minutes > 0:
        normalized_features['career_minutes_per_year'] = career_minutes / years_active
    else:
        normalized_features['career_minutes_per_year'] = 0.0
    
    # Normalize by career matches (rates)
    if career_matches and career_matches > 0:
        if career_goals > 0:
            normalized_features['goals_per_career_match'] = career_goals / career_matches
        else:
            normalized_features['goals_per_career_match'] = 0.0
        
        if career_assists > 0:
            normalized_features['assists_per_career_match'] = career_assists / career_matches
        else:
            normalized_features['assists_per_career_match'] = 0.0
        
        if career_minutes > 0:
            normalized_features['minutes_per_career_match'] = career_minutes / career_matches
        else:
            normalized_features['minutes_per_career_match'] = 0.0
        
        if cum_inj_starts > 0:
            normalized_features['injuries_per_career_match'] = cum_inj_starts / career_matches
        else:
            normalized_features['injuries_per_career_match'] = 0.0
    else:
        normalized_features['goals_per_career_match'] = 0.0
        normalized_features['assists_per_career_match'] = 0.0
        normalized_features['minutes_per_career_match'] = 0.0
        normalized_features['injuries_per_career_match'] = 0.0
    
    # Calculate activity flag (has_minimum_activity)
    has_minimum_activity = 0
    if window_start_date is not None and player_df is not None and not player_df.empty:
        # Check total minutes in the 35-day window
        window_mask = (player_df['date'] >= window_start_date) & (player_df['date'] <= reference_date)
        window_df = player_df[window_mask]
        if len(window_df) > 0 and 'minutes_played_numeric' in window_df.columns:
            total_minutes = window_df['minutes_played_numeric'].sum()
            if total_minutes >= MIN_ACTIVITY_MINUTES:
                has_minimum_activity = 1
    
    # Combine features
    timeline = {**static_features, **windowed_features, **normalized_features}
    
    # Add activity flag
    timeline['has_minimum_activity'] = has_minimum_activity
    
    # Add targets if provided (for training/backtesting)
    if target1 is not None:
        timeline['target1'] = target1
    if target2 is not None:
        timeline['target2'] = target2
    
    return timeline

# ===== HELPER FUNCTIONS =====

def get_all_player_ids(daily_features_dir: str) -> List[int]:
    """Get all player IDs from the daily features directory"""
    player_ids = []
    
    if not os.path.exists(daily_features_dir):
        raise FileNotFoundError(f"Daily features directory not found: {daily_features_dir}")
    
    for filename in os.listdir(daily_features_dir):
        if filename.startswith('player_') and filename.endswith('_daily_features.csv'):
            player_id = int(filename.split('_')[1])
            player_ids.append(player_id)
    
    print(f"   Found {len(player_ids)} players")
    return sorted(player_ids)

def load_player_names_mapping(players_profile_file: str) -> Dict[int, str]:
    """Load player names from players_profile.csv"""
    player_names = {}
    
    if not os.path.exists(players_profile_file):
        print(f"⚠️  Players profile file not found: {players_profile_file}")
        return player_names
    
    try:
        players_df = pd.read_csv(players_profile_file, sep=';', encoding='utf-8-sig')
        if 'id' in players_df.columns and 'name' in players_df.columns:
            for _, row in players_df.iterrows():
                player_id = row.get('id')
                player_name = row.get('name', '')
                if pd.notna(player_id) and pd.notna(player_name) and player_name:
                    player_names[int(player_id)] = str(player_name).strip()
            print(f"✅ Loaded {len(player_names)} player names")
    except Exception as e:
        print(f"⚠️  Error loading player names: {e}")
    
    return player_names

def get_player_name_from_df(df: pd.DataFrame, player_id: int = None, player_names_map: Dict[int, str] = None) -> str:
    """Get player name from the daily features dataframe or player names mapping"""
    # First try to get player_id from dataframe if not provided
    if player_id is None:
        player_id = df['player_id'].iloc[0] if 'player_id' in df.columns else None
    
    # Try player names mapping first (most reliable)
    if player_id and player_names_map and player_id in player_names_map:
        return player_names_map[player_id]
    
    # Fallback to player_id format
    return f"Player_{player_id}" if player_id else "Unknown"

def save_timelines_to_csv_chunked(timelines: List[Dict], output_file: str, chunk_size: int = 10000):
    """
    Save timelines to CSV in chunks to avoid memory issues.
    
    Args:
        timelines: List of dictionaries containing timeline data
        output_file: Output CSV file path
        chunk_size: Number of rows to process at a time
    """
    if not timelines:
        return
    
    # Get all unique keys from all timelines to ensure consistent columns
    all_keys = set()
    for timeline in timelines:
        all_keys.update(timeline.keys())
    
    # Sort keys for consistent column order
    fieldnames = sorted(all_keys)
    
    # Write CSV in chunks
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        
        # Write in chunks to manage memory
        for i in range(0, len(timelines), chunk_size):
            chunk = timelines[i:i + chunk_size]
            for timeline in chunk:
                # Ensure all keys are present (fill missing with None)
                row = {key: timeline.get(key, None) for key in fieldnames}
                writer.writerow(row)
    
    # Get shape info by reading just the first few rows
    sample_df = pd.read_csv(output_file, nrows=1)
    num_cols = len(sample_df.columns)
    num_rows = len(timelines)
    del sample_df  # Free memory
    return (num_rows, num_cols)

# ===== MODEL-SPECIFIC FILTERING FUNCTIONS =====

def filter_timelines_for_model(timelines_df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Filter timelines for a specific model, excluding other injury types from negatives.
    
    This ensures each model only learns to distinguish its target injury type from non-injuries,
    not from other injury types.
    
    For Model 1 (muscular): Only includes timelines where target1=1 (positives) 
    or (target1=0 AND target2=0) (negatives - only non-injuries)
    
    For Model 2 (skeletal): Only includes timelines where target2=1 (positives)
    or (target1=0 AND target2=0) (negatives - only non-injuries)
    
    Args:
        timelines_df: DataFrame with target1 and target2 columns
        target_column: 'target1' for muscular model, 'target2' for skeletal model
        
    Returns:
        Filtered DataFrame with only relevant timelines for the specified model
        
    Example:
        >>> # Load timelines
        >>> timelines_df = pd.read_csv('timelines_35day_season_2024_2025_v4_muscular_train.csv')
        >>> 
        >>> # Filter for Model 1 (muscular)
        >>> model1_data = filter_timelines_for_model(timelines_df, 'target1')
        >>> # Use model1_data['target1'] as the target variable
        >>> 
        >>> # Filter for Model 2 (skeletal)
        >>> model2_data = filter_timelines_for_model(timelines_df, 'target2')
        >>> # Use model2_data['target2'] as the target variable
    """
    if target_column not in ['target1', 'target2']:
        raise ValueError(f"Invalid target_column: {target_column}. Must be 'target1' or 'target2'")
    
    # Validate required columns exist
    if 'target1' not in timelines_df.columns or 'target2' not in timelines_df.columns:
        raise ValueError("DataFrame must contain both 'target1' and 'target2' columns")
    
    if target_column == 'target1':
        # Model 1 (muscular): Include muscular injuries (target1=1) and non-injuries (both=0)
        # Exclude skeletal injuries (target2=1, target1=0)
        mask = (timelines_df['target1'] == 1) | ((timelines_df['target1'] == 0) & (timelines_df['target2'] == 0))
        filtered_df = timelines_df[mask].copy()
        
        excluded_count = ((timelines_df['target1'] == 0) & (timelines_df['target2'] == 1)).sum()
        positives = filtered_df['target1'].sum()
        negatives = ((filtered_df['target1'] == 0) & (filtered_df['target2'] == 0)).sum()
        
        print(f"\n📊 Filtered for Model 1 (Muscular Injuries):")
        print(f"   Original timelines: {len(timelines_df):,}")
        print(f"   After filtering: {len(filtered_df):,}")
        print(f"   Positives (target1=1): {positives:,}")
        print(f"   Negatives (target1=0, target2=0): {negatives:,}")
        print(f"   Excluded (skeletal injuries): {excluded_count:,}")
        if len(filtered_df) > 0:
            print(f"   Target ratio: {positives / len(filtered_df) * 100:.2f}%")
        
    else:  # target_column == 'target2'
        # Model 2 (skeletal): Include skeletal injuries (target2=1) and non-injuries (both=0)
        # Exclude muscular injuries (target1=1, target2=0)
        mask = (timelines_df['target2'] == 1) | ((timelines_df['target1'] == 0) & (timelines_df['target2'] == 0))
        filtered_df = timelines_df[mask].copy()
        
        excluded_count = ((timelines_df['target1'] == 1) & (timelines_df['target2'] == 0)).sum()
        positives = filtered_df['target2'].sum()
        negatives = ((filtered_df['target1'] == 0) & (filtered_df['target2'] == 0)).sum()
        
        print(f"\n📊 Filtered for Model 2 (Skeletal Injuries):")
        print(f"   Original timelines: {len(timelines_df):,}")
        print(f"   After filtering: {len(filtered_df):,}")
        print(f"   Positives (target2=1): {positives:,}")
        print(f"   Negatives (target1=0, target2=0): {negatives:,}")
        print(f"   Excluded (muscular injuries): {excluded_count:,}")
        if len(filtered_df) > 0:
            print(f"   Target ratio: {positives / len(filtered_df) * 100:.2f}%")
    
    return filtered_df

# ===== SEASON PROCESSING (PL filtering done in post-processing) =====

def process_season(season_start_year: int, daily_features_dir: str, 
                  all_injury_dates_by_player: Dict[int, Set[pd.Timestamp]],
                  injury_class_map: Dict[Tuple[int, pd.Timestamp], str],
                  player_names_map: Dict[int, str],
                  player_ids: List[int],
                  output_dir: str,
                  max_players: Optional[int] = None) -> Tuple[Optional[str], int, int, int]:
    """Process a single season and generate timeline file (PL filtering done in post-processing)"""
    season_start, season_end = get_season_date_range(season_start_year)
    
    # Determine if this is train or test season
    is_test_season = season_start_year >= TEST_SEASON_START
    if is_test_season:
        output_file = str(V4_TIMELINES_TEST / f'timelines_35day_season_{season_start_year}_{season_start_year+1}_v4_muscular_test.csv')
    else:
        output_file = str(V4_TIMELINES_TRAIN / f'timelines_35day_season_{season_start_year}_{season_start_year+1}_v4_muscular_train.csv')
    
    print(f"\n{'='*80}")
    print(f"PROCESSING SEASON {season_start_year}_{season_start_year+1} ({'TEST' if is_test_season else 'TRAIN'})")
    print(f"{'='*80}")
    print(f"   Date range: {season_start.date()} to {season_end.date()}")
    
    season_start_time = datetime.now()
    
    # Apply limit if specified (for testing)
    if max_players is not None:
        season_player_ids = player_ids[:max_players]
    else:
        season_player_ids = player_ids
    
    # ===== PASS 1: Generate all injury timelines and collect valid non-injury dates =====
    print(f"\n📊 PASS 1: Generating injury timelines and identifying valid dates")
    
    all_injury_timelines = []
    all_valid_non_injury_dates = []
    processed_players = 0
    
    for player_id in tqdm(season_player_ids, desc=f"Pass 1: {season_start_year}_{season_start_year+1}", unit="player"):
        try:
            # Load player data
            df = pd.read_csv(f'{daily_features_dir}/player_{player_id}_daily_features.csv')
            df['date'] = pd.to_datetime(df['date'])
            
            # Filter to season date range (with 35-day buffer before season start for windows)
            buffer_start = season_start - timedelta(days=34)
            season_mask = (df['date'] >= buffer_start) & (df['date'] <= season_end)
            df_season = df[season_mask].copy()
            
            if len(df_season) == 0:
                continue  # No data for this season
            
            # Get player name
            player_name = get_player_name_from_df(df_season, player_id=player_id, player_names_map=player_names_map)
            
            # Get all injury dates for this player (any class) for non-injury validation
            player_all_injury_dates = all_injury_dates_by_player.get(player_id, set())
            
            # Generate injury timelines (filtered by injury_class and season, NO PL filtering)
            injury_timelines = generate_injury_timelines_enhanced(
                player_id, player_name, df_season, injury_class_map
            )
            
            # Filter injury timelines to this season
            season_injury_timelines = []
            for timeline in injury_timelines:
                ref_date_str = timeline.get('reference_date', '')
                if ref_date_str:
                    ref_date = pd.to_datetime(ref_date_str)
                    if season_start <= ref_date <= season_end:
                        season_injury_timelines.append(timeline)
            
            all_injury_timelines.extend(season_injury_timelines)
            
            # Get valid non-injury dates for this season (NO PL filtering)
            valid_dates = get_valid_non_injury_dates(
                df_season,
                season_start=season_start,
                season_end=season_end,
                all_injury_dates=player_all_injury_dates
            )
            for date in valid_dates:
                all_valid_non_injury_dates.append((player_id, date))
            
            processed_players += 1
            del df, df_season
                
        except Exception as e:
            print(f"\n❌ Error processing player {player_id} for season {season_start_year}: {e}")
            continue
    
    print(f"\n✅ PASS 1 Complete for {season_start_year}_{season_start_year+1}:")
    print(f"   Injury timelines: {len(all_injury_timelines)}")
    print(f"   Valid non-injury dates: {len(all_valid_non_injury_dates)}")
    
    # Use natural target ratio (all available dates)
    selected_dates = all_valid_non_injury_dates
    print(f"\n📊 DATASET COMPOSITION (Natural Ratio, All Players):")
    print(f"   Injury timelines: {len(all_injury_timelines)}")
    print(f"   Non-injury dates: {len(selected_dates)}")
    print(f"   ✅ Using all {len(selected_dates)} available dates")
    
    # ===== PASS 2: Generate timelines for selected dates =====
    print(f"\n📊 PASS 2: Generating non-injury timelines")
    
    all_non_injury_timelines = []
    dates_by_player = defaultdict(list)
    for player_id, date in selected_dates:
        dates_by_player[player_id].append(date)
    
    for player_id, dates in tqdm(dates_by_player.items(), desc=f"Pass 2: {season_start_year}_{season_start_year+1}", unit="player"):
        try:
            df = pd.read_csv(f'{daily_features_dir}/player_{player_id}_daily_features.csv')
            df['date'] = pd.to_datetime(df['date'])
            
            # Filter to season date range (with buffer)
            buffer_start = season_start - timedelta(days=34)
            season_mask = (df['date'] >= buffer_start) & (df['date'] <= season_end)
            df_season = df[season_mask].copy()
            
            if len(df_season) == 0:
                continue
            
            player_name = get_player_name_from_df(df_season, player_id=player_id, player_names_map=player_names_map)
            
            # Generate non-injury timelines
            non_injury_timelines = generate_non_injury_timelines_for_dates(player_id, player_name, df_season, dates)
            all_non_injury_timelines.extend(non_injury_timelines)
            
            del df, df_season
            
        except Exception as e:
            print(f"\n❌ Error generating non-injury timelines for player {player_id}: {e}")
            continue
    
    # Combine and save
    print(f"\n📊 FINALIZING DATASET FOR {season_start_year}_{season_start_year+1}")
    
    injury_count = len(all_injury_timelines)
    non_injury_count = len(all_non_injury_timelines)
    final_timelines = all_injury_timelines + all_non_injury_timelines
    random.shuffle(final_timelines)
    
    total_count = len(final_timelines)
    final_ratio = (injury_count / total_count) if total_count > 0 else 0.0
    
    # Calculate statistics for dual targets
    target1_count = sum(1 for t in final_timelines if t.get('target1') == 1)
    target2_count = sum(1 for t in final_timelines if t.get('target2') == 1)
    both_targets_count = sum(1 for t in final_timelines if t.get('target1') == 1 and t.get('target2') == 1)
    activity_flag_count = sum(1 for t in final_timelines if t.get('has_minimum_activity') == 1)
    
    final_ratio1 = (target1_count / total_count) if total_count > 0 else 0.0
    final_ratio2 = (target2_count / total_count) if total_count > 0 else 0.0
    
    print(f"\n📈 SEASON {season_start_year}_{season_start_year+1} DATASET (PL-only):")
    print(f"   Total timelines: {total_count:,}")
    print(f"   Injury timelines (target1=1): {target1_count:,}")
    print(f"   Injury timelines (target2=1): {target2_count:,}")
    print(f"   Injury timelines (both=1): {both_targets_count:,}")
    print(f"   Non-injury timelines (both=0): {non_injury_count:,}")
    if total_count > 0:
        print(f"   Timelines with minimum activity: {activity_flag_count:,} ({activity_flag_count/total_count*100:.1f}%)")
        print(f"   Final target1 ratio: {final_ratio1:.1%} (natural)")
        print(f"   Final target2 ratio: {final_ratio2:.1%} (natural)")
    else:
        print(f"   Timelines with minimum activity: {activity_flag_count:,} (N/A - no timelines)")
        print(f"   Final target1 ratio: N/A (no timelines)")
        print(f"   Final target2 ratio: N/A (no timelines)")
    
    if final_timelines:
        print(f"\n💾 Saving timelines to CSV...")
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        shape = save_timelines_to_csv_chunked(final_timelines, output_file)
        print(f"✅ Saved to: {output_file}")
        print(f"📊 Shape: {shape}")
    else:
        print("⚠️  No timelines generated for this season")
        output_file = None
    
    season_time = datetime.now() - season_start_time
    print(f"⏱️  Season processing time: {season_time}")
    
    return output_file, target1_count, target2_count, non_injury_count

def main(max_players: Optional[int] = None, seasons: Optional[List[int]] = None):
    """Main function - processes all seasons with PL filtering
    
    Args:
        max_players: Optional limit on number of players to process (for testing)
        seasons: Optional list of season start years to process. If None, processes all available seasons.
    """
    print("🚀 V4 ENHANCED 35-DAY TIMELINE GENERATOR - PL-ONLY")
    print("=" * 80)
    print("📋 Features: V4 enhanced features with 35-day windows")
    print("⚡ Processing: Season-by-season with natural target ratios")
    print("🎯 Target ratios: Natural (all available positives and negatives)")
    print(f"📊 Activity flag: has_minimum_activity (≥{MIN_ACTIVITY_MINUTES} minutes in 35-day window)")
    print(f"🔍 Injury filtering: Only muscular injuries for injury timelines")
    print(f"🏆 PL filtering: Only timelines when player was in Premier League")
    print(f"📊 Season split: Train (≤{TRAIN_SEASON_CUTOFF}/25), Test ({TEST_SEASON_START}/26)")
    
    if seasons:
        print(f"📅 Processing specific seasons: {seasons}")
    else:
        print(f"📅 Processing all available seasons")
    
    start_time = datetime.now()
    
    # Setup paths
    # Prefer enriched daily features (Layer 2) if available
    if V4_DAILY_FEATURES_ENRICHED.exists():
        daily_features_dir = str(V4_DAILY_FEATURES_ENRICHED)
        print(f"\n📂 Using ENRICHED daily features directory: {daily_features_dir}")
    else:
        daily_features_dir = str(V4_DAILY_FEATURES)
        print(f"\n⚠️ Enriched daily features not found, falling back to BASE: {daily_features_dir}")
    
    raw_data_dir = str(V4_RAW_DATA)
    match_data_dir = str(V4_RAW_DATA / "match_data")
    career_file = str(V4_RAW_DATA / "players_career.csv")
    injuries_file = str(V4_RAW_DATA / "injuries_data.csv")
    players_profile_file = str(V4_RAW_DATA / "players_profile.csv")
    
    # Validate paths
    if not os.path.exists(daily_features_dir):
        raise FileNotFoundError(f"Daily features directory not found: {daily_features_dir}")
    if not os.path.exists(match_data_dir):
        raise FileNotFoundError(f"Match data directory not found: {match_data_dir}")
    if not os.path.exists(career_file):
        raise FileNotFoundError(f"Career file not found: {career_file}")
    if not os.path.exists(injuries_file):
        raise FileNotFoundError(f"Injuries file not found: {injuries_file}")
    
    # Get available seasons
    available_seasons = get_all_seasons_from_daily_features(daily_features_dir)
    
    # Filter to requested seasons if specified
    if seasons is not None:
        # Validate that requested seasons are available
        requested_seasons = [s for s in seasons if s in available_seasons]
        missing_seasons = [s for s in seasons if s not in available_seasons]
        
        if missing_seasons:
            print(f"\n⚠️  WARNING: Requested seasons not found in data: {missing_seasons}")
        
        if not requested_seasons:
            print(f"\n❌ ERROR: None of the requested seasons are available in the data.")
            print(f"   Available seasons: {sorted(available_seasons)}")
            return
        
        seasons_to_process = sorted(requested_seasons)
        print(f"\n✅ Will process {len(seasons_to_process)} requested season(s): {seasons_to_process}")
    else:
        seasons_to_process = sorted(available_seasons)
        print(f"\n✅ Will process all {len(seasons_to_process)} available seasons")
    
    # Step 1: Build PL clubs per season
    print("\n" + "=" * 80)
    print("Step 1: Building PL clubs per season mapping")
    print("=" * 80)
    pl_clubs_by_season = build_pl_clubs_per_season(match_data_dir)
    
    if not pl_clubs_by_season:
        print("ERROR: No PL clubs found. Cannot proceed.")
        return
    
    # Step 2: Build player PL membership periods
    print("\n" + "=" * 80)
    print("Step 2: Building player PL membership periods")
    print("=" * 80)
    player_pl_periods = build_player_pl_membership_periods(career_file, pl_clubs_by_season)
    
    if not player_pl_periods:
        print("ERROR: No PL membership periods found. Cannot proceed.")
        return
    
    # Step 3: Load injuries data (once, shared across seasons)
    print("\n" + "=" * 80)
    print("Step 3: Loading injuries data")
    print("=" * 80)
    injury_class_map = load_injuries_data(injuries_file)
    all_injury_dates_by_player = load_all_injury_dates(injuries_file)
    print(f"✅ Loaded injury data for {len(all_injury_dates_by_player)} players")
    
    # Step 4: Get all player IDs
    print("\n" + "=" * 80)
    print("Step 4: Getting player IDs")
    print("=" * 80)
    all_player_ids = get_all_player_ids(daily_features_dir)
    print(f"✅ Found {len(all_player_ids)} players")
    
    # Step 5: Load player names mapping
    print("\n" + "=" * 80)
    print("Step 5: Loading player names")
    print("=" * 80)
    player_names_map = load_player_names_mapping(players_profile_file)
    if player_names_map:
        print(f"✅ Loaded {len(player_names_map)} player names")
    else:
        print("⚠️  No player names loaded - will use Player_ID format")
    
    # Apply limit if specified (for testing)
    if max_players is not None:
        player_ids = all_player_ids[:max_players]
        print(f"\n🧪 TEST MODE: Processing {len(player_ids)} players (limited from {len(all_player_ids)})")
    else:
        player_ids = all_player_ids
    
    # Step 6: Process each season
    print("\n" + "=" * 80)
    print("Step 6: Processing seasons")
    print("=" * 80)
    
    output_files = []
    total_target1 = 0
    total_target2 = 0
    total_non_injuries = 0
    
    for season_start_year in seasons_to_process:
        output_file, target1_count, target2_count, non_injury_count = process_season(
            season_start_year,
            daily_features_dir,
            all_injury_dates_by_player,
            injury_class_map,
            player_names_map,
            player_ids,
            player_pl_periods,
            str(V4_TIMELINES),
            max_players
        )
        if output_file:
            output_files.append(output_file)
            total_target1 += target1_count
            total_target2 += target2_count
            total_non_injuries += non_injury_count
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*80}")
    total_timelines = total_target1 + total_target2 + total_non_injuries
    print(f"   Processed {len(output_files)} seasons")
    print(f"   Total timelines: {total_timelines:,}")
    print(f"   Total injury timelines (target1=1): {total_target1:,}")
    print(f"   Total injury timelines (target2=1): {total_target2:,}")
    print(f"   Total non-injury timelines (both=0): {total_non_injuries:,}")
    if total_timelines > 0:
        print(f"   Final target1 ratio: {total_target1 / total_timelines:.1%}")
        print(f"   Final target2 ratio: {total_target2 / total_timelines:.1%}")
    print(f"   Note: Activity flag statistics available in individual season files")
    print(f"\n   Generated files:")
    for f in output_files:
        print(f"      - {f}")
    
    total_time = datetime.now() - start_time
    print(f"\n⏱️  Total execution time: {total_time}")

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate 35-day timelines for V4 model')
    parser.add_argument('--test', type=int, metavar='N', 
                       help='Test mode: process only first N players')
    parser.add_argument('--seasons', type=int, nargs='+', metavar='YEAR',
                       help='Process specific seasons (season start years, e.g., --seasons 2024 2025)')
    parser.add_argument('--season', type=int, metavar='YEAR',
                       help='Process single season (season start year, e.g., --season 2025)')
    
    args = parser.parse_args()
    
    # Determine seasons to process
    seasons = None
    if args.seasons:
        seasons = args.seasons
    elif args.season:
        seasons = [args.season]
    
    # Determine max_players
    max_players = args.test if args.test else None
    
    main(max_players=max_players, seasons=seasons)
