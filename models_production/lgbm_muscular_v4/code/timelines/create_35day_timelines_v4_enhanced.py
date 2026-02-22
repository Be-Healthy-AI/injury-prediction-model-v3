#!/usr/bin/env python3
"""
V4 Enhanced 35-Day Timeline Generator - Dual target (muscular + skeletal)
Generates timelines from V4 daily features with PL-only filtering.

Key features:
- Generates timelines season by season (YYYY_YYYY+1 format)
- Two targets: target1 = muscular injuries, target2 = skeletal injuries.
- Positives (model-specific):
  - Muscular (target1=1): reference date in [D-14, D-1] (inclusive), D = first day of muscular injury.
  - Skeletal (target2=1): reference date in [D-21, D-1] (inclusive), D = first day of skeletal injury.
- Negatives (same for both): target1=0, target2=0 when:
  - No muscular/skeletal/unknown injury in [D-28, D+28] (28 days before or after reference date D);
  - Player selected (played or on bench) at least once in [D-14, D] (activity condition).
- Same labeling rules apply to train and test (seasons 2018/19 to 2025/26).
- PL-only filtering using career data
- Season segmentation: train (≤2024/25) and test (2025/26)
- Activity flag: has_minimum_activity (≥90 minutes in 35-day window)
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
# Injury classes that disqualify a reference date for non-injury
INJURY_CLASSES_DISQUALIFYING_NON_INJURY = {'muscular', 'skeletal', 'unknown'}

# Positive labeling windows (reference dates before injury day D)
MUSCULAR_POSITIVE_DAYS_BEFORE = 14   # D-14 to D-1 (inclusive) for muscular
SKELETAL_POSITIVE_DAYS_BEFORE = 21   # D-21 to D-1 (inclusive) for skeletal
# Negative: no injury in this window around reference date D
NEGATIVE_NO_INJURY_DAYS = 28        # [D-28, D+28]
# Activity: at least 1 selection (played or on bench) in [D-14, D]
ACTIVITY_DAYS_BEFORE = 14

# Activity requirement configuration (has_minimum_activity flag)
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

def get_static_pl_clubs_by_season() -> Dict[int, Set[str]]:
    """
    Get static PL clubs mapping by season based on TransferMarkt data.
    This provides a baseline mapping that can be merged with dynamic data from match files.
    Includes common club name variations to handle different naming conventions.
    
    Returns:
        Dictionary mapping season start year to set of PL club names (original and normalized)
    """
    # Static PL clubs by season (2019-2025) based on TransferMarkt data
    # Each season includes common variations for all clubs
    static_clubs = {
        2019: {
            'Manchester City FC', 'FC Arsenal', 'Arsenal FC', 'Chelsea FC', 'Manchester United FC',
            'Liverpool FC', 'Tottenham Hotspur', 'Newcastle United', 'Brighton & Hove Albion',
            'Aston Villa FC', 'Wolverhampton Wanderers', 'Leicester City FC', 'West Ham United',
            'FC Southampton', 'FC Everton', 'Norwich City FC', 'Sheffield United',
            'Crystal Palace FC', 'FC Burnley', 'Watford FC', 'AFC Bournemouth'
        },
        2020: {
            'Manchester City FC', 'FC Arsenal', 'Arsenal FC', 'Chelsea FC', 'Manchester United FC',
            'Liverpool FC', 'Tottenham Hotspur', 'Newcastle United', 'Brighton & Hove Albion',
            'Aston Villa FC', 'Wolverhampton Wanderers', 'Leicester City FC', 'West Ham United',
            'FC Southampton', 'FC Everton', 'Sheffield United', 'Crystal Palace FC',
            'FC Burnley', 'FC Fulham', 'West Bromwich Albion', 'Leeds United FC'
        },
        2021: {
            'Manchester City FC', 'FC Arsenal', 'Arsenal FC', 'Chelsea FC', 'Manchester United FC',
            'Liverpool FC', 'Tottenham Hotspur', 'Newcastle United', 'Brighton & Hove Albion',
            'Aston Villa FC', 'Wolverhampton Wanderers', 'Leicester City FC', 'West Ham United',
            'FC Southampton', 'FC Everton', 'Crystal Palace FC', 'FC Burnley',
            'FC Fulham', 'Brentford FC', 'Norwich City FC', 'Watford FC'
        },
        2022: {
            'Manchester City FC', 'FC Arsenal', 'Arsenal FC', 'Chelsea FC', 'Manchester United FC',
            'Liverpool FC', 'Tottenham Hotspur', 'Newcastle United', 'Brighton & Hove Albion',
            'Aston Villa FC', 'Wolverhampton Wanderers', 'Leicester City FC', 'West Ham United',
            'FC Southampton', 'FC Everton', 'Nottingham Forest', 'Brentford FC',
            'Leeds United FC', 'Crystal Palace FC', 'FC Fulham', 'AFC Bournemouth'
        },
        2023: {
            'Manchester City FC', 'FC Arsenal', 'Arsenal FC', 'Chelsea FC', 'Liverpool FC',
            'Tottenham Hotspur', 'Manchester United FC', 'Aston Villa FC', 'Newcastle United',
            'Brighton & Hove Albion', 'Nottingham Forest', 'West Ham United', 'Crystal Palace FC',
            'Wolverhampton Wanderers', 'Brentford FC', 'AFC Bournemouth', 'FC Everton',
            'FC Fulham', 'FC Burnley', 'Sheffield United', 'Luton Town'
        },
        2024: {
            'Manchester City FC', 'Chelsea FC', 'FC Arsenal', 'Arsenal FC', 'Liverpool FC',
            'Manchester United FC', 'Tottenham Hotspur', 'Aston Villa FC', 'Newcastle United',
            'Brighton & Hove Albion', 'Crystal Palace FC', 'AFC Bournemouth', 'Nottingham Forest',
            'Wolverhampton Wanderers', 'Brentford FC', 'West Ham United', 'FC Everton',
            'FC Fulham', 'FC Southampton', 'Ipswich Town FC', 'Leicester City FC'
        },
        2025: {
            'Manchester City FC', 'FC Arsenal', 'Arsenal FC', 'Chelsea FC', 'Liverpool FC',
            'Tottenham Hotspur', 'Manchester United FC', 'Newcastle United', 'Nottingham Forest',
            'Brighton & Hove Albion', 'Aston Villa FC', 'Crystal Palace FC', 'Brentford FC',
            'AFC Bournemouth', 'FC Everton', 'West Ham United', 'AFC Sunderland',
            'FC Fulham', 'Wolverhampton Wanderers', 'Leeds United FC', 'FC Burnley'
        }
    }
    
    # Convert to sets with all variations and normalized names for each season
    result = {}
    for season, clubs in static_clubs.items():
        result[season] = set()
        for club in clubs:
            # Add primary name and normalized version
            result[season].add(club)
            result[season].add(normalize_team_name(club))
            
            # Add common variations based on club name patterns
            # Arsenal variations
            if 'Arsenal' in club:
                result[season].add('Arsenal FC')
                result[season].add('FC Arsenal')
                result[season].add('Arsenal')
                result[season].add(normalize_team_name('Arsenal FC'))
                result[season].add(normalize_team_name('FC Arsenal'))
                result[season].add(normalize_team_name('Arsenal'))
            
            # Manchester City variations
            if 'Manchester City' in club:
                result[season].add('Manchester City')
                result[season].add('Man City')
                result[season].add(normalize_team_name('Manchester City'))
                result[season].add(normalize_team_name('Man City'))
            
            # Manchester United variations
            if 'Manchester United' in club:
                result[season].add('Manchester United')
                result[season].add('Man United')
                result[season].add('Man Utd')
                result[season].add(normalize_team_name('Manchester United'))
                result[season].add(normalize_team_name('Man United'))
                result[season].add(normalize_team_name('Man Utd'))
            
            # Tottenham variations
            if 'Tottenham' in club:
                result[season].add('Tottenham Hotspur')
                result[season].add('Tottenham')
                result[season].add('Spurs')
                result[season].add(normalize_team_name('Tottenham Hotspur'))
                result[season].add(normalize_team_name('Tottenham'))
                result[season].add(normalize_team_name('Spurs'))
            
            # Add FC prefix/suffix variations for clubs that might have them
            if club.startswith('FC '):
                # Remove FC prefix and add as suffix
                base_name = club[3:]
                result[season].add(f'{base_name} FC')
                result[season].add(normalize_team_name(f'{base_name} FC'))
            elif club.endswith(' FC') and not club.startswith('FC '):
                # Add FC prefix version
                base_name = club[:-3]
                result[season].add(f'FC {base_name}')
                result[season].add(normalize_team_name(f'FC {base_name}'))
    
    return result

def build_pl_clubs_per_season(raw_match_dir: str) -> Dict[int, Set[str]]:
    """
    Build mapping of season_year -> set of PL club names from raw match data.
    Now merges static PL clubs mapping (from TransferMarkt) with dynamic data from match files.
    
    Args:
        raw_match_dir: Directory containing raw match CSV files
        
    Returns:
        Dictionary mapping season start year to set of normalized PL club names
    """
    print("Building PL clubs per season mapping...")
    
    # Start with static PL clubs mapping
    pl_clubs_by_season = get_static_pl_clubs_by_season()
    print(f"  Loaded static PL clubs for {len(pl_clubs_by_season)} seasons")
    
    # Find all match CSV files
    match_files = glob.glob(os.path.join(raw_match_dir, "**", "*.csv"), recursive=True)
    
    if not match_files:
        print(f"  WARNING: No match files found in {raw_match_dir}")
        # Return static mapping even if no match files
        return pl_clubs_by_season
    
    print(f"  Found {len(match_files)} match files - merging with static mapping...")
    
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
                
                # Initialize season if not in static mapping
                if season not in pl_clubs_by_season:
                    pl_clubs_by_season[season] = set()
                
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
    print(f"  Final PL clubs mapping: {len(result)} seasons")
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
    pl_clubs_by_season: Dict[int, Set[str]],
    max_reference_date: Optional[pd.Timestamp] = None
) -> Dict[int, List[Tuple[pd.Timestamp, pd.Timestamp]]]:
    """
    Build mapping of player_id -> list of (start_date, end_date) periods when player was at PL club.
    
    Args:
        career_file: Path to players_career.csv file
        pl_clubs_by_season: Mapping of season -> set of PL club names
        max_reference_date: Maximum reference date for predictions calculation (replaces hardcoded date)
        
    Returns:
        Dictionary mapping player_id to list of (start, end) date tuples
        
    Note:
        PL clubs mapping completeness should be verified against TransferMarkt for each season.
        If missing clubs are found, update the build_pl_clubs_per_season function accordingly.
    """
    print("\nBuilding player PL membership periods from career data...")
    
    # Load career data
    try:
        career_df = pd.read_csv(career_file, sep=';', encoding='utf-8-sig', low_memory=False)
    except Exception as e:
        print(f"  ERROR: Failed to load career file: {e}")
        return {}
    
    # Parse dates - try DD/MM/YYYY format first to avoid MM-DD-YYYY misinterpretation
    # Store original column for retry
    original_dates = career_df['Date'].copy()
    
    # First, try DD/MM/YYYY format (European format with slashes)
    career_df['Date'] = pd.to_datetime(career_df['Date'], format='%d/%m/%Y', errors='coerce')
    
    # If that fails for many rows, try DD-MM-YYYY format (European format with dashes)
    if career_df['Date'].isna().sum() > len(career_df) * 0.5:
        career_df['Date'] = pd.to_datetime(original_dates, format='%d-%m-%Y', errors='coerce')
    
    # If that also fails, try auto-detect with dayfirst=True to prefer DD/MM/YYYY over MM/DD/YYYY
    if career_df['Date'].isna().sum() > len(career_df) * 0.5:
        career_df['Date'] = pd.to_datetime(original_dates, dayfirst=True, errors='coerce')
    
    # Filter out rows with invalid dates or missing club info
    # Keep "From" column for initial PL club detection (Issue 2)
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
        
        # Issue 2: Check if player was already at PL club at first transfer (initial PL club)
        if len(player_career) > 0:
            first_row = player_career.iloc[0]
            first_transfer_date = first_row['Date']
            from_club = str(first_row.get('From', '')).strip() if 'From' in first_row else ''
            
            if from_club and pd.notna(first_transfer_date):
                # Determine season for first transfer
                season_year = first_transfer_date.year if first_transfer_date.month >= 7 else first_transfer_date.year - 1
                # Check if origin club is PL club
                is_pl_origin = is_club_pl_club(from_club, season_year, pl_clubs_by_season)
                if is_pl_origin:
                    # Player was already at PL club, start period from first transfer date
                    current_pl_start = first_transfer_date
                    current_pl_club = from_club
        
        for idx, row in player_career.iterrows():
            transfer_date = row['Date']
            to_club = str(row['To']).strip()
            from_club = str(row.get('From', '')).strip() if 'From' in row else ''
            
            if not to_club or pd.isna(transfer_date):
                continue
            
            # Determine season for this transfer
            season_year = transfer_date.year if transfer_date.month >= 7 else transfer_date.year - 1
            
            # Check if destination club is PL club
            is_pl_destination = is_club_pl_club(to_club, season_year, pl_clubs_by_season)
            # Check if origin club is PL club (for transfers between PL clubs - Issue 2)
            is_pl_origin = is_club_pl_club(from_club, season_year, pl_clubs_by_season) if from_club else False
            
            if is_pl_destination:
                if current_pl_start is None:
                    # Starting a new PL period
                    current_pl_start = transfer_date
                    current_pl_club = to_club
                elif is_pl_origin:
                    # Issue 2: Transfer between PL clubs - continue the period, just update club
                    current_pl_club = to_club
                # else: already in PL period, destination is PL, so continue period
            else:
                # Transferring to non-PL club
                if current_pl_start is not None:
                    # Close the PL period at this transfer date
                    player_pl_periods[player_id].append((current_pl_start, transfer_date))
                    current_pl_start = None
                    current_pl_club = None
        
        # If still in PL at end of career, close the period
        if current_pl_start is not None:
            last_date = player_career['Date'].max()
            # Extend to end of that season + buffer
            end_season_year = last_date.year if last_date.month >= 7 else last_date.year - 1
            end_date = pd.Timestamp(f'{end_season_year + 1}-06-30')  # End of season
            
            # Issue 1: Use max_reference_date instead of hardcoded date
            if max_reference_date is not None:
                end_date = max(end_date, max_reference_date)
            else:
                # Fallback: use a reasonable future date (current date + 1 year)
                max_future = pd.Timestamp.now() + pd.Timedelta(days=365)
                end_date = max(end_date, max_future)
            
            player_pl_periods[player_id].append((current_pl_start, end_date))
    
    result = dict(player_pl_periods)
    print(f"  Found PL periods for {len(result)} players")
    
    # Report some statistics
    total_periods = sum(len(periods) for periods in result.values())
    print(f"  Total PL periods: {total_periods}")
    
    # Issue 3: Note about PL clubs mapping completeness
    print(f"\n  NOTE: PL clubs mapping completeness should be verified against TransferMarkt for each season.")
    print(f"        If missing clubs are found, update the build_pl_clubs_per_season function accordingly.")
    
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
            ts = pd.Timestamp(from_date)
            ts = ts.tz_localize(None) if getattr(ts, 'tz', None) is not None else ts
            injury_class_map[(int(player_id), ts.normalize())] = injury_class
    
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
    # Filter data for the 35-day window (normalized comparison for type/timezone robustness)
    dates_norm = pd.to_datetime(df['date']).dt.normalize()
    start_norm = pd.Timestamp(start_date).normalize()
    end_norm = pd.Timestamp(end_date).normalize()
    mask = (dates_norm >= start_norm) & (dates_norm <= end_norm)
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
    Generate injury timelines for muscular injuries only (target1=1).
    Each muscular injury on day D generates 14 reference dates: D-14, D-13, ..., D-1.
    Returns list of timeline dicts (target1=1; target2 set by caller when merging).
    """
    timelines = []
    injury_starts = df[df['cum_inj_starts'] > df['cum_inj_starts'].shift(1)]
    for _, injury_row in injury_starts.iterrows():
        injury_date = injury_row['date']
        injury_date_normalized = pd.Timestamp(injury_date).normalize()
        injury_class = injury_class_map.get((player_id, injury_date_normalized), '').lower()
        if injury_class not in ALLOWED_INJURY_CLASSES_MUSCULAR:
            continue
        for days_before in range(1, MUSCULAR_POSITIVE_DAYS_BEFORE + 1):
            reference_date = injury_date - timedelta(days=days_before)
            start_date = reference_date - timedelta(days=34)
            if start_date < df['date'].min():
                continue
            windowed_features = create_windowed_features_vectorized(df, start_date, reference_date)
            if windowed_features is None:
                continue
            ref_mask = (pd.to_datetime(df['date']).dt.normalize() == pd.Timestamp(reference_date).normalize())
            if not ref_mask.any():
                continue
            ref_row = df[ref_mask].iloc[0]
            timeline = build_timeline(player_id, player_name, reference_date, ref_row, windowed_features,
                                     target1=1, target2=None, player_df=df, window_start_date=start_date)
            timelines.append(timeline)
    return timelines


def generate_skeletal_injury_timelines_enhanced(player_id: int, player_name: str, df: pd.DataFrame,
                                                injury_class_map: Dict[Tuple[int, pd.Timestamp], str]) -> List[Dict]:
    """
    Generate injury timelines for skeletal injuries only (target2=1).
    Each skeletal injury on day D generates 21 reference dates: D-21, D-20, ..., D-1.
    Returns list of timeline dicts (target2=1; target1 set by caller when merging).
    """
    timelines = []
    injury_starts = df[df['cum_inj_starts'] > df['cum_inj_starts'].shift(1)]
    for _, injury_row in injury_starts.iterrows():
        injury_date = injury_row['date']
        injury_date_normalized = pd.Timestamp(injury_date).normalize()
        injury_class = injury_class_map.get((player_id, injury_date_normalized), '').lower()
        if injury_class not in ALLOWED_INJURY_CLASSES_SKELETAL:
            continue
        for days_before in range(1, SKELETAL_POSITIVE_DAYS_BEFORE + 1):
            reference_date = injury_date - timedelta(days=days_before)
            start_date = reference_date - timedelta(days=34)
            if start_date < df['date'].min():
                continue
            windowed_features = create_windowed_features_vectorized(df, start_date, reference_date)
            if windowed_features is None:
                continue
            ref_mask = (pd.to_datetime(df['date']).dt.normalize() == pd.Timestamp(reference_date).normalize())
            if not ref_mask.any():
                continue
            ref_row = df[ref_mask].iloc[0]
            timeline = build_timeline(player_id, player_name, reference_date, ref_row, windowed_features,
                                     target1=None, target2=1, player_df=df, window_start_date=start_date)
            timelines.append(timeline)
    return timelines


def get_muscular_positive_reference_dates(
    player_id: int,
    df: pd.DataFrame,
    injury_class_map: Dict[Tuple[int, pd.Timestamp], str],
    season_start: pd.Timestamp,
    season_end: pd.Timestamp,
) -> Set[pd.Timestamp]:
    """Return set of reference dates that are muscular positives (D-14..D-1 before a muscular injury D)."""
    out = set()
    injury_starts = df[df['cum_inj_starts'] > df['cum_inj_starts'].shift(1)]
    for _, injury_row in injury_starts.iterrows():
        injury_date = injury_row['date']
        injury_date_normalized = pd.Timestamp(injury_date).normalize()
        injury_class = injury_class_map.get((player_id, injury_date_normalized), '').lower()
        if injury_class not in ALLOWED_INJURY_CLASSES_MUSCULAR:
            continue
        for days_before in range(1, MUSCULAR_POSITIVE_DAYS_BEFORE + 1):
            reference_date = injury_date - timedelta(days=days_before)
            ref_n = pd.Timestamp(reference_date).normalize()
            if season_start <= ref_n <= season_end:
                out.add(ref_n)
    return out


def get_skeletal_positive_reference_dates(
    player_id: int,
    df: pd.DataFrame,
    injury_class_map: Dict[Tuple[int, pd.Timestamp], str],
    season_start: pd.Timestamp,
    season_end: pd.Timestamp,
) -> Set[pd.Timestamp]:
    """Return set of reference dates that are skeletal positives (D-21..D-1 before a skeletal injury D)."""
    out = set()
    injury_starts = df[df['cum_inj_starts'] > df['cum_inj_starts'].shift(1)]
    for _, injury_row in injury_starts.iterrows():
        injury_date = injury_row['date']
        injury_date_normalized = pd.Timestamp(injury_date).normalize()
        injury_class = injury_class_map.get((player_id, injury_date_normalized), '').lower()
        if injury_class not in ALLOWED_INJURY_CLASSES_SKELETAL:
            continue
        for days_before in range(1, SKELETAL_POSITIVE_DAYS_BEFORE + 1):
            reference_date = injury_date - timedelta(days=days_before)
            ref_n = pd.Timestamp(reference_date).normalize()
            if season_start <= ref_n <= season_end:
                out.add(ref_n)
    return out


def get_valid_non_injury_dates(df: pd.DataFrame,
                                season_start: pd.Timestamp,
                                season_end: pd.Timestamp,
                                all_injury_dates: Optional[Set[pd.Timestamp]] = None) -> List[pd.Timestamp]:
    """
    Get all valid non-injury (negative) reference dates. Same for both models (target1=0, target2=0).

    all_injury_dates: set of relevant injury dates for this player (muscular/skeletal/unknown only).

    Eligibility for a negative timeline at reference date D:
    - Reference date within season date range
    - Complete 35-day window available BEFORE reference date
    - No muscular, skeletal, or unknown injury in [D-28, D+28]
    - Player selected (played or on bench) at least once in [D-14, D]
    """
    valid_dates = []
    player_injury_dates = all_injury_dates if all_injury_dates is not None else set()
    max_date = df['date'].max()
    min_date = df['date'].min()
    max_reference_date = min(max_date, season_end)
    potential_dates = pd.date_range(min_date, max_reference_date, freq='D')
    df_dates = pd.to_datetime(df['date']).dt.normalize()

    for reference_date in potential_dates:
        reference_date_normalized = pd.Timestamp(reference_date).normalize()
        if not (df_dates == reference_date_normalized).any():
            continue
        if not (season_start <= reference_date <= season_end):
            continue
        start_date = reference_date - timedelta(days=34)
        if start_date < min_date:
            continue
        # No muscular/skeletal/unknown injury in [D-28, D+28]
        window_start_inj = reference_date_normalized - timedelta(days=NEGATIVE_NO_INJURY_DAYS)
        window_end_inj = reference_date_normalized + timedelta(days=NEGATIVE_NO_INJURY_DAYS)
        has_relevant_injury_around = False
        for injury_date in player_injury_dates:
            injury_date_n = pd.Timestamp(injury_date).normalize()
            if window_start_inj <= injury_date_n <= window_end_inj:
                has_relevant_injury_around = True
                break
        if has_relevant_injury_around:
            continue
        # Player selected (played or on bench) at least once in [D-14, D]
        selection_start = reference_date_normalized - timedelta(days=ACTIVITY_DAYS_BEFORE)
        selection_end = reference_date_normalized
        sel_mask = (df_dates >= selection_start) & (df_dates <= selection_end)
        sel_df = df.loc[sel_mask]
        if sel_df.empty:
            continue
        has_selection = False
        for _, row in sel_df.iterrows():
            mp = row.get('matches_played', 0)
            mb = row.get('matches_bench_unused', 0)
            if (pd.notna(mp) and mp > 0) or (pd.notna(mb) and mb > 0):
                has_selection = True
                break
        if not has_selection:
            continue
        
        valid_dates.append(reference_date)
    
    return valid_dates


def get_all_reference_dates_in_range(df: pd.DataFrame,
                                     season_start: pd.Timestamp,
                                     season_end: pd.Timestamp) -> List[pd.Timestamp]:
    """
    Get all reference dates in [season_start, season_end] where a timeline can be built.
    Eligibility: D in range, row for D exists, full 35-day window before D available.
    (No injury/activity filters; used only when building eligible-dates union.)
    """
    out = []
    max_date = df['date'].max()
    min_date = df['date'].min()
    max_reference_date = min(max_date, season_end)
    range_start = max(min_date, season_start)
    if range_start > max_reference_date:
        return out
    potential_dates = pd.date_range(range_start, max_reference_date, freq='D')
    df_dates = pd.to_datetime(df['date']).dt.normalize()
    for reference_date in potential_dates:
        ref_n = pd.Timestamp(reference_date).normalize()
        if not (season_start <= ref_n <= season_end):
            continue
        if not (df_dates == ref_n).any():
            continue
        start_date = ref_n - timedelta(days=34)
        if start_date < min_date:
            continue
        out.append(ref_n)
    return out


def get_eligible_reference_dates_with_targets(
    df: pd.DataFrame,
    season_start: pd.Timestamp,
    season_end: pd.Timestamp,
    muscular_positive_dates: Set[pd.Timestamp],
    skeletal_positive_dates: Set[pd.Timestamp],
    negative_eligible_dates: List[pd.Timestamp],
) -> List[Tuple[pd.Timestamp, int, int]]:
    """
    Build list of (reference_date, target1, target2) for all eligible dates.
    Eligible = muscular positives | skeletal positives | negative-eligible.
    For each date: target1=1 if in muscular_positive_dates else 0, target2=1 if in skeletal_positive_dates else 0.
    """
    all_dates = (muscular_positive_dates | skeletal_positive_dates) | set(negative_eligible_dates)
    date_target_list = []
    for ref_n in sorted(all_dates):
        t1 = 1 if ref_n in muscular_positive_dates else 0
        t2 = 1 if ref_n in skeletal_positive_dates else 0
        date_target_list.append((ref_n, t1, t2))
    return date_target_list


def generate_timelines_for_dates_with_targets(
    player_id: int,
    player_name: str,
    df: pd.DataFrame,
    date_target_list: List[Tuple[pd.Timestamp, int, int]],
) -> List[Dict]:
    """Generate one timeline per (reference_date, target1, target2). Each tuple is (ref_date, t1, t2)."""
    timelines = []
    for reference_date, target1, target2 in date_target_list:
        start_date = reference_date - timedelta(days=34)
        windowed_features = create_windowed_features_vectorized(df, start_date, reference_date)
        if windowed_features is None:
            continue
        ref_mask = (pd.to_datetime(df['date']).dt.normalize() == pd.Timestamp(reference_date).normalize())
        if not ref_mask.any():
            continue
        ref_row = df[ref_mask].iloc[0]
        timeline = build_timeline(
            player_id, player_name, reference_date, ref_row, windowed_features,
            target1=int(target1), target2=int(target2), player_df=df, window_start_date=start_date
        )
        timelines.append(timeline)
    return timelines


def generate_non_injury_timelines_for_dates(player_id: int, player_name: str, df: pd.DataFrame,
                                            reference_dates: List[pd.Timestamp]) -> List[Dict]:
    """Generate non-injury (negative) timelines for specific reference dates (target1=0, target2=0)."""
    timelines = []
    for reference_date in reference_dates:
        start_date = reference_date - timedelta(days=34)
        windowed_features = create_windowed_features_vectorized(df, start_date, reference_date)
        if windowed_features is None:
            continue
        ref_mask = (pd.to_datetime(df['date']).dt.normalize() == pd.Timestamp(reference_date).normalize())
        if not ref_mask.any():
            continue
        ref_row = df[ref_mask].iloc[0]
        timeline = build_timeline(player_id, player_name, reference_date, ref_row, windowed_features,
                                 target1=0, target2=0, player_df=df, window_start_date=start_date)
        timelines.append(timeline)
    return timelines

def build_timeline(player_id: int, player_name: str, reference_date: pd.Timestamp,
                  ref_row: pd.Series, windowed_features: Dict,
                  target1: Optional[int] = None,
                  target2: Optional[int] = None,
                  player_df: Optional[pd.DataFrame] = None,
                  window_start_date: Optional[pd.Timestamp] = None) -> Dict:
    """Build a complete timeline with normalized cumulative features (dual target: target1=muscular, target2=skeletal).

    Args:
        player_id: Player ID
        player_name: Player name
        reference_date: Reference date for the timeline
        ref_row: Reference row from daily features
        windowed_features: Windowed features dictionary
        target1: 1=muscular positive, 0=negative (or None to omit)
        target2: 1=skeletal positive, 0=negative (or None to omit)
        player_df: Optional full player daily features dataframe
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

def filter_timelines_for_model(timelines_df: pd.DataFrame, target_column: str = 'target1') -> pd.DataFrame:
    """
    Filter timelines for one model: target_column='target1' (muscular) or 'target2' (skeletal).
    Returns the same DataFrame (all rows valid; column target1/target2 used as label).
    """
    if target_column not in ('target1', 'target2'):
        raise ValueError("target_column must be 'target1' (muscular) or 'target2' (skeletal).")
    if target_column not in timelines_df.columns:
        raise ValueError(f"DataFrame must contain '{target_column}' column")
    filtered_df = timelines_df.copy()
    positives = int(filtered_df[target_column].sum())
    negatives = int((filtered_df[target_column] == 0).sum())
    label = "Muscular (target1)" if target_column == 'target1' else "Skeletal (target2)"
    print(f"\n📊 {label}:")
    print(f"   Total timelines: {len(filtered_df):,}")
    print(f"   Positives ({target_column}=1): {positives:,}")
    print(f"   Negatives ({target_column}=0): {negatives:,}")
    if len(filtered_df) > 0:
        print(f"   Target ratio: {positives / len(filtered_df) * 100:.2f}%")
    return filtered_df

# ===== SEASON PROCESSING (PL filtering done in post-processing) =====

def process_season(season_start_year: int, daily_features_dir: str, 
                  all_injury_dates_by_player: Dict[int, Set[pd.Timestamp]],
                  relevant_injury_dates_by_player: Dict[int, Set[pd.Timestamp]],
                  injury_class_map: Dict[Tuple[int, pd.Timestamp], str],
                  player_names_map: Dict[int, str],
                  player_ids: List[int],
                  player_pl_periods: Dict[int, List[Tuple[pd.Timestamp, pd.Timestamp]]],
                  output_dir: str,
                  max_players: Optional[int] = None) -> Tuple[Optional[str], int, int]:
    """Process a single season: dual target (target1=muscular, target2=skeletal). Same eligible-date rules for train and test."""
    season_start, season_end = get_season_date_range(season_start_year)
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
    if max_players is not None:
        season_player_ids = player_ids[:max_players]
    else:
        season_player_ids = player_ids

    # ===== PASS 1: Collect (player_id, date_target_list) with date_target_list = [(ref_date, target1, target2), ...] =====
    # Same rules for train and test: eligible dates = muscular positives | skeletal positives | negative-eligible (28d no-injury + activity)
    player_dates_with_targets: List[Tuple[int, List[Tuple[pd.Timestamp, int, int]]]] = []
    processed_players = 0
    skipped_players_info = []
    print(f"\n📊 PASS 1: Collecting eligible reference dates (muscular 14d, skeletal 21d, negatives 28d + activity)")

    for player_id in tqdm(season_player_ids, desc=f"Pass 1: {season_start_year}_{season_start_year+1}", unit="player"):
        try:
            df = pd.read_csv(f'{daily_features_dir}/player_{player_id}_daily_features.csv')
            df['date'] = pd.to_datetime(df['date'])
            buffer_start = season_start - timedelta(days=34)
            season_mask = (df['date'] >= buffer_start) & (df['date'] <= season_end)
            df_season = df[season_mask].copy()
            if len(df_season) == 0:
                file_min = df['date'].min() if len(df) > 0 else None
                file_max = df['date'].max() if len(df) > 0 else None
                skipped_players_info.append({
                    'player_id': player_id,
                    'reason': f'No data in season range (buffer: {buffer_start.date()} to {season_end.date()})',
                    'file_date_range': f'{file_min.date()} to {file_max.date()}' if file_min is not None else 'N/A',
                    'total_rows_in_file': len(df)
                })
                continue

            muscular_positive_dates = get_muscular_positive_reference_dates(
                player_id, df_season, injury_class_map, season_start, season_end
            )
            skeletal_positive_dates = get_skeletal_positive_reference_dates(
                player_id, df_season, injury_class_map, season_start, season_end
            )
            player_relevant_injury_dates = relevant_injury_dates_by_player.get(player_id, set())
            negative_eligible_dates = get_valid_non_injury_dates(
                df_season, season_start=season_start, season_end=season_end,
                all_injury_dates=player_relevant_injury_dates
            )
            date_target_list = get_eligible_reference_dates_with_targets(
                df_season, season_start, season_end,
                muscular_positive_dates, skeletal_positive_dates, negative_eligible_dates
            )
            if not date_target_list:
                skipped_players_info.append({
                    'player_id': player_id,
                    'reason': 'No eligible reference dates (no muscular/skeletal positives and no valid negatives)',
                    'muscular_pos': len(muscular_positive_dates),
                    'skeletal_pos': len(skeletal_positive_dates),
                    'negative_eligible': len(negative_eligible_dates)
                })
                del df, df_season
                continue
            player_dates_with_targets.append((player_id, date_target_list))
            processed_players += 1
            del df, df_season
        except Exception as e:
            skipped_players_info.append({
                'player_id': player_id,
                'reason': f'Error: {str(e)}',
                'error_type': type(e).__name__
            })
            print(f"\n❌ Error processing player {player_id} for season {season_start_year}: {e}")
            continue

    total_ref_days = sum(len(dtl) for _, dtl in player_dates_with_targets)
    n_t1 = sum(1 for _, dtl in player_dates_with_targets for _, t1, t2 in dtl if t1 == 1)
    n_t2 = sum(1 for _, dtl in player_dates_with_targets for _, t1, t2 in dtl if t2 == 1)
    print(f"\n✅ PASS 1 Complete for {season_start_year}_{season_start_year+1}:")
    print(f"   Players with timelines: {len(player_dates_with_targets)}")
    print(f"   Total reference days: {total_ref_days:,} (target1=1: {n_t1:,}, target2=1: {n_t2:,})")

    if skipped_players_info:
        print(f"\n⚠️  SKIPPED PLAYERS ({len(skipped_players_info)}):")
        for info in skipped_players_info:
            print(f"   Player {info['player_id']}: {info['reason']}")
    else:
        print(f"   ✅ All {len(season_player_ids)} players processed successfully")

    # ===== PASS 2: Generate timelines for all (player_id, date_target_list) =====
    print(f"\n📊 PASS 2: Generating timelines for all eligible reference days (dual target)")
    all_timelines = []
    for player_id, date_target_list in tqdm(player_dates_with_targets, desc=f"Pass 2: {season_start_year}_{season_start_year+1}", unit="player"):
        try:
            df = pd.read_csv(f'{daily_features_dir}/player_{player_id}_daily_features.csv')
            df['date'] = pd.to_datetime(df['date'])
            buffer_start = season_start - timedelta(days=34)
            season_mask = (df['date'] >= buffer_start) & (df['date'] <= season_end)
            df_season = df[season_mask].copy()
            if len(df_season) == 0:
                continue
            player_name = get_player_name_from_df(df_season, player_id=player_id, player_names_map=player_names_map)
            timelines = generate_timelines_for_dates_with_targets(player_id, player_name, df_season, date_target_list)
            all_timelines.extend(timelines)
            del df, df_season
        except Exception as e:
            print(f"\n❌ Error generating timelines for player {player_id}: {e}")
            continue

    # Combine and save
    print(f"\n📊 FINALIZING DATASET FOR {season_start_year}_{season_start_year+1}")
    final_timelines = all_timelines
    injury_count = sum(1 for t in final_timelines if t.get('target1') == 1)
    non_injury_count = len(final_timelines) - injury_count
    random.shuffle(final_timelines)
    
    # Apply PL filter (post-processing)
    if player_pl_periods and final_timelines:
        timelines_df = pd.DataFrame(final_timelines)
        timelines_df = filter_timelines_pl_only(timelines_df, player_pl_periods)
        final_timelines = timelines_df.to_dict('records')
        del timelines_df
        print(f"   Applied PL filter: {len(final_timelines):,} timelines after filter")
    
    total_count = len(final_timelines)
    target1_count = sum(1 for t in final_timelines if t.get('target1') == 1)
    target2_count = sum(1 for t in final_timelines if t.get('target2') == 1)
    activity_flag_count = sum(1 for t in final_timelines if t.get('has_minimum_activity') == 1)

    print(f"\n📈 SEASON {season_start_year}_{season_start_year+1} DATASET (PL-only, dual target):")
    print(f"   Total timelines: {total_count:,}")
    print(f"   target1=1 (muscular): {target1_count:,}")
    print(f"   target2=1 (skeletal): {target2_count:,}")
    if total_count > 0:
        print(f"   Timelines with minimum activity: {activity_flag_count:,} ({activity_flag_count/total_count*100:.1f}%)")
        print(f"   target1 ratio: {target1_count/total_count:.1%}  target2 ratio: {target2_count/total_count:.1%}")
    else:
        print(f"   Timelines with minimum activity: {activity_flag_count:,} (N/A)")
    
    if final_timelines:
        print(f"\n💾 Saving timelines to CSV...")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        shape = save_timelines_to_csv_chunked(final_timelines, output_file)
        print(f"✅ Saved to: {output_file}")
        print(f"📊 Shape: {shape}")
    else:
        print("⚠️  No timelines generated for this season")
        output_file = None
    
    season_time = datetime.now() - season_start_time
    print(f"⏱️  Season processing time: {season_time}")
    
    return output_file, target1_count, non_injury_count

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
    print(f"🔍 Dual target: target1=muscular (D-14..D-1), target2=skeletal (D-21..D-1); negatives 28d no-injury + activity")
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
    # For non-injury validation: only muscular/skeletal/unknown disqualify; other/no_injury do not
    relevant_injury_dates_by_player = defaultdict(set)
    for (pid, date), cls in injury_class_map.items():
        c = (cls or '').strip().lower()
        if c in INJURY_CLASSES_DISQUALIFYING_NON_INJURY:
            relevant_injury_dates_by_player[pid].add(date)
    relevant_injury_dates_by_player = dict(relevant_injury_dates_by_player)
    print(f"   Relevant injury dates (muscular/skeletal/unknown) for non-injury check: {sum(len(s) for s in relevant_injury_dates_by_player.values())} total")
    
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
    total_non_injuries = 0
    
    for season_start_year in seasons_to_process:
        output_file, target1_count, non_injury_count = process_season(
            season_start_year,
            daily_features_dir,
            all_injury_dates_by_player,
            relevant_injury_dates_by_player,
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
            total_non_injuries += non_injury_count
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*80}")
    total_timelines = total_target1 + total_non_injuries
    print(f"   Processed {len(output_files)} seasons")
    print(f"   Total timelines: {total_timelines:,}")
    print(f"   Positives (target1=1): {total_target1:,}")
    print(f"   Negatives (target1=0): {total_non_injuries:,}")
    if total_timelines > 0:
        print(f"   Final target1 ratio: {total_target1 / total_timelines:.1%}")
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
