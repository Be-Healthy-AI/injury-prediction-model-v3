#!/usr/bin/env python3
"""
Enhanced 35-Day Timeline Generator V4 - Season-by-Season Version
Generates one timeline file per season from daily features

Key features:
- Generates timelines season by season (YYYY_YYYY+1 format)
- Filters injury timelines by injury_class (muscular only)
- Natural target ratios (all available positives and negatives, no forced balancing)
- Each injury generates 5 timelines (D-1, D-2, D-3, D-4, D-5)
- Non-injury validation checks for ANY injury (any class) in 35 days after reference
- Skips players whose daily features file is missing (e.g. goalkeepers)
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
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import warnings
from tqdm import tqdm
try:
    import openpyxl  # For reading Excel files
except ImportError:
    openpyxl = None
warnings.filterwarnings('ignore')

# Calculate paths
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent
ROOT_DIR = PRODUCTION_ROOT.parent

# Configuration
WINDOW_SIZE = 35  # 5 weeks
# Use natural target ratios (no forced balancing - all available positives and negatives)
# TARGET_RATIO_TRAIN = 0.08  # Disabled - using natural ratio
# TARGET_RATIO_VAL = 0.08  # Disabled - using natural ratio
USE_NATURAL_RATIO = True  # Flag to use all available positives and negatives

# Season configuration
SEASON_START_MONTH = 7  # July
SEASON_START_DAY = 1

# Allowed injury classes for injury timelines (muscular only)
ALLOWED_INJURY_CLASSES = {'muscular'}

def get_season_date_range(season_start_year: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Get start and end dates for a season"""
    start = pd.Timestamp(f'{season_start_year}-07-01')
    end = pd.Timestamp(f'{season_start_year + 1}-06-30')
    return start, end

def get_all_seasons_from_daily_features(daily_features_dir: str) -> List[int]:
    """Scan daily features to determine available seasons"""
    print(f"\n[SCAN] Scanning daily features to determine available seasons...")
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
    """Get features that remain static (not windowed)"""
    return [
        'player_id', 'reference_date', 'player_name',  # Technical features
        'position', 'nationality1', 'nationality2', 'height_cm', 'dominant_foot',
        'previous_club', 'previous_club_country', 'current_club', 'current_club_country',
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
        'national_team_frequency', 'senior_national_team', 'competition_intensity',
        'competition_level', 'competition_diversity', 'international_competitions',
        'cup_competitions', 'competition_frequency', 'competition_experience',
        'competition_pressure', 'teams_today', 'cum_teams', 'seasons_count',
        # Career club features
        'club_cum_goals', 'club_cum_assists', 'club_cum_minutes',
        'club_cum_matches_played', 'club_cum_yellow_cards', 'club_cum_red_cards',
        # Interaction features
        'age_x_career_matches', 'age_x_career_goals', 'seniority_x_goals_per_match'
    ]

def get_windowed_features() -> List[str]:
    """Get features that will be windowed (35 days)"""
    return [
        # Daily performance metrics
        'matches_played', 'minutes_played_numeric', 'goals_numeric', 'assists_numeric',
        'yellow_cards_numeric', 'red_cards_numeric', 'matches_bench_unused', 
        'matches_not_selected', 'matches_injured',
        # Recent patterns
        'days_since_last_match', 'last_match_position', 'position_match_default',
        'disciplinary_action', 'goals_per_match', 'assists_per_match', 'minutes_per_match',
        # Recent injury indicators
        'days_since_last_injury', 'days_since_last_injury_ended', 'avg_injury_duration', 'injury_frequency',
        'physio_injury_ratio',
        # Recent injury indicators by class
        'days_since_last_muscular', 'days_since_last_skeletal', 'days_since_last_unknown', 'days_since_last_other',
        # Recent injury indicators by body part
        'days_since_last_lower_leg', 'days_since_last_knee', 'days_since_last_upper_leg', 'days_since_last_hip',
        'days_since_last_upper_body', 'days_since_last_head', 'days_since_last_illness',
        # Recent injury indicators by severity
        'days_since_last_mild', 'days_since_last_moderate', 'days_since_last_severe', 'days_since_last_critical',
        # Recent competition context
        'competition_importance', 'month', 'teams_this_season',
        'teams_season_today', 'season_team_diversity',
        # Recent national team activity
        'days_since_last_national_match', 'national_team_this_season',
        'national_team_intensity',
        # Recent club performance
        'club_goals_per_match', 'club_assists_per_match', 'club_minutes_per_match',
        'club_seniority_x_goals_per_match',
        # Match location
        'home_matches', 'away_matches',
        # Team result features
        'team_win', 'team_draw', 'team_loss', 'team_points',
        'cum_team_wins', 'cum_team_draws', 'cum_team_losses',
        'team_win_rate', 'cum_team_points', 'team_points_rolling5', 'team_mood_score',
        # Substitution patterns
        'substitution_on_count', 'substitution_off_count', 'late_substitution_on_count',
        'early_substitution_off_count', 'impact_substitution_count', 'tactical_substitution_count',
        'substitution_minutes_played', 'substitution_efficiency', 'substitution_mood_indicator',
        'consecutive_substitutions'
    ]

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

def load_injuries_data(data_date: str = None) -> Dict[Tuple[int, pd.Timestamp], str]:
    """
    Load injuries data file and create a mapping of (player_id, injury_date) -> injury_class
    
    Args:
        data_date: Date folder (YYYYMMDD format) to load from, defaults to latest
    
    Returns:
        Dictionary mapping (player_id, fromDate) to injury_class
    """
    print("[LOAD] Loading injuries data file...")
    
    # Try production raw_data first
    injuries_paths = []
    if data_date:
        injuries_paths.append(PRODUCTION_ROOT / "raw_data" / "england" / data_date / "injuries_data.csv")
    else:
        # Find latest raw_data folder
        raw_data_dir = PRODUCTION_ROOT / "raw_data" / "england"
        if raw_data_dir.exists():
            date_folders = sorted([d for d in raw_data_dir.iterdir() if d.is_dir() and d.name.isdigit() and len(d.name) == 8], reverse=True)
            if date_folders:
                injuries_paths.append(date_folders[0] / "injuries_data.csv")
    
    # Fallback to original paths
    injuries_paths.extend([
        ROOT_DIR / 'data_exports' / 'transfermarkt' / 'england' / '20251205' / 'injuries_data.csv',
        ROOT_DIR / 'data_exports' / 'transfermarkt' / 'england' / '20251213' / 'injuries_data.csv',
        ROOT_DIR / 'data_exports' / 'transfermarkt' / 'england' / '20251217' / 'injuries_data.csv',
        Path('data_exports/transfermarkt/england/20251205/injuries_data.csv'),
        Path('../data_exports/transfermarkt/england/20251205/injuries_data.csv'),
    ])
    
    injuries_path = None
    for path in injuries_paths:
        if isinstance(path, str):
            path = Path(path)
        if path.exists():
            injuries_path = path
            break
    
    if injuries_path is None:
        raise FileNotFoundError(f"Injuries data file not found. Tried: {injuries_paths}")
    
    print(f"   Loading from: {injuries_path}")
    
    # Load injuries data
    if str(injuries_path).endswith('.csv'):
        injuries_df = pd.read_csv(injuries_path, sep=';', encoding='utf-8')
    else:
        injuries_df = pd.read_excel(injuries_path, engine='openpyxl')
    
    # Convert dates - try DD/MM/YYYY format first (for CSV files), then auto-detect
    if 'fromDate' in injuries_df.columns:
        # Try DD/MM/YYYY format first (common in European CSV exports)
        injuries_df['fromDate_parsed'] = pd.to_datetime(injuries_df['fromDate'], format='%d/%m/%Y', errors='coerce')
        valid_count = injuries_df['fromDate_parsed'].notna().sum()
        
        # If that didn't work well, try auto-detect
        if valid_count < len(injuries_df) * 0.9:  # If less than 90% parsed successfully
            injuries_df['fromDate_parsed2'] = pd.to_datetime(injuries_df['fromDate'], errors='coerce')
            valid_count2 = injuries_df['fromDate_parsed2'].notna().sum()
            if valid_count2 > valid_count:
                injuries_df['fromDate'] = injuries_df['fromDate_parsed2']
            else:
                injuries_df['fromDate'] = injuries_df['fromDate_parsed']
        else:
            injuries_df['fromDate'] = injuries_df['fromDate_parsed']
        
        # Clean up temporary columns
        injuries_df = injuries_df.drop(columns=[col for col in injuries_df.columns if 'fromDate_parsed' in col])
    
    # Derive injury_class if it doesn't exist
    if 'injury_class' not in injuries_df.columns:
        print("   [WARNING] injury_class column not found, deriving from injury_type and no_physio_injury")
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

def load_all_injury_dates(data_date: str = None) -> Dict[int, Set[pd.Timestamp]]:
    """
    Load all injury dates (any class) for each player
    Used for non-injury validation (checking if ANY injury occurs in 35 days after)
    
    Args:
        data_date: Date folder (YYYYMMDD format) to load from, defaults to latest
    
    Returns:
        Dictionary mapping player_id to set of injury dates
    """
    print("[LOAD] Loading all injury dates (any class) for non-injury validation...")
    
    # Try production raw_data first
    injuries_paths = []
    if data_date:
        injuries_paths.append(PRODUCTION_ROOT / "raw_data" / "england" / data_date / "injuries_data.csv")
    else:
        # Find latest raw_data folder
        raw_data_dir = PRODUCTION_ROOT / "raw_data" / "england"
        if raw_data_dir.exists():
            date_folders = sorted([d for d in raw_data_dir.iterdir() if d.is_dir() and d.name.isdigit() and len(d.name) == 8], reverse=True)
            if date_folders:
                injuries_paths.append(date_folders[0] / "injuries_data.csv")
    
    # Fallback to original paths
    injuries_paths.extend([
        ROOT_DIR / 'data_exports' / 'transfermarkt' / 'england' / '20251205' / 'injuries_data.csv',
        ROOT_DIR / 'data_exports' / 'transfermarkt' / 'england' / '20251213' / 'injuries_data.csv',
        ROOT_DIR / 'data_exports' / 'transfermarkt' / 'england' / '20251217' / 'injuries_data.csv',
        Path('data_exports/transfermarkt/england/20251205/injuries_data.csv'),
        Path('../data_exports/transfermarkt/england/20251205/injuries_data.csv'),
    ])
    
    injuries_path = None
    for path in injuries_paths:
        if isinstance(path, str):
            path = Path(path)
        if path.exists():
            injuries_path = path
            break
    
    if injuries_path is None:
        raise FileNotFoundError(f"Injuries data file not found. Tried: {injuries_paths}")
    
    # Load injuries data
    if str(injuries_path).endswith('.csv'):
        injuries_df = pd.read_csv(injuries_path, sep=';', encoding='utf-8')
    else:
        injuries_df = pd.read_excel(injuries_path, engine='openpyxl')
    
    # Convert dates - try DD/MM/YYYY format first (for CSV files), then auto-detect
    if 'fromDate' in injuries_df.columns:
        # Try DD/MM/YYYY format first (common in European CSV exports)
        injuries_df['fromDate_parsed'] = pd.to_datetime(injuries_df['fromDate'], format='%d/%m/%Y', errors='coerce')
        valid_count = injuries_df['fromDate_parsed'].notna().sum()
        
        # If that didn't work well, try auto-detect
        if valid_count < len(injuries_df) * 0.9:  # If less than 90% parsed successfully
            injuries_df['fromDate_parsed2'] = pd.to_datetime(injuries_df['fromDate'], errors='coerce')
            valid_count2 = injuries_df['fromDate_parsed2'].notna().sum()
            if valid_count2 > valid_count:
                injuries_df['fromDate'] = injuries_df['fromDate_parsed2']
            else:
                injuries_df['fromDate'] = injuries_df['fromDate_parsed']
        else:
            injuries_df['fromDate'] = injuries_df['fromDate_parsed']
        
        # Clean up temporary columns
        injuries_df = injuries_df.drop(columns=[col for col in injuries_df.columns if 'fromDate_parsed' in col])
    
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

def create_windowed_features_vectorized(df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> Dict:
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
                              'teams_season_today', 'season_team_diversity', 'national_team_this_season',
                              'national_team_intensity', 'club_goals_per_match', 'club_assists_per_match',
                              'club_minutes_per_match', 'club_seniority_x_goals_per_match', 'team_win_rate']
        
        # Season-specific features that should be excluded from week 5 (too sensitive to temporal patterns)
        season_specific_features = ['teams_this_season', 'national_team_this_season', 'season_team_diversity']
        
        sum_features = ['matches_played', 'minutes_played_numeric', 'goals_numeric', 'assists_numeric',
                       'yellow_cards_numeric', 'red_cards_numeric', 'matches_bench_unused',
                       'matches_not_selected', 'matches_injured', 'substitution_on_count',
                       'substitution_off_count', 'late_substitution_on_count', 'early_substitution_off_count',
                       'impact_substitution_count', 'tactical_substitution_count', 'substitution_minutes_played',
                       'consecutive_substitutions']
        
        mean_features = ['goals_per_match', 'assists_per_match', 'minutes_per_match',
                        'substitution_efficiency', 'substitution_mood_indicator',
                        'avg_injury_duration', 'injury_frequency', 'physio_injury_ratio']
        
        min_features = ['days_since_last_match', 'days_since_last_injury', 'days_since_last_national_match']
        
        # Apply aggregations
        for feature in get_windowed_features():
            # Skip season-specific features in week 5 (weeks are 0-indexed, so week 4 = week 5)
            if feature in season_specific_features and week == 4:
                continue  # Skip week 5 for season-specific features
            
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

def generate_injury_timelines_enhanced(player_id: int, player_name: str, df: pd.DataFrame,
                                       injury_class_map: Dict[Tuple[int, pd.Timestamp], str]) -> List[Dict]:
    """
    Generate injury timelines - ONLY for injuries with allowed injury_class
    Each allowed injury gets 5 timelines (D-1, D-2, D-3, D-4, D-5)
    """
    timelines = []
    
    # Vectorized injury start detection
    injury_starts = df[df['cum_inj_starts'] > df['cum_inj_starts'].shift(1)]
    
    for _, injury_row in injury_starts.iterrows():
        injury_date = injury_row['date']
        injury_date_normalized = pd.Timestamp(injury_date).normalize()
        
        # Check if this injury has an allowed injury_class
        injury_class = injury_class_map.get((player_id, injury_date_normalized), '').lower()
        if injury_class not in ALLOWED_INJURY_CLASSES:
            continue  # Skip injuries that are not muscular, skeletal, or unknown
        
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
            
            # Build timeline (pass df for years_active calculation)
            timeline = build_timeline(player_id, player_name, reference_date, ref_row, windowed_features, target=1, player_df=df)
            timelines.append(timeline)
    
    return timelines

def get_all_valid_dates_for_timelines(df: pd.DataFrame,
                                      season_start: pd.Timestamp,
                                      season_end: pd.Timestamp,
                                      min_date: Optional[pd.Timestamp] = None,
                                      max_date: Optional[pd.Timestamp] = None) -> List[pd.Timestamp]:
    """
    Get ALL valid reference dates for timeline generation (for production deployment)
    
    Eligibility rules:
    - Reference date must be within season date range
    - Complete 35-day window must be available before reference date
    - Complete 35-day window must be available after reference date (for validation)
    - ALL dates are included regardless of injury status
    """
    valid_dates = []
    
    # Get max date from data (use different variable name to avoid shadowing max_date parameter)
    data_max_date = df['date'].max()
    min_data_date = df['date'].min()
    
    # Calculate max reference date (for production, we can use all available dates)
    # We only need 35 days before, not after
    max_reference_date = min(data_max_date, season_end)
    
    # Apply max_date cap if provided (from parameter)
    if max_date is not None:
        max_reference_date = min(max_reference_date, max_date)
    
    # Start from season start or min_data_date + 35 days, whichever is later
    start_date = max(season_start, min_data_date + timedelta(days=34))
    if min_date:
        start_date = max(start_date, min_date)
    
    # Create date range for potential reference dates
    potential_dates = pd.date_range(start_date, max_reference_date, freq='D')
    
    # Process all valid dates
    for reference_date in potential_dates:
        # Check if reference date exists in data
        if reference_date not in df['date'].values:
            continue
        
        # Season filtering
        if not (season_start <= reference_date <= season_end):
            continue
        
        # Apply max_date filter (from parameter)
        if max_date is not None and reference_date > max_date:
            continue
        
        # Check if we can create a complete 35-day window before reference
        window_start = reference_date - timedelta(days=34)
        if window_start < min_data_date:
            continue
        
        # For production: we only need the 35-day window before reference date
        # The after window is optional (only needed for training/validation)
        # So we don't check for future_end > max_date in production mode
        
        # ALL dates are valid - no injury filtering
        valid_dates.append(reference_date)
    
    return valid_dates

def generate_timelines_for_dates(player_id: int, player_name: str, df: pd.DataFrame, 
                                  reference_dates_with_targets: List[Tuple[pd.Timestamp, int]]) -> List[Dict]:
    """Generate timelines for specific reference dates with target values
    
    Args:
        reference_dates_with_targets: List of tuples (reference_date, target_value)
    """
    timelines = []
    
    for reference_date, target in reference_dates_with_targets:
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
        
        # Build timeline with target value
        timeline = build_timeline(player_id, player_name, reference_date, ref_row, windowed_features, target=target, player_df=df)
        timelines.append(timeline)
    
    return timelines

def build_timeline(player_id: int, player_name: str, reference_date: pd.Timestamp, 
                  ref_row: pd.Series, windowed_features: Dict, target: int = None,
                  player_df: Optional[pd.DataFrame] = None) -> Dict:
    """Build a complete timeline with normalized cumulative features
    
    Args:
        player_id: Player ID
        player_name: Player name
        reference_date: Reference date for the timeline
        ref_row: Reference row from daily features
        windowed_features: Windowed features dictionary
        target: Optional target value (for training/backtesting). If None, target column is not included.
        player_df: Optional full player daily features dataframe to calculate career start
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
    
    # ===== NORMALIZE ALL CUMULATIVE FEATURES BY YEARS_ACTIVE =====
    # This addresses temporal drift by converting absolute cumulative values to rates
    
    # Match performance cumulative features
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
        ('cum_disciplinary_actions', 'disciplinary_actions_per_year'),
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
    
    # Career totals (also normalize these)
    career_goals = ref_row.get('career_goals', 0) if ref_row.get('career_goals') is not None else 0
    if career_goals is not None and pd.notna(career_goals) and career_goals > 0:
        normalized_features['career_goals_per_year'] = career_goals / years_active
    else:
        normalized_features['career_goals_per_year'] = 0.0  # Always set, even if 0
    
    career_assists = ref_row.get('career_assists', 0) if ref_row.get('career_assists') is not None else 0
    if career_assists is not None and pd.notna(career_assists) and career_assists > 0:
        normalized_features['career_assists_per_year'] = career_assists / years_active
    else:
        normalized_features['career_assists_per_year'] = 0.0  # Always set, even if 0
    
    career_minutes = ref_row.get('career_minutes', 0) if ref_row.get('career_minutes') is not None else 0
    if career_minutes is not None and pd.notna(career_minutes) and career_minutes > 0:
        normalized_features['career_minutes_per_year'] = career_minutes / years_active
    else:
        normalized_features['career_minutes_per_year'] = 0.0  # Always set, even if 0
    
    # Normalize cumulative career features by seasons - always set
    career_matches_val = ref_row.get('career_matches', 0) if ref_row.get('career_matches') is not None else 0
    if seasons_count and seasons_count > 0 and career_matches_val > 0:
        normalized_features['career_matches_per_season'] = career_matches_val / seasons_count
    else:
        normalized_features['career_matches_per_season'] = 0.0
        
    cum_comp = ref_row.get('cum_competitions', 0) if ref_row.get('cum_competitions') is not None else 0
    if seasons_count and seasons_count > 0 and cum_comp > 0:
        normalized_features['competitions_per_season'] = cum_comp / seasons_count
    else:
        normalized_features['competitions_per_season'] = 0.0
        
    cum_teams = ref_row.get('cum_teams', 0) if ref_row.get('cum_teams') is not None else 0
    if seasons_count and seasons_count > 0 and cum_teams > 0:
        normalized_features['teams_per_season'] = cum_teams / seasons_count
    else:
        normalized_features['teams_per_season'] = 0.0
        
    cup_comp = ref_row.get('cup_competitions', 0) if ref_row.get('cup_competitions') is not None else 0
    if seasons_count and seasons_count > 0 and cup_comp > 0:
        normalized_features['cup_competitions_per_season'] = cup_comp / seasons_count
    else:
        normalized_features['cup_competitions_per_season'] = 0.0
        
    int_comp = ref_row.get('international_competitions', 0) if ref_row.get('international_competitions') is not None else 0
    if seasons_count and seasons_count > 0 and int_comp > 0:
        normalized_features['international_competitions_per_season'] = int_comp / seasons_count
    else:
        normalized_features['international_competitions_per_season'] = 0.0
    
    # Normalize by career matches (rates) - always set
    if career_matches and career_matches > 0:
        # Goals per match (already exists, but ensure it's normalized)
        if career_goals > 0:
            normalized_features['goals_per_career_match'] = career_goals / career_matches
        else:
            normalized_features['goals_per_career_match'] = 0.0
        
        # Assists per match
        if career_assists > 0:
            normalized_features['assists_per_career_match'] = career_assists / career_matches
        else:
            normalized_features['assists_per_career_match'] = 0.0
        
        # Minutes per match
        if career_minutes > 0:
            normalized_features['minutes_per_career_match'] = career_minutes / career_matches
        else:
            normalized_features['minutes_per_career_match'] = 0.0
        
        # Injury frequency (injuries per match)
        if cum_inj_starts > 0:
            normalized_features['injuries_per_career_match'] = cum_inj_starts / career_matches
        else:
            normalized_features['injuries_per_career_match'] = 0.0
        
        # Normalize cumulative matches bench by career matches
        cum_bench = ref_row.get('cum_matches_bench', 0) if ref_row.get('cum_matches_bench') is not None else 0
        if cum_bench > 0:
            normalized_features['bench_rate'] = cum_bench / career_matches
        else:
            normalized_features['bench_rate'] = 0.0
    else:
        normalized_features['goals_per_career_match'] = 0.0
        normalized_features['assists_per_career_match'] = 0.0
        normalized_features['minutes_per_career_match'] = 0.0
        normalized_features['injuries_per_career_match'] = 0.0
        normalized_features['bench_rate'] = 0.0
    
    # Normalize team performance features (use win rate instead of cumulative)
    # Team win rate is already in features, but normalize cumulative team features from windowed features
    cum_wins = windowed_features.get('cum_team_wins_week_5', 0) or 0
    cum_losses = windowed_features.get('cum_team_losses_week_5', 0) or 0
    cum_draws = windowed_features.get('cum_team_draws_week_5', 0) or 0
    total_games = cum_wins + cum_losses + cum_draws
    if total_games > 0:
        normalized_features['team_win_rate_normalized'] = cum_wins / total_games
        normalized_features['team_loss_rate_normalized'] = cum_losses / total_games
        normalized_features['team_draw_rate_normalized'] = cum_draws / total_games
    else:
        normalized_features['team_win_rate_normalized'] = 0.0
        normalized_features['team_loss_rate_normalized'] = 0.0
        normalized_features['team_draw_rate_normalized'] = 0.0
    
    # ===== ENHANCED FEATURES =====
    
    # 1. RELATIVE FEATURES: Ratios and percentages
    # Matches this season vs last season - always set
    matches_this_season = ref_row.get('matches', 0) if ref_row.get('matches') is not None else 0
    matches_last_season = ref_row.get('teams_last_season', 0) if ref_row.get('teams_last_season') is not None else 0
    if matches_last_season > 0:
        normalized_features['matches_this_season_to_last_ratio'] = matches_this_season / matches_last_season
    else:
        normalized_features['matches_this_season_to_last_ratio'] = 0.0
    
    # Career matches ratio (current vs average) - always set
    if career_matches > 0 and seasons_count > 0:
        avg_matches_per_season = career_matches / seasons_count
        if avg_matches_per_season > 0:
            normalized_features['matches_to_avg_season_ratio'] = matches_this_season / avg_matches_per_season
        else:
            normalized_features['matches_to_avg_season_ratio'] = 0.0
    else:
        normalized_features['matches_to_avg_season_ratio'] = 0.0
    
    # Goals this season vs career average - always set
    if career_goals > 0 and career_matches > 0:
        career_goals_per_match = career_goals / career_matches
        goals_this_season = ref_row.get('goals_numeric', 0) if ref_row.get('goals_numeric') is not None else 0
        matches_this_season_for_goals = ref_row.get('matches', 0) if ref_row.get('matches') is not None else 0
        if matches_this_season_for_goals > 0 and career_goals_per_match > 0:
            goals_per_match_this_season = goals_this_season / matches_this_season_for_goals
            normalized_features['goals_per_match_to_career_ratio'] = goals_per_match_this_season / career_goals_per_match
        else:
            normalized_features['goals_per_match_to_career_ratio'] = 0.0
    else:
        normalized_features['goals_per_match_to_career_ratio'] = 0.0
    
    # Assists this season vs career average - always set
    if career_assists > 0 and career_matches > 0:
        career_assists_per_match = career_assists / career_matches
        assists_this_season = ref_row.get('assists_numeric', 0) if ref_row.get('assists_numeric') is not None else 0
        matches_this_season_for_assists = ref_row.get('matches', 0) if ref_row.get('matches') is not None else 0
        if matches_this_season_for_assists > 0 and career_assists_per_match > 0:
            assists_per_match_this_season = assists_this_season / matches_this_season_for_assists
            normalized_features['assists_per_match_to_career_ratio'] = assists_per_match_this_season / career_assists_per_match
        else:
            normalized_features['assists_per_match_to_career_ratio'] = 0.0
    else:
        normalized_features['assists_per_match_to_career_ratio'] = 0.0
    
    # Minutes this season vs career average - always set
    if career_minutes > 0 and career_matches > 0:
        career_minutes_per_match = career_minutes / career_matches
        minutes_this_season = ref_row.get('minutes_played_numeric', 0) if ref_row.get('minutes_played_numeric') is not None else 0
        matches_this_season_for_minutes = ref_row.get('matches', 0) if ref_row.get('matches') is not None else 0
        if matches_this_season_for_minutes > 0 and career_minutes_per_match > 0:
            minutes_per_match_this_season = minutes_this_season / matches_this_season_for_minutes
            normalized_features['minutes_per_match_to_career_ratio'] = minutes_per_match_this_season / career_minutes_per_match
        else:
            normalized_features['minutes_per_match_to_career_ratio'] = 0.0
    else:
        normalized_features['minutes_per_match_to_career_ratio'] = 0.0
    
    # Injury frequency ratio (recent vs career) - always set
    cum_inj_starts = ref_row.get('cum_inj_starts', 0) if ref_row.get('cum_inj_starts') is not None else 0
    if cum_inj_starts > 0 and years_active > 0:
        career_injury_rate = cum_inj_starts / years_active
        recent_injury_freq = ref_row.get('injury_frequency', 0) if ref_row.get('injury_frequency') is not None else 0
        if career_injury_rate > 0:
            normalized_features['recent_to_career_injury_frequency_ratio'] = recent_injury_freq / career_injury_rate
        else:
            normalized_features['recent_to_career_injury_frequency_ratio'] = 0.0
    else:
        normalized_features['recent_to_career_injury_frequency_ratio'] = 0.0
    
    # 2. TEMPORAL TREND FEATURES: Slopes and moving averages from windowed features - always set
    if windowed_features:
        # Calculate slopes for key metrics across the 5 weeks
        minutes_weeks = [windowed_features.get(f'minutes_played_numeric_week_{i}', 0) for i in range(1, 6)]
        matches_weeks = [windowed_features.get(f'matches_played_week_{i}', 0) for i in range(1, 6)]
        goals_weeks = [windowed_features.get(f'goals_numeric_week_{i}', 0) for i in range(1, 6)]
        
        # Calculate linear trend (slope) using simple linear regression
        def calculate_slope(values):
            if len(values) < 2:
                return 0.0
            x = np.array(range(len(values)))
            y = np.array(values)
            if np.std(x) == 0:
                return 0.0
            slope = np.corrcoef(x, y)[0, 1] * (np.std(y) / np.std(x)) if np.std(x) > 0 else 0
            return slope
        
        normalized_features['minutes_trend_slope'] = calculate_slope(minutes_weeks)
        normalized_features['matches_trend_slope'] = calculate_slope(matches_weeks)
        normalized_features['goals_trend_slope'] = calculate_slope(goals_weeks)
    else:
        normalized_features['minutes_trend_slope'] = 0.0
        normalized_features['matches_trend_slope'] = 0.0
        normalized_features['goals_trend_slope'] = 0.0
        minutes_weeks = [0] * 5
        matches_weeks = [0] * 5
        goals_weeks = [0] * 5
    
    # Moving averages (3-week and 5-week) - always set
    if len(minutes_weeks) >= 3:
        normalized_features['minutes_3week_avg'] = np.mean(minutes_weeks[-3:])
        normalized_features['minutes_5week_avg'] = np.mean(minutes_weeks)
    else:
        normalized_features['minutes_3week_avg'] = 0.0
        normalized_features['minutes_5week_avg'] = 0.0
    if len(matches_weeks) >= 3:
        normalized_features['matches_3week_avg'] = np.mean(matches_weeks[-3:])
        normalized_features['matches_5week_avg'] = np.mean(matches_weeks)
    else:
        normalized_features['matches_3week_avg'] = 0.0
        normalized_features['matches_5week_avg'] = 0.0
    if len(goals_weeks) >= 3:
        normalized_features['goals_3week_avg'] = np.mean(goals_weeks[-3:])
        normalized_features['goals_5week_avg'] = np.mean(goals_weeks)
    else:
        normalized_features['goals_3week_avg'] = 0.0
        normalized_features['goals_5week_avg'] = 0.0
        
    # Acceleration (change in slope) - difference between recent and early trend - always set
    if len(minutes_weeks) >= 4:
        early_slope = calculate_slope(minutes_weeks[:3])
        recent_slope = calculate_slope(minutes_weeks[-3:])
        normalized_features['minutes_acceleration'] = recent_slope - early_slope
    else:
        normalized_features['minutes_acceleration'] = 0.0
        
    # Workload volatility (coefficient of variation) - always set
    if len(minutes_weeks) > 0 and np.mean(minutes_weeks) > 0:
        normalized_features['minutes_volatility'] = np.std(minutes_weeks) / np.mean(minutes_weeks)
    else:
        normalized_features['minutes_volatility'] = 0.0
    if len(matches_weeks) > 0 and np.mean(matches_weeks) > 0:
        normalized_features['matches_volatility'] = np.std(matches_weeks) / np.mean(matches_weeks)
    else:
        normalized_features['matches_volatility'] = 0.0
    
    # 3. WORKLOAD INTENSITY FEATURES: Acute:Chronic ratios - always set
    if windowed_features:
        # Acute workload (last week)
        acute_minutes = windowed_features.get('minutes_played_numeric_week_5', 0)
        acute_matches = windowed_features.get('matches_played_week_5', 0)
        
        # Chronic workload (average of weeks 1-4, or 4-week average)
        chronic_minutes = np.mean([windowed_features.get(f'minutes_played_numeric_week_{i}', 0) for i in range(1, 5)])
        chronic_matches = np.mean([windowed_features.get(f'matches_played_week_{i}', 0) for i in range(1, 5)])
        
        # Cumulative workload (total over 5 weeks)
        total_5week_minutes = sum([windowed_features.get(f'minutes_played_numeric_week_{i}', 0) for i in range(1, 6)])
        total_5week_matches = sum([windowed_features.get(f'matches_played_week_{i}', 0) for i in range(1, 6)])
    else:
        acute_minutes = 0
        acute_matches = 0
        chronic_minutes = 0.0
        chronic_matches = 0.0
        total_5week_minutes = 0
        total_5week_matches = 0
    
    # Acute:Chronic ratios - always set
    if chronic_minutes > 0:
        normalized_features['acute_chronic_minutes_ratio'] = acute_minutes / chronic_minutes
    else:
        normalized_features['acute_chronic_minutes_ratio'] = 0.0
    if chronic_matches > 0:
        normalized_features['acute_chronic_matches_ratio'] = acute_matches / chronic_matches
    else:
        normalized_features['acute_chronic_matches_ratio'] = 0.0
        
    # Training load spikes (if acute is significantly higher than chronic) - always set
    if chronic_minutes > 0:
        spike_threshold = 1.2  # 20% increase
        normalized_features['workload_spike_indicator'] = 1 if (acute_minutes / chronic_minutes) > spike_threshold else 0
    else:
        normalized_features['workload_spike_indicator'] = 0
        
    # Cumulative workload (total over 5 weeks) - always set
    normalized_features['total_5week_minutes'] = total_5week_minutes
    normalized_features['total_5week_matches'] = total_5week_matches
    
    # Average weekly workload - always set
    if total_5week_minutes > 0:
        normalized_features['avg_weekly_minutes'] = total_5week_minutes / 5
    else:
        normalized_features['avg_weekly_minutes'] = 0.0
    if total_5week_matches > 0:
        normalized_features['avg_weekly_matches'] = total_5week_matches / 5
    else:
        normalized_features['avg_weekly_matches'] = 0.0
    
    # 4. RECOVERY INDICATORS: Days since last match normalized by typical recovery time - always set
    days_since_last_match = ref_row.get('days_since_last_match', 999) if ref_row.get('days_since_last_match') is not None else 999
    if days_since_last_match != 999 and days_since_last_match >= 0:
        # Typical recovery time: 3-4 days for most players
        typical_recovery = 3.5
        if typical_recovery > 0:
            normalized_features['recovery_ratio'] = days_since_last_match / typical_recovery
            # Recovery status: 0 = insufficient, 1 = adequate, >1 = extended rest
            normalized_features['recovery_status'] = min(1.0, days_since_last_match / typical_recovery)
        else:
            normalized_features['recovery_ratio'] = 0.0
            normalized_features['recovery_status'] = 0.0
    else:
        normalized_features['recovery_ratio'] = 0.0
        normalized_features['recovery_status'] = 0.0
    
    # Days since last match vs average match frequency - always set
    if career_matches > 0 and years_active > 0:
        avg_days_between_matches = (years_active * 365.25) / career_matches
        if avg_days_between_matches > 0 and days_since_last_match != 999:
            normalized_features['days_since_match_to_avg_ratio'] = days_since_last_match / avg_days_between_matches
        else:
            normalized_features['days_since_match_to_avg_ratio'] = 0.0
    else:
        normalized_features['days_since_match_to_avg_ratio'] = 0.0
    
    # 5. FEATURE RATIOS: Current performance vs career baseline - always set
    # Goals per match ratio
    goals_per_match_current = ref_row.get('goals_per_match', 0) if ref_row.get('goals_per_match') is not None else 0
    if career_matches > 0 and career_goals > 0:
        career_goals_per_match = career_goals / career_matches
        if career_goals_per_match > 0:
            normalized_features['goals_per_match_ratio'] = goals_per_match_current / career_goals_per_match
        else:
            normalized_features['goals_per_match_ratio'] = 0.0
    else:
        normalized_features['goals_per_match_ratio'] = 0.0
    
    # Assists per match ratio
    assists_per_match_current = ref_row.get('assists_per_match', 0) if ref_row.get('assists_per_match') is not None else 0
    if career_matches > 0 and career_assists > 0:
        career_assists_per_match = career_assists / career_matches
        if career_assists_per_match > 0:
            normalized_features['assists_per_match_ratio'] = assists_per_match_current / career_assists_per_match
        else:
            normalized_features['assists_per_match_ratio'] = 0.0
    else:
        normalized_features['assists_per_match_ratio'] = 0.0
    
    # Minutes per match ratio
    minutes_per_match_current = ref_row.get('minutes_per_match', 0) if ref_row.get('minutes_per_match') is not None else 0
    if career_matches > 0 and career_minutes > 0:
        career_minutes_per_match = career_minutes / career_matches
        if career_minutes_per_match > 0:
            normalized_features['minutes_per_match_ratio'] = minutes_per_match_current / career_minutes_per_match
        else:
            normalized_features['minutes_per_match_ratio'] = 0.0
    else:
        normalized_features['minutes_per_match_ratio'] = 0.0
    
    # Injury days ratio (recent vs career average) - always set
    cum_inj_days = ref_row.get('cum_inj_days', 0) if ref_row.get('cum_inj_days') is not None else 0
    if cum_inj_days > 0 and years_active > 0:
        career_injury_days_per_year = cum_inj_days / years_active
        recent_injury_days = ref_row.get('days_since_last_injury', 999) if ref_row.get('days_since_last_injury') is not None else 999
        if recent_injury_days != 999 and career_injury_days_per_year > 0:
            # Normalize recent injury days by typical injury duration
            typical_injury_duration = 14  # Average injury duration in days
            if typical_injury_duration > 0:
                normalized_features['recent_injury_duration_ratio'] = recent_injury_days / typical_injury_duration
            else:
                normalized_features['recent_injury_duration_ratio'] = 0.0
        else:
            normalized_features['recent_injury_duration_ratio'] = 0.0
    else:
        normalized_features['recent_injury_duration_ratio'] = 0.0
    
    # 6. INTERACTION FEATURES: Combinations of existing features
    # Calculate total_5week_minutes once for reuse - always set
    if windowed_features:
        total_5week_minutes_interaction = sum([windowed_features.get(f'minutes_played_numeric_week_{i}', 0) for i in range(1, 6)])
    else:
        total_5week_minutes_interaction = 0
    
    # Age × workload interaction - always set
    age = ref_row.get('age', 0) if ref_row.get('age') is not None else 0
    if age > 0 and total_5week_minutes_interaction > 0:
        normalized_features['age_x_5week_minutes'] = age * total_5week_minutes_interaction
    else:
        normalized_features['age_x_5week_minutes'] = 0.0
    
    # Career matches × recent form - always set
    if career_matches > 0 and windowed_features:
        recent_goals = sum([windowed_features.get(f'goals_numeric_week_{i}', 0) for i in range(4, 6)])  # Last 2 weeks
        normalized_features['career_matches_x_recent_goals'] = career_matches * recent_goals
    else:
        normalized_features['career_matches_x_recent_goals'] = 0.0
    
    # Injury history × recent workload - always set
    if cum_inj_starts > 0 and windowed_features:
        recent_minutes = sum([windowed_features.get(f'minutes_played_numeric_week_{i}', 0) for i in range(4, 6)])
        normalized_features['injury_history_x_recent_workload'] = cum_inj_starts * recent_minutes
    else:
        normalized_features['injury_history_x_recent_workload'] = 0.0
    
    # Position × workload - always set
    position = ref_row.get('position', '')
    if position and total_5week_minutes_interaction > 0:
        # Convert position to numeric if possible (simplified - you may want to use one-hot encoding)
        position_numeric = hash(str(position)) % 100  # Simple hash for interaction
        normalized_features['position_x_workload'] = position_numeric * total_5week_minutes_interaction
    else:
        normalized_features['position_x_workload'] = 0.0
    
    # 7. ADDITIONAL NORMALIZED CUMULATIVE FEATURES (rates per season/year)
    # These complement the existing normalized features
    
    # Club-specific rates - always set
    club_cum_goals = ref_row.get('club_cum_goals', 0) if ref_row.get('club_cum_goals') is not None else 0
    club_cum_assists = ref_row.get('club_cum_assists', 0) if ref_row.get('club_cum_assists') is not None else 0
    club_cum_minutes = ref_row.get('club_cum_minutes', 0) if ref_row.get('club_cum_minutes') is not None else 0
    
    # Calculate time at current club (simplified - use seasons_count as proxy)
    if seasons_count > 0:
        if club_cum_goals > 0:
            normalized_features['club_goals_per_season'] = club_cum_goals / seasons_count
        else:
            normalized_features['club_goals_per_season'] = 0.0
        if club_cum_assists > 0:
            normalized_features['club_assists_per_season'] = club_cum_assists / seasons_count
        else:
            normalized_features['club_assists_per_season'] = 0.0
        if club_cum_minutes > 0:
            normalized_features['club_minutes_per_season'] = club_cum_minutes / seasons_count
        else:
            normalized_features['club_minutes_per_season'] = 0.0
    else:
        normalized_features['club_goals_per_season'] = 0.0
        normalized_features['club_assists_per_season'] = 0.0
        normalized_features['club_minutes_per_season'] = 0.0
    
    # National team rates - always set
    national_team_apps = ref_row.get('national_team_appearances', 0) if ref_row.get('national_team_appearances') is not None else 0
    national_team_mins = ref_row.get('national_team_minutes', 0) if ref_row.get('national_team_minutes') is not None else 0
    if years_active > 0:
        if national_team_apps > 0:
            normalized_features['national_team_apps_per_year'] = national_team_apps / years_active
        else:
            normalized_features['national_team_apps_per_year'] = 0.0
        if national_team_mins > 0:
            normalized_features['national_team_minutes_per_year'] = national_team_mins / years_active
        else:
            normalized_features['national_team_minutes_per_year'] = 0.0
    else:
        normalized_features['national_team_apps_per_year'] = 0.0
        normalized_features['national_team_minutes_per_year'] = 0.0
    
    # Combine features
    timeline = {**static_features, **windowed_features, **normalized_features}
    # Only include target if provided (for training/backtesting)
    if target is not None:
        timeline['target'] = target
    return timeline

def get_all_player_ids(config_path: str = None) -> List[int]:
    """Get player IDs from config.json. Goalkeepers are skipped when their daily features file is missing."""
    player_ids = []
    
    if config_path is None:
        raise ValueError("config_path is required - cannot use default Chelsea path")
    
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        player_ids = config.get('player_ids', [])
        print(f"   Loaded {len(player_ids)} player IDs from config: {config_path}")
    else:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    print(f"   Found {len(player_ids)} players from config")
    return sorted(player_ids)

def load_player_names_mapping(data_dir: str = None) -> Dict[int, str]:
    """Load player names from players_profile.csv"""
    player_names = {}
    
    # Try production raw_data first
    possible_paths = []
    if data_dir:
        possible_paths.append(Path(data_dir) / 'players_profile.csv')
    else:
        # Find latest raw_data folder
        raw_data_dir = PRODUCTION_ROOT / "raw_data" / "england"
        if raw_data_dir.exists():
            date_folders = sorted([d for d in raw_data_dir.iterdir() if d.is_dir() and d.name.isdigit() and len(d.name) == 8], reverse=True)
            if date_folders:
                possible_paths.append(date_folders[0] / 'players_profile.csv')
    
    # Fallback paths
    possible_paths.extend([
        ROOT_DIR / 'data_exports' / 'transfermarkt' / 'england' / '20251205' / 'players_profile.csv',
        ROOT_DIR / 'data_exports' / 'transfermarkt' / 'england' / '20251213' / 'players_profile.csv',
        ROOT_DIR / 'data_exports' / 'transfermarkt' / 'england' / '20251217' / 'players_profile.csv',
        Path('data_exports/transfermarkt/england/20251205/players_profile.csv'),
        Path('data_exports/transfermarkt/england/20251109/players_profile.csv'),
    ])
    
    for profile_path in possible_paths:
        try:
            if isinstance(profile_path, str):
                profile_path = Path(profile_path)
            if not profile_path.exists():
                continue
                
            if str(profile_path).endswith('.csv'):
                players_df = pd.read_csv(profile_path, sep=';', encoding='utf-8')
                if 'id' in players_df.columns and 'name' in players_df.columns:
                    for _, row in players_df.iterrows():
                        player_id = row.get('id')
                        player_name = row.get('name', '')
                        if pd.notna(player_id) and pd.notna(player_name) and player_name:
                            player_names[int(player_id)] = str(player_name).strip()
                    print(f"[OK] Loaded {len(player_names)} player names from {profile_path}")
                    return player_names
            elif str(profile_path).endswith('.xlsx') and openpyxl is not None:
                try:
                    players_df = pd.read_excel(profile_path)
                    if 'id' in players_df.columns and 'name' in players_df.columns:
                        for _, row in players_df.iterrows():
                            player_id = row.get('id')
                            player_name = row.get('name', '')
                            if pd.notna(player_id) and pd.notna(player_name) and player_name:
                                player_names[int(player_id)] = str(player_name).strip()
                        print(f"[OK] Loaded {len(player_names)} player names from {profile_path}")
                        return player_names
                except Exception as e:
                    continue
        except Exception as e:
            continue
    
    print(f"[WARNING] Could not load player names from profile file. Will use Player_ID format.")
    return player_names

def get_player_name_from_df(df: pd.DataFrame, player_id: int = None, player_names_map: Dict[int, str] = None) -> str:
    """Get player name from the daily features dataframe or player names mapping"""
    # First try to get player_id from dataframe if not provided
    if player_id is None:
        player_id = df['player_id'].iloc[0] if 'player_id' in df.columns else None
    
    # Try player names mapping first (most reliable)
    if player_id and player_names_map and player_id in player_names_map:
        return player_names_map[player_id]
    
    # Try to get from dataframe columns (check multiple possible column names)
    for col_name in ['player_name', 'name']:
        if col_name in df.columns:
            player_name = df[col_name].dropna()
            if not player_name.empty:
                name_value = str(player_name.iloc[0]).strip()
                if name_value and name_value != 'nan' and not name_value.startswith('Player_'):
                    return name_value
    
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
    
    # Ensure output_file is absolute path
    if not os.path.isabs(output_file):
        output_file = os.path.abspath(output_file)
    
    # Write CSV in chunks
    try:
        with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            
            # Write in chunks to manage memory
            for i in range(0, len(timelines), chunk_size):
                chunk = timelines[i:i + chunk_size]
                for timeline in chunk:
                    # Ensure all keys are present (fill missing with None)
                    row = {key: timeline.get(key, None) for key in fieldnames}
                    
                    # Ensure reference_date is in date-only format (YYYY-MM-DD) if present
                    if 'reference_date' in row and row['reference_date'] is not None:
                        try:
                            # If it's already a string in YYYY-MM-DD format, keep it
                            if isinstance(row['reference_date'], str) and len(row['reference_date']) == 10:
                                # Validate it's in correct format
                                pd.to_datetime(row['reference_date'], format='%Y-%m-%d')
                            else:
                                # Convert datetime/timestamp to date string
                                dt = pd.to_datetime(row['reference_date'], errors='coerce')
                                if pd.notna(dt):
                                    row['reference_date'] = dt.strftime('%Y-%m-%d')
                                else:
                                    row['reference_date'] = None
                        except (ValueError, TypeError):
                            # If conversion fails, try to parse and reformat
                            try:
                                dt = pd.to_datetime(row['reference_date'], errors='coerce')
                                if pd.notna(dt):
                                    row['reference_date'] = dt.strftime('%Y-%m-%d')
                                else:
                                    row['reference_date'] = None
                            except:
                                row['reference_date'] = None
                    
                    writer.writerow(row)
    except PermissionError as e:
        raise PermissionError(f"Cannot write to {output_file}. Please close the file if it's open in Excel or another program. Original error: {e}")
    
    # Get shape info by reading just the first few rows
    sample_df = pd.read_csv(output_file, nrows=1)
    num_cols = len(sample_df.columns)
    num_rows = len(timelines)
    del sample_df  # Free memory
    return (num_rows, num_cols)

def process_season(season_start_year: int, daily_features_dir: str, 
                  all_injury_dates_by_player: Dict[int, Set[pd.Timestamp]],
                  injury_class_map: Dict[Tuple[int, pd.Timestamp], str],
                  player_names_map: Dict[int, str],
                  player_ids: List[int],
                  max_players: Optional[int] = None,
                  min_date: Optional[pd.Timestamp] = None,
                  max_date: Optional[pd.Timestamp] = None) -> Tuple[Optional[str], int, int]:
    """Process a single season and generate timeline file
    
    Args:
        min_date: Minimum reference date for timelines (e.g., 2025-12-06)
        max_date: Maximum reference date for timelines (e.g., 2026-01-29)
    """
    season_start, season_end = get_season_date_range(season_start_year)
    output_file = f'timelines_35day_season_{season_start_year}_{season_start_year+1}_v4_muscular.csv'
    
    print(f"\n{'='*80}")
    print(f"PROCESSING SEASON {season_start_year}_{season_start_year+1}")
    print(f"{'='*80}")
    print(f"   Date range: {season_start.date()} to {season_end.date()}")
    
    season_start_time = datetime.now()
    
    # Apply limit if specified (for testing)
    if max_players is not None:
        season_player_ids = player_ids[:max_players]
    else:
        season_player_ids = player_ids
    
    # ===== PASS 1: Generate all injury timelines and collect valid non-injury dates =====
    print(f"\n[PASS 1] Generating injury timelines and identifying valid dates")
    
    all_injury_timelines = []
    all_valid_non_injury_dates = []
    processed_players = 0
    
    for player_id in tqdm(season_player_ids, desc=f"Pass 1: {season_start_year}_{season_start_year+1}", unit="player"):
        try:
            # Skip if daily features file does not exist (e.g. goalkeeper – not generated by update_daily_features)
            daily_features_path = os.path.join(daily_features_dir, f'player_{player_id}_daily_features.csv')
            if not os.path.exists(daily_features_path):
                continue
            # Load player data
            df = pd.read_csv(daily_features_path)
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
            
            # Generate injury timelines (filtered by injury_class and season)
            injury_timelines = generate_injury_timelines_enhanced(player_id, player_name, df_season, injury_class_map)
            
            # Filter injury timelines to this season and min_date
            season_injury_timelines = []
            for timeline in injury_timelines:
                ref_date_str = timeline.get('reference_date', '')
                if ref_date_str:
                    ref_date = pd.to_datetime(ref_date_str)
                    if season_start <= ref_date <= season_end:
                        # Apply min_date filter if provided
                        if min_date is None or ref_date >= min_date:
                            season_injury_timelines.append(timeline)
            
            all_injury_timelines.extend(season_injury_timelines)
            
            # Get ALL valid dates for timeline generation (not filtered by injury status)
            all_valid_dates = get_all_valid_dates_for_timelines(
                df_season,
                season_start=season_start,
                season_end=season_end,
                min_date=min_date,
                max_date=max_date
            )
            
            # Determine target for each date (1 if injury in next 35 days, 0 otherwise)
            for date in all_valid_dates:
                # Check if there's an injury in the next 35 days to set target
                future_end = date + timedelta(days=34)
                future_dates = pd.date_range(date, future_end, freq='D')
                injury_in_window = any(pd.Timestamp(d).normalize() in player_all_injury_dates for d in future_dates)
                
                # Store with target information (0 = no injury, 1 = injury)
                target_value = 1 if injury_in_window else 0
                all_valid_non_injury_dates.append((player_id, date, target_value))
            
            processed_players += 1
            del df, df_season
                
        except Exception as e:
            print(f"\n[ERROR] Error processing player {player_id} for season {season_start_year}: {e}")
            continue
    
    print(f"\n[OK] PASS 1 Complete for {season_start_year}_{season_start_year+1}:")
    print(f"   Injury timelines: {len(all_injury_timelines)}")
    print(f"   All valid dates: {len(all_valid_non_injury_dates)}")
    
    # Use all available dates (no filtering)
    selected_dates = all_valid_non_injury_dates
    print(f"\n[COMPOSITION] DATASET COMPOSITION (Production Mode - All Dates):")
    print(f"   Injury timelines: {len(all_injury_timelines)}")
    print(f"   All dates: {len(selected_dates)}")
    print(f"   [OK] Generating timelines for all {len(selected_dates)} available dates")
    
    # ===== PASS 2: Generate timelines for all dates =====
    print(f"\n[PASS 2] Generating timelines for all dates")
    
    all_non_injury_timelines = []
    dates_by_player = defaultdict(list)
    for player_id, date, target in selected_dates:  # Now includes target
        dates_by_player[player_id].append((date, target))
    
    for player_id, dates_with_targets in tqdm(dates_by_player.items(), desc=f"Pass 2: {season_start_year}_{season_start_year+1}", unit="player"):
        try:
            daily_features_path = os.path.join(daily_features_dir, f'player_{player_id}_daily_features.csv')
            if not os.path.exists(daily_features_path):
                continue
            df = pd.read_csv(daily_features_path)
            df['date'] = pd.to_datetime(df['date'])
            
            # Filter to season date range (with buffer)
            buffer_start = season_start - timedelta(days=34)
            season_mask = (df['date'] >= buffer_start) & (df['date'] <= season_end)
            df_season = df[season_mask].copy()
            
            if len(df_season) == 0:
                continue
            
            player_name = get_player_name_from_df(df_season, player_id=player_id, player_names_map=player_names_map)
            
            # Generate timelines for all dates
            timelines = generate_timelines_for_dates(player_id, player_name, df_season, dates_with_targets)
            all_non_injury_timelines.extend(timelines)
            
            del df, df_season
            
        except Exception as e:
            print(f"\n[ERROR] Error generating non-injury timelines for player {player_id}: {e}")
            continue
    
    # Combine and save
    print(f"\n[FINALIZE] FINALIZING DATASET FOR {season_start_year}_{season_start_year+1}")
    
    injury_count = len(all_injury_timelines)
    non_injury_count = len(all_non_injury_timelines)
    final_timelines = all_injury_timelines + all_non_injury_timelines
    random.shuffle(final_timelines)
    
    total_count = len(final_timelines)
    final_ratio = (injury_count / total_count) if total_count > 0 else 0.0
    
    print(f"\n[SUMMARY] SEASON {season_start_year}_{season_start_year+1} DATASET:")
    print(f"   Total timelines: {total_count:,}")
    print(f"   Injury timelines: {injury_count:,}")
    print(f"   Non-injury timelines: {non_injury_count:,}")
    print(f"   Final injury ratio: {final_ratio:.1%} (natural)")
    
    if final_timelines:
        print(f"\n[SAVE] Saving timelines to CSV...")
        shape = save_timelines_to_csv_chunked(final_timelines, output_file)
        print(f"[OK] Saved to: {output_file}")
        print(f"[SHAPE] Shape: {shape}")
    else:
        print("[WARNING] No timelines generated for this season")
        output_file = None
    
    season_time = datetime.now() - season_start_time
    print(f"[TIME] Season processing time: {season_time}")
    
    return output_file, injury_count, non_injury_count

def main(daily_features_dir: str = None, 
         output_dir: str = None,
         config_path: str = None,
         data_date: str = None,
         min_date: str = None,
         max_date: str = None,
         regenerate_from_date: str = None,
         max_players: Optional[int] = None,
         existing_timelines_file: str = None,
         full_regeneration: bool = False):
    """Main function - processes all seasons
    
    Args:
        daily_features_dir: Directory containing daily features files
        output_dir: Directory to save timeline files
        config_path: Path to config.json with player IDs
        data_date: Raw data date folder (YYYYMMDD) for loading injuries
        min_date: Minimum reference date for timelines (YYYY-MM-DD format, e.g., '2025-12-06')
        max_players: Optional limit on number of players to process (for testing)
        existing_timelines_file: Path to existing timelines file to append to
        full_regeneration: If True, do not load existing timelines; generate from min_date (default 2025-07-01).
    """
    print("ENHANCED 35-DAY TIMELINE GENERATOR V4 - PRODUCTION VERSION")
    print("=" * 80)
    print("Features: 108 enhanced features with 35-day windows")
    print("Processing: Season-by-season - ALL dates (production mode)")
    print("Mode: Generating timelines for ALL dates regardless of injury status")
    
    # Validate required parameters - no Chelsea defaults
    if daily_features_dir is None:
        raise ValueError("daily_features_dir is required - must be specified")
    if output_dir is None:
        raise ValueError("output_dir is required - must be specified")
    if config_path is None:
        raise ValueError("config_path is required - must be specified")
    
    # Load existing timelines if provided (skip when full_regeneration)
    existing_timelines = None
    max_existing_date = None
    if existing_timelines_file and os.path.exists(existing_timelines_file) and not full_regeneration:
        print(f"\n[LOAD] Loading existing timelines from: {existing_timelines_file}")
        try:
            existing_timelines = pd.read_csv(existing_timelines_file, low_memory=False)
            if 'reference_date' in existing_timelines.columns:
                # Try parsing with flexible date format (handles both date-only and datetime)
                existing_timelines['reference_date'] = pd.to_datetime(existing_timelines['reference_date'], errors='coerce', format='mixed')
                # Drop rows with invalid dates
                existing_timelines = existing_timelines.dropna(subset=['reference_date'])
                if len(existing_timelines) > 0:
                    max_existing_date = existing_timelines['reference_date'].max()
                    print(f"[LOAD] Existing timelines have {len(existing_timelines)} rows, latest date: {max_existing_date.date()}")
                    
                    # Handle regenerate_from_date option - remove timelines after that date (keep the date itself)
                    if regenerate_from_date:
                        regenerate_from_ts = pd.to_datetime(regenerate_from_date).normalize()
                        before_count = len(existing_timelines)
                        existing_timelines = existing_timelines[existing_timelines['reference_date'] <= regenerate_from_ts].copy()
                        after_count = len(existing_timelines)
                        removed_count = before_count - after_count
                        print(f"[REGENERATE] Removed {removed_count} timelines after {regenerate_from_date} (kept {after_count} rows including {regenerate_from_date})")
                        
                        # Update max_existing_date to be the day before regenerate_from_date
                        if len(existing_timelines) > 0:
                            max_existing_date = existing_timelines['reference_date'].max()
                            print(f"[REGENERATE] New max date in existing timelines: {max_existing_date.date()}")
                        else:
                            max_existing_date = None
                            existing_timelines = None
                            print(f"[REGENERATE] All timelines removed, will generate from scratch")
                    
                    # Handle max_date truncation
                    if max_date:
                        max_date_ts = pd.to_datetime(max_date).normalize()
                        before_count = len(existing_timelines)
                        existing_timelines = existing_timelines[existing_timelines['reference_date'] <= max_date_ts].copy()
                        after_count = len(existing_timelines)
                        if before_count > after_count:
                            print(f"[TRUNCATE] Truncated existing timelines: removed {before_count - after_count} rows beyond {max_date_ts.date()}")
                            # Save truncated file immediately
                            existing_timelines.to_csv(existing_timelines_file, index=False, encoding='utf-8-sig')
                            if len(existing_timelines) > 0:
                                max_existing_date = existing_timelines['reference_date'].max()
                                print(f"[TRUNCATE] New max date in existing timelines: {max_existing_date.date()}")
                            else:
                                max_existing_date = None
                                existing_timelines = None
                                print(f"[TRUNCATE] All timelines removed, will generate from scratch")
                    
                    # Override min_date to be the day after max_existing_date only if min_date was not explicitly provided
                    if max_existing_date and min_date is None:
                        min_date = (max_existing_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                        print(f"[INCREMENTAL] Will generate timelines from {min_date} onwards")
                    elif min_date:
                        print(f"[INCREMENTAL] Using explicitly provided min_date: {min_date}")
                else:
                    print(f"[WARNING] No valid dates found in existing timelines, will generate from scratch")
                    existing_timelines = None
            else:
                print(f"[WARNING] 'reference_date' column not found in existing timelines, will generate from scratch")
                existing_timelines = None
        except Exception as e:
            print(f"[WARNING] Error loading existing timelines: {e}, will generate from scratch")
            existing_timelines = None
    
    # When no existing file: default min_date to season start so we generate from 2025-07-01 inclusively
    if existing_timelines is None and min_date is None:
        min_date = '2025-07-01'
        print(f"[FROM-SCRATCH] No existing timelines; will generate from {min_date} inclusively")
    
    # Parse min_date and max_date
    min_date_ts = None
    max_date_ts = None
    if max_date:
        max_date_ts = pd.to_datetime(max_date).normalize()
        print(f"[MAX_DATE] Maximum reference date: {max_date_ts.date()}")
    if min_date:
        min_date_ts = pd.Timestamp(min_date)
        print(f"Minimum reference date: {min_date_ts.date()}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Convert paths to absolute before changing directories
    daily_features_dir = os.path.abspath(daily_features_dir)
    output_dir = os.path.abspath(output_dir)
    if config_path:
        config_path = os.path.abspath(config_path)
    
    # Change to output directory for saving files
    original_cwd = os.getcwd()
    os.chdir(output_dir)
    
    try:
        start_time = datetime.now()
        
        # Get available seasons (only from daily features)
        available_seasons = get_all_seasons_from_daily_features(daily_features_dir)
        
        # Load injuries data
        print("\n[LOAD] Loading injuries data...")
        injury_class_map = load_injuries_data(data_date=data_date)
        all_injury_dates_by_player = load_all_injury_dates(data_date=data_date)
        print(f"[OK] Loaded injury data for {len(all_injury_dates_by_player)} players")
        
        # Get player IDs from config
        all_player_ids = get_all_player_ids(config_path=config_path)
        print(f"[OK] Found {len(all_player_ids)} players from config")
        
        # Load player names
        print("\n[LOAD] Loading player names...")
        player_names_map = load_player_names_mapping(data_dir=None)
        if player_names_map:
            print(f"[OK] Loaded {len(player_names_map)} player names")
        else:
            print("[WARNING] No player names loaded - will use Player_ID format")
        
        # Apply limit if specified (for testing)
        if max_players is not None:
            player_ids = all_player_ids[:max_players]
            print(f"[TEST] TEST MODE: Processing {len(player_ids)} players (limited from {len(all_player_ids)})")
        else:
            player_ids = all_player_ids
        
        # Process each season
        output_files = []
        total_injuries = 0
        total_non_injuries = 0
        
        for season_start_year in available_seasons:
            output_file, injury_count, non_injury_count = process_season(
                season_start_year=season_start_year,
                daily_features_dir=daily_features_dir,
                all_injury_dates_by_player=all_injury_dates_by_player,
                injury_class_map=injury_class_map,
                player_names_map=player_names_map,
                player_ids=player_ids,
                max_players=max_players,
                min_date=min_date_ts,
                max_date=max_date_ts
            )
            if output_file:
                output_files.append(output_file)
                total_injuries += injury_count
                total_non_injuries += non_injury_count
        
        # Merge with existing timelines if provided
        if existing_timelines is not None and len(output_files) > 0:
            print(f"\n[MERGE] Merging new timelines with existing timelines...")
            
            # Load all newly generated timeline files
            all_new_timelines = []
            for output_file in output_files:
                file_path = Path(output_file)
                if file_path.exists():
                    new_df = pd.read_csv(file_path, low_memory=False)
                    all_new_timelines.append(new_df)
                    print(f"[MERGE] Loaded {len(new_df)} timelines from {output_file}")
            
            if all_new_timelines:
                new_timelines_df = pd.concat(all_new_timelines, ignore_index=True)
                print(f"[MERGE] New timelines: {len(new_timelines_df)} rows")
                
                # Normalize reference_date to string YYYY-MM-DD so dedup works (existing=datetime, new=string from CSV)
                if 'reference_date' in existing_timelines.columns:
                    existing_timelines = existing_timelines.copy()
                    existing_timelines['reference_date'] = pd.to_datetime(existing_timelines['reference_date'], errors='coerce').dt.strftime('%Y-%m-%d')
                if 'reference_date' in new_timelines_df.columns:
                    new_timelines_df = new_timelines_df.copy()
                    new_timelines_df['reference_date'] = pd.to_datetime(new_timelines_df['reference_date'], errors='coerce').dt.strftime('%Y-%m-%d')
                
                # Merge with existing
                # IMPORTANT: Put new timelines AFTER existing so that keep='last' will keep the new ones
                combined = pd.concat([existing_timelines, new_timelines_df], ignore_index=True)
                
                # Deduplicate by player_id and reference_date
                # Use keep='last' so that newly generated timelines replace old ones when there are duplicates
                if 'player_id' in combined.columns and 'reference_date' in combined.columns:
                    combined = combined.sort_values(['player_id', 'reference_date'])
                    before_dedup = len(combined)
                    combined = combined.drop_duplicates(subset=['player_id', 'reference_date'], keep='last')
                    after_dedup = len(combined)
                    combined = combined.reset_index(drop=True)
                    if before_dedup != after_dedup:
                        print(f"[MERGE] Removed {before_dedup - after_dedup} duplicate rows (kept new timelines)")
                
                # Save merged file
                merged_output_file = Path(output_dir) / "timelines_35day_season_2025_2026_v4_muscular.csv"
                
                # Ensure reference_date is in consistent date-only format (YYYY-MM-DD) before saving
                if 'reference_date' in combined.columns:
                    # Convert to datetime, then to date, then to string for consistent format
                    combined['reference_date'] = pd.to_datetime(combined['reference_date'], errors='coerce').dt.date.astype(str)
                    # Drop rows with invalid dates
                    before_drop = len(combined)
                    combined = combined[combined['reference_date'] != 'NaT'].copy()
                    after_drop = len(combined)
                    if before_drop != after_drop:
                        print(f"[MERGE] Dropped {before_drop - after_drop} rows with invalid dates")
                    valid_dates = pd.to_datetime(combined['reference_date'], errors='coerce').dropna()
                    if len(valid_dates) > 0:
                        print(f"[MERGE] Date range: {valid_dates.min().date()} to {valid_dates.max().date()}")
                
                combined.to_csv(merged_output_file, index=False)
                print(f"[MERGE] Saved merged timelines to: {merged_output_file}")
                print(f"[MERGE] Total timelines: {len(combined)} rows")
            else:
                print("[MERGE] No new timeline files generated, keeping existing timelines")
        elif existing_timelines is not None and len(output_files) == 0:
            print(f"\n[MERGE] No new timelines generated, keeping existing timelines ({len(existing_timelines)} rows)")
        
        # Summary
        print(f"\n{'='*80}")
        print("[SUMMARY] FINAL SUMMARY")
        print(f"{'='*80}")
        print(f"   Processed {len(output_files)} seasons")
        print(f"   Total injury timelines: {total_injuries:,}")
        print(f"   Total non-injury timelines: {total_non_injuries:,}")
        print(f"   Total timelines: {total_injuries + total_non_injuries:,}")
        if existing_timelines is not None:
            print(f"   Existing timelines: {len(existing_timelines):,}")
        print(f"\n   Generated files:")
        for f in output_files:
            print(f"      - {f}")
        
        total_time = datetime.now() - start_time
        print(f"\n[TIME] Total execution time: {total_time}")
    finally:
        os.chdir(original_cwd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate 35-day timelines for production')
    parser.add_argument('--daily-features-dir', type=str, default=None,
                        help='Directory containing daily features files')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save timeline files')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config.json with player IDs')
    parser.add_argument('--data-date', type=str, default=None,
                        help='Raw data date folder (YYYYMMDD) for loading injuries')
    parser.add_argument('--min-date', type=str, default=None,
                        help='Minimum reference date for timelines (YYYY-MM-DD). If not provided and --existing-timelines is set, will auto-detect from existing file.')
    parser.add_argument('--max-date', type=str, default=None,
                        help='Maximum reference date for timelines (YYYY-MM-DD). Truncates existing timelines beyond this date.')
    parser.add_argument('--regenerate-from-date', type=str, default=None,
                        help='Regenerate timelines from this date onwards (YYYY-MM-DD). Will remove existing timelines from this date before regenerating.')
    parser.add_argument('--max-players', type=int, default=None,
                        help='Limit number of players (for testing)')
    parser.add_argument('--existing-timelines', type=str, default=None,
                        help='Path to existing timelines file to append to (defaults to timelines_35day_season_2025_2026_v4_muscular.csv in output-dir)')
    parser.add_argument('--full-regeneration', action='store_true', default=False,
                        help='Regenerate timelines from scratch from 2025-07-01. Do not load existing file even if present.')
    
    args = parser.parse_args()
    
    # Set default existing timelines file if not provided (only if output_dir is specified)
    if args.existing_timelines is None:
        if args.output_dir:
            existing_timelines_path = Path(args.output_dir) / "timelines_35day_season_2025_2026_v4_muscular.csv"
            if existing_timelines_path.exists():
                args.existing_timelines = str(existing_timelines_path)
                print(f"[AUTO] Found existing timelines file: {args.existing_timelines}")
        # Removed Chelsea fallback - output_dir must be provided
    
    # Full regeneration: do not use existing file; generate from 2025-07-01
    if args.full_regeneration:
        args.existing_timelines = None
        if args.min_date is None:
            args.min_date = '2025-07-01'
        print(f"[FULL-REGENERATION] Ignoring existing file; will generate from {args.min_date} inclusively")
    
    main(
        daily_features_dir=args.daily_features_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        data_date=args.data_date,
        min_date=args.min_date,
        max_date=args.max_date,
        regenerate_from_date=args.regenerate_from_date,
        max_players=args.max_players,
        existing_timelines_file=args.existing_timelines,
        full_regeneration=args.full_regeneration
    )

