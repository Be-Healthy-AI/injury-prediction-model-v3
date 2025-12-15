#!/usr/bin/env python3
"""
Enhanced 35-Day Timeline Generator V4 for Muscular Injury Prediction
Adapted version targeting muscular injuries only

Key changes from V3:
- Filters injury timelines by injury_class (muscular only)
- Updated temporal splits: Train (2021-07-01 to 2024-06-30), Val (2024-07-01 to 2025-06-30), Test (>= 2025-07-01)
- Target ratios: Natural (all available positives and negatives, no forced balancing)
- Each injury generates 5 timelines (D-1, D-2, D-3, D-4, D-5)
- Non-injury validation checks for ANY injury (any class) in 35 days after reference
- Excludes goalkeeper player IDs
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

# Configuration
WINDOW_SIZE = 35  # 5 weeks
# Use natural target ratios (no forced balancing - all available positives and negatives)
# TARGET_RATIO_TRAIN = 0.08  # Disabled - using natural ratio
# TARGET_RATIO_VAL = 0.08  # Disabled - using natural ratio
USE_NATURAL_RATIO = True  # Flag to use all available positives and negatives

# Temporal split dates
TRAIN_START = pd.Timestamp('2021-07-01')  # Added 2021/22 season
TRAIN_END = pd.Timestamp('2024-06-30')
VAL_START = pd.Timestamp('2024-07-01')
VAL_END = pd.Timestamp('2025-06-30')
TEST_START = pd.Timestamp('2025-07-01')

# Allowed injury classes for injury timelines (muscular only)
ALLOWED_INJURY_CLASSES = {'muscular'}

# Goalkeeper player IDs to exclude
GOALKEEPER_IDS = {
    238223, 85941, 221624, 14555, 919438, 116648, 1080903, 1082283, 315858, 566799,
    427568, 425306, 493513, 503765, 262749, 503769, 111819, 1131973, 940915, 848753,
    192279, 442531, 656316, 490606, 573132, 403151, 465555, 731466, 732120, 827435,
    857792, 585323, 834397, 258919, 59377, 550829, 74960, 128899, 34130, 495033,
    622236, 234509, 336077, 620362, 587018, 1013690, 1055382, 503883, 105470, 340918,
    71271, 706815, 486604, 662334, 678402, 502676, 226049, 17965, 52570, 282823,
    428016, 286047, 725912, 192080, 85864, 99397, 142389, 1019169, 1019170, 124419,
    29712, 660768, 73564, 406556, 565093, 646353, 745716, 111873, 75458, 110867,
    484838, 555074, 610863, 712181, 591844, 1004709, 1237073, 124884, 249994, 293257,
    136401, 444641, 511964, 215810, 110864, 203026, 260841, 33754, 400536, 486144,
    120629, 29692, 51321, 431422, 543726, 576099, 670878, 61697, 585550, 95976,
    19948, 336367, 130164, 33873, 2857, 14044, 329813, 536794, 605757, 357658,
    574448, 79422, 45494, 452707, 354017, 565684, 127202, 242284, 641458, 432914,
    201574, 222209, 741236, 456292, 125714, 488935, 736251, 505046, 1008553, 542586,
    418561, 101118, 559320, 352041, 194386, 64399, 670840, 196722, 814848, 296802,
    72476, 195488, 186901, 502070, 381469, 127181, 582078, 655136, 303657, 983248,
    371021, 243591, 606576, 829299, 95795, 226073, 121254, 385412, 77757, 33027,
    368629, 91340, 208379, 245625, 621997, 475946, 646991, 696892, 123536
}

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

def load_injuries_data() -> Dict[Tuple[int, pd.Timestamp], str]:
    """
    Load injuries data file and create a mapping of (player_id, injury_date) -> injury_class
    
    Returns:
        Dictionary mapping (player_id, fromDate) to injury_class
    """
    print("📂 Loading injuries data file...")
    
    # Try to find injuries data file - prioritize 20251205 folder
    injuries_paths = [
        'data_exports/transfermarkt/england/20251205/injuries_data.csv',
        '../data_exports/transfermarkt/england/20251205/injuries_data.csv',
        'original_data/20251106_injuries_data.xlsx',
        '../original_data/20251106_injuries_data.xlsx',
        'original_data/injuries_data.xlsx',
        '../original_data/injuries_data.xlsx',
    ]
    
    injuries_path = None
    for path in injuries_paths:
        if os.path.exists(path):
            injuries_path = path
            break
    
    if injuries_path is None:
        raise FileNotFoundError(f"Injuries data file not found. Tried: {injuries_paths}")
    
    print(f"   Loading from: {injuries_path}")
    
    # Load injuries data
    if injuries_path.endswith('.csv'):
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

def load_all_injury_dates() -> Dict[int, Set[pd.Timestamp]]:
    """
    Load all injury dates (any class) for each player
    Used for non-injury validation (checking if ANY injury occurs in 35 days after)
    
    Returns:
        Dictionary mapping player_id to set of injury dates
    """
    print("📂 Loading all injury dates (any class) for non-injury validation...")
    
    # Try to find injuries data file - prioritize 20251205 folder
    injuries_paths = [
        'data_exports/transfermarkt/england/20251205/injuries_data.csv',
        '../data_exports/transfermarkt/england/20251205/injuries_data.csv',
        'original_data/20251106_injuries_data.xlsx',
        '../original_data/20251106_injuries_data.xlsx',
        'original_data/injuries_data.xlsx',
        '../original_data/injuries_data.xlsx',
    ]
    
    injuries_path = None
    for path in injuries_paths:
        if os.path.exists(path):
            injuries_path = path
            break
    
    if injuries_path is None:
        raise FileNotFoundError(f"Injuries data file not found. Tried: {injuries_paths}")
    
    # Load injuries data
    if injuries_path.endswith('.csv'):
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

def get_valid_non_injury_dates(df: pd.DataFrame,
                                train_start: Optional[pd.Timestamp] = None,
                                train_end: Optional[pd.Timestamp] = None,
                                val_start: Optional[pd.Timestamp] = None,
                                val_end: Optional[pd.Timestamp] = None,
                                test_start: Optional[pd.Timestamp] = None,
                                all_injury_dates: Optional[Set[pd.Timestamp]] = None) -> List[pd.Timestamp]:
    """
    Get all valid non-injury reference dates with eligibility filtering
    
    Eligibility rules:
    - Reference date must be within specified date range
    - No injury of ANY class in the 35 days after reference date
    - Complete 35-day window must be available
    - Activity requirement removed (no longer checking for >= 90 minutes or >= 1 match)
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
        
        # Temporal split filtering
        if train_start and train_end:
            if not (train_start <= reference_date <= train_end):
                continue
        elif val_start and val_end:
            if not (val_start <= reference_date <= val_end):
                continue
        elif test_start:
            if reference_date < test_start:
                continue
        
        # CRITICAL: Check if there's an injury of ANY class in the next 35 days
        future_end = reference_date + timedelta(days=34)
        if future_end > max_date:
            continue
        
        # Check if any injury (any class) occurs in this 35-day window
        # Create date range for the future window
        future_dates = pd.date_range(reference_date, future_end, freq='D')
        injury_in_window = any(pd.Timestamp(date).normalize() in player_injury_dates for date in future_dates)
        if injury_in_window:
            continue  # Skip this date - injury (any class) in next 35 days
        
        # Check if we can create a complete 35-day window
        start_date = reference_date - timedelta(days=34)
        if start_date < min_date:
            continue
        
        # Check if we can create a complete 35-day window (already validated above)
        # Activity requirement removed - all dates with complete window and no injury are eligible
        valid_dates.append(reference_date)
    
    return valid_dates

def generate_non_injury_timelines_for_dates(player_id: int, player_name: str, df: pd.DataFrame, 
                                            reference_dates: List[pd.Timestamp]) -> List[Dict]:
    """Generate non-injury timelines for specific reference dates"""
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
        
        # Build timeline (pass df for years_active calculation)
        timeline = build_timeline(player_id, player_name, reference_date, ref_row, windowed_features, target=0, player_df=df)
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
    
    for cum_feature, normalized_name in cum_features_to_normalize:
        cum_value = ref_row.get(cum_feature, 0)
        if cum_value is not None and pd.notna(cum_value) and cum_value > 0:
            normalized_features[normalized_name] = cum_value / years_active
    
    # Injury cumulative features
    cum_inj_starts = ref_row.get('cum_inj_starts', 0) if ref_row.get('cum_inj_starts') is not None else 0
    if cum_inj_starts is not None and pd.notna(cum_inj_starts) and cum_inj_starts > 0:
        normalized_features['injuries_per_year'] = cum_inj_starts / years_active
    
    cum_inj_days = ref_row.get('cum_inj_days', 0) if ref_row.get('cum_inj_days') is not None else 0
    if cum_inj_days is not None and pd.notna(cum_inj_days) and cum_inj_days > 0:
        normalized_features['injury_days_per_year'] = cum_inj_days / years_active
    
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
    
    # Normalize cumulative career features by seasons (keep existing logic)
    if seasons_count and seasons_count > 0:
        # Career matches per season
        career_matches_val = ref_row.get('career_matches', 0) if ref_row.get('career_matches') is not None else 0
        if career_matches_val > 0:
            normalized_features['career_matches_per_season'] = career_matches_val / seasons_count
        
        # Competitions per season
        cum_comp = ref_row.get('cum_competitions', 0) if ref_row.get('cum_competitions') is not None else 0
        if cum_comp > 0:
            normalized_features['competitions_per_season'] = cum_comp / seasons_count
        
        # Teams per season
        cum_teams = ref_row.get('cum_teams', 0) if ref_row.get('cum_teams') is not None else 0
        if cum_teams > 0:
            normalized_features['teams_per_season'] = cum_teams / seasons_count
        
        # Cup competitions per season
        cup_comp = ref_row.get('cup_competitions', 0) if ref_row.get('cup_competitions') is not None else 0
        if cup_comp > 0:
            normalized_features['cup_competitions_per_season'] = cup_comp / seasons_count
        
        # International competitions per season
        int_comp = ref_row.get('international_competitions', 0) if ref_row.get('international_competitions') is not None else 0
        if int_comp > 0:
            normalized_features['international_competitions_per_season'] = int_comp / seasons_count
    
    # Normalize by career matches (rates) - keep existing logic
    if career_matches and career_matches > 0:
        # Goals per match (already exists, but ensure it's normalized)
        if career_goals > 0:
            normalized_features['goals_per_career_match'] = career_goals / career_matches
        
        # Assists per match
        if career_assists > 0:
            normalized_features['assists_per_career_match'] = career_assists / career_matches
        
        # Minutes per match
        if career_minutes > 0:
            normalized_features['minutes_per_career_match'] = career_minutes / career_matches
        
        # Injury frequency (injuries per match)
        if cum_inj_starts > 0:
            normalized_features['injuries_per_career_match'] = cum_inj_starts / career_matches
        
        # Normalize cumulative matches bench by career matches
        cum_bench = ref_row.get('cum_matches_bench', 0) if ref_row.get('cum_matches_bench') is not None else 0
        if cum_bench > 0:
            normalized_features['bench_rate'] = cum_bench / career_matches
    
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
    
    # ===== ENHANCED FEATURES =====
    
    # 1. RELATIVE FEATURES: Ratios and percentages
    # Matches this season vs last season
    matches_this_season = ref_row.get('matches', 0) if ref_row.get('matches') is not None else 0
    matches_last_season = ref_row.get('teams_last_season', 0) if ref_row.get('teams_last_season') is not None else 0
    if matches_last_season > 0:
        normalized_features['matches_this_season_to_last_ratio'] = matches_this_season / matches_last_season
    
    # Career matches ratio (current vs average)
    if career_matches > 0 and seasons_count > 0:
        avg_matches_per_season = career_matches / seasons_count
        if avg_matches_per_season > 0:
            normalized_features['matches_to_avg_season_ratio'] = matches_this_season / avg_matches_per_season
    
    # Goals this season vs career average
    if career_goals > 0 and career_matches > 0:
        career_goals_per_match = career_goals / career_matches
        goals_this_season = ref_row.get('goals_numeric', 0) if ref_row.get('goals_numeric') is not None else 0
        matches_this_season_for_goals = ref_row.get('matches', 0) if ref_row.get('matches') is not None else 0
        if matches_this_season_for_goals > 0 and career_goals_per_match > 0:
            goals_per_match_this_season = goals_this_season / matches_this_season_for_goals
            normalized_features['goals_per_match_to_career_ratio'] = goals_per_match_this_season / career_goals_per_match
    
    # Assists this season vs career average
    if career_assists > 0 and career_matches > 0:
        career_assists_per_match = career_assists / career_matches
        assists_this_season = ref_row.get('assists_numeric', 0) if ref_row.get('assists_numeric') is not None else 0
        matches_this_season_for_assists = ref_row.get('matches', 0) if ref_row.get('matches') is not None else 0
        if matches_this_season_for_assists > 0 and career_assists_per_match > 0:
            assists_per_match_this_season = assists_this_season / matches_this_season_for_assists
            normalized_features['assists_per_match_to_career_ratio'] = assists_per_match_this_season / career_assists_per_match
    
    # Minutes this season vs career average
    if career_minutes > 0 and career_matches > 0:
        career_minutes_per_match = career_minutes / career_matches
        minutes_this_season = ref_row.get('minutes_played_numeric', 0) if ref_row.get('minutes_played_numeric') is not None else 0
        matches_this_season_for_minutes = ref_row.get('matches', 0) if ref_row.get('matches') is not None else 0
        if matches_this_season_for_minutes > 0 and career_minutes_per_match > 0:
            minutes_per_match_this_season = minutes_this_season / matches_this_season_for_minutes
            normalized_features['minutes_per_match_to_career_ratio'] = minutes_per_match_this_season / career_minutes_per_match
    
    # Injury frequency ratio (recent vs career)
    cum_inj_starts = ref_row.get('cum_inj_starts', 0) if ref_row.get('cum_inj_starts') is not None else 0
    if cum_inj_starts > 0 and years_active > 0:
        career_injury_rate = cum_inj_starts / years_active
        recent_injury_freq = ref_row.get('injury_frequency', 0) if ref_row.get('injury_frequency') is not None else 0
        if career_injury_rate > 0:
            normalized_features['recent_to_career_injury_frequency_ratio'] = recent_injury_freq / career_injury_rate
    
    # 2. TEMPORAL TREND FEATURES: Slopes and moving averages from windowed features
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
        
        # Moving averages (3-week and 5-week)
        if len(minutes_weeks) >= 3:
            normalized_features['minutes_3week_avg'] = np.mean(minutes_weeks[-3:])
            normalized_features['minutes_5week_avg'] = np.mean(minutes_weeks)
        if len(matches_weeks) >= 3:
            normalized_features['matches_3week_avg'] = np.mean(matches_weeks[-3:])
            normalized_features['matches_5week_avg'] = np.mean(matches_weeks)
        if len(goals_weeks) >= 3:
            normalized_features['goals_3week_avg'] = np.mean(goals_weeks[-3:])
            normalized_features['goals_5week_avg'] = np.mean(goals_weeks)
        
        # Acceleration (change in slope) - difference between recent and early trend
        if len(minutes_weeks) >= 4:
            early_slope = calculate_slope(minutes_weeks[:3])
            recent_slope = calculate_slope(minutes_weeks[-3:])
            normalized_features['minutes_acceleration'] = recent_slope - early_slope
        
        # Workload volatility (coefficient of variation)
        if len(minutes_weeks) > 0 and np.mean(minutes_weeks) > 0:
            normalized_features['minutes_volatility'] = np.std(minutes_weeks) / np.mean(minutes_weeks)
        if len(matches_weeks) > 0 and np.mean(matches_weeks) > 0:
            normalized_features['matches_volatility'] = np.std(matches_weeks) / np.mean(matches_weeks)
    
    # 3. WORKLOAD INTENSITY FEATURES: Acute:Chronic ratios
    if windowed_features:
        # Acute workload (last week)
        acute_minutes = windowed_features.get('minutes_played_numeric_week_5', 0)
        acute_matches = windowed_features.get('matches_played_week_5', 0)
        
        # Chronic workload (average of weeks 1-4, or 4-week average)
        chronic_minutes = np.mean([windowed_features.get(f'minutes_played_numeric_week_{i}', 0) for i in range(1, 5)])
        chronic_matches = np.mean([windowed_features.get(f'matches_played_week_{i}', 0) for i in range(1, 5)])
        
        # Acute:Chronic ratios
        if chronic_minutes > 0:
            normalized_features['acute_chronic_minutes_ratio'] = acute_minutes / chronic_minutes
        if chronic_matches > 0:
            normalized_features['acute_chronic_matches_ratio'] = acute_matches / chronic_matches
        
        # Training load spikes (if acute is significantly higher than chronic)
        if chronic_minutes > 0:
            spike_threshold = 1.2  # 20% increase
            normalized_features['workload_spike_indicator'] = 1 if (acute_minutes / chronic_minutes) > spike_threshold else 0
        
        # Cumulative workload (total over 5 weeks)
        total_5week_minutes = sum([windowed_features.get(f'minutes_played_numeric_week_{i}', 0) for i in range(1, 6)])
        total_5week_matches = sum([windowed_features.get(f'matches_played_week_{i}', 0) for i in range(1, 6)])
        normalized_features['total_5week_minutes'] = total_5week_minutes
        normalized_features['total_5week_matches'] = total_5week_matches
        
        # Average weekly workload
        if total_5week_minutes > 0:
            normalized_features['avg_weekly_minutes'] = total_5week_minutes / 5
        if total_5week_matches > 0:
            normalized_features['avg_weekly_matches'] = total_5week_matches / 5
    
    # 4. RECOVERY INDICATORS: Days since last match normalized by typical recovery time
    days_since_last_match = ref_row.get('days_since_last_match', 999) if ref_row.get('days_since_last_match') is not None else 999
    if days_since_last_match != 999 and days_since_last_match >= 0:
        # Typical recovery time: 3-4 days for most players
        typical_recovery = 3.5
        if typical_recovery > 0:
            normalized_features['recovery_ratio'] = days_since_last_match / typical_recovery
            # Recovery status: 0 = insufficient, 1 = adequate, >1 = extended rest
            normalized_features['recovery_status'] = min(1.0, days_since_last_match / typical_recovery)
    
    # Days since last match vs average match frequency
    if career_matches > 0 and years_active > 0:
        avg_days_between_matches = (years_active * 365.25) / career_matches
        if avg_days_between_matches > 0 and days_since_last_match != 999:
            normalized_features['days_since_match_to_avg_ratio'] = days_since_last_match / avg_days_between_matches
    
    # 5. FEATURE RATIOS: Current performance vs career baseline
    # Goals per match ratio
    goals_per_match_current = ref_row.get('goals_per_match', 0) if ref_row.get('goals_per_match') is not None else 0
    if career_matches > 0 and career_goals > 0:
        career_goals_per_match = career_goals / career_matches
        if career_goals_per_match > 0:
            normalized_features['goals_per_match_ratio'] = goals_per_match_current / career_goals_per_match
    
    # Assists per match ratio
    assists_per_match_current = ref_row.get('assists_per_match', 0) if ref_row.get('assists_per_match') is not None else 0
    if career_matches > 0 and career_assists > 0:
        career_assists_per_match = career_assists / career_matches
        if career_assists_per_match > 0:
            normalized_features['assists_per_match_ratio'] = assists_per_match_current / career_assists_per_match
    
    # Minutes per match ratio
    minutes_per_match_current = ref_row.get('minutes_per_match', 0) if ref_row.get('minutes_per_match') is not None else 0
    if career_matches > 0 and career_minutes > 0:
        career_minutes_per_match = career_minutes / career_matches
        if career_minutes_per_match > 0:
            normalized_features['minutes_per_match_ratio'] = minutes_per_match_current / career_minutes_per_match
    
    # Injury days ratio (recent vs career average)
    cum_inj_days = ref_row.get('cum_inj_days', 0) if ref_row.get('cum_inj_days') is not None else 0
    if cum_inj_days > 0 and years_active > 0:
        career_injury_days_per_year = cum_inj_days / years_active
        recent_injury_days = ref_row.get('days_since_last_injury', 999) if ref_row.get('days_since_last_injury') is not None else 999
        if recent_injury_days != 999 and career_injury_days_per_year > 0:
            # Normalize recent injury days by typical injury duration
            typical_injury_duration = 14  # Average injury duration in days
            if typical_injury_duration > 0:
                normalized_features['recent_injury_duration_ratio'] = recent_injury_days / typical_injury_duration
    
    # 6. INTERACTION FEATURES: Combinations of existing features
    # Calculate total_5week_minutes once for reuse
    total_5week_minutes_interaction = 0
    if windowed_features:
        total_5week_minutes_interaction = sum([windowed_features.get(f'minutes_played_numeric_week_{i}', 0) for i in range(1, 6)])
    
    # Age × workload interaction
    age = ref_row.get('age', 0) if ref_row.get('age') is not None else 0
    if age > 0 and total_5week_minutes_interaction > 0:
        normalized_features['age_x_5week_minutes'] = age * total_5week_minutes_interaction
    
    # Career matches × recent form
    if career_matches > 0 and windowed_features:
        recent_goals = sum([windowed_features.get(f'goals_numeric_week_{i}', 0) for i in range(4, 6)])  # Last 2 weeks
        normalized_features['career_matches_x_recent_goals'] = career_matches * recent_goals
    else:
        normalized_features['career_matches_x_recent_goals'] = 0.0  # Always set, even if 0
    
    # Injury history × recent workload
    if cum_inj_starts > 0 and windowed_features:
        recent_minutes = sum([windowed_features.get(f'minutes_played_numeric_week_{i}', 0) for i in range(4, 6)])
        normalized_features['injury_history_x_recent_workload'] = cum_inj_starts * recent_minutes
    
    # Position × workload (if position is numeric, otherwise skip)
    position = ref_row.get('position', '')
    if position and total_5week_minutes_interaction > 0:
        # Convert position to numeric if possible (simplified - you may want to use one-hot encoding)
        position_numeric = hash(str(position)) % 100  # Simple hash for interaction
        normalized_features['position_x_workload'] = position_numeric * total_5week_minutes_interaction
    
    # 7. ADDITIONAL NORMALIZED CUMULATIVE FEATURES (rates per season/year)
    # These complement the existing normalized features
    
    # Club-specific rates
    club_cum_goals = ref_row.get('club_cum_goals', 0) if ref_row.get('club_cum_goals') is not None else 0
    club_cum_assists = ref_row.get('club_cum_assists', 0) if ref_row.get('club_cum_assists') is not None else 0
    club_cum_minutes = ref_row.get('club_cum_minutes', 0) if ref_row.get('club_cum_minutes') is not None else 0
    
    # Calculate time at current club (simplified - use seasons_count as proxy)
    if seasons_count > 0:
        if club_cum_goals > 0:
            normalized_features['club_goals_per_season'] = club_cum_goals / seasons_count
        if club_cum_assists > 0:
            normalized_features['club_assists_per_season'] = club_cum_assists / seasons_count
        if club_cum_minutes > 0:
            normalized_features['club_minutes_per_season'] = club_cum_minutes / seasons_count
    
    # National team rates
    national_team_apps = ref_row.get('national_team_appearances', 0) if ref_row.get('national_team_appearances') is not None else 0
    national_team_mins = ref_row.get('national_team_minutes', 0) if ref_row.get('national_team_minutes') is not None else 0
    if years_active > 0:
        if national_team_apps > 0:
            normalized_features['national_team_apps_per_year'] = national_team_apps / years_active
        if national_team_mins > 0:
            normalized_features['national_team_minutes_per_year'] = national_team_mins / years_active
    
    # Combine features
    timeline = {**static_features, **windowed_features, **normalized_features}
    # Only include target if provided (for training/backtesting)
    if target is not None:
        timeline['target'] = target
    return timeline

def get_all_player_ids() -> List[int]:
    """Get all player IDs from the daily features directory, excluding goalkeepers"""
    player_ids = []
    # Use absolute path for daily features
    daily_features_dir = r'C:\Users\joao.henriques\IPM V3\daily_features_output'
    if not os.path.exists(daily_features_dir):
        daily_features_dir = 'daily_features_output'
        if not os.path.exists(daily_features_dir):
            daily_features_dir = '../daily_features_output'  # Fallback if run from scripts directory
    
    if not os.path.exists(daily_features_dir):
        raise FileNotFoundError(f"Daily features directory not found: {daily_features_dir}")
    
    for filename in os.listdir(daily_features_dir):
        if filename.startswith('player_') and filename.endswith('_daily_features.csv'):
            player_id = int(filename.split('_')[1])
            # Exclude goalkeepers
            if player_id not in GOALKEEPER_IDS:
                player_ids.append(player_id)
    
    print(f"   Found {len(player_ids)} players (excluded {len(GOALKEEPER_IDS)} goalkeepers)")
    return sorted(player_ids)

def load_player_names_mapping(data_dir: str = None) -> Dict[int, str]:
    """Load player names from players_profile.csv"""
    player_names = {}
    
    # Try multiple possible locations for players_profile.csv
    possible_paths = [
        'data_exports/transfermarkt/england/20251205/players_profile.csv',
        'data_exports/transfermarkt/england/20251109/players_profile.csv',
        'original_data/20251106_players_profile.xlsx',
        '../data_exports/transfermarkt/england/20251205/players_profile.csv',
        '../original_data/20251106_players_profile.xlsx'
    ]
    
    if data_dir:
        possible_paths.insert(0, os.path.join(data_dir, 'players_profile.csv'))
    
    for profile_path in possible_paths:
        try:
            if profile_path.endswith('.csv'):
                if os.path.exists(profile_path):
                    players_df = pd.read_csv(profile_path, sep=';', encoding='utf-8')
                    if 'id' in players_df.columns and 'name' in players_df.columns:
                        for _, row in players_df.iterrows():
                            player_id = row.get('id')
                            player_name = row.get('name', '')
                            if pd.notna(player_id) and pd.notna(player_name) and player_name:
                                player_names[int(player_id)] = str(player_name).strip()
                        print(f"✅ Loaded {len(player_names)} player names from {profile_path}")
                        return player_names
            elif profile_path.endswith('.xlsx'):
                if os.path.exists(profile_path) and openpyxl is not None:
                    try:
                        players_df = pd.read_excel(profile_path)
                        if 'id' in players_df.columns and 'name' in players_df.columns:
                            for _, row in players_df.iterrows():
                                player_id = row.get('id')
                                player_name = row.get('name', '')
                                if pd.notna(player_id) and pd.notna(player_name) and player_name:
                                    player_names[int(player_id)] = str(player_name).strip()
                            print(f"✅ Loaded {len(player_names)} player names from {profile_path}")
                            return player_names
                    except Exception as e:
                        continue
        except Exception as e:
            continue
    
    print(f"⚠️  Could not load player names from profile file. Will use Player_ID format.")
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

def main(max_players: Optional[int] = None):
    """Main function with temporal split and eligibility filtering
    
    Args:
        max_players: Optional limit on number of players to process (for testing)
    """
    print("🚀 ENHANCED 35-DAY TIMELINE GENERATOR V4 (INJURY CLASS FILTERING)")
    print("=" * 60)
    print("📋 Features: 108 enhanced features with 35-day windows")
    print("⚡ Processing: Optimized two-pass approach with eligibility filtering")
    print("🎯 Target ratios: Natural (all available positives and negatives)")
    print("📅 Temporal Split:")
    print(f"   Training: {TRAIN_START.date()} to {TRAIN_END.date()} (seasons 2021/22, 2022/23, 2023/24)")
    print(f"   Validation: {VAL_START.date()} to {VAL_END.date()} (season 2024/25)")
    print(f"   Test: >= {TEST_START.date()}")
    print(f"🔍 Injury filtering: Only muscular injuries for injury timelines")
    print(f"🚫 Excluding {len(GOALKEEPER_IDS)} goalkeeper player IDs")
    
    start_time = datetime.now()
    
    # Load injuries data for injury_class filtering
    injury_class_map = load_injuries_data()
    all_injury_dates_by_player = load_all_injury_dates()
    
    # Get all player IDs (excluding goalkeepers)
    all_player_ids = get_all_player_ids()
    print(f"Found {len(all_player_ids)} players (after excluding goalkeepers)")
    
    # Load player names mapping
    print("\n📂 Loading player names from profile file...")
    player_names_map = load_player_names_mapping()
    if player_names_map:
        print(f"✅ Loaded {len(player_names_map)} player names")
    else:
        print("⚠️  No player names loaded - will use Player_ID format")
    
    # Apply limit if specified (for testing)
    if max_players is not None:
        player_ids = all_player_ids[:max_players]
        print(f"🧪 TEST MODE: Processing {len(player_ids)} players (limited from {len(all_player_ids)})")
    else:
        player_ids = all_player_ids
    
    # Determine daily features directory
    daily_features_dir = r'C:\Users\joao.henriques\IPM V3\daily_features_output'
    if not os.path.exists(daily_features_dir):
        daily_features_dir = 'daily_features_output'
        if not os.path.exists(daily_features_dir):
            daily_features_dir = '../daily_features_output'
    
    # ===== PASS 1: Generate all injury timelines and collect valid non-injury dates =====
    print("\n" + "=" * 60)
    print("📊 PASS 1: Generating injury timelines and identifying valid dates")
    print("=" * 60)
    
    all_injury_timelines = []
    all_valid_non_injury_dates_train = []  # Store (player_id, reference_date) tuples for training
    all_valid_non_injury_dates_val = []   # Store (player_id, reference_date) tuples for validation
    all_valid_non_injury_dates_test = []  # Store (player_id, reference_date) tuples for test
    processed_players = 0
    
    for player_id in tqdm(player_ids, desc="Pass 1: Processing players", unit="player"):
        try:
            # Load player data
            df = pd.read_csv(f'{daily_features_dir}/player_{player_id}_daily_features.csv')
            df['date'] = pd.to_datetime(df['date'])
            
            # Get player name from dataframe
            player_name = get_player_name_from_df(df, player_id=player_id, player_names_map=player_names_map)
            
            # Get all injury dates for this player (any class) for non-injury validation
            player_all_injury_dates = all_injury_dates_by_player.get(player_id, set())
            
            # Generate injury timelines (filtered by injury_class)
            injury_timelines = generate_injury_timelines_enhanced(player_id, player_name, df, injury_class_map)
            all_injury_timelines.extend(injury_timelines)
            
            # Get valid non-injury dates for training period (2022-07-01 to 2024-06-30)
            valid_dates_train = get_valid_non_injury_dates(
                df,
                train_start=TRAIN_START,
                train_end=TRAIN_END,
                all_injury_dates=player_all_injury_dates
            )
            for date in valid_dates_train:
                all_valid_non_injury_dates_train.append((player_id, date))
            
            # Get valid non-injury dates for validation period (2024-07-01 to 2025-06-30)
            valid_dates_val = get_valid_non_injury_dates(
                df,
                val_start=VAL_START,
                val_end=VAL_END,
                all_injury_dates=player_all_injury_dates
            )
            for date in valid_dates_val:
                all_valid_non_injury_dates_val.append((player_id, date))
            
            # Get valid non-injury dates for test period (>= 2025-07-01)
            valid_dates_test = get_valid_non_injury_dates(
                df,
                test_start=TEST_START,
                all_injury_dates=player_all_injury_dates
            )
            for date in valid_dates_test:
                all_valid_non_injury_dates_test.append((player_id, date))
            
            processed_players += 1
            
            # Progress update every 10 players
            if processed_players % 10 == 0:
                print(f"\n📊 Progress: {processed_players}/{len(player_ids)} players")
                print(f"   Injury timelines: {len(all_injury_timelines)}")
                print(f"   Valid train dates: {len(all_valid_non_injury_dates_train)}")
                print(f"   Valid val dates: {len(all_valid_non_injury_dates_val)}")
                print(f"   Valid test dates: {len(all_valid_non_injury_dates_test)}")
                
            # Memory optimization: clear player data
            del df
                
        except Exception as e:
            print(f"\n❌ Error processing player {player_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    pass1_time = datetime.now() - start_time
    print(f"\n✅ PASS 1 Complete:")
    print(f"   Total injury timelines: {len(all_injury_timelines)}")
    print(f"   Valid train dates: {len(all_valid_non_injury_dates_train)}")
    print(f"   Valid val dates: {len(all_valid_non_injury_dates_val)}")
    print(f"   Valid test dates: {len(all_valid_non_injury_dates_test)}")
    print(f"   Processing time: {pass1_time}")
    
    # Split injury timelines by reference_date into 3 groups
    print("\n" + "=" * 60)
    print("📊 SPLITTING INJURY TIMELINES BY REFERENCE DATE (3-WAY SPLIT)")
    print("=" * 60)
    
    injury_timelines_train = []
    injury_timelines_val = []
    injury_timelines_test = []
    
    for timeline in all_injury_timelines:
        ref_date_str = timeline.get('reference_date', '')
        if ref_date_str:
            ref_date = pd.to_datetime(ref_date_str)
            if TRAIN_START <= ref_date <= TRAIN_END:
                injury_timelines_train.append(timeline)
            elif VAL_START <= ref_date <= VAL_END:
                injury_timelines_val.append(timeline)
            elif ref_date >= TEST_START:
                injury_timelines_test.append(timeline)
    
    print(f"   Training injury timelines ({TRAIN_START.date()} to {TRAIN_END.date()}): {len(injury_timelines_train)}")
    print(f"   Validation injury timelines ({VAL_START.date()} to {VAL_END.date()}): {len(injury_timelines_val)}")
    print(f"   Test injury timelines (>= {TEST_START.date()}): {len(injury_timelines_test)}")
    
    # Use natural target ratios (all available positives and negatives)
    print(f"\n📊 DATASET COMPOSITION (Natural Ratios):")
    
    # Training set (natural ratio - use all available dates)
    if len(injury_timelines_train) == 0:
        print("⚠️  No training injury timelines generated - will use only non-injury timelines")
    
    print(f"\n   TRAINING SET (Natural ratio - all available dates):")
    print(f"      Injury timelines: {len(injury_timelines_train)}")
    print(f"      Available non-injury dates: {len(all_valid_non_injury_dates_train)}")
    selected_dates_train = all_valid_non_injury_dates_train
    print(f"      ✅ Using all {len(selected_dates_train)} available dates")
    
    # Validation set (natural ratio - use all available dates)
    print(f"\n   VALIDATION SET (Natural ratio - all available dates):")
    print(f"      Injury timelines: {len(injury_timelines_val)}")
    print(f"      Available non-injury dates: {len(all_valid_non_injury_dates_val)}")
    selected_dates_val = all_valid_non_injury_dates_val
    print(f"      ✅ Using all {len(selected_dates_val)} available dates")
    
    # Test set (natural ratio - use all available dates)
    print(f"\n   TEST SET (Natural ratio - all available dates):")
    print(f"      Injury timelines: {len(injury_timelines_test)}")
    print(f"      Available non-injury dates: {len(all_valid_non_injury_dates_test)}")
    selected_dates_test = all_valid_non_injury_dates_test
    print(f"      ✅ Using all {len(selected_dates_test)} available dates")
    
    # ===== PASS 2: Generate timelines for selected dates =====
    print("\n" + "=" * 60)
    print("📊 PASS 2: Generating non-injury timelines for selected dates")
    print("=" * 60)
    
    all_non_injury_timelines_train = []
    all_non_injury_timelines_val = []
    all_non_injury_timelines_test = []
    
    # Process training dates
    dates_by_player_train = defaultdict(list)
    for player_id, date in selected_dates_train:
        dates_by_player_train[player_id].append(date)
    
    for player_id, dates in tqdm(dates_by_player_train.items(), desc="Pass 2: Training set", unit="player"):
        try:
            df = pd.read_csv(f'{daily_features_dir}/player_{player_id}_daily_features.csv')
            df['date'] = pd.to_datetime(df['date'])
            player_name = get_player_name_from_df(df, player_id=player_id, player_names_map=player_names_map)
            non_injury_timelines = generate_non_injury_timelines_for_dates(player_id, player_name, df, dates)
            all_non_injury_timelines_train.extend(non_injury_timelines)
            del df
        except Exception as e:
            print(f"\n❌ Error processing player {player_id} (train): {e}")
            continue
    
    # Process validation dates
    dates_by_player_val = defaultdict(list)
    for player_id, date in selected_dates_val:
        dates_by_player_val[player_id].append(date)
    
    for player_id, dates in tqdm(dates_by_player_val.items(), desc="Pass 2: Validation set", unit="player"):
        try:
            df = pd.read_csv(f'{daily_features_dir}/player_{player_id}_daily_features.csv')
            df['date'] = pd.to_datetime(df['date'])
            player_name = get_player_name_from_df(df, player_id=player_id, player_names_map=player_names_map)
            non_injury_timelines = generate_non_injury_timelines_for_dates(player_id, player_name, df, dates)
            all_non_injury_timelines_val.extend(non_injury_timelines)
            del df
        except Exception as e:
            print(f"\n❌ Error processing player {player_id} (val): {e}")
            continue
    
    # Process test dates
    dates_by_player_test = defaultdict(list)
    for player_id, date in selected_dates_test:
        dates_by_player_test[player_id].append(date)
    
    for player_id, dates in tqdm(dates_by_player_test.items(), desc="Pass 2: Test set", unit="player"):
        try:
            df = pd.read_csv(f'{daily_features_dir}/player_{player_id}_daily_features.csv')
            df['date'] = pd.to_datetime(df['date'])
            player_name = get_player_name_from_df(df, player_id=player_id, player_names_map=player_names_map)
            non_injury_timelines = generate_non_injury_timelines_for_dates(player_id, player_name, df, dates)
            all_non_injury_timelines_test.extend(non_injury_timelines)
            del df
        except Exception as e:
            print(f"\n❌ Error processing player {player_id} (test): {e}")
            continue
    
    pass2_time = datetime.now() - start_time
    print(f"\n✅ PASS 2 Complete:")
    print(f"   Generated train non-injury timelines: {len(all_non_injury_timelines_train)}")
    print(f"   Generated val non-injury timelines: {len(all_non_injury_timelines_val)}")
    print(f"   Generated test non-injury timelines: {len(all_non_injury_timelines_test)}")
    print(f"   Processing time: {pass2_time}")
    
    # Combine and save training set
    print("\n" + "=" * 60)
    print("📊 FINALIZING TRAINING DATASET")
    print("=" * 60)
    
    final_timelines_train = injury_timelines_train + all_non_injury_timelines_train
    random.shuffle(final_timelines_train)
    
    injury_count_train = len(injury_timelines_train)
    non_injury_count_train = len(all_non_injury_timelines_train)
    total_count_train = len(final_timelines_train)
    final_ratio_train = injury_count_train / total_count_train if total_count_train > 0 else 0
    
    print(f"\n📈 TRAINING DATASET (Muscular Injuries Only):")
    print(f"   Total timelines: {total_count_train}")
    print(f"   Injury timelines: {injury_count_train}")
    print(f"   Non-injury timelines: {non_injury_count_train}")
    print(f"   Final injury ratio: {final_ratio_train:.1%}")
    
    if final_timelines_train:
        output_file_train = 'timelines_35day_enhanced_natural_v4_muscular_train.csv'
        print(f"\n💾 Saving training timelines to CSV (chunked)...")
        shape = save_timelines_to_csv_chunked(final_timelines_train, output_file_train)
        print(f"✅ Training timelines saved to: {output_file_train}")
        print(f"📊 Shape: {shape}")
        # Clear memory
        del final_timelines_train
    
    # Combine and save validation set
    print("\n" + "=" * 60)
    print("📊 FINALIZING VALIDATION DATASET")
    print("=" * 60)
    
    injury_count_val = len(injury_timelines_val)
    non_injury_count_val = len(all_non_injury_timelines_val)
    final_timelines_val = injury_timelines_val + all_non_injury_timelines_val
    random.shuffle(final_timelines_val)
    
    total_count_val = len(final_timelines_val)
    final_ratio_val = (injury_count_val / total_count_val) if total_count_val > 0 else 0.0

    print(f"\n📈 VALIDATION DATASET (Muscular Injuries Only):")
    print(f"   Total timelines: {total_count_val}")
    print(f"   Injury timelines: {injury_count_val}")
    print(f"   Non-injury timelines: {non_injury_count_val}")
    print(f"   Final injury ratio: {final_ratio_val:.1%}")

    if final_timelines_val:
        output_file_val = 'timelines_35day_enhanced_natural_v4_muscular_val.csv'
        print(f"\n💾 Saving validation timelines to CSV (chunked)...")
        shape = save_timelines_to_csv_chunked(final_timelines_val, output_file_val)
        print(f"✅ Validation timelines saved to: {output_file_val}")
        print(f"📊 Shape: {shape}")
        # Clear memory
        del final_timelines_val
    else:
        print("⚠️  No validation timelines generated; skipping file output.")
    
    # Combine and save test set
    print("\n" + "=" * 60)
    print("📊 FINALIZING TEST DATASET")
    print("=" * 60)
    
    injury_count_test = len(injury_timelines_test)
    non_injury_count_test = len(all_non_injury_timelines_test)
    final_timelines_test = injury_timelines_test + all_non_injury_timelines_test
    random.shuffle(final_timelines_test)
    
    total_count_test = len(final_timelines_test)
    final_ratio_test = (injury_count_test / total_count_test) if total_count_test > 0 else 0.0

    print(f"\n📈 TEST DATASET (Muscular Injuries Only):")
    print(f"   Total timelines: {total_count_test}")
    print(f"   Injury timelines: {injury_count_test}")
    print(f"   Non-injury timelines: {non_injury_count_test}")
    print(f"   Final injury ratio: {final_ratio_test:.1%} (natural)")

    if final_timelines_test:
        output_file_test = 'timelines_35day_enhanced_natural_v4_muscular_test.csv'
        print(f"\n💾 Saving test timelines to CSV (chunked)...")
        shape = save_timelines_to_csv_chunked(final_timelines_test, output_file_test)
        print(f"✅ Test timelines saved to: {output_file_test}")
        print(f"📊 Shape: {shape}")
        # Clear memory
        del final_timelines_test
    else:
        print("⚠️  No test timelines generated; skipping file output.")
    
    # Show feature summary
    print(f"\n📋 FEATURE SUMMARY:")
    print(f"   Static features: {len(get_static_features())}")
    print(f"   Windowed features: {len(get_windowed_features())}")
    print(f"   Total weekly features: {len(get_windowed_features()) * 5}")
    print(f"   Total features: {len(get_static_features()) + len(get_windowed_features()) * 5 + 1}")  # +1 for target
    
    total_time = datetime.now() - start_time
    print(f"\n⏱️  Total execution time: {total_time}")
    print(f"   Pass 1 time: {pass1_time}")
    print(f"   Pass 2 time: {pass2_time - pass1_time}")

if __name__ == "__main__":
    import sys
    # Check for test mode argument or direct number
    if len(sys.argv) > 1:
        if sys.argv[1] == '--test':
            max_players = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            main(max_players=max_players)
        elif sys.argv[1].isdigit():
            # Allow direct number: python script.py 5
            max_players = int(sys.argv[1])
            main(max_players=max_players)
        else:
            main()
    else:
        main()

