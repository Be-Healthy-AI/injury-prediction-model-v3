#!/usr/bin/env python3
"""
Enhanced 35-Day Timeline Generator V3 for Injury Prediction
Combines enhanced features with 35-day timeline logic
Implements player-by-player processing for optimal performance
Balanced dataset with configurable target ratio
Based on V2 with V3 dataset support
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
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Configuration
WINDOW_SIZE = 35  # 5 weeks
TARGET_RATIO = float(os.environ.get("TARGET_RATIO", 0.15))  # Default 15%, can be overridden via environment variable
TRAIN_SPLIT_DATE = pd.Timestamp('2025-07-01')
RECENT_NEGATIVE_MONTHS = int(os.environ.get("RECENT_NEGATIVE_MONTHS", 6))  # default 6 months, can be overridden via environment variable
RECENT_NEGATIVE_START = TRAIN_SPLIT_DATE - pd.DateOffset(months=RECENT_NEGATIVE_MONTHS)
USE_NATURAL_VAL_RATIO = os.environ.get("USE_NATURAL_VAL_RATIO", "false").lower() == "true"  # Use natural ratio for validation

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
        # Non-musculoskeletal signal & fragility features
        'competitive_pace_recovery_count', 'effort_management_count', 'muscle_fatigue_count',
        'days_since_last_signal', 'operation_count', 'intervention_count', 'flu_count',
        'covid_count', 'sick_count', 'days_since_last_fragility', 'fragility_recovery_days',
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
        # Transfermarkt features
        'transfermarkt_score_recent', 'transfermarkt_score_cum', 'transfermarkt_score_avg',
        'transfermarkt_score_rolling5', 'transfermarkt_score_matches',
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

def generate_injury_timelines_enhanced(player_id: int, player_name: str, df: pd.DataFrame) -> List[Dict]:
    """Generate injury timelines - ALL injuries get 5 timelines each"""
    timelines = []
    
    # Vectorized injury start detection
    injury_starts = df[df['cum_inj_starts'] > df['cum_inj_starts'].shift(1)]
    
    for _, injury_row in injury_starts.iterrows():
        injury_date = injury_row['date']
        
        # Generate 5 timelines (D-1, D-2, D-3, D-4, D-5) - ALL OF THEM
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
                                train_cutoff: Optional[pd.Timestamp] = None,
                                val_start: Optional[pd.Timestamp] = None,
                                recent_only: bool = False) -> List[pd.Timestamp]:
    """Get all valid non-injury reference dates with eligibility filtering
    
    Eligibility rules:
    - Must have >= 90 minutes OR >= 1 match in the 35-day window
    - Exception: If within 60 days of recent injury end (rehab cases), allow regardless of activity
    - If recent_only=True, restrict reference dates to the recent period (RECENT_NEGATIVE_MONTHS)
    """
    valid_dates = []
    recent_cutoff = pd.Timestamp(RECENT_NEGATIVE_START) if recent_only else None
    
    # Vectorized injury start detection
    injury_starts = df[df['cum_inj_starts'] > df['cum_inj_starts'].shift(1)]['date'].values
    
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
        
        # Temporal split filtering (if specified)
        if train_cutoff and reference_date > train_cutoff:
            continue  # Skip if beyond training cutoff
        if val_start and reference_date < val_start:
            continue  # Skip if before validation start
        if recent_cutoff is not None and reference_date < recent_cutoff:
            continue  # Enforce recent negatives for training
        
        # CRITICAL: Check if there's an injury in the next 35 days
        future_end = reference_date + timedelta(days=34)
        if future_end > max_date:
            continue
        
        # Check if any injury starts fall in this 35-day window
        injury_in_window = np.any((injury_starts >= reference_date) & (injury_starts <= future_end))
        if injury_in_window:
            continue  # Skip this date - injury in next 35 days
        
        # Check if we can create a complete 35-day window
        start_date = reference_date - timedelta(days=34)
        if start_date < min_date:
            continue
        
        # ELIGIBILITY CHECK: Create windowed features to check activity
        windowed_features = create_windowed_features_vectorized(df, start_date, reference_date)
        if windowed_features is None:
            continue  # Incomplete window
        
        # Get reference row for injury recency check
        ref_row = df[df['date'] == reference_date].iloc[0]
        
        # Check activity requirement
        total_matches = sum(windowed_features.get(f'matches_played_week_{i}', 0) for i in range(1, 6))
        total_minutes = sum(windowed_features.get(f'minutes_played_numeric_week_{i}', 0) for i in range(1, 6))
        
        # Activity requirement: >= 90 minutes OR >= 1 match
        activity_eligible = (total_minutes >= 90) or (total_matches >= 1)
        
        # Injury recency exception: Within 60 days of recent injury (rehab cases)
        days_since_injury_ended = ref_row.get('days_since_last_injury_ended', 999)
        rehab_exception = (days_since_injury_ended <= 60) and (days_since_injury_ended != 999)
        
        # Eligible if activity requirement OR rehab exception
        if activity_eligible or rehab_exception:
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
    
    career_assists = ref_row.get('career_assists', 0) if ref_row.get('career_assists') is not None else 0
    if career_assists is not None and pd.notna(career_assists) and career_assists > 0:
        normalized_features['career_assists_per_year'] = career_assists / years_active
    
    career_minutes = ref_row.get('career_minutes', 0) if ref_row.get('career_minutes') is not None else 0
    if career_minutes is not None and pd.notna(career_minutes) and career_minutes > 0:
        normalized_features['career_minutes_per_year'] = career_minutes / years_active
    
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
    
    # Combine features
    timeline = {**static_features, **windowed_features, **normalized_features}
    # Only include target if provided (for training/backtesting)
    if target is not None:
        timeline['target'] = target
    return timeline

def get_all_player_ids() -> List[int]:
    """Get all player IDs from the daily features directory"""
    player_ids = []
    # V3: Use daily_features_output directory
    daily_features_dir = 'daily_features_output'
    if not os.path.exists(daily_features_dir):
        daily_features_dir = '../daily_features_output'  # Fallback if run from scripts directory
    
    if not os.path.exists(daily_features_dir):
        raise FileNotFoundError(f"Daily features directory not found: {daily_features_dir}")
    
    for filename in os.listdir(daily_features_dir):
        if filename.startswith('player_') and filename.endswith('_daily_features.csv'):
            player_id = int(filename.split('_')[1])
            player_ids.append(player_id)
    return sorted(player_ids)

def get_player_name(player_id: int) -> str:
    """Get player name from the players data"""
    try:
        # Try current V3 file first
        players_file = 'original_data/20251106_players_profile.xlsx'
        if not os.path.exists(players_file):
            players_file = '../original_data/20251106_players_profile.xlsx'
        
        players_df = pd.read_excel(players_file)
        player_row = players_df[players_df['id'] == player_id]
        if not player_row.empty:
            return player_row.iloc[0]['name']
        else:
            return f"Player_{player_id}"
    except:
        return f"Player_{player_id}"

def main(max_players: Optional[int] = None):
    """Main function with temporal split and eligibility filtering
    
    Args:
        max_players: Optional limit on number of players to process (for testing)
    """
    print("🚀 ENHANCED 35-DAY TIMELINE GENERATOR V4 (SEASONAL 3-WAY SPLIT)")
    print("=" * 60)
    print("📋 Features: 108 enhanced features with 35-day windows")
    print("⚡ Processing: Optimized two-pass approach with eligibility filtering")
    print(f"🎯 Target: {TARGET_RATIO:.1%} injury ratio for all datasets")
    print("📅 Temporal Split: Train <= 2024-06-30, Val 2024/25, Test >= 2025-07-01")
    
    start_time = datetime.now()
    
    # Seasonal 3-way split dates
    TRAIN_CUTOFF = pd.Timestamp('2024-06-30')
    VAL_START = pd.Timestamp('2024-07-01')
    VAL_END = pd.Timestamp('2025-06-30')
    TEST_START = pd.Timestamp('2025-07-01')
    
    # Get all player IDs
    all_player_ids = get_all_player_ids()
    print(f"Found {len(all_player_ids)} players")
    
    # Apply limit if specified (for testing)
    if max_players is not None:
        player_ids = all_player_ids[:max_players]
        print(f"🧪 TEST MODE: Processing {len(player_ids)} players (limited from {len(all_player_ids)})")
    else:
        player_ids = all_player_ids
    
    # Determine daily features directory
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
            
            # Get player name
            player_name = get_player_name(player_id)
            
            # Generate all injury timelines (5 per injury)
            injury_timelines = generate_injury_timelines_enhanced(player_id, player_name, df)
            all_injury_timelines.extend(injury_timelines)
            
            # Get valid non-injury dates for training period (<= 2024-06-30)
            # Use recent_only=False to get all available dates, not just recent ones
            valid_dates_train = get_valid_non_injury_dates(
                df,
                train_cutoff=TRAIN_CUTOFF,
                val_start=None,
                recent_only=False
            )
            for date in valid_dates_train:
                all_valid_non_injury_dates_train.append((player_id, date))
            
            # Get valid non-injury dates for validation period (2024-07-01 to 2025-06-30)
            # Need to filter dates between VAL_START and VAL_END
            valid_dates_val_all = get_valid_non_injury_dates(
                df,
                train_cutoff=None,
                val_start=VAL_START,
                recent_only=False
            )
            # Filter to only dates within validation period
            for date in valid_dates_val_all:
                if VAL_START <= date <= VAL_END:
                    all_valid_non_injury_dates_val.append((player_id, date))
            
            # Get valid non-injury dates for test period (>= 2025-07-01)
            valid_dates_test_all = get_valid_non_injury_dates(
                df,
                train_cutoff=None,
                val_start=TEST_START,
                recent_only=False
            )
            for date in valid_dates_test_all:
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
            if ref_date <= TRAIN_CUTOFF:
                injury_timelines_train.append(timeline)
            elif VAL_START <= ref_date <= VAL_END:
                injury_timelines_val.append(timeline)
            elif ref_date >= TEST_START:
                injury_timelines_test.append(timeline)
    
    print(f"   Training injury timelines (<= 2024-06-30): {len(injury_timelines_train)}")
    print(f"   Validation injury timelines (2024/25): {len(injury_timelines_val)}")
    print(f"   Test injury timelines (>= 2025-07-01): {len(injury_timelines_test)}")
    
    # Calculate required non-injury timelines for all three sets (8% ratio)
    print(f"\n📊 BALANCING CALCULATIONS (Target: {TARGET_RATIO:.1%}):")
    
    # Training set
    if len(injury_timelines_train) == 0:
        print("❌ No training injury timelines generated! Cannot create balanced training dataset.")
        return
    
    required_non_injury_train = int(len(injury_timelines_train) * (1 - TARGET_RATIO) / TARGET_RATIO)
    print(f"\n   TRAINING SET:")
    print(f"      Injury timelines: {len(injury_timelines_train)}")
    print(f"      Required non-injury: {required_non_injury_train}")
    print(f"      Available dates: {len(all_valid_non_injury_dates_train)}")
    
    if len(all_valid_non_injury_dates_train) < required_non_injury_train:
        print(f"      ⚠️  Using all available dates (actual ratio will be higher than {TARGET_RATIO:.1%})")
        selected_dates_train = all_valid_non_injury_dates_train
    else:
        selected_dates_train = random.sample(all_valid_non_injury_dates_train, required_non_injury_train)
        print(f"      ✅ Selected {len(selected_dates_train)} dates")
    
    # Validation set
    if len(injury_timelines_val) == 0:
        print(f"\n   VALIDATION SET:")
        print(f"      ⚠️  No injury timelines - using all available dates")
        selected_dates_val = all_valid_non_injury_dates_val
    else:
        required_non_injury_val = int(len(injury_timelines_val) * (1 - TARGET_RATIO) / TARGET_RATIO)
        print(f"\n   VALIDATION SET:")
        print(f"      Injury timelines: {len(injury_timelines_val)}")
        print(f"      Required non-injury: {required_non_injury_val}")
        print(f"      Available dates: {len(all_valid_non_injury_dates_val)}")
        
        if len(all_valid_non_injury_dates_val) < required_non_injury_val:
            print(f"      ⚠️  Using all available dates (actual ratio will be higher than {TARGET_RATIO:.1%})")
            selected_dates_val = all_valid_non_injury_dates_val
        else:
            selected_dates_val = random.sample(all_valid_non_injury_dates_val, required_non_injury_val)
            print(f"      ✅ Selected {len(selected_dates_val)} dates")
    
    # Test set
    if len(injury_timelines_test) == 0:
        print(f"\n   TEST SET:")
        print(f"      ⚠️  No injury timelines - using all available dates")
        selected_dates_test = all_valid_non_injury_dates_test
    else:
        required_non_injury_test = int(len(injury_timelines_test) * (1 - TARGET_RATIO) / TARGET_RATIO)
        print(f"\n   TEST SET:")
        print(f"      Injury timelines: {len(injury_timelines_test)}")
        print(f"      Required non-injury: {required_non_injury_test}")
        print(f"      Available dates: {len(all_valid_non_injury_dates_test)}")
        
        if len(all_valid_non_injury_dates_test) < required_non_injury_test:
            print(f"      ⚠️  Using all available dates (actual ratio will be higher than {TARGET_RATIO:.1%})")
            selected_dates_test = all_valid_non_injury_dates_test
        else:
            selected_dates_test = random.sample(all_valid_non_injury_dates_test, required_non_injury_test)
            print(f"      ✅ Selected {len(selected_dates_test)} dates")
    
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
            player_name = get_player_name(player_id)
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
            player_name = get_player_name(player_id)
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
            player_name = get_player_name(player_id)
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
    
    print(f"\n📈 TRAINING DATASET:")
    print(f"   Total timelines: {total_count_train}")
    print(f"   Injury timelines: {injury_count_train}")
    print(f"   Non-injury timelines: {non_injury_count_train}")
    print(f"   Final injury ratio: {final_ratio_train:.1%}")
    
    if final_timelines_train:
        timelines_df_train = pd.DataFrame(final_timelines_train)
        output_file_train = 'timelines_35day_enhanced_balanced_v4_train.csv'
        timelines_df_train.to_csv(output_file_train, index=False, encoding='utf-8-sig')
        print(f"\n✅ Training timelines saved to: {output_file_train}")
        print(f"📊 Shape: {timelines_df_train.shape}")
    
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

    print(f"\n📈 VALIDATION DATASET:")
    print(f"   Total timelines: {total_count_val}")
    print(f"   Injury timelines: {injury_count_val}")
    print(f"   Non-injury timelines: {non_injury_count_val}")
    print(f"   Final injury ratio: {final_ratio_val:.1%}")

    if final_timelines_val:
        timelines_df_val = pd.DataFrame(final_timelines_val)
        output_file_val = 'timelines_35day_enhanced_balanced_v4_val.csv'
        timelines_df_val.to_csv(output_file_val, index=False, encoding='utf-8-sig')
        print(f"\n✅ Validation timelines saved to: {output_file_val}")
        print(f"📊 Shape: {timelines_df_val.shape}")
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

    print(f"\n📈 TEST DATASET:")
    print(f"   Total timelines: {total_count_test}")
    print(f"   Injury timelines: {injury_count_test}")
    print(f"   Non-injury timelines: {non_injury_count_test}")
    print(f"   Final injury ratio: {final_ratio_test:.1%}")

    if final_timelines_test:
        timelines_df_test = pd.DataFrame(final_timelines_test)
        output_file_test = 'timelines_35day_enhanced_balanced_v4_test.csv'
        timelines_df_test.to_csv(output_file_test, index=False, encoding='utf-8-sig')
        print(f"\n✅ Test timelines saved to: {output_file_test}")
        print(f"📊 Shape: {timelines_df_test.shape}")
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
    # Check for test mode argument
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        max_players = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        main(max_players=max_players)
    else:
        main()
