#!/usr/bin/env python3
"""
Daily Features Generator - Version 2
Standalone implementation from scratch, avoiding data leakage and drift.
Generates comprehensive daily features for football players.
"""

import pandas as pd
import numpy as np
import os
import sys
import glob
import logging
import time
import traceback
import argparse
import re
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

# Add scripts directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

# Import benfica-parity functions
from benfica_parity_config import (
    map_competition_importance_benfica_parity,
    detect_disciplinary_action_benfica_parity,
)

# Import position normalization
try:
    from scripts.data_collection.transformers import _normalize_position
except ImportError:
    try:
        from data_collection.transformers import _normalize_position
    except ImportError:
        # Fallback: define a simple version if import fails
        def _normalize_position(position):
            if pd.isna(position) or not position:
                return ''
            return str(position).strip()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = r'data_exports\transfermarkt\england\20251205'
REFERENCE_DATE = pd.Timestamp('2025-12-05')
OUTPUT_DIR = 'daily_features_output'

# Helper functions
def map_severity_to_numeric(severity: str) -> int:
    """Map severity string to numeric value."""
    if pd.isna(severity):
        return 1  # Default to mild
    severity_lower = str(severity).lower().strip()
    if 'mild' in severity_lower:
        return 1
    elif 'moderate' in severity_lower:
        return 2
    elif 'severe' in severity_lower:
        return 3
    elif 'critical' in severity_lower:
        return 4
    else:
        return 1  # Default to mild

def load_teams_data(data_dir: str) -> pd.DataFrame:
    """Load teams data CSV file."""
    teams_path = os.path.join(data_dir, 'teams_data.csv')
    if os.path.exists(teams_path):
        logger.info(f"Loading teams data from {teams_path}")
        teams = pd.read_csv(teams_path, encoding='utf-8-sig')
        return teams
    else:
        logger.warning(f"Teams data file not found: {teams_path}")
        return pd.DataFrame(columns=['team', 'country'])

def get_team_country(team_name: str, team_country_map: Dict[str, str]) -> Optional[str]:
    """Get country for a team name."""
    if pd.isna(team_name):
        return None
    team_name_str = str(team_name).strip()
    return team_country_map.get(team_name_str)

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

def normalize_team_name(team_name: str) -> str:
    """Normalize team name for comparison."""
    if pd.isna(team_name) or not team_name:
        return ''
    
    # Convert to string and lowercase
    name = str(team_name).strip().lower()
    
    # Remove common suffixes/prefixes that don't affect identity
    name = re.sub(r'\s+', ' ', name)  # Normalize whitespace
    name = re.sub(r'\b(fc|cf|sc|ac|bc|bk)\b', '', name)  # Remove common prefixes
    name = re.sub(r'\b(u\d+|u-\d+|youth|junior|reserve|b team)\b', '', name)  # Remove age groups
    name = re.sub(r'\s+', ' ', name).strip()  # Clean up whitespace again
    
    return name

def identify_player_team(match: pd.Series, current_club: Optional[str] = None) -> Optional[str]:
    """Identify which team the player is playing for in a match."""
    # Try to match with current club first (normalized)
    if current_club:
        current_club_norm = normalize_team_name(current_club)
        for team_col in ['home_team', 'away_team', 'team']:
            if team_col in match and pd.notna(match[team_col]):
                team_name = str(match[team_col]).strip()
                if not is_national_team(team_name):
                    if normalize_team_name(team_name) == current_club_norm:
                        return team_name
        
        # If no match, try direct comparison as fallback
        for team_col in ['home_team', 'away_team', 'team']:
            if team_col in match and pd.notna(match[team_col]):
                team_name = str(match[team_col]).strip()
                if not is_national_team(team_name):
                    if team_name == current_club:
                        return team_name
    
    # Fallback: return first non-national team
    for team_col in ['home_team', 'away_team', 'team']:
        if team_col in match and pd.notna(match[team_col]):
            team_name = str(match[team_col]).strip()
            if not is_national_team(team_name):
                return team_name
    
    return None

def get_football_season(date: pd.Timestamp) -> str:
    """Get football season string (e.g., '2023/24') for a date."""
    year = date.year
    month = date.month
    if month >= 7:  # July onwards is next season
        return f"{year}/{str(year + 1)[-2:]}"
    else:  # January to June is current season
        return f"{year - 1}/{str(year)[-2:]}"

def parse_substitution_minutes(sub_str: str) -> Optional[int]:
    """Parse substitution minute from string like '45' or '45+2'."""
    if pd.isna(sub_str) or sub_str == '':
        return None
    try:
        # Handle formats like "45", "45+2", "90+3"
        sub_str = str(sub_str).strip()
        if '+' in sub_str:
            parts = sub_str.split('+')
            return int(parts[0])
        else:
            return int(sub_str)
    except:
        return None

def load_player_data(player_id: int, data_dir: str) -> Dict[str, pd.DataFrame]:
    """Load all data for a specific player."""
    logger.info(f"Loading data for player {player_id}...")
    
    # Load profile
    players_path = os.path.join(data_dir, 'players_profile.csv')
    if os.path.exists(players_path):
        players = pd.read_csv(players_path, sep=';', encoding='utf-8')
        players = players[players['id'] == player_id].copy()
        if 'date_of_birth' in players.columns:
            players['date_of_birth'] = pd.to_datetime(players['date_of_birth'], format='%d/%m/%Y', errors='coerce')
    else:
        players = pd.DataFrame()
    
    # Load injuries
    injuries_path = os.path.join(data_dir, 'injuries_data.csv')
    if os.path.exists(injuries_path):
        injuries = pd.read_csv(injuries_path, sep=';', encoding='utf-8')
        injuries = injuries[injuries['player_id'] == player_id].copy()
        if 'fromDate' in injuries.columns:
            injuries['fromDate'] = pd.to_datetime(injuries['fromDate'], errors='coerce')
        if 'untilDate' in injuries.columns:
            injuries['untilDate'] = pd.to_datetime(injuries['untilDate'], errors='coerce')
    else:
        injuries = pd.DataFrame()
    
    # Load matches
    match_data_dir = os.path.join(data_dir, 'match_data')
    matches = []
    if os.path.exists(match_data_dir):
        match_files = glob.glob(os.path.join(match_data_dir, f'player_{player_id}_*.csv'))
        for match_file in match_files:
            try:
                df = pd.read_csv(match_file, encoding='utf-8-sig')
                if 'player_id' in df.columns:
                    df = df[df['player_id'] == player_id].copy()
                matches.append(df)
            except Exception as e:
                logger.warning(f"Error loading {match_file}: {e}")
    
    if matches:
        matches = pd.concat(matches, ignore_index=True)
        if 'date' in matches.columns:
            matches['date'] = pd.to_datetime(matches['date'], errors='coerce')
    else:
        matches = pd.DataFrame()
    
    # Load career
    career_path = os.path.join(data_dir, 'players_career.csv')
    if os.path.exists(career_path):
        career = pd.read_csv(career_path, sep=';', encoding='utf-8')
        career = career[career['id'] == player_id].copy()
        if 'Date' in career.columns:
            career['Date'] = pd.to_datetime(career['Date'], format='%d/%m/%Y', errors='coerce')
    else:
        career = pd.DataFrame()
    
    # Load teams data
    teams = load_teams_data(data_dir)
    team_country_map = {}
    if not teams.empty and 'team' in teams.columns and 'country' in teams.columns:
        for _, row in teams.iterrows():
            team_name = str(row['team']).strip()
            country = row['country'] if pd.notna(row['country']) else None
            if country:
                team_country_map[team_name] = country
    
    logger.info(f"Loaded: {len(players)} players, {len(injuries)} injuries, {len(matches)} matches, {len(career)} career entries")
    
    return {
        'players': players,
        'injuries': injuries,
        'matches': matches,
        'career': career,
        'teams': teams,
        'team_country_map': team_country_map
    }

def preprocess_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Preprocess match data."""
    if matches.empty:
        return matches
    
    matches = matches.copy()
    
    # Validate required columns exist
    required_cols = ['date']
    missing_cols = [col for col in required_cols if col not in matches.columns]
    if missing_cols:
        logger.warning(f"Missing required columns in matches: {missing_cols}")
        # Add missing columns with default values
        for col in missing_cols:
            if col == 'date':
                matches[col] = pd.NaT
    
    # Create numeric columns
    if 'goals' in matches.columns:
        matches['goals_numeric'] = pd.to_numeric(matches['goals'], errors='coerce').fillna(0).astype(int)
    else:
        matches['goals_numeric'] = 0
    
    if 'assists' in matches.columns:
        matches['assists_numeric'] = pd.to_numeric(matches['assists'], errors='coerce').fillna(0).astype(int)
    else:
        matches['assists_numeric'] = 0
    
    if 'minutes_played' in matches.columns:
        matches['minutes_played_numeric'] = pd.to_numeric(matches['minutes_played'], errors='coerce').fillna(0).astype(int)
    else:
        matches['minutes_played_numeric'] = 0
    
    # Cards
    if 'yellow_cards' in matches.columns:
        matches['yellow_cards_numeric'] = (pd.to_numeric(matches['yellow_cards'], errors='coerce').fillna(0) > 0).astype(int)
    else:
        matches['yellow_cards_numeric'] = 0
    
    if 'red_cards' in matches.columns:
        matches['red_cards_numeric'] = (pd.to_numeric(matches['red_cards'], errors='coerce').fillna(0) > 0).astype(int)
    else:
        matches['red_cards_numeric'] = 0
    
    # Competition importance
    if 'competition' in matches.columns:
        matches['competition_importance'] = matches['competition'].apply(map_competition_importance_benfica_parity)
    else:
        matches['competition_importance'] = 1
    
    # Disciplinary action
    matches['disciplinary_action'] = matches.apply(detect_disciplinary_action_benfica_parity, axis=1)
    
    # Determine participation status
    def determine_participation_status(row):
        if pd.isna(row.get('minutes_played_numeric', 0)) or row.get('minutes_played_numeric', 0) == 0:
            # Check if injured
            if pd.notna(row.get('injury_status')) and str(row.get('injury_status', '')).lower() in ['injured', 'out']:
                return 'injured'
            # Check if on bench
            elif pd.notna(row.get('substitutions_on')) or pd.notna(row.get('substitutions_off')):
                return 'bench_unused'
            else:
                return 'not_selected'
        else:
            return 'played'
    
    matches['participation_status'] = matches.apply(determine_participation_status, axis=1)
    matches['matches_played'] = (matches['participation_status'] == 'played').astype(int)
    matches['matches_bench_unused'] = (matches['participation_status'] == 'bench_unused').astype(int)
    matches['matches_not_selected'] = (matches['participation_status'] == 'not_selected').astype(int)
    matches['matches_injured'] = (matches['participation_status'] == 'injured').astype(int)
    
    # Parse substitution data
    if 'substitutions_on' in matches.columns:
        matches['substitution_on_minute'] = matches['substitutions_on'].apply(parse_substitution_minutes)
    else:
        matches['substitution_on_minute'] = None
    
    if 'substitutions_off' in matches.columns:
        matches['substitution_off_minute'] = matches['substitutions_off'].apply(parse_substitution_minutes)
    else:
        matches['substitution_off_minute'] = None
    
    # Sort by date
    if 'date' in matches.columns:
        matches = matches.sort_values('date').reset_index(drop=True)
    
    return matches

def determine_calendar(matches: pd.DataFrame, injuries: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DatetimeIndex:
    """Determine the date range for feature generation."""
    start_dates = []
    
    if not matches.empty and 'date' in matches.columns:
        first_match = matches['date'].min()
        if pd.notna(first_match):
            start_dates.append(first_match)
    
    if not injuries.empty and 'fromDate' in injuries.columns:
        first_injury = injuries['fromDate'].min()
        if pd.notna(first_injury):
            start_dates.append(first_injury)
    
    if start_dates:
        start_date = min(start_dates).normalize()
    else:
        start_date = pd.Timestamp('2000-01-01')
    
    end_date = reference_date.normalize()
    
    calendar = pd.date_range(start=start_date, end=end_date, freq='D')
    logger.info(f"Calendar: {len(calendar)} days from {start_date.date()} to {end_date.date()}")
    
    return calendar

def calculate_profile_features(
    player_row: pd.Series,
    calendar: pd.DatetimeIndex,
    matches: pd.DataFrame,
    career: pd.DataFrame,
    team_country_map: Dict[str, str]
) -> pd.DataFrame:
    """Calculate profile features for each day."""
    logger.info("=== Starting calculate_profile_features ===")
    n_days = len(calendar)
    
    # Initialize features
    features = {
        'player_id': [player_row['id']] * n_days if 'id' in player_row else [0] * n_days,
        'date': calendar,
        'age': [0.0] * n_days,
        'seniority_days': [0] * n_days,
        'position': [player_row.get('position', '')] * n_days,
        'nationality1': [player_row.get('nationality1', '')] * n_days,
        'nationality2': [player_row.get('nationality2', '')] * n_days,
        'height_cm': [player_row.get('height', 0.0)] * n_days if pd.notna(player_row.get('height')) else [0.0] * n_days,
        'dominant_foot': [player_row.get('foot', '')] * n_days,
        'previous_club': [''] * n_days,
        'previous_club_country': [''] * n_days,
        'current_club': [''] * n_days,
        'current_club_country': [''] * n_days,
        'teams_today': [0] * n_days,
        'cum_teams': [0] * n_days,
        'seasons_count': [0] * n_days,
    }
    
    # Calculate age
    if 'date_of_birth' in player_row and pd.notna(player_row['date_of_birth']):
        dob = pd.Timestamp(player_row['date_of_birth'])
        for i, date in enumerate(calendar):
            features['age'][i] = (date - dob).days / 365.25
    
    # Process career to get club changes
    club_changes = []
    if not career.empty and 'Date' in career.columns and 'To' in career.columns:
        for _, row in career.iterrows():
            if pd.notna(row['Date']) and pd.notna(row['To']):
                club_changes.append({
                    'date': pd.Timestamp(row['Date']).normalize(),
                    'club': str(row['To']).strip()
                })
        club_changes.sort(key=lambda x: x['date'])
    
    # Process matches to get unique teams (excluding national teams)
    if not matches.empty and 'date' in matches.columns:
        # Get unique teams per day from matches
        match_teams = set()
        for _, match in matches.iterrows():
            if pd.notna(match.get('date')):
                match_date = pd.Timestamp(match['date']).normalize()
                # Check home_team and away_team
                for team_col in ['home_team', 'away_team', 'team']:
                    if team_col in match and pd.notna(match[team_col]):
                        team_name = str(match[team_col]).strip()
                        if not is_national_team(team_name):
                            match_teams.add(team_name)
    
    # Track current club and seniority
    current_club = None
    current_club_start_date = None
    previous_club = None
    all_teams_seen = set()
    seasons_seen = set()
    
    # Progress tracking
    start_time = time.time()
    last_log_time = start_time
    
    for i, date in enumerate(calendar):
        current_time = time.time()
        
        # Log progress every 1000 days or every 30 seconds
        if i % 1000 == 0 or (current_time - last_log_time) >= 30:
            elapsed = current_time - start_time
            progress_pct = (i * 100) // n_days if n_days > 0 else 0
            if i > 0:
                rate = i / elapsed if elapsed > 0 else 0
                remaining_days = n_days - i
                eta_seconds = remaining_days / rate if rate > 0 else 0
                eta_str = str(timedelta(seconds=int(eta_seconds)))
                elapsed_str = str(timedelta(seconds=int(elapsed)))
                logger.info(f"Profile features progress: {i}/{n_days} days ({progress_pct}%) | "
                          f"Elapsed: {elapsed_str} | ETA: {eta_str} | Rate: {rate:.2f} days/sec")
            sys.stdout.flush()
            last_log_time = current_time
        
        date_norm = date.normalize()
        
        # Check for club changes
        for change in club_changes:
            if change['date'] <= date_norm:
                previous_club = current_club
                current_club = change['club']
                current_club_start_date = change['date']
                if current_club:
                    all_teams_seen.add(current_club)
        
        # If no club change found, try to infer from matches
        if current_club is None and not matches.empty:
            # Find the first match before or on this date
            past_matches = matches[matches['date'] <= date_norm]
            if not past_matches.empty:
                last_match = past_matches.iloc[-1]
                for team_col in ['home_team', 'away_team', 'team']:
                    if team_col in last_match and pd.notna(last_match[team_col]):
                        team_name = str(last_match[team_col]).strip()
                        if not is_national_team(team_name):
                            if current_club is None:
                                current_club = team_name
                                current_club_start_date = date_norm
                            all_teams_seen.add(team_name)
                            break
        
        # Calculate seniority_days (reset on club change)
        if current_club and current_club_start_date:
            features['seniority_days'][i] = (date_norm - current_club_start_date).days
        else:
            features['seniority_days'][i] = 0
        
        # Set current and previous club
        if current_club:
            features['current_club'][i] = current_club
            features['current_club_country'][i] = get_team_country(current_club, team_country_map) or ''
        if previous_club:
            features['previous_club'][i] = previous_club
            features['previous_club_country'][i] = get_team_country(previous_club, team_country_map) or ''
        
        # Calculate teams_today (unique teams played for on this day, from matches)
        teams_today_set = set()
        if not matches.empty:
            day_matches = matches[matches['date'] == date_norm]
            for _, match in day_matches.iterrows():
                for team_col in ['home_team', 'away_team', 'team']:
                    if team_col in match and pd.notna(match[team_col]):
                        team_name = str(match[team_col]).strip()
                        if not is_national_team(team_name):
                            teams_today_set.add(team_name)
        features['teams_today'][i] = len(teams_today_set)
        
        # Calculate cum_teams (cumulative unique teams, excluding national teams)
        features['cum_teams'][i] = len(all_teams_seen)
        
        # Calculate seasons_count (unique football seasons)
        if not matches.empty:
            past_matches = matches[matches['date'] <= date_norm]
            if not past_matches.empty:
                for _, match in past_matches.iterrows():
                    if pd.notna(match.get('date')):
                        season = get_football_season(pd.Timestamp(match['date']))
                        seasons_seen.add(season)
        features['seasons_count'][i] = len(seasons_seen)
    
    logger.info("=== Completed calculate_profile_features ===")
    return pd.DataFrame(features, index=calendar)

def calculate_match_features(
    matches: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    player_row: pd.Series,
    team_country_map: Dict[str, str]
) -> pd.DataFrame:
    """Calculate match-related features for each day."""
    logger.info("=== Starting calculate_match_features ===")
    n_days = len(calendar)
    
    # Initialize all match features - this is a large dictionary
    # I'll create it programmatically to save space
    feature_names = [
        'matches_played', 'minutes_played_numeric', 'goals_numeric', 'assists_numeric',
        'yellow_cards_numeric', 'red_cards_numeric', 'matches_bench_unused',
        'matches_not_selected', 'matches_injured',
        'cum_minutes_played_numeric', 'cum_goals_numeric', 'cum_assists_numeric',
        'cum_yellow_cards_numeric', 'cum_red_cards_numeric', 'cum_matches_played',
        'cum_matches_bench', 'cum_matches_not_selected', 'cum_competitions',
        'goals_per_match', 'assists_per_match', 'minutes_per_match', 'days_since_last_match',
        'matches', 'career_matches', 'career_goals', 'career_assists', 'career_minutes',
        'last_match_position', 'position_match_default', 'competition_importance',
        'avg_competition_importance', 'month', 'disciplinary_action', 'cum_disciplinary_actions',
        'teams_this_season', 'teams_last_season', 'teams_season_today', 'season_team_diversity',
        'national_team_appearances', 'national_team_minutes', 'days_since_last_national_match',
        'national_team_this_season', 'national_team_last_season', 'national_team_frequency',
        'senior_national_team', 'national_team_intensity',
        'competition_intensity', 'competition_level', 'competition_diversity',
        'international_competitions', 'cup_competitions', 'competition_frequency',
        'competition_experience', 'competition_pressure',
        'club_cum_goals', 'club_cum_assists', 'club_cum_minutes', 'club_cum_matches_played',
        'club_cum_yellow_cards', 'club_cum_red_cards', 'club_goals_per_match',
        'club_assists_per_match', 'club_minutes_per_match', 'club_seniority_x_goals_per_match',
        'substitution_on_count', 'substitution_off_count', 'late_substitution_on_count',
        'early_substitution_off_count', 'impact_substitution_count', 'tactical_substitution_count',
        'substitution_minutes_played', 'substitution_efficiency', 'substitution_mood_indicator',
        'consecutive_substitutions',
        'home_matches', 'away_matches', 'team_win', 'team_draw', 'team_loss', 'team_points',
        'cum_team_wins', 'cum_team_draws', 'cum_team_losses', 'team_win_rate',
        'cum_team_points', 'team_points_rolling5', 'team_mood_score', 'cum_matches_injured'
    ]
    
    features = {}
    for name in feature_names:
        if 'position' in name:
            features[name] = [''] * n_days if name == 'last_match_position' else [0] * n_days
        elif 'per_match' in name or 'rate' in name or 'ratio' in name or 'diversity' in name or 'frequency' in name or 'intensity' in name or 'efficiency' in name or 'mood' in name or 'rolling' in name:
            features[name] = [0.0] * n_days
        else:
            features[name] = [0] * n_days
    
    # Get player's default position
    default_position = _normalize_position(player_row.get('position', '')) if 'position' in player_row else ''
    
    # Track cumulative values
    cum_minutes = 0
    cum_goals = 0
    cum_assists = 0
    cum_yellow_cards = 0
    cum_red_cards = 0
    cum_matches_played = 0
    cum_matches_bench = 0
    cum_matches_not_selected = 0
    cum_matches_injured = 0
    cum_disciplinary = 0
    
    # Track last match date
    last_match_date = None
    
    # Track competitions
    competitions_seen = set()
    competitions_this_season = set()
    competitions_last_season = set()
    
    # Track teams by season
    teams_this_season_set = set()
    teams_last_season_set = set()
    current_season = None
    
    # Track national team matches
    last_national_match_date = None
    national_matches_count = 0
    national_minutes_total = 0
    
    # Track club performance (reset on club change)
    current_club = None
    club_cum_goals = 0
    club_cum_assists = 0
    club_cum_minutes = 0
    club_cum_matches = 0
    club_cum_yellow = 0
    club_cum_red = 0
    
    # Track substitution patterns
    consecutive_subs = 0
    last_day_with_substitution = None
    
    # Track team results
    team_wins = 0
    team_draws = 0
    team_losses = 0
    recent_results = []  # For rolling 5
    
    # Progress tracking
    start_time = time.time()
    last_log_time = start_time
    
    for i, date in enumerate(calendar):
        current_time = time.time()
        
        # Log progress
        if i % 1000 == 0 or (current_time - last_log_time) >= 30:
            elapsed = current_time - start_time
            progress_pct = (i * 100) // n_days if n_days > 0 else 0
            if i > 0:
                rate = i / elapsed if elapsed > 0 else 0
                remaining_days = n_days - i
                eta_seconds = remaining_days / rate if rate > 0 else 0
                eta_str = str(timedelta(seconds=int(eta_seconds)))
                elapsed_str = str(timedelta(seconds=int(elapsed)))
                logger.info(f"Match features progress: {i}/{n_days} days ({progress_pct}%) | "
                          f"Elapsed: {elapsed_str} | ETA: {eta_str} | Rate: {rate:.2f} days/sec")
            sys.stdout.flush()
            last_log_time = current_time
        
        date_norm = date.normalize()
        features['month'][i] = date.month
        
        # Get matches on this day
        day_matches = matches[matches['date'] == date_norm] if not matches.empty else pd.DataFrame()
        
        # Track if this day has any substitution
        day_has_substitution = False
        
        if not day_matches.empty:
            for _, match in day_matches.iterrows():
                # Daily features
                features['matches_played'][i] += match.get('matches_played', 0)
                features['minutes_played_numeric'][i] += match.get('minutes_played_numeric', 0)
                features['goals_numeric'][i] += match.get('goals_numeric', 0)
                features['assists_numeric'][i] += match.get('assists_numeric', 0)
                features['yellow_cards_numeric'][i] += match.get('yellow_cards_numeric', 0)
                features['red_cards_numeric'][i] += match.get('red_cards_numeric', 0)
                features['matches_bench_unused'][i] += match.get('matches_bench_unused', 0)
                features['matches_not_selected'][i] += match.get('matches_not_selected', 0)
                features['matches_injured'][i] += match.get('matches_injured', 0)
                
                # Update cumulative
                cum_minutes += match.get('minutes_played_numeric', 0)
                cum_goals += match.get('goals_numeric', 0)
                cum_assists += match.get('assists_numeric', 0)
                cum_yellow_cards += match.get('yellow_cards_numeric', 0)
                cum_red_cards += match.get('red_cards_numeric', 0)
                cum_matches_played += match.get('matches_played', 0)
                cum_matches_bench += match.get('matches_bench_unused', 0)
                cum_matches_not_selected += match.get('matches_not_selected', 0)
                cum_matches_injured += match.get('matches_injured', 0)
                cum_disciplinary += match.get('disciplinary_action', 0)
                
                # Career totals
                features['matches'][i] += 1
                features['career_matches'][i] = features['matches'][i]
                features['career_goals'][i] += match.get('goals_numeric', 0)
                features['career_assists'][i] += match.get('assists_numeric', 0)
                features['career_minutes'][i] += match.get('minutes_played_numeric', 0)
                
                # Last match position
                match_position = match.get('position', '')
                if pd.notna(match_position) and match_position != '':
                    normalized_pos = _normalize_position(match_position)
                    if normalized_pos:
                        features['last_match_position'][i] = normalized_pos
                        if normalized_pos == default_position:
                            features['position_match_default'][i] = 1
                
                # Competition importance
                comp_importance = match.get('competition_importance', 0)
                features['competition_importance'][i] = max(features['competition_importance'][i], comp_importance)
                
                # Disciplinary
                features['disciplinary_action'][i] = max(features['disciplinary_action'][i], match.get('disciplinary_action', 0))
                
                # Competitions tracking
                comp_name = match.get('competition', '')
                if pd.notna(comp_name) and comp_name != '':
                    competitions_seen.add(comp_name)
                    season = get_football_season(date_norm)
                    if season == current_season:
                        competitions_this_season.add(comp_name)
                    elif current_season:
                        prev_season = get_football_season(pd.Timestamp(f"{current_season.split('/')[0]}-07-01") - timedelta(days=365))
                        if season == prev_season:
                            competitions_last_season.add(comp_name)
                
                # National team detection
                home_team = match.get('home_team', '')
                away_team = match.get('away_team', '')
                is_national_match = is_national_team(home_team) or is_national_team(away_team)
                
                if is_national_match:
                    national_matches_count += 1
                    national_minutes_total += match.get('minutes_played_numeric', 0)
                    last_national_match_date = date_norm
                    
                    # Determine if senior or youth
                    comp_lower = str(match.get('competition', '')).lower()
                    is_senior = not any(indicator in comp_lower for indicator in ['u19', 'u20', 'u21', 'u23', 'youth', 'u-'])
                    if is_senior:
                        # Set to 1 if ever played for senior team (cumulative)
                        if features['senior_national_team'][i] == 0:
                            features['senior_national_team'][i] = 1
                
                # Club matching for club features using normalized names
                player_team = identify_player_team(match, current_club)
                
                # Update club stats (reset on club change)
                if player_team:
                    player_team_norm = normalize_team_name(player_team)
                    current_club_norm = normalize_team_name(current_club) if current_club else None
                    
                    if current_club_norm != player_team_norm:
                        # Club changed, reset
                        current_club = player_team
                        club_cum_goals = 0
                        club_cum_assists = 0
                        club_cum_minutes = 0
                        club_cum_matches = 0
                        club_cum_yellow = 0
                        club_cum_red = 0
                    elif current_club is None:
                        # First club assignment
                        current_club = player_team
                    
                    club_cum_goals += match.get('goals_numeric', 0)
                    club_cum_assists += match.get('assists_numeric', 0)
                    club_cum_minutes += match.get('minutes_played_numeric', 0)
                    club_cum_matches += match.get('matches_played', 0)
                    club_cum_yellow += match.get('yellow_cards_numeric', 0)
                    club_cum_red += match.get('red_cards_numeric', 0)
                
                # Substitution features
                sub_on_min = match.get('substitution_on_minute')
                sub_off_min = match.get('substitution_off_minute')
                
                if pd.notna(sub_on_min) or pd.notna(sub_off_min):
                    day_has_substitution = True
                
                if pd.notna(sub_on_min):
                    features['substitution_on_count'][i] += 1
                    if sub_on_min >= 60:  # Late substitution
                        features['late_substitution_on_count'][i] += 1
                    if match.get('goals_numeric', 0) > 0 or match.get('assists_numeric', 0) > 0:
                        features['impact_substitution_count'][i] += 1
                    features['substitution_minutes_played'][i] += (90 - sub_on_min)
                elif pd.notna(sub_off_min):
                    features['substitution_off_count'][i] += 1
                    if sub_off_min <= 60:  # Early substitution
                        features['early_substitution_off_count'][i] += 1
                    if sub_off_min < 70:  # Tactical
                        features['tactical_substitution_count'][i] += 1
                    features['substitution_minutes_played'][i] += sub_off_min
                
                # Location features
                if pd.notna(match.get('home_team')):
                    # Determine if player's team is home or away using normalized names
                    player_team = identify_player_team(match, current_club)
                    
                    if player_team:
                        home_team = match.get('home_team', '')
                        away_team = match.get('away_team', '')
                        
                        # Use normalized comparison
                        if normalize_team_name(player_team) == normalize_team_name(home_team):
                            features['home_matches'][i] += 1
                        elif normalize_team_name(player_team) == normalize_team_name(away_team):
                            features['away_matches'][i] += 1
                        # Fallback to direct comparison
                        elif player_team == home_team:
                            features['home_matches'][i] += 1
                        elif player_team == away_team:
                            features['away_matches'][i] += 1
                
                # Team results
                result = match.get('result', '')
                if pd.notna(result) and result != '':
                    # Parse result (e.g., "2-1", "1-1", "0-3")
                    try:
                        if '-' in result:
                            parts = result.split('-')
                            if len(parts) == 2:
                                home_score = int(parts[0].strip())
                                away_score = int(parts[1].strip())
                                
                                # Determine if player's team won/drew/lost using normalized names
                                player_team = identify_player_team(match, current_club)
                                home_team = match.get('home_team', '')
                                away_team = match.get('away_team', '')
                                
                                if player_team:
                                    # Use normalized comparison
                                    is_home = (normalize_team_name(player_team) == normalize_team_name(home_team)) or (player_team == home_team)
                                    is_away = (normalize_team_name(player_team) == normalize_team_name(away_team)) or (player_team == away_team)
                                    
                                    if is_home:
                                        if home_score > away_score:
                                            features['team_win'][i] += 1
                                            team_wins += 1
                                            features['team_points'][i] += 3
                                            recent_results.append(3)
                                        elif home_score == away_score:
                                            features['team_draw'][i] += 1
                                            team_draws += 1
                                            features['team_points'][i] += 1
                                            recent_results.append(1)
                                        else:
                                            features['team_loss'][i] += 1
                                            team_losses += 1
                                            features['team_points'][i] += 0
                                            recent_results.append(0)
                                    elif is_away:
                                        if away_score > home_score:
                                            features['team_win'][i] += 1
                                            team_wins += 1
                                            features['team_points'][i] += 3
                                            recent_results.append(3)
                                        elif away_score == home_score:
                                            features['team_draw'][i] += 1
                                            team_draws += 1
                                            features['team_points'][i] += 1
                                            recent_results.append(1)
                                        else:
                                            features['team_loss'][i] += 1
                                            team_losses += 1
                                            features['team_points'][i] += 0
                                            recent_results.append(0)
                    except:
                        pass
                
                last_match_date = date_norm
        
        # Set cumulative features
        features['cum_minutes_played_numeric'][i] = cum_minutes
        features['cum_goals_numeric'][i] = cum_goals
        features['cum_assists_numeric'][i] = cum_assists
        features['cum_yellow_cards_numeric'][i] = cum_yellow_cards
        features['cum_red_cards_numeric'][i] = cum_red_cards
        features['cum_matches_played'][i] = cum_matches_played
        features['cum_matches_bench'][i] = cum_matches_bench
        features['cum_matches_not_selected'][i] = cum_matches_not_selected
        features['cum_matches_injured'][i] = cum_matches_injured
        features['cum_disciplinary_actions'][i] = cum_disciplinary
        features['cum_competitions'][i] = len(competitions_seen)
        
        # Per-match ratios
        if cum_matches_played > 0:
            features['goals_per_match'][i] = cum_goals / cum_matches_played
            features['assists_per_match'][i] = cum_assists / cum_matches_played
            features['minutes_per_match'][i] = cum_minutes / cum_matches_played
        
        # Days since last match
        if last_match_date:
            features['days_since_last_match'][i] = (date_norm - last_match_date).days
        
        # Average competition importance
        past_matches = matches[matches['date'] <= date_norm] if not matches.empty else pd.DataFrame()
        if not past_matches.empty and 'competition_importance' in past_matches.columns:
            features['avg_competition_importance'][i] = past_matches['competition_importance'].mean()
        
        # Season-based team features
        season = get_football_season(date_norm)
        if season != current_season:
            current_season = season
            teams_last_season_set = teams_this_season_set.copy()
            teams_this_season_set = set()
            competitions_last_season = competitions_this_season.copy()
            competitions_this_season = set()
        
        # Update teams this season from matches
        past_matches = matches[matches['date'] <= date_norm] if not matches.empty else pd.DataFrame()
        if not past_matches.empty:
            season_matches = past_matches[past_matches['date'].apply(lambda d: get_football_season(pd.Timestamp(d)) == season)]
            for _, match in season_matches.iterrows():
                for team_col in ['home_team', 'away_team', 'team']:
                    if team_col in match and pd.notna(match[team_col]):
                        team_name = str(match[team_col]).strip()
                        if not is_national_team(team_name):
                            teams_this_season_set.add(team_name)
        
        features['teams_this_season'][i] = len(teams_this_season_set)
        features['teams_last_season'][i] = len(teams_last_season_set)
        features['teams_season_today'][i] = len(teams_this_season_set)
        if len(teams_this_season_set) > 0 and not past_matches.empty:
            season_matches = past_matches[past_matches['date'].apply(lambda d: get_football_season(pd.Timestamp(d)) == season)]
            if len(season_matches) > 0:
                features['season_team_diversity'][i] = len(teams_this_season_set) / len(season_matches)
        
        # National team features
        features['national_team_appearances'][i] = national_matches_count
        features['national_team_minutes'][i] = national_minutes_total
        if last_national_match_date:
            features['days_since_last_national_match'][i] = (date_norm - last_national_match_date).days
        
        # National team by season
        if not past_matches.empty:
            season_national = past_matches[
                past_matches.apply(lambda m: is_national_team(m.get('home_team', '')) or is_national_team(m.get('away_team', '')), axis=1)
            ]
            season_national_this = season_national[season_national['date'].apply(lambda d: get_football_season(pd.Timestamp(d)) == season)]
            prev_season = get_football_season(pd.Timestamp(f"{season.split('/')[0]}-07-01") - timedelta(days=365))
            season_national_last = season_national[season_national['date'].apply(lambda d: get_football_season(pd.Timestamp(d)) == prev_season)]
            features['national_team_this_season'][i] = len(season_national_this)
            features['national_team_last_season'][i] = len(season_national_last)
        
        # National team frequency
        days_span = (date_norm - calendar[0]).days + 1
        years_span = days_span / 365.25
        if years_span > 0:
            features['national_team_frequency'][i] = national_matches_count / years_span
        
        # National team intensity (average minutes per appearance)
        if national_matches_count > 0:
            features['national_team_intensity'][i] = national_minutes_total / national_matches_count
        
        # Competition analysis
        if not past_matches.empty:
            comp_importances = past_matches['competition_importance'].dropna()
            if len(comp_importances) > 0:
                features['competition_intensity'][i] = comp_importances.mean()
                features['competition_level'][i] = comp_importances.max()
            
            unique_comps = past_matches['competition'].nunique()
            total_matches = len(past_matches)
            if total_matches > 0:
                features['competition_diversity'][i] = unique_comps / total_matches
            
            # International and cup competitions
            comp_names = past_matches['competition'].astype(str).str.lower()
            features['international_competitions'][i] = comp_names.str.contains('champions|europa|world cup|euro', case=False, na=False).sum()
            features['cup_competitions'][i] = comp_names.str.contains('cup|trophy', case=False, na=False).sum()
            
            # Competition frequency
            if years_span > 0:
                features['competition_frequency'][i] = total_matches / years_span
            
            # Competition experience (unique competitions)
            features['competition_experience'][i] = unique_comps
            
            # Competition pressure (high importance matches)
            high_importance = (past_matches['competition_importance'] >= 4).sum()
            if total_matches > 0:
                features['competition_pressure'][i] = high_importance / total_matches
        
        # Club performance features
        features['club_cum_goals'][i] = club_cum_goals
        features['club_cum_assists'][i] = club_cum_assists
        features['club_cum_minutes'][i] = club_cum_minutes
        features['club_cum_matches_played'][i] = club_cum_matches
        features['club_cum_yellow_cards'][i] = club_cum_yellow
        features['club_cum_red_cards'][i] = club_cum_red
        
        if club_cum_matches > 0:
            features['club_goals_per_match'][i] = club_cum_goals / club_cum_matches
            features['club_assists_per_match'][i] = club_cum_assists / club_cum_matches
            features['club_minutes_per_match'][i] = club_cum_minutes / club_cum_matches
        
        # Update consecutive substitutions based on day
        if day_has_substitution:
            consecutive_subs += 1
            last_day_with_substitution = date_norm
        else:
            # Reset if no substitution this day (only if we had a previous substitution day)
            if last_day_with_substitution is not None and (date_norm - last_day_with_substitution).days > 1:
                consecutive_subs = 0
                last_day_with_substitution = None
        
        features['consecutive_substitutions'][i] = consecutive_subs
        
        # Substitution features (continued)
        if features['substitution_on_count'][i] > 0 or features['substitution_off_count'][i] > 0:
            total_subs = features['substitution_on_count'][i] + features['substitution_off_count'][i]
            if total_subs > 0:
                features['substitution_efficiency'][i] = (features['goals_numeric'][i] + features['assists_numeric'][i]) / total_subs
        
        # Substitution mood (positive if goals/assists, negative if early off)
        if features['substitution_off_count'][i] > 0 and features['early_substitution_off_count'][i] > 0:
            features['substitution_mood_indicator'][i] = -1.0
        elif features['substitution_on_count'][i] > 0 and features['impact_substitution_count'][i] > 0:
            features['substitution_mood_indicator'][i] = 1.0
        
        # Team results (cumulative)
        features['cum_team_wins'][i] = team_wins
        features['cum_team_draws'][i] = team_draws
        features['cum_team_losses'][i] = team_losses
        total_results = team_wins + team_draws + team_losses
        if total_results > 0:
            features['team_win_rate'][i] = team_wins / total_results
        features['cum_team_points'][i] = team_wins * 3 + team_draws
        
        # Rolling 5 results
        if len(recent_results) > 5:
            recent_results = recent_results[-5:]
        if len(recent_results) > 0:
            features['team_points_rolling5'][i] = sum(recent_results) / len(recent_results)
        
        # Team mood score (weighted recent performance)
        if len(recent_results) > 0:
            weights = [0.3, 0.25, 0.2, 0.15, 0.1][:len(recent_results)]
            weighted_sum = sum(r * w for r, w in zip(recent_results, weights))
            features['team_mood_score'][i] = weighted_sum
    
    # Make senior_national_team cumulative (once 1, always 1)
    for i in range(1, n_days):
        if features['senior_national_team'][i-1] == 1:
            features['senior_national_team'][i] = 1
    
    logger.info("=== Completed calculate_match_features ===")
    return pd.DataFrame(features, index=calendar)

def calculate_injury_features(injuries: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Calculate comprehensive injury features for ALL injury classes.
    NO TARGET/FEATURE DISTINCTION - all injury classes are treated as features.
    """
    logger.info("=== Starting calculate_injury_features ===")
    sys.stdout.flush()
    
    try:
        n_days = len(calendar)
        logger.info(f"Processing {n_days} days for injury features...")
        
        # Process ALL injuries (no distinction between target/feature)
        logger.info(f"Total injuries: {len(injuries)}")
        all_injuries = injuries.copy()
        
        if all_injuries.empty:
            logger.info("No injuries found, returning empty features")
            # Return empty DataFrame with all feature columns initialized
            features = {
                # Base features (all injuries)
                'cum_inj_starts': [0] * n_days,
                'cum_inj_days': [0] * n_days,
                'days_since_last_injury': [999] * n_days,
                'days_since_last_injury_ended': [999] * n_days,
                'avg_injury_duration': [0.0] * n_days,
                'injury_frequency': [0.0] * n_days,
                'avg_injury_severity': [0.0] * n_days,
                'max_injury_severity': [0] * n_days,
                'short_term_injury_ratio': [0.0] * n_days,
                
                # By injury class (ALL classes treated equally)
                'muscular_injury_count': [0] * n_days,
                'skeletal_injury_count': [0] * n_days,
                'unknown_injury_count': [0] * n_days,
                'other_injury_count': [0] * n_days,
                'muscular_injury_days': [0] * n_days,
                'skeletal_injury_days': [0] * n_days,
                'unknown_injury_days': [0] * n_days,
                'other_injury_days': [0] * n_days,
                'days_since_last_muscular': [999] * n_days,
                'days_since_last_skeletal': [999] * n_days,
                'days_since_last_unknown': [999] * n_days,
                'days_since_last_other': [999] * n_days,
                
                # By body part (all injuries)
                'lower_leg_injury_count': [0] * n_days,
                'knee_injury_count': [0] * n_days,
                'upper_leg_injury_count': [0] * n_days,
                'hip_injury_count': [0] * n_days,
                'upper_body_injury_count': [0] * n_days,
                'head_injury_count': [0] * n_days,
                'illness_count': [0] * n_days,
                'unknown_body_part_count': [0] * n_days,
                'days_since_last_lower_leg': [999] * n_days,
                'days_since_last_knee': [999] * n_days,
                'days_since_last_upper_leg': [999] * n_days,
                'days_since_last_hip': [999] * n_days,
                'days_since_last_upper_body': [999] * n_days,
                'days_since_last_head': [999] * n_days,
                'days_since_last_illness': [999] * n_days,
                
                # By severity (all injuries)
                'mild_injury_count': [0] * n_days,
                'moderate_injury_count': [0] * n_days,
                'severe_injury_count': [0] * n_days,
                'critical_injury_count': [0] * n_days,
                'days_since_last_mild': [999] * n_days,
                'days_since_last_moderate': [999] * n_days,
                'days_since_last_severe': [999] * n_days,
                'days_since_last_critical': [999] * n_days,
                
                # Combined: class + body part
                'muscular_lower_leg_count': [0] * n_days,
                'muscular_knee_count': [0] * n_days,
                'skeletal_lower_leg_count': [0] * n_days,
                'skeletal_knee_count': [0] * n_days,
                
                # Combined: class + severity
                'muscular_mild_count': [0] * n_days,
                'muscular_moderate_count': [0] * n_days,
                'muscular_severe_count': [0] * n_days,
                'muscular_critical_count': [0] * n_days,
                'skeletal_mild_count': [0] * n_days,
                'skeletal_moderate_count': [0] * n_days,
                'skeletal_severe_count': [0] * n_days,
                'skeletal_critical_count': [0] * n_days,
                
                # Combined: body part + severity
                'mild_lower_leg_count': [0] * n_days,
                'moderate_lower_leg_count': [0] * n_days,
                'severe_lower_leg_count': [0] * n_days,
                'critical_lower_leg_count': [0] * n_days,
                'mild_knee_count': [0] * n_days,
                'moderate_knee_count': [0] * n_days,
                'severe_knee_count': [0] * n_days,
                'critical_knee_count': [0] * n_days,
                
                # Ratios (by class to total)
                'muscular_to_total_ratio': [0.0] * n_days,
                'skeletal_to_total_ratio': [0.0] * n_days,
                'unknown_to_total_ratio': [0.0] * n_days,
                'other_to_total_ratio': [0.0] * n_days,
            }
            return pd.DataFrame(features, index=calendar)
        
        # Add numeric severity
        logger.info("Mapping severity to numeric...")
        all_injuries['severity_numeric'] = all_injuries['severity'].apply(map_severity_to_numeric)
        
        # Sort by fromDate for efficient processing
        logger.info("Sorting injuries by date...")
        all_injuries = all_injuries.sort_values('fromDate').reset_index(drop=True)
        
        # Pre-calculate injury periods
        logger.info("Normalizing injury dates...")
        all_injuries['start_norm'] = all_injuries['fromDate'].apply(lambda x: x.normalize() if pd.notna(x) else pd.NaT)
        all_injuries['end_norm'] = all_injuries['untilDate'].apply(lambda x: x.normalize() if pd.notna(x) else pd.NaT)
        logger.info("Injury dates normalized")
        sys.stdout.flush()
        
        # Initialize features
        features = {
            # Base features (all injuries)
            'cum_inj_starts': [0] * n_days,
            'cum_inj_days': [0] * n_days,
            'days_since_last_injury': [999] * n_days,
            'days_since_last_injury_ended': [999] * n_days,
            'avg_injury_duration': [0.0] * n_days,
            'injury_frequency': [0.0] * n_days,
            'avg_injury_severity': [0.0] * n_days,
            'max_injury_severity': [0] * n_days,
            'short_term_injury_ratio': [0.0] * n_days,
            
            # By injury class (ALL classes treated equally - NO TARGET/FEATURE distinction)
            'muscular_injury_count': [0] * n_days,
            'skeletal_injury_count': [0] * n_days,
            'unknown_injury_count': [0] * n_days,
            'other_injury_count': [0] * n_days,
            'muscular_injury_days': [0] * n_days,
            'skeletal_injury_days': [0] * n_days,
            'unknown_injury_days': [0] * n_days,
            'other_injury_days': [0] * n_days,
            'days_since_last_muscular': [999] * n_days,
            'days_since_last_skeletal': [999] * n_days,
            'days_since_last_unknown': [999] * n_days,
            'days_since_last_other': [999] * n_days,
            
            # By body part (all injuries)
            'lower_leg_injury_count': [0] * n_days,
            'knee_injury_count': [0] * n_days,
            'upper_leg_injury_count': [0] * n_days,
            'hip_injury_count': [0] * n_days,
            'upper_body_injury_count': [0] * n_days,
            'head_injury_count': [0] * n_days,
            'illness_count': [0] * n_days,
            'unknown_body_part_count': [0] * n_days,
            'days_since_last_lower_leg': [999] * n_days,
            'days_since_last_knee': [999] * n_days,
            'days_since_last_upper_leg': [999] * n_days,
            'days_since_last_hip': [999] * n_days,
            'days_since_last_upper_body': [999] * n_days,
            'days_since_last_head': [999] * n_days,
            'days_since_last_illness': [999] * n_days,
            
            # By severity (all injuries)
            'mild_injury_count': [0] * n_days,
            'moderate_injury_count': [0] * n_days,
            'severe_injury_count': [0] * n_days,
            'critical_injury_count': [0] * n_days,
            'days_since_last_mild': [999] * n_days,
            'days_since_last_moderate': [999] * n_days,
            'days_since_last_severe': [999] * n_days,
            'days_since_last_critical': [999] * n_days,
            
            # Combined: class + body part
            'muscular_lower_leg_count': [0] * n_days,
            'muscular_knee_count': [0] * n_days,
            'skeletal_lower_leg_count': [0] * n_days,
            'skeletal_knee_count': [0] * n_days,
            
            # Combined: class + severity
            'muscular_mild_count': [0] * n_days,
            'muscular_moderate_count': [0] * n_days,
            'muscular_severe_count': [0] * n_days,
            'muscular_critical_count': [0] * n_days,
            'skeletal_mild_count': [0] * n_days,
            'skeletal_moderate_count': [0] * n_days,
            'skeletal_severe_count': [0] * n_days,
            'skeletal_critical_count': [0] * n_days,
            
            # Combined: body part + severity
            'mild_lower_leg_count': [0] * n_days,
            'moderate_lower_leg_count': [0] * n_days,
            'severe_lower_leg_count': [0] * n_days,
            'critical_lower_leg_count': [0] * n_days,
            'mild_knee_count': [0] * n_days,
            'moderate_knee_count': [0] * n_days,
            'severe_knee_count': [0] * n_days,
            'critical_knee_count': [0] * n_days,
            
            # Ratios (by class to total)
            'muscular_to_total_ratio': [0.0] * n_days,
            'skeletal_to_total_ratio': [0.0] * n_days,
            'unknown_to_total_ratio': [0.0] * n_days,
            'other_to_total_ratio': [0.0] * n_days,
        }
        
        # Track last injury dates by various dimensions
        last_injury_date = None
        last_recovery_date = None
        last_by_class = {'muscular': None, 'skeletal': None, 'unknown': None, 'other': None}
        last_by_body_part = {
            'lower_leg': None, 'knee': None, 'upper_leg': None, 'hip': None,
            'upper_body': None, 'head': None, 'illness': None
        }
        last_by_severity = {1: None, 2: None, 3: None, 4: None}  # mild, moderate, severe, critical
        
        # Use cumulative approach - much faster
        injury_idx = 0
        logger.info(f"Starting daily loop for {n_days} days...")
        sys.stdout.flush()
        
        # Progress tracking
        start_time = time.time()
        last_log_time = start_time
        
        for i, date in enumerate(calendar):
            current_time = time.time()
            
            # Log progress every 1000 days or every 30 seconds
            if i % 1000 == 0 or (current_time - last_log_time) >= 30:
                elapsed = current_time - start_time
                progress_pct = (i * 100) // n_days if n_days > 0 else 0
                
                if i > 0:
                    rate = i / elapsed if elapsed > 0 else 0
                    remaining_days = n_days - i
                    eta_seconds = remaining_days / rate if rate > 0 else 0
                    eta_str = str(timedelta(seconds=int(eta_seconds)))
                    elapsed_str = str(timedelta(seconds=int(elapsed)))
                    
                    logger.info(f"Injury features progress: {i}/{n_days} days ({progress_pct}%) | "
                              f"Elapsed: {elapsed_str} | ETA: {eta_str} | "
                              f"Rate: {rate:.2f} days/sec")
                else:
                    logger.info(f"Injury features progress: {i}/{n_days} days ({progress_pct}%) | Starting...")
                
                sys.stdout.flush()
                last_log_time = current_time
            
            date_norm = date.normalize()
            
            # Advance injury index to include all injuries up to this date (NO LEAKAGE)
            while injury_idx < len(all_injuries) and all_injuries.iloc[injury_idx]['start_norm'] <= date_norm:
                inj = all_injuries.iloc[injury_idx]
                injury_idx += 1
                
                # Track last injury dates
                last_injury_date = inj['start_norm']
                
                # Track by class (handle NaN)
                injury_class = str(inj.get('injury_class', '')).lower() if pd.notna(inj.get('injury_class')) else ''
                if injury_class in last_by_class:
                    last_by_class[injury_class] = inj['start_norm']
                
                # Track by body part (handle NaN)
                body_part = str(inj.get('body_part', '')).lower() if pd.notna(inj.get('body_part')) else ''
                if body_part in last_by_body_part:
                    last_by_body_part[body_part] = inj['start_norm']
                
                # Track by severity
                severity_num = inj.get('severity_numeric', 1)
                if severity_num in last_by_severity:
                    last_by_severity[severity_num] = inj['start_norm']
                
                # Track recovery
                if pd.notna(inj['end_norm']) and inj['end_norm'] <= date_norm:
                    last_recovery_date = inj['end_norm']
            
            # Get all injuries up to this date (already sorted) - NO FUTURE DATA
            past_injuries = all_injuries.iloc[:injury_idx]
            
            if len(past_injuries) > 0:
                # Base counts
                features['cum_inj_starts'][i] = len(past_injuries)
                
                # Injury days - calculate total days injured up to current date
                injury_days = 0
                for _, inj in past_injuries.iterrows():
                    start = inj['start_norm']
                    end = inj['end_norm'] if pd.notna(inj['end_norm']) else date_norm
                    if end > date_norm:
                        end = date_norm  # Cap at current date (no leakage)
                    if start <= date_norm:
                        injury_days += (end - start).days + 1
                features['cum_inj_days'][i] = injury_days
                
                # Severity - vectorized
                severity_values = past_injuries['severity_numeric'].dropna()
                if len(severity_values) > 0:
                    features['avg_injury_severity'][i] = float(severity_values.mean())
                    features['max_injury_severity'][i] = int(severity_values.max())
                else:
                    features['avg_injury_severity'][i] = 0.0
                    features['max_injury_severity'][i] = 0
                
                # By injury class (handle NaN) - ALL classes treated equally
                for class_name in ['muscular', 'skeletal', 'unknown', 'other']:
                    class_injuries = past_injuries[
                        past_injuries['injury_class'].astype(str).str.lower().fillna('') == class_name
                    ]
                    features[f'{class_name}_injury_count'][i] = len(class_injuries)
                    
                    # Injury days by class
                    class_days = 0
                    for _, inj in class_injuries.iterrows():
                        start = inj['start_norm']
                        end = inj['end_norm'] if pd.notna(inj['end_norm']) else date_norm
                        if end > date_norm:
                            end = date_norm
                        if start <= date_norm:
                            class_days += (end - start).days + 1
                    features[f'{class_name}_injury_days'][i] = class_days
                
                # By body part (handle NaN)
                body_parts = ['lower_leg', 'knee', 'upper_leg', 'hip', 'upper_body', 'head', 'illness']
                for body_part in body_parts:
                    bp_injuries = past_injuries[
                        past_injuries['body_part'].astype(str).str.lower().fillna('') == body_part
                    ]
                    features[f'{body_part}_injury_count'][i] = len(bp_injuries)
                
                # Unknown body part count
                features['unknown_body_part_count'][i] = past_injuries['body_part'].isna().sum()
                
                # By severity
                for severity_num, severity_name in [(1, 'mild'), (2, 'moderate'), (3, 'severe'), (4, 'critical')]:
                    sev_injuries = past_injuries[past_injuries['severity_numeric'] == severity_num]
                    features[f'{severity_name}_injury_count'][i] = len(sev_injuries)
                
                # Combined: class + body part (handle NaN)
                features['muscular_lower_leg_count'][i] = len(past_injuries[
                    (past_injuries['injury_class'].astype(str).str.lower().fillna('') == 'muscular') & 
                    (past_injuries['body_part'].astype(str).str.lower().fillna('') == 'lower_leg')
                ])
                features['muscular_knee_count'][i] = len(past_injuries[
                    (past_injuries['injury_class'].astype(str).str.lower().fillna('') == 'muscular') & 
                    (past_injuries['body_part'].astype(str).str.lower().fillna('') == 'knee')
                ])
                features['skeletal_lower_leg_count'][i] = len(past_injuries[
                    (past_injuries['injury_class'].astype(str).str.lower().fillna('') == 'skeletal') & 
                    (past_injuries['body_part'].astype(str).str.lower().fillna('') == 'lower_leg')
                ])
                features['skeletal_knee_count'][i] = len(past_injuries[
                    (past_injuries['injury_class'].astype(str).str.lower().fillna('') == 'skeletal') & 
                    (past_injuries['body_part'].astype(str).str.lower().fillna('') == 'knee')
                ])
                
                # Combined: class + severity
                for class_name in ['muscular', 'skeletal']:
                    for severity_num, severity_name in [(1, 'mild'), (2, 'moderate'), (3, 'severe'), (4, 'critical')]:
                        features[f'{class_name}_{severity_name}_count'][i] = len(past_injuries[
                            (past_injuries['injury_class'].astype(str).str.lower().fillna('') == class_name) & 
                            (past_injuries['severity_numeric'] == severity_num)
                        ])
                
                # Combined: body part + severity
                for body_part in ['lower_leg', 'knee']:
                    for severity_num, severity_name in [(1, 'mild'), (2, 'moderate'), (3, 'severe'), (4, 'critical')]:
                        features[f'{severity_name}_{body_part}_count'][i] = len(past_injuries[
                            (past_injuries['body_part'].astype(str).str.lower().fillna('') == body_part) & 
                            (past_injuries['severity_numeric'] == severity_num)
                        ])
                
                # Average duration
                if 'days' in past_injuries.columns:
                    valid_durations = past_injuries['days'].dropna()
                    if len(valid_durations) > 0:
                        features['avg_injury_duration'][i] = valid_durations.mean()
                
                # Injury frequency
                days_span = (date_norm - calendar[0]).days + 1
                years_span = days_span / 365.25
                if years_span > 0:
                    features['injury_frequency'][i] = len(past_injuries) / years_span
                
                # Short-term injury ratio (<=7 days)
                if 'days' in past_injuries.columns:
                    short_term = past_injuries[past_injuries['days'] <= 7]
                    if len(past_injuries) > 0:
                        features['short_term_injury_ratio'][i] = len(short_term) / len(past_injuries)
                
                # Ratios by class (to total)
                total_count = len(past_injuries)
                if total_count > 0:
                    for class_name in ['muscular', 'skeletal', 'unknown', 'other']:
                        class_count = features[f'{class_name}_injury_count'][i]
                        features[f'{class_name}_to_total_ratio'][i] = class_count / total_count
                
                # Recency features - days since last injury
                if last_injury_date is not None:
                    features['days_since_last_injury'][i] = (date_norm - last_injury_date).days
                
                if last_recovery_date is not None and last_recovery_date <= date_norm:
                    features['days_since_last_injury_ended'][i] = (date_norm - last_recovery_date).days
                
                # Recency by class
                for class_name in ['muscular', 'skeletal', 'unknown', 'other']:
                    if last_by_class[class_name] is not None:
                        features[f'days_since_last_{class_name}'][i] = (date_norm - last_by_class[class_name]).days
                
                # Recency by body part
                for body_part in ['lower_leg', 'knee', 'upper_leg', 'hip', 'upper_body', 'head', 'illness']:
                    if last_by_body_part[body_part] is not None:
                        features[f'days_since_last_{body_part}'][i] = (date_norm - last_by_body_part[body_part]).days
                
                # Recency by severity
                for severity_num, severity_name in [(1, 'mild'), (2, 'moderate'), (3, 'severe'), (4, 'critical')]:
                    if last_by_severity[severity_num] is not None:
                        features[f'days_since_last_{severity_name}'][i] = (date_norm - last_by_severity[severity_num]).days
        
        logger.info("Building injury DataFrame...")
        result_df = pd.DataFrame(features, index=calendar)
        logger.info(f"=== Completed calculate_injury_features: {len(result_df)} rows, {len(result_df.columns)} columns ===")
        sys.stdout.flush()
        return result_df
    except Exception as e:
        logger.error(f"Error in calculate_injury_features at day {i if 'i' in locals() else 'unknown'}: {e}")
        logger.error(traceback.format_exc())
        sys.stdout.flush()
        raise

def calculate_interaction_features(
    profile_df: pd.DataFrame,
    match_df: pd.DataFrame,
    injury_df: pd.DataFrame
) -> pd.DataFrame:
    """Calculate interaction features."""
    logger.info("=== Starting calculate_interaction_features ===")
    
    features = {}
    
    # Age x career matches
    if 'age' in profile_df.columns and 'career_matches' in match_df.columns:
        features['age_x_career_matches'] = profile_df['age'] * match_df['career_matches']
    else:
        features['age_x_career_matches'] = [0.0] * len(profile_df)
    
    # Age x career goals
    if 'age' in profile_df.columns and 'career_goals' in match_df.columns:
        features['age_x_career_goals'] = profile_df['age'] * match_df['career_goals']
    else:
        features['age_x_career_goals'] = [0.0] * len(profile_df)
    
    # Seniority x goals per match
    if 'seniority_days' in profile_df.columns and 'goals_per_match' in match_df.columns:
        features['seniority_x_goals_per_match'] = profile_df['seniority_days'] * match_df['goals_per_match']
    else:
        features['seniority_x_goals_per_match'] = [0.0] * len(profile_df)
    
    logger.info("=== Completed calculate_interaction_features ===")
    return pd.DataFrame(features, index=profile_df.index)

def generate_daily_features_for_player(
    player_id: int,
    data_dir: str = DATA_DIR,
    reference_date: pd.Timestamp = REFERENCE_DATE,
    output_dir: str = OUTPUT_DIR
) -> pd.DataFrame:
    """Generate daily features for a single player."""
    logger.info(f"=== Generating daily features for player {player_id} ===")
    
    # Load data
    data = load_player_data(player_id, data_dir)
    players = data['players']
    injuries = data['injuries']
    matches = data['matches']
    career = data['career']
    team_country_map = data['team_country_map']
    
    if players.empty:
        logger.error(f"Player {player_id} not found")
        return pd.DataFrame()
    
    player_row = players.iloc[0]
    
    # Preprocess matches
    matches = preprocess_matches(matches)
    
    # Determine calendar
    calendar = determine_calendar(matches, injuries, reference_date)
    
    # Calculate features
    profile_df = calculate_profile_features(player_row, calendar, matches, career, team_country_map)
    match_df = calculate_match_features(matches, calendar, player_row, team_country_map)
    injury_df = calculate_injury_features(injuries, calendar)  # Single consolidated function
    interaction_df = calculate_interaction_features(profile_df, match_df, injury_df)
    
    # Combine all features
    daily_features = pd.concat([profile_df, match_df, injury_df, interaction_df], axis=1)
    
    # Remove duplicate columns
    daily_features = daily_features.loc[:, ~daily_features.columns.duplicated()]
    
    logger.info(f"=== Completed: {len(daily_features)} rows, {len(daily_features.columns)} columns ===")
    
    return daily_features

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Generate daily features for football players')
    parser.add_argument('--player-id', type=int, required=True, help='Player ID to process')
    parser.add_argument('--data-dir', type=str, default=DATA_DIR, help='Data directory')
    parser.add_argument('--reference-date', type=str, default=str(REFERENCE_DATE.date()), help='Reference date (YYYY-MM-DD)')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR, help='Output directory')
    parser.add_argument('--force', action='store_true', help='Force regeneration')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Output file path
    output_file = os.path.join(args.output_dir, f'player_{args.player_id}_daily_features.csv')
    
    # Check if file exists
    if os.path.exists(output_file) and not args.force:
        logger.info(f"File already exists: {output_file}")
        logger.info("Use --force to regenerate")
        return
    
    # Parse reference date
    reference_date = pd.Timestamp(args.reference_date)
    
    # Generate features
    logger.info(f"Generating daily features for player {args.player_id}...")
    start_time = time.time()
    
    try:
        daily_features = generate_daily_features_for_player(
            player_id=args.player_id,
            data_dir=args.data_dir,
            reference_date=reference_date,
            output_dir=args.output_dir
        )
        
        if daily_features.empty:
            logger.error("No features generated")
            return
        
        # Save to CSV
        daily_features.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Successfully generated features for player {args.player_id}")
        logger.info(f"   Output: {output_file}")
        logger.info(f"   Rows: {len(daily_features)}, Columns: {len(daily_features.columns)}")
        logger.info(f"   Time: {elapsed:.2f} seconds")
        
    except Exception as e:
        logger.error(f"❌ Error generating features: {e}")
        logger.error(traceback.format_exc())
        raise

if __name__ == '__main__':
    main()
