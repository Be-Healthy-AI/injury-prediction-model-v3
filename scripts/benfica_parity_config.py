#!/usr/bin/env python3
"""
Benfica Parity Configuration
Configuration and functions to match historical Benfica pipeline behavior
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional

# Benfica Parity Configuration
BENFICA_PARITY_CONFIG = {
    'use_historical_detection': True,
    'competition_importance_default': 1.0,  # Default to 1.0, not 0
    'season_phase_default': 3.0,  # Default to 3.0, not 0
    'age_calculation_precision': 'historical',
    'national_team_detection': 'simplified',
    'enhanced_features_dynamic': True,
    'reproduce_historical_zeros': False,  # Set to False to calculate properly
    'injury_severity_calculation': True,  # Enable injury severity calculation
    'national_team_calculation': True,  # Enable national team features calculation
    'complex_derived_features': True  # Enable complex derived features calculation
}

def is_national_team_benfica_parity(team_name: str) -> bool:
    """
    Simplified national team detection matching historical Benfica logic.
    This is likely what the historical pipeline used - much simpler than the current enhanced version.
    """
    if pd.isna(team_name) or not team_name:
        return False
    
    team_lower = str(team_name).lower().strip()
    
    # Simple country name detection (likely what historical pipeline used)
    # Focus on major football nations that would appear in the dataset
    country_names = [
        'brazil', 'portugal', 'spain', 'france', 'germany', 'italy', 'england',
        'argentina', 'colombia', 'uruguay', 'chile', 'mexico', 'netherlands',
        'belgium', 'croatia', 'poland', 'sweden', 'norway', 'denmark', 'switzerland',
        'austria', 'czech', 'slovakia', 'hungary', 'romania', 'bulgaria', 'serbia',
        'montenegro', 'bosnia', 'albania', 'macedonia', 'slovenia', 'croatia',
        'ukraine', 'russia', 'turkey', 'greece', 'cyprus', 'israel', 'wales',
        'scotland', 'ireland', 'northern ireland', 'finland', 'iceland', 'estonia',
        'latvia', 'lithuania', 'luxembourg', 'malta', 'andorra', 'san marino',
        'liechtenstein', 'moldova', 'belarus', 'georgia', 'armenia', 'azerbaijan',
        'kazakhstan', 'kyrgyzstan', 'tajikistan', 'uzbekistan', 'turkmenistan',
        'peru', 'bolivia', 'ecuador', 'venezuela', 'paraguay', 'guyana', 'suriname',
        'canada', 'united states', 'usa', 'costa rica', 'guatemala', 'honduras',
        'el salvador', 'nicaragua', 'panama', 'jamaica', 'trinidad', 'barbados',
        'cuba', 'haiti', 'dominican republic', 'puerto rico', 'algeria', 'egypt',
        'libya', 'tunisia', 'morocco', 'sudan', 'ethiopia', 'kenya', 'uganda',
        'tanzania', 'rwanda', 'burundi', 'democratic republic of congo', 'congo',
        'central african republic', 'chad', 'cameroon', 'nigeria', 'niger',
        'mali', 'burkina faso', 'senegal', 'gambia', 'guinea', 'sierra leone',
        'liberia', 'ghana', 'togo', 'benin', 'cote d\'ivoire', 'guinea-bissau',
        'cape verde', 'sao tome and principe', 'equatorial guinea', 'gabon',
        'angola', 'zambia', 'zimbabwe', 'botswana', 'namibia', 'south africa',
        'lesotho', 'swaziland', 'madagascar', 'mauritius', 'seychelles', 'comoros',
        'djibouti', 'somalia', 'eritrea', 'south sudan', 'china', 'japan',
        'south korea', 'north korea', 'mongolia', 'taiwan', 'hong kong', 'macau',
        'vietnam', 'laos', 'cambodia', 'thailand', 'myanmar', 'malaysia',
        'singapore', 'indonesia', 'philippines', 'brunei', 'east timor', 'australia',
        'new zealand', 'fiji', 'papua new guinea', 'solomon islands', 'vanuatu',
        'samoa', 'tonga', 'cook islands', 'tahiti', 'india', 'pakistan',
        'bangladesh', 'sri lanka', 'maldives', 'nepal', 'bhutan', 'afghanistan',
        'iran', 'iraq', 'syria', 'lebanon', 'jordan', 'israel', 'palestine',
        'saudi arabia', 'yemen', 'oman', 'uae', 'qatar', 'bahrain', 'kuwait'
    ]
    
    # Check if the team name contains a country name
    for country in country_names:
        if country in team_lower:
            return True
    
    # Check for common national team indicators
    national_indicators = [
        'national', 'country', 'selection', 'team', 'olympic', 'world cup',
        'euro', 'copa america', 'africa cup', 'asia cup', 'gold cup'
    ]
    
    return any(indicator in team_lower for indicator in national_indicators)

def map_competition_importance_benfica_parity(competition: str) -> int:
    """
    Competition importance mapping matching historical Benfica logic.
    Uses 0 for no match instead of 1, and simplified mapping rules.
    """
    if pd.isna(competition) or not competition:
        return 0  # No match = 0, not 1
    
    comp_lower = str(competition).lower().strip()
    
    # Simplified mapping (likely what historical pipeline used)
    if any(term in comp_lower for term in ['champions league', 'world cup', 'euro', 'european championship']):
        return 5
    elif any(term in comp_lower for term in ['europa league', 'conference league', 'copa america', 'africa cup']):
        return 4
    elif any(term in comp_lower for term in ['liga', 'premier league', 'bundesliga', 'serie a', 'la liga', 'primeira liga']):
        return 3
    elif any(term in comp_lower for term in ['cup', 'trophy', 'super cup', 'taça']):
        return 2
    else:
        return 1  # Default for other competitions

def map_season_phase_benfica_parity(date: pd.Timestamp) -> int:
    """
    Season phase mapping matching historical Benfica logic.
    Uses 0 for no match instead of 3, and simplified season phases.
    """
    if pd.isna(date):
        return 0  # No match = 0, not 3
    
    month = date.month
    
    # Simplified season phases (likely what historical pipeline used)
    if month in [7, 8]:  # Pre-season
        return 1
    elif month in [9, 10, 11, 12]:  # First half of season
        return 2
    elif month in [1, 2, 3, 4, 5]:  # Second half of season
        return 3
    else:  # June - end of season
        return 4

def calculate_age_benfica_parity(birth_date: pd.Timestamp, current_date: pd.Timestamp) -> float:
    """
    Age calculation matching historical Benfica precision.
    Uses exact same calculation as reference to avoid floating point differences.
    """
    if pd.isna(birth_date) or pd.isna(current_date):
        return 0.0
    
    # Use exact same calculation as reference
    days_diff = (current_date - birth_date).days
    return days_diff / 365.25

def detect_disciplinary_action_benfica_parity(row: pd.Series) -> int:
    """
    Disciplinary action detection matching historical Benfica logic.
    Simplified version of the enhanced detection.
    """
    if row.empty:
        return 0
    
    # Check position field for disciplinary indicators
    position = str(row.get('position', '')).lower()
    disciplinary_indicators = [
        'suspended', 'banned', 'punished', 'disciplinary', 'red card', 'yellow card'
    ]
    
    if any(indicator in position for indicator in disciplinary_indicators):
        return 1
    
    # Check cards - use the numeric versions
    yellow_cards = row.get('yellow_cards_numeric', 0)
    red_cards = row.get('red_cards_numeric', 0)
    
    if red_cards > 0:
        return 1
    elif yellow_cards >= 2:  # Two yellows = red
        return 1
    
    return 0

def calculate_enhanced_features_dynamically(matches: pd.DataFrame, calendar: pd.DatetimeIndex,
                                          player_row: pd.Series) -> Dict[str, List]:
    """
    Calculate enhanced features dynamically based on actual match data.
    This replaces hardcoded defaults with proper calculations.
    """
    n_days = len(calendar)

    # Initialize with proper defaults based on reference file patterns
    competition_importance = [BENFICA_PARITY_CONFIG['competition_importance_default']] * n_days
    season_phase = [BENFICA_PARITY_CONFIG['season_phase_default']] * n_days
    disciplinary_action = [0] * n_days
    avg_competition_importance = [BENFICA_PARITY_CONFIG['competition_importance_default']] * n_days
    cum_disciplinary_actions = [0] * n_days

    # Always calculate properly, don't force to zeros
    if BENFICA_PARITY_CONFIG.get('reproduce_historical_zeros', False):
        return {
            'competition_importance': [BENFICA_PARITY_CONFIG['competition_importance_default']] * n_days,
            'season_phase': [BENFICA_PARITY_CONFIG['season_phase_default']] * n_days,
            'disciplinary_action': [0] * n_days,
            'avg_competition_importance': [BENFICA_PARITY_CONFIG['competition_importance_default']] * n_days,
            'cum_disciplinary_actions': [0] * n_days
        }

    if matches.empty:
        return {
            'competition_importance': competition_importance,
            'season_phase': season_phase,
            'disciplinary_action': disciplinary_action,
            'avg_competition_importance': avg_competition_importance,
            'cum_disciplinary_actions': cum_disciplinary_actions
        }

    # Calculate based on actual matches
    for i, date in enumerate(calendar):
        day_matches = matches[matches['date'] == date]
        if not day_matches.empty:
            # Use the actual competition importance from match data
            competition_importance[i] = day_matches['competition_importance'].iloc[0]
            season_phase[i] = day_matches['season_phase'].iloc[0]
            # Calculate disciplinary action from match data
            disciplinary_action[i] = day_matches['disciplinary_action'].iloc[0]

    # Calculate rolling averages and cumulative values
    competition_importance_series = pd.Series(competition_importance)
    avg_competition_importance = competition_importance_series.rolling(window=7, min_periods=1).mean().fillna(0.0).tolist()

    disciplinary_action_series = pd.Series(disciplinary_action)
    cum_disciplinary_actions = disciplinary_action_series.cumsum().tolist()

    return {
        'competition_importance': competition_importance,
        'season_phase': season_phase,
        'disciplinary_action': disciplinary_action,
        'avg_competition_importance': avg_competition_importance,
        'cum_disciplinary_actions': cum_disciplinary_actions
    }

def calculate_injury_features_benfica_parity(injuries: pd.DataFrame, calendar: pd.DatetimeIndex) -> Dict[str, List]:
    """
    Calculate injury features matching historical Benfica pipeline behavior.
    The historical pipeline did NOT calculate injury severity features.
    """
    n_days = len(calendar)
    
    # Initialize with proper defaults based on reference file patterns
    avg_injury_severity = [1.0] * n_days  # Default to 1.0, not 0
    max_injury_severity = [1] * n_days  # Default to 1, not 0
    lower_leg_injuries = [0] * n_days
    knee_injuries = [0] * n_days
    upper_leg_injuries = [0] * n_days
    hip_injuries = [0] * n_days
    upper_body_injuries = [0] * n_days
    head_injuries = [0] * n_days
    illness_count = [0] * n_days
    physio_injury_ratio = [0.0] * n_days
    
    # If historical pipeline did NOT calculate injury severity, return zeros
    if BENFICA_PARITY_CONFIG.get('injury_severity_calculation', True):
        # Calculate normally (enhanced behavior)
        if not injuries.empty:
            # Calculate severity features day-by-day
            for i, date in enumerate(calendar):
                past_injuries = injuries[injuries['fromDate'] <= date]
                if not past_injuries.empty:
                    avg_injury_severity[i] = past_injuries['injury_severity'].mean()
                    max_injury_severity[i] = past_injuries['injury_severity'].max()
                    
                    # Count injuries by body part
                    body_part_counts = past_injuries['body_part_category'].value_counts()
                    lower_leg_injuries[i] = body_part_counts.get('lower_leg', 0)
                    knee_injuries[i] = body_part_counts.get('knee', 0)
                    upper_leg_injuries[i] = body_part_counts.get('upper_leg', 0)
                    hip_injuries[i] = body_part_counts.get('hip', 0)
                    upper_body_injuries[i] = body_part_counts.get('upper_body', 0)
                    head_injuries[i] = body_part_counts.get('head', 0)
                    illness_count[i] = body_part_counts.get('illness', 0)
                    
                    # Calculate physio-injury ratio
                    def parse_days(days_str):
                        if pd.isna(days_str):
                            return 0
                        try:
                            return int(str(days_str).split()[0])
                        except:
                            return 0
                    
                    past_injuries_days = past_injuries['days'].apply(parse_days)
                    short_term_injuries = past_injuries[past_injuries_days <= 7]
                    total_past_injuries = len(past_injuries)
                    if total_past_injuries > 0:
                        physio_injury_ratio[i] = len(short_term_injuries) / total_past_injuries
    
    return {
        'avg_injury_severity': avg_injury_severity,
        'max_injury_severity': max_injury_severity,
        'lower_leg_injuries': lower_leg_injuries,
        'knee_injuries': knee_injuries,
        'upper_leg_injuries': upper_leg_injuries,
        'hip_injuries': hip_injuries,
        'upper_body_injuries': upper_body_injuries,
        'head_injuries': head_injuries,
        'illness_count': illness_count,
        'physio_injury_ratio': physio_injury_ratio
    }

def calculate_national_team_features_benfica_parity(matches: pd.DataFrame, calendar: pd.DatetimeIndex) -> Dict[str, List]:
    """
    Calculate national team features matching historical Benfica pipeline behavior.
    The historical pipeline did NOT calculate national team features.
    """
    n_days = len(calendar)
    
    # Initialize with proper defaults based on reference file patterns
    national_team_appearances = [0] * n_days
    national_team_minutes = [0] * n_days
    days_since_last_national_match = [365.25] * n_days  # Default to 1 year, not 0
    national_team_this_season = [0] * n_days
    national_team_last_season = [0] * n_days
    national_team_frequency = [0.0] * n_days
    senior_national_team = [0] * n_days
    national_team_intensity = [0.0] * n_days
    
    # If historical pipeline did NOT calculate national team features, return zeros
    if BENFICA_PARITY_CONFIG.get('national_team_calculation', True):
        # Calculate normally (enhanced behavior)
        if not matches.empty:
            # Filter national team matches
            national_matches = matches[
                (matches['home_team'].apply(is_national_team_benfica_parity)) |
                (matches['away_team'].apply(is_national_team_benfica_parity))
            ]
            
            # Calculate national team features for each day
            for i, date in enumerate(calendar):
                past_national_matches = national_matches[national_matches['date'] <= date]
                
                if not past_national_matches.empty:
                    national_team_appearances[i] = len(past_national_matches)
                    
                    # Calculate total minutes in national team
                    total_national_minutes = 0
                    for _, match in past_national_matches.iterrows():
                        if pd.notna(match['minutes_played']):
                            try:
                                minutes = int(str(match['minutes_played']).replace("'", ""))
                                total_national_minutes += minutes
                            except:
                                pass
                    national_team_minutes[i] = total_national_minutes
                    
                    # Calculate days since last national match
                    last_national_date = past_national_matches['date'].max()
                    days_since = (date - last_national_date).days
                    days_since_last_national_match[i] = days_since
                    
                    # Calculate national team frequency (appearances per year)
                    if i > 0:
                        years_span = i / 365.25
                        national_team_frequency[i] = len(past_national_matches) / years_span
                    
                    # Determine if senior national team (simplified logic)
                    senior_national_team[i] = 1  # Assume all national team appearances are senior
                    
                    # Calculate national team intensity (minutes per appearance)
                    if len(past_national_matches) > 0:
                        national_team_intensity[i] = total_national_minutes / len(past_national_matches)
    
    return {
        'national_team_appearances': national_team_appearances,
        'national_team_minutes': national_team_minutes,
        'days_since_last_national_match': days_since_last_national_match,
        'national_team_this_season': national_team_this_season,
        'national_team_last_season': national_team_last_season,
        'national_team_frequency': national_team_frequency,
        'senior_national_team': senior_national_team,
        'national_team_intensity': national_team_intensity
    }

def calculate_complex_derived_features_benfica_parity(matches: pd.DataFrame, calendar: pd.DatetimeIndex) -> Dict[str, List]:
    """
    Calculate complex derived features matching historical Benfica pipeline behavior.
    The historical pipeline did NOT calculate these complex features.
    """
    n_days = len(calendar)
    
    # Initialize with proper defaults based on reference file patterns
    competition_intensity = [1.0] * n_days  # Default to 1.0, not 0
    competition_level = [1.0] * n_days  # Default to 1.0, not 0
    competition_pressure = [1.0] * n_days  # Default to 1.0, not 0
    teams_this_season = [0] * n_days
    season_team_diversity = [0.0] * n_days
    cum_competitions = [0] * n_days
    competition_diversity = [0.0] * n_days
    competition_frequency = [0.0] * n_days
    competition_experience = [0.0] * n_days
    international_competitions = [0] * n_days
    cup_competitions = [0] * n_days
    
    # If historical pipeline did NOT calculate complex features, return zeros
    if BENFICA_PARITY_CONFIG.get('complex_derived_features', True):
        # Calculate normally (enhanced behavior)
        if not matches.empty:
            # Calculate complex features based on actual match data
            for i, date in enumerate(calendar):
                past_matches = matches[matches['date'] <= date]
                if not past_matches.empty:
                    # Calculate competition features
                    competitions = past_matches['competition'].unique()
                    cum_competitions[i] = len(competitions)
                    
                    # Calculate team diversity
                    teams = set()
                    for _, match in past_matches.iterrows():
                        teams.add(match['home_team'])
                        teams.add(match['away_team'])
                    teams_this_season[i] = len(teams)
                    season_team_diversity[i] = len(teams) / max(1, len(past_matches))
                    
                    # Calculate competition diversity
                    competition_diversity[i] = len(competitions) / max(1, len(past_matches))
                    
                    # Calculate competition frequency
                    if i > 0:
                        years_span = i / 365.25
                        competition_frequency[i] = len(past_matches) / years_span
                    
                    # Calculate competition experience
                    competition_experience[i] = len(past_matches)
                    
                    # Count international competitions
                    international_competitions[i] = len([c for c in competitions if 'international' in str(c).lower()])
                    
                    # Count cup competitions
                    cup_competitions[i] = len([c for c in competitions if 'cup' in str(c).lower()])
    
    return {
        'competition_intensity': competition_intensity,
        'competition_level': competition_level,
        'competition_pressure': competition_pressure,
        'teams_this_season': teams_this_season,
        'season_team_diversity': season_team_diversity,
        'cum_competitions': cum_competitions,
        'competition_diversity': competition_diversity,
        'competition_frequency': competition_frequency,
        'competition_experience': competition_experience,
        'international_competitions': international_competitions,
        'cup_competitions': cup_competitions
    }

def get_benfica_parity_functions():
    """
    Return a dictionary of benfica-parity functions for use in the generic script.
    """
    return {
        'is_national_team': is_national_team_benfica_parity,
        'map_competition_importance': map_competition_importance_benfica_parity,
        'map_season_phase': map_season_phase_benfica_parity,
        'calculate_age': calculate_age_benfica_parity,
        'detect_disciplinary_action': detect_disciplinary_action_benfica_parity,
        'calculate_enhanced_features_dynamically': calculate_enhanced_features_dynamically
    }
