#!/usr/bin/env python3
"""
Benfica Parity Configuration
Configuration and functions to match historical Benfica pipeline behavior
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Tuple
import math

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

COMPETITION_TYPE_MAP: Dict[str, str] = {}

def normalize_competition_key(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    name_str = str(name).strip()
    if not name_str:
        return None
    return re.sub(r"\s+", " ", name_str.lower())


def set_competition_type_map(mapping: Dict[str, str]):
    global COMPETITION_TYPE_MAP
    COMPETITION_TYPE_MAP = {}
    for comp_name, comp_type in mapping.items():
        key = normalize_competition_key(comp_name)
        if key and pd.notna(comp_type):
            COMPETITION_TYPE_MAP[key] = str(comp_type)


def get_competition_type(competition_name: Optional[str]) -> Optional[str]:
    key = normalize_competition_key(competition_name)
    if not key:
        return None
    return COMPETITION_TYPE_MAP.get(key)


def map_competition_intensity_level(match: pd.Series) -> Tuple[float, float]:
    comp_name = match.get('competition')
    comp_lower = str(comp_name).lower() if pd.notna(comp_name) else ''
    comp_type = get_competition_type(comp_name)
    comp_type_lower = str(comp_type).lower() if comp_type else ''
    journey = str(match.get('journey', '')).lower() if pd.notna(match.get('journey')) else ''

    home_team = match.get('home_team')
    away_team = match.get('away_team')
    is_national = is_national_team_benfica_parity(home_team) or is_national_team_benfica_parity(away_team)

    intensity = 1.0
    level = 1.0

    elite_keywords = [
        'champions league', 'uefa champions', 'europa league', 'conference league',
        'uefa', 'libertadores', 'sudamericana', 'club world cup', 'concacaf champions',
        'afc champions', 'ofc champions', 'caf champions'
    ]
    if any(keyword in comp_lower for keyword in elite_keywords):
        intensity = max(intensity, 5.0)
        level = max(level, 5.0)
    elif is_national:
        intensity = max(intensity, 3.0)
        level = max(level, 3.0)

    if comp_type_lower:
        if 'main league' in comp_type_lower or 'league' in comp_type_lower:
            intensity = max(intensity, 4.0)
            level = max(level, 4.0)
        if 'cup' in comp_type_lower or 'trophy' in comp_type_lower:
            cup_intensity = 3.0
            if any(stage in journey for stage in ['final', 'semi', 'quarter']):
                cup_intensity = 4.0
            intensity = max(intensity, cup_intensity)
            level = max(level, cup_intensity)
        if 'friendly' in comp_type_lower:
            intensity = max(intensity, 2.0)
            level = max(level, 2.0)
    else:
        if 'friendly' in comp_lower or 'test match' in comp_lower:
            intensity = max(intensity, 2.0)
            level = max(level, 2.0)

    youth_indicators = ['u19', 'u20', 'u21', 'u23', 'youth', 'reserves', 'b team', 'ii', 'u-']
    if any(indicator in comp_lower for indicator in youth_indicators):
        intensity = min(intensity, 2.0)
        level = min(level, 2.0)

    intensity = min(max(intensity, 1.0), 5.0)
    level = min(max(level, 1.0), 5.0)
    return intensity, level

def get_football_season(date: pd.Timestamp) -> str:
    if pd.isna(date):
        return ''
    date = pd.to_datetime(date)
    year = date.year
    if date.month >= 7:
        return f"{year}/{year + 1}"
    return f"{year - 1}/{year}"


def is_national_team_benfica_parity(team_name: str) -> bool:
    """
    Simplified national team detection matching historical Benfica logic.
    This is likely what the historical pipeline used - much simpler than the current enhanced version.
    """
    if pd.isna(team_name) or not team_name:
        return False
    
    team_lower = str(team_name).lower().strip()
    
    club_indicators = [
        ' fc', 'cf ', ' cf', 'cd ', ' cd', ' sc', 'sc ', ' afc', 's.c.', 'u.d.', 'club', 'clube',
        'deportivo', 'sporting', 'association', 'academy', 'reserves',
        'juvenil', 'junior', 'national cd', 'nacional', 'nationals', 'international',
    ]
    if any(indicator in team_lower for indicator in club_indicators):
        return False
    
    country_names = [
        'brazil', 'portugal', 'spain', 'france', 'germany', 'italy', 'england',
        'argentina', 'colombia', 'uruguay', 'chile', 'mexico', 'netherlands',
        'belgium', 'croatia', 'poland', 'sweden', 'norway', 'denmark', 'switzerland',
        'austria', 'czech', 'slovakia', 'hungary', 'romania', 'bulgaria', 'serbia',
        'montenegro', 'bosnia', 'albania', 'macedonia', 'slovenia',
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
        'liberia', 'ghana', 'togo', 'benin', "cote d'ivoire", 'guinea-bissau',
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
        'iran', 'iraq', 'syria', 'lebanon', 'jordan', 'palestine',
        'saudi arabia', 'yemen', 'oman', 'uae', 'qatar', 'bahrain', 'kuwait'
    ]
    
    for country in country_names:
        if team_lower == country:
            return True
        if team_lower.startswith(country + ' '):
            suffix = team_lower[len(country):].strip()
            if suffix.startswith('u') and suffix[1:].replace('-', '').isdigit():
                return True
            if any(word in suffix for word in ['national', 'selection', 'select', 'seleccion', 'selecao', 'olympic']):
                return True
    
    national_indicators = [
        'national team', 'national selection', 'national squad', 'selección',
        'selecao', 'selección nacional', 'team nacional', 'representative'
    ]
    if any(indicator in team_lower for indicator in national_indicators):
        return True
    
    return False

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
    disciplinary_action = [0] * n_days
    avg_competition_importance = [BENFICA_PARITY_CONFIG['competition_importance_default']] * n_days
    cum_disciplinary_actions = [0] * n_days

    # Always calculate properly, don't force to zeros
    if BENFICA_PARITY_CONFIG.get('reproduce_historical_zeros', False):
        return {
            'competition_importance': [BENFICA_PARITY_CONFIG['competition_importance_default']] * n_days,
            'disciplinary_action': [0] * n_days,
            'avg_competition_importance': [BENFICA_PARITY_CONFIG['competition_importance_default']] * n_days,
            'cum_disciplinary_actions': [0] * n_days
        }

    if matches.empty:
        return {
            'competition_importance': competition_importance,
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
            # Calculate disciplinary action from match data
            disciplinary_action[i] = day_matches['disciplinary_action'].iloc[0]

    # Calculate rolling averages and cumulative values
    competition_importance_series = pd.Series(competition_importance)
    avg_competition_importance = competition_importance_series.rolling(window=7, min_periods=1).mean().fillna(0.0).tolist()

    disciplinary_action_series = pd.Series(disciplinary_action)
    cum_disciplinary_actions = disciplinary_action_series.cumsum().tolist()

    return {
        'competition_importance': competition_importance,
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
    other_injuries = [0] * n_days
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
                    other_injuries[i] = body_part_counts.get('other', 0)
                    
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
        'other_injuries': other_injuries,
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
            national_matches = national_matches.copy()
            national_matches['football_season'] = national_matches['date'].apply(get_football_season)

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
                    senior_national_team[i] = 1

                    # Calculate national team intensity (minutes per appearance)
                    if len(past_national_matches) > 0:
                        national_team_intensity[i] = total_national_minutes / len(past_national_matches)

                # Season-based counters
                current_season = get_football_season(date)
                prev_season = f"{date.year - 1}/{date.year}" if date.month >= 7 else f"{date.year - 2}/{date.year - 1}"
                national_team_this_season[i] = len(past_national_matches[past_national_matches['football_season'] == current_season])
                national_team_last_season[i] = len(past_national_matches[past_national_matches['football_season'] == prev_season])

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

INTERNATIONAL_TYPE_KEYWORDS = {
    'champions league', 'conference league', 'europa league', 'continental cup',
    'club world cup', 'world cup', 'nations league', 'international', 'olympic',
    'asian champions league', 'continental', 'junior continental cup', 'junior world cup'
}
INTERNATIONAL_NAME_KEYWORDS = {
    'champions league', 'copa libertadores', 'sudamericana', 'club world cup',
    'conference league', 'europa league', 'world cup', 'nations league',
    'cup winners cup', 'continental', 'international cup', 'olympic', 'ucl', 'uel'
}

def is_international_competition(match: pd.Series) -> bool:
    comp_name = match.get('competition')
    comp_type = get_competition_type(comp_name)
    comp_lower = str(comp_name).lower() if pd.notna(comp_name) else ''
    type_lower = str(comp_type).lower() if comp_type else ''
    if any(keyword in type_lower for keyword in INTERNATIONAL_TYPE_KEYWORDS):
        return True
    if any(keyword in comp_lower for keyword in INTERNATIONAL_NAME_KEYWORDS):
        return True
    if is_national_team_benfica_parity(match.get('home_team')) or is_national_team_benfica_parity(match.get('away_team')):
        return True
    return False

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
    transfermarkt_score_cum = [0.0] * n_days
    transfermarkt_score_avg = [0.0] * n_days
    transfermarkt_score_matches = [0] * n_days
    transfermarkt_score_recent = [0.0] * n_days
    transfermarkt_score_rolling5 = [0.0] * n_days
    
    # If historical pipeline did NOT calculate complex features, return zeros
    if BENFICA_PARITY_CONFIG.get('complex_derived_features', True):
        # Calculate normally (enhanced behavior)
        if not matches.empty:
            matches_copy = matches.copy()
            matches_copy['date'] = pd.to_datetime(matches_copy['date'])
            matches_copy = matches_copy.dropna(subset=['date']).sort_values('date')
            intensity_level = matches_copy.apply(map_competition_intensity_level, axis=1, result_type='expand')
            matches_copy['intensity'] = intensity_level[0]
            matches_copy['level'] = intensity_level[1]

            seen_international_competitions: set = set()
            
            # Add football_season to matches for season-based calculations
            matches_copy['football_season'] = matches_copy['date'].apply(get_football_season)

            for i, date in enumerate(calendar):
                past_matches = matches_copy[matches_copy['date'] <= date]
                if not past_matches.empty:
                    competitions = past_matches['competition'].unique()
                    cum_competitions[i] = len(competitions)

                    # Calculate teams_this_season based on current season only
                    current_season = get_football_season(date)
                    season_matches = past_matches[past_matches['football_season'] == current_season]
                    
                    teams = set()
                    season_teams = set()
                    for _, match in past_matches.iterrows():
                        teams.add(match['home_team'])
                        teams.add(match['away_team'])
                        if is_international_competition(match):
                            seen_international_competitions.add(match.get('competition'))
                    
                    # Count teams only from current season
                    for _, match in season_matches.iterrows():
                        season_teams.add(match['home_team'])
                        season_teams.add(match['away_team'])
                    
                    teams_this_season[i] = len(season_teams)
                    season_team_diversity[i] = len(season_teams) / max(1, len(season_matches))

                    competition_diversity[i] = len(competitions) / max(1, len(past_matches))

                    if i > 0:
                        years_span = i / 365.25
                        competition_frequency[i] = len(past_matches) / max(years_span, 1e-6)

                    competition_experience[i] = len(past_matches)

                    international_competitions[i] = len(seen_international_competitions)
                    cup_competitions[i] = len([c for c in competitions if 'cup' in str(c).lower()])

                    intensities = past_matches['intensity']
                    levels = past_matches['level']
                    competition_intensity[i] = float(intensities.mean()) if not intensities.empty else competition_intensity[i - 1] if i > 0 else 1.0
                    competition_level[i] = float(levels.mean()) if not levels.empty else competition_level[i - 1] if i > 0 else 1.0
                    competition_pressure[i] = float(past_matches['intensity'].iloc[-1])

                    bundesliga_matches = past_matches[past_matches.apply(
                        lambda m: 'bundesliga' in str(m.get('competition', '')).lower() or
                                  'bundesliga' in str(get_competition_type(m.get('competition')) or '').lower(),
                        axis=1
                    )]
                    scores = []
                    for _, match in bundesliga_matches.iterrows():
                        score = match.get('transfermarkt_score')
                        if pd.notna(score):
                            try:
                                scores.append(float(score))
                            except Exception:
                                continue
                    if scores:
                        transfermarkt_score_recent[i] = scores[-1]
                        transfermarkt_score_cum[i] = sum(scores)
                        transfermarkt_score_matches[i] = len(scores)
                        transfermarkt_score_avg[i] = transfermarkt_score_cum[i] / transfermarkt_score_matches[i]
                        transfermarkt_score_rolling5[i] = sum(scores[-5:]) / len(scores[-5:])
                        competition_pressure[i] = float(scores[-1])
                    else:
                        if i > 0:
                            transfermarkt_score_recent[i] = transfermarkt_score_recent[i - 1]
                            transfermarkt_score_cum[i] = transfermarkt_score_cum[i - 1]
                            transfermarkt_score_matches[i] = transfermarkt_score_matches[i - 1]
                            transfermarkt_score_avg[i] = transfermarkt_score_avg[i - 1]
                            transfermarkt_score_rolling5[i] = transfermarkt_score_rolling5[i - 1]
                else:
                    # No matches on this day - check if we need to reset for new season
                    current_season = get_football_season(date)
                    prev_season = get_football_season(calendar[i - 1]) if i > 0 else current_season
                    
                    if i > 0:
                        competition_intensity[i] = competition_intensity[i - 1]
                        competition_level[i] = competition_level[i - 1]
                        competition_pressure[i] = competition_pressure[i - 1]
                        
                        # For teams_this_season, reset if we've moved to a new season
                        if current_season != prev_season:
                            # New season - check if there are any matches in this season before this date
                            season_matches_before = matches_copy[
                                (matches_copy['date'] <= date) & 
                                (matches_copy['football_season'] == current_season)
                            ]
                            if not season_matches_before.empty:
                                season_teams = set()
                                for _, match in season_matches_before.iterrows():
                                    season_teams.add(match['home_team'])
                                    season_teams.add(match['away_team'])
                                teams_this_season[i] = len(season_teams)
                                season_team_diversity[i] = len(season_teams) / max(1, len(season_matches_before))
                            else:
                                teams_this_season[i] = 0
                                season_team_diversity[i] = 0.0
                        else:
                            # Same season - carry forward
                            teams_this_season[i] = teams_this_season[i - 1]
                            season_team_diversity[i] = season_team_diversity[i - 1]
                        
                        cum_competitions[i] = cum_competitions[i - 1]
                        competition_diversity[i] = competition_diversity[i - 1]
                        competition_frequency[i] = competition_frequency[i - 1]
                        competition_experience[i] = competition_experience[i - 1]
                        international_competitions[i] = international_competitions[i - 1]
                        cup_competitions[i] = cup_competitions[i - 1]
                        transfermarkt_score_recent[i] = transfermarkt_score_recent[i - 1]
                        transfermarkt_score_cum[i] = transfermarkt_score_cum[i - 1]
                        transfermarkt_score_matches[i] = transfermarkt_score_matches[i - 1]
                        transfermarkt_score_avg[i] = transfermarkt_score_avg[i - 1]
                        transfermarkt_score_rolling5[i] = transfermarkt_score_rolling5[i - 1]
    else:
        for i in range(n_days):
            teams_this_season[i] = 0
            season_team_diversity[i] = 0.0
            competition_diversity[i] = 0.0
            competition_frequency[i] = 0.0
            competition_experience[i] = 0
            international_competitions[i] = 0
            cup_competitions[i] = 0
            transfermarkt_score_cum[i] = 0.0
            transfermarkt_score_avg[i] = 0.0
            transfermarkt_score_matches[i] = 0
            transfermarkt_score_recent[i] = 0.0
            transfermarkt_score_rolling5[i] = 0.0

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
        'cup_competitions': cup_competitions,
        'transfermarkt_score_recent': transfermarkt_score_recent,
        'transfermarkt_score_cum': transfermarkt_score_cum,
        'transfermarkt_score_avg': transfermarkt_score_avg,
        'transfermarkt_score_rolling5': transfermarkt_score_rolling5,
        'transfermarkt_score_matches': transfermarkt_score_matches
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
