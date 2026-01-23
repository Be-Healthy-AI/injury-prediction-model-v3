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
import bisect
import io
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

# Fix encoding issues on Windows (only if not already fixed)
if sys.platform == "win32" and not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Benfica Parity Configuration (inlined from benfica_parity_config.py)
BENFICA_PARITY_CONFIG = {
    'use_historical_detection': True,
    'competition_importance_default': 1.0,
    'season_phase_default': 3.0,
    'age_calculation_precision': 'historical',
    'national_team_detection': 'simplified',
    'enhanced_features_dynamic': True,
    'reproduce_historical_zeros': False,
    'injury_severity_calculation': True,
    'national_team_calculation': True,
    'complex_derived_features': True
}

COMPETITION_TYPE_MAP: Dict[str, str] = {}

def normalize_competition_key(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    name_str = str(name).strip()
    if not name_str:
        return None
    return re.sub(r"\s+", " ", name_str.lower())

def get_competition_type(competition_name: Optional[str]) -> Optional[str]:
    key = normalize_competition_key(competition_name)
    if not key:
        return None
    return COMPETITION_TYPE_MAP.get(key)

def is_national_team_benfica_parity(team_name: str) -> bool:
    """Simplified national team detection matching historical Benfica logic."""
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
    """Competition importance mapping matching historical Benfica logic."""
    if pd.isna(competition) or not competition:
        return 0
    comp_lower = str(competition).lower().strip()
    if any(term in comp_lower for term in ['champions league', 'world cup', 'euro', 'european championship']):
        return 5
    elif any(term in comp_lower for term in ['europa league', 'conference league', 'copa america', 'africa cup']):
        return 4
    elif any(term in comp_lower for term in ['liga', 'premier league', 'bundesliga', 'serie a', 'la liga', 'primeira liga']):
        return 3
    elif any(term in comp_lower for term in ['cup', 'trophy', 'super cup', 'taça']):
        return 2
    else:
        return 1

def detect_disciplinary_action_benfica_parity(row: pd.Series) -> int:
    """Disciplinary action detection matching historical Benfica logic."""
    if row.empty:
        return 0
    position = str(row.get('position', '')).lower()
    disciplinary_indicators = [
        'suspended', 'banned', 'punished', 'disciplinary', 'red card', 'yellow card'
    ]
    if any(indicator in position for indicator in disciplinary_indicators):
        return 1
    yellow_cards = row.get('yellow_cards_numeric', 0)
    red_cards = row.get('red_cards_numeric', 0)
    if red_cards > 0:
        return 1
    elif yellow_cards >= 2:
        return 1
    return 0

# Import position normalization
try:
    from scripts.data_collection.transformers import _normalize_position
except ImportError:
    try:
        from data_collection.transformers import _normalize_position
    except ImportError:
        # Fallback: define a comprehensive version if import fails
        def _normalize_position(position):
            """
            Normalize position names to match reference schema.
            Maps Transfermarkt position formats and abbreviations to canonical names.
            """
            if position is None or (isinstance(position, float) and pd.isna(position)):
                return ''
            
            position_str = str(position).strip()
            if not position_str:
                return ''
            
            # Skip non-position values (bench, not selected, etc.)
            position_lower = position_str.lower()
            non_position_indicators = [
                'not in squad', 'on the bench', 'bench unused', 'unused substitute',
                'was not in the squad', 'information not yet available',
                'injury', 'lesion', 'bench', 'substitute'
            ]
            if any(indicator in position_lower for indicator in non_position_indicators):
                return ''
            
            # Position normalization mapping (from transformers.py)
            position_map = {
                # Full Transfermarkt format
                "goalkeeper": "Goalkeeper",
                "defender - centre-back": "Centre Back",
                "defender - left-back": "Left Back",
                "defender - right-back": "Right Back",
                "midfield - defensive midfield": "Defensive Midfielder",
                "midfield - central midfield": "Central Midfielder",
                "midfield - attacking midfield": "Attacking Midfielder",
                "attack - left winger": "Left Winger",
                "attack - right winger": "Right Winger",
                "attack - centre-forward": "Centre Forward",
                "attack - striker": "Centre Forward",
                "attack - second-striker": "Second Striker",
                # Abbreviations commonly used in match data
                "gk": "Goalkeeper",
                "cb": "Centre Back",
                "lb": "Left Back",
                "rb": "Right Back",
                "dm": "Defensive Midfielder",
                "cm": "Central Midfielder",
                "am": "Attacking Midfielder",
                "lw": "Left Winger",
                "rw": "Right Winger",
                "cf": "Centre Forward",
                "ss": "Second Striker",
                "st": "Centre Forward",  # Striker
                # Additional abbreviations that might appear
                "lm": "Left Winger",  # Left Midfielder/Winger
                "rm": "Right Winger",  # Right Midfielder/Winger
                # Full names (already normalized)
                "goalkeeper": "Goalkeeper",
                "centre back": "Centre Back",
                "left back": "Left Back",
                "right back": "Right Back",
                "defensive midfielder": "Defensive Midfielder",
                "central midfielder": "Central Midfielder",
                "attacking midfielder": "Attacking Midfielder",
                "left winger": "Left Winger",
                "right winger": "Right Winger",
                "centre forward": "Centre Forward",
                "second striker": "Second Striker",
                "second attacker": "Second Striker",  # Alternative name
            }
            
            # Try exact match (case-insensitive)
            normalized = position_map.get(position_lower)
            if normalized:
                # Validate it's a valid position
                if normalized in VALID_POSITIONS:
                    return normalized
                else:
                    return ''
            
            # Try partial matches for variations
            for key, value in position_map.items():
                if key in position_lower or position_lower in key:
                    # Validate it's a valid position
                    if value in VALID_POSITIONS:
                        return value
            
            # FIXED: If no mapping found, return empty string instead of original value
            # This prevents invalid values like "Lm", "Rm", injury names, etc. from being used
            return ''

# Valid canonical positions (must match what's in position_map)
VALID_POSITIONS = {
    'Goalkeeper',
    'Centre Back',
    'Left Back',
    'Right Back',
    'Defensive Midfielder',
    'Central Midfielder',
    'Attacking Midfielder',
    'Left Winger',
    'Right Winger',
    'Centre Forward',
    'Second Striker'
}

# Configure logging
def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    format_str = '%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s' if verbose else '%(asctime)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    return logging.getLogger(__name__)

# Initialize logger (will be reconfigured in main if verbose)
logger = logging.getLogger(__name__)

# Configuration
# V4: Use relative paths from script location
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = str(SCRIPT_DIR.parent.parent / 'data' / 'raw_data')
REFERENCE_DATE = pd.Timestamp('2025-12-05')
OUTPUT_DIR = str(SCRIPT_DIR.parent.parent / 'data' / 'daily_features')

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
        # Try semicolon first (as per the file format)
        try:
            teams = pd.read_csv(teams_path, encoding='utf-8-sig', sep=';', on_bad_lines='skip')
        except Exception as e:
            logger.warning(f"Error loading teams data with semicolon: {e}, trying comma")
            try:
                teams = pd.read_csv(teams_path, encoding='utf-8-sig', sep=',', on_bad_lines='skip')
            except Exception as e2:
                logger.error(f"Failed to load teams data: {e2}")
                return pd.DataFrame(columns=['team', 'country'])
        logger.info(f"Loaded {len(teams)} teams from teams_data.csv")
        return teams
    else:
        logger.warning(f"Teams data file not found: {teams_path}")
        return pd.DataFrame(columns=['team', 'country'])

def get_team_country(team_name: str, team_country_map: Dict[str, str]) -> Optional[str]:
    """Get country for a team name using normalized matching."""
    if pd.isna(team_name) or not team_name:
        return None
    team_name_str = str(team_name).strip()
    
    # Try exact match first
    if team_name_str in team_country_map:
        return team_country_map[team_name_str]
    
    # Try normalized match
    normalized_name = normalize_team_name(team_name_str)
    if normalized_name in team_country_map:
        return team_country_map[normalized_name]
    
    # Try case-insensitive match
    for key, value in team_country_map.items():
        if key.lower() == team_name_str.lower():
            return value
    
    return None

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

def load_player_data(player_id: int, data_dir: str, reference_date: pd.Timestamp = REFERENCE_DATE) -> Dict[str, pd.DataFrame]:
    """Load all data for a specific player, filtered by reference_date."""
    logger.info(f"Loading data for player {player_id}...")
    logger.debug(f"Data directory: {data_dir}, Reference date: {reference_date.date()}")
    
    # Load profile
    players_path = os.path.join(data_dir, 'players_profile.csv')
    logger.debug(f"Loading profile from: {players_path}")
    if os.path.exists(players_path):
        players = pd.read_csv(players_path, sep=';', encoding='utf-8')
        logger.debug(f"Profile file loaded: {len(players)} total players")
        players = players[players['id'] == player_id].copy()
        logger.debug(f"Filtered to player {player_id}: {len(players)} rows")
        if 'date_of_birth' in players.columns:
            # FIXED: Use pd.to_datetime without format to auto-detect ISO format (YYYY-MM-DD) and other formats
            players['date_of_birth'] = pd.to_datetime(players['date_of_birth'], errors='coerce')
            logger.debug(f"Parsed date_of_birth: {players['date_of_birth'].notna().sum()} valid dates out of {len(players)}")
    else:
        logger.warning(f"Profile file not found: {players_path}")
        players = pd.DataFrame()
    
    # Load injuries
    injuries_path = os.path.join(data_dir, 'injuries_data.csv')
    logger.debug(f"Loading injuries from: {injuries_path}")
    if os.path.exists(injuries_path):
        injuries = pd.read_csv(injuries_path, sep=';', encoding='utf-8')
        logger.debug(f"Injuries file loaded: {len(injuries)} total injuries")
        injuries = injuries[injuries['player_id'] == player_id].copy()
        logger.debug(f"Filtered to player {player_id}: {len(injuries)} injuries")
        if 'fromDate' in injuries.columns:
            # FIXED: Use European date format (DD-MM-YYYY and DD/MM/YYYY) to avoid MM-DD-YYYY misinterpretation
            # Store original column for retry
            original_dates = injuries['fromDate'].copy()
            
            # First, try DD-MM-YYYY format (European format with dashes)
            injuries['fromDate'] = pd.to_datetime(injuries['fromDate'], format='%d-%m-%Y', errors='coerce')
            
            # If that fails for many rows, try DD/MM/YYYY format (European format with slashes)
            if injuries['fromDate'].isna().sum() > len(injuries) * 0.5:
                injuries['fromDate'] = pd.to_datetime(original_dates, format='%d/%m/%Y', errors='coerce')
            
            # If that also fails, try auto-detect with dayfirst=True to prefer DD/MM/YYYY over MM/DD/YYYY
            if injuries['fromDate'].isna().sum() > len(injuries) * 0.5:
                injuries['fromDate'] = pd.to_datetime(original_dates, dayfirst=True, errors='coerce')
            
            logger.debug(f"Parsed fromDate: {injuries['fromDate'].notna().sum()} valid dates out of {len(injuries)}")
        if 'untilDate' in injuries.columns:
            # FIXED: Use European date format (DD-MM-YYYY and DD/MM/YYYY) to avoid MM-DD-YYYY misinterpretation
            # Store original column for retry
            original_until_dates = injuries['untilDate'].copy()
            
            # First, try DD-MM-YYYY format (European format with dashes)
            injuries['untilDate'] = pd.to_datetime(injuries['untilDate'], format='%d-%m-%Y', errors='coerce')
            
            # If that fails for many rows, try DD/MM/YYYY format (European format with slashes)
            if injuries['untilDate'].isna().sum() > len(injuries) * 0.5:
                injuries['untilDate'] = pd.to_datetime(original_until_dates, format='%d/%m/%Y', errors='coerce')
            
            # If that also fails, try auto-detect with dayfirst=True to prefer DD/MM/YYYY over MM/DD/YYYY
            if injuries['untilDate'].isna().sum() > len(injuries) * 0.5:
                injuries['untilDate'] = pd.to_datetime(original_until_dates, dayfirst=True, errors='coerce')
            
            logger.debug(f"Parsed untilDate: {injuries['untilDate'].notna().sum()} valid dates out of {len(injuries)}")
        
        # FILTER: Only include injuries that start on or before reference_date
        if 'fromDate' in injuries.columns:
            before_filter = len(injuries)
            injuries = injuries[injuries['fromDate'] <= reference_date].copy()
            after_filter = len(injuries)
            if before_filter != after_filter:
                logger.info(f"Filtered injuries by reference_date: {before_filter} -> {after_filter} (removed {before_filter - after_filter} injuries after {reference_date.date()})")
    else:
        logger.warning(f"Injuries file not found: {injuries_path}")
        injuries = pd.DataFrame()
    
    # Load matches
    match_data_dir = os.path.join(data_dir, 'match_data')
    logger.debug(f"Loading matches from: {match_data_dir}")
    matches = []
    if os.path.exists(match_data_dir):
        # Load ALL match files (they're named by match_id, not player_id)
        match_files = glob.glob(os.path.join(match_data_dir, 'match_*.csv'))
        logger.info(f"Found {len(match_files)} total match files")
        
        for match_file in match_files:
            try:
                logger.debug(f"Loading match file: {os.path.basename(match_file)}")
                df = pd.read_csv(match_file, encoding='utf-8-sig')
                logger.debug(f"  Loaded {len(df)} rows from {os.path.basename(match_file)}")
                
                # Filter by player_id column after loading
                if 'player_id' in df.columns:
                    df = df[df['player_id'] == player_id].copy()
                    if len(df) > 0:
                        logger.debug(f"  Filtered to {len(df)} rows for player {player_id}")
                        matches.append(df)
                else:
                    logger.debug(f"  No player_id column in {os.path.basename(match_file)}, skipping")
            except Exception as e:
                logger.warning(f"Error loading {match_file}: {e}")
                logger.debug(traceback.format_exc())
    
    if matches:
        matches = pd.concat(matches, ignore_index=True)
        logger.info(f"Combined {len(matches)} total match rows for player {player_id}")
        if 'date' in matches.columns:
            matches['date'] = pd.to_datetime(matches['date'], errors='coerce')
            valid_dates = matches['date'].notna().sum()
            logger.debug(f"Parsed dates: {valid_dates}/{len(matches)} valid")
            
            # FILTER: Only include matches on or before reference_date
            before_filter = len(matches)
            matches = matches[matches['date'] <= reference_date].copy()
            after_filter = len(matches)
            if before_filter != after_filter:
                logger.info(f"Filtered matches by reference_date: {before_filter} -> {after_filter} (removed {before_filter - after_filter} matches after {reference_date.date()})")
            
            if valid_dates > 0 and len(matches) > 0:
                logger.info(f"  Date range: {matches['date'].min().date()} to {matches['date'].max().date()}")
    else:
        logger.warning(f"No match files found for player {player_id}")
        matches = pd.DataFrame()
    
    # Load career
    career_path = os.path.join(data_dir, 'players_career.csv')
    logger.debug(f"Loading career from: {career_path}")
    if os.path.exists(career_path):
        career = pd.read_csv(career_path, sep=';', encoding='utf-8')
        logger.debug(f"Career file loaded: {len(career)} total entries")
        career = career[career['id'] == player_id].copy()
        logger.debug(f"Filtered to player {player_id}: {len(career)} career entries")
        if 'Date' in career.columns:
            # FIXED: Use pd.to_datetime without format to auto-detect ISO format (YYYY-MM-DD) and other formats
            career['Date'] = pd.to_datetime(career['Date'], errors='coerce')
            logger.debug(f"Parsed career dates: {career['Date'].notna().sum()} valid dates out of {len(career)}")
            
            # FILTER: Only include career entries on or before reference_date
            before_filter = len(career)
            career = career[career['Date'] <= reference_date].copy()
            after_filter = len(career)
            if before_filter != after_filter:
                logger.info(f"Filtered career by reference_date: {before_filter} -> {after_filter} (removed {before_filter - after_filter} entries after {reference_date.date()})")
    else:
        logger.warning(f"Career file not found: {career_path}")
        career = pd.DataFrame()
    
    # Load teams data
    logger.info("Loading teams data...")
    teams = load_teams_data(data_dir)
    team_country_map = {}
    if not teams.empty and 'team' in teams.columns and 'country' in teams.columns:
        # OPTIMIZATION: Use itertuples() instead of iterrows()
        for row in teams.itertuples(index=False):
            team_name = str(getattr(row, 'team', '')).strip()
            country = getattr(row, 'country', None)
            if pd.notna(country) and str(country).strip():
                # Store both exact name and normalized name for flexible lookup
                team_country_map[team_name] = str(country).strip()
                normalized_name = normalize_team_name(team_name)
                if normalized_name and normalized_name != team_name:
                    # Only add normalized if different from original
                    if normalized_name not in team_country_map:
                        team_country_map[normalized_name] = str(country).strip()
        logger.info(f"Loaded {len(teams)} teams, created {len(team_country_map)} team-country mappings")
    else:
        logger.warning("No teams data available or missing required columns")
    
    logger.info(f"[LOADED] {len(players)} players, {len(injuries)} injuries, {len(matches)} matches, {len(career)} career entries")
    if len(matches) > 0:
        logger.info(f"   Match date range: {matches['date'].min().date()} to {matches['date'].max().date()}")
    if len(injuries) > 0:
        logger.info(f"   Injury date range: {injuries['fromDate'].min().date()} to {injuries['fromDate'].max().date()}")
    
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
    logger.debug("=== Starting preprocess_matches ===")
    if matches.empty:
        logger.warning("Matches DataFrame is empty")
        return matches
    
    logger.debug(f"Preprocessing {len(matches)} matches")
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
    else:
        logger.debug(f"All required columns present: {required_cols}")
    
    # Create numeric columns
    logger.debug("Creating numeric columns...")
    if 'goals' in matches.columns:
        matches['goals_numeric'] = pd.to_numeric(matches['goals'], errors='coerce').fillna(0).astype(int)
        total_goals = matches['goals_numeric'].sum()
        logger.debug(f"  Goals: {total_goals} total from {len(matches[matches['goals_numeric'] > 0])} matches")
    else:
        logger.debug("  Goals column not found, defaulting to 0")
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
    
    # Determine participation status (following v3 logic)
    def determine_participation_status(row):
        position = str(row.get('position', '')).lower() if pd.notna(row.get('position')) else ''
        minutes = row.get('minutes_played_numeric', 0)
        
        # Check for injury-related positions first
        injury_indicators = [
            'injury', 'lesion', 'unknown lesion', 'cervicalgia', 'pubalgia',
            'knee injury', 'muscle injury', 'fracture', 'sprain', 'strain',
            'tear', 'rupture', 'tendon', 'ligament', 'joint', 'bruise',
            'contusion', 'inflammation', 'infection', 'illness', 'disease',
            'pain', 'discomfort', 'surgery', 'rehabilitation', 'recuperation',
            'cervicalgia', 'lombalgia', 'low back pain', 'back pain', 'unknown injury'
        ]
        if any(indicator in position for indicator in injury_indicators):
            return 'injured'
        
        # Check for bench-related positions (case-insensitive)
        bench_indicators = [
            'bench unused', 'unused substitute', 'on the bench', 'bench',
            'unused sub', 'substitute unused'
        ]
        not_selected_indicators = [
            'was not in the squad', 'not in squad', 'not in the squad'
        ]
        
        if pd.isna(minutes) or minutes == 0:
            # Check for not selected first
            if any(indicator in position for indicator in not_selected_indicators):
                return 'not_selected'
            # Check for bench unused
            elif any(indicator in position for indicator in bench_indicators):
                return 'bench_unused'
            elif 'information not yet available' in position or position == '' or pd.isna(row.get('position')):
                return 'not_selected'  # Default assumption
            else:
                # Has position but no minutes - could be bench_unused if position indicates bench
                # or could be played if position is a playing position
                # Default to bench_unused if minutes is 0
                return 'bench_unused'
        else:
            # Has minutes > 0, definitely played
            return 'played'
    
    matches['participation_status'] = matches.apply(determine_participation_status, axis=1)
    matches['matches_played'] = (matches['participation_status'] == 'played').astype(int)
    matches['matches_bench_unused'] = (matches['participation_status'] == 'bench_unused').astype(int)
    matches['matches_not_selected'] = (matches['participation_status'] == 'not_selected').astype(int)
    matches['matches_injured'] = (matches['participation_status'] == 'injured').astype(int)
    total_injured = matches['matches_injured'].sum()
    logger.debug(f"Total matches with injured status: {total_injured} out of {len(matches)} matches")
    
    # Normalize position column in matches (same as in transformers.py)
    if 'position' in matches.columns:
        logger.debug("Normalizing position column in matches...")
        # Log some examples before normalization
        sample_positions_before = matches['position'].dropna().unique()[:10]
        logger.debug(f"Sample positions before normalization: {list(sample_positions_before)}")
        
        matches['position'] = matches['position'].apply(_normalize_position)
        # Convert None to empty string for consistency
        matches['position'] = matches['position'].fillna('').replace('None', '')
        
        # Log some examples after normalization
        sample_positions_after = matches['position'].dropna().unique()[:10]
        logger.debug(f"Sample positions after normalization: {list(sample_positions_after)}")
        logger.debug(f"Normalized positions: {matches['position'].notna().sum()} non-null values")
    
    # Parse substitution data
    if 'substitutions_on' in matches.columns:
        # Handle both numeric and string values
        def parse_sub_on(val):
            if pd.isna(val) or val == '':
                return None
            # If already numeric, return as int
            if isinstance(val, (int, float)):
                return int(val) if not pd.isna(val) else None
            # Otherwise parse as string
            return parse_substitution_minutes(str(val))
        matches['substitution_on_minute'] = matches['substitutions_on'].apply(parse_sub_on)
        logger.debug(f"Parsed substitution_on_minute: {matches['substitution_on_minute'].notna().sum()} non-null values")
    else:
        logger.warning("substitutions_on column not found, creating empty column")
        matches['substitution_on_minute'] = pd.Series([None] * len(matches), dtype=object)
    
    if 'substitutions_off' in matches.columns:
        # Handle both numeric and string values
        def parse_sub_off(val):
            if pd.isna(val) or val == '':
                return None
            # If already numeric, return as int
            if isinstance(val, (int, float)):
                return int(val) if not pd.isna(val) else None
            # Otherwise parse as string
            return parse_substitution_minutes(str(val))
        matches['substitution_off_minute'] = matches['substitutions_off'].apply(parse_sub_off)
        logger.debug(f"Parsed substitution_off_minute: {matches['substitution_off_minute'].notna().sum()} non-null values")
    else:
        logger.warning("substitutions_off column not found, creating empty column")
        matches['substitution_off_minute'] = pd.Series([None] * len(matches), dtype=object)
    
    # Sort by date
    if 'date' in matches.columns:
        matches = matches.sort_values('date').reset_index(drop=True)
        logger.debug(f"Sorted matches by date: {matches['date'].min()} to {matches['date'].max()}")
    
    # Summary statistics
    logger.debug("Preprocessing summary:")
    logger.debug(f"  Total matches: {len(matches)}")
    logger.debug(f"  Matches played: {matches['matches_played'].sum()}")
    logger.debug(f"  Total goals: {matches['goals_numeric'].sum()}")
    logger.debug(f"  Total assists: {matches['assists_numeric'].sum()}")
    logger.debug(f"  Unique competitions: {matches['competition'].nunique() if 'competition' in matches.columns else 0}")
    
    logger.debug("=== Completed preprocess_matches ===")
    return matches

def determine_calendar(
    matches: pd.DataFrame, 
    injuries: pd.DataFrame, 
    reference_date: pd.Timestamp,
    player_row: Optional[pd.Series] = None
) -> pd.DatetimeIndex:
    """
    Determine the date range for feature generation.
    
    CRITICAL: Calendar starts from FIRST SEASON START (01/07/YYYY), not first match date.
    This matches the reference implementation logic.
    
    Args:
        matches: DataFrame with player matches
        injuries: DataFrame with player injuries
        reference_date: Reference date to cap the calendar
        player_row: Optional player row for fallback to DOB if no matches
    """
    logger.debug("=== Determining calendar ===")
    
    # FIXED: Calendar starts from FIRST SEASON START (01/07/YYYY), not first match date
    if not matches.empty and 'date' in matches.columns:
        first_match_date = matches['date'].min()
        if pd.notna(first_match_date):
            # Get the season start (01/07/YYYY) for the first match
            if first_match_date.month >= 7:
                # Match is in second half of year, season started this year
                start_date = pd.Timestamp(f'{first_match_date.year}-07-01')
            else:
                # Match is in first half of year, season started previous year
                start_date = pd.Timestamp(f'{first_match_date.year - 1}-07-01')
            logger.info(f"[CALENDAR] Player starts from FIRST SEASON START: {start_date.date()} (first match: {first_match_date.date()})")
        else:
            # Fallback if no valid dates
            if player_row is not None and 'date_of_birth' in player_row and pd.notna(player_row['date_of_birth']):
                dob = pd.to_datetime(player_row['date_of_birth'])
                # Start from season start of birth year
                start_date = pd.Timestamp(f'{dob.year}-07-01')
                logger.warning(f"[CALENDAR] No valid match dates, using season start of birth year: {start_date.date()}")
            else:
                start_date = pd.Timestamp('2000-07-01')
                logger.warning(f"[CALENDAR] No matches or DOB found, using default: {start_date.date()}")
    else:
        # Fallback to player's birth date if no matches
        if player_row is not None and 'date_of_birth' in player_row and pd.notna(player_row['date_of_birth']):
            dob = pd.to_datetime(player_row['date_of_birth'])
            start_date = pd.Timestamp(f'{dob.year}-07-01')
            logger.warning(f"[CALENDAR] Player has no matches, using season start of birth year: {start_date.date()}")
        else:
            start_date = pd.Timestamp('2000-07-01')
            logger.warning(f"[CALENDAR] No matches or DOB found, using default: {start_date.date()}")
    
    # Determine end date - SMART approach that respects career termination but includes all injuries
    if not matches.empty:
        last_match_date = matches['date'].max()
        
        # Check if player has any injuries after their last match
        if not injuries.empty and 'fromDate' in injuries.columns:
            latest_injury_date = injuries['fromDate'].max()
            
            # If player has injuries after their last match, extend to include those injuries
            if latest_injury_date > last_match_date:
                # Player retired but had injuries after retirement - extend to injury date + buffer
                injury_end_year = latest_injury_date.year
                if latest_injury_date.month >= 7:  # If injury is in second half of year
                    injury_end_year = latest_injury_date.year + 1
                injury_end_date = pd.Timestamp(f'{injury_end_year}-06-30')
                
                # Use the later of: last match season end OR injury season end
                match_end_year = last_match_date.year
                if last_match_date.month >= 7:  # If last match is in second half of year
                    match_end_year = last_match_date.year + 1
                match_end_date = pd.Timestamp(f'{match_end_year}-06-30')
                
                end_date = max(match_end_date, injury_end_date)
                logger.info(f"[CALENDAR] Player retired but had injuries after retirement")
                logger.info(f"[CALENDAR] Calendar extends to: {end_date.date()} (includes post-retirement injuries)")
            else:
                # Player retired and no injuries after retirement - respect career end
                season_end_year = last_match_date.year
                if last_match_date.month >= 7:  # If last match is in second half of year
                    season_end_year = last_match_date.year + 1
                end_date = pd.Timestamp(f'{season_end_year}-06-30')
                logger.info(f"[CALENDAR] Player calendar respects career end: {end_date.date()}")
        else:
            # Player has no injuries - respect career end
            season_end_year = last_match_date.year
            if last_match_date.month >= 7:  # If last match is in second half of year
                season_end_year = last_match_date.year + 1
            end_date = pd.Timestamp(f'{season_end_year}-06-30')
            logger.info(f"📅 Player calendar respects career end: {end_date.date()}")
    else:
        # If no matches, check if player has injuries to determine end date
        if not injuries.empty and 'fromDate' in injuries.columns:
            latest_injury_date = injuries['fromDate'].max()
            injury_end_year = latest_injury_date.year
            if latest_injury_date.month >= 7:  # If injury is in second half of year
                injury_end_year = latest_injury_date.year + 1
            end_date = pd.Timestamp(f'{injury_end_year}-06-30')
            logger.info(f"[CALENDAR] Player has no matches but has injuries, calendar extends to: {end_date.date()}")
        else:
            # No matches and no injuries - use reference date
            end_date = reference_date
            logger.info(f"[CALENDAR] Player has no matches or injuries, using reference date: {end_date.date()}")
    
    start_date = pd.Timestamp(start_date).normalize()
    end_date = pd.Timestamp(end_date).normalize()
    
    # FIXED: Only cap to reference_date, don't extend
    if end_date > reference_date:
        logger.info(f"[CALENDAR] Clamping calendar end from {end_date.date()} to {reference_date.date()} (reference date cap)")
        end_date = reference_date
    
    if start_date > end_date:
        logger.warning(f"[CALENDAR] Start date {start_date.date()} exceeds end date {end_date.date()}. Adjusting to single-day calendar.")
        end_date = start_date
    
    calendar = pd.date_range(start=start_date, end=end_date, freq='D')
    logger.info(f"[CALENDAR] Calendar: {len(calendar)} days from {start_date.date()} to {end_date.date()}")
    logger.debug(f"Calendar spans {(end_date - start_date).days} days")
    
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
    logger.debug(f"Processing {n_days} days for profile features")
    logger.debug(f"Player: {player_row.get('name', 'Unknown')} (ID: {player_row.get('id', 'Unknown')})")
    
    # OPTIMIZATION: Pre-index matches by date for O(1) lookup
    logger.debug("Pre-indexing matches by date for profile features...")
    matches_by_date = {}
    matches_sorted = None
    if not matches.empty and 'date' in matches.columns:
        matches['date_norm'] = matches['date'].dt.normalize()
        for date_norm, group in matches.groupby('date_norm'):
            matches_by_date[date_norm] = group.copy()  # Use .copy() to ensure normalization is preserved
        matches_sorted = matches.sort_values('date_norm').reset_index(drop=True).copy()
        logger.debug(f"Indexed {len(matches_by_date)} unique match dates for profile features")
    
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
    
    # FIXED: Calculate age with proper handling of missing birth dates
    # Check for different possible column names
    dob = None
    if 'date_of_birth' in player_row.index and pd.notna(player_row['date_of_birth']):
        dob = pd.Timestamp(player_row['date_of_birth'])
    elif 'Date of Birth' in player_row.index and pd.notna(player_row['Date of Birth']):
        dob = pd.Timestamp(player_row['Date of Birth'])
    elif 'birth_date' in player_row.index and pd.notna(player_row['birth_date']):
        dob = pd.Timestamp(player_row['birth_date'])
    
    if dob is not None and pd.notna(dob):
        # Vectorized calculation: calculate all ages at once
        age_deltas = (calendar - dob).days / 365.25
        # Handle any NaT/NaN values that might result
        age_deltas = pd.Series(age_deltas).fillna(0.0)
        features['age'] = age_deltas.tolist()
        logger.debug(f"Age calculated: min={min(features['age']):.2f}, max={max(features['age']):.2f}")
    else:
        birth_cols = [col for col in player_row.index if 'birth' in str(col).lower() or 'dob' in str(col).lower()]
        logger.warning(f"No valid date_of_birth found (checked columns: {birth_cols}, available columns: {list(player_row.index)[:10]}...), age will remain 0.0")
    
    # Process career to get club changes
    club_changes = []
    if not career.empty and 'Date' in career.columns and 'To' in career.columns:
        logger.debug(f"Processing {len(career)} career entries for club changes")
        # OPTIMIZATION: Use itertuples() instead of iterrows()
        for row in career.itertuples(index=False):
            date_val = getattr(row, 'Date', None)
            to_val = getattr(row, 'To', None)
            if pd.notna(date_val) and pd.notna(to_val):
                club_changes.append({
                    'date': pd.Timestamp(date_val).normalize(),
                    'club': str(to_val).strip()
                })
        club_changes.sort(key=lambda x: x['date'])
        logger.debug(f"Found {len(club_changes)} club changes")
        for change in club_changes:
            logger.debug(f"  {change['date'].date()}: {change['club']}")
    else:
        logger.debug("No career data available for club changes")
    
    # Process matches to get unique teams (excluding national teams)
    if not matches.empty and 'date' in matches.columns:
        # Get unique teams per day from matches
        match_teams = set()
        # OPTIMIZATION: Use itertuples() instead of iterrows()
        for match in matches.itertuples(index=False):
            match_date_val = getattr(match, 'date', None) if hasattr(match, 'date') else None
            if pd.notna(match_date_val):
                # Check home_team and away_team
                for team_col in ['home_team', 'away_team', 'team']:
                    if hasattr(match, team_col):
                        team_name = str(getattr(match, team_col)).strip()
                        if pd.notna(team_name) and team_name != '' and not is_national_team(team_name):
                            match_teams.add(team_name)
    
    # OPTIMIZATION: Pre-compute club change timeline (O(n) instead of O(n*m))
    club_timeline = {}
    change_dates = []
    if club_changes:
        for change in club_changes:
            change_date = change['date']
            club_timeline[change_date] = {
                'club': change['club'],
                'start_date': change_date
            }
            change_dates.append(change_date)
        change_dates.sort()
        logger.debug(f"Pre-computed club timeline with {len(change_dates)} changes")
    
    # OPTIMIZATION: Pre-compute cumulative unique teams per date
    cum_teams_by_date = {}
    all_teams_seen_cum = set()
    if matches_sorted is not None and not matches_sorted.empty:
        logger.debug("Pre-computing cumulative teams per date...")
        for date_norm, group in matches_sorted.groupby('date_norm'):
            for match in group.itertuples(index=False):
                match_dict = {col: getattr(match, col, None) for col in group.columns if hasattr(match, col)}
                player_team = identify_player_team(match_dict, None)
                if player_team and not is_national_team(player_team):
                    all_teams_seen_cum.add(player_team)
            cum_teams_by_date[date_norm] = len(all_teams_seen_cum)
        logger.debug(f"Pre-computed cumulative teams for {len(cum_teams_by_date)} dates")
    
    # OPTIMIZATION: Pre-compute cumulative seasons per date
    cum_seasons_by_date = {}
    seasons_seen_cum = set()
    if matches_sorted is not None and not matches_sorted.empty:
        logger.debug("Pre-computing cumulative seasons per date...")
        for date_norm, group in matches_sorted.groupby('date_norm'):
            if 'date' in group.columns:
                seasons = group['date'].apply(lambda d: get_football_season(pd.Timestamp(d)) if pd.notna(d) else None)
                seasons_seen_cum.update(seasons.dropna().unique())
            cum_seasons_by_date[date_norm] = len(seasons_seen_cum)
        logger.debug(f"Pre-computed cumulative seasons for {len(cum_seasons_by_date)} dates")
    
    # OPTIMIZATION: Pre-compute sorted dates for binary search
    # Convert to pandas Timestamp array for proper comparison with np.searchsorted
    if matches_sorted is not None and not matches_sorted.empty:
        matches_sorted_dates = pd.to_datetime(matches_sorted['date_norm']).values
    else:
        matches_sorted_dates = np.array([])
    
    # Track current club and seniority
    current_club = None
    current_club_start_date = None
    previous_club = None
    all_teams_seen = set()
    
    # FIXED: Track last identified club persistently to avoid frequent incorrect changes
    last_identified_club = None
    last_identified_club_date = None
    
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
        
        # OPTIMIZATION: Binary search for club change (O(log n) instead of O(n))
        if change_dates:
            idx = bisect.bisect_right(change_dates, date_norm) - 1
            if idx >= 0:
                change_date = change_dates[idx]
                current_club = club_timeline[change_date]['club']
                current_club_start_date = club_timeline[change_date]['start_date']
                if idx > 0:
                    previous_club = club_timeline[change_dates[idx-1]]['club']
                else:
                    previous_club = None
                if current_club:
                    all_teams_seen.add(current_club)
            else:
                current_club = None
                current_club_start_date = None
                previous_club = None
        else:
            current_club = None
            current_club_start_date = None
            previous_club = None
        
        # If no club change found, try to infer from matches (using binary search)
        if current_club is None and len(matches_sorted_dates) > 0:
            # FIXED: Use identify_player_team to properly identify which team the player played for
            # This avoids incorrectly picking home_team when player is on away_team
            date_norm_np = np.datetime64(date_norm)
            idx = np.searchsorted(matches_sorted_dates, date_norm_np, side='right')
            if idx > 0:
                past_matches = matches_sorted.iloc[:idx]
                if not past_matches.empty:
                    # Work backwards from most recent match to find player's actual team
                    for match_idx in range(len(past_matches) - 1, -1, -1):
                        match_row = past_matches.iloc[match_idx]
                        # Convert Series to dict-like structure for identify_player_team
                        match_dict = match_row.to_dict()
                        
                        # Use identify_player_team with last_identified_club to maintain consistency
                        player_team = identify_player_team(match_dict, last_identified_club)
                        
                        if player_team and not is_national_team(player_team):
                            # Found the player's team in this match
                            if current_club is None:
                                current_club = player_team
                                # Find the first match date with this club for better accuracy
                                first_match_with_club = None
                                for check_idx in range(match_idx, -1, -1):
                                    check_match = past_matches.iloc[check_idx]
                                    check_match_dict = check_match.to_dict()
                                    check_team = identify_player_team(check_match_dict, player_team)
                                    if check_team and normalize_team_name(check_team) == normalize_team_name(player_team):
                                        if 'date_norm' in check_match_dict and pd.notna(check_match_dict['date_norm']):
                                            first_match_with_club = pd.Timestamp(check_match_dict['date_norm'])
                                current_club_start_date = first_match_with_club if first_match_with_club else date_norm
                                last_identified_club = player_team
                                last_identified_club_date = date_norm
                            all_teams_seen.add(player_team)
                            break  # Found the player's team, stop searching
        
        # IMPROVED: Set current and previous club with improved logic for early dates
        if current_club:
            features['current_club'][i] = current_club
            club_country = get_team_country(current_club, team_country_map)
            features['current_club_country'][i] = club_country if club_country else ''
        else:
            # IMPROVED: Try to infer club from matches even if no career data
            # FIXED: Use identify_player_team and track persistently to avoid frequent incorrect changes
            if len(matches_sorted_dates) > 0:
                date_norm_np = np.datetime64(date_norm)
                idx = np.searchsorted(matches_sorted_dates, date_norm_np, side='right')
                if idx > 0:
                    past_matches = matches_sorted.iloc[:idx]
                    if not past_matches.empty:
                        # FIXED: Use identify_player_team to find player's actual team
                        # Work backwards from most recent match
                        for match_idx in range(len(past_matches) - 1, -1, -1):
                            match_row = past_matches.iloc[match_idx]
                            # Convert Series to dict-like structure for identify_player_team
                            match_dict = match_row.to_dict()
                            
                            # Use last_identified_club to maintain consistency
                            player_team = identify_player_team(match_dict, last_identified_club)
                            
                            if player_team and not is_national_team(player_team):
                                # Only update if we don't have a club yet, or if it's different (club change)
                                if not current_club:
                                    features['current_club'][i] = player_team
                                    club_country = get_team_country(player_team, team_country_map)
                                    features['current_club_country'][i] = club_country if club_country else ''
                                    current_club = player_team
                                    # Find first match date with this club
                                    first_match_with_club = None
                                    for check_idx in range(match_idx, -1, -1):
                                        check_match = past_matches.iloc[check_idx]
                                        check_match_dict = check_match.to_dict()
                                        check_team = identify_player_team(check_match_dict, player_team)
                                        if check_team and normalize_team_name(check_team) == normalize_team_name(player_team):
                                            if 'date_norm' in check_match_dict and pd.notna(check_match_dict['date_norm']):
                                                first_match_with_club = pd.Timestamp(check_match_dict['date_norm'])
                                    current_club_start_date = first_match_with_club if first_match_with_club else date_norm
                                    last_identified_club = player_team
                                    last_identified_club_date = date_norm
                                elif normalize_team_name(player_team) != normalize_team_name(current_club):
                                    # Club change detected - only update if clearly different
                                    features['current_club'][i] = player_team
                                    club_country = get_team_country(player_team, team_country_map)
                                    features['current_club_country'][i] = club_country if club_country else ''
                                    current_club = player_team
                                    # Find first match date with this new club
                                    first_match_with_club = None
                                    for check_idx in range(match_idx, -1, -1):
                                        check_match = past_matches.iloc[check_idx]
                                        check_match_dict = check_match.to_dict()
                                        check_team = identify_player_team(check_match_dict, player_team)
                                        if check_team and normalize_team_name(check_team) == normalize_team_name(player_team):
                                            if 'date_norm' in check_match_dict and pd.notna(check_match_dict['date_norm']):
                                                first_match_with_club = pd.Timestamp(check_match_dict['date_norm'])
                                    current_club_start_date = first_match_with_club if first_match_with_club else date_norm
                                    last_identified_club = player_team
                                    last_identified_club_date = date_norm
                                break  # Found the player's team, stop searching
        
        # FIXED: Calculate seniority_days AFTER setting current_club (moved here from before)
        if current_club and current_club_start_date:
            features['seniority_days'][i] = (date_norm - current_club_start_date).days
        else:
            features['seniority_days'][i] = 0
        
        if previous_club:
            features['previous_club'][i] = previous_club
            prev_club_country = get_team_country(previous_club, team_country_map)
            features['previous_club_country'][i] = prev_club_country if prev_club_country else ''
        
        # Calculate teams_today (unique teams played for on this day, from matches)
        teams_today_set = set()
        day_matches = matches_by_date.get(date_norm, pd.DataFrame())
        if not day_matches.empty:
            # OPTIMIZATION: Use itertuples() instead of iterrows()
            for match in day_matches.itertuples(index=False):
                for team_col in ['home_team', 'away_team', 'team']:
                    if hasattr(match, team_col):
                        team_name = str(getattr(match, team_col)).strip()
                        if pd.notna(team_name) and team_name != '' and not is_national_team(team_name):
                            teams_today_set.add(team_name)
        features['teams_today'][i] = len(teams_today_set)
        
        # OPTIMIZATION: Use pre-computed cumulative teams (O(1) lookup)
        if cum_teams_by_date:
            matching_dates = [d for d in cum_teams_by_date.keys() if d <= date_norm]
            if matching_dates:
                features['cum_teams'][i] = cum_teams_by_date[max(matching_dates)]
            else:
                features['cum_teams'][i] = 0
        else:
            features['cum_teams'][i] = 0
        
        # OPTIMIZATION: Use pre-computed cumulative seasons
        if cum_seasons_by_date:
            matching_dates = [d for d in cum_seasons_by_date.keys() if d <= date_norm]
            if matching_dates:
                features['seasons_count'][i] = cum_seasons_by_date[max(matching_dates)]
            else:
                features['seasons_count'][i] = 0
        else:
            features['seasons_count'][i] = 0
    
    # Summary statistics
    logger.debug("Profile features summary:")
    logger.debug(f"  Unique clubs: {len(set(features['current_club'])) - (1 if '' in set(features['current_club']) else 0)}")
    logger.debug(f"  Max seniority days: {max(features['seniority_days'])}")
    logger.debug(f"  Total teams seen: {max(features['cum_teams'])}")
    logger.debug(f"  Total seasons: {max(features['seasons_count'])}")
    
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
    logger.debug(f"Processing {n_days} days for match features")
    logger.debug(f"Total matches to process: {len(matches)}")
    if not matches.empty:
        logger.debug(f"Match date range: {matches['date'].min().date()} to {matches['date'].max().date()}")
    
    # OPTIMIZATION: Pre-index matches by date for O(1) lookup instead of O(n) filtering
    logger.debug("Pre-indexing matches by date...")
    matches_by_date = {}
    if not matches.empty and 'date' in matches.columns:
        # Normalize all dates first
        matches['date_norm'] = matches['date'].dt.normalize()
        # Group by normalized date - use .copy() to ensure normalization is preserved
        for date_norm, group in matches.groupby('date_norm'):
            matches_by_date[date_norm] = group.copy()
        logger.debug(f"Indexed {len(matches_by_date)} unique match dates")
    
    # OPTIMIZATION: Pre-index matches for "past matches" queries (<= date)
    # Sort matches by date and create cumulative index
    matches_sorted = None
    matches_sorted_dates = np.array([])
    if not matches.empty and 'date' in matches.columns:
        matches_sorted = matches.sort_values('date_norm').reset_index(drop=True)
        # Convert to pandas Timestamp array for proper comparison with np.searchsorted
        matches_sorted_dates = pd.to_datetime(matches_sorted['date_norm']).values
        logger.debug(f"Sorted {len(matches_sorted)} matches by date for cumulative queries")
    
    # OPTIMIZATION: Pre-aggregate matches by date (do this once, not per day)
    matches_by_date_agg = {}
    if not matches.empty:
        logger.debug("Pre-aggregating matches by date...")
        for date_norm, group in matches.groupby('date_norm'):
            agg = {
                'matches_played': int(group['matches_played'].fillna(0).sum()),
                'minutes_played_numeric': int(group['minutes_played_numeric'].fillna(0).sum()),
                'goals_numeric': int(group['goals_numeric'].fillna(0).sum()),
                'assists_numeric': int(group['assists_numeric'].fillna(0).sum()),
                'yellow_cards_numeric': int(group['yellow_cards_numeric'].fillna(0).sum()),
                'red_cards_numeric': int(group['red_cards_numeric'].fillna(0).sum()),
                'matches_bench_unused': int(group['matches_bench_unused'].fillna(0).sum()),
                'matches_not_selected': int(group['matches_not_selected'].fillna(0).sum()),
                'matches_injured': int(group['matches_injured'].fillna(0).sum()),
                'disciplinary_action': int(group['disciplinary_action'].fillna(0).max()) if 'disciplinary_action' in group.columns else 0,
            }
            matches_by_date_agg[date_norm] = agg
        logger.debug(f"Pre-aggregated matches for {len(matches_by_date_agg)} dates")
    
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
        'club_assists_per_match', 'club_minutes_per_match',
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
    # Normalize default position from profile
    if 'position' in player_row and pd.notna(player_row.get('position')):
        default_position = _normalize_position(player_row.get('position', ''))
        # Handle None/empty from normalization
        if not default_position or default_position == 'None':
            default_position = ''
    else:
        default_position = ''
    logger.debug(f"Default position (normalized): '{default_position}'")
    
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
    
    # OPTIMIZATION: Move abbrev_map outside loop to avoid recreating it thousands of times
    abbrev_map = {
        'gk': 'Goalkeeper',
        'lw': 'Left Winger', 'rw': 'Right Winger', 'lm': 'Left Winger', 'rm': 'Right Winger',
        'am': 'Attacking Midfielder',
        'cm': 'Central Midfielder', 'dm': 'Defensive Midfielder',
        'cb': 'Centre Back', 'lb': 'Left Back', 'rb': 'Right Back',
        'cf': 'Centre Forward', 'ss': 'Second Striker', 'st': 'Centre Forward'
    }
    
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
        
        # Get matches on this day (using pre-indexed dictionary for O(1) lookup)
        day_matches = matches_by_date.get(date_norm, pd.DataFrame())
        
        # Track if this day has any substitution
        day_has_substitution = False
        
        # Get matches on this day (using pre-indexed dictionary for O(1) lookup)
        day_matches = matches_by_date.get(date_norm, pd.DataFrame())
        
        # OPTIMIZATION: Use pre-aggregated data for day matches (faster than recalculating)
        day_agg = matches_by_date_agg.get(date_norm, {})
        if day_agg:
            # Update features from pre-aggregated data
            features['matches_played'][i] = day_agg['matches_played']
            features['minutes_played_numeric'][i] = day_agg['minutes_played_numeric']
            features['goals_numeric'][i] = day_agg['goals_numeric']
            features['assists_numeric'][i] = day_agg['assists_numeric']
            features['yellow_cards_numeric'][i] = day_agg['yellow_cards_numeric']
            features['red_cards_numeric'][i] = day_agg['red_cards_numeric']
            features['matches_bench_unused'][i] = day_agg['matches_bench_unused']
            features['matches_not_selected'][i] = day_agg['matches_not_selected']
            features['matches_injured'][i] = day_agg['matches_injured']
            features['disciplinary_action'][i] = day_agg['disciplinary_action']
            
            # Update cumulative values
            cum_minutes += day_agg['minutes_played_numeric']
            cum_goals += day_agg['goals_numeric']
            cum_assists += day_agg['assists_numeric']
            cum_yellow_cards += day_agg['yellow_cards_numeric']
            cum_red_cards += day_agg['red_cards_numeric']
            cum_matches_played += day_agg['matches_played']
            cum_matches_bench += day_agg['matches_bench_unused']
            cum_matches_not_selected += day_agg['matches_not_selected']
            cum_matches_injured += day_agg['matches_injured']
            cum_disciplinary = max(cum_disciplinary, day_agg['disciplinary_action'])
            
            # Career totals
            features['matches'][i] = day_agg['matches_played']
            features['career_matches'][i] = cum_matches_played
            features['career_goals'][i] = cum_goals
            features['career_assists'][i] = cum_assists
            features['career_minutes'][i] = cum_minutes
        
        if not day_matches.empty:
            
            # OPTIMIZATION: Use itertuples() for row-by-row operations (5-10x faster than iterrows)
            for match in day_matches.itertuples(index=False):
                
                # OPTIMIZATION: Use pre-normalized position from preprocessing (already normalized in preprocess_matches)
                # This avoids thousands of string operations and function calls in the loop
                match_position = getattr(match, 'position', '') if hasattr(match, 'position') else ''
                if pd.notna(match_position) and match_position != '' and str(match_position).strip() != '':
                    normalized_pos = str(match_position).strip()
                    
                    # Only re-normalize if it looks like an abbreviation (fallback for edge cases)
                    # Most positions should already be normalized from preprocessing
                    if len(normalized_pos) <= 3 and normalized_pos.isupper():
                        position_lower = normalized_pos.lower()
                        if position_lower in abbrev_map:
                            normalized_pos = abbrev_map[position_lower]
                    
                    # FIXED: Validate that normalized position is in the canonical list
                    # Handle None/empty from normalization
                    if normalized_pos and normalized_pos != '' and normalized_pos != 'None' and normalized_pos in VALID_POSITIONS:
                        features['last_match_position'][i] = normalized_pos
                        # Compare normalized positions (both should be normalized to same format)
                        if normalized_pos == default_position:
                            features['position_match_default'][i] = 1
                
                # Competition importance
                comp_importance = getattr(match, 'competition_importance', 0) if hasattr(match, 'competition_importance') else 0
                features['competition_importance'][i] = max(features['competition_importance'][i], comp_importance)
                
                # Disciplinary
                match_disciplinary = getattr(match, 'disciplinary_action', 0) if hasattr(match, 'disciplinary_action') else 0
                features['disciplinary_action'][i] = max(features['disciplinary_action'][i], match_disciplinary)
                
                # Competitions tracking
                comp_name = getattr(match, 'competition', '') if hasattr(match, 'competition') else ''
                if pd.notna(comp_name) and comp_name != '':
                    competitions_seen.add(comp_name)
                    season = get_football_season(date_norm)
                    if season == current_season:
                        competitions_this_season.add(comp_name)
                    elif current_season:
                        prev_season = get_football_season(pd.Timestamp(f"{current_season.split('/')[0]}-07-01") - timedelta(days=365))
                        if season == prev_season:
                            competitions_last_season.add(comp_name)
                
                # National team detection - define match_home_team and match_away_team here first since they're used in multiple places
                match_home_team = getattr(match, 'home_team', '') if hasattr(match, 'home_team') else ''
                match_away_team = getattr(match, 'away_team', '') if hasattr(match, 'away_team') else ''
                is_national_match = is_national_team(match_home_team) or is_national_team(match_away_team)
                
                if is_national_match:
                    national_matches_count += 1
                    match_minutes = getattr(match, 'minutes_played_numeric', 0) if hasattr(match, 'minutes_played_numeric') else 0
                    national_minutes_total += match_minutes
                    last_national_match_date = date_norm
                    
                    # Determine if senior or youth
                    comp_lower = str(comp_name).lower()
                    is_senior = not any(indicator in comp_lower for indicator in ['u19', 'u20', 'u21', 'u23', 'youth', 'u-'])
                    if is_senior:
                        # Set to 1 if ever played for senior team (cumulative)
                        if features['senior_national_team'][i] == 0:
                            features['senior_national_team'][i] = 1
                
                # Club matching for club features using normalized names
                # Convert itertuples NamedTuple to dict-like access for identify_player_team
                match_dict = {col: getattr(match, col, None) for col in day_matches.columns if hasattr(match, col)}
                player_team = identify_player_team(match_dict, current_club)
                
                # Initialize match variables to avoid UnboundLocalError when player_team is None
                # These are used in substitution features even if player_team couldn't be identified
                match_goals = getattr(match, 'goals_numeric', 0) if hasattr(match, 'goals_numeric') else 0
                match_assists = getattr(match, 'assists_numeric', 0) if hasattr(match, 'assists_numeric') else 0
                match_matches = getattr(match, 'matches_played', 0) if hasattr(match, 'matches_played') else 0
                
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
                    
                    # Re-assign match variables for club stats (already initialized above for substitution features)
                    match_minutes = getattr(match, 'minutes_played_numeric', 0) if hasattr(match, 'minutes_played_numeric') else 0
                    match_yellow = getattr(match, 'yellow_cards_numeric', 0) if hasattr(match, 'yellow_cards_numeric') else 0
                    match_red = getattr(match, 'red_cards_numeric', 0) if hasattr(match, 'red_cards_numeric') else 0
                    
                    club_cum_goals += match_goals
                    club_cum_assists += match_assists
                    club_cum_minutes += match_minutes
                    club_cum_matches += match_matches
                    club_cum_yellow += match_yellow
                    club_cum_red += match_red
                
                # Substitution features (only for matches where player played)
                # Only process substitutions if player actually played in the match
                if match_matches > 0:
                    sub_on_min = getattr(match, 'substitution_on_minute', None) if hasattr(match, 'substitution_on_minute') else None
                    sub_off_min = getattr(match, 'substitution_off_minute', None) if hasattr(match, 'substitution_off_minute') else None
                    
                    if pd.notna(sub_on_min) or pd.notna(sub_off_min):
                        day_has_substitution = True
                    
                    if pd.notna(sub_on_min):
                        features['substitution_on_count'][i] += 1
                        if sub_on_min >= 60:  # Late substitution
                            features['late_substitution_on_count'][i] += 1
                        if match_goals > 0 or match_assists > 0:
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
                match_home_team = getattr(match, 'home_team', None) if hasattr(match, 'home_team') else None
                match_away_team = getattr(match, 'away_team', '') if hasattr(match, 'away_team') else ''
                if pd.notna(match_home_team):
                    # Determine if player's team is home or away using normalized names
                    if player_team:
                        # Use normalized comparison
                        if normalize_team_name(player_team) == normalize_team_name(match_home_team):
                            features['home_matches'][i] += 1
                        elif normalize_team_name(player_team) == normalize_team_name(match_away_team):
                            features['away_matches'][i] += 1
                        # Fallback to direct comparison
                        elif player_team == match_home_team:
                            features['home_matches'][i] += 1
                        elif player_team == match_away_team:
                            features['away_matches'][i] += 1
                
                # Team results
                result = getattr(match, 'result', '') if hasattr(match, 'result') else ''
                if pd.notna(result) and result != '':
                    # Parse result (e.g., "2:1", "1:1", "0:3", "2-1", "6:7 on pens")
                    try:
                        result_str = str(result).strip()
                        
                        # Remove extra text like "on pens", "a.e.t.", etc.
                        result_str = re.sub(r'\s+on\s+pens.*$', '', result_str, flags=re.IGNORECASE)
                        result_str = re.sub(r'\s+a\.e\.t\..*$', '', result_str, flags=re.IGNORECASE)
                        result_str = re.sub(r'\s+after\s+extra\s+time.*$', '', result_str, flags=re.IGNORECASE)
                        result_str = result_str.strip()
                        
                        # Try colon separator first (most common in Transfermarkt data)
                        separator = None
                        if ':' in result_str:
                            separator = ':'
                        elif '-' in result_str:
                            separator = '-'
                        
                        if separator:
                            parts = result_str.split(separator)
                            if len(parts) == 2:
                                home_score = int(parts[0].strip())
                                away_score = int(parts[1].strip())
                                
                                # Determine if player's team won/drew/lost using normalized names
                                # match_home_team and match_away_team are already defined above
                                if player_team:
                                    # Use normalized comparison
                                    is_home = (normalize_team_name(player_team) == normalize_team_name(match_home_team)) or (player_team == match_home_team)
                                    is_away = (normalize_team_name(player_team) == normalize_team_name(match_away_team)) or (player_team == match_away_team)
                                    
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
                    except Exception as e:
                        logger.debug(f"Error parsing result '{result}': {e}")
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
        
        # OPTIMIZATION: Binary search for past matches (O(log n) instead of O(n))
        if len(matches_sorted_dates) > 0:
            # Ensure date_norm is a numpy datetime64 for proper comparison
            date_norm_np = np.datetime64(date_norm)
            idx = np.searchsorted(matches_sorted_dates, date_norm_np, side='right')
            past_matches = matches_sorted.iloc[:idx] if idx > 0 else pd.DataFrame()
        else:
            past_matches = pd.DataFrame()
        
        # Average competition importance
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
        # Update teams this season from matches (using pre-indexed sorted matches)
        if matches_sorted is not None and not matches_sorted.empty:
            past_matches = matches_sorted[matches_sorted['date_norm'] <= date_norm]
            if not past_matches.empty:
                season_matches = past_matches[past_matches['date_norm'].apply(lambda d: get_football_season(pd.Timestamp(d)) == season)]
            else:
                season_matches = pd.DataFrame()  # Initialize if past_matches is empty
        else:
            past_matches = pd.DataFrame()
            season_matches = pd.DataFrame()
        # OPTIMIZATION: Use itertuples() instead of iterrows()
        if not season_matches.empty:
            for match in season_matches.itertuples(index=False):
                for team_col in ['home_team', 'away_team', 'team']:
                    if hasattr(match, team_col):
                        team_name = str(getattr(match, team_col)).strip()
                        if pd.notna(team_name) and team_name != '' and not is_national_team(team_name):
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
    
    # Summary statistics
    logger.debug("Match features summary:")
    logger.debug(f"  Total matches played: {cum_matches_played}")
    logger.debug(f"  Total goals: {cum_goals}")
    logger.debug(f"  Total assists: {cum_assists}")
    logger.debug(f"  Unique competitions: {len(competitions_seen)}")
    logger.debug(f"  National team matches: {national_matches_count}")
    logger.debug(f"  Club changes detected: {len(set([f for f in features['club_cum_matches_played'] if f > 0]))}")
    
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
                'days_since_last_injury': [-1] * n_days,
                'days_since_last_injury_ended': [-1] * n_days,
                'avg_injury_duration': [0.0] * n_days,
                'injury_frequency': [0.0] * n_days,
                'avg_injury_severity': [0.0] * n_days,
                'max_injury_severity': [0] * n_days,
                'short_term_injury_ratio': [0.0] * n_days,
                
                # By injury class (ALL classes treated equally)
                'muscular_injury_count': [0] * n_days,
                'skeletal_injury_count': [0] * n_days,
                'unknown_injury_count': [0] * n_days,
                'other_injuries': [0] * n_days,
                'muscular_injury_days': [0] * n_days,
                'skeletal_injury_days': [0] * n_days,
                'unknown_injury_days': [0] * n_days,
                'other_injury_days': [0] * n_days,
                'days_since_last_muscular': [-1] * n_days,
                'days_since_last_skeletal': [-1] * n_days,
                'days_since_last_unknown': [-1] * n_days,
                'days_since_last_other': [-1] * n_days,
                
                # By body part (all injuries) - NOTE: Column names without _count suffix to match reference
                'lower_leg_injuries': [0] * n_days,
                'knee_injuries': [0] * n_days,
                'upper_leg_injuries': [0] * n_days,
                'hip_injuries': [0] * n_days,
                'upper_body_injuries': [0] * n_days,
                'head_injuries': [0] * n_days,
                'illness_count': [0] * n_days,
                'unknown_body_part_count': [0] * n_days,
                'days_since_last_lower_leg': [-1] * n_days,
                'days_since_last_knee': [-1] * n_days,
                'days_since_last_upper_leg': [-1] * n_days,
                'days_since_last_hip': [-1] * n_days,
                'days_since_last_upper_body': [-1] * n_days,
                'days_since_last_head': [-1] * n_days,
                'days_since_last_illness': [-1] * n_days,
                
                # By severity (all injuries)
                'mild_injury_count': [0] * n_days,
                'moderate_injury_count': [0] * n_days,
                'severe_injury_count': [0] * n_days,
                'critical_injury_count': [0] * n_days,
                'days_since_last_mild': [-1] * n_days,
                'days_since_last_moderate': [-1] * n_days,
                'days_since_last_severe': [-1] * n_days,
                'days_since_last_critical': [-1] * n_days,
                
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
        
        # OPTIMIZATION: Pre-compute sorted injury dates for binary search
        injury_start_dates = all_injuries['start_norm'].dropna()
        if len(injury_start_dates) > 0:
            # Convert to pandas Timestamp array for proper comparison with np.searchsorted
            injury_start_dates_sorted = pd.to_datetime(injury_start_dates).values
            injury_start_dates_sorted = np.sort(injury_start_dates_sorted)
        else:
            injury_start_dates_sorted = np.array([])
        logger.debug(f"Pre-computed sorted injury dates for binary search: {len(injury_start_dates_sorted)} injuries")
        
        # Initialize features
        features = {
            # Base features (all injuries)
            'cum_inj_starts': [0] * n_days,
            'cum_inj_days': [0] * n_days,
            'days_since_last_injury': [-1] * n_days,
            'days_since_last_injury_ended': [-1] * n_days,
            'avg_injury_duration': [0.0] * n_days,
            'injury_frequency': [0.0] * n_days,
            'avg_injury_severity': [0.0] * n_days,
            'max_injury_severity': [0] * n_days,
            'short_term_injury_ratio': [0.0] * n_days,
            
            # By injury class (ALL classes treated equally - NO TARGET/FEATURE distinction)
            'muscular_injury_count': [0] * n_days,
            'skeletal_injury_count': [0] * n_days,
            'unknown_injury_count': [0] * n_days,
            'other_injuries': [0] * n_days,
            'muscular_injury_days': [0] * n_days,
            'skeletal_injury_days': [0] * n_days,
            'unknown_injury_days': [0] * n_days,
            'other_injury_days': [0] * n_days,
            'days_since_last_muscular': [-1] * n_days,
            'days_since_last_skeletal': [-1] * n_days,
            'days_since_last_unknown': [-1] * n_days,
            'days_since_last_other': [-1] * n_days,
            
            # By body part (all injuries) - NOTE: Column names without _count suffix to match reference
            'lower_leg_injuries': [0] * n_days,
            'knee_injuries': [0] * n_days,
            'upper_leg_injuries': [0] * n_days,
            'hip_injuries': [0] * n_days,
            'upper_body_injuries': [0] * n_days,
            'head_injuries': [0] * n_days,
            'illness_count': [0] * n_days,
            'unknown_body_part_count': [0] * n_days,
            'days_since_last_lower_leg': [-1] * n_days,
            'days_since_last_knee': [-1] * n_days,
            'days_since_last_upper_leg': [-1] * n_days,
            'days_since_last_hip': [-1] * n_days,
            'days_since_last_upper_body': [-1] * n_days,
            'days_since_last_head': [-1] * n_days,
            'days_since_last_illness': [-1] * n_days,
            
            # By severity (all injuries)
            'mild_injury_count': [0] * n_days,
            'moderate_injury_count': [0] * n_days,
            'severe_injury_count': [0] * n_days,
            'critical_injury_count': [0] * n_days,
            'days_since_last_mild': [-1] * n_days,
            'days_since_last_moderate': [-1] * n_days,
            'days_since_last_severe': [-1] * n_days,
            'days_since_last_critical': [-1] * n_days,
            
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
            
            # OPTIMIZATION: Binary search for injuries (O(log n) instead of O(n))
            if len(injury_start_dates_sorted) > 0:
                # Ensure date_norm is a numpy datetime64 for proper comparison
                date_norm_np = np.datetime64(date_norm)
                idx = np.searchsorted(injury_start_dates_sorted, date_norm_np, side='right')
                past_injuries = all_injuries.iloc[:idx] if idx > 0 else pd.DataFrame()
            else:
                past_injuries = pd.DataFrame()
            
            # Update injury tracking dictionaries (only for new injuries)
            if len(past_injuries) > injury_idx:
                for inj_idx in range(injury_idx, len(past_injuries)):
                    inj = past_injuries.iloc[inj_idx]
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
                
                injury_idx = len(past_injuries)
                
                # Update last_injury_date from most recent injury
                if len(past_injuries) > 0:
                    last_injury_date = past_injuries.iloc[-1]['start_norm']
            
            # Find the most recent recovery date from all past injuries that have ended
            # Reset last_recovery_date at the start of each day
            last_recovery_date = None
            if len(past_injuries) > 0:
                # Get all injuries that have ended on or before this date
                ended_injuries = past_injuries[
                    (past_injuries['end_norm'].notna()) & 
                    (past_injuries['end_norm'] <= date_norm)
                ]
                if len(ended_injuries) > 0:
                    last_recovery_date = ended_injuries['end_norm'].max()
            
            if len(past_injuries) > 0:
                # Base counts
                features['cum_inj_starts'][i] = len(past_injuries)
                
                # Injury days - calculate total days injured up to current date (VECTORIZED)
                if len(past_injuries) > 0:
                    # Vectorized calculation: cap end dates and calculate days
                    end_dates = past_injuries['end_norm'].fillna(date_norm)
                    end_dates = end_dates.clip(upper=date_norm)  # Cap at current date
                    start_dates = past_injuries['start_norm']
                    # Only count injuries that started on or before current date
                    valid_mask = start_dates <= date_norm
                    if valid_mask.any():
                        injury_days = ((end_dates[valid_mask] - start_dates[valid_mask]).dt.days + 1).sum()
                        features['cum_inj_days'][i] = int(injury_days)
                    else:
                        features['cum_inj_days'][i] = 0
                else:
                    features['cum_inj_days'][i] = 0
                
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
                    # Use 'other_injuries' instead of 'other_injury_count' for consistency
                    if class_name == 'other':
                        features['other_injuries'][i] = len(class_injuries)
                    else:
                        features[f'{class_name}_injury_count'][i] = len(class_injuries)
                    
                    # Injury days by class (VECTORIZED)
                    if len(class_injuries) > 0:
                        end_dates = class_injuries['end_norm'].fillna(date_norm)
                        end_dates = end_dates.clip(upper=date_norm)
                        start_dates = class_injuries['start_norm']
                        valid_mask = start_dates <= date_norm
                        if valid_mask.any():
                            class_days = ((end_dates[valid_mask] - start_dates[valid_mask]).dt.days + 1).sum()
                            features[f'{class_name}_injury_days'][i] = int(class_days)
                        else:
                            features[f'{class_name}_injury_days'][i] = 0
                    else:
                        features[f'{class_name}_injury_days'][i] = 0
                
                # By body part (handle NaN) - NOTE: Column names without _count suffix
                body_parts = ['lower_leg', 'knee', 'upper_leg', 'hip', 'upper_body', 'head', 'illness']
                for body_part in body_parts:
                    bp_injuries = past_injuries[
                        past_injuries['body_part'].astype(str).str.lower().fillna('') == body_part
                    ]
                    # Use column name without _count suffix (except for illness_count)
                    if body_part == 'illness':
                        features['illness_count'][i] = len(bp_injuries)
                    else:
                        features[f'{body_part}_injuries'][i] = len(bp_injuries)
                
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
                        # Handle 'other' class which uses 'other_injuries' instead of 'other_injury_count'
                        if class_name == 'other':
                            class_count = features['other_injuries'][i]
                        else:
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

def transform_injury_recency_features(injury_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log transformation to all days_since_last_* injury features.
    
    This reduces over-reliance on players without injuries for a long time
    by applying log(1 + x) transformation. Preserves -1 as sentinel for "never happened".
    
    Args:
        injury_df: DataFrame with injury features including days_since_last_* columns
        
    Returns:
        DataFrame with transformed injury recency features
    """
    logger.info("=== Applying log transformation to injury recency features ===")
    
    # List of all days_since_last_* features to transform
    injury_recency_features = [
        'days_since_last_injury',
        'days_since_last_injury_ended',
        'days_since_last_muscular',
        'days_since_last_skeletal',
        'days_since_last_unknown',
        'days_since_last_other',
        'days_since_last_lower_leg',
        'days_since_last_knee',
        'days_since_last_upper_leg',
        'days_since_last_hip',
        'days_since_last_upper_body',
        'days_since_last_head',
        'days_since_last_illness',
        'days_since_last_mild',
        'days_since_last_moderate',
        'days_since_last_severe',
        'days_since_last_critical',
    ]
    
    # Create a copy to avoid modifying the original
    transformed_df = injury_df.copy()
    
    # Apply log transformation to each feature
    transformed_count = 0
    for feature in injury_recency_features:
        if feature in transformed_df.columns:
            # FIXED: Preserve -1 values (never happened) and only transform non-negative values
            # This ensures:
            #   -1 (no injury) -> -1 (preserved as sentinel)
            #   0 (injury today) -> log(1 + 0) = 0
            #   1 day -> log(1 + 1) = log(2) ≈ 0.69
            #   30 days -> log(1 + 30) = log(31) ≈ 3.43
            #   365 days -> log(1 + 365) = log(366) ≈ 5.90
            #   730 days -> log(1 + 730) = log(731) ≈ 6.59
            original_values = transformed_df[feature].copy()
            
            # FIXED: Convert to float64 to avoid dtype warnings when assigning log-transformed values
            # This preserves -1 as float -1.0, which is fine for the sentinel value
            transformed_df[feature] = transformed_df[feature].astype('float64')
            
            # Preserve -1 values, only transform non-negative values
            mask = transformed_df[feature] >= 0
            transformed_df.loc[mask, feature] = np.log1p(transformed_df.loc[mask, feature])
            
            transformed_count += 1
            neg_one_count = (transformed_df[feature] == -1).sum()
            logger.debug(f"Transformed {feature}: min={transformed_df[feature].min():.2f}, max={transformed_df[feature].max():.2f}, "
                        f"original_range=[{original_values.min()}, {original_values.max()}], "
                        f"-1_count={neg_one_count}")
        else:
            logger.warning(f"Feature {feature} not found in injury features DataFrame")
    
    logger.info(f"Applied log transformation to {transformed_count} injury recency features")
    return transformed_df

def calculate_interaction_features(
    profile_df: pd.DataFrame,
    match_df: pd.DataFrame,
    injury_df: pd.DataFrame
) -> pd.DataFrame:
    """Calculate interaction features."""
    logger.info("=== Starting calculate_interaction_features ===")
    logger.debug(f"Profile features: {len(profile_df)} rows, {len(profile_df.columns)} columns")
    logger.debug(f"Match features: {len(match_df)} rows, {len(match_df.columns)} columns")
    logger.debug(f"Injury features: {len(injury_df)} rows, {len(injury_df.columns)} columns")
    
    features = {}
    
    # Age x career matches
    if 'age' in profile_df.columns and 'career_matches' in match_df.columns:
        features['age_x_career_matches'] = profile_df['age'] * match_df['career_matches']
        logger.debug(f"Calculated age_x_career_matches: min={features['age_x_career_matches'].min():.2f}, max={features['age_x_career_matches'].max():.2f}")
    else:
        logger.warning("Missing columns for age_x_career_matches")
        features['age_x_career_matches'] = [0.0] * len(profile_df)
    
    # Age x career goals
    if 'age' in profile_df.columns and 'career_goals' in match_df.columns:
        features['age_x_career_goals'] = profile_df['age'] * match_df['career_goals']
        logger.debug(f"Calculated age_x_career_goals: min={features['age_x_career_goals'].min():.2f}, max={features['age_x_career_goals'].max():.2f}")
    else:
        logger.warning("Missing columns for age_x_career_goals")
        features['age_x_career_goals'] = [0.0] * len(profile_df)
    
    # Seniority x goals per match
    if 'seniority_days' in profile_df.columns and 'goals_per_match' in match_df.columns:
        features['seniority_x_goals_per_match'] = profile_df['seniority_days'] * match_df['goals_per_match']
        logger.debug(f"Calculated seniority_x_goals_per_match: min={features['seniority_x_goals_per_match'].min():.2f}, max={features['seniority_x_goals_per_match'].max():.2f}")
    else:
        logger.warning("Missing columns for seniority_x_goals_per_match")
        features['seniority_x_goals_per_match'] = [0.0] * len(profile_df)
    
    # Club seniority x goals per match
    if 'seniority_days' in profile_df.columns and 'club_goals_per_match' in match_df.columns:
        features['club_seniority_x_goals_per_match'] = profile_df['seniority_days'] * match_df['club_goals_per_match']
        logger.debug(f"Calculated club_seniority_x_goals_per_match: min={features['club_seniority_x_goals_per_match'].min():.2f}, max={features['club_seniority_x_goals_per_match'].max():.2f}")
    else:
        logger.warning("Missing columns for club_seniority_x_goals_per_match")
        features['club_seniority_x_goals_per_match'] = [0.0] * len(profile_df)
    
    logger.debug(f"Generated {len(features)} interaction features")
    logger.info("=== Completed calculate_interaction_features ===")
    return pd.DataFrame(features, index=profile_df.index)

def generate_daily_features_for_player(
    player_id: int,
    data_dir: str = DATA_DIR,
    reference_date: pd.Timestamp = REFERENCE_DATE,
    output_dir: str = OUTPUT_DIR
) -> pd.DataFrame:
    """Generate daily features for a single player."""
    # OPTIMIZATION: Track total time for file generation
    file_start_time = time.time()
    logger.info(f"=== Generating daily features for player {player_id} ===")
    logger.debug(f"Parameters: data_dir={data_dir}, reference_date={reference_date}, output_dir={output_dir}")
    
    # Load data
    step_start = time.time()
    logger.info("[Step 1] Loading player data...")
    data = load_player_data(player_id, data_dir, reference_date)
    step_time = time.time() - step_start
    logger.debug(f"[Step 1] Completed in {step_time:.2f} seconds")
    players = data['players']
    injuries = data['injuries']
    matches = data['matches']
    career = data['career']
    team_country_map = data['team_country_map']
    
    if players.empty:
        logger.error(f"[ERROR] Player {player_id} not found")
        return pd.DataFrame()
    
    player_row = players.iloc[0]
    logger.debug(f"Player name: {player_row.get('name', 'Unknown')}")
    
    # Preprocess matches
    step_start = time.time()
    logger.info("[Step 2] Preprocessing matches...")
    matches = preprocess_matches(matches)
    step_time = time.time() - step_start
    logger.debug(f"[Step 2] Completed in {step_time:.2f} seconds")
    
    # Determine calendar
    step_start = time.time()
    logger.info("[Step 3] Determining calendar...")
    calendar = determine_calendar(matches, injuries, reference_date, player_row)
    step_time = time.time() - step_start
    logger.debug(f"[Step 3] Completed in {step_time:.2f} seconds")
    
    # Calculate features
    step_start = time.time()
    logger.info("[Step 4] Calculating profile features...")
    profile_df = calculate_profile_features(player_row, calendar, matches, career, team_country_map)
    step_time = time.time() - step_start
    logger.debug(f"[Step 4] Profile features: {len(profile_df)} rows × {len(profile_df.columns)} columns | Completed in {step_time:.2f} seconds")
    
    step_start = time.time()
    logger.info("[Step 5] Calculating match features...")
    match_df = calculate_match_features(matches, calendar, player_row, team_country_map)
    step_time = time.time() - step_start
    logger.debug(f"[Step 5] Match features: {len(match_df)} rows × {len(match_df.columns)} columns | Completed in {step_time:.2f} seconds")
    
    step_start = time.time()
    logger.info("[Step 6] Calculating injury features...")
    injury_df = calculate_injury_features(injuries, calendar)  # Single consolidated function
    step_time = time.time() - step_start
    logger.debug(f"[Step 6] Injury features: {len(injury_df)} rows × {len(injury_df.columns)} columns | Completed in {step_time:.2f} seconds")
    
    step_start = time.time()
    logger.info("[Step 6.5] Applying log transformation to injury recency features...")
    injury_df = transform_injury_recency_features(injury_df)  # V4: Apply log transformation
    step_time = time.time() - step_start
    logger.debug(f"[Step 6.5] Transformation completed in {step_time:.2f} seconds")
    
    step_start = time.time()
    logger.info("[Step 7] Calculating interaction features...")
    interaction_df = calculate_interaction_features(profile_df, match_df, injury_df)
    step_time = time.time() - step_start
    logger.debug(f"[Step 7] Interaction features: {len(interaction_df)} rows × {len(interaction_df.columns)} columns | Completed in {step_time:.2f} seconds")
    
    # Combine all features
    step_start = time.time()
    logger.info("[Step 8] Combining all features...")
    daily_features = pd.concat([profile_df, match_df, injury_df, interaction_df], axis=1)
    step_time = time.time() - step_start
    logger.debug(f"[Step 8] Combined features: {len(daily_features)} rows × {len(daily_features.columns)} columns | Completed in {step_time:.2f} seconds")
    
    # Remove duplicate columns
    before_cols = len(daily_features.columns)
    daily_features = daily_features.loc[:, ~daily_features.columns.duplicated()]
    after_cols = len(daily_features.columns)
    if before_cols != after_cols:
        logger.warning(f"Removed {before_cols - after_cols} duplicate columns")
    
    # Rename injury columns to match reference file format (if any old names exist)
    column_rename_map = {
        'lower_leg_injury_count': 'lower_leg_injuries',
        'knee_injury_count': 'knee_injuries',
        'upper_leg_injury_count': 'upper_leg_injuries',
        'hip_injury_count': 'hip_injuries',
        'upper_body_injury_count': 'upper_body_injuries',
        'head_injury_count': 'head_injuries',
        'other_injury_count': 'other_injuries',
    }
    
    # Only rename columns that exist
    rename_map = {k: v for k, v in column_rename_map.items() if k in daily_features.columns}
    if rename_map:
        logger.info(f"Renaming {len(rename_map)} columns to match reference format: {list(rename_map.keys())}")
        daily_features = daily_features.rename(columns=rename_map)
    
    # Total time for file generation
    total_time = time.time() - file_start_time
    logger.info(f"=== Completed generating daily features for player {player_id} ===")
    logger.info(f"⏱️  TOTAL TIME: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    logger.info(f"[COMPLETE] Generated {len(daily_features)} rows, {len(daily_features.columns)} columns")
    logger.debug(f"Feature columns: {list(daily_features.columns)[:10]}..." if len(daily_features.columns) > 10 else f"Feature columns: {list(daily_features.columns)}")
    
    return daily_features

def print_progress_bar(current, total, bar_length=50, prefix="Progress"):
    """Print a progress bar."""
    if total == 0:
        return
    percent = float(current) / total
    filled_length = int(bar_length * percent)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    percent_str = f"{percent * 100:.1f}%"
    return f"{prefix}: |{bar}| {current}/{total} ({percent_str})"

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Generate daily features for football players')
    parser.add_argument('--player-id', type=int, default=None, help='Player ID to process (if not provided, processes all players)')
    parser.add_argument('--all-players', action='store_true', help='Process all players from profile file')
    parser.add_argument('--data-dir', type=str, default=DATA_DIR, help='Data directory')
    parser.add_argument('--reference-date', type=str, default=str(REFERENCE_DATE.date()), help='Reference date (YYYY-MM-DD)')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR, help='Output directory')
    parser.add_argument('--force', action='store_true', help='Force regeneration')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging (DEBUG level)')
    parser.add_argument('--skip-existing', action='store_true', help='Skip players whose files already exist (unless --force is used)')
    
    args = parser.parse_args()
    
    # Setup logging with verbose option
    global logger
    logger = setup_logging(verbose=args.verbose)
    
    logger.info("=" * 70)
    logger.info("DAILY FEATURES GENERATOR")
    logger.info("=" * 70)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Reference date: {args.reference_date}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Verbose mode: {args.verbose}")
    logger.info(f"Force regeneration: {args.force}")
    logger.info(f"Skip existing: {args.skip_existing}")
    logger.info("=" * 70)
    
    # Create output directory
    logger.debug(f"Creating output directory: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    logger.debug(f"Output directory ready: {os.path.exists(args.output_dir)}")
    
    # Parse reference date
    logger.debug(f"Parsing reference date: {args.reference_date}")
    reference_date = pd.Timestamp(args.reference_date)
    logger.debug(f"Parsed reference date: {reference_date}")
    
    # Determine which players to process
    if args.all_players or args.player_id is None:
        # Load all player IDs from profile file
        players_path = os.path.join(args.data_dir, 'players_profile.csv')
        logger.info(f"Loading player list from: {players_path}")
        if not os.path.exists(players_path):
            logger.error(f"Profile file not found: {players_path}")
            return
        
        try:
            players_df = pd.read_csv(players_path, sep=';', encoding='utf-8')
            player_ids = players_df['id'].unique().tolist()
            logger.info(f"Found {len(player_ids)} unique players in profile file")
        except Exception as e:
            logger.error(f"Error loading profile file: {e}")
            return
    else:
        # Process single player
        player_ids = [args.player_id]
        logger.info(f"Processing single player: {args.player_id}")
    
    # Process all players
    total_players = len(player_ids)
    successful = 0
    failed = 0
    skipped = 0
    total_time = 0
    total_rows = 0
    total_size_mb = 0
    
    logger.info("=" * 70)
    logger.info(f"🚀 Starting batch processing for {total_players} players")
    logger.info("=" * 70)
    
    overall_start_time = time.time()
    last_progress_log = overall_start_time
    
    for idx, player_id in enumerate(player_ids, 1):
        player_start_time = time.time()
        
        # Output file path
        output_file = os.path.join(args.output_dir, f'player_{player_id}_daily_features.csv')
        
        # Check if file exists
        if os.path.exists(output_file) and not args.force:
            if args.skip_existing:
                file_size = os.path.getsize(output_file) / (1024 * 1024)  # Size in MB
                logger.info(f"[{idx}/{total_players}] ⏭️  SKIPPED: Player {player_id} (file exists: {file_size:.2f} MB)")
                skipped += 1
                # Update progress visualization
                current_time = time.time()
                if current_time - last_progress_log >= 5:  # Log progress every 5 seconds
                    elapsed_total = current_time - overall_start_time
                    progress_bar = print_progress_bar(idx, total_players, prefix="Overall")
                    logger.info(progress_bar)
                    last_progress_log = current_time
                continue
            else:
                file_size = os.path.getsize(output_file) / (1024 * 1024)  # Size in MB
                logger.info(f"[{idx}/{total_players}] ⚠️  WARNING: File already exists: {output_file} ({file_size:.2f} MB)")
                logger.info("Use --force to regenerate or --skip-existing to skip")
                skipped += 1
                continue
        elif os.path.exists(output_file) and args.force:
            logger.info(f"[{idx}/{total_players}] 🔄 FORCE: Regenerating existing file for player {player_id}")
        
        # Overall progress visualization (every 5 seconds or every 10 players)
        current_time = time.time()
        if (current_time - last_progress_log >= 5) or (idx % 10 == 0):
            elapsed_total = current_time - overall_start_time
            progress_pct = (idx * 100) // total_players
            avg_time_per_player = elapsed_total / idx if idx > 0 else 0
            remaining_players = total_players - idx
            eta_seconds = remaining_players * avg_time_per_player
            eta_str = str(timedelta(seconds=int(eta_seconds)))
            elapsed_str = str(timedelta(seconds=int(elapsed_total)))
            
            progress_bar = print_progress_bar(idx, total_players, prefix="Overall")
            logger.info("")
            logger.info("=" * 70)
            logger.info("📊 OVERALL PROGRESS")
            logger.info("=" * 70)
            logger.info(progress_bar)
            logger.info(f"   ⏱️  Elapsed: {elapsed_str} | ETA: {eta_str}")
            logger.info(f"   📈 Success: {successful} | Failed: {failed} | Skipped: {skipped}")
            if successful > 0:
                logger.info(f"   ⚡ Avg time/player: {total_time/successful:.1f}s | Rate: {successful/(elapsed_total/3600):.1f} players/hour")
            logger.info("=" * 70)
            logger.info("")
            last_progress_log = current_time
        
        # Per-player progress
        progress_pct = (idx * 100) // total_players
        logger.info(f"[{idx}/{total_players}] ({progress_pct}%) 🎯 Processing player {player_id}...")
        
        try:
            daily_features = generate_daily_features_for_player(
                player_id=player_id,
                data_dir=args.data_dir,
                reference_date=reference_date,
                output_dir=args.output_dir
            )
            
            if daily_features.empty:
                logger.warning(f"[{idx}/{total_players}] ⚠️  No features generated for player {player_id}")
                failed += 1
                continue
            
            # FIXED: Save to CSV with date as a column (not index)
            logger.debug(f"Saving features to CSV: {output_file}")
            # Always reset index first
            if isinstance(daily_features.index, pd.DatetimeIndex) and 'date' not in daily_features.columns:
                # Create date column from index
                daily_features = daily_features.reset_index()
                # Rename the index column (might be unnamed)
                index_col = None
                for col in daily_features.columns:
                    if col == '' or (isinstance(col, float) and pd.isna(col)) or col == daily_features.index.name:
                        index_col = col
                        break
                if index_col:
                    daily_features = daily_features.rename(columns={index_col: 'date'})
                elif len(daily_features.columns) > 0 and daily_features.columns[0] not in ['player_id', 'date']:
                    daily_features = daily_features.rename(columns={daily_features.columns[0]: 'date'})
            else:
                # Date column exists or no DatetimeIndex, just drop the index
                daily_features = daily_features.reset_index(drop=True)
                # Remove any unnamed columns (empty string or NaN column names) that might contain dates
                cols_to_drop = [col for col in daily_features.columns 
                                if (col == '' or (isinstance(col, float) and pd.isna(col)))]
                if cols_to_drop:
                    logger.debug(f"Removing {len(cols_to_drop)} unnamed column(s) from DataFrame: {cols_to_drop}")
                    daily_features = daily_features.drop(columns=cols_to_drop)
            daily_features.to_csv(output_file, index=False, encoding='utf-8-sig')
            file_size = os.path.getsize(output_file) / (1024 * 1024)  # Size in MB
            logger.debug(f"File saved: {file_size:.2f} MB")
            
            player_elapsed = time.time() - player_start_time
            total_time += player_elapsed
            total_rows += len(daily_features)
            total_size_mb += file_size
            
            successful += 1
            logger.info(f"[{idx}/{total_players}] ✅ SUCCESS: Player {player_id}")
            logger.info(f"   📁 Output: {os.path.basename(output_file)}")
            logger.info(f"   📊 Rows: {len(daily_features):,}, Columns: {len(daily_features.columns)}")
            logger.info(f"   💾 File size: {file_size:.2f} MB")
            logger.info(f"   ⏱️  Time: {player_elapsed:.2f}s ({player_elapsed/60:.2f} min)")
            logger.info(f"   📈 Rate: {len(daily_features)/player_elapsed:.0f} rows/sec")
            
        except Exception as e:
            failed += 1
            logger.error(f"[{idx}/{total_players}] ❌ ERROR: Player {player_id}")
            logger.error(f"   Error: {str(e)}")
            if args.verbose:
                logger.error(traceback.format_exc())
            # Continue with next player instead of stopping
            continue
    
    # Final summary
    total_elapsed = time.time() - overall_start_time
    logger.info("")
    logger.info("=" * 70)
    logger.info("🎉 BATCH PROCESSING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"📊 Total players: {total_players}")
    logger.info(f"   ✅ Successful: {successful} ({successful*100//total_players if total_players > 0 else 0}%)")
    logger.info(f"   ❌ Failed: {failed} ({failed*100//total_players if total_players > 0 else 0}%)")
    logger.info(f"   ⏭️  Skipped: {skipped} ({skipped*100//total_players if total_players > 0 else 0}%)")
    logger.info(f"")
    logger.info(f"⏱️  Total time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes / {total_elapsed/3600:.2f} hours)")
    if successful > 0:
        logger.info(f"📈 Average time per player: {total_time/successful:.2f} seconds")
        logger.info(f"📈 Processing rate: {successful/(total_elapsed/3600):.2f} players/hour")
        logger.info(f"📊 Total rows generated: {total_rows:,}")
        logger.info(f"💾 Total size: {total_size_mb:.2f} MB ({total_size_mb/1024:.2f} GB)")
        logger.info(f"📊 Average rows per player: {total_rows//successful:,}")
        logger.info(f"💾 Average size per player: {total_size_mb/successful:.2f} MB")
    logger.info("=" * 70)

if __name__ == '__main__':
    main()
