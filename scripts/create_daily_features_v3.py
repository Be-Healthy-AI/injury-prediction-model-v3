#!/usr/bin/env python3
"""
Enhanced Gold Standard Daily Features Generator V3
Generates daily features for injury prediction model V3
Based on V2 with enhancements and new dataset support
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import pickle
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
import re
from collections import defaultdict, deque
import shutil
from tqdm import tqdm

# Import benfica-parity functions
from benfica_parity_config import (
    BENFICA_PARITY_CONFIG,
    is_national_team_benfica_parity,
    map_competition_importance_benfica_parity,
    calculate_age_benfica_parity,
    detect_disciplinary_action_benfica_parity,
    calculate_enhanced_features_dynamically,
    calculate_injury_features_benfica_parity,
    calculate_national_team_features_benfica_parity,
    calculate_complex_derived_features_benfica_parity,
    set_competition_type_map
)

# Configure logging
import sys
if sys.platform == 'win32':
    import io
    # Fix Windows console encoding for emoji support
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('feature_generation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Team country mapping cache
TEAM_COUNTRY_MAP: Dict[str, str] = {}
COMPETITION_TYPE_MAP: Dict[str, str] = {}

# Configuration constants
CONFIG = {
    'DATA_DIR': 'original_data',  # Relative to project root
    'CACHE_FILE': 'data_cache_v3.pkl',  # Relative to project root
    'CACHE_DURATION': 3600,  # 1 hour in seconds
    'DEFAULT_OUTPUT_DIR': 'daily_features_output',  # Relative to project root
    'FEATURE_COUNT': 108,
    'FOOTBALL_SEASON_END_MONTH': 6,  # June
    'FOOTBALL_SEASON_END_DAY': 30,
    'MIN_CLUB_STABILITY_DAYS': 30,
    'MIN_CLUB_APPEARANCES': 3,
    'SUBSTITUTION_LATE_MINUTE': 75,
    'SUBSTITUTION_EARLY_MINUTE': 60,
    'INJURY_SHORT_TERM_DAYS': 7,
    'RECENT_MATCHES_WINDOW': 10,
    'ROLLING_WINDOW_DAYS': 7
}

# Global cache for data loading
_DATA_CACHE = {}

def load_data_with_cache() -> Dict[str, pd.DataFrame]:
    """
    Load data files with caching for performance.
    
    Returns:
        Dictionary containing all data DataFrames
    """
    import time
    
    # Check if cache exists and is recent
    if os.path.exists(CONFIG['CACHE_FILE']):
        cache_age = time.time() - os.path.getmtime(CONFIG['CACHE_FILE'])
        if cache_age < CONFIG['CACHE_DURATION']:
            logger.info("📦 Loading data from cache...")
            with open(CONFIG['CACHE_FILE'], 'rb') as f:
                return pickle.load(f)
    
    logger.info("📂 Loading data files...")
    data_dir = CONFIG['DATA_DIR']
    
    # Load all files at once with optimized settings
    # V3: Update these file names to match your V3 dataset files
    # Expected files: *_players_profile.xlsx, *_injuries_data.xlsx, *_match_data.xlsx, *_teams_data.xlsx, *_competition_data.xlsx
    import glob
    
    # Try to auto-detect files or use defaults
    players_files = glob.glob(f'{data_dir}/*players_profile.xlsx') + glob.glob(f'{data_dir}/*players_profile*.xlsx')
    injuries_files = glob.glob(f'{data_dir}/*injuries_data.xlsx') + glob.glob(f'{data_dir}/*injuries*.xlsx')
    matches_files = glob.glob(f'{data_dir}/*match_data.xlsx') + glob.glob(f'{data_dir}/*match*.xlsx')
    teams_files = glob.glob(f'{data_dir}/*teams_data.xlsx') + glob.glob(f'{data_dir}/*teams*.xlsx')
    competitions_files = glob.glob(f'{data_dir}/*competition_data.xlsx') + glob.glob(f'{data_dir}/*competition*.xlsx')
    career_files = glob.glob(f'{data_dir}/*players_career.xlsx') + glob.glob(f'{data_dir}/*players_career*.xlsx')
    
    # Use first match found, or fallback to default pattern
    players_path = players_files[0] if players_files else f'{data_dir}/players_profile.xlsx'
    injuries_path = injuries_files[0] if injuries_files else f'{data_dir}/injuries_data.xlsx'
    matches_path = matches_files[0] if matches_files else f'{data_dir}/match_data.xlsx'
    teams_path = teams_files[0] if teams_files else f'{data_dir}/teams_data.xlsx'
    competitions_path = competitions_files[0] if competitions_files else f'{data_dir}/competition_data.xlsx'
    career_path = career_files[0] if career_files else f'{data_dir}/players_career.xlsx'
    
    logger.info(f"📄 Loading: {os.path.basename(players_path)}")
    players = pd.read_excel(players_path, engine='openpyxl')
    logger.info(f"📄 Loading: {os.path.basename(injuries_path)}")
    injuries = pd.read_excel(injuries_path, engine='openpyxl')
    logger.info(f"📄 Loading: {os.path.basename(matches_path)}")
    matches = pd.read_excel(matches_path, engine='openpyxl')
    logger.info(f"📄 Loading: {os.path.basename(teams_path)}")
    teams = pd.read_excel(teams_path, engine='openpyxl')
    logger.info(f"📄 Loading: {os.path.basename(competitions_path)}")
    competitions = pd.read_excel(competitions_path, engine='openpyxl')

    career = None
    if os.path.exists(career_path):
        logger.info(f"📄 Loading: {os.path.basename(career_path)}")
        career = pd.read_excel(career_path, engine='openpyxl')
    else:
        logger.warning("⚠️  Players career file not found; previous club seeding will be skipped.")
    
    # Cache the data
    data = {
        'players': players,
        'injuries': injuries,
        'matches': matches,
        'teams': teams,
        'competitions': competitions,
        'career': career,
    }
    
    with open(CONFIG['CACHE_FILE'], 'wb') as f:
        pickle.dump(data, f)
    
    return data

def preprocess_data_optimized(players, injuries, matches):
    """Optimized data preprocessing with vectorized operations"""
    print("🔧 Preprocessing data...")
    
    # Filter goalkeepers (vectorized)
    players = players[players['position'] != 'Goalkeeper'].copy()
    
    # Filter injuries (vectorized)
    injuries = injuries[injuries['no_physio_injury'].isna()].copy()
    
    # Convert dates (vectorized)
    date_columns = ['fromDate', 'untilDate']
    for col in date_columns:
        injuries[col] = pd.to_datetime(injuries[col])
    
    # Add duration_days (vectorized)
    injuries['duration_days'] = (injuries['untilDate'] - injuries['fromDate']).dt.days
    
    # Preprocess matches (vectorized)
    matches['date'] = pd.to_datetime(matches['date'])
    
    # Optimized minutes parsing
    def parse_minutes_vectorized(minutes_series):
        """Vectorized minutes parsing"""
        # Convert to string and handle NaN
        minutes_str = minutes_series.astype(str)
        
        # Extract numbers using regex (vectorized)
        import re
        pattern = r'(\d+)'
        
        def extract_minutes(x):
            if pd.isna(x) or x == 'nan':
                return np.nan
            match = re.search(pattern, str(x))
            return int(match.group(1)) if match else np.nan
        
        return minutes_str.apply(extract_minutes)
    
    matches['minutes_played_numeric'] = parse_minutes_vectorized(matches['minutes_played'])
    
    # Vectorized numeric conversions for goals and assists
    numeric_cols = ['goals', 'assists']
    for col in numeric_cols:
        matches[f'{col}_numeric'] = pd.to_numeric(matches[col], errors='coerce').fillna(0)
    
    # Vectorized cards parsing
    def parse_cards_vectorized(cards_series):
        """Vectorized cards parsing"""
        def parse_cards(x):
            if pd.isna(x):
                return 0
            if isinstance(x, (int, float)):
                return int(x)
            # Count cards in strings like "71'", "45' 90'", etc.
            import re
            cards = re.findall(r'\d+\'', str(x))
            return len(cards)
        
        return cards_series.apply(parse_cards)
    
    matches['yellow_cards_numeric'] = parse_cards_vectorized(matches['yellow_cards'])
    matches['second_yellow_cards_numeric'] = parse_cards_vectorized(matches['second_yellow_cards'])
    matches['red_cards_numeric'] = parse_cards_vectorized(matches['red_cards'])
    
    # Combine yellow cards and second yellow cards for total yellow cards count
    matches['yellow_cards_numeric'] = matches['yellow_cards_numeric'] + matches['second_yellow_cards_numeric']
    
    # Vectorized height conversion - values are already in centimeters
    def parse_height(height_str):
        if pd.isna(height_str):
            return np.nan
        # Remove 'm' or 'cm' if present, replace comma with dot, then convert to float
        height_clean = str(height_str).replace('m', '').replace('cm', '').replace(',', '.').strip()
        try:
            return float(height_clean)
        except:
            return np.nan
    
    players['height_cm'] = players['height'].apply(parse_height)
    
    # ENHANCED: Add competition importance mapping using benfica-parity logic
    matches['competition_importance'] = matches['competition'].apply(map_competition_importance_benfica_parity)
    
    # DEPRECATED: season_phase removed - replaced with month feature
    # matches['season_phase'] = matches['date'].apply(map_season_phase_benfica_parity)
    
    # ENHANCED: Add disciplinary action detection using benfica-parity logic
    matches['disciplinary_action'] = matches.apply(detect_disciplinary_action_benfica_parity, axis=1)
    
    # ENHANCED: Add injury severity mapping
    def map_injury_severity(injury_type):
        """Map injury types to severity scores (1-5)"""
        if pd.isna(injury_type):
            return 1
        
        injury_lower = str(injury_type).lower()
        
        # Level 5: Career-threatening
        if any(term in injury_lower for term in ['rupture', 'multiple ligament', 'complex fracture', 'achilles rupture']):
            return 5
        
        # Level 4: Severe
        elif any(term in injury_lower for term in ['acl', 'cruciate', 'major fracture', 'dislocation', 'hernia']):
            return 4
        
        # Level 3: Serious
        elif any(term in injury_lower for term in ['ligament', 'fracture', 'meniscus', 'tendon']):
            return 3
        
        # Level 2: Moderate
        elif any(term in injury_lower for term in ['sprain', 'strain', 'minor fracture', 'bruise']):
            return 2
        
        # Level 1: Minor
        else:
            return 1
    
    injuries['injury_severity'] = injuries['injury_type'].apply(map_injury_severity)
    
    # ENHANCED: Add injury body part categorization
    def categorize_body_part(injury_type):
        """Categorize injuries by body part"""
        if pd.isna(injury_type):
            return 'unknown'
        
        injury_lower = str(injury_type).lower()
        
        # LOWER LEG - Ankle, foot, lower leg, calf
        if any(term in injury_lower for term in [
            'ankle', 'ankel', 'foot', 'toe', 'achilles', 'fibula', 'tibia', 
            'metatarsal', 'peroneal', 'calcaneus', 'plantar', 'heel', 'shin', 
            'sole', 'talus', 'navicular', 'cuboid', 'cuneiform', 'phalanx', 
            'sesamoid', 'tendinitis', 'tendonitis', 'tendinopathy', 'fascia', 
            'fascitis', 'plantar fasciitis', 'achilles tendon', 'calf', 
            'gastrocnemius', 'soleus', 'tibiotarsal', 'syndesmosis', 
            'tibial', 'peroneal tendon', 'ankle sprain', 'ankle fracture',
            'foot sprain', 'foot fracture', 'toe fracture', 'toe sprain'
        ]):
            return 'lower_leg'
        
        # KNEE - Knee, patella, meniscus, ligaments
        elif any(term in injury_lower for term in [
            'knee', 'patella', 'meniscus', 'meniscal', 'acl', 'pcl', 'lcl', 
            'mcl', 'ligament', 'cruciate', 'patellar', 'cartilage', 
            'arthroscopy', 'chondral', 'osteochondral', 'condyle', 
            'femur', 'tibial plateau', 'bursitis', 'prepatellar', 
            'infrapatellar', 'pes anserine', 'knee sprain', 'knee fracture'
        ]):
            return 'knee'
        
        # UPPER LEG - Thigh, hamstring, quadriceps (but NOT calf - that's lower_leg)
        elif any(term in injury_lower for term in [
            'thigh', 'quad', 'hamstring', 'adductor', 'quadriceps', 
            'biceps femoris', 'rectus femoris', 'iliopsoas', 
            'tensor fasciae latae', 'sartorius', 'gracilis', 
            'semimembranosus', 'semitendinosus', 'vastus', 'femoral', 
            'muscle strain', 'muscle tear', 'muscle injury', 
            'muscle problems', 'muscle tension', 'muscle fatigue',
            'breakdown of muscle fibers', 'muscle fibers', 'leg injury',
            'sore muscles'
        ]):
            return 'upper_leg'
        
        # HIP - Hip, pelvis, groin, lower back, abdominal
        elif any(term in injury_lower for term in [
            'hip', 'pelvis', 'pelvic', 'pubis', 'pubalgia', 'glute', 
            'gluteus', 'lumbar', 'lower back', 'groin', 'adductor', 
            'sacroiliac', 'abdominal', 'core', 'piriformis', 'iliac', 
            'ischial', 'coccyx', 'tailbone', 'osteitis pubis', 
            'sports hernia', 'inguinal', 'psoas', 'back problems', 
            'back injury', 'lumbago', 'belly muscles'
        ]):
            return 'hip'
        
        # UPPER BODY - Shoulder, arm, hand, neck, spine, ribs, chest
        elif any(term in injury_lower for term in [
            'shoulder', 'arm', 'elbow', 'wrist', 'hand', 'finger', 'thumb', 
            'clavicle', 'humerus', 'radius', 'ulna', 'scapula', 'bicep', 
            'tricep', 'forearm', 'upper arm', 'rotator cuff', 'labrum', 
            'acromion', 'sternoclavicular', 'acromioclavicular', 'carpal', 
            'metacarpal', 'tendon', 'tendinitis', 'tendonitis', 'bursitis', 
            'impingement', 'neck', 'cervical', 'cervicalgia', 'spine', 
            'spinal', 'vertebra', 'vertebral', 'disc', 'herniated', 
            'sciatica', 'back', 'cervical spine', 'thoracic', 'sacral', 
            'rib', 'ribs', 'rib cage', 'rib area', 'capsule injury',
            'tendon inflammation', 'rupture in a tendon', 'chest', 'chest injury'
        ]):
            return 'upper_body'
        
        # HEAD - Head, face, brain
        elif any(term in injury_lower for term in [
            'head', 'face', 'eye', 'nose', 'mouth', 'concussion', 'skull', 
            'jaw', 'cheekbone', 'ear', 'temple', 'brain', 'cranium', 
            'facial', 'orbital', 'zygomatic', 'maxilla', 'mandible', 
            'temporal', 'frontal', 'parietal', 'occipital', 'nasal', 
            'dental', 'tooth'
        ]):
            return 'head'
        
        # ILLNESS - Illness, infection, gastrointestinal
        elif any(term in injury_lower for term in [
            'illness', 'sick', 'sickness', 'fever', 'cold', 'flu', 'virus', 
            'infection', 'influenza', 'covid', 'respiratory', 'bronchitis', 
            'pneumonia', 'disease', 'sars', 'gastroenteritis', 'gastric', 
            'gastric problems', 'diarrhea', 'nausea', 'vomiting', 
            'dehydration', 'fatigue', 'exhaustion', 'heat stroke', 
            'hypothermia', 'shock'
        ]):
            return 'illness'
        
        else:
            return 'other'
    
    injuries['body_part_category'] = injuries['injury_type'].apply(categorize_body_part)
    
    # Map column names to match expected format
    players = players.rename(columns={
        'foot': 'dominant_foot',
        'signed_from': 'previous_club'
    })
    
    # Add missing columns with default values
    players['previous_club_country'] = None
    
    print(f"Players after excluding goalkeepers: {players.shape}")
    print(f"Injuries after filtering (no_physio_injury = null): {injuries.shape}")
    
    return players, injuries, matches

def preprocess_career_data(career: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Standardize players career dataset for previous club lookups."""
    if career is None:
        return None
    career = career.copy()
    rename_map = {
        'id': 'player_id',
        'ID': 'player_id',
        'Date': 'transfer_date',
        'date': 'transfer_date',
        'From': 'from_club',
        'from': 'from_club',
        'To': 'to_club',
        'to': 'to_club',
        'Season': 'season'
    }
    existing = {old: new for old, new in rename_map.items() if old in career.columns}
    if existing:
        career = career.rename(columns=existing)
    if 'transfer_date' in career.columns:
        career['transfer_date'] = pd.to_datetime(career['transfer_date'], errors='coerce')
    if 'player_id' in career.columns:
        career = career.sort_values(['player_id', 'transfer_date'], na_position='last').reset_index(drop=True)
    else:
        career = career.sort_values(by='transfer_date', na_position='last').reset_index(drop=True)
    return career

def determine_match_participation_optimized(matches):
    """Optimized match participation determination"""
    def participation_status(row):
        position = str(row['position']).lower()
        minutes = row['minutes_played_numeric']
        
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
        
        if pd.isna(minutes):
            if 'was not in the squad' in position:
                return 'not_selected'
            elif 'unused substitute' in position:
                return 'bench_unused'
            elif 'information not yet available' in position or pd.isna(row['position']):
                return 'not_selected'  # Default assumption
            else:
                return 'played'  # Has position, assume played
        else:
            if minutes > 0:
                return 'played'
            else:
                return 'bench_unused'
    
    return matches.apply(participation_status, axis=1)

def get_club_country(club_name):
    """Get country for a given club name"""
    if pd.isna(club_name):
        return None
    
    club_name_lower = str(club_name).lower()
    for key in clean_club_variants(club_name):
        if key in TEAM_COUNTRY_MAP:
            return TEAM_COUNTRY_MAP[key]
    # Fallback: try substring match in map
    for key, value in TEAM_COUNTRY_MAP.items():
        if key and key in club_name_lower:
            return value
    
    # Quick explicit mappings for frequently missing clubs in datasets
    # FC Alverca (Portugal)
    if 'alverca' in club_name_lower:
        return 'Portugal'
    # GE Anápolis (Brazil)
    if 'anápolis' in club_name_lower or 'anapolis' in club_name_lower:
        return 'Brazil'
    
    # Brazilian clubs
    if any(indicator in club_name_lower for indicator in ['ceará', 'palmeiras', 'botafogo', 'flamengo', 'fluminense', 'vasco', 'corinthians', 'são paulo', 'santos', 'grêmio', 'internacional', 'atlético mineiro', 'cruzeiro', 'bahia', 'vitória', 'sport', 'nautico', 'santa cruz', 'figueirense', 'criciúma', 'joinville', 'chapecoense', 'atlético paranaense', 'coritiba', 'paraná', 'goiás', 'atlético goianiense', 'brasiliense', 'ceub', 'gama', 'taguatinga', 'samambaia', 'sobradinho', 'planaltina', 'ceilandia', 'guará', 'anápolis', 'anapolis']):
        return 'Brazil'
    
    # Portuguese clubs
    elif any(indicator in club_name_lower for indicator in ['benfica', 'porto', 'sporting', 'braga', 'marítimo', 'moreirense', 'paços', 'boavista', 'rio ave', 'estoril', 'tondela', 'portimonense', 'santa clara', 'famalicão', 'arouca', 'chaves', 'gil vicente', 'aves', 'vizela', 'feirense', 'alverca']):
        return 'Portugal'
    
    # Spanish clubs
    elif any(indicator in club_name_lower for indicator in ['barcelona', 'madrid', 'atletico', 'sevilla', 'valencia', 'bilbao', 'sociedad', 'villareal', 'córdoba', 'murcia', 'ponferradina', 'gijón', 'mirandés', 'lugo', 'sabadell', 'alcorcón', 'girona', 'numancia', 'las palmas', 'hercules', 'cartagena', 'huelva', 'recreativo']):
        return 'Spain'
    
    # German clubs
    elif any(indicator in club_name_lower for indicator in ['bayer', 'bayern', 'dortmund', 'frankfurt', 'leipzig', 'stuttgart', 'heidenheim', 'union berlin', 'st. pauli', 'augsburg', 'freiburg', 'mainz', 'teutonia', 'borussia', 'schalke', 'hamburger', 'werder', 'hannover', 'nürnberg', 'kaiserslautern', 'karlsruhe', 'duisburg', 'bochum', 'bielefeld', 'cologne', 'leverkusen', 'mönchengladbach', 'wolfsburg', 'hannover', 'nürnberg', 'kaiserslautern', 'karlsruhe', 'duisburg', 'bochum', 'bielefeld', 'cologne', 'leverkusen', 'mönchengladbach', 'wolfsburg']):
        return 'Germany'
    
    # Italian clubs
    elif any(indicator in club_name_lower for indicator in ['fiorentina', 'milan', 'inter', 'juventus', 'roma', 'lazio', 'napoli', 'torino', 'genoa', 'sampdoria', 'udinese', 'atalanta', 'bologna', 'parma', 'lecce', 'empoli', 'sassuolo', 'verona', 'spezia', 'salernitana', 'monza', 'cremonese', 'cagliari', 'palermo', 'bari', 'frosinone', 'como', 'venezia', 'spal', 'benevento', 'crotone', 'pescara', 'carpi', 'frosinone', 'como', 'venezia', 'spal', 'benevento', 'crotone', 'pescara', 'carpi']):
        return 'Italy'
    
    # Swiss clubs
    elif any(indicator in club_name_lower for indicator in ['basel', 'young boys', 'zürich', 'grasshopper', 'lausanne', 'servette', 'sion', 'luzern', 'st. gallen', 'thun', 'aarau', 'wil', 'kriens', 'schaffhausen', 'bellinzona', 'neuchâtel', 'xamax', 'lugano', 'chiasso', 'locarno', 'bellinzona', 'neuchâtel', 'xamax', 'lugano', 'chiasso', 'locarno']):
        return 'Switzerland'
    
    # Danish clubs
    elif any(indicator in club_name_lower for indicator in ['hb köge', 'sönderjyske', 'copenhagen', 'brøndby', 'aalborg', 'midtjylland', 'nordsjælland', 'viborg', 'randers', 'horsens', 'silkeborg', 'esbjerg', 'odense', 'vejle', 'hobro', 'skive', 'fredericia', 'kolding', 'hvidovre', 'b 93', 'ab', 'frem', 'køge', 'sönderjyske', 'copenhagen', 'brøndby', 'aalborg', 'midtjylland', 'nordsjælland', 'viborg', 'randers', 'horsens', 'silkeborg', 'esbjerg', 'odense', 'vejle', 'hobro', 'skive', 'fredericia', 'kolding', 'hvidovre', 'b 93', 'ab', 'frem', 'køge']):
        return 'Denmark'
    
    # Czech clubs
    elif any(indicator in club_name_lower for indicator in ['slavia prague', 'sparta prague', 'bohemians', 'dynamo', 'viktoria', 'plzeň', 'ostrava', 'brno', 'liberec', 'jablonec', 'olomouc', 'pardubice', 'hradec králové', 'mladá boleslav', 'teplice', 'most', 'chomutov', 'ústí nad labem', 'karlovy vary', 'české budějovice', 'tábor', 'písek', 'strakonice', 'klatovy', 'domažlice', 'cheb', 'sokolov', 'karlovy vary', 'české budějovice', 'tábor', 'písek', 'strakonice', 'klatovy', 'domažlice', 'cheb', 'sokolov']):
        return 'Czech Republic'
    
    # French clubs
    elif any(indicator in club_name_lower for indicator in ['paris', 'lyon', 'marseille', 'monaco', 'lille', 'nice', 'rennes', 'strasbourg', 'nantes', 'bordeaux', 'toulouse', 'montpellier', 'reims', 'angers', 'troyes', 'clermont', 'lens', 'brest', 'lorient', 'auxerre', 'ajaccio', 'toulon', 'cannes', 'bastia', 'guingamp', 'caen', 'dijon', 'amiens', 'metz', 'nancy', 'sochaux', 'valenciennes', 'evian', 'gazélec', 'red star', 'paris fc', 'orléans', 'châteauroux', 'tours', 'niort', 'laval', 'le havre', 'troyes', 'clermont', 'lens', 'brest', 'lorient', 'auxerre', 'ajaccio', 'toulon', 'cannes', 'bastia', 'guingamp', 'caen', 'dijon', 'amiens', 'metz', 'nancy', 'sochaux', 'valenciennes', 'evian', 'gazélec', 'red star', 'paris fc', 'orléans', 'châteauroux', 'tours', 'niort', 'laval', 'le havre']):
        return 'France'
    
    # English clubs
    elif any(indicator in club_name_lower for indicator in ['manchester', 'liverpool', 'arsenal', 'chelsea', 'tottenham', 'leeds', 'newcastle', 'aston villa', 'everton', 'west ham', 'crystal palace', 'brighton', 'fulham', 'brentford', 'bournemouth', 'wolves', 'leicester', 'southampton', 'burnley', 'watford', 'norwich', 'cardiff', 'huddersfield', 'swansea', 'stoke', 'west brom', 'sunderland', 'middlesbrough', 'hull', 'reading', 'derby', 'birmingham', 'blackburn', 'bolton', 'wigan', 'portsmouth', 'charlton', 'ipswich', 'qpr', 'millwall', 'barnsley', 'rotherham', 'preston', 'blackpool', 'bristol', 'nottingham', 'sheffield', 'bradford', 'oldham', 'rochdale', 'bury', 'accrington', 'fleetwood', 'burton', 'shrewsbury', 'walsall', 'coventry', 'oxford', 'peterborough', 'doncaster', 'bristol rovers', 'plymouth', 'portsmouth', 'charlton', 'ipswich', 'qpr', 'millwall', 'barnsley', 'rotherham', 'preston', 'blackpool', 'bristol', 'nottingham', 'sheffield', 'bradford', 'oldham', 'rochdale', 'bury', 'accrington', 'fleetwood', 'burton', 'shrewsbury', 'walsall', 'coventry', 'oxford', 'peterborough', 'doncaster', 'bristol rovers', 'plymouth']):
        return 'England'
    
    # Dutch clubs
    elif any(indicator in club_name_lower for indicator in ['ajax', 'psv', 'feyenoord', 'az', 'utrecht', 'vitesse', 'heerenveen', 'groningen', 'twente', 'heracles', 'willem ii', 'sparta', 'excelsior', 'den haag', 'adoptie', 'volendam', 'go ahead', 'cambuur', 'emmen', 'fortuna', 'roda', 'mvv', 'helmond', 'eindhoven', 'den bosch', 'oss', 'almere', 'telstar', 'dordrecht', 'rotterdam', 'amsterdam', 'eindhoven', 'den bosch', 'oss', 'almere', 'telstar', 'dordrecht', 'rotterdam', 'amsterdam', 'eindhoven', 'den bosch', 'oss', 'almere', 'telstar', 'dordrecht', 'rotterdam', 'amsterdam']):
        return 'Netherlands'
    
    # Belgian clubs
    elif any(indicator in club_name_lower for indicator in ['club brugge', 'anderlecht', 'genk', 'standard', 'antwerp', 'gent', 'charleroi', 'mechelen', 'ostend', 'eupen', 'sint-truiden', 'kortrijk', 'lokeren', 'waasland-beveren', 'roeselare', 'mouscron', 'waregem', 'leuven', 'beerschot', 'berchem', 'beringen', 'beveren', 'boom', 'boussu', 'brasschaat', 'brugge', 'charleroi', 'cercle', 'dender', 'dessel', 'diyarbakir', 'eupen', 'genk', 'gent', 'geel', 'gembloux', 'hamme', 'heist', 'herk-de-stad', 'hoogstraten', 'houthalen', 'jupille', 'kortrijk', 'la louvière', 'leuven', 'liège', 'lokeren', 'lommel', 'maasmechelen', 'mechelen', 'molenbeek', 'mons', 'mouscron', 'namur', 'nieuwerkerken', 'ninove', 'oostende', 'patro', 'roeselare', 'roosdaal', 'sint-niklaas', 'sint-truiden', 'spa', 'tienen', 'tongeren', 'torhout', 'turnhout', 'vise', 'waregem', 'westerlo', 'zulte', 'zwijndrecht']):
        return 'Belgium'
    
    # Ukrainian clubs
    elif any(indicator in club_name_lower for indicator in ['dynamo kyiv', 'shakhtar', 'dynamo', 'vorskla', 'zorya', 'kolos', 'rukh', 'minaj', 'ingulets', 'metalist', 'dnipro', 'kryvbas', 'chornomorets', 'mariupol', 'volyn', 'karpaty', 'tavriya', 'metalurh', 'stal', 'vorskla', 'zorya', 'kolos', 'rukh', 'minaj', 'ingulets', 'metalist', 'dnipro', 'kryvbas', 'chornomorets', 'mariupol', 'volyn', 'karpaty', 'tavriya', 'metalurh', 'stal']):
        return 'Ukraine'
    
    # Russian clubs
    elif any(indicator in club_name_lower for indicator in ['cska moscow', 'spartak moscow', 'lokomotiv moscow', 'zenit', 'dynamo moscow', 'torpedo moscow', 'fakel', 'orenburg', 'baltika', 'volgar', 'alania', 'rotor', 'krylia', 'ural', 'rubin', 'terek', 'amkar', 'tom', 'kuban', 'krasnodar', 'rostov', 'akhmat', 'ufa', 'arsenal tula', 'tambov', 'sochi', 'khimki', 'nizhny', 'torpedo', 'fakel', 'orenburg', 'baltika', 'volgar', 'alania', 'rotor', 'krylia', 'ural', 'rubin', 'terek', 'amkar', 'tom', 'kuban', 'krasnodar', 'rostov', 'akhmat', 'ufa', 'arsenal tula', 'tambov', 'sochi', 'khimki', 'nizhny', 'torpedo', 'fakel', 'orenburg', 'baltika', 'volgar']):
        return 'Russia'
    
    # Turkish clubs
    elif any(indicator in club_name_lower for indicator in ['fenerbahçe', 'galatasaray', 'beşiktaş', 'trabzonspor', 'antalyaspor', 'kasımpaşa', 'sivasspor', 'konyaspor', 'alanyaspor', 'gaziantep', 'adana demirspor', 'kayserispor', 'fatih karagümrük', 'istanbul başakşehir', 'ankaragücü', 'giresunspor', 'hatayspor', 'konyaspor', 'alanyaspor', 'gaziantep', 'adana demirspor', 'kayserispor', 'fatih karagümrük', 'istanbul başakşehir', 'ankaragücü', 'giresunspor', 'hatayspor']):
        return 'Turkey'
    
    # Greek clubs
    elif any(indicator in club_name_lower for indicator in ['olympiacos', 'panathinaikos', 'aek athens', 'paok', 'aris', 'ofi', 'panionios', 'larissa', 'volos', 'lamia', 'atromitos', 'panetolikos', 'asteras tripolis', 'xanthi', 'kerkyra', 'levadiakos', 'platanias', 'iraklis', 'kavala', 'doxa drama', 'apollon smyrnis', 'panachaiki', 'kallithea', 'chalkida', 'kalamata', 'karditsa', 'kozani', 'veria', 'giannina', 'kavala', 'doxa drama', 'apollon smyrnis', 'panachaiki', 'kallithea', 'chalkida', 'kalamata', 'karditsa', 'kozani', 'veria', 'giannina']):
        return 'Greece'
    
    # Polish clubs
    elif any(indicator in club_name_lower for indicator in ['legia warsaw', 'wisła kraków', 'lechia gdańsk', 'lech poznan', 'cracovia', 'piast gliwice', 'zagłębie lubin', 'korona kielce', 'jagiellonia białystok', 'śląsk wrocław', 'widzew łódź', 'ruch chorzów', 'górnik zabrze', 'polonia warsaw', 'odra opole', 'stomil olsztyn', 'amica wronki', 'groclin dębica', 'wisła płock', 'odra wodzisław', 'górnik łęczna', 'arkonia szczecin', 'pogoń szczecin', 'zawisza bydgoszcz', 'sandecja nowy sącz', 'podbeskidzie bielsko-biała', 'górnik polkowice', 'bruk-bet termalica', 'nieciecza', 'miedź legnica', 'chrobry głogów', 'stomil olsztyn', 'amica wronki', 'groclin dębica', 'wisła płock', 'odra wodzisław', 'górnik łęczna', 'arkonia szczecin', 'pogoń szczecin', 'zawisza bydgoszcz', 'sandecja nowy sącz', 'podbeskidzie bielsko-biała', 'górnik polkowice', 'bruk-bet termalica', 'nieciecza', 'miedź legnica', 'chrobry głogów']):
        return 'Poland'
    
    # Austrian clubs
    elif any(indicator in club_name_lower for indicator in ['red bull salzburg', 'rapid vienna', 'austria vienna', 'sturm graz', 'lask linz', 'wolfsberg', 'hartberg', 'ried', 'altach', 'austria klagenfurt', 'wattens', 'admira', 'kapfenberg', 'mattersburg', 'groedig', 'pasching', 'austria salzburg', 'swarovski tirol', 'innsbruck', 'wacker', 'vorwärts', 'first vienna', 'floridsdorfer', 'wien', 'linz', 'graz', 'klagenfurt', 'salzburg', 'innsbruck', 'wacker', 'vorwärts', 'first vienna', 'floridsdorfer', 'wien', 'linz', 'graz', 'klagenfurt', 'salzburg', 'innsbruck', 'wacker', 'vorwärts', 'first vienna', 'floridsdorfer', 'wien', 'linz', 'graz', 'klagenfurt', 'salzburg', 'innsbruck', 'wacker', 'vorwärts', 'first vienna', 'floridsdorfer', 'wien', 'linz', 'graz', 'klagenfurt', 'salzburg']):
        return 'Austria'
    
    # Romanian clubs
    elif any(indicator in club_name_lower for indicator in ['dinamo bucurești', 'steaua bucurești', 'rapid bucurești', 'universitatea craiova', 'astra giurgiu', 'cfr cluj', 'fcsb', 'farul constanța', 'sepsi sfântu gheorghe', 'uts arad', 'botoșani', 'voluntari', 'chindia târgoviște', 'academica clinceni', 'gaz metan mediaș', 'viitorul', 'poli timișoara', 'otopeni', 'brașov', 'pandurii', 'ceahlăul', 'vaslui', 'bistrița', 'baia mare', 'suceava', 'piatra neamț', 'bacău', 'galați', 'brăila', 'tulcea', 'constanța', 'călărași', 'slobozia', 'buzău', 'ploiești', 'târgoviște', 'pitești', 'craiova', 'drobeta', 'reșița', 'timișoara', 'arad', 'oradea', 'cluj', 'sibiu', 'brașov', 'făgăraș', 'sighișoara', 'târgu mureș', 'alba iulia', 'deva', 'hunedoara', 'petroșani', 'lupeni', 'vulcan', 'anina', 'reșița', 'timișoara', 'arad', 'oradea', 'cluj', 'sibiu', 'brașov', 'făgăraș', 'sighișoara', 'târgu mureș', 'alba iulia', 'deva', 'hunedoara', 'petroșani', 'lupeni', 'vulcan', 'anina']):
        return 'Romania'
    
    # Bulgarian clubs
    elif any(indicator in club_name_lower for indicator in ['cska sofia', 'levski sofia', 'ludogorets razgrad', 'lokomotiv sofia', 'slavia sofia', 'botev plovdiv', 'lokomotiv plovdiv', 'cherno more', 'botev vratsa', 'etar', 'dobrudzha', 'marek', 'belasitsa', 'pirin', 'macedonia', 'vihren', 'hebar', 'strumica', 'minyor', 'spartak', 'sliven', 'yantra', 'velbazhd', 'balkan', 'shumen', 'dunav', 'ludogorets', 'razgrad', 'sofia', 'plovdiv', 'varna', 'burgas', 'ruse', 'pleven', 'gabrovo', 'veliko tarnovo', 'stara zagora', 'haskovo', 'kardzhali', 'smolyan', 'blagoevgrad', 'kyustendil', 'pernik', 'sofia', 'plovdiv', 'varna', 'burgas', 'ruse', 'pleven', 'gabrovo', 'veliko tarnovo', 'stara zagora', 'haskovo', 'kardzhali', 'smolyan', 'blagoevgrad', 'kyustendil', 'pernik']):
        return 'Bulgaria'
    
    # Croatian clubs
    elif any(indicator in club_name_lower for indicator in ['dinamo zagreb', 'hajduk split', 'rijeka', 'osijek', 'lokomotiva', 'inter zaprešić', 'slaven belupo', 'istra 1961', 'varazdin', 'zadar', 'cibalia', 'šibenik', 'karlovac', 'međimurje', 'pomorac', 'imotski', 'zadar', 'cibalia', 'šibenik', 'karlovac', 'međimurje', 'pomorac', 'imotski']):
        return 'Croatia'
    
    # Serbian clubs
    elif any(indicator in club_name_lower for indicator in ['red star belgrade', 'partizan belgrade', 'vojvodina', 'čukarički', 'radnički niš', 'napredak', 'voždovac', 'radnik', 'proleter', 'novi pazar', 'javor', 'mladost', 'voždovac', 'radnik', 'proleter', 'novi pazar', 'javor', 'mladost']):
        return 'Serbia'
    
    # Hungarian clubs
    elif any(indicator in club_name_lower for indicator in ['ferencváros', 'mtk budapest', 'újpest', 'honvéd', 'debrecen', 'győr', 'kecskemét', 'paks', 'diosgyor', 'vasas', 'budaörs', 'pécs', 'szombathely', 'szeged', 'kaposvár', 'sopron', 'zalaegerszeg', 'tatabánya', 'veszprém', 'budapest', 'debrecen', 'győr', 'kecskemét', 'paks', 'diosgyor', 'vasas', 'budaörs', 'pécs', 'szombathely', 'szeged', 'kaposvár', 'sopron', 'zalaegerszeg', 'tatabánya', 'veszprém']):
        return 'Hungary'
    
    # Slovak clubs
    elif any(indicator in club_name_lower for indicator in ['slovan bratislava', 'spartak trnava', 'trenčín', 'ružomberok', 'dac dunajská streda', 'senica', 'nitra', 'banská bystrica', 'prešov', 'košice', 'žilina', 'poprad', 'bardejov', 'humenné', 'michalovce', 'trebišov', 'rožňava', 'spišská nová ves', 'brezno', 'detva', 'zvolen', 'banská štiavnica', 'kremnica', 'banská bystrica', 'prešov', 'košice', 'žilina', 'poprad', 'bardejov', 'humenné', 'michalovce', 'trebišov', 'rožňava', 'spišská nová ves', 'brezno', 'detva', 'zvolen', 'banská štiavnica', 'kremnica']):
        return 'Slovakia'
    
    # Slovenian clubs
    elif any(indicator in club_name_lower for indicator in ['maribor', 'olimpija ljubljana', 'domžale', 'celje', 'koper', 'grosuplje', 'krka', 'rudar', 'primorje', 'interblock', 'drava', 'ptuj', 'slovan', 'ljubljana', 'maribor', 'celje', 'koper', 'grosuplje', 'krka', 'rudar', 'primorje', 'interblock', 'drava', 'ptuj', 'slovan']):
        return 'Slovenia'
    
    # Default to None if not recognized
    else:
        return None

def is_national_team_fast(team_name):
    """Robust check if team is a national team - works for any country"""
    if pd.isna(team_name):
        return False
    
    team_lower = str(team_name).lower()
    
    # Special cases: known clubs that contain country names but are not national teams
    known_clubs_with_country_names = [
        'austria vienna', 'rapid vienna', 'red bull salzburg', 'borussia dortmund',
        'dynamo kyiv', 'shakhtar donetsk', 'dinamo bucurești', 'steaua bucurești',
        'cska sofia', 'levski sofia', 'ludogorets razgrad', 'cska moscow',
        'spartak moscow', 'lokomotiv moscow', 'zenit saint petersburg'
    ]
    
    if team_lower in known_clubs_with_country_names:
        return False
    
    # Generic national team indicators
    national_indicators = [
        'national', 'country', 'republic', 'federation', 'selection', 'team',
        'olympic', 'world cup', 'euro', 'copa america', 'africa cup', 'asia cup'
    ]
    
    # Check for generic indicators first
    if any(indicator in team_lower for indicator in national_indicators):
        return True
    
    # Comprehensive list of country names (not exhaustive but covers major football nations)
    country_names = [
        # Europe
        'albania', 'andorra', 'armenia', 'austria', 'azerbaijan', 'belarus', 'belgium', 
        'bosnia', 'bulgaria', 'croatia', 'cyprus', 'czech', 'denmark', 'england', 
        'estonia', 'finland', 'france', 'georgia', 'germany', 'gibraltar', 'greece', 
        'hungary', 'iceland', 'ireland', 'israel', 'italy', 'kazakhstan', 'kosovo', 
        'latvia', 'liechtenstein', 'lithuania', 'luxembourg', 'malta', 'moldova', 
        'montenegro', 'netherlands', 'north macedonia', 'norway', 'poland', 'portugal', 
        'romania', 'russia', 'san marino', 'scotland', 'serbia', 'slovakia', 'slovenia', 
        'spain', 'sweden', 'switzerland', 'turkey', 'ukraine', 'wales',
        
        # South America
        'argentina', 'bolivia', 'brazil', 'chile', 'colombia', 'ecuador', 'guyana', 
        'paraguay', 'peru', 'suriname', 'uruguay', 'venezuela',
        
        # North America
        'canada', 'costa rica', 'el salvador', 'guatemala', 'honduras', 'jamaica', 
        'mexico', 'nicaragua', 'panama', 'trinidad', 'united states', 'usa',
        
        # Africa
        'algeria', 'angola', 'benin', 'botswana', 'burkina faso', 'burundi', 'cameroon', 
        'cape verde', 'central african republic', 'chad', 'comoros', 'congo', 
        'cote d\'ivoire', 'djibouti', 'egypt', 'equatorial guinea', 'eritrea', 'ethiopia', 
        'gabon', 'gambia', 'ghana', 'guinea', 'guinea-bissau', 'kenya', 'lesotho', 
        'liberia', 'libya', 'madagascar', 'malawi', 'mali', 'mauritania', 'mauritius', 
        'morocco', 'mozambique', 'namibia', 'niger', 'nigeria', 'rwanda', 'senegal', 
        'seychelles', 'sierra leone', 'somalia', 'south africa', 'sudan', 'tanzania', 
        'togo', 'tunisia', 'uganda', 'zambia', 'zimbabwe',
        
        # Asia
        'afghanistan', 'bahrain', 'bangladesh', 'bhutan', 'brunei', 'cambodia', 'china', 
        'india', 'indonesia', 'iran', 'iraq', 'japan', 'jordan', 'kazakhstan', 'kuwait', 
        'kyrgyzstan', 'laos', 'lebanon', 'malaysia', 'maldives', 'mongolia', 'myanmar', 
        'nepal', 'north korea', 'oman', 'pakistan', 'palestine', 'philippines', 'qatar', 
        'saudi arabia', 'singapore', 'south korea', 'sri lanka', 'syria', 'taiwan', 
        'thailand', 'timor-leste', 'turkmenistan', 'uae', 'uzbekistan', 'vietnam', 'yemen',
        
        # Oceania
        'australia', 'fiji', 'new zealand', 'papua new guinea', 'samoa', 'solomon islands', 
        'tahiti', 'tonga', 'vanuatu'
    ]
    
    # Check if the team name contains a country name
    # Use word boundaries to avoid false positives like "Borussia Dortmund" containing "Austria"
    import re
    for country in country_names:
        # Use word boundaries to match whole words only
        pattern = r'\b' + re.escape(country) + r'\b'
        if re.search(pattern, team_lower):
            return True
    
    return False

def is_junior_or_temporary_team(team_name):
    """Check if team is a junior team or temporary assignment"""
    if pd.isna(team_name):
        return False
    
    team_lower = str(team_name).lower()
    junior_indicators = ['junior', 'youth', 'u19', 'u20', 'u21', 'u23', 'reserve', 'b team', 'second team', 'under-21', 'under-20', 'under-19']
    temporary_indicators = ['loan', 'temporary', 'short-term']
    return any(indicator in team_lower for indicator in junior_indicators + temporary_indicators)

def is_main_club_team(team_name):
    """Check if team is a main club (not national, junior, or temporary)"""
    if pd.isna(team_name):
        return False
    
    return not (is_national_team_fast(team_name) or is_junior_or_temporary_team(team_name))

def get_football_season(date):
    """Get football season for a given date (follows football calendar, not civil calendar)"""
    year = date.year
    month = date.month
    
    # Football season typically runs from July/August to May/June
    # For dates from July to December, season is current_year/next_year
    # For dates from January to June, season is previous_year/current_year
    if month >= 7:  # July onwards
        return f"{year}/{year+1}"
    else:  # January to June
        return f"{year-1}/{year}"

def build_daily_profile_series_optimized(player_row, calendar, player_matches, player_career: Optional[pd.DataFrame] = None):
    """Highly optimized daily profile series building"""
    # Pre-calculate all values before creating DataFrame
    n_days = len(calendar)
    
    def normalize_club_name(value):
        if pd.isna(value):
            return None
        value_str = str(value).strip()
        if value_str == '' or value_str.lower() in {'nan', 'none', 'no club', 'free agent'}:
            return None
        return value_str
    
    # Static features (vectorized)
    # Use benfica-parity age calculation for better precision matching
    age_values = [calculate_age_benfica_parity(player_row['date_of_birth'], date) for date in calendar]
    position_values = [player_row['position']] * n_days
    nationality1_values = [player_row['nationality1']] * n_days
    nationality2_values = [player_row['nationality2']] * n_days
    height_cm_values = [player_row['height_cm']] * n_days
    dominant_foot_values = [player_row['dominant_foot']] * n_days
    
    # Previous club seeding
    initial_previous_club = clean_club_label(player_row.get('previous_club'))
    initial_previous_club_country = get_club_country(initial_previous_club) if initial_previous_club else None

    # Build career events for precise club transitions
    career_events: List[Dict[str, Optional[str]]] = []
    if player_career is not None and not player_career.empty:
        date_col = 'transfer_date' if 'transfer_date' in player_career.columns else (
            'Date' if 'Date' in player_career.columns else None
        )
        from_col = 'from_club' if 'from_club' in player_career.columns else (
            'From' if 'From' in player_career.columns else None
        )
        to_col = 'to_club' if 'to_club' in player_career.columns else (
            'To' if 'To' in player_career.columns else None
        )
        if date_col is not None:
            for _, row in player_career.iterrows():
                transfer_date = row.get(date_col)
                if pd.isna(transfer_date):
                    continue
                transfer_date = pd.to_datetime(transfer_date, errors='coerce')
                if pd.isna(transfer_date):
                    continue
                transfer_date = transfer_date.normalize()
                from_club = clean_club_label(row.get(from_col)) if from_col is not None else None
                to_club = clean_club_label(row.get(to_col)) if to_col is not None else None
                if not from_club and not to_club:
                    continue
                career_events.append({'date': transfer_date, 'from': from_club, 'to': to_club})
    career_events.sort(key=lambda x: x['date'])
    use_career_assignment = len(career_events) > 0

    # Prepare storage arrays
    previous_club_values = [initial_previous_club] * n_days
    previous_club_country_values = [initial_previous_club_country] * n_days
    current_club_values = [None] * n_days
    current_club_country_values = [None] * n_days
    seniority_days_values = [0] * n_days

    current_club_seed = clean_club_label(player_row.get('current_club')) if 'current_club' in player_row else None
    if not current_club_seed:
        current_club_seed = clean_club_label(player_row.get('previous_club'))

    if use_career_assignment:
        start_date = calendar[0]
        current_club = current_club_seed
        current_club_country = get_club_country(current_club)
        current_club_start = start_date
        current_previous_club = initial_previous_club
        current_previous_club_country = initial_previous_club_country

        past_events = [e for e in career_events if e['date'] <= start_date]
        future_events = [e for e in career_events if e['date'] > start_date]

        for event in past_events:
            if event['from']:
                current_previous_club = event['from']
                current_previous_club_country = get_club_country(current_previous_club)
            if event['to']:
                if current_club != event['to']:
                    current_club = event['to']
                    current_club_country = get_club_country(current_club)
                    current_club_start = max(event['date'], start_date)

        if current_club is None and future_events:
            first_event = future_events[0]
            if first_event['from']:
                current_club = first_event['from']
                current_club_country = get_club_country(current_club)
                current_club_start = start_date
        if current_club is None:
            current_club = current_club_seed
            current_club_country = get_club_country(current_club)

        future_idx = 0
        next_event = future_events[future_idx] if future_events else None

        for i, date in enumerate(calendar):
            while next_event is not None and date >= next_event['date']:
                if next_event['from']:
                    current_previous_club = next_event['from']
                    current_previous_club_country = get_club_country(current_previous_club)
                if next_event['to'] and current_club != next_event['to']:
                    current_club = next_event['to']
                    current_club_country = get_club_country(current_club)
                    current_club_start = next_event['date']
                future_idx += 1
                next_event = future_events[future_idx] if future_idx < len(future_events) else None

            previous_club_values[i] = current_previous_club
            previous_club_country_values[i] = current_previous_club_country
            current_club_values[i] = current_club
            current_club_country_values[i] = current_club_country
            seniority_days_values[i] = max(0, (date - current_club_start).days) if current_club_start else 0

    elif not player_matches.empty:
        # Fallback to match-derived club tracking when career data is unavailable
        main_club_matches = player_matches[
            (player_matches['home_team'].apply(is_main_club_team)) |
            (player_matches['away_team'].apply(is_main_club_team))
        ].copy()

        if not main_club_matches.empty:
            player_clubs = []
            for _, match in main_club_matches.iterrows():
                home_team = match['home_team']
                away_team = match['away_team']

                if is_main_club_team(home_team) and is_main_club_team(away_team):
                    home_count = len(player_matches[player_matches['home_team'] == home_team])
                    away_count = len(player_matches[player_matches['away_team'] == away_team])
                    player_club = home_team if home_count >= away_count else away_team
                elif is_main_club_team(home_team):
                    player_club = home_team
                elif is_main_club_team(away_team):
                    player_club = away_team
                else:
                    continue

                player_clubs.append({'date': match['date'], 'club': clean_club_label(player_club)})

            player_clubs.sort(key=lambda x: x['date'])

            club_periods = []
            current_club = None
            current_start_date = None

            current_group = []
            for club_info in player_clubs:
                club = club_info['club']
                date = club_info['date']

                if not current_group or club == current_group[-1]['club']:
                    current_group.append(club_info)
                else:
                    if len(current_group) >= 3:
                        club_periods.append(current_group)
                    current_group = [club_info]

            if len(current_group) >= 3:
                club_periods.append(current_group)

            structured_periods = []
            for group in club_periods:
                club = group[0]['club']
                start_date = group[0]['date']
                end_date = group[-1]['date']
                structured_periods.append({'club': club, 'start_date': start_date, 'end_date': end_date})

            current_club = current_club_seed or (structured_periods[0]['club'] if structured_periods else None)
            current_club_country = get_club_country(current_club)
            current_club_start = structured_periods[0]['start_date'] if structured_periods else calendar[0]
            current_previous_club = initial_previous_club
            current_previous_club_country = initial_previous_club_country

            for i, date in enumerate(calendar):
                current_period = None
                for period in structured_periods:
                    if period['start_date'] <= date <= period['end_date']:
                        current_period = period
                        break

                if current_period and current_club != current_period['club']:
                    current_previous_club = current_club
                    current_previous_club_country = get_club_country(current_previous_club)
                    current_club = current_period['club']
                    current_club_country = get_club_country(current_club)
                    current_club_start = current_period['start_date']

                previous_club_values[i] = current_previous_club
                previous_club_country_values[i] = current_previous_club_country
                current_club_values[i] = current_club
                current_club_country_values[i] = current_club_country
                seniority_days_values[i] = max(0, (date - current_club_start).days) if current_club_start else 0

        else:
            current_club = current_club_seed or initial_previous_club
            current_club_country = get_club_country(current_club)
            current_club_start = calendar[0]

            for i, date in enumerate(calendar):
                current_club_values[i] = current_club
                current_club_country_values[i] = current_club_country
                seniority_days_values[i] = max(0, (date - current_club_start).days)

    else:
        current_club = current_club_seed or initial_previous_club
        current_club_country = get_club_country(current_club)
        joined_days = (calendar - player_row['joined_on']).days
        seniority_days_values = [max(0, days) for days in joined_days]

        for i in range(n_days):
            current_club_values[i] = current_club
            current_club_country_values[i] = current_club_country
    
    # Optimized teams calculation (including national teams)
    matches_for_team_counts = player_matches[['date', 'home_team', 'away_team']].copy()

    daily_team_sets = matches_for_team_counts.groupby('date').agg({
        'home_team': lambda x: set(x.dropna()),
        'away_team': lambda x: set(x.dropna())
    }) if not matches_for_team_counts.empty else pd.DataFrame(columns=['home_team', 'away_team'])

    unique_teams_seen: set = set()
    teams_today_values = [0] * n_days
    cum_teams_values = [0] * n_days

    for i, date in enumerate(calendar):
        if date in daily_team_sets.index:
            home_set = daily_team_sets.loc[date, 'home_team']
            away_set = daily_team_sets.loc[date, 'away_team']
            day_teams = set()
            for team in home_set.union(away_set):
                if pd.notna(team):
                    day_teams.add(clean_club_label(team))
            day_teams.discard(None)
            teams_today_values[i] = len(day_teams)
            unique_teams_seen.update(day_teams)
        cum_teams_values[i] = len(unique_teams_seen)
    
    # Calculate season counter (starts at 0, increments each new season)
    seasons_seen: set = set()
    season_counter_values = [0] * n_days
    
    for i, date in enumerate(calendar):
        current_season = get_football_season(date)
        if current_season and current_season not in seasons_seen:
            seasons_seen.add(current_season)
        # Count is number of seasons seen minus 1 (since we start at 0)
        season_counter_values[i] = len(seasons_seen) - 1 if len(seasons_seen) > 0 else 0
    
    # Create DataFrame with all pre-calculated values
    out = pd.DataFrame({
        'age': age_values,
        'position': position_values,
        'nationality1': nationality1_values,
        'nationality2': nationality2_values,
        'height_cm': height_cm_values,
        'dominant_foot': dominant_foot_values,
        'previous_club': previous_club_values,
        'previous_club_country': previous_club_country_values,
        'current_club': current_club_values,
        'current_club_country': current_club_country_values,
        'teams_today': teams_today_values,
        'cum_teams': cum_teams_values,
        'seniority_days': seniority_days_values,
        'seasons_count': season_counter_values
    }, index=calendar)
    
    return out

def build_daily_match_series_optimized(
    matches: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    player_row: pd.Series,
    profile_series: Optional[pd.DataFrame],
    player_career: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """Optimized daily match series building"""
    team_metrics = compute_team_result_metrics(matches, calendar, player_row, profile_series, player_career)
    if matches.empty:
        # Return empty DataFrame with correct structure
        n_days = len(calendar)
        empty_data = {
            'matches_played': [0] * n_days,
            'minutes_played_numeric': [0] * n_days,
            'goals_numeric': [0] * n_days,
            'assists_numeric': [0] * n_days,
            'yellow_cards_numeric': [0] * n_days,
            'red_cards_numeric': [0] * n_days,
            'matches_bench_unused': [0] * n_days,
            'matches_not_selected': [0] * n_days,
            'matches_injured': [0] * n_days,
            'cum_minutes_played_numeric': [0] * n_days,
            'cum_goals_numeric': [0] * n_days,
            'cum_assists_numeric': [0] * n_days,
            'cum_yellow_cards_numeric': [0] * n_days,
            'cum_red_cards_numeric': [0] * n_days,
            'cum_matches_played': [0] * n_days,
            'cum_matches_bench': [0] * n_days,
            'cum_matches_not_selected': [0] * n_days,
            'cum_matches_injured': [0] * n_days,
            'cum_competitions': [0] * n_days,
            'goals_per_match': [0.0] * n_days,
            'assists_per_match': [0.0] * n_days,
            'minutes_per_match': [0.0] * n_days,
            'days_since_last_match': [999] * n_days,
            'matches': [0] * n_days,
            'career_matches': [0] * n_days,
            'career_goals': [0] * n_days,
            'career_assists': [0] * n_days,
            'career_minutes': [0] * n_days,
            'last_match_position': [''] * n_days,
            'position_match_default': [player_row['position']] * n_days,
            # ENHANCED FEATURES - Use benfica-parity defaults
            'competition_importance': [0] * n_days,  # 0 = no match
            'avg_competition_importance': [0.0] * n_days,
            'month': [0] * n_days,  # Month of the year (1-12)
            'disciplinary_action': [0] * n_days,
            'cum_disciplinary_actions': [0] * n_days,
            'teams_this_season': [0] * n_days,
            'teams_last_season': [0] * n_days,
            'teams_season_today': [0] * n_days,
            'season_team_diversity': [0] * n_days,
            'national_team_appearances': [0] * n_days,
            'national_team_minutes': [0] * n_days,
            'days_since_last_national_match': [np.nan] * n_days,
            'national_team_this_season': [0] * n_days,
            'national_team_last_season': [0] * n_days,
            'national_team_frequency': [0.0] * n_days,
            'senior_national_team': [0] * n_days,
            'national_team_intensity': [0] * n_days,
            'competition_intensity': [1] * n_days,
            'competition_level': [1] * n_days,
            'competition_diversity': [0] * n_days,
            'international_competitions': [0] * n_days,
            'cup_competitions': [0] * n_days,
            'competition_frequency': [0.0] * n_days,
            'competition_experience': [0] * n_days,
            'competition_pressure': [1] * n_days,
            'club_cum_goals': [0] * n_days,
            'club_cum_assists': [0] * n_days,
            'club_cum_minutes': [0] * n_days,
            'club_cum_matches_played': [0] * n_days,
            'club_cum_yellow_cards': [0] * n_days,
            'club_cum_red_cards': [0] * n_days,
            'club_goals_per_match': [0.0] * n_days,
            'club_assists_per_match': [0.0] * n_days,
            'club_minutes_per_match': [0.0] * n_days,
            'club_seniority_x_goals_per_match': [0.0] * n_days
        }
        empty_data.update(team_metrics)
        return pd.DataFrame(empty_data, index=calendar)
    
    # Group by date for daily aggregation
    daily_matches = matches.groupby('date').agg({
        'minutes_played_numeric': 'sum',
        'goals_numeric': 'sum',
        'assists_numeric': 'sum',
        'yellow_cards_numeric': 'sum',
        'red_cards_numeric': 'sum',
        'position': 'last',
        'competition_importance': 'mean',
        'disciplinary_action': 'sum'
    }).reset_index()
    
    # Determine participation status for each match
    matches['participation'] = determine_match_participation_optimized(matches)
    
    # Add participation to daily_matches
    participation_by_date = matches.groupby('date')['participation'].apply(list).reset_index()
    daily_matches = daily_matches.merge(participation_by_date, on='date', how='left')
    
    # Create daily series
    daily = pd.DataFrame(index=calendar)
    
    # Initialize arrays
    n_days = len(calendar)
    matches_played = [0] * n_days
    minutes_played = [0] * n_days
    goals = [0] * n_days
    assists = [0] * n_days
    yellow_cards = [0] * n_days
    red_cards = [0] * n_days
    matches_bench_unused = [0] * n_days
    matches_not_selected = [0] * n_days
    matches_injured = [0] * n_days
    competition_importance = [1] * n_days
    month = [0] * n_days  # Month of the year (1-12)
    disciplinary_action = [0] * n_days
    last_match_position = [''] * n_days
    
    # Fill daily values
    for _, row in daily_matches.iterrows():
        date = row['date']
        if date in calendar:
            idx = calendar.get_loc(date)
            
            # Count participation types
            participations = row['participation']
            matches_bench_unused[idx] = participations.count('bench_unused')
            matches_not_selected[idx] = participations.count('not_selected')
            matches_injured[idx] = participations.count('injured')
            
            # Only count as matches_played if player actually played (not bench, not selected, or injured)
            matches_played[idx] = participations.count('played')
            
            minutes_played[idx] = row['minutes_played_numeric']
            goals[idx] = row['goals_numeric']
            assists[idx] = row['assists_numeric']
            yellow_cards[idx] = row['yellow_cards_numeric']
            red_cards[idx] = row['red_cards_numeric']
            # Don't set enhanced features here - they will be set by benfica-parity function later
            # competition_importance[idx] = row['competition_importance']
            # month is calculated directly from calendar dates below
            # disciplinary_action[idx] = row['disciplinary_action']
            
            # Only update last_match_position if player actually played and position is valid
            if row['minutes_played_numeric'] > 0 and pd.notna(row['position']):
                pos = str(row['position'])
                # Filter out non-position values
                non_position_indicators = [
                    'was not in the squad', 'unused substitute', 'information not yet available',
                    'injury', 'lesion', 'infection', 'punished', 'permit', 'management', 'problems'
                ]
                if not any(indicator in pos.lower() for indicator in non_position_indicators):
                    last_match_position[idx] = pos
    
    # Calculate cumulative values
    cum_minutes_played = np.cumsum(minutes_played)
    cum_goals = np.cumsum(goals)
    cum_assists = np.cumsum(assists)
    cum_yellow_cards = np.cumsum(yellow_cards)
    cum_red_cards = np.cumsum(red_cards)
    cum_matches_played = np.cumsum(matches_played)
    cum_matches_bench = np.cumsum(matches_bench_unused)
    cum_matches_not_selected = np.cumsum(matches_not_selected)
    cum_matches_injured = np.cumsum(matches_injured)
    cum_disciplinary_actions = np.cumsum(disciplinary_action)
    
    # Calculate total cumulative matches (all types) - as in original script
    cum_matches = [p + b + n + i for p, b, n, i in zip(matches_played, matches_bench_unused, matches_not_selected, matches_injured)]
    cum_matches = np.cumsum(cum_matches)
    
    # Calculate ratios
    goals_per_match = [g/m if m > 0 else 0.0 for g, m in zip(cum_goals, cum_matches_played)]
    assists_per_match = [a/m if m > 0 else 0.0 for a, m in zip(cum_assists, cum_matches_played)]
    minutes_per_match = [mp/m if m > 0 else 0.0 for mp, m in zip(cum_minutes_played, cum_matches_played)]
    
    # Calculate days since last match
    days_since_last_match = [999] * n_days  # FIXED: Use 999 as default instead of np.nan
    last_match_idx = -1
    for i in range(n_days):
        if matches_played[i] > 0:
            last_match_idx = i
        if last_match_idx >= 0:
            days_since_last_match[i] = i - last_match_idx
    
    # Carry forward last_match_position (maintain last valid position until new one is found)
    current_last_position = None
    for i in range(n_days):
        if last_match_position[i] and last_match_position[i] != '':
            current_last_position = last_match_position[i]
        last_match_position[i] = current_last_position
    
    # Career totals (same as cumulative for single player)
    career_matches = cum_matches  # Total cumulative matches including all types
    career_goals = cum_goals
    career_assists = cum_assists
    career_minutes = cum_minutes_played
    
    # Matches feature (only matches where player actually played) - as in original script
    total_matches = matches_played
    
    # Position match default - binary indicator (1 if position different from default, 0 if same)
    position_match_default = [None] * n_days
    default_position = player_row['position'] if pd.notna(player_row['position']) else None
    
    for i in range(n_days):
        current_position = last_match_position[i]
        if current_position is not None and default_position is not None:
            position_match_default[i] = 1 if current_position != default_position else 0
        elif current_position is not None:
            position_match_default[i] = 0  # No default position to compare against
        else:
            position_match_default[i] = None  # No position played yet
    
    # Calculate month directly from calendar dates (1-12)
    month = [date.month for date in calendar]
    
    # ENHANCED: Calculate enhanced features dynamically using benfica-parity logic
    enhanced_features = calculate_enhanced_features_dynamically(matches, calendar, player_row)
    competition_importance = enhanced_features['competition_importance']
    disciplinary_action = enhanced_features['disciplinary_action']
    avg_competition_importance = enhanced_features['avg_competition_importance']
    cum_disciplinary_actions = enhanced_features['cum_disciplinary_actions']
    
    # ENHANCED: Calculate cumulative competitions
    cum_competitions = [1 if ci > 1 else 0 for ci in competition_importance]
    cum_competitions = np.cumsum(cum_competitions)
    
    # ENHANCED: Season-based team features
    teams_this_season = [0] * n_days
    teams_last_season = [0] * n_days
    teams_season_today = [0] * n_days
    season_team_diversity = [0] * n_days
    
    if not matches.empty:
        matches_copy = matches.copy()
        matches_copy['football_season'] = matches_copy['date'].apply(get_football_season)
        matches_copy.sort_values('date', inplace=True)

        season_team_progression: Dict[str, List[Tuple[pd.Timestamp, int]]] = {}
        season_match_progression: Dict[str, List[Tuple[pd.Timestamp, int]]] = {}
        season_daily_team_counts: Dict[Tuple[str, pd.Timestamp], int] = {}

        for season, season_matches in matches_copy.groupby('football_season'):
            season_matches = season_matches.sort_values('date')
            cumulative_teams: set = set()
            progression: List[Tuple[pd.Timestamp, int]] = []
            match_prog: List[Tuple[pd.Timestamp, int]] = []
            match_count = 0

            for _, match in season_matches.iterrows():
                match_count += 1
                daily_teams = set()
                home_team = match['home_team']
                away_team = match['away_team']

                if pd.notna(home_team) and is_main_club_team(home_team):
                    daily_teams.add(home_team)
                    cumulative_teams.add(home_team)
                if pd.notna(away_team) and is_main_club_team(away_team):
                    daily_teams.add(away_team)
                    cumulative_teams.add(away_team)

                progression.append((match['date'], len(cumulative_teams)))
                match_prog.append((match['date'], match_count))
                season_daily_team_counts[(season, match['date'])] = len(daily_teams)

            season_team_progression[season] = progression
            season_match_progression[season] = match_prog

        season_team_indices = {season: 0 for season in season_team_progression}
        season_match_indices = {season: 0 for season in season_match_progression}

        def get_progress_value(progression_map, indices_map, season_key, current_date):
            progression_list = progression_map.get(season_key)
            if not progression_list:
                return 0
            idx = indices_map.get(season_key, 0)
            while idx < len(progression_list) and progression_list[idx][0] <= current_date:
                idx += 1
            indices_map[season_key] = idx
            if idx == 0:
                return 0
            return progression_list[idx - 1][1]

        for i, date in enumerate(calendar):
            current_season = get_football_season(date)
            prev_season = f"{date.year - 1}/{date.year}" if date.month >= 7 else f"{date.year - 2}/{date.year - 1}"

            teams_this_season[i] = get_progress_value(season_team_progression, season_team_indices, current_season, date)
            teams_last_season[i] = get_progress_value(season_team_progression, season_team_indices, prev_season, date)

            teams_season_today[i] = season_daily_team_counts.get((current_season, date), 0)

            matches_this_season = get_progress_value(season_match_progression, season_match_indices, current_season, date)
            if matches_this_season > 0:
                season_team_diversity[i] = teams_this_season[i] / matches_this_season
            else:
                season_team_diversity[i] = 0
    
    # ENHANCED: National team features
    national_team_appearances = [0] * n_days
    national_team_minutes = [0] * n_days
    days_since_last_national_match = [np.nan] * n_days
    national_team_this_season = [0] * n_days
    national_team_last_season = [0] * n_days
    national_team_frequency = [0.0] * n_days
    senior_national_team = [0] * n_days
    national_team_intensity = [0] * n_days
    
    # ENHANCED: Competition analysis features
    competition_intensity = [1] * n_days
    competition_level = [1] * n_days
    competition_diversity = [0] * n_days
    international_competitions = [0] * n_days
    cup_competitions = [0] * n_days
    competition_frequency = [0.0] * n_days
    competition_experience = [0] * n_days
    competition_pressure = [1] * n_days
    
    # ENHANCED: Club performance features
    club_cum_goals = [0] * n_days
    club_cum_assists = [0] * n_days
    club_cum_minutes = [0] * n_days
    club_cum_matches_played = [0] * n_days
    club_cum_yellow_cards = [0] * n_days
    club_cum_red_cards = [0] * n_days
    club_goals_per_match = [0.0] * n_days
    club_assists_per_match = [0.0] * n_days
    club_minutes_per_match = [0.0] * n_days
    club_seniority_x_goals_per_match = [0.0] * n_days
    
    # ENHANCED: Substitution-based features
    substitution_on_count = [0] * n_days
    substitution_off_count = [0] * n_days
    late_substitution_on_count = [0] * n_days  # Subbed in after 75th minute
    early_substitution_off_count = [0] * n_days  # Subbed out before 60th minute
    impact_substitution_count = [0] * n_days  # Subbed in and scored/assisted
    tactical_substitution_count = [0] * n_days  # Subbed out for tactical reasons (not injury)
    substitution_minutes_played = [0] * n_days  # Minutes played when subbed in
    substitution_efficiency = [0.0] * n_days  # Goals/assists per minute when subbed in
    substitution_mood_indicator = [0.0] * n_days  # Positive/negative substitution impact
    consecutive_substitutions = [0] * n_days  # Days with consecutive substitution patterns
    
    # Calculate substitution-based features
    if not matches.empty:
        # Create a copy to avoid modifying the original
        matches_copy = matches.copy()
        
        # Parse substitution data
        def parse_minutes(minutes_str):
            """Parse minutes from string format like '90'' or '31''"""
            if pd.isna(minutes_str):
                return 0
            try:
                return int(str(minutes_str).replace("'", ""))
            except:
                return 0
        
        def parse_substitution_minute(sub_str):
            """Parse substitution minute from string format like '59'' or '90''"""
            if pd.isna(sub_str):
                return None
            try:
                return int(str(sub_str).replace("'", ""))
            except:
                return None
        
        # Add parsed columns
        matches_copy['minutes_played_parsed'] = matches_copy['minutes_played'].apply(parse_minutes)
        matches_copy['substitution_on_minute'] = matches_copy['substitutions_on'].apply(parse_substitution_minute)
        matches_copy['substitution_off_minute'] = matches_copy['substitutions_off'].apply(parse_substitution_minute)
        
        # Filter to only matches where player actually played (has meaningful minutes_played)
        played_matches = matches_copy[matches_copy['minutes_played'].notna()]
        
        # Calculate substitution features for each match day only
        # Group by date to handle potential duplicates
        substitution_on_total = 0
        substitution_off_total = 0
        
        for date, date_matches in played_matches.groupby('date'):
            if pd.notna(date):
                # Find the index for this date in the calendar
                try:
                    i = calendar.get_loc(date)
                except KeyError:
                    continue  # Skip if date not in calendar
                
                # Process each match for this date
                for _, match in date_matches.iterrows():
                    match_minutes_played = match['minutes_played_parsed']
                    sub_on_minute = match['substitution_on_minute']
                    sub_off_minute = match['substitution_off_minute']
                    match_goals = match['goals'] if pd.notna(match['goals']) else 0
                    match_assists = match['assists'] if pd.notna(match['assists']) else 0
                    
                    # Substitution ON features - only count if there's actual substitution data
                    if sub_on_minute is not None and not pd.isna(sub_on_minute):
                        substitution_on_count[i] += 1
                        substitution_on_total += 1
                        substitution_minutes_played[i] += match_minutes_played
                        
                        # Late substitution (after 75th minute)
                        if sub_on_minute >= 75:
                            late_substitution_on_count[i] += 1
                        
                        # Impact substitution (scored or assisted)
                        if match_goals > 0 or match_assists > 0:
                            impact_substitution_count[i] += 1
                        
                        # Substitution efficiency (goals+assists per minute)
                        if match_minutes_played > 0:
                            substitution_efficiency[i] += (match_goals + match_assists) / match_minutes_played
                    
                    # Substitution OFF features - only count if there's actual substitution data
                    if sub_off_minute is not None and not pd.isna(sub_off_minute):
                        substitution_off_count[i] += 1
                        substitution_off_total += 1
                        
                        # Early substitution (before 60th minute)
                        if sub_off_minute < 60:
                            early_substitution_off_count[i] += 1
                        
                        # Tactical substitution (not injury-related)
                        position = str(match['position']).lower()
                        injury_indicators = ['injury', 'cruciate ligament tear', 'pubalgia', 'knee injury', 'muscle injury']
                        if not any(indicator in position for indicator in injury_indicators):
                            tactical_substitution_count[i] += 1
        

        
        # Calculate substitution mood indicator for each day
        for i in range(n_days):
            # Positive: impact substitutions, late substitutions with goals
            # Negative: early substitutions, tactical substitutions without impact
            positive_impact = impact_substitution_count[i] + (late_substitution_on_count[i] * 0.5)
            negative_impact = early_substitution_off_count[i] + (tactical_substitution_count[i] * 0.3)
            substitution_mood_indicator[i] = positive_impact - negative_impact
            
            # Consecutive substitutions (simplified - days with any substitution)
            if substitution_on_count[i] > 0 or substitution_off_count[i] > 0:
                consecutive_substitutions[i] = 1
    
    # ENHANCED: Calculate national team features using benfica-parity logic
    national_team_features = calculate_national_team_features_benfica_parity(matches, calendar)
    national_team_appearances = national_team_features['national_team_appearances']
    national_team_minutes = national_team_features['national_team_minutes']
    days_since_last_national_match = national_team_features['days_since_last_national_match']
    national_team_this_season = national_team_features['national_team_this_season']
    national_team_last_season = national_team_features['national_team_last_season']
    national_team_frequency = national_team_features['national_team_frequency']
    senior_national_team = national_team_features['senior_national_team']
    national_team_intensity = national_team_features['national_team_intensity']
    
    # Fix days_since_last_national_match to use NaN instead of 999
    for i in range(n_days):
        if national_team_appearances[i] == 0:
            days_since_last_national_match[i] = np.nan
    
    # ENHANCED: Calculate complex derived features using benfica-parity logic
    complex_features = calculate_complex_derived_features_benfica_parity(matches, calendar)
    competition_intensity = complex_features['competition_intensity']
    competition_level = complex_features['competition_level']
    competition_pressure = complex_features['competition_pressure']
    teams_this_season = complex_features['teams_this_season']
    season_team_diversity = complex_features['season_team_diversity']
    cum_competitions = complex_features['cum_competitions']
    competition_diversity = complex_features['competition_diversity']
    competition_frequency = complex_features['competition_frequency']
    competition_experience = complex_features['competition_experience']
    international_competitions = complex_features['international_competitions']
    cup_competitions = complex_features['cup_competitions']
    transfermarkt_score_recent = complex_features.get('transfermarkt_score_recent', [0.0] * n_days)
    transfermarkt_score_cum = complex_features.get('transfermarkt_score_cum', [0.0] * n_days)
    transfermarkt_score_avg = complex_features.get('transfermarkt_score_avg', [0.0] * n_days)
    transfermarkt_score_rolling5 = complex_features.get('transfermarkt_score_rolling5', [0.0] * n_days)
    transfermarkt_score_matches = complex_features.get('transfermarkt_score_matches', [0] * n_days)
    
    # Complex derived features are now calculated by benfica-parity function above
    
    # ENHANCED: Calculate club performance features day-by-day (no data leakage)
    if not matches.empty:
        # Filter to only main club matches (exclude national teams, junior teams, etc.)
        # IMPORTANT: Require BOTH teams to be main clubs to avoid national team matches
        main_club_matches = matches[
            (matches['home_team'].apply(is_main_club_team)) & 
            (matches['away_team'].apply(is_main_club_team))
        ].copy()
        
        if not main_club_matches.empty:
            # Determine player's club for each match
            player_clubs = []
            for _, match in main_club_matches.iterrows():
                home_team = match['home_team']
                away_team = match['away_team']
                
                # Both teams are main clubs, need to determine player's club
                # Use the team that appears more frequently in player's career (club matches only)
                home_count = len(main_club_matches[(main_club_matches['home_team'] == home_team) | (main_club_matches['away_team'] == home_team)])
                away_count = len(main_club_matches[(main_club_matches['home_team'] == away_team) | (main_club_matches['away_team'] == away_team)])
                player_club = home_team if home_count >= away_count else away_team
                
                # Parse cards data
                def parse_cards(cards_str):
                    if pd.isna(cards_str):
                        return 0
                    try:
                        return int(str(cards_str))
                    except:
                        return 0
                
                player_clubs.append({
                    'date': match['date'],
                    'club': player_club,
                    'goals': match['goals'] if pd.notna(match['goals']) else 0,
                    'assists': match['assists'] if pd.notna(match['assists']) else 0,
                    'minutes_played': match['minutes_played'],
                    'yellow_cards': parse_cards(match['yellow_cards']),
                    'red_cards': parse_cards(match['red_cards'])
                })
            
            # Sort by date and calculate club-specific metrics
            player_clubs.sort(key=lambda x: x['date'])
            
            # Track club progression and reset metrics on club changes
            current_club = None
            club_goals = 0
            club_assists = 0
            club_minutes = 0
            club_matches = 0
            club_yellow_cards = 0
            club_red_cards = 0
            
            for idx, club_info in enumerate(player_clubs):
                club = club_info['club']
                date = club_info['date']
                
                # Reset metrics if club changes
                if current_club is not None and club != current_club:
                    club_goals = 0
                    club_assists = 0
                    club_minutes = 0
                    club_matches = 0
                    club_yellow_cards = 0
                    club_red_cards = 0
                
                current_club = club
                
                # Update club metrics - handle datetime.time values
                def safe_numeric(value):
                    if pd.isna(value):
                        return 0
                    if isinstance(value, (int, float)):
                        return int(value)
                    if hasattr(value, 'hour'):  # Check if it's a time object
                        return 0  # Convert time to 0
                    try:
                        return int(value)
                    except:
                        return 0
                
                club_goals += safe_numeric(club_info['goals'])
                club_assists += safe_numeric(club_info['assists'])
                
                # Parse minutes
                if pd.notna(club_info['minutes_played']):
                    try:
                        minutes = int(str(club_info['minutes_played']).replace("'", ""))
                        club_minutes += minutes
                    except:
                        pass
                
                club_matches += 1
                club_yellow_cards += club_info['yellow_cards']
                club_red_cards += club_info['red_cards']
                
                # Find the index for this date in the calendar
                try:
                    i = calendar.get_loc(date)
                except KeyError:
                    continue
                
                # Determine end index: up to next match date (if exists) or end of calendar
                if idx + 1 < len(player_clubs):
                    # Check if next match is same club - if different club, stop before it
                    next_match_date = player_clubs[idx + 1]['date']
                    next_match_club = player_clubs[idx + 1]['club']
                    if next_match_club != club:
                        # Club changes at next match, so update only up to day before next match
                        try:
                            next_idx = calendar.get_loc(next_match_date)
                            end_idx = min(next_idx, n_days)
                        except KeyError:
                            end_idx = n_days
                    else:
                        # Same club, update up to next match date
                        try:
                            next_idx = calendar.get_loc(next_match_date)
                            end_idx = min(next_idx, n_days)
                        except KeyError:
                            end_idx = n_days
                else:
                    # Last match, update to end of calendar
                    end_idx = n_days
                
                # Update club features from this match date up to (but not including) next match or end
                for j in range(i, end_idx):
                    club_cum_goals[j] = club_goals
                    club_cum_assists[j] = club_assists
                    club_cum_minutes[j] = club_minutes
                    club_cum_matches_played[j] = club_matches
                    club_cum_yellow_cards[j] = club_yellow_cards
                    club_cum_red_cards[j] = club_red_cards
                    
                    # Calculate per-match metrics
                    if club_matches > 0:
                        club_goals_per_match[j] = club_goals / club_matches
                        club_assists_per_match[j] = club_assists / club_matches
                        club_minutes_per_match[j] = club_minutes / club_matches
                    
                    # Calculate seniority interaction (simplified - using days since first match)
                    if j > 0:
                        club_seniority_x_goals_per_match[j] = j * club_goals_per_match[j]
    

    
    # Create DataFrame
    out = pd.DataFrame({
        'matches_played': matches_played,
        'minutes_played_numeric': minutes_played,
        'goals_numeric': goals,
        'assists_numeric': assists,
        'yellow_cards_numeric': yellow_cards,
        'red_cards_numeric': red_cards,
        'matches_bench_unused': matches_bench_unused,
        'matches_not_selected': matches_not_selected,
        'matches_injured': matches_injured,
        'cum_minutes_played_numeric': cum_minutes_played,
        'cum_goals_numeric': cum_goals,
        'cum_assists_numeric': cum_assists,
        'cum_yellow_cards_numeric': cum_yellow_cards,
        'cum_red_cards_numeric': cum_red_cards,
        'cum_matches_played': cum_matches_played,
        'cum_matches_bench': cum_matches_bench,
        'cum_matches_not_selected': cum_matches_not_selected,
        'cum_matches_injured': cum_matches_injured,
        'cum_competitions': cum_competitions,
        'goals_per_match': goals_per_match,
        'assists_per_match': assists_per_match,
        'minutes_per_match': minutes_per_match,
        'days_since_last_match': days_since_last_match,
        'matches': total_matches,
        'career_matches': career_matches,
        'career_goals': career_goals,
        'career_assists': career_assists,
        'career_minutes': career_minutes,
        'last_match_position': last_match_position,
        'position_match_default': position_match_default,
        # ENHANCED FEATURES
        'competition_importance': competition_importance,
        'avg_competition_importance': avg_competition_importance,
        'month': month,
        'disciplinary_action': disciplinary_action,
        'cum_disciplinary_actions': cum_disciplinary_actions,
        'teams_this_season': teams_this_season,
        'teams_last_season': teams_last_season,
        'teams_season_today': teams_season_today,
        'season_team_diversity': season_team_diversity,
        'national_team_appearances': national_team_appearances,
        'national_team_minutes': national_team_minutes,
        'days_since_last_national_match': days_since_last_national_match,
        'national_team_this_season': national_team_this_season,
        'national_team_last_season': national_team_last_season,
        'national_team_frequency': national_team_frequency,
        'senior_national_team': senior_national_team,
        'national_team_intensity': national_team_intensity,
        'competition_intensity': competition_intensity,
        'competition_level': competition_level,
        'competition_diversity': competition_diversity,
        'international_competitions': international_competitions,
        'cup_competitions': cup_competitions,
        'competition_frequency': competition_frequency,
        'competition_experience': competition_experience,
        'competition_pressure': competition_pressure,
        'transfermarkt_score_recent': transfermarkt_score_recent,
        'transfermarkt_score_cum': transfermarkt_score_cum,
        'transfermarkt_score_avg': transfermarkt_score_avg,
        'transfermarkt_score_rolling5': transfermarkt_score_rolling5,
        'transfermarkt_score_matches': transfermarkt_score_matches,
        'club_cum_goals': club_cum_goals,
        'club_cum_assists': club_cum_assists,
        'club_cum_minutes': club_cum_minutes,
        'club_cum_matches_played': club_cum_matches_played,
        'club_cum_yellow_cards': club_cum_yellow_cards,
        'club_cum_red_cards': club_cum_red_cards,
        'club_goals_per_match': club_goals_per_match,
        'club_assists_per_match': club_assists_per_match,
        'club_minutes_per_match': club_minutes_per_match,
        'club_seniority_x_goals_per_match': club_seniority_x_goals_per_match,
        # ENHANCED SUBSTITUTION FEATURES
        'substitution_on_count': substitution_on_count,
        'substitution_off_count': substitution_off_count,
        'late_substitution_on_count': late_substitution_on_count,
        'early_substitution_off_count': early_substitution_off_count,
        'impact_substitution_count': impact_substitution_count,
        'tactical_substitution_count': tactical_substitution_count,
        'substitution_minutes_played': substitution_minutes_played,
        'substitution_efficiency': substitution_efficiency,
        'substitution_mood_indicator': substitution_mood_indicator,
        'consecutive_substitutions': consecutive_substitutions
    }, index=calendar)
    for key, values in team_metrics.items():
        out[key] = values
    
    return out

def build_daily_injury_series_optimized(injuries, calendar):
    """Enhanced daily injury series building with complete feature set"""
    if injuries.empty:
        # Return empty DataFrame with correct structure
        n_days = len(calendar)
        empty_data = {
            'cum_inj_starts': [0] * n_days,
            'cum_inj_days': [0] * n_days,
            'days_since_last_injury': [999] * n_days,  # FIXED: Use 999 as default instead of np.nan
            'days_since_last_injury_ended': [999] * n_days,
            'avg_injury_duration': [0.0] * n_days,
            'injury_frequency': [0.0] * n_days,
            # ENHANCED INJURY FEATURES
            'avg_injury_severity': [1.0] * n_days,
            'max_injury_severity': [1] * n_days,
            'lower_leg_injuries': [0] * n_days,
            'knee_injuries': [0] * n_days,
            'upper_leg_injuries': [0] * n_days,
            'hip_injuries': [0] * n_days,
            'upper_body_injuries': [0] * n_days,
            'head_injuries': [0] * n_days,
            'illness_count': [0] * n_days,
            'other_injuries': [0] * n_days,
            'physio_injury_ratio': [0.0] * n_days,
            # MISSING FEATURES
            'cum_matches_injured': [0] * n_days
        }
        return pd.DataFrame(empty_data, index=calendar)
    
    # Create daily injury tracking
    daily = pd.DataFrame(index=calendar)
    daily['inj_starts'] = 0
    daily['inj_days'] = 0
    daily['cum_matches_injured'] = 0
    daily['injury_end_marker'] = 0
    
    # Mark injury periods
    for _, injury in injuries.iterrows():
        # Handle NaT values in fromDate (critical - can't process without start date)
        if pd.isna(injury['fromDate']):
            continue
            
        start_date = injury['fromDate'].normalize()
        
        # Handle NaT values in untilDate (use default recovery period)
        if pd.isna(injury['untilDate']):
            # For injuries without recovery date, use default recovery period
            # Severe injuries (severity 4-5) get longer recovery, others get standard
            if 'injury_severity' in injury and injury['injury_severity'] >= 4:
                # Severe injuries: 90 days recovery (e.g., cruciate ligament tear)
                end_date = start_date + pd.Timedelta(days=90)
                print(f"   ⚠️  Injury {start_date.strftime('%Y-%m-%d')} has no recovery date, using 90-day default")
            else:
                # Standard injuries: 30 days recovery
                end_date = start_date + pd.Timedelta(days=30)
                print(f"   ⚠️  Injury {start_date.strftime('%Y-%m-%d')} has no recovery date, using 30-day default")
        else:
            end_date = injury['untilDate'].normalize()
        
        # Mark injury start (ALWAYS mark if we have a start date)
        if start_date in calendar:
            daily.loc[start_date, 'inj_starts'] = 1
        
        # Mark injury period
        injury_period = calendar[(calendar >= start_date) & (calendar <= end_date)]
        daily.loc[injury_period, 'inj_days'] = 1
        
        recovery_date = end_date + pd.Timedelta(days=1)
        if calendar[0] <= recovery_date <= calendar[-1]:
            if recovery_date in daily.index:
                daily.loc[recovery_date, 'injury_end_marker'] = 1
    
    # Calculate cumulative values
    daily['cum_inj_starts'] = daily['inj_starts'].cumsum()
    daily['cum_inj_days'] = daily['inj_days'].cumsum()
    
    # Calculate days since last injury
    daily['days_since_last_injury'] = 999  # FIXED: Use 999 as default instead of np.nan
    last_injury_idx = -1
    for i in range(len(calendar)):
        if daily.iloc[i]['inj_starts'] > 0:
            last_injury_idx = i
        if last_injury_idx >= 0:
            daily.iloc[i, daily.columns.get_loc('days_since_last_injury')] = i - last_injury_idx
    
    # Calculate days since last injury ended (recovery)
    days_since_last_injury_ended = [999] * len(calendar)
    last_recovery_idx = None
    injury_end_marker_values = daily['injury_end_marker'].tolist()
    for i in range(len(calendar)):
        if injury_end_marker_values[i] > 0:
            last_recovery_idx = i
            days_since_last_injury_ended[i] = 0
        elif last_recovery_idx is not None:
            days_since_last_injury_ended[i] = i - last_recovery_idx
    daily['days_since_last_injury_ended'] = days_since_last_injury_ended
    
    # Calculate average injury duration
    if not injuries.empty:
        avg_duration = injuries['duration_days'].mean()
    else:
        avg_duration = 0.0
    
    daily['avg_injury_duration'] = avg_duration
    
    # Calculate injury frequency (injuries per year)
    total_injuries = len(injuries)
    if total_injuries > 0 and len(calendar) > 0:
        years_span = len(calendar) / 365.25
        injury_frequency = total_injuries / years_span
    else:
        injury_frequency = 0.0
    
    daily['injury_frequency'] = injury_frequency
    
    # ENHANCED: Calculate injury severity features using benfica-parity logic
    injury_features = calculate_injury_features_benfica_parity(injuries, calendar)
    daily['avg_injury_severity'] = injury_features['avg_injury_severity']
    daily['max_injury_severity'] = injury_features['max_injury_severity']
    daily['lower_leg_injuries'] = injury_features['lower_leg_injuries']
    daily['knee_injuries'] = injury_features['knee_injuries']
    daily['upper_leg_injuries'] = injury_features['upper_leg_injuries']
    daily['hip_injuries'] = injury_features['hip_injuries']
    daily['upper_body_injuries'] = injury_features['upper_body_injuries']
    daily['head_injuries'] = injury_features['head_injuries']
    daily['illness_count'] = injury_features['illness_count']
    daily['other_injuries'] = injury_features['other_injuries']
    daily['physio_injury_ratio'] = injury_features['physio_injury_ratio']
    
    # Select only the required columns
    required_columns = [
        'cum_inj_starts', 'cum_inj_days', 'days_since_last_injury',
        'days_since_last_injury_ended', 'avg_injury_duration', 'injury_frequency',
        'avg_injury_severity', 'max_injury_severity',
        'lower_leg_injuries', 'knee_injuries', 'upper_leg_injuries',
        'hip_injuries', 'upper_body_injuries', 'head_injuries',
        'illness_count', 'other_injuries', 'physio_injury_ratio',
        'cum_matches_injured'  # Added missing feature
    ]
    
    return daily[required_columns]

def build_daily_interaction_features_optimized(profile_series, match_series, injury_series):
    """Build interaction features between different feature types"""
    interactions = pd.DataFrame(index=profile_series.index)
    
    # Age interactions
    interactions['age_x_career_matches'] = profile_series['age'] * match_series['career_matches']
    interactions['age_x_career_goals'] = profile_series['age'] * match_series['career_goals']
    
    # Seniority interactions
    interactions['seniority_x_goals_per_match'] = profile_series['seniority_days'] * match_series['goals_per_match']
    
    return interactions

def generate_daily_features_for_player_enhanced(
    player_id: int, 
    player_row: pd.Series, 
    player_matches: pd.DataFrame, 
    player_injuries: pd.DataFrame,
    player_career: Optional[pd.DataFrame] = None,
    global_end_date_cap: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """
    Enhanced daily features generation with fixed date range logic.
    
    Args:
        player_id: Player identifier
        player_row: Player profile data
        player_matches: Player's match history
        player_injuries: Player's injury history
        player_career: Player's career history (transfers) if available
        global_end_date_cap: Optional maximum date for the calendar (e.g. max raw match date)
        
    Returns:
        DataFrame with daily features for the player
        
    Raises:
        ValueError: If required data is missing or invalid
    """
    # Input validation
    if player_row is None or player_row.empty:
        raise ValueError(f"Player {player_id}: Missing player profile data")
    
    if not isinstance(player_matches, pd.DataFrame):
        raise ValueError(f"Player {player_id}: player_matches must be a DataFrame")
    
    if not isinstance(player_injuries, pd.DataFrame):
        raise ValueError(f"Player {player_id}: player_injuries must be a DataFrame")
    
    if not isinstance(player_career, pd.DataFrame):
        raise ValueError(f"Player {player_id}: player_career must be a DataFrame")
    
    # FIXED: Use player's actual first match date instead of hardcoded 2011-07-21
    if not player_matches.empty:
        start_date = player_matches['date'].min()
        print(f"   📅 Player {player_id} starts from: {start_date}")
    else:
        # Fallback to player's birth date if no matches
        start_date = player_row['date_of_birth']
        print(f"   📅 Player {player_id} has no matches, using birth date: {start_date}")
    
    # Determine end date - SMART approach that respects career termination but includes all injuries
    if not player_matches.empty:
        last_match_date = player_matches['date'].max()
        
        # Check if player has any injuries after their last match
        if not player_injuries.empty:
            latest_injury_date = player_injuries['fromDate'].max()
            
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
                print(f"   📅 Player {player_id} retired but had injuries after retirement")
                print(f"   📅 Calendar extends to: {end_date} (includes post-retirement injuries)")
            else:
                # Player retired and no injuries after retirement - respect career end
                season_end_year = last_match_date.year
                if last_match_date.month >= 7:  # If last match is in second half of year
                    season_end_year = last_match_date.year + 1
                end_date = pd.Timestamp(f'{season_end_year}-06-30')
                print(f"   📅 Player {player_id} calendar respects career end: {end_date}")
        else:
            # Player has no injuries - respect career end
            season_end_year = last_match_date.year
            if last_match_date.month >= 7:  # If last match is in second half of year
                season_end_year = last_match_date.year + 1
            end_date = pd.Timestamp(f'{season_end_year}-06-30')
            print(f"   📅 Player {player_id} calendar respects career end: {end_date}")
    else:
        # If no matches, check if player has injuries to determine end date
        if not player_injuries.empty:
            latest_injury_date = player_injuries['fromDate'].max()
            injury_end_year = latest_injury_date.year
            if latest_injury_date.month >= 7:  # If injury is in second half of year
                injury_end_year = latest_injury_date.year + 1
            end_date = pd.Timestamp(f'{injury_end_year}-06-30')
            print(f"   📅 Player {player_id} has no matches but has injuries, calendar extends to: {end_date}")
        else:
            # No matches and no injuries - use a reasonable default
            end_date = pd.Timestamp('2025-06-30')
            print(f"   📅 Player {player_id} has no matches or injuries, using default end date: {end_date}")
    
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    if global_end_date_cap is not None:
        cap_date = pd.Timestamp(global_end_date_cap)
        if end_date > cap_date:
            print(f"   ⏱️  Clamping calendar end from {end_date} to {cap_date} (global cap)")
            end_date = cap_date

    if start_date > end_date:
        print(f"   ⚠️  Start date {start_date} exceeds end date {end_date}. Adjusting to single-day calendar.")
        end_date = start_date

    calendar = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Build feature series
    profile_series = build_daily_profile_series_optimized(player_row, calendar, player_matches, player_career)
    match_series = build_daily_match_series_optimized(
        player_matches,
        calendar,
        player_row,
        profile_series=profile_series,
        player_career=player_career
    )
    injury_series = build_daily_injury_series_optimized(player_injuries, calendar)
    interaction_series = build_daily_interaction_features_optimized(profile_series, match_series, injury_series)
    
    # Combine all features
    daily_features = pd.concat([profile_series, match_series, injury_series, interaction_series], axis=1)
    
    # Remove duplicate columns
    daily_features = daily_features.loc[:, ~daily_features.columns.duplicated()]
    
    # Define our 98 selected features (51 original + 47 new - 1 removed)
    selected_features = [
        # Profile features (14)
        'age', 'seniority_days', 'position', 'nationality1', 'nationality2', 
        'height_cm', 'dominant_foot', 'previous_club', 'previous_club_country',
        'current_club', 'current_club_country', 'teams_today', 'cum_teams', 'seasons_count',
        
        # Windowed features (9)
        'matches_played', 'minutes_played_numeric', 'goals_numeric', 'assists_numeric',
        'yellow_cards_numeric', 'red_cards_numeric', 'matches_bench_unused', 
        'matches_not_selected', 'matches_injured',
        
        # Static match features (20)
        'cum_minutes_played_numeric', 'cum_goals_numeric', 'cum_assists_numeric',
        'cum_yellow_cards_numeric', 'cum_red_cards_numeric', 'cum_matches_played',
        'cum_matches_bench', 'cum_matches_not_selected', 'cum_competitions',
        'goals_per_match', 'assists_per_match', 'minutes_per_match', 'days_since_last_match',
        'matches', 'career_matches', 'career_goals', 'career_assists', 'career_minutes',
        'last_match_position', 'position_match_default',
        
        # Static injury features (5) - REMOVED total_career_injuries
        'cum_inj_starts', 'cum_inj_days', 'days_since_last_injury', 'days_since_last_injury_ended',
        'avg_injury_duration', 'injury_frequency',
        
        # Interaction features (3)
        'age_x_career_matches', 'age_x_career_goals', 'seniority_x_goals_per_match',
        
        # ENHANCED INJURY FEATURES (12)
        'avg_injury_severity', 'max_injury_severity',
        'lower_leg_injuries', 'knee_injuries', 'upper_leg_injuries',
        'hip_injuries', 'upper_body_injuries', 'head_injuries',
        'illness_count', 'other_injuries', 'physio_injury_ratio', 'cum_matches_injured',
        
        # ENHANCED COMPETITION & SEASON FEATURES (6)
        'competition_importance', 'avg_competition_importance', 'month',
        'disciplinary_action', 'cum_disciplinary_actions', 'teams_this_season',
        
        # ENHANCED SEASON & TEAM DIVERSITY FEATURES (3)
        'teams_last_season', 'teams_season_today', 'season_team_diversity',
        
        # ENHANCED NATIONAL TEAM FEATURES (8)
        'national_team_appearances', 'national_team_minutes', 'days_since_last_national_match',
        'national_team_this_season', 'national_team_last_season', 'national_team_frequency',
        'senior_national_team', 'national_team_intensity',
        
        # ENHANCED COMPETITION ANALYSIS FEATURES (8)
        'competition_intensity', 'competition_level', 'competition_diversity',
        'international_competitions', 'cup_competitions', 'competition_frequency',
        'competition_experience', 'competition_pressure',
        
        # ENHANCED CLUB PERFORMANCE FEATURES (11)
        'club_cum_goals', 'club_cum_assists', 'club_cum_minutes',
        'club_cum_matches_played', 'club_cum_yellow_cards', 'club_cum_red_cards',
        'club_goals_per_match', 'club_assists_per_match', 'club_minutes_per_match',
        'club_seniority_x_goals_per_match',
        
        # ENHANCED SUBSTITUTION FEATURES (10)
        'substitution_on_count', 'substitution_off_count', 'late_substitution_on_count',
        'early_substitution_off_count', 'impact_substitution_count', 'tactical_substitution_count',
        'substitution_minutes_played', 'substitution_efficiency', 'substitution_mood_indicator',
        'consecutive_substitutions',
        
        # TRANSFERMARKT PERFORMANCE (5)
        'transfermarkt_score_recent', 'transfermarkt_score_cum', 'transfermarkt_score_avg',
        'transfermarkt_score_rolling5', 'transfermarkt_score_matches',
        
        # MATCH LOCATION FEATURES (2)
        'home_matches', 'away_matches',
        
        # TEAM RESULT FEATURES (11)
        'team_win', 'team_draw', 'team_loss', 'team_points',
        'cum_team_wins', 'cum_team_draws', 'cum_team_losses',
        'team_win_rate', 'cum_team_points', 'team_points_rolling5', 'team_mood_score'
    ]
    
    # Filter to only include selected features
    available_features = [f for f in selected_features if f in daily_features.columns]
    missing_features = [f for f in selected_features if f not in daily_features.columns]
    
    if missing_features:
        print(f"Warning: Missing features for player {player_id}: {missing_features}")
    
    # Select only our chosen features
    daily_features = daily_features[available_features]
    
    # Add player_id and date columns
    daily_features.insert(0, 'player_id', player_id)
    daily_features.insert(1, 'date', daily_features.index)
    
    return daily_features

def check_disk_space(output_dir: str, estimated_size_mb: float) -> bool:
    """
    Check if there's sufficient disk space for output files.
    
    Args:
        output_dir: Directory where files will be saved
        estimated_size_mb: Estimated total size needed in MB
        
    Returns:
        True if sufficient space, False otherwise
    """
    try:
        # Get absolute path to check disk space
        abs_output_dir = os.path.abspath(output_dir)
        os.makedirs(abs_output_dir, exist_ok=True)
        
        # Get disk usage
        total, used, free = shutil.disk_usage(abs_output_dir)
        free_mb = free / (1024 * 1024)
        
        # Add 20% buffer for safety
        required_mb = estimated_size_mb * 1.2
        
        if free_mb < required_mb:
            print(f"❌ Insufficient disk space!")
            print(f"   Required: {required_mb:.1f} MB (with 20% buffer)")
            print(f"   Available: {free_mb:.1f} MB")
            print(f"   Shortage: {required_mb - free_mb:.1f} MB")
            return False
        else:
            print(f"✅ Disk space check passed")
            print(f"   Required: {required_mb:.1f} MB")
            print(f"   Available: {free_mb:.1f} MB")
            return True
    except Exception as e:
        print(f"⚠️  Could not check disk space: {str(e)}")
        return True  # Continue if check fails

def check_locked_files(output_dir: str) -> List[str]:
    """
    Check for locked/open CSV files in output directory.
    
    Args:
        output_dir: Directory to check
        
    Returns:
        List of locked file paths (empty if none)
    """
    locked_files = []
    if not os.path.exists(output_dir):
        return locked_files
    
    try:
        csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
        for csv_file in csv_files:
            file_path = os.path.join(output_dir, csv_file)
            try:
                # Try to open in append mode to check if locked
                with open(file_path, 'a'):
                    pass
            except (PermissionError, IOError):
                locked_files.append(csv_file)
    except Exception as e:
        print(f"⚠️  Could not check for locked files: {str(e)}")
    
    return locked_files

def verify_input_files() -> Tuple[bool, List[str]]:
    """
    Verify all required input files exist.
    
    Returns:
        Tuple of (all_exist: bool, missing_files: List[str])
    """
    data_dir = CONFIG['DATA_DIR']
    required_files = [
        '*players_profile*.xlsx',
        '*injuries_data*.xlsx',
        '*match_data*.xlsx',
        '*teams_data*.xlsx',
        '*competition_data*.xlsx',
        '*players_career*.xlsx'
    ]
    
    missing_files = []
    
    import glob
    for pattern in required_files:
        matches = glob.glob(os.path.join(data_dir, pattern))
        if not matches:
            missing_files.append(pattern)
    
    all_exist = len(missing_files) == 0
    return all_exist, missing_files

def run_preflight_checks(output_dir: str, num_players: int) -> bool:
    """
    Run all pre-flight checks before starting generation.
    
    Args:
        output_dir: Output directory path
        num_players: Number of players to process
        
    Returns:
        True if all checks pass, False otherwise
    """
    print("\n" + "=" * 70)
    print("🔍 PRE-FLIGHT CHECKS")
    print("=" * 70)
    
    all_passed = True
    
    # Check 1: Verify input files
    print("\n1️⃣  Checking input files...")
    all_exist, missing = verify_input_files()
    if not all_exist:
        print(f"❌ Missing input files:")
        for file in missing:
            print(f"   - {file}")
        all_passed = False
    else:
        print("✅ All input files found")
    
    # Check 2: Check for locked files
    print("\n2️⃣  Checking for locked files...")
    locked = check_locked_files(output_dir)
    if locked:
        print(f"❌ Found {len(locked)} locked/open file(s):")
        for file in locked[:10]:  # Show first 10
            print(f"   - {file}")
        if len(locked) > 10:
            print(f"   ... and {len(locked) - 10} more")
        print("   Please close these files and try again.")
        all_passed = False
    else:
        print("✅ No locked files detected")
    
    # Check 3: Check disk space
    print("\n3️⃣  Checking disk space...")
    # Estimate: average 5-10 MB per player file (conservative estimate)
    estimated_mb_per_player = 8.0
    estimated_total_mb = num_players * estimated_mb_per_player
    if not check_disk_space(output_dir, estimated_total_mb):
        all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ All pre-flight checks passed! Ready to proceed.")
    else:
        print("❌ Pre-flight checks failed. Please resolve issues above.")
    print("=" * 70 + "\n")
    
    return all_passed

def generate_features_for_all_players(
    output_dir: str = 'daily_features_output',
    max_players: Optional[int] = None,
    random_seed: Optional[int] = None
):
    """
    Generate daily features for players in the dataset.
    
    Args:
        output_dir (str): Directory to save the output files
        max_players (Optional[int]): Maximum number of players to process. 
                                     If None, processes all players. If specified, 
                                     randomly selects that many players.
        random_seed (Optional[int]): Random seed for reproducible player selection.
                                     Defaults to 42 if max_players is specified.
    """
    print("🚀 ENHANCED GOLD STANDARD DAILY FEATURES GENERATOR")
    print("=" * 70)
    print("📋 Features: 108 total (51 original + 57 new - 1 removed)")
    print("🔧 Production-ready with no hardcoded values or synthetic data")
    
    start_time = datetime.now()
    
    # Load data
    data = load_data_with_cache()
    players, injuries, matches = data['players'], data['injuries'], data['matches']
    teams = data.get('teams')
    competitions = data.get('competitions')
    initialize_team_country_map(teams)
    initialize_competition_type_map(competitions)
    career = preprocess_career_data(data.get('career'))
    
    # Preprocess data
    players, injuries, matches = preprocess_data_optimized(players, injuries, matches)
    
    # Get all non-goalkeeper player IDs
    all_player_ids = players['id'].tolist()
    
    # Random selection logic if max_players is specified
    if max_players is not None:
        import random
        if random_seed is not None:
            random.seed(random_seed)
        else:
            random.seed(42)  # Default seed for reproducibility
        
        if max_players > len(all_player_ids):
            print(f"⚠️  Warning: max_players ({max_players}) exceeds available players ({len(all_player_ids)})")
            print(f"   Processing all {len(all_player_ids)} players instead")
            selected_player_ids = all_player_ids
        else:
            selected_player_ids = random.sample(all_player_ids, max_players)
            print(f"🧪 TEST MODE: Randomly selected {max_players} players from {len(all_player_ids)} available")
            print(f"📋 Selected player IDs: {sorted(selected_player_ids)}")
    else:
        selected_player_ids = all_player_ids
        print(f"🎯 FULL MODE: Processing all {len(all_player_ids)} players")
    
    print(f"📁 Output will be saved to '{output_dir}/' folder")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Run pre-flight checks
    if not run_preflight_checks(output_dir, len(selected_player_ids)):
        print("❌ Pre-flight checks failed. Please resolve issues and try again.")
        return
    
    # Process each player with progress bar and enhanced monitoring
    successful_players = 0
    failed_players = 0
    player_times = []
    files_generated = 0
    
    # Check which players have already been processed
    existing_files = set()
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if file.startswith('player_') and file.endswith('_daily_features.csv'):
                try:
                    player_id_from_file = int(file.replace('player_', '').replace('_daily_features.csv', ''))
                    existing_files.add(player_id_from_file)
                except ValueError:
                    pass
    
    # Filter out already processed players
    players_to_process = [pid for pid in selected_player_ids if pid not in existing_files]
    skipped_count = len(selected_player_ids) - len(players_to_process)
    
    if skipped_count > 0:
        print(f"⏭️  Skipping {skipped_count} already processed player(s)")
        print(f"🔄 Processing {len(players_to_process)} remaining player(s)")
    
    if len(players_to_process) == 0:
        print("✅ All players have already been processed!")
        return
    
    # Initialize progress bar
    pbar = tqdm(
        players_to_process,
        desc="Processing players",
        unit="player",
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )
    
    for player_id in pbar:
        player_start_time = datetime.now()
        
        # Update progress bar description with current player
        pbar.set_description(f"Processing player {player_id}")
        
        try:
            # Get player data
            player_filter = players[players['id'] == player_id]
            if player_filter.empty:
                pbar.write(f"❌ Player {player_id} not found in players data")
                failed_players += 1
                continue
                
            player_row = player_filter.iloc[0]
            player_matches = matches[matches['player_id'] == player_id].copy()
            player_injuries = injuries[injuries['player_id'] == player_id].copy()
            player_career = None
            if career is not None:
                id_col = 'player_id' if 'player_id' in career.columns else ('id' if 'id' in career.columns else None)
                if id_col is not None:
                    player_career = career[career[id_col] == player_id].copy()
            
            # Determine global end date cap (no calendars beyond last real match or today)
            global_end_date_cap = None
            try:
                today_cap = pd.Timestamp.today().normalize()
            except Exception:
                today_cap = pd.Timestamp('today').normalize()

            if matches is not None and not matches.empty:
                max_match_date = matches['date'].max()
                if pd.notna(max_match_date):
                    global_end_date_cap = min(max_match_date, today_cap)
                else:
                    global_end_date_cap = today_cap
            else:
                global_end_date_cap = today_cap

            print(f"🗓️  Global calendar cap set to: {global_end_date_cap}")
            
            # Generate daily features
            daily_features = generate_daily_features_for_player_enhanced(
                player_id,
                player_row,
                player_matches,
                player_injuries,
                player_career,
                global_end_date_cap=global_end_date_cap
            )
            
            # Save to output folder
            output_file = f'{output_dir}/player_{player_id}_daily_features.csv'
            daily_features.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            # Calculate timing
            player_time = (datetime.now() - player_start_time).total_seconds()
            player_times.append(player_time)
            files_generated += 1
            
            # Calculate statistics for progress bar
            elapsed_time = (datetime.now() - start_time).total_seconds()
            avg_time_per_player = sum(player_times) / len(player_times) if player_times else 0
            remaining_players = len(selected_player_ids) - (successful_players + failed_players)
            eta_seconds = remaining_players * avg_time_per_player if avg_time_per_player > 0 else 0
            eta_str = str(timedelta(seconds=int(eta_seconds))) if eta_seconds > 0 else "calculating..."
            
            # Enhanced console output
            pbar.write(f"✅ Player {player_id}: {daily_features.shape[0]} days, {daily_features.shape[1]} features")
            pbar.write(f"   📊 Matches: {len(player_matches)}, Injuries: {len(player_injuries)}")
            pbar.write(f"   ⏱️  Time: {player_time:.1f}s | Avg: {avg_time_per_player:.1f}s | ETA: {eta_str}")
            pbar.write(f"   📁 Files generated: {files_generated}/{len(selected_player_ids)}")
            
            successful_players += 1
            
            # Update progress bar postfix
            percentage = (successful_players + failed_players) / len(selected_player_ids) * 100
            pbar.set_postfix({
                'Success': successful_players,
                'Failed': failed_players,
                'ETA': eta_str[:8] if eta_seconds > 0 else 'calc...'
            })
            
        except Exception as e:
            player_time = (datetime.now() - player_start_time).total_seconds()
            player_times.append(player_time)
            pbar.write(f"❌ Error processing player {player_id}: {str(e)}")
            failed_players += 1
            pbar.set_postfix({
                'Success': successful_players,
                'Failed': failed_players
            })
    
    pbar.close()
    
    # Final summary
    total_time = datetime.now() - start_time
    avg_time = sum(player_times) / len(player_times) if player_times else 0
    
    print("\n" + "=" * 70)
    print("📊 GENERATION SUMMARY")
    print("=" * 70)
    print(f"⏱️  Total processing time: {total_time}")
    print(f"📈 Average time per player: {avg_time:.2f} seconds")
    if skipped_count > 0:
        print(f"⏭️  Skipped (already processed): {skipped_count} players")
    print(f"✅ Successfully processed: {successful_players} players")
    print(f"❌ Failed to process: {failed_players} players")
    print(f"📁 Files generated: {files_generated}")
    if successful_players > 0:
        total_processed = successful_players + failed_players
        print(f"📊 Success rate: {successful_players / total_processed * 100:.1f}%")
    total_completed = skipped_count + successful_players
    print(f"📋 Total completed: {total_completed}/{len(selected_player_ids)} players")
    print(f"🎉 Feature generation completed! Check the '{output_dir}/' folder for results.")
    print("=" * 70)

def main():
    """Main function - can be used for testing or production"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate daily features for players',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test mode: Process 10 randomly selected players
  python create_daily_features_v3.py --test
  
  # Custom number of players
  python create_daily_features_v3.py --max-players 20
  
  # Full mode: Process all players
  python create_daily_features_v3.py
  
  # Test mode with custom output directory
  python create_daily_features_v3.py --test --output-dir test_output
  
  # Custom seed for reproducibility
  python create_daily_features_v3.py --test --seed 123
        """
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: Process 10 randomly selected players'
    )
    
    parser.add_argument(
        '--max-players',
        type=int,
        default=None,
        help='Maximum number of players to process (randomly selected)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='daily_features_output',
        help='Output directory for daily features files (default: daily_features_output)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for player selection (default: 42 when using --test or --max-players)'
    )
    
    args = parser.parse_args()
    
    # Determine max_players based on arguments
    if args.test:
        max_players = 10
    elif args.max_players is not None:
        max_players = args.max_players
    else:
        max_players = None
    
    # Call the function with appropriate parameters
    generate_features_for_all_players(
        output_dir=args.output_dir,
        max_players=max_players,
        random_seed=args.seed
    )

def clean_club_label(value: Optional[str]) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    value_str = str(value).strip()
    if not value_str or value_str.lower() in {'nan', 'none', 'no club', 'free agent'}:
        return None
    return re.sub(r"\s+", " ", value_str)

def initialize_team_country_map(teams_df: Optional[pd.DataFrame]):
    global TEAM_COUNTRY_MAP
    TEAM_COUNTRY_MAP = {}
    if teams_df is None or teams_df.empty:
        logger.warning("⚠️  Teams dataset is empty; club country lookup will rely on heuristics.")
        return
    team_col_candidates = ['team', 'team_name', 'name']
    country_col_candidates = ['country', 'team_country']
    team_col = next((col for col in team_col_candidates if col in teams_df.columns), None)
    country_col = next((col for col in country_col_candidates if col in teams_df.columns), None)
    if team_col is None or country_col is None:
        logger.warning("⚠️  Teams dataset missing expected columns; club country lookup will rely on heuristics.")
        return
    for _, row in teams_df.iterrows():
        club = clean_club_label(row[team_col])
        country = row[country_col]
        if not club or pd.isna(country):
            continue
        country_str = str(country).strip()
        for key in clean_club_variants(club):
            TEAM_COUNTRY_MAP[key] = country_str

def normalize_competition_key(value: Optional[str]) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    value_str = str(value).strip()
    if not value_str or value_str.lower() in {'nan', 'none'}:
        return None
    return re.sub(r"\s+", " ", value_str.lower())

def initialize_competition_type_map(competitions_df: Optional[pd.DataFrame]):
    global COMPETITION_TYPE_MAP
    COMPETITION_TYPE_MAP = {}
    if competitions_df is None or competitions_df.empty:
        logger.warning("⚠️  Competitions dataset is empty; competition intensity mapping will use heuristics only.")
        set_competition_type_map({})
        return
    type_col = None
    for candidate in ['type', 'Type', 'TYPE']:
        if candidate in competitions_df.columns:
            type_col = candidate
            break
    if type_col is None or 'competition' not in competitions_df.columns:
        logger.warning("⚠️  Competitions dataset missing expected columns; competition intensity mapping will use heuristics only.")
        set_competition_type_map({})
        return
    for _, row in competitions_df.iterrows():
        comp = normalize_competition_key(row['competition'])
        comp_type = row[type_col]
        if comp and pd.notna(comp_type):
            COMPETITION_TYPE_MAP[comp] = str(comp_type)
    set_competition_type_map(COMPETITION_TYPE_MAP)

def clean_club_variants(club_name: str) -> List[str]:
    variants = []
    normalized = normalize_club_key(club_name)
    if not normalized:
        return variants
    variants.append(normalized)
    # Remove content in parentheses
    without_parentheses = re.sub(r"\s*\(.*?\)", "", normalized).strip()
    if without_parentheses and without_parentheses not in variants:
        variants.append(without_parentheses)
    # Remove dots and extra punctuation for fallback
    simple = re.sub(r"[^a-z0-9 ]", "", without_parentheses or normalized).strip()
    if simple and simple not in variants:
        variants.append(simple)
    return variants

def normalize_club_key(value: Optional[str]) -> Optional[str]:
    cleaned = clean_club_label(value)
    if cleaned is None:
        return None
    return cleaned.lower()

def clubs_match(team_name: Optional[str], reference_name: Optional[str]) -> bool:
    if not team_name or not reference_name:
        return False
    variants_team = set(clean_club_variants(team_name))
    variants_ref = set(clean_club_variants(reference_name))
    return bool(variants_team & variants_ref)

def parse_match_result(result_str: Optional[str]) -> Tuple[Optional[int], Optional[int], bool, bool]:
    if result_str is None or (isinstance(result_str, float) and np.isnan(result_str)):
        return None, None, False, False
    text = str(result_str).strip().lower()
    if not text:
        return None, None, False, False
    numbers = re.findall(r"\d+", text)
    if len(numbers) < 2:
        return None, None, False, False
    home_goals = int(numbers[0])
    away_goals = int(numbers[1])
    decided_by_penalties = 'p.g' in text or 'pens' in text
    decided_after_extra = 'a.p' in text or 'aet' in text or 'et' in text
    return home_goals, away_goals, decided_by_penalties, decided_after_extra

def determine_player_team_for_match(
    match_row: pd.Series,
    current_club: Optional[str],
    player_row: pd.Series,
    career_clubs: set
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    home_team = clean_club_label(match_row.get('home_team'))
    away_team = clean_club_label(match_row.get('away_team'))

    current_club_clean = clean_club_label(current_club)
    if current_club_clean:
        if clubs_match(home_team, current_club_clean):
            return 'home', home_team, away_team
        if clubs_match(away_team, current_club_clean):
            return 'away', away_team, home_team

    for club in career_clubs:
        if clubs_match(home_team, club):
            return 'home', home_team, away_team
        if clubs_match(away_team, club):
            return 'away', away_team, home_team

    nationalities = [player_row.get('nationality1'), player_row.get('nationality2')]

    def team_matches_player_nationality(team: Optional[str]) -> bool:
        if not team:
            return False
        if not is_national_team_fast(team):
            return False
        team_lower = team.lower()
        for nat in nationalities:
            if nat and str(nat).lower() in team_lower:
                return True
        return False

    if team_matches_player_nationality(home_team):
        return 'home', home_team, away_team
    if team_matches_player_nationality(away_team):
        return 'away', away_team, home_team

    if home_team:
        return 'home', home_team, away_team
    if away_team:
        return 'away', away_team, home_team

    return None, None, None

def compute_team_result_metrics(
    matches: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    player_row: pd.Series,
    profile_series: Optional[pd.DataFrame],
    player_career: Optional[pd.DataFrame]
) -> Dict[str, List[float]]:
    n_days = len(calendar)
    zero_float = [0.0] * n_days
    zero_int = [0] * n_days

    if matches is None or matches.empty:
        return {
            'team_win': zero_int.copy(),
            'team_draw': zero_int.copy(),
            'team_loss': zero_int.copy(),
            'team_points': zero_int.copy(),
            'cum_team_wins': zero_int.copy(),
            'cum_team_draws': zero_int.copy(),
            'cum_team_losses': zero_int.copy(),
            'team_win_rate': zero_float.copy(),
            'cum_team_points': zero_int.copy(),
            'team_points_rolling5': zero_float.copy(),
            'team_mood_score': zero_float.copy(),
            'home_matches': zero_int.copy(),
            'away_matches': zero_int.copy()
        }

    matches_ext = matches.copy()
    matches_ext['date'] = pd.to_datetime(matches_ext['date'])
    matches_ext = matches_ext.dropna(subset=['date'])
    matches_ext.sort_values('date', inplace=True)

    current_club_series = None
    if profile_series is not None and 'current_club' in profile_series.columns:
        current_club_series = profile_series['current_club']

    career_clubs = set()
    if player_career is not None and not player_career.empty:
        for col in ['from_club', 'to_club', 'From', 'To']:
            if col in player_career.columns:
                for value in player_career[col].dropna():
                    label = clean_club_label(value)
                    if label:
                        career_clubs.add(label)

    win_map = defaultdict(int)
    draw_map = defaultdict(int)
    loss_map = defaultdict(int)
    points_map = defaultdict(list)
    home_match_map = defaultdict(int)
    away_match_map = defaultdict(int)

    for _, row in matches_ext.iterrows():
        match_date = row['date'].normalize()
        current_club_today = None
        if current_club_series is not None and match_date in current_club_series.index:
            current_club_today = current_club_series.loc[match_date]

        side, player_team_name, opponent_team_name = determine_player_team_for_match(
            row, current_club_today, player_row, career_clubs
        )

        home_goals, away_goals, _, _ = parse_match_result(row.get('result'))
        result = None
        points = 0

        if side and home_goals is not None and away_goals is not None:
            if side == 'home':
                team_goals, opp_goals = home_goals, away_goals
            else:
                team_goals, opp_goals = away_goals, home_goals

            if team_goals > opp_goals:
                result = 'win'
                points = 3
            elif team_goals == opp_goals:
                result = 'draw'
                points = 1
            else:
                result = 'loss'
                points = 0

        if result is None:
            continue

        if result == 'win':
            win_map[match_date] += 1
        elif result == 'draw':
            draw_map[match_date] += 1
        elif result == 'loss':
            loss_map[match_date] += 1

        points_map[match_date].append(points)

        if side == 'home':
            home_match_map[match_date] += 1
        elif side == 'away':
            away_match_map[match_date] += 1

    team_win = [0] * n_days
    team_draw = [0] * n_days
    team_loss = [0] * n_days
    team_points = [0] * n_days
    home_matches = [0] * n_days
    away_matches = [0] * n_days
    cum_team_wins = [0] * n_days
    cum_team_draws = [0] * n_days
    cum_team_losses = [0] * n_days
    team_win_rate = [0.0] * n_days
    cum_team_points = [0] * n_days
    team_points_rolling5 = [0.0] * n_days
    team_mood_score = [0.0] * n_days

    recent_points = deque(maxlen=5)
    total_matches = 0
    wins_so_far = 0
    draws_so_far = 0
    losses_so_far = 0
    points_so_far = 0

    calendar_list = list(calendar)

    for i, date in enumerate(calendar_list):
        wins_today = win_map.get(date, 0)
        draws_today = draw_map.get(date, 0)
        losses_today = loss_map.get(date, 0)
        points_today_list = points_map.get(date, [])

        team_win[i] = wins_today
        team_draw[i] = draws_today
        team_loss[i] = losses_today
        team_points[i] = sum(points_today_list)
        home_matches[i] = home_match_map.get(date, 0)
        away_matches[i] = away_match_map.get(date, 0)

        if wins_today or draws_today or losses_today:
            for pts in points_today_list:
                recent_points.append(pts)
                points_so_far += pts
            wins_so_far += wins_today
            draws_so_far += draws_today
            losses_so_far += losses_today
            total_matches += wins_today + draws_today + losses_today

        cum_team_wins[i] = wins_so_far
        cum_team_draws[i] = draws_so_far
        cum_team_losses[i] = losses_so_far
        cum_team_points[i] = points_so_far

        if total_matches > 0:
            team_win_rate[i] = wins_so_far / total_matches
        elif i > 0:
            team_win_rate[i] = team_win_rate[i - 1]
        else:
            team_win_rate[i] = 0.0

        rolling_points = sum(recent_points)
        team_points_rolling5[i] = rolling_points
        if recent_points:
            team_mood_score[i] = rolling_points / (len(recent_points) * 3.0)
        else:
            team_mood_score[i] = 0.0

        if total_matches == 0 and i > 0:
            team_win_rate[i] = team_win_rate[i - 1]
            cum_team_wins[i] = cum_team_wins[i - 1]
            cum_team_draws[i] = cum_team_draws[i - 1]
            cum_team_losses[i] = cum_team_losses[i - 1]
            cum_team_points[i] = cum_team_points[i - 1]
            team_points_rolling5[i] = team_points_rolling5[i - 1]
            team_mood_score[i] = team_mood_score[i - 1]

    return {
        'team_win': team_win,
        'team_draw': team_draw,
        'team_loss': team_loss,
        'team_points': team_points,
        'cum_team_wins': cum_team_wins,
        'cum_team_draws': cum_team_draws,
        'cum_team_losses': cum_team_losses,
        'team_win_rate': team_win_rate,
        'cum_team_points': cum_team_points,
        'team_points_rolling5': team_points_rolling5,
        'team_mood_score': team_mood_score,
        'home_matches': home_matches,
        'away_matches': away_matches
    }

if __name__ == "__main__":
    main()
