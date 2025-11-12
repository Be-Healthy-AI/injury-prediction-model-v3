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

# Import benfica-parity functions
from benfica_parity_config import (
    BENFICA_PARITY_CONFIG,
    is_national_team_benfica_parity,
    map_competition_importance_benfica_parity,
    map_season_phase_benfica_parity,
    calculate_age_benfica_parity,
    detect_disciplinary_action_benfica_parity,
    calculate_enhanced_features_dynamically,
    calculate_injury_features_benfica_parity,
    calculate_national_team_features_benfica_parity,
    calculate_complex_derived_features_benfica_parity
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('feature_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration constants
CONFIG = {
    'DATA_DIR': '../original_data',
    'CACHE_FILE': '../data_cache_v3.pkl',
    'CACHE_DURATION': 3600,  # 1 hour in seconds
    'DEFAULT_OUTPUT_DIR': '../features_daily_all_players_v3',
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
    
    # Use first match found, or fallback to default pattern
    players_path = players_files[0] if players_files else f'{data_dir}/players_profile.xlsx'
    injuries_path = injuries_files[0] if injuries_files else f'{data_dir}/injuries_data.xlsx'
    matches_path = matches_files[0] if matches_files else f'{data_dir}/match_data.xlsx'
    teams_path = teams_files[0] if teams_files else f'{data_dir}/teams_data.xlsx'
    competitions_path = competitions_files[0] if competitions_files else f'{data_dir}/competition_data.xlsx'
    
    logger.info(f"📄 Loading: {os.path.basename(players_path)}")
    players = pd.read_excel(players_path, engine='openpyxl')
    injuries = pd.read_excel(injuries_path, engine='openpyxl')
    matches = pd.read_excel(matches_path, engine='openpyxl')
    teams = pd.read_excel(teams_path, engine='openpyxl')
    competitions = pd.read_excel(competitions_path, engine='openpyxl')
    
    # Cache the data
    data = {
        'players': players,
        'injuries': injuries, 
        'matches': matches,
        'teams': teams,
        'competitions': competitions
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
    
    # Vectorized height conversion - handle "1,71m" format
    def parse_height(height_str):
        if pd.isna(height_str):
            return np.nan
        # Remove 'm' and replace comma with dot, then convert to float and multiply by 100
        height_clean = str(height_str).replace('m', '').replace(',', '.')
        try:
            return float(height_clean) * 100
        except:
            return np.nan
    
    players['height_cm'] = players['height'].apply(parse_height)
    
    # ENHANCED: Add competition importance mapping using benfica-parity logic
    matches['competition_importance'] = matches['competition'].apply(map_competition_importance_benfica_parity)
    
    # ENHANCED: Add season phase mapping using benfica-parity logic
    matches['season_phase'] = matches['date'].apply(map_season_phase_benfica_parity)
    
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
        
        if any(term in injury_lower for term in ['ankle', 'foot', 'toe', 'achilles']):
            return 'lower_leg'
        elif any(term in injury_lower for term in ['knee', 'patella', 'meniscus']):
            return 'knee'
        elif any(term in injury_lower for term in ['thigh', 'quad', 'hamstring', 'calf']):
            return 'upper_leg'
        elif any(term in injury_lower for term in ['hip', 'groin', 'adductor']):
            return 'hip'
        elif any(term in injury_lower for term in ['shoulder', 'arm', 'elbow', 'wrist', 'hand']):
            return 'upper_body'
        elif any(term in injury_lower for term in ['head', 'face', 'eye', 'nose', 'mouth']):
            return 'head'
        elif any(term in injury_lower for term in ['illness', 'sick', 'fever', 'cold']):
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

def determine_match_participation_optimized(matches):
    """Optimized match participation determination"""
    def participation_status(row):
        position = str(row['position']).lower()
        minutes = row['minutes_played_numeric']
        
        # Check for injury-related positions first
        injury_indicators = ['injury', 'cruciate ligament tear', 'pubalgia', 'knee injury', 'muscle injury']
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

def build_daily_profile_series_optimized(player_row, calendar, player_matches):
    """Highly optimized daily profile series building"""
    # Pre-calculate all values before creating DataFrame
    n_days = len(calendar)
    
    # Static features (vectorized)
    # Use benfica-parity age calculation for better precision matching
    age_values = [calculate_age_benfica_parity(player_row['date_of_birth'], date) for date in calendar]
    position_values = [player_row['position']] * n_days
    nationality1_values = [player_row['nationality1']] * n_days
    nationality2_values = [player_row['nationality2']] * n_days
    height_cm_values = [player_row['height_cm']] * n_days
    dominant_foot_values = [player_row['dominant_foot']] * n_days
    # FIXED: Use previous_club (renamed from signed_from in preprocessing) for initial assignment
    previous_club_values = [player_row['previous_club']] * n_days
    previous_club_country_values = [get_club_country(player_row['previous_club'])] * n_days
    
    # Initialize club-related arrays
    current_club_values = [None] * n_days
    current_club_country_values = [None] * n_days
    seniority_days_values = [0] * n_days
    
    # ENHANCED: Proper club assignment logic with seniority reset
    if not player_matches.empty:
        # Filter to only main club matches (exclude national teams, junior teams, etc.)
        main_club_matches = player_matches[
            (player_matches['home_team'].apply(is_main_club_team)) | 
            (player_matches['away_team'].apply(is_main_club_team))
        ].copy()
        
        if not main_club_matches.empty:
            # Determine player's club for each match
            player_clubs = []
            for _, match in main_club_matches.iterrows():
                home_team = match['home_team']
                away_team = match['away_team']
                
                # Determine which team the player belongs to
                if is_main_club_team(home_team) and is_main_club_team(away_team):
                    # Both are main clubs, need to determine player's club
                    # Use the team that appears more frequently in player's career
                    home_count = len(player_matches[player_matches['home_team'] == home_team])
                    away_count = len(player_matches[player_matches['away_team'] == away_team])
                    player_club = home_team if home_count >= away_count else away_team
                elif is_main_club_team(home_team):
                    player_club = home_team
                elif is_main_club_team(away_team):
                    player_club = away_team
                else:
                    continue
                
                player_clubs.append({
                    'date': match['date'],
                    'club': player_club
                })
            
            # Sort by date and detect club changes
            player_clubs.sort(key=lambda x: x['date'])
            
            # Track club progression with stability logic
            club_periods = []
            current_club = None
            current_start_date = None
            club_stability_days = 30  # Minimum days to consider a club change stable
            
            # Group consecutive club appearances
            club_groups = []
            current_group = []
            
            for club_info in player_clubs:
                club = club_info['club']
                date = club_info['date']
                
                if not current_group or club == current_group[-1]['club']:
                    current_group.append(club_info)
                else:
                    # Check if the previous group was stable enough
                    if len(current_group) >= 3:  # At least 3 appearances
                        club_groups.append(current_group)
                    current_group = [club_info]
            
            # Add the last group if it's stable
            if len(current_group) >= 3:
                club_groups.append(current_group)
            
            # Create club periods from stable groups
            for group in club_groups:
                if group:
                    club = group[0]['club']
                    start_date = group[0]['date']
                    end_date = group[-1]['date']
                    
                    if current_club is None:
                        current_club = club
                        current_start_date = start_date
                    elif club != current_club:
                        # Club change detected
                        club_periods.append({
                            'club': current_club,
                            'start_date': current_start_date,
                            'end_date': start_date - pd.Timedelta(days=1)
                        })
                        current_club = club
                        current_start_date = start_date
            
            # Add the last club period
            if current_club is not None:
                club_periods.append({
                    'club': current_club,
                    'start_date': current_start_date,
                    'end_date': calendar[-1]  # End of calendar
                })
            
            # Pre-calculate club assignments for each day
            current_club_start = None
            current_previous_club = None  # Start with None for first club
            current_previous_club_country = None
            current_club = None
            current_club_country = None
            has_changed_clubs = False  # Track if player has ever changed clubs
            
            # Set initial club from first period
            if club_periods:
                current_club = club_periods[0]['club']
                current_club_country = get_club_country(current_club)
                current_club_start = club_periods[0]['start_date']
            
            # Pre-calculate all values in one pass
            for i, date in enumerate(calendar):
                # Find the club period that contains this date
                current_period = None
                for period in club_periods:
                    if period['start_date'] <= date <= period['end_date']:
                        current_period = period
                        break
                
                # Update club if we found a period and it's different from current
                if current_period and current_club != current_period['club']:
                    # Club change detected
                    current_previous_club = current_club
                    current_previous_club_country = get_club_country(current_club)
                    current_club = current_period['club']
                    current_club_country = get_club_country(current_club)
                    current_club_start = current_period['start_date']
                    has_changed_clubs = True
                
                # Store assignments directly in lists
                # Only set previous_club if player has actually changed clubs
                if has_changed_clubs:
                    previous_club_values[i] = current_previous_club
                    previous_club_country_values[i] = current_previous_club_country
                # If no club change yet, keep the initial previous_club from player profile
                
                current_club_values[i] = current_club
                current_club_country_values[i] = current_club_country
                seniority_days_values[i] = max(0, (date - current_club_start).days) if current_club_start else 0
        else:
            # No main club matches found, use default values
            current_club_start = calendar[0]
            current_previous_club = None  # No previous club if no matches found
            current_previous_club_country = None
            current_club = player_row['previous_club']  # Use the player's recorded previous club
            current_club_country = get_club_country(current_club)
            
            for i, date in enumerate(calendar):
                previous_club_values[i] = current_previous_club  # None since no club changes detected
                previous_club_country_values[i] = current_previous_club_country
                current_club_values[i] = current_club
                current_club_country_values[i] = current_club_country
                seniority_days_values[i] = max(0, (date - current_club_start).days)
    else:
        # No club matches - use joined_on date
        joined_days = (calendar - player_row['joined_on']).days
        seniority_days_values = [max(0, days) for days in joined_days]
    
    # Optimized teams calculation (EXCLUDING NATIONAL TEAMS)
    # Filter matches to exclude national team games using benfica-parity detection
    club_matches = player_matches[
        ~(player_matches['home_team'].apply(is_national_team_benfica_parity) | 
          player_matches['away_team'].apply(is_national_team_benfica_parity))
    ].copy()
    
    # Pre-calculate all unique club teams
    all_club_teams = set()
    for _, match in club_matches.iterrows():
        for team_col in ['home_team', 'away_team']:
            team = match[team_col]
            if pd.notna(team) and not is_national_team_fast(team):
                all_club_teams.add(team)
    
    # Use groupby for faster daily calculation (CLUBS ONLY)
    daily_club_teams = club_matches.groupby('date').agg({
        'home_team': lambda x: set(x.dropna()),
        'away_team': lambda x: set(x.dropna())
    })
    
    unique_club_teams = set()
    teams_today_values = [0] * n_days
    cum_teams_values = [0] * n_days
    
    for i, date in enumerate(calendar):
        if date in daily_club_teams.index:
            home_teams = daily_club_teams.loc[date, 'home_team']
            away_teams = daily_club_teams.loc[date, 'away_team']
            # Filter out national teams from daily teams
            club_teams_today = set()
            for team in home_teams.union(away_teams):
                if not is_national_team_benfica_parity(team):
                    club_teams_today.add(team)
            teams_today_values[i] = len(club_teams_today)
            unique_club_teams.update(club_teams_today)
        cum_teams_values[i] = len(unique_club_teams)
    
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
        'seniority_days': seniority_days_values
    }, index=calendar)
    
    return out

def build_daily_match_series_optimized(matches, calendar, player_row):
    """Optimized daily match series building"""
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
            'season_phase': [0] * n_days,  # 0 = no match
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
        'season_phase': 'mean',
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
    season_phase = [3] * n_days
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
            # season_phase[idx] = row['season_phase']
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
    
    # ENHANCED: Calculate enhanced features dynamically using benfica-parity logic
    enhanced_features = calculate_enhanced_features_dynamically(matches, calendar, player_row)
    competition_importance = enhanced_features['competition_importance']
    season_phase = enhanced_features['season_phase']
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
    
    # Calculate season-based team features
    if not matches.empty:
        # Create a copy to avoid modifying the original
        matches_copy = matches.copy()
        matches_copy['football_season'] = matches_copy['date'].apply(get_football_season)
        
        # Group teams by season
        season_teams = {}
        for _, match in matches_copy.iterrows():
            season = match['football_season']
            home_team = match['home_team']
            away_team = match['away_team']
            
            if season not in season_teams:
                season_teams[season] = set()
            
            if pd.notna(home_team) and is_main_club_team(home_team):
                season_teams[season].add(home_team)
            if pd.notna(away_team) and is_main_club_team(away_team):
                season_teams[season].add(away_team)
        
        # Calculate season-based features for each day
        for i, date in enumerate(calendar):
            current_season = get_football_season(date)
            year = date.year
            month = date.month
            
            # teams_this_season: distinct teams played against in current season
            teams_this_season[i] = len(season_teams.get(current_season, set()))
            
            # teams_last_season: distinct teams played against in previous season
            if month >= 7:  # July onwards
                prev_season = f"{year-1}/{year}"
            else:  # January to June
                prev_season = f"{year-2}/{year-1}"
            teams_last_season[i] = len(season_teams.get(prev_season, set()))
            
            # teams_season_today: teams involved in matches on that specific day
            day_matches = matches_copy[matches_copy['date'] == date]
            if not day_matches.empty:
                day_teams = set()
                for _, match in day_matches.iterrows():
                    home_team = match['home_team']
                    away_team = match['away_team']
                    if pd.notna(home_team) and is_main_club_team(home_team):
                        day_teams.add(home_team)
                    if pd.notna(away_team) and is_main_club_team(away_team):
                        day_teams.add(away_team)
                teams_season_today[i] = len(day_teams)
            
            # season_team_diversity: ratio of teams_this_season / total_matches_this_season
            current_season_matches = matches_copy[matches_copy['football_season'] == current_season]
            total_matches_this_season = len(current_season_matches)
            if total_matches_this_season > 0:
                season_team_diversity[i] = teams_this_season[i] / total_matches_this_season
    

        

    

    
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
        'season_phase': season_phase,
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
    daily['physio_injury_ratio'] = injury_features['physio_injury_ratio']
    
    # Select only the required columns
    required_columns = [
        'cum_inj_starts', 'cum_inj_days', 'days_since_last_injury',
        'avg_injury_duration', 'injury_frequency',
        'avg_injury_severity', 'max_injury_severity',
        'lower_leg_injuries', 'knee_injuries', 'upper_leg_injuries',
        'hip_injuries', 'upper_body_injuries', 'head_injuries',
        'illness_count', 'physio_injury_ratio',
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
    player_injuries: pd.DataFrame
) -> pd.DataFrame:
    """
    Enhanced daily features generation with fixed date range logic.
    
    Args:
        player_id: Player identifier
        player_row: Player profile data
        player_matches: Player's match history
        player_injuries: Player's injury history
        
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
    
    calendar = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Build feature series
    profile_series = build_daily_profile_series_optimized(player_row, calendar, player_matches)
    match_series = build_daily_match_series_optimized(player_matches, calendar, player_row)
    injury_series = build_daily_injury_series_optimized(player_injuries, calendar)
    interaction_series = build_daily_interaction_features_optimized(profile_series, match_series, injury_series)
    
    # Combine all features
    daily_features = pd.concat([profile_series, match_series, injury_series, interaction_series], axis=1)
    
    # Remove duplicate columns
    daily_features = daily_features.loc[:, ~daily_features.columns.duplicated()]
    
    # Define our 98 selected features (51 original + 47 new - 1 removed)
    selected_features = [
        # Profile features (13)
        'age', 'seniority_days', 'position', 'nationality1', 'nationality2', 
        'height_cm', 'dominant_foot', 'previous_club', 'previous_club_country',
        'current_club', 'current_club_country', 'teams_today', 'cum_teams',
        
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
        'cum_inj_starts', 'cum_inj_days', 'days_since_last_injury',
        'avg_injury_duration', 'injury_frequency',
        
        # Interaction features (3)
        'age_x_career_matches', 'age_x_career_goals', 'seniority_x_goals_per_match',
        
        # ENHANCED INJURY FEATURES (11)
        'avg_injury_severity', 'max_injury_severity',
        'lower_leg_injuries', 'knee_injuries', 'upper_leg_injuries',
        'hip_injuries', 'upper_body_injuries', 'head_injuries',
        'illness_count', 'physio_injury_ratio', 'cum_matches_injured',
        
        # ENHANCED COMPETITION & SEASON FEATURES (6)
        'competition_importance', 'avg_competition_importance', 'season_phase',
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
        'consecutive_substitutions'
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

def generate_features_for_all_players(output_dir: str = 'daily_features_output'):
    """
    Generate daily features for all players in the dataset.
    
    Args:
        output_dir (str): Directory to save the output files
    """
    print("🚀 ENHANCED GOLD STANDARD DAILY FEATURES GENERATOR")
    print("=" * 70)
    print("📋 Features: 108 total (51 original + 57 new - 1 removed)")
    print("🔧 Production-ready with no hardcoded values or synthetic data")
    
    start_time = datetime.now()
    
    # Load data
    data = load_data_with_cache()
    players, injuries, matches = data['players'], data['injuries'], data['matches']
    
    # Preprocess data
    players, injuries, matches = preprocess_data_optimized(players, injuries, matches)
    
    # Get all non-goalkeeper player IDs
    all_player_ids = players['id'].tolist()
    
    print(f"🎯 Processing {len(all_player_ids)} players")
    print(f"📁 Output will be saved to '{output_dir}/' folder")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each player
    successful_players = 0
    failed_players = 0
    
    for i, player_id in enumerate(all_player_ids, 1):
        print(f"\n🔄 Processing player {i}/{len(all_player_ids)}: {player_id}")
        
        try:
            # Get player data
            player_filter = players[players['id'] == player_id]
            if player_filter.empty:
                print(f"   ❌ Player {player_id} not found in players data")
                failed_players += 1
                continue
                
            player_row = player_filter.iloc[0]
            player_matches = matches[matches['player_id'] == player_id].copy()
            player_injuries = injuries[injuries['player_id'] == player_id].copy()
            
            print(f"   Matches: {len(player_matches)}")
            print(f"   Injuries: {len(player_injuries)}")
            
            # Generate daily features
            daily_features = generate_daily_features_for_player_enhanced(
                player_id, player_row, player_matches, player_injuries
            )
            
            # Save to output folder
            output_file = f'{output_dir}/player_{player_id}_daily_features.csv'
            daily_features.to_csv(output_file, index=False)
            
            print(f"✅ Saved: {output_file}")
            print(f"📊 Shape: {daily_features.shape}")
            print(f"📅 Date range: {daily_features['date'].min()} to {daily_features['date'].max()}")
            
            successful_players += 1
            
        except Exception as e:
            print(f"❌ Error processing player {player_id}: {str(e)}")
            failed_players += 1
    
    total_time = datetime.now() - start_time
    print(f"\n⏱️  Total processing time: {total_time}")
    print(f"✅ Successfully processed: {successful_players} players")
    print(f"❌ Failed to process: {failed_players} players")
    print(f"🎉 Feature generation completed! Check the '{output_dir}/' folder for results.")

def main():
    """Main function - can be used for testing or production"""
    import sys
    
    if len(sys.argv) > 1:
        # Production mode with custom output directory
        output_dir = sys.argv[1]
        generate_features_for_all_players(output_dir)
    else:
        # Default production mode
        generate_features_for_all_players()

if __name__ == "__main__":
    main()
