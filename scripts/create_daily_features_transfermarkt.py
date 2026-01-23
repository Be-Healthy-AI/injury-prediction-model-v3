#!/usr/bin/env python3
"""
Daily Features Generator for Transfermarkt Data
Adapted from create_daily_features_v3.py to read from Transfermarkt CSV exports
Reads from: data_exports/transfermarkt/england/20251203/

This script imports the original create_daily_features_v3 module and overrides
only the data loading functions to work with Transfermarkt CSV data.
"""

import pandas as pd
import numpy as np
import os
import sys
import glob
from tqdm import tqdm
import pickle
import logging
import re
from typing import Dict, Optional
from datetime import datetime, timedelta

# Add scripts directory to path to import the original module
sys.path.insert(0, os.path.dirname(__file__))

# Import benfica-parity functions (needed for preprocessing)
from benfica_parity_config import (
    map_competition_importance_benfica_parity,
    detect_disciplinary_action_benfica_parity,
)

# Import the original module
import create_daily_features_v3 as original_module

# Override CONFIG in the original module
original_module.CONFIG['DATA_DIR'] = r'data_exports\transfermarkt\england\20251203'
original_module.CONFIG['CACHE_FILE'] = 'data_cache_transfermarkt.pkl'

# Use standard logging instead of relying on original_module.logger
logger = logging.getLogger(__name__)

def load_match_data_from_folder(match_data_dir: str, player_id: Optional[int] = None) -> pd.DataFrame:
    """
    Load match data CSV files from the match_data folder and combine them.
    If player_id is provided, only loads files for that specific player.
    
    Args:
        match_data_dir: Path to the match_data folder
        player_id: Optional player ID to filter files (e.g., only load match_8198_*.csv)
        
    Returns:
        Combined DataFrame with all match data
    """
    if player_id is not None:
        logger.info(f"[IO] Loading match data from folder: {match_data_dir} (player {player_id} only)")
        # Only load files for this specific player
        match_files = glob.glob(os.path.join(match_data_dir, f'match_{player_id}_*.csv'))
    else:
        logger.info(f"[IO] Loading match data from folder: {match_data_dir}")
        match_files = glob.glob(os.path.join(match_data_dir, 'match_*.csv'))
    
    if not match_files:
        if player_id is not None:
            logger.warning(f"No match data files found for player {player_id} in {match_data_dir}")
        else:
            logger.warning(f"No match data files found in {match_data_dir}")
        return pd.DataFrame()
    
    logger.info(f"   Found {len(match_files)} match data files")
    
    all_matches = []
    for match_file in tqdm(match_files, desc="Loading match files", unit="file", leave=False):
        try:
            # Match data files use comma separator
            df = pd.read_csv(match_file, sep=',', encoding='utf-8')
            all_matches.append(df)
        except Exception as e:
            logger.warning(f"Error loading {match_file}: {e}")
            continue
    
    if not all_matches:
        logger.warning("No match data could be loaded")
        return pd.DataFrame()
    
    # Combine all match data
    matches = pd.concat(all_matches, ignore_index=True)
    logger.info(f"Loaded {len(matches)} total match records from {len(match_files)} files")
    
    return matches

def load_data_with_cache(player_id: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    """
    Load data files with caching for performance.
    Adapted to read CSV files from Transfermarkt export folder.
    This function overrides the original load_data_with_cache.
    
    Args:
        player_id: Optional player ID to only load match data for that player (for optimization)
    
    Returns:
        Dictionary containing all data DataFrames
    """
    import time
    
    # For single player mode, use a separate cache file to avoid conflicts
    if player_id is not None:
        cache_file = f'data_cache_transfermarkt_player_{player_id}.pkl'
    else:
        cache_file = original_module.CONFIG['CACHE_FILE']
    
    # Check if cache exists and is recent
    if os.path.exists(cache_file):
        cache_age = time.time() - os.path.getmtime(cache_file)
        if cache_age < original_module.CONFIG['CACHE_DURATION']:
            logger.info("[CACHE] Loading data from cache...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
    
    logger.info("[IO] Loading data files from Transfermarkt export...")
    data_dir = original_module.CONFIG['DATA_DIR']
    
    # Load CSV files (semicolon-separated for profile, career, injuries)
    players_path = os.path.join(data_dir, 'players_profile.csv')
    injuries_path = os.path.join(data_dir, 'injuries_data.csv')
    career_path = os.path.join(data_dir, 'players_career.csv')
    match_data_dir = os.path.join(data_dir, 'match_data')
    
    # Load players profile (semicolon-separated)
    logger.info(f"[IO] Loading: {os.path.basename(players_path)}")
    players = pd.read_csv(players_path, sep=';', encoding='utf-8')
    
    # Convert date_of_birth to datetime (Transfermarkt uses DD/MM/YYYY format)
    if 'date_of_birth' in players.columns:
        players['date_of_birth'] = pd.to_datetime(players['date_of_birth'], format='%d/%m/%Y', errors='coerce')
    
    # Convert height to height_cm (Transfermarkt height is already in cm, but we need to ensure it's numeric)
    if 'height' in players.columns and 'height_cm' not in players.columns:
        def parse_height(height_val):
            if pd.isna(height_val):
                return np.nan
            # Height is already in cm in Transfermarkt data, just convert to float
            try:
                return float(height_val)
            except (ValueError, TypeError):
                return np.nan
        players['height_cm'] = players['height'].apply(parse_height)
    
    # Map column names from Transfermarkt format to expected format
    # Transfermarkt uses 'foot', original script expects 'dominant_foot'
    if 'foot' in players.columns and 'dominant_foot' not in players.columns:
        players['dominant_foot'] = players['foot']
    
    # Load injuries data (semicolon-separated)
    logger.info(f"[IO] Loading: {os.path.basename(injuries_path)}")
    injuries = pd.read_csv(injuries_path, sep=';', encoding='utf-8')
    
    # Load match data from folder (only for specific player if provided)
    matches = load_match_data_from_folder(match_data_dir, player_id=player_id)
    
    # Load career data (semicolon-separated)
    career = None
    if os.path.exists(career_path):
        logger.info(f"[IO] Loading: {os.path.basename(career_path)}")
        career = pd.read_csv(career_path, sep=';', encoding='utf-8')
    else:
        logger.warning("Players career file not found; previous club seeding will be skipped.")
    
    # Teams and competitions data are optional - set to None
    # The original script handles None gracefully
    teams = None
    competitions = None
    
    # Cache the data
    data = {
        'players': players,
        'injuries': injuries,
        'matches': matches,
        'teams': teams,
        'competitions': competitions,
        'career': career,
    }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    
    return data

def preprocess_data_optimized(players, injuries, matches):
    """Optimized data preprocessing with vectorized operations
    Adapted to handle Transfermarkt data format with injury_class column
    
    This function overrides the original preprocess_data_optimized.
    
    Returns:
        Tuple of (players, physio_injuries, non_physio_injuries, matches)
        - physio_injuries: Used for target calculation (muscular injuries)
        - non_physio_injuries: Used for feature generation (skeletical, other, no_injury)
    """
    # Avoid emojis in console output for compatibility
    print("Preprocessing data (Transfermarkt override)...")
    print(f"   Players: {len(players)}, Injuries: {len(injuries)}, Matches: {len(matches)}")
    
    # Convert height to height_cm if needed (in case it wasn't done during loading)
    if 'height' in players.columns and 'height_cm' not in players.columns:
        def parse_height(height_val):
            if pd.isna(height_val):
                return np.nan
            try:
                return float(height_val)
            except (ValueError, TypeError):
                return np.nan
        players['height_cm'] = players['height'].apply(parse_height)
    
    # Map column names from Transfermarkt format to expected format
    if 'foot' in players.columns and 'dominant_foot' not in players.columns:
        players['dominant_foot'] = players['foot']
    
    # Filter goalkeepers (vectorized)
    players = players[players['position'] != 'Goalkeeper'].copy()
    
    # Handle injuries - Check for injury_class column (new format) or no_physio_injury (old format)
    if 'injury_class' in injuries.columns:
        # New format: injury_class with values: muscular, skeletical, other, no_injury
        # Muscular injuries = target (physio_injuries)
        # All others = features (non_physio_injuries)
        physio_injuries = injuries[injuries['injury_class'] == 'muscular'].copy()
        non_physio_injuries = injuries[injuries['injury_class'].isin(['skeletical', 'other', 'no_injury'])].copy()
        
        print(f"   Injury class distribution:")
        if not injuries.empty:
            class_counts = injuries['injury_class'].value_counts()
            for class_name, count in class_counts.items():
                print(f"      {class_name}: {count}")
    elif 'no_physio_injury' in injuries.columns:
        # Original logic: separate physio and non-physio injuries (backward compatibility)
        physio_injuries = injuries[injuries['no_physio_injury'].isna()].copy()
        non_physio_injuries = injuries[injuries['no_physio_injury'] == 1].copy()
    else:
        # If neither column exists, treat all injuries as physio injuries
        logger.warning("Neither 'injury_class' nor 'no_physio_injury' column found in injuries data. Treating all injuries as physio injuries.")
        physio_injuries = injuries.copy()
        non_physio_injuries = pd.DataFrame(columns=injuries.columns)
    
    print(f"   Physio injuries (target - muscular): {len(physio_injuries)}")
    print(f"   Non-physio injuries (features - skeletical/other/no_injury): {len(non_physio_injuries)}")
    
    # Convert dates (vectorized) for both injury types
    # Transfermarkt uses DD/MM/YYYY format
    date_columns = ['fromDate', 'untilDate']
    for col in date_columns:
        if col in physio_injuries.columns:
            physio_injuries[col] = pd.to_datetime(physio_injuries[col], format='%d/%m/%Y', errors='coerce')
        if col in non_physio_injuries.columns and not non_physio_injuries.empty:
            non_physio_injuries[col] = pd.to_datetime(non_physio_injuries[col], format='%d/%m/%Y', errors='coerce')
    
    # Add duration_days (vectorized)
    if 'untilDate' in physio_injuries.columns and 'fromDate' in physio_injuries.columns:
        physio_injuries['duration_days'] = (physio_injuries['untilDate'] - physio_injuries['fromDate']).dt.days
    if not non_physio_injuries.empty and 'untilDate' in non_physio_injuries.columns and 'fromDate' in non_physio_injuries.columns:
        non_physio_injuries['duration_days'] = (non_physio_injuries['untilDate'] - non_physio_injuries['fromDate']).dt.days
    
    # Preprocess matches (vectorized)
    print("   Processing matches...")
    if 'date' in matches.columns:
        matches['date'] = pd.to_datetime(matches['date'], errors='coerce')
    
    # Optimized minutes parsing
    print("   Parsing minutes played...")
    def parse_minutes_vectorized(minutes_series):
        """Vectorized minutes parsing"""
        # Convert to string and handle NaN
        minutes_str = minutes_series.astype(str)
        
        # Extract numbers using regex (vectorized)
        pattern = r'(\d+)'
        
        def extract_minutes(x):
            if pd.isna(x) or x == 'nan':
                return np.nan
            match = re.search(pattern, str(x))
            return int(match.group(1)) if match else np.nan
        
        return minutes_str.apply(extract_minutes)
    
    # Create numeric columns for matches (required by the feature generation)
    if 'minutes_played' in matches.columns:
        matches['minutes_played_numeric'] = parse_minutes_vectorized(matches['minutes_played'])
    
    # Vectorized numeric conversions for goals and assists
    numeric_cols = ['goals', 'assists']
    for col in numeric_cols:
        if col in matches.columns:
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
            cards = re.findall(r'\d+\'', str(x))
            return len(cards)
        
        return cards_series.apply(parse_cards)
    
    if 'yellow_cards' in matches.columns:
        matches['yellow_cards_numeric'] = parse_cards_vectorized(matches['yellow_cards'])
    if 'second_yellow_cards' in matches.columns:
        matches['second_yellow_cards_numeric'] = parse_cards_vectorized(matches['second_yellow_cards'])
    if 'red_cards' in matches.columns:
        matches['red_cards_numeric'] = parse_cards_vectorized(matches['red_cards'])
    
    # Combine yellow cards and second yellow cards for total yellow cards count
    if 'yellow_cards_numeric' in matches.columns and 'second_yellow_cards_numeric' in matches.columns:
        matches['yellow_cards_numeric'] = matches['yellow_cards_numeric'] + matches['second_yellow_cards_numeric']
    
    print("   Adding competition and disciplinary features...")
    import sys
    sys.stdout.flush()
    # Add competition_importance and disciplinary_action columns (required by feature generation)
    # These need to be calculated using the benfica-parity functions
    if 'competition' in matches.columns:
        if 'competition_importance' not in matches.columns:
            print("   Calculating competition importance...")
            sys.stdout.flush()
            matches['competition_importance'] = matches['competition'].apply(map_competition_importance_benfica_parity)
            print(f"      Competition importance calculated for {len(matches)} matches")
            sys.stdout.flush()
    
    if 'disciplinary_action' not in matches.columns:
        print("   Calculating disciplinary actions...")
        import sys
        sys.stdout.flush()
        if not matches.empty:
            print(f"      Processing {len(matches)} matches...")
            sys.stdout.flush()
            # Simplified vectorized calculation - just check cards (position check can be slow)
            # Initialize with 0
            matches['disciplinary_action'] = 0
            
            # Check cards - use the numeric versions (vectorized and fast)
            if 'red_cards_numeric' in matches.columns:
                matches.loc[matches['red_cards_numeric'] > 0, 'disciplinary_action'] = 1
                print(f"      Red cards checked")
                sys.stdout.flush()
            
            if 'yellow_cards_numeric' in matches.columns:
                # Two yellows = red
                matches.loc[matches['yellow_cards_numeric'] >= 2, 'disciplinary_action'] = 1
                print(f"      Yellow cards checked")
                sys.stdout.flush()
            
            # Check position field only if needed (can be slow, so do it last)
            if 'position' in matches.columns:
                print(f"      Checking position field...")
                sys.stdout.flush()
                try:
                    position_lower = matches['position'].astype(str).str.lower()
                    # Use single regex pattern
                    pattern = '|'.join(['suspended', 'banned', 'punished', 'disciplinary'])
                    matches.loc[position_lower.str.contains(pattern, na=False, regex=True), 'disciplinary_action'] = 1
                    print(f"      Position field checked")
                    sys.stdout.flush()
                except Exception as e:
                    print(f"      Warning: Error checking position field: {e}")
                    sys.stdout.flush()
            
            print(f"      Disciplinary actions: {matches['disciplinary_action'].sum()} found")
            sys.stdout.flush()
        else:
            matches['disciplinary_action'] = 0
    
    # Avoid emojis in console output for compatibility
    print("Preprocessing complete!")
    import sys
    sys.stdout.flush()
    return players, physio_injuries, non_physio_injuries, matches

# Override the functions in the original module
original_module.load_data_with_cache = load_data_with_cache
original_module.preprocess_data_optimized = preprocess_data_optimized

def generate_features_for_single_player(
    player_id: int,
    output_dir: str = 'daily_features_output',
    force_rebuild: bool = False
):
    """
    Generate daily features for a single player.
    
    Args:
        player_id (int): ID of the player to process
        output_dir (str): Directory to save the output file
        force_rebuild (bool): Force regeneration even if file exists
    """
    print("DAILY FEATURES GENERATOR - TRANSFERMARKT DATA (SINGLE PLAYER)")
    print("=" * 70)
    print(f"Processing player ID: {player_id}")
    print("=" * 70)
    
    start_time = datetime.now()
    
    # Load data using our custom loader (only load match files for this player)
    data = load_data_with_cache(player_id=player_id)
    players, injuries, matches = data['players'], data['injuries'], data['matches']
    teams = data.get('teams')
    competitions = data.get('competitions')
    original_module.initialize_team_country_map(teams)
    original_module.initialize_competition_type_map(competitions)
    career = original_module.preprocess_career_data(data.get('career'))
    
    # Filter to only this player's data BEFORE preprocessing to speed things up
    print(f"Filtering data for player {player_id}...")
    players = players[players['id'] == player_id].copy()
    if players.empty:
        print(f"ERROR: Player {player_id} not found in players data")
        return
    
    # Filter injuries and matches for this player
    injuries = injuries[injuries['player_id'] == player_id].copy()
    matches = matches[matches['player_id'] == player_id].copy()
    
    print(f"   Filtered: {len(players)} players, {len(injuries)} injuries, {len(matches)} matches")
    
    # Preprocess data using our adapted preprocessor (now only for this player)
    print("Starting data preprocessing...")
    import sys
    sys.stdout.flush()
    players, physio_injuries, non_physio_injuries, matches = preprocess_data_optimized(players, injuries, matches)
    print("Preprocessing complete!")
    sys.stdout.flush()
    
    # Check if player exists (should still be there after filtering)
    player_filter = players[players['id'] == player_id]
    if player_filter.empty:
        print(f"ERROR: Player {player_id} not found in players data")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if file already exists
    output_file = os.path.join(output_dir, f'player_{player_id}_daily_features.csv')
    if os.path.exists(output_file) and not force_rebuild:
        print(f"File already exists: {output_file}")
        print("   Use --force-rebuild to regenerate")
        return
    
    try:
        player_row = player_filter.iloc[0]
        player_matches = matches[matches['player_id'] == player_id].copy()
        player_physio_injuries = physio_injuries[physio_injuries['player_id'] == player_id].copy()
        player_non_physio_injuries = non_physio_injuries[non_physio_injuries['player_id'] == player_id].copy() if not non_physio_injuries.empty else pd.DataFrame()
        player_career = None
        if career is not None:
            id_col = 'player_id' if 'player_id' in career.columns else ('id' if 'id' in career.columns else None)
            if id_col is not None:
                player_career = career[career[id_col] == player_id].copy()
        
        print("Player data loaded:")
        print(f"   Name: {player_row.get('name', 'N/A')}")
        print(f"   Position: {player_row.get('position', 'N/A')}")
        print(f"   Matches: {len(player_matches)}")
        print(f"   Physio Injuries: {len(player_physio_injuries)}")
        print(f"   Non-Physio Injuries: {len(player_non_physio_injuries)}")
        print(f"   Career entries: {len(player_career) if player_career is not None else 0}")
        
        # Determine global end date cap
        global_end_date_cap = None
        target_end_date = pd.Timestamp('2025-12-05').normalize()
        
        try:
            today_cap = pd.Timestamp.today().normalize()
        except Exception:
            today_cap = pd.Timestamp('today').normalize()

        if player_matches is not None and not player_matches.empty:
            max_match_date = player_matches['date'].max()
            if pd.notna(max_match_date):
                global_end_date_cap = min(max_match_date, today_cap, target_end_date)
            else:
                global_end_date_cap = min(today_cap, target_end_date)
        else:
            all_injuries = pd.concat(
                [player_physio_injuries, player_non_physio_injuries],
                ignore_index=True
            ) if not player_physio_injuries.empty or not player_non_physio_injuries.empty else pd.DataFrame()

            if not all_injuries.empty and 'fromDate' in all_injuries.columns:
                latest_injury_date = all_injuries['fromDate'].max()
                if pd.notna(latest_injury_date):
                    global_end_date_cap = min(latest_injury_date, today_cap, target_end_date)
                else:
                    global_end_date_cap = min(today_cap, target_end_date)
            else:
                global_end_date_cap = min(today_cap, target_end_date)

        print(f"Global calendar cap set to: {global_end_date_cap}")
        print("Generating daily features...")
        print(f"   Processing {len(player_matches)} matches - this may take a few minutes...")
        
        # Calculate expected calendar size
        if not player_matches.empty:
            expected_start = player_matches['date'].min()
            expected_end = global_end_date_cap
            expected_days = (expected_end - expected_start).days
            print(f"   Expected calendar size: ~{expected_days} days ({expected_days/365:.1f} years)")
            if expected_days > 5000:
                print("      WARNING: Very large calendar - processing may take 30+ minutes")
        
        import sys
        import time
        sys.stdout.flush()
        
        # Comprehensive pre-generation checks and logging
        print("\n" + "="*70)
        print("PRE-FEATURE GENERATION DIAGNOSTICS")
        print("="*70)
        
        # Check player_matches
        print(f"\nPlayer Matches Info:")
        print(f"   Shape: {player_matches.shape}")
        if not player_matches.empty and 'date' in player_matches.columns:
            print(f"   Date range: {player_matches['date'].min()} to {player_matches['date'].max()}")
        print(f"   Columns ({len(player_matches.columns)}): {', '.join(list(player_matches.columns)[:10])}...")
        
        # Check required columns
        print(f"\nRequired Columns Check:")
        required_cols = {
            'minutes_played_numeric': 0,
            'goals_numeric': 0,
            'assists_numeric': 0,
            'yellow_cards_numeric': 0,
            'red_cards_numeric': 0,
            'competition_importance': 1,  # Default value if missing
            'disciplinary_action': 0,
            'date': None
        }
        
        missing_cols = []
        for col, default_val in required_cols.items():
            if col in player_matches.columns:
                non_null = player_matches[col].notna().sum() if default_val is not None else len(player_matches)
                null_count = len(player_matches) - non_null
                print(f"   OK {col}: {non_null} non-null, {null_count} null")
                if null_count > 0 and default_val is not None:
                    print(f"      WARNING: Filling {null_count} null values with {default_val}")
                    player_matches[col].fillna(default_val, inplace=True)
            else:
                print(f"   MISSING {col}: MISSING")
                missing_cols.append(col)
                if default_val is not None:
                    print(f"      ➕ Creating column with default value {default_val}")
                    player_matches[col] = default_val
        
        if missing_cols:
            print(f"\nWARNING: {len(missing_cols)} required columns were missing and have been initialized")
        
        # Check data types
        print(f"\n🔢 Data Types Check:")
        for col in ['minutes_played_numeric', 'goals_numeric', 'assists_numeric', 
                   'yellow_cards_numeric', 'red_cards_numeric', 'disciplinary_action']:
            if col in player_matches.columns:
                dtype = player_matches[col].dtype
                print(f"   {col}: {dtype}")
                if not pd.api.types.is_numeric_dtype(dtype):
                    print("      WARNING: Converting to numeric...")
                    player_matches[col] = pd.to_numeric(player_matches[col], errors='coerce').fillna(0)
        
        # Check injuries
        print(f"\n🏥 Injuries Info:")
        print(f"   Physio injuries: {len(player_physio_injuries)}")
        print(f"   Non-physio injuries: {len(player_non_physio_injuries)}")
        
        # Check career
        print(f"\nCareer Info:")
        print(f"   Career entries: {len(player_career) if player_career is not None else 0}")
        
        print("\n" + "="*70)
        print("STARTING FEATURE GENERATION")
        print("="*70 + "\n")
        
        import sys
        import time
        sys.stdout.flush()
        
        start_gen_time = time.time()
        
        # Generate daily features
        print("   Calling generate_daily_features_for_player_enhanced...")
        print("   (This function will:")
        print("      1. Build daily profile series")
        print("      2. Build daily match series (this may take time with many matches)")
        print("      3. Build daily injury series")
        print("      4. Build interaction features)")
        sys.stdout.flush()
        
        # Monkey-patch critical functions to add detailed logging
        original_compute_team_result_metrics = original_module.compute_team_result_metrics
        original_build_daily_match_series = original_module.build_daily_match_series_optimized
        original_build_daily_profile_series = original_module.build_daily_profile_series_optimized
        original_determine_participation = original_module.determine_match_participation_optimized
        
        def logged_compute_team_result_metrics(matches, calendar, player_row, profile_series, player_career):
            print("   STEP: compute_team_result_metrics called...")
            print(f"      Matches: {len(matches)}, Calendar days: {len(calendar)}")
            sys.stdout.flush()
            start = time.time()
            try:
                if not matches.empty:
                    print(f"      Processing {len(matches)} matches (using iterrows - this may be slow)...")
                    sys.stdout.flush()
                    # Monkey-patch iterrows to show progress
                    original_iterrows = matches.iterrows
                    match_count = [0]
                    def logged_iterrows():
                        for idx, row in original_iterrows():
                            match_count[0] += 1
                            if match_count[0] % 200 == 0:
                                print(f"         [PROGRESS] Processed {match_count[0]}/{len(matches)} matches in compute_team_result_metrics...")
                                sys.stdout.flush()
                            yield idx, row
                    matches.iterrows = logged_iterrows
                result = original_compute_team_result_metrics(matches, calendar, player_row, profile_series, player_career)
                elapsed = time.time() - start
                print(f"   STEP: compute_team_result_metrics completed in {elapsed:.2f}s")
                sys.stdout.flush()
                return result
            except Exception as e:
                elapsed = time.time() - start
                print(f"   STEP: compute_team_result_metrics FAILED after {elapsed:.2f}s: {e}")
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
                raise
        
        def logged_determine_participation(matches):
            print(f"   STEP: determine_match_participation_optimized called for {len(matches)} matches...")
            sys.stdout.flush()
            start = time.time()
            try:
                result = original_determine_participation(matches)
                elapsed = time.time() - start
                print(f"   STEP: determine_match_participation_optimized completed in {elapsed:.2f}s")
                sys.stdout.flush()
                return result
            except Exception as e:
                elapsed = time.time() - start
                print(f"   STEP: determine_match_participation_optimized FAILED after {elapsed:.2f}s: {e}")
                sys.stdout.flush()
                raise
        
        def logged_build_daily_match_series(matches, calendar, player_row, profile_series, player_career):
            print("   STEP: build_daily_match_series_optimized called...")
            print(f"      Matches: {len(matches)}, Calendar days: {len(calendar)}")
            sys.stdout.flush()
            start = time.time()
            try:
                # Add logging for internal operations
                print("      [SUBSTEP] Grouping matches by date...")
                sys.stdout.flush()
                
                # Import benfica_parity functions to add logging
                from benfica_parity_config import calculate_enhanced_features_dynamically as original_calc_enhanced
                from benfica_parity_config import calculate_national_team_features_benfica_parity as original_calc_national
                from benfica_parity_config import calculate_complex_derived_features_benfica_parity as original_calc_complex
                
                def logged_calc_enhanced(*args, **kwargs):
                    print("      [SUBSTEP] calculate_enhanced_features_dynamically called...")
                    sys.stdout.flush()
                    sub_start = time.time()
                    result = original_calc_enhanced(*args, **kwargs)
                    elapsed = time.time() - sub_start
                    print(f"      [SUBSTEP] calculate_enhanced_features_dynamically completed in {elapsed:.2f}s")
                    sys.stdout.flush()
                    return result
                
                def logged_calc_national(*args, **kwargs):
                    print("      [SUBSTEP] calculate_national_team_features_benfica_parity called...")
                    sys.stdout.flush()
                    sub_start = time.time()
                    result = original_calc_national(*args, **kwargs)
                    elapsed = time.time() - sub_start
                    print(f"      [SUBSTEP] calculate_national_team_features_benfica_parity completed in {elapsed:.2f}s")
                    sys.stdout.flush()
                    return result
                
                def logged_calc_complex(*args, **kwargs):
                    print("      [SUBSTEP] calculate_complex_derived_features_benfica_parity called...")
                    sys.stdout.flush()
                    sub_start = time.time()
                    result = original_calc_complex(*args, **kwargs)
                    elapsed = time.time() - sub_start
                    print(f"      [SUBSTEP] calculate_complex_derived_features_benfica_parity completed in {elapsed:.2f}s")
                    sys.stdout.flush()
                    return result
                
                # Monkey-patch the benfica_parity functions
                import benfica_parity_config
                benfica_parity_config.calculate_enhanced_features_dynamically = logged_calc_enhanced
                benfica_parity_config.calculate_national_team_features_benfica_parity = logged_calc_national
                benfica_parity_config.calculate_complex_derived_features_benfica_parity = logged_calc_complex
                
                try:
                    print("      [SUBSTEP] Calling original build_daily_match_series...")
                    sys.stdout.flush()
                    result = original_build_daily_match_series(matches, calendar, player_row, profile_series, player_career)
                    print("      [SUBSTEP] Original build_daily_match_series returned")
                    sys.stdout.flush()
                finally:
                    # Restore original functions
                    benfica_parity_config.calculate_enhanced_features_dynamically = original_calc_enhanced
                    benfica_parity_config.calculate_national_team_features_benfica_parity = original_calc_national
                    benfica_parity_config.calculate_complex_derived_features_benfica_parity = original_calc_complex
                
                elapsed = time.time() - start
                print(f"   STEP: build_daily_match_series_optimized completed in {elapsed:.2f}s")
                sys.stdout.flush()
                return result
            except Exception as e:
                elapsed = time.time() - start
                print(f"   STEP: build_daily_match_series_optimized FAILED after {elapsed:.2f}s: {e}")
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
                raise
        
        def logged_build_daily_profile_series(*args, **kwargs):
            print("   STEP: build_daily_profile_series_optimized called...")
            sys.stdout.flush()
            start = time.time()
            try:
                result = original_build_daily_profile_series(*args, **kwargs)
                elapsed = time.time() - start
                print(f"   STEP: build_daily_profile_series_optimized completed in {elapsed:.2f}s")
                sys.stdout.flush()
                return result
            except Exception as e:
                elapsed = time.time() - start
                print(f"   STEP: build_daily_profile_series_optimized FAILED after {elapsed:.2f}s: {e}")
                sys.stdout.flush()
                raise
        
        # Apply monkey patches
        original_module.compute_team_result_metrics = logged_compute_team_result_metrics
        original_module.build_daily_match_series_optimized = logged_build_daily_match_series
        original_module.build_daily_profile_series_optimized = logged_build_daily_profile_series
        original_module.determine_match_participation_optimized = logged_determine_participation
        
        try:
            daily_features = original_module.generate_daily_features_for_player_enhanced(
                player_id,
                player_row,
                player_matches,
                player_physio_injuries,
                player_non_physio_injuries,
                player_career,
                global_end_date_cap=global_end_date_cap
            )
            
            elapsed_gen = time.time() - start_gen_time
            print(f"\nFeature generation completed in {elapsed_gen:.2f} seconds ({elapsed_gen/60:.2f} minutes)")
            sys.stdout.flush()
            
        except Exception as e:
            elapsed_gen = time.time() - start_gen_time
            print(f"\nFeature generation FAILED after {elapsed_gen:.2f} seconds")
            print(f"   Error: {e}")
            import traceback
            print("\nFull traceback:")
            traceback.print_exc()
            sys.stdout.flush()
            raise
        finally:
            # Restore original functions
            original_module.compute_team_result_metrics = original_compute_team_result_metrics
            original_module.build_daily_match_series_optimized = original_build_daily_match_series
            original_module.build_daily_profile_series_optimized = original_build_daily_profile_series
            original_module.determine_match_participation_optimized = original_determine_participation
        
        if daily_features is None or daily_features.empty:
            print(f"No daily features generated for player {player_id}")
            return
        
        # Save to output folder
        daily_features.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        # Calculate timing
        total_time = datetime.now() - start_time
        
        print("\n" + "=" * 70)
        print("GENERATION COMPLETE")
        print("=" * 70)
        print(f"Processing time: {total_time}")
        print(f"Generated {daily_features.shape[0]} days of features")
        print(f"Total features: {daily_features.shape[1]}")
        print(f"Output file: {output_file}")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nError processing player {player_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def generate_features_for_all_players(
    output_dir: str = 'daily_features_output',
    max_players: Optional[int] = None,
    random_seed: Optional[int] = None,
    force_rebuild: bool = False
):
    """
    Generate daily features for players in the dataset.
    This is a wrapper that calls the original function.
    """
    # Use the original function from the module
    original_module.generate_features_for_all_players(
        output_dir=output_dir,
        max_players=max_players,
        random_seed=random_seed,
        force_rebuild=force_rebuild
    )

def main():
    """Main function - can be used for testing or production"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate daily features for players from Transfermarkt data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test on a single player
  python create_daily_features_transfermarkt.py --player-id 238223
  
  # Test mode: Process 10 randomly selected players
  python create_daily_features_transfermarkt.py --test
  
  # Custom number of players
  python create_daily_features_transfermarkt.py --max-players 20
  
  # Full mode: Process all players
  python create_daily_features_transfermarkt.py
  
  # Test mode with custom output directory
  python create_daily_features_transfermarkt.py --test --output-dir test_output
  
  # Custom seed for reproducibility
  python create_daily_features_transfermarkt.py --test --seed 123
        """
    )
    
    parser.add_argument(
        '--player-id',
        type=int,
        default=None,
        help='Process a single player by ID (useful for testing)'
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
    
    parser.add_argument(
        '--force-rebuild',
        action='store_true',
        help='Force regeneration of all daily features files, even if they already exist'
    )
    
    args = parser.parse_args()
    
    # If player-id is specified, process only that player
    if args.player_id is not None:
        generate_features_for_single_player(
            player_id=args.player_id,
            output_dir=args.output_dir,
            force_rebuild=args.force_rebuild
        )
        return
    
    # Otherwise, use the normal flow
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
        random_seed=args.seed,
        force_rebuild=args.force_rebuild
    )

if __name__ == "__main__":
    main()

