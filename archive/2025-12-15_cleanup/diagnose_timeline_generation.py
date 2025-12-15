#!/usr/bin/env python3
"""
Diagnostic script to analyze why only 675 injury timelines were generated
instead of the expected ~13,000 (2700 injuries × 5 timelines)
"""

import pandas as pd
import os
from datetime import timedelta
from typing import Dict, Tuple, Set, List
from collections import defaultdict

# Configuration (matching the main script)
TRAIN_START = pd.Timestamp('2022-07-01')
TRAIN_END = pd.Timestamp('2024-06-30')
ALLOWED_INJURY_CLASSES = {'muscular', 'skeletal', 'unknown'}

# Goalkeeper IDs (full list from main script)
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

def derive_injury_class(injury_type: str, no_physio_injury) -> str:
    """Same logic as main script"""
    if pd.notna(no_physio_injury) and no_physio_injury == 1.0:
        return 'other'
    if pd.isna(injury_type) or injury_type == '':
        return 'unknown'
    injury_lower = str(injury_type).lower()
    skeletal_keywords = ['fracture', 'bone', 'ligament', 'tendon', 'cartilage', 'meniscus', 
                         'acl', 'cruciate', 'dislocation', 'subluxation', 'sprain', 'joint']
    if any(keyword in injury_lower for keyword in skeletal_keywords):
        return 'skeletal'
    muscular_keywords = ['strain', 'muscle', 'hamstring', 'quadriceps', 'calf', 'groin', 
                        'adductor', 'abductor', 'thigh', 'tear', 'rupture', 'pull']
    if any(keyword in injury_lower for keyword in muscular_keywords):
        return 'muscular'
    return 'unknown'

def load_injuries_data():
    """Load and analyze injuries data"""
    print("=" * 70)
    print("STEP 1: Analyzing Injuries Data File")
    print("=" * 70)
    
    injuries_path = 'original_data/20251106_injuries_data.xlsx'
    if not os.path.exists(injuries_path):
        injuries_path = '../original_data/20251106_injuries_data.xlsx'
    
    injuries_df = pd.read_excel(injuries_path, engine='openpyxl')
    injuries_df['fromDate'] = pd.to_datetime(injuries_df['fromDate'], errors='coerce')
    
    # Derive injury_class
    if 'injury_class' not in injuries_df.columns:
        injuries_df['injury_class'] = injuries_df.apply(
            lambda row: derive_injury_class(
                row.get('injury_type', ''),
                row.get('no_physio_injury', None)
            ),
            axis=1
        )
    
    # Filter to training period
    train_injuries = injuries_df[injuries_df['fromDate'] <= TRAIN_END].copy()
    
    print(f"\nTotal injuries in file: {len(injuries_df)}")
    print(f"Injuries <= {TRAIN_END.date()}: {len(train_injuries)}")
    print(f"\nInjury class distribution (all injuries):")
    print(injuries_df['injury_class'].value_counts())
    print(f"\nInjury class distribution (training period):")
    print(train_injuries['injury_class'].value_counts())
    
    # Filter to allowed classes
    allowed_train_injuries = train_injuries[
        train_injuries['injury_class'].str.lower().isin(ALLOWED_INJURY_CLASSES)
    ]
    print(f"\nAllowed injury classes in training period: {len(allowed_train_injuries)}")
    print(f"Expected timelines (×5): {len(allowed_train_injuries) * 5}")
    
    # Create mapping
    injury_class_map = {}
    for _, row in injuries_df.iterrows():
        player_id = row.get('player_id')
        from_date = row.get('fromDate')
        if pd.notna(player_id) and pd.notna(from_date):
            injury_class = row.get('injury_class', '').lower()
            injury_class_map[(int(player_id), pd.Timestamp(from_date).normalize())] = injury_class
    
    return injury_class_map, allowed_train_injuries

def analyze_daily_features_injuries(injury_class_map: Dict[Tuple[int, pd.Timestamp], str]):
    """Analyze injuries detected in daily features files"""
    print("\n" + "=" * 70)
    print("STEP 2: Analyzing Injuries in Daily Features Files")
    print("=" * 70)
    
    daily_features_dir = r'C:\Users\joao.henriques\IPM V3\daily_features_output'
    if not os.path.exists(daily_features_dir):
        daily_features_dir = 'daily_features_output'
        if not os.path.exists(daily_features_dir):
            daily_features_dir = '../daily_features_output'
    
    # Get all player files
    player_files = [f for f in os.listdir(daily_features_dir) 
                   if f.startswith('player_') and f.endswith('_daily_features.csv')]
    
    print(f"\nTotal daily features files: {len(player_files)}")
    
    # Statistics
    total_injuries_detected = 0
    matched_injuries = 0
    allowed_injuries = 0
    date_mismatches = []
    unmatched_injuries = []
    window_failures = 0
    reference_date_failures = 0
    
    # Sample a few players for detailed analysis
    sample_players = []
    
    # Analyze all players with progress tracking
    from tqdm import tqdm
    print(f"\nProcessing {len(player_files)} player files...")
    
    for filename in tqdm(player_files, desc="Analyzing players"):
        player_id = int(filename.split('_')[1])
        if player_id in GOALKEEPER_IDS:
            continue
        
        try:
            df = pd.read_csv(f'{daily_features_dir}/{filename}', low_memory=False)
            df['date'] = pd.to_datetime(df['date'])
            
            # Detect injury starts
            injury_starts = df[df['cum_inj_starts'] > df['cum_inj_starts'].shift(1)]
            total_injuries_detected += len(injury_starts)
            
            for _, injury_row in injury_starts.iterrows():
                injury_date = injury_row['date']
                injury_date_normalized = pd.Timestamp(injury_date).normalize()
                
                # Try to match with injury_class_map
                injury_class = injury_class_map.get((player_id, injury_date_normalized), '').lower()
                
                if not injury_class:
                    # Try nearby dates
                    found = False
                    for day_offset in [-1, 1, -2, 2, -3, 3]:
                        alt_date = injury_date_normalized + timedelta(days=day_offset)
                        alt_class = injury_class_map.get((player_id, alt_date), '').lower()
                        if alt_class:
                            injury_class = alt_class
                            date_mismatches.append((player_id, injury_date_normalized, alt_date, day_offset))
                            found = True
                            break
                    
                    if not found:
                        unmatched_injuries.append((player_id, injury_date_normalized))
                else:
                    matched_injuries += 1
                
                if injury_class in ALLOWED_INJURY_CLASSES:
                    allowed_injuries += 1
                    
                    # Check if we can generate timelines
                    can_generate = 0
                    for days_before in range(1, 6):
                        reference_date = injury_date - timedelta(days=days_before)
                        start_date = reference_date - timedelta(days=34)
                        
                        if start_date < df['date'].min():
                            window_failures += 1
                            continue
                        
                        # Check if reference date exists
                        if not (df['date'] == reference_date).any():
                            reference_date_failures += 1
                            continue
                        
                        # Check if we have complete 35-day window (simplified check)
                        window_data = df[(df['date'] >= start_date) & (df['date'] <= reference_date)]
                        if len(window_data) == 35:
                            can_generate += 1
                        else:
                            window_failures += 1
                    
                    if len(sample_players) < 10:
                        sample_players.append({
                            'player_id': player_id,
                            'injury_date': injury_date,
                            'injury_class': injury_class,
                            'can_generate': can_generate
                        })
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    print(f"\nInjuries detected in daily features (all players): {total_injuries_detected}")
    print(f"Matched with injury_class_map: {matched_injuries}")
    print(f"Unmatched (not in injury_class_map): {len(unmatched_injuries)}")
    print(f"Date mismatches (found with offset): {len(date_mismatches)}")
    print(f"Allowed injury classes: {allowed_injuries}")
    print(f"Expected timelines (×5): {allowed_injuries * 5}")
    print(f"Window validation failures: {window_failures}")
    print(f"Reference date failures: {reference_date_failures}")
    
    if date_mismatches:
        print(f"\nSample date mismatches (first 5):")
        for player_id, orig_date, matched_date, offset in date_mismatches[:5]:
            print(f"  Player {player_id}: {orig_date.date()} -> {matched_date.date()} (offset: {offset} days)")
    
    if unmatched_injuries:
        print(f"\nSample unmatched injuries (first 5):")
        for player_id, injury_date in unmatched_injuries[:5]:
            print(f"  Player {player_id}: {injury_date.date()}")
    
    if sample_players:
        print(f"\nSample players analysis:")
        for sample in sample_players[:5]:
            print(f"  Player {sample['player_id']}: {sample['injury_date'].date()}, "
                  f"class={sample['injury_class']}, can_generate={sample['can_generate']}/5 timelines")
    
    return total_injuries_detected, matched_injuries, allowed_injuries, date_mismatches, unmatched_injuries, window_failures, reference_date_failures

def analyze_reference_date_filtering(allowed_train_injuries: pd.DataFrame):
    """Analyze how reference date filtering affects timeline counts"""
    print("\n" + "=" * 70)
    print("STEP 3: Analyzing Reference Date Filtering")
    print("=" * 70)
    
    # For each injury, calculate which reference dates would be in training period
    timelines_in_train = 0
    timelines_outside_train = 0
    
    for _, injury in allowed_train_injuries.iterrows():
        injury_date = pd.Timestamp(injury['fromDate']).normalize()
        
        # Generate 5 reference dates (D-1 to D-5)
        for days_before in range(1, 6):
            reference_date = injury_date - timedelta(days=days_before)
            
            if TRAIN_START <= reference_date <= TRAIN_END:
                timelines_in_train += 1
            else:
                timelines_outside_train += 1
    
    print(f"\nTotal allowed injuries in training period: {len(allowed_train_injuries)}")
    print(f"Total possible timelines (×5): {len(allowed_train_injuries) * 5}")
    print(f"Reference dates in training period: {timelines_in_train}")
    print(f"Reference dates outside training period: {timelines_outside_train}")
    
    return timelines_in_train, timelines_outside_train

def main():
    print("DIAGNOSTIC ANALYSIS: Why only 675 injury timelines?")
    print("=" * 70)
    
    # Step 1: Analyze injuries data file
    injury_class_map, allowed_train_injuries = load_injuries_data()
    
    # Step 2: Analyze daily features
    total_detected, matched, allowed, date_mismatches, unmatched, window_failures, ref_failures = analyze_daily_features_injuries(injury_class_map)
    
    # Step 3: Reference date filtering
    timelines_in_train, timelines_outside = analyze_reference_date_filtering(allowed_train_injuries)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"1. Injuries in file (<= {TRAIN_END.date()}): {len(allowed_train_injuries)}")
    print(f"2. Expected timelines (×5): {len(allowed_train_injuries) * 5}")
    print(f"3. Reference dates in training period: {timelines_in_train}")
    print(f"4. Actual timelines generated: 675")
    print(f"\nGap: {timelines_in_train - 675} timelines missing")
    print(f"\nPossible reasons:")
    print(f"  - Date matching issues: {len(date_mismatches)} mismatches found")
    print(f"  - Unmatched injuries: {len(unmatched)} not found in injury_class_map")
    print(f"  - Window validation failures: {window_failures} (incomplete 35-day windows)")
    print(f"  - Reference date not in daily features data: {ref_failures}")

if __name__ == "__main__":
    main()

