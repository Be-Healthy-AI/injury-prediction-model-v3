#!/usr/bin/env python3
"""
Quick script to check injury counts in the injuries data file
"""

import pandas as pd
import os

# Load injuries data from 20251205 folder
injuries_path = 'data_exports/transfermarkt/england/20251205/injuries_data.csv'
if not os.path.exists(injuries_path):
    injuries_path = '../data_exports/transfermarkt/england/20251205/injuries_data.csv'

print("=" * 70)
print("INJURIES DATA FILE ANALYSIS")
print("=" * 70)

# Try CSV first (20251205 folder), then Excel (original_data)
if injuries_path.endswith('.csv'):
    injuries_df = pd.read_csv(injuries_path, sep=';', encoding='utf-8')
else:
    injuries_df = pd.read_excel(injuries_path, engine='openpyxl')
print(f"\nLoading from: {injuries_path}")
print(f"Total rows in file: {len(injuries_df)}")

# Check column names and sample values
print(f"\nColumn names: {list(injuries_df.columns)}")
if 'fromDate' in injuries_df.columns:
    print(f"\nSample fromDate values (first 5):")
    print(injuries_df['fromDate'].head())
    print(f"\nData type: {injuries_df['fromDate'].dtype}")
    
    # Try different date formats
    print("\nTrying different date parsing methods...")
    
    # Method 1: Try DD/MM/YYYY format (common in European data)
    try:
        injuries_df['fromDate_parsed'] = pd.to_datetime(injuries_df['fromDate'], format='%d/%m/%Y', errors='coerce')
        valid_count_1 = injuries_df['fromDate_parsed'].notna().sum()
        print(f"  Format '%d/%m/%Y': {valid_count_1} valid dates")
    except:
        print(f"  Format '%d/%m/%Y': Failed")
        valid_count_1 = 0
    
    # Method 2: Try YYYY-MM-DD format
    try:
        injuries_df['fromDate_parsed2'] = pd.to_datetime(injuries_df['fromDate'], format='%Y-%m-%d', errors='coerce')
        valid_count_2 = injuries_df['fromDate_parsed2'].notna().sum()
        print(f"  Format '%Y-%m-%d': {valid_count_2} valid dates")
    except:
        print(f"  Format '%Y-%m-%d': Failed")
        valid_count_2 = 0
    
    # Method 3: Auto-detect (no format specified)
    try:
        injuries_df['fromDate_parsed3'] = pd.to_datetime(injuries_df['fromDate'], errors='coerce')
        valid_count_3 = injuries_df['fromDate_parsed3'].notna().sum()
        print(f"  Auto-detect: {valid_count_3} valid dates")
    except:
        print(f"  Auto-detect: Failed")
        valid_count_3 = 0
    
    # Use the method that worked best
    if valid_count_1 >= valid_count_2 and valid_count_1 >= valid_count_3:
        injuries_df['fromDate'] = injuries_df['fromDate_parsed']
        print(f"\n✅ Using format '%d/%m/%Y' - {valid_count_1} valid dates")
    elif valid_count_2 >= valid_count_3:
        injuries_df['fromDate'] = injuries_df['fromDate_parsed2']
        print(f"\n✅ Using format '%Y-%m-%d' - {valid_count_2} valid dates")
    else:
        injuries_df['fromDate'] = injuries_df['fromDate_parsed3']
        print(f"\n✅ Using auto-detect - {valid_count_3} valid dates")
    
    # Clean up temporary columns
    injuries_df = injuries_df.drop(columns=[col for col in injuries_df.columns if col.startswith('fromDate_parsed')])
else:
    print("ERROR: 'fromDate' column not found!")
    print(f"Available columns: {list(injuries_df.columns)}")
    exit(1)

# Check for missing dates
missing_dates = injuries_df['fromDate'].isna().sum()
print(f"\nRows with missing/invalid fromDate: {missing_dates}")
print(f"Rows with valid fromDate: {len(injuries_df) - missing_dates}")

# Filter to valid dates only
valid_injuries = injuries_df[injuries_df['fromDate'].notna()].copy()
print(f"Rows with valid fromDate: {len(valid_injuries)}")

# Check injuries >= 2022-07-01 (including goalkeepers)
train_start = pd.Timestamp('2022-07-01')
injuries_after_start = valid_injuries[valid_injuries['fromDate'] >= train_start].copy()

print(f"\n" + "=" * 70)
print(f"Injuries >= {train_start.date()} (including goalkeepers)")
print("=" * 70)
print(f"Count: {len(injuries_after_start)}")

if len(injuries_after_start) > 0:
    print(f"\nDate range:")
    print(f"  Earliest: {injuries_after_start['fromDate'].min().date()}")
    print(f"  Latest: {injuries_after_start['fromDate'].max().date()}")
    
    # Show distribution by year
    injuries_after_start['year'] = injuries_after_start['fromDate'].dt.year
    print(f"\nDistribution by year:")
    print(injuries_after_start['year'].value_counts().sort_index())

# Also check <= 2024-06-30 for comparison
train_end = pd.Timestamp('2024-06-30')
injuries_in_train_period = valid_injuries[
    (valid_injuries['fromDate'] >= train_start) & 
    (valid_injuries['fromDate'] <= train_end)
].copy()

print(f"\n" + "=" * 70)
print(f"Injuries in training period ({train_start.date()} to {train_end.date()})")
print("=" * 70)
print(f"Count: {len(injuries_in_train_period)}")

# Check all injuries regardless of date
print(f"\n" + "=" * 70)
print("ALL INJURIES (any date)")
print("=" * 70)
print(f"Total with valid dates: {len(valid_injuries)}")
if len(valid_injuries) > 0:
    print(f"Date range (all):")
    print(f"  Earliest: {valid_injuries['fromDate'].min().date()}")
    print(f"  Latest: {valid_injuries['fromDate'].max().date()}")

# Derive injury_class and filter
print(f"\n" + "=" * 70)
print("INJURY CLASS ANALYSIS")
print("=" * 70)

def derive_injury_class(injury_type: str, no_physio_injury) -> str:
    """Derive injury_class from injury_type and no_physio_injury"""
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

# Derive injury_class if it doesn't exist
if 'injury_class' not in injuries_in_train_period.columns:
    print("Deriving injury_class from injury_type and no_physio_injury...")
    injuries_in_train_period['injury_class'] = injuries_in_train_period.apply(
        lambda row: derive_injury_class(
            row.get('injury_type', ''),
            row.get('no_physio_injury', None)
        ),
        axis=1
    )

# Show injury class distribution
print(f"\nInjury class distribution in training period:")
class_dist = injuries_in_train_period['injury_class'].value_counts()
print(class_dist)

# Filter to allowed classes (muscular, skeletal, unknown)
ALLOWED_INJURY_CLASSES = {'muscular', 'skeletal', 'unknown'}
allowed_injuries = injuries_in_train_period[
    injuries_in_train_period['injury_class'].str.lower().isin(ALLOWED_INJURY_CLASSES)
].copy()

print(f"\n" + "=" * 70)
print("FILTERED INJURIES (muscular, skeletal, unknown only)")
print("=" * 70)
print(f"Total injuries in training period: {len(injuries_in_train_period)}")
print(f"Excluded (other/no-injury): {len(injuries_in_train_period) - len(allowed_injuries)}")
print(f"Allowed injuries (muscular, skeletal, unknown): {len(allowed_injuries)}")
print(f"\nExpected timelines (×5 per injury): {len(allowed_injuries) * 5}")

