import pandas as pd
from pathlib import Path
from datetime import timedelta

# Debug Curtis Jones's injury mapping
injuries_file = Path(r"C:\Users\joao.henriques\IPM V3\models_production\lgbm_muscular_v4\data\raw_data\injuries_data.csv")
daily_features_file = Path(r"C:\Users\joao.henriques\IPM V3\models_production\lgbm_muscular_v4\data\daily_features\player_433188_daily_features.csv")

player_id = 433188
injury_date_expected = pd.Timestamp('2025-10-26')

print("="*80)
print("DEBUGGING CURTIS JONES INJURY (2025-10-26)")
print("="*80)

# Check injuries_data.csv
print("\n1. Checking injuries_data.csv:")
injuries_df = pd.read_csv(injuries_file, sep=';', encoding='utf-8-sig')

# Try different date parsing methods
original_dates = injuries_df['fromDate'].copy()

# Method 1: DD-MM-YYYY
injuries_df['fromDate_method1'] = pd.to_datetime(original_dates, format='%d-%m-%Y', errors='coerce')
# Method 2: DD/MM/YYYY
injuries_df['fromDate_method2'] = pd.to_datetime(original_dates, format='%d/%m/%Y', errors='coerce')
# Method 3: dayfirst=True
injuries_df['fromDate_method3'] = pd.to_datetime(original_dates, dayfirst=True, errors='coerce')

curtis_injuries = injuries_df[injuries_df['player_id'] == player_id].copy()

print(f"   Found {len(curtis_injuries)} injuries for Curtis Jones")
print(f"\n   Injuries around 2025-10-26:")

# Check injuries around the expected date
for _, row in curtis_injuries.iterrows():
    date_str = row['fromDate']
    method1 = row['fromDate_method1']
    method2 = row['fromDate_method2']
    method3 = row['fromDate_method3']
    injury_type = row.get('injury_type', '')
    injury_class = row.get('injury_class', '')
    
    # Check if any parsed date matches
    if (pd.notna(method1) and abs((method1 - injury_date_expected).days) <= 5) or \
       (pd.notna(method2) and abs((method2 - injury_date_expected).days) <= 5) or \
       (pd.notna(method3) and abs((method3 - injury_date_expected).days) <= 5):
        print(f"     Date string: {date_str}")
        print(f"       Method 1 (DD-MM-YYYY): {method1}")
        print(f"       Method 2 (DD/MM/YYYY): {method2}")
        print(f"       Method 3 (dayfirst=True): {method3}")
        print(f"       Injury type: {injury_type}")
        print(f"       Injury class: {injury_class}")
        print(f"       Expected: 2025-10-26")
        
        # Check which method gives the correct date
        if pd.notna(method1) and method1.date() == injury_date_expected.date():
            print(f"       [OK] Method 1 matches!")
        elif pd.notna(method2) and method2.date() == injury_date_expected.date():
            print(f"       [OK] Method 2 matches!")
        elif pd.notna(method3) and method3.date() == injury_date_expected.date():
            print(f"       [OK] Method 3 matches!")
        else:
            print(f"       [WARNING] None of the methods match exactly")

# Check daily features
print("\n2. Checking daily_features file:")
df_daily = pd.read_csv(daily_features_file)
df_daily['date'] = pd.to_datetime(df_daily['date'])

# Find injury starts around 2025-10-26
injury_starts = df_daily[df_daily['cum_inj_starts'] > df_daily['cum_inj_starts'].shift(1)]
injuries_around_date = injury_starts[
    (injury_starts['date'] >= injury_date_expected - timedelta(days=5)) &
    (injury_starts['date'] <= injury_date_expected + timedelta(days=5))
]

print(f"   Injury starts in daily features around 2025-10-26:")
if len(injuries_around_date) > 0:
    for _, row in injuries_around_date.iterrows():
        print(f"     Date: {row['date'].date()}, cum_inj_starts: {row['cum_inj_starts']}")
else:
    print(f"     No injury starts found around 2025-10-26")

# Check what the injury_class_map would contain
print("\n3. Testing injury_class_map lookup:")
print(f"   Testing lookup with date: {injury_date_expected.date()}")

# Simulate the date parsing used in load_injuries_data
injuries_df_test = pd.read_csv(injuries_file, sep=';', encoding='utf-8-sig')
original_dates_test = injuries_df_test['fromDate'].copy()

# Apply the same parsing logic as in the script
injuries_df_test['fromDate'] = pd.to_datetime(injuries_df_test['fromDate'], format='%d-%m-%Y', errors='coerce')
if injuries_df_test['fromDate'].isna().sum() > len(injuries_df_test) * 0.5:
    injuries_df_test['fromDate'] = pd.to_datetime(original_dates_test, format='%d/%m/%Y', errors='coerce')
if injuries_df_test['fromDate'].isna().sum() > len(injuries_df_test) * 0.5:
    injuries_df_test['fromDate'] = pd.to_datetime(original_dates_test, dayfirst=True, errors='coerce')

# Derive injury_class if needed
if 'injury_class' not in injuries_df_test.columns:
    def derive_injury_class(injury_type, no_physio_injury):
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
    
    injuries_df_test['injury_class'] = injuries_df_test.apply(
        lambda row: derive_injury_class(
            row.get('injury_type', ''),
            row.get('no_physio_injury', None)
        ),
        axis=1
    )

# Create injury_class_map
injury_class_map = {}
for _, row in injuries_df_test.iterrows():
    p_id = row.get('player_id')
    from_date = row.get('fromDate')
    injury_class = row.get('injury_class', '').lower() if pd.notna(row.get('injury_class')) else ''
    
    if pd.notna(p_id) and pd.notna(from_date):
        key = (int(p_id), pd.Timestamp(from_date).normalize())
        injury_class_map[key] = injury_class

# Test lookup
lookup_key = (player_id, injury_date_expected.normalize())
injury_class = injury_class_map.get(lookup_key, 'NOT_FOUND')

print(f"   Lookup key: {lookup_key}")
print(f"   Injury class found: '{injury_class}'")
print(f"   Expected: 'muscular'")

if injury_class == 'muscular':
    print(f"   [OK] Injury class mapping is correct!")
else:
    print(f"   [ERROR] Injury class mapping failed!")
    
    # Check what dates are in the map for this player
    print(f"\n   All injury dates in map for player {player_id}:")
    player_injuries_in_map = {k[1]: v for k, v in injury_class_map.items() if k[0] == player_id}
    for date, iclass in sorted(player_injuries_in_map.items()):
        if abs((date - injury_date_expected).days) <= 10:
            print(f"     {date.date()}: {iclass}")
