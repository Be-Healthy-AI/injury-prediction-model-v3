import pandas as pd
import os
import time

output_file = 'daily_features_output/player_462250_daily_features.csv'

# Wait for file to be created (max 5 minutes)
max_wait = 300
waited = 0
while not os.path.exists(output_file) and waited < max_wait:
    time.sleep(5)
    waited += 5
    print(f'Waiting for file... ({waited}s)')

if os.path.exists(output_file):
    df = pd.read_csv(output_file)
    print(f'\nFile exists! Total rows: {len(df)}')
    print(f'\nSample positions from last_match_position column:')
    positions = df['last_match_position'].dropna().unique()[:30]
    for i, p in enumerate(positions, 1):
        print(f'  {i}. {p}')
    print(f'\nTotal unique positions: {df["last_match_position"].nunique()}')
    
    # Check for abbreviations that should have been normalized
    abbrevs = ['LW', 'AM', 'SS', 'CF', 'LB', 'RB', 'CB', 'CM', 'DM', 'RW', 'ST']
    found_abbrevs = [p for p in positions if p in abbrevs]
    if found_abbrevs:
        print(f'\nWARNING: Found abbreviations that should be normalized: {found_abbrevs}')
    else:
        print('\n✓ No abbreviations found - positions appear to be normalized!')
else:
    print(f'File not found after {max_wait} seconds')



