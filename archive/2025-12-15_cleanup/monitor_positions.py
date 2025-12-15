import pandas as pd
import os
import time
from datetime import datetime

output_file = 'daily_features_output/player_462250_daily_features.csv'

print("Monitoring for updated file...")
print("=" * 60)

if os.path.exists(output_file):
    initial_time = os.path.getmtime(output_file)
    initial_time_str = datetime.fromtimestamp(initial_time).strftime('%Y-%m-%d %H:%M:%S')
    print(f"Initial file timestamp: {initial_time_str}")
else:
    initial_time = 0
    print("File does not exist yet, waiting for creation...")

# Wait for file to be updated (check every 10 seconds, max 10 minutes)
max_wait = 600
waited = 0
check_interval = 10

while waited < max_wait:
    time.sleep(check_interval)
    waited += check_interval
    
    if os.path.exists(output_file):
        current_time = os.path.getmtime(output_file)
        if current_time > initial_time:
            print(f"\n✓ File updated! (after {waited} seconds)")
            print("=" * 60)
            
            df = pd.read_csv(output_file)
            print(f"Total rows: {len(df)}")
            print(f"\nSample positions from last_match_position column:")
            positions = df['last_match_position'].dropna().unique()[:30]
            for i, p in enumerate(positions, 1):
                print(f"  {i}. {p}")
            print(f"\nTotal unique positions: {df['last_match_position'].nunique()}")
            
            # Check for abbreviations that should have been normalized
            abbrevs = ['LW', 'AM', 'SS', 'CF', 'LB', 'RB', 'CB', 'CM', 'DM', 'RW', 'ST']
            found_abbrevs = [p for p in positions if p in abbrevs]
            if found_abbrevs:
                print(f"\n⚠ WARNING: Found abbreviations that should be normalized: {found_abbrevs}")
            else:
                print("\n✓ SUCCESS: No abbreviations found - all positions appear to be normalized!")
            
            # Check for normalized positions
            normalized_positions = ['Left Winger', 'Attacking Midfielder', 'Second Striker', 
                                  'Centre Forward', 'Left Back', 'Right Back', 'Centre Back',
                                  'Central Midfielder', 'Defensive Midfielder', 'Right Winger']
            found_normalized = [p for p in positions if p in normalized_positions]
            if found_normalized:
                print(f"\n✓ Found normalized positions: {found_normalized}")
            
            break
    else:
        print(f"Waiting for file... ({waited}s)", end='\r')
else:
    print(f"\nTimeout: File was not updated within {max_wait} seconds")



