import pandas as pd
import json
from pathlib import Path
from collections import Counter
from datetime import datetime

# Load player IDs from config
config_path = Path('production/deployments/England/Chelsea FC/config.json')
with open(config_path, 'r') as f:
    config = json.load(f)
player_ids = config['player_ids']

# Directory with prediction files
predictions_dir = Path('production/deployments/England/Chelsea FC/predictions/players')

# Load injuries data to check for currently injured players
latest_raw_data = Path('production/raw_data/england/20251217')
injuries_file = latest_raw_data / 'injuries_data.csv'

injuries_df = None
if injuries_file.exists():
    injuries_df = pd.read_csv(injuries_file, sep=';', encoding='utf-8-sig', low_memory=False)
    injuries_df['fromDate'] = pd.to_datetime(injuries_df['fromDate'], errors='coerce')
    injuries_df['untilDate'] = pd.to_datetime(injuries_df['untilDate'], errors='coerce')

# Target date
target_date = pd.Timestamp('2025-12-17')

def is_player_injured(player_id: int, target_date: pd.Timestamp) -> bool:
    """Check if a player is currently injured on the target date."""
    if injuries_df is None:
        return False
    
    player_injuries = injuries_df[injuries_df['player_id'] == player_id].copy()
    if player_injuries.empty:
        return False
    
    # Check if any injury is active on the target date
    for _, row in player_injuries.iterrows():
        injury_start = row['fromDate']
        injury_end = row['untilDate']
        
        if pd.isna(injury_start):
            continue
        
        # If untilDate is NaN, injury is ongoing
        if pd.isna(injury_end):
            if injury_start <= target_date:
                return True
        else:
            # Injury is active if target_date is between fromDate and untilDate
            if injury_start <= target_date <= injury_end:
                return True
    
    return False

# Collect risk levels for 2025-12-17
risk_levels = []
player_details = []

for player_id in player_ids:
    # Try to find the prediction file (use latest available)
    # First try 20251219, then fallback to 20251218
    file_path = predictions_dir / f'player_{player_id}_predictions_20251219.csv'
    if not file_path.exists():
        file_path = predictions_dir / f'player_{player_id}_predictions_20251218.csv'
    
    if file_path.exists():
        df = pd.read_csv(file_path)
        # Filter for 2025-12-17
        latest = df[df['reference_date'] == '2025-12-17']
        
        if not latest.empty:
            row = latest.iloc[0]
            probability = row['injury_probability']
            player_name = row.get('player_name', f'Player {player_id}')
            
            # Check if player is currently injured
            if is_player_injured(player_id, target_date):
                risk_level = 'Injured'
            else:
                # Use the existing risk_level from predictions
                risk_level = row['risk_level']
            
            risk_levels.append(risk_level)
            player_details.append({
                'player_id': player_id,
                'player_name': player_name,
                'risk_level': risk_level,
                'probability': probability,
                'is_injured': is_player_injured(player_id, target_date)
            })

# Count distribution
dist = Counter(risk_levels)

# Print results
print('=' * 60)
print('Risk Distribution for Chelsea Players (2025-12-17)')
print('=' * 60)
print()

# Print in order: Injured, Very High, High, Medium, Low
for level in ['Injured', 'Very High', 'High', 'Medium', 'Low']:
    count = dist.get(level, 0)
    pct = (count / len(risk_levels) * 100) if risk_levels else 0
    print(f'{level:12s}: {count:2d} players ({pct:5.1f}%)')

print()
print('=' * 60)
print(f'Total: {len(risk_levels)} players')
print('=' * 60)
print()

# Optional: Show players by risk level
print('\nPlayers by Risk Level:')
print('-' * 60)
for level in ['Injured', 'Very High', 'High', 'Medium', 'Low']:
    players_in_level = [p for p in player_details if p['risk_level'] == level]
    if players_in_level:
        print(f'\n{level}:')
        # Sort by probability (descending) for non-injured, or by name for injured
        if level == 'Injured':
            sorted_players = sorted(players_in_level, key=lambda x: x['player_name'])
        else:
            sorted_players = sorted(players_in_level, key=lambda x: x['probability'], reverse=True)
        
        for p in sorted_players:
            if level == 'Injured':
                print(f"  - {p['player_name']} (ID: {p['player_id']})")
            else:
                print(f"  - {p['player_name']} (ID: {p['player_id']}, Prob: {p['probability']:.3f})")

