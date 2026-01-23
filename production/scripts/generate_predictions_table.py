#!/usr/bin/env python3
"""
Generate predictions table for club players from the most recent predictions file.
Shows Date, Player, Id, Prediction, Risk Class, Body Part, Severity.
"""

import pandas as pd
import json
import argparse
import sys
import io
from pathlib import Path
from datetime import datetime

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Paths
PRODUCTION_ROOT = Path(__file__).resolve().parent.parent
INJURIES_DIR = PRODUCTION_ROOT / "raw_data" / "england"

def find_latest_predictions_file(predictions_dir: Path, model_version: str = "v2") -> Path:
    """Find the most recent predictions file for the specified model version."""
    if model_version == "v3":
        pattern = "predictions_lgbm_v3_*.csv"
    else:
        pattern = "predictions_lgbm_v2_*.csv"
    
    prediction_files = list(predictions_dir.glob(pattern))
    
    if not prediction_files:
        raise FileNotFoundError(f"No predictions files found in {predictions_dir} for model version {model_version}")
    
    # Sort by filename (which contains date) and get the latest
    latest_file = sorted(prediction_files, reverse=True)[0]
    return latest_file

def find_latest_injuries_file(injuries_dir: Path) -> Path:
    """Find the most recent injuries_data.csv file."""
    # Find all date folders
    date_folders = [d for d in injuries_dir.iterdir() if d.is_dir() and d.name.isdigit() and len(d.name) == 8]
    
    if not date_folders:
        raise FileNotFoundError(f"No date folders found in {injuries_dir}")
    
    # Sort by folder name (date) and get the latest
    latest_folder = sorted(date_folders, reverse=True)[0]
    injuries_file = latest_folder / "injuries_data.csv"
    
    if not injuries_file.exists():
        raise FileNotFoundError(f"Injuries file not found: {injuries_file}")
    
    return injuries_file

def get_injured_players(injuries_file: Path, target_date: pd.Timestamp) -> set:
    """Get set of player IDs who are currently injured on target_date."""
    injured_players = set()
    
    if not injuries_file.exists():
        print(f"WARNING: Injuries file not found: {injuries_file}")
        return injured_players
    
    try:
        df_inj = pd.read_csv(injuries_file, sep=';', encoding='utf-8-sig', low_memory=False)
        df_inj['fromDate'] = pd.to_datetime(df_inj['fromDate'], errors='coerce')
        df_inj['untilDate'] = pd.to_datetime(df_inj['untilDate'], errors='coerce')
        
        for _, row in df_inj.iterrows():
            player_id = row['player_id']
            injury_start = row['fromDate']
            injury_end = row['untilDate']
            
            if pd.isna(injury_start):
                continue
            
            # If untilDate is NaN, injury is ongoing
            if pd.isna(injury_end):
                if injury_start <= target_date:
                    injured_players.add(player_id)
            else:
                # Injury is active if target_date is between fromDate and untilDate
                if injury_start <= target_date <= injury_end:
                    injured_players.add(player_id)
    except Exception as e:
        print(f"WARNING: Error reading injuries file: {e}")
    
    return injured_players

def format_body_part(bp):
    """Format body part with proper capitalization."""
    if pd.isna(bp) or bp == '':
        return '-'
    return bp.replace('_', ' ').title()

def format_severity(sev):
    """Format severity with proper capitalization."""
    if pd.isna(sev) or sev == '':
        return '-'
    return sev.capitalize()

def main(country: str = "England", club: str = "Chelsea FC", target_date_str=None, model_version: str = "v2"):
    print("=" * 120)
    print(f"{club.upper()} PREDICTIONS TABLE")
    print("=" * 120)
    print()
    
    # Paths based on country and club
    config_path = PRODUCTION_ROOT / "deployments" / country / club / "config.json"
    predictions_dir = PRODUCTION_ROOT / "deployments" / country / club / "predictions"
    
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        return
    
    if not predictions_dir.exists():
        print(f"ERROR: Predictions directory not found: {predictions_dir}")
        return
    
    # Load player IDs
    with open(config_path, 'r') as f:
        config = json.load(f)
    player_ids = set(config['player_ids'])
    print(f"Loaded {len(player_ids)} {club} player IDs from config")
    print()
    
    # Find latest predictions file
    predictions_file = find_latest_predictions_file(predictions_dir, model_version)
    print(f"Using predictions file: {predictions_file.name} (Model: {model_version.upper()})")
    
    # Load predictions
    df_pred = pd.read_csv(predictions_file, parse_dates=['reference_date'], low_memory=False)
    print(f"Total predictions in file: {len(df_pred):,}")
    
    # Determine target date
    if target_date_str:
        target_date = pd.to_datetime(target_date_str)
        print(f"Target date specified: {target_date.date()}")
    else:
        target_date = df_pred['reference_date'].max()
        print(f"Using most recent date in file: {target_date.date()}")
    print()
    
    # Filter for target date and club players
    df_filtered = df_pred[
        (df_pred['reference_date'] == target_date) & 
        (df_pred['player_id'].isin(player_ids))
    ].copy()
    
    print(f"Predictions for {target_date.date()} ({club} players): {len(df_filtered)}")
    
    # Check for missing players
    predicted_player_ids = set(df_filtered['player_id'].unique())
    missing_players = player_ids - predicted_player_ids
    if missing_players:
        print(f"WARNING: Missing predictions for {len(missing_players)} players: {sorted(missing_players)}")
    print()
    
    # Load injuries data
    try:
        injuries_file = find_latest_injuries_file(INJURIES_DIR)
        print(f"Using injuries file: {injuries_file}")
        injured_players = get_injured_players(injuries_file, target_date)
        print(f"Found {len(injured_players)} currently injured players: {sorted(injured_players)}")
    except Exception as e:
        print(f"WARNING: Could not load injuries data: {e}")
        injured_players = set()
    print()
    
    # Create table data
    table_data = []
    for _, row in df_filtered.iterrows():
        player_id = row['player_id']
        risk_class = "Injured" if player_id in injured_players else row['risk_level']
        
        table_data.append({
            'Date': row['reference_date'].strftime('%Y-%m-%d'),
            'Player': row['player_name'],
            'Id': player_id,
            'Prediction': f"{row['injury_probability']*100:.2f}%",
            'Risk Class': risk_class,
            'Body Part': format_body_part(row['predicted_body_part']),
            'Severity': format_severity(row['predicted_severity'])
        })
    
    # Sort by player name
    table_data.sort(key=lambda x: x['Player'])
    
    # Create DataFrame and save to CSV
    table_df = pd.DataFrame(table_data)
    if model_version == "v3":
        output_file = predictions_dir / f"predictions_table_v3_{target_date.strftime('%Y%m%d')}.csv"
    else:
        output_file = predictions_dir / f"predictions_table_{target_date.strftime('%Y%m%d')}.csv"
    table_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    # Print table
    print("=" * 120)
    print(f"PREDICTIONS TABLE FOR {target_date.date()}")
    print("=" * 120)
    print()
    print(f"{'Date':<12} {'Player':<30} {'Id':<8} {'Prediction':<12} {'Risk Class':<12} {'Body Part':<15} {'Severity':<10}")
    print("-" * 120)
    
    for row in table_data:
        # Safely handle Unicode characters in player names
        try:
            player_name = str(row['Player'])
            date_str = str(row['Date'])
            id_str = str(row['Id'])
            pred_str = str(row['Prediction'])
            risk_str = str(row['Risk Class'])
            body_str = str(row['Body Part'])
            sev_str = str(row['Severity'])
            print(f"{date_str:<12} {player_name:<30} {id_str:<8} {pred_str:<12} {risk_str:<12} {body_str:<15} {sev_str:<10}")
        except (UnicodeEncodeError, UnicodeDecodeError):
            # Fallback: use ASCII-safe representation
            player_name = str(row['Player']).encode('ascii', errors='replace').decode('ascii')
            print(f"{row['Date']:<12} {player_name:<30} {row['Id']:<8} {row['Prediction']:<12} {row['Risk Class']:<12} {row['Body Part']:<15} {row['Severity']:<10}")
    
    print("-" * 120)
    print(f"Total: {len(table_data)} players")
    print()
    print(f"Table saved to: {output_file}")
    print("=" * 120)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate predictions table for club players')
    parser.add_argument('--country', type=str, default='England',
                        help='Country name (default: England)')
    parser.add_argument('--club', type=str, default='Chelsea FC',
                        help='Club name (default: Chelsea FC)')
    parser.add_argument('--date', type=str, default=None,
                        help='Target date in YYYY-MM-DD format (default: most recent date in predictions file)')
    parser.add_argument('--model-version', type=str, choices=['v2', 'v3'], default='v2',
                        help='Model version to use (default: v2)')
    args = parser.parse_args()
    
    main(country=args.country, club=args.club, target_date_str=args.date, model_version=args.model_version)


