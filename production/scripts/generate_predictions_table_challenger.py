#!/usr/bin/env python3
"""
Generate predictions table for V4 challenger pipeline (reads/writes ONLY challenger paths).

Shows Date, Player, Id, Muscular (LGBM) %, Muscular (GB) %, Skeletal %, Risk Class, Body Part, Severity.
Supports 3-model CSV (injury_probability_muscular_lgbm, _muscular_gb, _skeletal) with fallback to
legacy single injury_probability.
- Config: production/deployments/{country}/challenger/{club}/config.json
- Predictions: challenger/{club}/predictions/predictions_lgbm_v4_*.csv
- Output: challenger/{club}/predictions_table_v4_{date}.csv

Does not touch V3 paths.
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


def get_challenger_path(country: str, club: str) -> Path:
    """Challenger club path: production/deployments/{country}/challenger/{club}. Never touch V3."""
    return PRODUCTION_ROOT / "deployments" / country / "challenger" / club


def find_latest_predictions_file(predictions_dir: Path) -> Path:
    """Find the most recent V4 predictions file."""
    pattern = "predictions_lgbm_v4_*.csv"
    prediction_files = list(predictions_dir.glob(pattern))
    if not prediction_files:
        raise FileNotFoundError(f"No V4 predictions files found in {predictions_dir}")
    return sorted(prediction_files, reverse=True)[0]


def find_latest_injuries_file(injuries_dir: Path) -> Path:
    """Find the most recent injuries_data.csv file."""
    date_folders = [d for d in injuries_dir.iterdir() if d.is_dir() and d.name.isdigit() and len(d.name) == 8]
    if not date_folders:
        raise FileNotFoundError(f"No date folders found in {injuries_dir}")
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
            if pd.isna(injury_end):
                if injury_start <= target_date:
                    injured_players.add(player_id)
            else:
                if injury_start <= target_date <= injury_end:
                    injured_players.add(player_id)
    except Exception as e:
        print(f"WARNING: Error reading injuries file: {e}")
    return injured_players


def format_body_part(bp):
    if pd.isna(bp) or bp == '':
        return '-'
    return bp.replace('_', ' ').title()


def format_severity(sev):
    if pd.isna(sev) or sev == '':
        return '-'
    return sev.capitalize()


def main(country: str = "England", club: str = "Chelsea FC", target_date_str=None):
    print("=" * 120)
    print(f"{club.upper()} PREDICTIONS TABLE (V4 CHALLENGER)")
    print("=" * 120)
    print()

    challenger_path = get_challenger_path(country, club)
    config_path = challenger_path / "config.json"
    predictions_dir = challenger_path / "predictions"

    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        return 1

    if not predictions_dir.exists():
        print(f"ERROR: Predictions directory not found: {predictions_dir}")
        return 1

    with open(config_path, 'r') as f:
        config = json.load(f)
    player_ids = set(config['player_ids'])
    print(f"Loaded {len(player_ids)} {club} player IDs from challenger config")
    print()

    predictions_file = find_latest_predictions_file(predictions_dir)
    print(f"Using predictions file: {predictions_file.name} (V4)")

    df_pred = pd.read_csv(predictions_file, parse_dates=['reference_date'], low_memory=False)
    print(f"Total predictions in file: {len(df_pred):,}")

    if target_date_str:
        target_date = pd.to_datetime(target_date_str)
        print(f"Target date specified: {target_date.date()}")
    else:
        target_date = df_pred['reference_date'].max()
        print(f"Using most recent date in file: {target_date.date()}")
    print()

    df_filtered = df_pred[
        (df_pred['reference_date'] == target_date) &
        (df_pred['player_id'].isin(player_ids))
    ].copy()

    print(f"Predictions for {target_date.date()} ({club} players): {len(df_filtered)}")

    predicted_player_ids = set(df_filtered['player_id'].unique())
    missing_players = player_ids - predicted_player_ids
    if missing_players:
        print(f"WARNING: Missing predictions for {len(missing_players)} players: {sorted(missing_players)}")
    print()

    try:
        injuries_file = find_latest_injuries_file(INJURIES_DIR)
        print(f"Using injuries file: {injuries_file}")
        injured_players = get_injured_players(injuries_file, target_date)
        print(f"Found {len(injured_players)} currently injured players: {sorted(injured_players)}")
    except Exception as e:
        print(f"WARNING: Could not load injuries data: {e}")
        injured_players = set()
    print()

    def fmt_pct(val):
        if pd.isna(val):
            return "-"
        try:
            return f"{float(val)*100:.2f}%"
        except (TypeError, ValueError):
            return "-"

    table_data = []
    for _, row in df_filtered.iterrows():
        player_id = row['player_id']
        risk_class = "Injured" if player_id in injured_players else row.get('risk_level', '-')
        table_data.append({
            'Date': row['reference_date'].strftime('%Y-%m-%d'),
            'Player': row.get('player_name', player_id),
            'Id': player_id,
            'Muscular (LGBM) %': fmt_pct(row.get('injury_probability_muscular_lgbm', row.get('injury_probability'))),
            'Muscular (GB) %': fmt_pct(row.get('injury_probability_muscular_gb')),
            'Skeletal %': fmt_pct(row.get('injury_probability_skeletal')),
            'Risk Class': risk_class,
            'Body Part': format_body_part(row.get('predicted_body_part', '')),
            'Severity': format_severity(row.get('predicted_severity', ''))
        })

    table_data.sort(key=lambda x: x['Player'])

    table_df = pd.DataFrame(table_data)
    output_file = predictions_dir / f"predictions_table_v4_{target_date.strftime('%Y%m%d')}.csv"
    table_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print("=" * 140)
    print(f"PREDICTIONS TABLE FOR {target_date.date()} (V4 - 3 models)")
    print("=" * 140)
    print()
    header = f"{'Date':<12} {'Player':<28} {'Id':<6} {'Musc(LGBM)%':<12} {'Musc(GB)%':<12} {'Skeletal%':<12} {'Risk Class':<12} {'Body Part':<12} {'Severity':<8}"
    print(header)
    print("-" * 140)

    for row in table_data:
        try:
            player_name = str(row['Player'])
            date_str = str(row['Date'])
            id_str = str(row['Id'])
            m_lgbm = str(row['Muscular (LGBM) %'])
            m_gb = str(row['Muscular (GB) %'])
            sk = str(row['Skeletal %'])
            risk_str = str(row['Risk Class'])
            body_str = str(row['Body Part'])
            sev_str = str(row['Severity'])
            print(f"{date_str:<12} {player_name:<28} {id_str:<6} {m_lgbm:<12} {m_gb:<12} {sk:<12} {risk_str:<12} {body_str:<12} {sev_str:<8}")
        except (UnicodeEncodeError, UnicodeDecodeError):
            player_name = str(row['Player']).encode('ascii', errors='replace').decode('ascii')
            print(f"{row['Date']:<12} {player_name:<28} {row['Id']:<6} {row['Muscular (LGBM) %']:<12} {row['Muscular (GB) %']:<12} {row['Skeletal %']:<12} {row['Risk Class']:<12} {row['Body Part']:<12} {row['Severity']:<8}")

    print("-" * 120)
    print(f"Total: {len(table_data)} players")
    print()
    print(f"Table saved to: {output_file}")
    print("=" * 120)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate predictions table for V4 challenger (challenger paths only)')
    parser.add_argument('--country', type=str, default='England', help='Country name (default: England)')
    parser.add_argument('--club', type=str, default='Chelsea FC', help='Club name (default: Chelsea FC)')
    parser.add_argument('--date', type=str, default=None, help='Target date YYYY-MM-DD (default: most recent in predictions file)')
    args = parser.parse_args()
    sys.exit(main(country=args.country, club=args.club, target_date_str=args.date))
