#!/usr/bin/env python3
"""
Fix daily features files that extend beyond a player's natural career end date.

This script:
1. Loads raw data from original_data/
2. Reuses the same preprocessing logic as create_daily_features_v3.py
3. Detects players whose daily features extend past their computed career end
4. Regenerates only those players' files with the correct end date cap
"""

import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.create_daily_features_v3 import (
    load_data_with_cache,
    preprocess_data_optimized,
    preprocess_career_data,
    initialize_team_country_map,
    initialize_competition_type_map,
    generate_daily_features_for_player_enhanced,
)

TARGET_END_DATE = pd.Timestamp('2025-11-09').normalize()
OUTPUT_DIR = ROOT_DIR / 'daily_features_output'


def compute_career_end_date(player_matches: pd.DataFrame,
                            player_physio_injuries: pd.DataFrame,
                            player_non_physio_injuries: pd.DataFrame) -> pd.Timestamp:
    """Compute the natural career end date based on matches and injuries."""
    match_end_date = None
    injury_end_date = None

    if not player_matches.empty:
        last_match_date = player_matches['date'].max()
        season_end_year = last_match_date.year + (1 if last_match_date.month >= 7 else 0)
        match_end_date = pd.Timestamp(f'{season_end_year}-06-30')

    all_injuries = pd.concat(
        [player_physio_injuries, player_non_physio_injuries],
        ignore_index=True
    ) if not player_physio_injuries.empty or not player_non_physio_injuries.empty else pd.DataFrame()

    if not all_injuries.empty and 'fromDate' in all_injuries.columns:
        latest_injury_date = all_injuries['fromDate'].max()
        injury_end_year = latest_injury_date.year + (1 if latest_injury_date.month >= 7 else 0)
        injury_end_date = pd.Timestamp(f'{injury_end_year}-06-30')

    candidates = [d for d in [match_end_date, injury_end_date] if d is not None]
    if candidates:
        return max(candidates)
    # Fallback for players without matches or injuries
    return pd.Timestamp('2025-06-30')


def main():
    if not OUTPUT_DIR.exists():
        print(f"❌ Output directory not found: {OUTPUT_DIR}")
        return

    print("📂 Loading cached data...")
    data = load_data_with_cache()
    players, injuries, matches = data['players'], data['injuries'], data['matches']
    teams = data.get('teams')
    competitions = data.get('competitions')
    initialize_team_country_map(teams)
    initialize_competition_type_map(competitions)
    career = preprocess_career_data(data.get('career'))

    print("🔧 Preprocessing data...")
    players, physio_injuries, non_physio_injuries, matches = preprocess_data_optimized(players, injuries, matches)

    files = sorted(OUTPUT_DIR.glob('player_*_daily_features.csv'))
    if not files:
        print(f"❌ No daily feature files found in {OUTPUT_DIR}")
        return

    fixed_players = []
    skipped_players = []

    print(f"🔍 Checking {len(files)} daily feature files for truncation issues...")

    for file_path in tqdm(files, desc="Inspecting files"):
        try:
            player_id = int(file_path.stem.split('_')[1])
        except (IndexError, ValueError):
            tqdm.write(f"⚠️  Skipping unrecognized file name: {file_path.name}")
            continue

        df = pd.read_csv(file_path, parse_dates=['date'])
        if df.empty:
            tqdm.write(f"⚠️  Empty daily features file for player {player_id}, skipping")
            continue

        last_date = df['date'].max()

        player_matches = matches[matches['player_id'] == player_id].copy()
        player_physio = physio_injuries[physio_injuries['player_id'] == player_id].copy()
        player_non_physio = non_physio_injuries[non_physio_injuries['player_id'] == player_id].copy()

        career_end_date = compute_career_end_date(player_matches, player_physio, player_non_physio)
        allowed_end_date = min(career_end_date, TARGET_END_DATE)

        if last_date <= allowed_end_date:
            skipped_players.append(player_id)
            continue

        tqdm.write(f"🛠️  Truncating player {player_id}: last date {last_date.date()} > allowed {allowed_end_date.date()}")

        player_row = players[players['id'] == player_id]
        if player_row.empty:
            tqdm.write(f"   ❌ Player {player_id} not found in players dataset, skipping")
            continue
        player_row = player_row.iloc[0]

        player_career = None
        if career is not None:
            id_col = 'player_id' if 'player_id' in career.columns else ('id' if 'id' in career.columns else None)
            if id_col is not None:
                player_career = career[career[id_col] == player_id].copy()

        daily_features = generate_daily_features_for_player_enhanced(
            player_id=player_id,
            player_row=player_row,
            player_matches=player_matches,
            player_physio_injuries=player_physio,
            player_non_physio_injuries=player_non_physio,
            player_career=player_career,
            global_end_date_cap=allowed_end_date
        )

        if daily_features is None or daily_features.empty:
            tqdm.write(f"   ❌ Failed to regenerate player {player_id}, skipping")
            continue

        daily_features.to_csv(file_path, index=False, encoding='utf-8-sig')
        fixed_players.append(player_id)

    print("\n📊 FIX SUMMARY")
    print("======================================================================")
    print(f"✅ Fixed players: {len(fixed_players)}")
    if fixed_players:
        print(f"   IDs: {fixed_players}")
    print(f"⏭️  Already correct: {len(skipped_players)} players")
    print("======================================================================")


if __name__ == "__main__":
    main()

