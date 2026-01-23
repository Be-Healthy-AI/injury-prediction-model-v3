#!/usr/bin/env python3
"""Quick verification of PL membership for players with/without timelines."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "models_production" / "lgbm_muscular_v4" / "code" / "timelines"))

from create_35day_timelines_v4_enhanced import build_pl_clubs_per_season, build_player_pl_membership_periods, get_season_date_range

raw_path = ROOT_DIR / "production" / "raw_data" / "england" / "20260122"
match_dir = str(raw_path / "match_data")
career_file = str(raw_path / "players_career.csv")

print("Building PL membership data...")
pl_clubs = build_pl_clubs_per_season(match_dir)
periods = build_player_pl_membership_periods(career_file, pl_clubs)
season_start, season_end = get_season_date_range(2025)

players_with = [148367, 192279, 262749, 325443, 338424, 423440, 479999, 502821, 503987, 646750, 659813, 890719, 890721, 1187629]
players_without = [144028, 309400, 316264, 335721, 357662, 363205, 420243, 433177, 435338, 495666, 655488]

print("\nPlayers WITH timelines - PL membership:")
for p in players_with:
    has_membership = p in periods and any(ps <= season_end and pe >= season_start for ps, pe in periods[p])
    print(f"  {p}: {'✅ YES' if has_membership else '❌ NO'}")

print("\nPlayers WITHOUT timelines - PL membership:")
for p in players_without:
    has_membership = p in periods and any(ps <= season_end and pe >= season_start for ps, pe in periods[p])
    print(f"  {p}: {'✅ YES' if has_membership else '❌ NO'}")
