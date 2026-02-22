#!/usr/bin/env python3
"""
TEST SCRIPT: Regenerate only rows from season_start (2025-07-01) onward; preserve rows before
that date from the existing file. Output file starts exactly as the one before.

- Preserve: all rows with date < season_start from the existing daily_features file (unchanged).
- Regenerate: only rows from season_start to max_date using raw data (with cumulative seeding
  from the last row before season_start).
- Output: preserved + regenerated, sorted by date. Original script (update_daily_features.py)
  is left unchanged for rollback.

Usage (single player, e.g. Arsenal FC 144028):
  python production/scripts/update_daily_features_from_season_start_test.py --country England --club "Arsenal FC" --player-id 144028 --data-date 20260129

Run from repo root (parent of production/). Requires existing daily_features file for the player.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure we can import production script from repo root
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent
ROOT_DIR = PRODUCTION_ROOT.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import numpy as np
from typing import Optional, Dict

# Import the main module (will be patched below)
import production.scripts.update_daily_features as base

logger = base.logger
_original_determine_calendar = base.determine_calendar


def _patched_determine_calendar(
    matches,
    injuries,
    reference_date,
    player_row=None,
    max_date=None,
    min_date=None,
):
    """Wrapper that applies min_date floor so calendar starts at season_start (e.g. 2025-07-01)."""
    calendar = _original_determine_calendar(
        matches, injuries, reference_date, player_row, max_date=max_date, min_date=None
    )
    if min_date is not None and len(calendar) > 0:
        min_date_norm = pd.Timestamp(min_date).normalize()
        if calendar[0] < min_date_norm:
            calendar = calendar[calendar >= min_date_norm]
            base.logger.info(
                f"[CALENDAR] Clamped start to min_date (season_start): {min_date_norm.date()}, {len(calendar)} days"
            )
    return calendar


def _club_as_of_date(career: pd.DataFrame, as_of: pd.Timestamp, normalize_fn) -> Optional[str]:
    """Return the club (To) from the last career row with Date <= as_of."""
    if career is None or career.empty or "Date" not in career.columns or "To" not in career.columns:
        return None
    c = career.copy()
    c["Date"] = pd.to_datetime(c["Date"], errors="coerce")
    c = c.dropna(subset=["Date"])
    if c.empty:
        return None
    c = c[c["Date"] <= pd.Timestamp(as_of)]
    if c.empty:
        return None
    last = c.sort_values("Date").iloc[-1]
    to_val = last.get("To")
    return str(to_val).strip() if pd.notna(to_val) else None


def _compute_seed_row_before_date(
    matches: pd.DataFrame,
    before_date: pd.Timestamp,
    career: pd.DataFrame,
    player_row: pd.Series,
    team_country_map: Dict[str, str],
) -> Optional[pd.Series]:
    """Build a seed row (state as of before_date) from matches before that date."""
    if matches.empty or "date_norm" not in matches.columns:
        return None
    before_norm = pd.Timestamp(before_date).normalize()
    m = matches[matches["date_norm"] < before_norm]
    if m.empty:
        return None

    # Aggregate career cumulatives from matches before before_date
    cum_minutes = int(m["minutes_played_numeric"].fillna(0).sum())
    cum_goals = int(m["goals_numeric"].fillna(0).sum())
    cum_assists = int(m["assists_numeric"].fillna(0).sum())
    cum_yellow = int(m["yellow_cards_numeric"].fillna(0).sum())
    cum_red = int(m["red_cards_numeric"].fillna(0).sum())
    cum_matches_played = int(m["matches_played"].fillna(0).sum())
    cum_matches_bench = int(m["matches_bench_unused"].fillna(0).sum())
    cum_matches_not_selected = int(m["matches_not_selected"].fillna(0).sum())
    cum_matches_injured = int(m["matches_injured"].fillna(0).sum())
    cum_disciplinary = int(m["disciplinary_action"].fillna(0).max()) if "disciplinary_action" in m.columns else 0

    last_match_date = m["date_norm"].max()

    # National team: matches where player's team is national
    national_matches_count = 0
    national_minutes_total = 0
    for _, row in m.iterrows():
        team = base.identify_player_team(row, None)
        if team and base.is_national_team(team):
            national_matches_count += 1
            national_minutes_total += int(row.get("minutes_played_numeric", 0) or 0)

    # Team results (wins/draws/losses)
    team_wins = 0
    team_draws = 0
    team_losses = 0
    if "result" in m.columns:
        for _, row in m.iterrows():
            res = str(row.get("result", "")).strip().lower()
            if "won" in res or res == "w":
                team_wins += 1
            elif "draw" in res or res == "d":
                team_draws += 1
            else:
                team_losses += 1
    elif "home_goals" in m.columns and "away_goals" in m.columns:
        for _, row in m.iterrows():
            try:
                h = int(row.get("home_goals", 0) or 0)
                a = int(row.get("away_goals", 0) or 0)
                if h > a:
                    team_wins += 1
                elif h < a:
                    team_losses += 1
                else:
                    team_draws += 1
            except (TypeError, ValueError):
                team_losses += 1

    # Club as of (before_date - 1) and club cumulatives for that club only
    current_club = _club_as_of_date(career, before_norm, base.normalize_team_name)
    club_cum_goals = 0
    club_cum_assists = 0
    club_cum_minutes = 0
    club_cum_matches = 0
    club_cum_yellow = 0
    club_cum_red = 0
    if current_club:
        club_norm = base.normalize_team_name(current_club)
        for _, row in m.iterrows():
            team = base.identify_player_team(row, current_club)
            if team and base.normalize_team_name(team) == club_norm:
                club_cum_goals += int(row.get("goals_numeric", 0) or 0)
                club_cum_assists += int(row.get("assists_numeric", 0) or 0)
                club_cum_minutes += int(row.get("minutes_played_numeric", 0) or 0)
                club_cum_matches += 1 if (row.get("matches_played", 0) or 0) else 0
                club_cum_yellow += int(row.get("yellow_cards_numeric", 0) or 0)
                club_cum_red += int(row.get("red_cards_numeric", 0) or 0)

    # Build Series with keys expected by calculate_match_features (existing_last_row)
    seed = pd.Series(dtype=object)
    seed["date"] = before_norm
    seed["cum_minutes_played_numeric"] = cum_minutes
    seed["cum_goals_numeric"] = cum_goals
    seed["cum_assists_numeric"] = cum_assists
    seed["cum_yellow_cards_numeric"] = cum_yellow
    seed["cum_red_cards_numeric"] = cum_red
    seed["cum_matches_played"] = cum_matches_played
    seed["cum_matches_bench"] = cum_matches_bench
    seed["cum_matches_not_selected"] = cum_matches_not_selected
    seed["cum_matches_injured"] = cum_matches_injured
    seed["cum_disciplinary_actions"] = cum_disciplinary
    seed["national_team_appearances"] = national_matches_count
    seed["national_team_minutes"] = national_minutes_total
    seed["days_since_last_national_match"] = (before_norm - last_match_date).days if pd.notna(last_match_date) else 0
    seed["cum_team_wins"] = team_wins
    seed["cum_team_draws"] = team_draws
    seed["cum_team_losses"] = team_losses
    seed["current_club"] = current_club or ""
    seed["club_cum_goals"] = club_cum_goals
    seed["club_cum_assists"] = club_cum_assists
    seed["club_cum_minutes"] = club_cum_minutes
    seed["club_cum_matches_played"] = club_cum_matches
    seed["club_cum_yellow_cards"] = club_cum_yellow
    seed["club_cum_red_cards"] = club_cum_red
    seed["days_since_last_match"] = (before_norm - last_match_date).days if pd.notna(last_match_date) else 0
    seed["senior_national_team"] = 1 if national_matches_count > 0 else 0
    return seed


def _patched_generate_daily_features_for_player(
    player_id: int,
    data_dir: str = None,
    reference_date: pd.Timestamp = None,
    output_dir: str = None,
    existing_file_path: Optional[str] = None,
    incremental: bool = False,
    max_date: Optional[pd.Timestamp] = None,
    season_start: Optional[pd.Timestamp] = None,
    club_name: Optional[str] = None,
) -> pd.DataFrame:
    """Preserve rows before season_start from existing file; regenerate only from season_start onward."""
    data_dir = data_dir or base.DATA_DIR
    reference_date = reference_date or base.REFERENCE_DATE
    output_dir = output_dir or base.OUTPUT_DIR

    base.logger.info(f"=== Generating daily features for player {player_id} (FROM SEASON START TEST) ===")
    data = base.load_player_data(player_id, data_dir)
    players = data["players"]
    injuries = data["injuries"]
    matches = data["matches"]
    career = data["career"]
    team_country_map = data["team_country_map"]

    if players.empty:
        base.logger.error(f"[ERROR] Player {player_id} not found")
        return pd.DataFrame()

    player_row = players.iloc[0]

    # Load existing file when not incremental and season_start set (for preserve + seed)
    existing_df = None
    preserved = None
    start_date_override = None
    season_start_norm = pd.Timestamp(season_start).normalize() if season_start is not None else None

    if incremental and existing_file_path and os.path.exists(existing_file_path):
        try:
            base.logger.info(f"[INCREMENTAL] Loading existing file: {existing_file_path}")
            existing_df = pd.read_csv(existing_file_path, parse_dates=["date"], encoding="utf-8-sig")
            if not existing_df.empty and "date" in existing_df.columns:
                if max_date is not None:
                    existing_df = existing_df[existing_df["date"] <= max_date].copy()
                if not existing_df.empty:
                    max_existing_date = pd.to_datetime(existing_df["date"].max()).normalize()
                    start_date_override = max_existing_date + pd.Timedelta(days=1)
                else:
                    existing_df = None
            else:
                existing_df = None
        except Exception as e:
            base.logger.warning(f"[INCREMENTAL] Error loading existing file: {e}")
            existing_df = None
    elif not incremental and season_start_norm is not None:
        # When regenerating from season_start only, we need the existing file for preserve+seed (base.main passes existing_file_path only when incremental)
        path_to_load = existing_file_path or (os.path.join(output_dir, f"player_{player_id}_daily_features.csv") if output_dir else None)
        if path_to_load and os.path.exists(path_to_load):
            try:
                existing_df = pd.read_csv(path_to_load, parse_dates=["date"], encoding="utf-8-sig")
                if not existing_df.empty and "date" in existing_df.columns:
                    existing_df["date_norm"] = pd.to_datetime(existing_df["date"]).dt.normalize()
                    preserved = existing_df[existing_df["date_norm"] < season_start_norm].copy()
                    if max_date is not None:
                        preserved = preserved[preserved["date_norm"] <= pd.Timestamp(max_date).normalize()].copy()
                    preserved = preserved.drop(columns=["date_norm"], errors="ignore")
                    base.logger.info(f"[PRESERVE] Keeping {len(preserved)} rows before {season_start_norm.date()} from existing file")
                else:
                    existing_df = None
            except Exception as e:
                base.logger.warning(f"[PRESERVE] Could not load existing file: {e}")
                existing_df = None

    matches = base.preprocess_matches(matches)

    calendar = _patched_determine_calendar(
        matches, injuries, reference_date, player_row, max_date=max_date, min_date=season_start
    )

    if start_date_override is not None:
        calendar = calendar[calendar >= start_date_override]
        if len(calendar) == 0:
            return existing_df if existing_df is not None else pd.DataFrame()

    transfer_out_date = None
    if club_name and base.get_club_stint_dates is not None and not career.empty:
        try:
            _, transfer_out_date = base.get_club_stint_dates(career, club_name, base.normalize_team_name)
            if transfer_out_date is not None:
                transfer_out_date = pd.Timestamp(transfer_out_date).normalize()
                calendar = calendar[calendar <= transfer_out_date]
                if preserved is not None and not preserved.empty:
                    preserved = preserved[preserved["date"] <= transfer_out_date].copy()
                if existing_df is not None and not existing_df.empty:
                    existing_df = existing_df[existing_df["date"] <= transfer_out_date].copy()
                if len(calendar) == 0 and (preserved is None or preserved.empty):
                    return existing_df if existing_df is not None else pd.DataFrame()
        except Exception as e:
            base.logger.warning(f"[TRANSFER] Could not get club stint dates: {e}")

    profile_df = base.calculate_profile_features(player_row, calendar, matches, career, team_country_map)

    existing_last_row = None
    existing_df_for_seed = None
    career_start_date = None

    if incremental and existing_df is not None and not existing_df.empty:
        existing_last_row = existing_df.iloc[-1]
        existing_df_for_seed = existing_df
        base.logger.debug(f"[INCREMENTAL] Using last row from existing file (date: {existing_last_row['date'].date()})")
        original_calendar = _original_determine_calendar(matches, injuries, reference_date, player_row)
        career_start_date = original_calendar[0] if len(original_calendar) > 0 else None
    elif not incremental and season_start_norm is not None and len(calendar) > 0:
        seed_cutoff = season_start_norm - pd.Timedelta(days=1)
        if existing_df is not None and not existing_df.empty:
            pre = existing_df[pd.to_datetime(existing_df["date"]).dt.normalize() <= seed_cutoff]
            if not pre.empty:
                existing_last_row = pre.iloc[-1]
                base.logger.info(
                    f"[SEED] Using row from existing file as of {existing_last_row['date'].date()} for cumulatives before {season_start_norm.date()}"
                )
        if existing_last_row is None and existing_file_path and os.path.exists(existing_file_path):
            try:
                seed_df = pd.read_csv(existing_file_path, parse_dates=["date"], encoding="utf-8-sig")
                if not seed_df.empty and "date" in seed_df.columns:
                    pre = seed_df[pd.to_datetime(seed_df["date"]).dt.normalize() <= seed_cutoff]
                    if not pre.empty:
                        existing_last_row = pre.iloc[-1]
            except Exception:
                pass
        if existing_last_row is None and not matches.empty and "date_norm" in matches.columns:
            existing_last_row = _compute_seed_row_before_date(
                matches, seed_cutoff, career, player_row, team_country_map
            )
            if existing_last_row is not None:
                base.logger.info(f"[SEED] Computed seed row from matches before {season_start_norm.date()} for cumulatives")
        full_calendar = _original_determine_calendar(
            matches, injuries, reference_date, player_row, max_date=max_date, min_date=None
        )
        career_start_date = full_calendar[0] if len(full_calendar) > 0 else None
        if career_start_date is not None:
            base.logger.debug(f"[SEED] career_start_date set to {career_start_date.date()} for frequency features")

    match_df = base.calculate_match_features(
        matches, calendar, player_row, team_country_map,
        existing_last_row, existing_df_for_seed, career_start_date
    )
    injury_df = base.calculate_injury_features(injuries, calendar, career_start_date)
    interaction_df = base.calculate_interaction_features(profile_df, match_df, injury_df)

    regenerated = pd.concat([profile_df, match_df, injury_df, interaction_df], axis=1)
    regenerated = regenerated.loc[:, ~regenerated.columns.duplicated()]

    column_rename_map = {
        "lower_leg_injury_count": "lower_leg_injuries",
        "knee_injury_count": "knee_injuries",
        "upper_leg_injury_count": "upper_leg_injuries",
        "hip_injury_count": "hip_injuries",
        "upper_body_injury_count": "upper_body_injuries",
        "head_injury_count": "head_injuries",
        "other_injury_count": "other_injuries",
    }
    rename_map = {k: v for k, v in column_rename_map.items() if k in regenerated.columns}
    if rename_map:
        regenerated = regenerated.rename(columns=rename_map)

    if transfer_out_date is not None and not regenerated.empty and "date" in regenerated.columns:
        regenerated = regenerated[regenerated["date"] <= transfer_out_date].copy()

    # Incremental: merge existing + new (same as original)
    if incremental and existing_df is not None and not existing_df.empty:
        base.logger.info(f"[INCREMENTAL] Merging {len(regenerated)} new rows with {len(existing_df)} existing rows")
        existing_cols = set(existing_df.columns)
        new_cols = set(regenerated.columns)
        for col in new_cols - existing_cols:
            existing_df[col] = pd.NA
        for col in existing_cols - new_cols:
            regenerated[col] = pd.NA
        regenerated = regenerated[existing_df.columns]
        combined = pd.concat([existing_df, regenerated], ignore_index=True)
        combined = combined.sort_values("date").drop_duplicates(subset=["date"], keep="first").reset_index(drop=True)
        if transfer_out_date is not None and not combined.empty and "date" in combined.columns:
            combined = combined[combined["date"] <= transfer_out_date].copy()
        return combined

    # Regenerate from season_start only: output = preserved + regenerated
    if preserved is not None and not preserved.empty:
        base.logger.info(f"[PRESERVE+REGENERATE] Combining {len(preserved)} preserved rows + {len(regenerated)} regenerated rows")
        reg_cols = set(regenerated.columns)
        pres_cols = set(preserved.columns)
        for col in reg_cols - pres_cols:
            preserved[col] = pd.NA
        for col in pres_cols - reg_cols:
            regenerated[col] = pd.NA
        preserved = preserved.reindex(columns=regenerated.columns, fill_value=pd.NA)
        combined = pd.concat([preserved, regenerated], ignore_index=True)
        combined = combined.sort_values("date").drop_duplicates(subset=["date"], keep="first").reset_index(drop=True)
        if transfer_out_date is not None and not combined.empty and "date" in combined.columns:
            combined = combined[combined["date"] <= transfer_out_date].copy()
        base.logger.info(f"[PRESERVE+REGENERATE] Output: {len(combined)} rows (from {combined['date'].min().date()} to {combined['date'].max().date()})")
        return combined

    return regenerated


def main():
    base.determine_calendar = _patched_determine_calendar
    base.generate_daily_features_for_player = _patched_generate_daily_features_for_player
    base.main()


if __name__ == "__main__":
    main()
