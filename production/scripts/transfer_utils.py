#!/usr/bin/env python3
"""
Shared utilities for transfer handling in the V3 pipeline.

- get_pl_club_names(country): set of club folder names (PL clubs) for a country.
- get_club_stint_dates(career, club_name, normalize_fn): (transfer_in_date, transfer_out_date)
  for a player's stint at a given club from players_career.csv.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Set, Tuple

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent


def get_deployments_dir(country: str) -> Path:
    """Deployments root for country, e.g. production/deployments/England."""
    return PRODUCTION_ROOT / "deployments" / country


def get_pl_club_names(country: str) -> Set[str]:
    """
    Return set of club folder names (PL clubs) under deployments/{country}.
    Used to detect PL-to-PL transfers (current_club is in this set).
    """
    deployments_dir = get_deployments_dir(country)
    if not deployments_dir.exists():
        return set()
    clubs = set()
    for item in deployments_dir.iterdir():
        if item.is_dir() and (item / "config.json").exists():
            clubs.add(item.name)
    return clubs


def get_club_stint_dates(
    career: pd.DataFrame,
    club_name: str,
    normalize_fn: Callable[[str], str],
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Get transfer-in and transfer-out dates for a player's stint at the given club.

    career: DataFrame with columns Date, From, To (from players_career.csv).
    club_name: e.g. "Arsenal FC".
    normalize_fn: e.g. normalize_team_name, used to compare To/From with club_name.

    Returns:
        (transfer_in_date, transfer_out_date).
        transfer_in_date = date of the row where To == club (joined this club).
        transfer_out_date = date of the row where From == club (left this club), or None if still at club.
        Uses the *most recent* stint at this club (last chronological join).
    """
    if career is None or career.empty:
        return None, None
    if "Date" not in career.columns or "To" not in career.columns or "From" not in career.columns:
        return None, None

    career = career.copy()
    career["Date"] = pd.to_datetime(career["Date"], errors="coerce")
    career = career.dropna(subset=["Date"])
    if career.empty:
        return None, None

    # Sort ascending (oldest first) for chronological order
    career = career.sort_values("Date", ascending=True).reset_index(drop=True)
    club_norm = normalize_fn(club_name) if club_name else ""

    # Find the most recent stint at this club: last row where To == club
    transfer_in_date = None
    transfer_out_date = None
    for i, row in career.iterrows():
        to_val = row.get("To")
        from_val = row.get("From")
        date_val = row["Date"]
        if pd.isna(date_val):
            continue
        to_norm = normalize_fn(str(to_val).strip()) if pd.notna(to_val) else ""
        from_norm = normalize_fn(str(from_val).strip()) if pd.notna(from_val) else ""
        if to_norm == club_norm:
            transfer_in_date = pd.Timestamp(date_val).normalize()
            transfer_out_date = None  # reset; we might find a later stint
        if from_norm == club_norm:
            # Only count as transfer-out if moving to a different club (e.g. U21 -> first team same club is not a leave)
            if to_norm != club_norm:
                transfer_out_date = pd.Timestamp(date_val).normalize()

    return transfer_in_date, transfer_out_date
