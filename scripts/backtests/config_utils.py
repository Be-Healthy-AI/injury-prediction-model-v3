#!/usr/bin/env python3
"""Utility helpers for retrospective backtesting configuration."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd


def _to_timestamp(value) -> pd.Timestamp:
    """Convert incoming JSON values to pandas timestamps."""
    if isinstance(value, pd.Timestamp):
        return value
    if value is None or value == "":
        return pd.NaT
    return pd.to_datetime(value, errors="raise")


def create_entry_id(player_id: int, injury_date: pd.Timestamp | str) -> str:
    """Generate a reproducible identifier for a backtesting entry."""
    injury_ts = _to_timestamp(injury_date)
    if pd.isna(injury_ts):
        raise ValueError("Injury date is required to build entry id.")
    return f"player_{int(player_id)}_{injury_ts.strftime('%Y%m%d')}"


@dataclass(frozen=True)
class BacktestEntry:
    """Configuration describing a single backtesting injury window."""

    entry_id: str
    player_id: int
    injury_date: pd.Timestamp
    season: str | None
    injury_type: str | None
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    history_start: pd.Timestamp
    history_end: pd.Timestamp

    @property
    def injury_label(self) -> str:
        return self.injury_date.strftime("%Y%m%d")

    @property
    def window_label(self) -> str:
        return f"{self.window_start.strftime('%Y%m%d')}_{self.window_end.strftime('%Y%m%d')}"

    @property
    def history_label(self) -> str:
        return f"{self.history_start.strftime('%Y%m%d')}_{self.history_end.strftime('%Y%m%d')}"

    @property
    def daily_features_filename(self) -> str:
        return f"{self.entry_id}_daily_features.csv"

    @property
    def timelines_filename(self) -> str:
        return f"{self.entry_id}_timelines.csv"

    def predictions_filename(self, model_name: str) -> str:
        return f"{self.entry_id}_{model_name}_predictions.csv"

    @property
    def combined_predictions_filename(self) -> str:
        return f"{self.entry_id}_predictions_combined.csv"

    @property
    def chart_filename(self) -> str:
        return f"{self.entry_id}_probabilities.png"


def load_backtest_config(path: Path | str) -> List[BacktestEntry]:
    """Load a JSON backtesting configuration file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Backtesting config not found: {config_path}")

    data = json.loads(config_path.read_text(encoding="utf-8"))
    entries_data: Sequence[dict] = data.get("entries", [])
    if not entries_data:
        raise ValueError(f"No entries found in config: {config_path}")

    entries: List[BacktestEntry] = []
    for raw in entries_data:
        player_id = int(raw["player_id"])
        injury_date = _to_timestamp(raw.get("injury_date"))
        if pd.isna(injury_date):
            raise ValueError(f"Invalid injury_date for player {player_id}: {raw.get('injury_date')}")

        entry_id = raw.get("entry_id") or create_entry_id(player_id, injury_date)
        window_start = _to_timestamp(raw.get("window_start"))
        window_end = _to_timestamp(raw.get("window_end"))
        history_start = _to_timestamp(raw.get("history_start"))
        history_end = _to_timestamp(raw.get("history_end"))

        if pd.isna(window_start) or pd.isna(window_end):
            raise ValueError(f"Window bounds missing for entry {entry_id}")
        if pd.isna(history_start) or pd.isna(history_end):
            raise ValueError(f"History bounds missing for entry {entry_id}")

        entries.append(
            BacktestEntry(
                entry_id=entry_id,
                player_id=player_id,
                injury_date=injury_date,
                season=raw.get("season"),
                injury_type=raw.get("injury_type"),
                window_start=window_start,
                window_end=window_end,
                history_start=history_start,
                history_end=history_end,
            )
        )

    # Ensure deterministic ordering by injury date, player_id
    entries.sort(key=lambda e: (e.injury_date, e.player_id, e.entry_id))
    _ensure_unique_ids(entries)
    return entries


def _ensure_unique_ids(entries: Iterable[BacktestEntry]) -> None:
    seen: set[str] = set()
    for entry in entries:
        if entry.entry_id in seen:
            raise ValueError(f"Duplicated entry_id detected: {entry.entry_id}")
        seen.add(entry.entry_id)


__all__ = [
    "BacktestEntry",
    "create_entry_id",
    "load_backtest_config",
]

