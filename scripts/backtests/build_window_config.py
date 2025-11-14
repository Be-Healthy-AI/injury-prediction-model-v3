#!/usr/bin/env python3
"""Build backtesting configuration for 45-day retrospective windows."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd

import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.backtests.config_utils import create_entry_id  # noqa: E402


DEFAULT_SEASON = "25/26"
DEFAULT_OUTPUT = Path("backtests") / "config" / "players_2025_45d.json"
INJURIES_FILE = Path("original_data") / "20251106_injuries_data.xlsx"

# Additional players to always include (player_id -> iterable of injury dates)
BASELINE_INJURIES: Dict[int, Sequence[str]] = {
    452607: ("2025-02-09",),
    699592: ("2025-02-09",),
    258027: ("2025-10-29",),
    8198: ("2025-05-11",),
    200512: ("2024-05-31",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--season",
        default=DEFAULT_SEASON,
        help="Season code (e.g., '25/26') to include in the config.",
    )
    parser.add_argument(
        "--injuries-file",
        type=Path,
        default=INJURIES_FILE,
        help="Path to the injuries Excel dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination JSON file for the generated configuration.",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=45,
        help="Number of days in the retrospective prediction window (excluding the injury day).",
    )
    parser.add_argument(
        "--history-days",
        type=int,
        default=80,
        help=(
            "Number of days of daily features to retain per injury (must be >= window-days). "
            "This ensures enough history to build 35-day timelines for the earliest reference date."
        ),
    )
    parser.add_argument(
        "--include-baseline",
        action="store_true",
        help="Force inclusion of the predefined baseline players regardless of season filtering.",
    )
    return parser.parse_args()


def load_injuries(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Injuries file not found: {path}")
    df = pd.read_excel(path, parse_dates=["fromDate", "untilDate"])
    df = df.rename(columns={"fromDate": "from_date", "untilDate": "until_date"})
    df = df.dropna(subset=["player_id", "from_date"])
    return df


def iter_target_rows(
    df: pd.DataFrame,
    season: str,
    include_baseline: bool,
) -> Iterable[pd.Series]:
    season_mask = df["season"] == season
    filtered = df.loc[season_mask].copy()

    if include_baseline:
        baseline_rows: List[pd.Series] = []
        for player_id, injury_dates in BASELINE_INJURIES.items():
            for injury_date in injury_dates:
                injury_ts = pd.to_datetime(injury_date, errors="coerce")
                if pd.isna(injury_ts):
                    raise ValueError(f"Invalid baseline injury date: {injury_date}")
                match = df[
                    (df["player_id"] == player_id) & (df["from_date"] == injury_ts)
                ]
                if match.empty:
                    raise ValueError(
                        f"Baseline injury not found in dataset for player {player_id} on {injury_date}"
                    )
                baseline_rows.append(match.iloc[0])

        # Append and drop duplicates based on player/date
        if baseline_rows:
            baseline_df = pd.DataFrame(baseline_rows)
            filtered = pd.concat([filtered, baseline_df], ignore_index=True)
            filtered = filtered.drop_duplicates(subset=["player_id", "from_date"])

    # Sort for determinism
    filtered.sort_values(["from_date", "player_id"], inplace=True)
    for _, row in filtered.iterrows():
        yield row


def build_entry_payload(
    row: pd.Series,
    window_days: int,
    history_days: int,
) -> dict:
    if history_days < window_days:
        raise ValueError("history_days must be greater than or equal to window_days")

    injury_date = pd.to_datetime(row["from_date"])
    window_end = injury_date - pd.Timedelta(days=1)
    window_start = window_end - pd.Timedelta(days=window_days - 1)
    history_end = window_end
    history_start = history_end - pd.Timedelta(days=history_days - 1)

    entry_id = create_entry_id(int(row["player_id"]), injury_date)
    payload = {
        "entry_id": entry_id,
        "player_id": int(row["player_id"]),
        "season": row.get("season"),
        "injury_type": row.get("injury_type"),
        "injury_date": injury_date.strftime("%Y-%m-%d"),
        "window_start": window_start.strftime("%Y-%m-%d"),
        "window_end": window_end.strftime("%Y-%m-%d"),
        "history_start": history_start.strftime("%Y-%m-%d"),
        "history_end": history_end.strftime("%Y-%m-%d"),
    }
    return payload


def main() -> None:
    args = parse_args()
    injuries = load_injuries(args.injuries_file)

    entries = [
        build_entry_payload(row, args.window_days, args.history_days)
        for row in iter_target_rows(injuries, args.season, args.include_baseline)
    ]

    if not entries:
        raise SystemExit("No injuries found for the requested configuration.")

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "season": args.season,
        "window_days": args.window_days,
        "history_days": args.history_days,
        "exclude_injury_day": True,
        "injuries_file": str(args.injuries_file),
        "entries_count": len(entries),
    }

    payload = {
        "metadata": metadata,
        "entries": entries,
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] Backtesting config written to {output_path}")
    print(f"       Entries: {len(entries)} | Season filter: {args.season}")


if __name__ == "__main__":
    main()

