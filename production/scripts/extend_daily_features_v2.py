#!/usr/bin/env python3
"""
[DEPRECATED] This script is superseded by production/scripts/update_daily_features.py

Extend existing production daily-features files using the original V2 generator
from scripts/create_daily_features.py, without touching historical rows.

Workflow:
- Reads existing per-player CSVs from:
    production/deployments/{country}/{club}/daily_features
- Calls generate_daily_features_for_player(...) from scripts/create_daily_features.py
  on a given Transfermarkt data snapshot (data-dir).
- Appends only rows with date > last existing date to the production file.

DEPRECATION NOTICE:
    This script is deprecated. Please use production/scripts/update_daily_features.py instead.
    The new script is based on the proven winner implementation and includes incremental update logic.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, List

import pandas as pd
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent
ROOT_DIR = PRODUCTION_ROOT.parent

# Ensure project root and scripts directory are on sys.path
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Import V2 generator and logging setup, with Transfermarkt-aware data loading.
# The wrapper module patches the original V2 generator in-place.
from create_daily_features_transfermarkt_v2 import (  # type: ignore
    generate_daily_features_for_player,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--country",
        type=str,
        required=True,
        help='Country name (e.g. "England")',
    )
    parser.add_argument(
        "--club",
        type=str,
        required=True,
        help='Club name (e.g. "Chelsea FC")',
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help=(
            "Optional explicit path to the raw-data snapshot used by V2 "
            "(e.g. production/raw_data/england/20251213). "
            "If omitted, it is inferred from --data-date or from the latest "
            "folder under production/raw_data/{country_lower}/."
        ),
    )
    parser.add_argument(
        "--data-date",
        type=str,
        default=None,
        help=(
            "Optional snapshot date in YYYYMMDD format "
            "(e.g. 20251213). If provided and --data-dir is not, "
            "data-dir is set to production/raw_data/{country_lower}/{data-date}."
        ),
    )
    parser.add_argument(
        "--reference-date",
        type=str,
        required=True,
        help="As-of date (YYYY-MM-DD) for the snapshot (e.g. 2025-12-13)",
    )
    parser.add_argument(
        "--players",
        type=int,
        nargs="*",
        help=(
            "Optional list of player IDs to process. "
            "If omitted, extends all players with existing files in the club deployment folder."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG) inside the V2 generator.",
    )
    return parser.parse_args()


def get_deploy_dir(country: str, club: str) -> Path:
    """Return daily-features deployment directory for a club."""
    return PRODUCTION_ROOT / "deployments" / country / club / "daily_features"


def list_existing_player_ids(deploy_dir: Path) -> List[int]:
    """Infer player IDs from existing production daily-features files."""
    ids: List[int] = []
    for f in deploy_dir.glob("player_*_daily_features.csv"):
        try:
            parts = f.stem.split("_")
            pid = int(parts[1])
            ids.append(pid)
        except Exception:
            continue
    return sorted(set(ids))


def extend_player_file(
    player_id: int,
    deploy_dir: Path,
    data_dir: Path,
    reference_date: str,
    verbose: bool = False,
) -> None:
    """Extend a single player's production daily-features file using V2 generator."""
    logger = setup_logging(verbose=verbose)

    existing_file = deploy_dir / f"player_{player_id}_daily_features.csv"
    if not existing_file.exists():
        logger.warning(f"Player {player_id}: existing file not found at {existing_file}, skipping.")
        return

    logger.info(f"Player {player_id}: extending {existing_file}")

    try:
        df_existing = pd.read_csv(existing_file, parse_dates=["date"])
    except Exception as e:
        logger.error(f"Player {player_id}: failed to read existing file: {e}")
        return

    if df_existing.empty or "date" not in df_existing.columns:
        logger.warning(f"Player {player_id}: existing file empty or missing 'date' column, skipping.")
        return

    last_date = pd.to_datetime(df_existing["date"].max()).normalize()
    logger.info(f"Player {player_id}: last existing date = {last_date.date()}")

    # Generate full history for this player using V2 generator
    try:
        df_new_full = generate_daily_features_for_player(
            player_id=player_id,
            data_dir=str(data_dir),
            reference_date=pd.Timestamp(reference_date),
            output_dir=str(deploy_dir),  # we will handle writing ourselves
        )
    except Exception as e:
        logger.error(f"Player {player_id}: V2 generator raised an error: {e}")
        return

    if df_new_full is None or df_new_full.empty:
        logger.warning(f"Player {player_id}: V2 generator returned no rows, nothing to append.")
        return

    if "date" not in df_new_full.columns:
        logger.error(f"Player {player_id}: V2 output missing 'date' column, cannot merge.")
        return

    df_new_full["date"] = pd.to_datetime(df_new_full["date"], errors="coerce")
    df_to_append = df_new_full[df_new_full["date"] > last_date].copy()

    if df_to_append.empty:
        logger.info(f"Player {player_id}: no new dates after {last_date.date()}, nothing to append.")
        return

    # Align schemas: ensure both frames have the same columns
    existing_cols = set(df_existing.columns)
    new_cols = set(df_to_append.columns)

    for col in new_cols - existing_cols:
        df_existing[col] = pd.NA
    for col in existing_cols - new_cols:
        df_to_append[col] = pd.NA

    # Reorder new columns to match existing column order
    df_to_append = df_to_append[df_existing.columns]

    df_combined = pd.concat([df_existing, df_to_append], ignore_index=True)
    df_combined = (
        df_combined.sort_values("date")
        .drop_duplicates(subset=["date"], keep="first")
        .reset_index(drop=True)
    )

    # Backup original file once before overwriting
    backup = existing_file.with_suffix(".backup.csv")
    if not backup.exists():
        existing_file.replace(backup)
        logger.info(f"Player {player_id}: backup saved as {backup.name}")

    df_combined.to_csv(existing_file, index=False, encoding="utf-8-sig")

    logger.info(
        f"Player {player_id}: appended {len(df_to_append)} new rows "
        f"(from {df_to_append['date'].min().date()} to {df_to_append['date'].max().date()})"
    )


def main() -> None:
    args = parse_args()

    deploy_dir = get_deploy_dir(args.country, args.club)
    deploy_dir.mkdir(parents=True, exist_ok=True)

    # Resolve data directory if not explicitly provided
    if args.data_dir is None:
        raw_root = ROOT_DIR / "production" / "raw_data" / args.country.lower()
        if args.data_date:
            candidate = raw_root / args.data_date
            if not candidate.exists():
                raise FileNotFoundError(f"Raw-data folder not found for date {args.data_date}: {candidate}")
            data_dir = candidate
        else:
            # Pick latest dated subfolder (e.g. 20251215)
            subdirs = [
                p for p in raw_root.iterdir()
                if p.is_dir() and p.name.isdigit() and len(p.name) == 8
            ]
            if not subdirs:
                raise FileNotFoundError(f"No dated raw-data folders found under {raw_root}")
            data_dir = sorted(subdirs)[-1]
    else:
        data_dir = args.data_dir

    print(f"Using raw data folder: {data_dir}")

    if args.players:
        player_ids = args.players
    else:
        player_ids = list_existing_player_ids(deploy_dir)

    if not player_ids:
        print(f"No existing player files found in {deploy_dir}, nothing to extend.")
        return

    for pid in player_ids:
        extend_player_file(
            player_id=pid,
            deploy_dir=deploy_dir,
            data_dir=data_dir,
            reference_date=args.reference_date,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()


