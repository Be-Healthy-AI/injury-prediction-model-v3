#!/usr/bin/env python3
"""
[DEPRECATED] This script is superseded by production/scripts/update_daily_features.py

Generate daily-features for a single player using:

  - V2 winner generator:   scripts/create_daily_features.py
  - Transfermarkt loader:  scripts/create_daily_features_transfermarkt_v2.py

Reads raw data from:
    production/raw_data/{country_lower}/{YYYYMMDD}/

Writes output to:
    production/deployments/{country}/{club}/daily_features/
    (or a custom --output-dir)

DEPRECATION NOTICE:
    This script is deprecated. Please use production/scripts/update_daily_features.py instead.
    The new script is based on the proven winner implementation and includes incremental update logic.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import sys
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup: point to project root and scripts
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent
ROOT_DIR = PRODUCTION_ROOT.parent

SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Import the V2 + Transfermarkt wrapper (uses V2 winner logic internally).
# The module lives in the top-level 'scripts' directory, which we added to sys.path.
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
        "--player-id",
        type=int,
        required=True,
        help="Player ID to generate daily-features for (e.g. 614258)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Optional explicit path to raw-data snapshot "
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
            "Optional snapshot date in YYYYMMDD format (e.g. 20251213). "
            "If provided and --data-dir is not, data-dir is set to "
            "production/raw_data/{country_lower}/{data-date}."
        ),
    )
    parser.add_argument(
        "--reference-date",
        type=str,
        required=True,
        help="As-of date (YYYY-MM-DD) for the snapshot (e.g. 2025-12-13)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory for daily-features CSV. "
            "Default: production/deployments/{country}/{club}/daily_features"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG) inside the V2 generator.",
    )
    return parser.parse_args()


def get_default_output_dir(country: str, club: str) -> Path:
    """Default deployment directory for daily-features."""
    return PRODUCTION_ROOT / "deployments" / country / club / "daily_features"


def resolve_data_dir(
    country: str,
    data_dir: Optional[Path],
    data_date: Optional[str],
) -> Path:
    """
    Resolve which raw-data folder to use:

    1) If data_dir is provided, use it.
    2) Else if data_date is provided, use production/raw_data/{country_lower}/{data_date}.
    3) Else pick the latest dated folder under production/raw_data/{country_lower}/.
    """
    if data_dir is not None:
        return data_dir

    raw_root = ROOT_DIR / "production" / "raw_data" / country.lower()

    if data_date:
        candidate = raw_root / data_date
        if not candidate.exists():
            raise FileNotFoundError(f"Raw-data folder not found for date {data_date}: {candidate}")
        return candidate

    subdirs = [
        p for p in raw_root.iterdir()
        if p.is_dir() and p.name.isdigit() and len(p.name) == 8
    ]
    if not subdirs:
        raise FileNotFoundError(f"No dated raw-data folders found under {raw_root}")
    return sorted(subdirs)[-1]


def main() -> None:
    args = parse_args()

    logger = setup_logging(verbose=args.verbose)

    data_dir = resolve_data_dir(args.country, args.data_dir, args.data_date)
    print(f"[V2-TM] Using raw data folder: {data_dir}")

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = get_default_output_dir(args.country, args.club)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[V2-TM] Output directory: {output_dir}")

    reference_date = pd.Timestamp(args.reference_date)

    logger.info(
        f"[V2-TM] Generating daily-features for player {args.player_id} "
        f"(data_dir={data_dir}, reference_date={reference_date.date()})"
    )

    df = generate_daily_features_for_player(
        player_id=args.player_id,
        data_dir=str(data_dir),
        reference_date=reference_date,
        output_dir=str(output_dir),
    )

    if df is None or df.empty:
        logger.warning(f"[V2-TM] No rows generated for player {args.player_id}")
        return

    if "date" not in df.columns:
        logger.error(f"[V2-TM] Generated DataFrame missing 'date' column; cannot sort/save.")
        return

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    out_path = output_dir / f"player_{args.player_id}_daily_features.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    logger.info(
        f"[V2-TM] Saved daily-features for player {args.player_id} to {out_path} "
        f"({len(df)} rows, {len(df.columns)} columns)"
    )


if __name__ == "__main__":
    main()


