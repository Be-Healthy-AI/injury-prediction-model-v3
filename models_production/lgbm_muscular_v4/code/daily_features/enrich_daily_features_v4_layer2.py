#!/usr/bin/env python3
"""
Layer 2 enrichment for daily features (V4)

Goal:
- Read existing daily features files (Layer 1)
- Add more sophisticated workload, recovery, and injury-history features
- Write enriched files to a separate output directory

This script is designed to be:
- Non-destructive (does not modify Layer 1 files)
- Restart-friendly (skips files that are already enriched)
- Configurable to process only a subset of files (for testing)
"""

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    def tqdm(iterable, **kwargs):
        return iterable

import argparse
import random


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent.parent.parent

DATA_DIR = ROOT_DIR / "models_production" / "lgbm_muscular_v4" / "data"
INPUT_DIR = DATA_DIR / "daily_features"
OUTPUT_DIR = DATA_DIR / "daily_features_enriched"


def enrich_one_file(input_path: Path, output_path: Path, verbose: bool = False, force: bool = False) -> None:
    """
    Load one Layer-1 daily features file, compute Layer-2 features, and save.
    """
    if output_path.exists() and not force:
        if verbose:
            print(f"   Skipping (already enriched): {output_path.name}")
        return

    if verbose:
        print(f"   Enriching: {input_path.name}")

    df = pd.read_csv(input_path, encoding="utf-8-sig", low_memory=False)

    if "date" not in df.columns:
        raise ValueError(f"'date' column missing in {input_path.name}")

    # Ensure proper types and ordering
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # ---- FORWARD-FILL last_match_position (if exists) ----
    # This ensures position is available even on non-match days
    if "last_match_position" in df.columns:
        # Replace empty strings with NaN for proper forward-filling
        df["last_match_position"] = df["last_match_position"].replace("", pd.NA)
        # Forward-fill to carry last known position to non-match days
        df["last_match_position"] = df["last_match_position"].ffill()
        # Fill any remaining NaN/empty with empty string for consistency
        df["last_match_position"] = df["last_match_position"].fillna("")

    # Basic required columns (use actual column names from Layer 1)
    # minutes_played_numeric: per-day minutes
    # matches: 1 if played a match that day, else 0
    required_cols = ["minutes_played_numeric", "matches", "cum_inj_starts"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"'{col}' column missing in {input_path.name}")

    minutes = df["minutes_played_numeric"].fillna(0)
    played = df["matches"].fillna(0)
    cum_inj = df["cum_inj_starts"].fillna(0)

    # ---- WORKLOAD WINDOWS (strictly past days) ----
    # Shift by 1 to exclude current day from "past" windows
    minutes_shifted = minutes.shift(1)
    played_shifted = played.shift(1)

    # We assume one row per calendar day; if not, this approximates N previous records.
    # Minutes windows
    for window in [3, 7, 14, 28, 35]:
        df[f"minutes_last_{window}d"] = (
            minutes_shifted.rolling(window=window, min_periods=1).sum()
        )

    # Matches windows (using played_match)
    for window in [7, 14, 28, 35]:
        df[f"matches_last_{window}d"] = (
            played_shifted.rolling(window=window, min_periods=1).sum()
        )

    # ---- ACWR (minutes 7 vs 28 days) ----
    acute_7 = df["minutes_last_7d"] / 7.0
    chronic_28 = df["minutes_last_28d"] / 28.0
    df["acwr_min_7_28"] = acute_7 / (chronic_28 + 1e-6)

    # ---- Minutes season-to-date (simple cumulative, can later be grouped by season) ----
    df["minutes_season_to_date"] = minutes_shifted.cumsum()

    df["minutes_last_7d_pct_season"] = df["minutes_last_7d"] / (
        df["minutes_season_to_date"] + 1e-6
    )
    df["minutes_last_28d_pct_season"] = df["minutes_last_28d"] / (
        df["minutes_season_to_date"] + 1e-6
    )

    # ---- INJURY HISTORY ----
    df["injury_start_flag"] = (cum_inj.diff() > 0).astype(int)

    # Last injury date (forward-filled)
    last_inj_date = df["date"].where(df["injury_start_flag"] == 1)
    last_inj_date = last_inj_date.ffill()
    df["days_since_last_injury"] = (df["date"] - last_inj_date).dt.days
    df["days_since_last_injury"] = df["days_since_last_injury"].fillna(999)

    # Injuries last 90 / 365 days (approx per record if 1 row per day)
    inj_shifted = df["injury_start_flag"].shift(1).fillna(0)
    for window in [90, 365]:
        df[f"injuries_last_{window}d"] = inj_shifted.rolling(
            window=window, min_periods=1
        ).sum()

    df["injuries_season_to_date"] = inj_shifted.cumsum()

    # ---- RECOVERY / REST ----
    last_match_date = df["date"].where(played == 1)
    last_match_date = last_match_date.ffill()
    df["days_since_last_match"] = (df["date"] - last_match_date).dt.days
    df["days_since_last_match"] = df["days_since_last_match"].fillna(999)

    df["is_back_to_back"] = (df["days_since_last_match"] <= 2).astype(int)
    df["short_rest_3_4d"] = df["days_since_last_match"].between(3, 4).astype(int)
    df["long_rest_7d_plus"] = (df["days_since_last_match"] >= 7).astype(int)

    # ---- ACTIVITY FLAGS ----
    df["has_played_last_7d"] = (df["minutes_last_7d"] > 0).astype(int)
    df["has_played_last_28d"] = (df["minutes_last_28d"] > 0).astype(int)
    df["no_recent_activity_28d"] = (df["minutes_last_28d"] == 0).astype(int)

    # ---- INTERACTION FEATURES (NEW) ----
    # Capture patterns identified in low-probability injury analysis
    
    # Inactivity risk: Long rest + no recent activity + injury history
    # Use injuries_last_365d (calculated above) as proxy for injury history
    df["inactivity_risk"] = (
        (df["days_since_last_match"] > 7) & 
        (df["minutes_last_28d"] == 0) & 
        (df["injuries_last_365d"] > 0)
    ).astype(int)
    
    # Early season + low activity risk
    # Check if days_into_season exists (from Layer 1 daily features)
    if 'days_into_season' in df.columns:
        df["early_season_low_activity"] = (
            (df["days_into_season"] <= 30) & 
            (df["minutes_last_14d"] < 90)  # Less than 90 minutes in last 14 days
        ).astype(int)
    else:
        # If days_into_season not available, set to 0
        df["early_season_low_activity"] = 0
    
    # Pre-season + long rest risk
    if 'is_pre_season' in df.columns:
        df["preseason_long_rest"] = (
            (df["is_pre_season"] == 1) & 
            (df["days_since_last_match"] > 14)
        ).astype(int)
    else:
        df["preseason_long_rest"] = 0
    
    # Low activity despite injury history (vulnerable state)
    # Try to use muscular_injuries_last_2_years if available (from Layer 1), otherwise use injuries_last_365d
    if 'muscular_injuries_last_2_years' in df.columns:
        df["low_activity_with_history"] = (
            (df["minutes_last_28d"] == 0) & 
            (df["muscular_injuries_last_2_years"] > 0)
        ).astype(int)
    elif 'injuries_last_365d' in df.columns:
        # Fallback to injuries_last_365d as proxy
        df["low_activity_with_history"] = (
            (df["minutes_last_28d"] == 0) & 
            (df["injuries_last_365d"] > 0)
        ).astype(int)
    else:
        df["low_activity_with_history"] = 0

    # Save enriched file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Enrich daily features (Layer 2) with workload and injury-history features."
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to process (for testing). "
        "If not set, processes all files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for file sampling when --max-files is set.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file processing details.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of existing enriched files (overwrites existing files).",
    )

    args = parser.parse_args(argv)

    if not INPUT_DIR.exists():
        print(f"ERROR: Input directory not found: {INPUT_DIR}")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_files = sorted(INPUT_DIR.glob("*.csv"))
    if not all_files:
        print(f"No input files found in {INPUT_DIR}")
        return 1

    if args.max_files is not None and args.max_files < len(all_files):
        random.seed(args.seed)
        sampled_files = random.sample(all_files, args.max_files)
    else:
        sampled_files = all_files

    print(f"Found {len(all_files)} base daily feature files.")
    print(f"Processing {len(sampled_files)} file(s).")
    print(f"Input dir:  {INPUT_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")

    for inp in tqdm(sampled_files, desc="Enriching daily features"):
        outp = OUTPUT_DIR / inp.name
        try:
            enrich_one_file(inp, outp, verbose=args.verbose, force=args.force)
        except Exception as e:
            print(f"Error processing {inp.name}: {e}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
