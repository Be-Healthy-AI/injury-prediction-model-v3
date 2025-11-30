#!/usr/bin/env python3
"""
Compare freshly scraped Transfermarkt datasets against the frozen references.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


REFERENCE_MAP = {
    "players_profile": "20251106_players_profile.xlsx",
    "players_career": "20251106_players_career.xlsx",
    "injuries_data": "20251106_injuries_data.xlsx",
    "match_data": "20251106_match_data.xlsx",
    "teams_data": "20251106_teams_data.xlsx",
    "competition_data": "20251106_competition_data.xlsx",
}

PRIMARY_KEYS = {
    "players_profile": ["id"],
    "players_career": ["id", "Season", "Date", "From", "To"],
    "injuries_data": ["player_id", "season", "injury_type", "fromDate"],
    "match_data": ["player_id", "date", "competition", "home_team", "away_team"],
    "teams_data": ["team"],
    "competition_data": ["competition"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="Directory with newly generated CSVs.")
    parser.add_argument(
        "--reference-dir",
        default="original_data",
        help="Directory containing the authoritative Excel workbooks.",
    )
    parser.add_argument(
        "--player-ids",
        help="Optional comma-separated player IDs to filter reference data for club-specific validation.",
    )
    parser.add_argument(
        "--report-path",
        help="Optional path to write a markdown summary of comparison metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    player_ids = (
        [int(pid) for pid in args.player_ids.split(",")]
        if args.player_ids
        else None
    )
    summary_rows = []
    for name, ref_file in REFERENCE_MAP.items():
        new_df = _load_new(args.output_dir, name)
        ref_df = _load_reference(args.reference_dir, ref_file)
        if player_ids and "player_id" in ref_df.columns:
            ref_df = ref_df[ref_df["player_id"].isin(player_ids)]
        elif player_ids and "id" in ref_df.columns:
            ref_df = ref_df[ref_df["id"].isin(player_ids)]
        comparison = compare_frames(name, new_df, ref_df, PRIMARY_KEYS[name])
        summary_rows.append(comparison)
        print(f"{name}: {comparison['status']} ({comparison['message']})")

    if args.report_path:
        pd.DataFrame(summary_rows).to_csv(args.report_path, index=False)


def compare_frames(
    dataset: str,
    new: pd.DataFrame,
    reference: pd.DataFrame,
    key_cols: List[str],
) -> Dict[str, str]:
    if new.empty and reference.empty:
        return {"dataset": dataset, "status": "match", "message": "both empty"}
    new_norm = _normalize(new, key_cols)
    ref_norm = _normalize(reference, key_cols)
    if len(new_norm) != len(ref_norm):
        return {
            "dataset": dataset,
            "status": "mismatch",
            "message": f"row_count new={len(new_norm)} ref={len(ref_norm)}",
        }
    try:
        pd.testing.assert_frame_equal(new_norm, ref_norm, check_like=True, check_dtype=False)
        return {"dataset": dataset, "status": "match", "message": "rows identical"}
    except AssertionError as exc:
        return {"dataset": dataset, "status": "mismatch", "message": str(exc)[:200]}


def _normalize(df: pd.DataFrame, key_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return df
    missing_keys = [col for col in key_cols if col not in df.columns]
    if missing_keys:
        raise ValueError(f"Missing key columns {missing_keys} in dataset.")
    sort_cols = [col for col in key_cols if col in df.columns]
    normalized = df.sort_values(sort_cols).reset_index(drop=True)
    return normalized


def _load_new(output_dir: str, dataset: str) -> pd.DataFrame:
    path = _glob_latest(Path(output_dir), dataset)
    if not path:
        return pd.DataFrame()
    return pd.read_csv(path)


def _glob_latest(directory: Path, dataset: str) -> Optional[Path]:
    candidates = sorted(directory.glob(f"*_{dataset}.csv"))
    return candidates[-1] if candidates else None


def _load_reference(reference_dir: str, filename: str) -> pd.DataFrame:
    path = Path(reference_dir) / filename
    return pd.read_excel(path)


if __name__ == "__main__":
    main()

