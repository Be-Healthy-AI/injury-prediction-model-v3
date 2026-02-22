"""
Generate matches_enriched.csv from existing matches_tm.csv and matches_fbref.csv.

Use this when:
- FBRef fetch was skipped (rate limit) but you already have matches_fbref.csv from another run, or
- You want to re-run only the merge step after updating one of the source files.

Usage:
  python scripts/player_identity/enrich_matches_from_csv.py --player-dir "players_raw_data/out/csv/550e8400-e29b-41d4-a716-446655440001"
  python scripts/player_identity/enrich_matches_from_csv.py --tm-csv path/to/matches_tm.csv --fbref-csv path/to/matches_fbref.csv --out path/to/matches_enriched.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.player_identity.parse_to_csv import merge_tm_with_fbref


def main() -> int:
    parser = argparse.ArgumentParser(description="Enrich TM matches with FBRef data from existing CSVs.")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--player-dir", type=Path, help="Player CSV directory containing matches_tm.csv and matches_fbref.csv")
    g.add_argument("--tm-csv", type=Path, help="Path to matches_tm.csv (use with --fbref-csv and --out)")
    parser.add_argument("--fbref-csv", type=Path, help="Path to matches_fbref.csv")
    parser.add_argument("--out", type=Path, help="Output path for matches_enriched.csv (required if using --tm-csv)")
    args = parser.parse_args()

    if args.player_dir:
        player_dir = Path(args.player_dir)
        tm_path = player_dir / "matches_tm.csv"
        fbref_path = player_dir / "matches_fbref.csv"
        out_path = player_dir / "matches_enriched.csv"
    else:
        if not args.tm_csv or not args.fbref_csv or not args.out:
            parser.error("--tm-csv, --fbref-csv, and --out are required when not using --player-dir")
        tm_path = Path(args.tm_csv)
        fbref_path = Path(args.fbref_csv)
        out_path = Path(args.out)

    if not tm_path.exists():
        print(f"Missing: {tm_path}", file=sys.stderr)
        return 1
    if not fbref_path.exists():
        print(f"Missing: {fbref_path}", file=sys.stderr)
        return 1

    tm_df = pd.read_csv(tm_path, encoding="utf-8-sig")
    fbref_df = pd.read_csv(fbref_path, encoding="utf-8-sig")
    if tm_df.empty:
        print("matches_tm.csv is empty; nothing to enrich.", file=sys.stderr)
        return 0
    enriched = merge_tm_with_fbref(tm_df, fbref_df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(enriched)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
