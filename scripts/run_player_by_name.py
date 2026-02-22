#!/usr/bin/env python3
"""
Player-centric pipeline: resolve by name (+ hints) then fetch from TransferMarkt and FBRef.

Usage (from repo root):
  python scripts/run_player_by_name.py --name "Harry Kane" --club "FC Bayern Munich"
  python scripts/run_player_by_name.py --name "Harry Kane" --transfermarkt-id 132098
  python scripts/run_player_by_name.py --name "Harry Kane" --transfermarkt-id 132098 --skip-fbref-search --no-fetch

Output: writes to players_raw_data/out/<internal_id>.json by default, or --no-save to print only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.player_identity.resolver import resolve
from scripts.player_identity.fetcher import fetch_all
from scripts.player_identity.store import PLAYERS_RAW_DATA
from scripts.player_identity.parse_to_csv import parse_and_write_csv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve player by name and fetch from TM + FBRef")
    parser.add_argument("--name", required=True, help="Player name (e.g. 'Harry Kane')")
    parser.add_argument("--club", default=None, help="Club name for disambiguation (e.g. 'FC Bayern Munich')")
    parser.add_argument("--transfermarkt-id", type=int, default=None, help="TransferMarkt player ID if known (e.g. 132098)")
    parser.add_argument("--fbref-id", default=None, help="FBRef player ID/slug if known")
    parser.add_argument("--skip-fbref-search", action="store_true", help="Do not call FBRef search (avoids rate limits)")
    parser.add_argument("--resolve-wikipedia", action="store_true", help="Search Wikipedia by name and add wikipedia mapping when missing")
    parser.add_argument("--no-fetch", action="store_true", help="Only resolve (name -> internal_id); do not fetch data")
    parser.add_argument("--no-save", action="store_true", help="Do not save fetched data to players_raw_data/out/")
    parser.add_argument("--no-tm-career", action="store_true", help="Skip TM career fetch (faster)")
    parser.add_argument("--no-tm-injuries", action="store_true", help="Skip TM injuries fetch")
    parser.add_argument("--tm-matches", action="store_true", help="Include TM match log (slower)")
    parser.add_argument("--fbref-match-logs", action="store_true", help="Include FBRef match logs (slower)")
    parser.add_argument("--season", type=int, default=None, help="Fetch match data for this season only (e.g. 2025 for 2025/26); enables TM matches and FBRef match logs")
    parser.add_argument("--skip-fbref-fetch", action="store_true", help="Skip FBRef fetch (e.g. when rate limited)")
    parser.add_argument("--no-csv", action="store_true", help="Do not parse to CSV (only save JSON)")
    parser.add_argument("--no-csv-shared", action="store_true", help="Write only per-player CSVs; do not append to shared players_profile.csv etc.")
    args = parser.parse_args(argv)

    # Resolve
    result = resolve(
        args.name,
        transfermarkt_id=args.transfermarkt_id,
        fbref_id=args.fbref_id,
        club=args.club,
        skip_fbref_search=args.skip_fbref_search,
        resolve_wikipedia=getattr(args, "resolve_wikipedia", False),
    )
    internal_id = result["internal_id"]
    print(f"internal_id: {internal_id}")
    print(f"display_name: {result['display_name']}")
    print(f"created: {result['created']}")
    print(f"transfermarkt_id: {result.get('transfermarkt_id')}")
    print(f"fbref_id: {result.get('fbref_id')}")
    print(f"wikipedia_id: {result.get('wikipedia_id')}")
    print(f"mappings_updated: {result.get('mappings_updated', [])}")

    if args.no_fetch:
        return 0

    # Fetch (when --season is set, fetch TM + FBRef match data for that season)
    season = getattr(args, "season", None)
    data = fetch_all(
        internal_id,
        include_tm_career=not args.no_tm_career,
        include_tm_injuries=not args.no_tm_injuries,
        include_tm_matches=getattr(args, "tm_matches", False) or (season is not None),
        include_fbref_match_logs=args.fbref_match_logs or (season is not None),
        season=season,
        include_fbref=not getattr(args, "skip_fbref_fetch", False),
    )
    payload = {"internal_id": internal_id, "display_name": result["display_name"], "sources": data}

    if not args.no_save:
        out_dir = PLAYERS_RAW_DATA / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{internal_id}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        print(f"Saved to {out_file}")
    else:
        js = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        print(js[:3000] + "..." if len(js) > 3000 else js)

    # Parse to CSV (profile, career, injuries, matches_fbref) using existing transformers
    if not getattr(args, "no_csv", False):
        tm_id = None
        if result.get("transfermarkt_id"):
            try:
                tm_id = int(result["transfermarkt_id"])
            except (TypeError, ValueError):
                pass
        csv_dir = PLAYERS_RAW_DATA / "out" / "csv"
        try:
            written = parse_and_write_csv(
                internal_id,
                data,
                tm_id=tm_id,
                display_name=result["display_name"],
                fbref_id=result.get("fbref_id"),
                out_dir=csv_dir,
                append_to_shared=not getattr(args, "no_csv_shared", False),
            )
            if written:
                print("CSV written:")
                for name, path in written.items():
                    print(f"  {name}: {path}")
            else:
                print("CSV: no tables produced (missing TM data or transformers)")
        except Exception as e:
            print(f"CSV parse/write failed: {e}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
