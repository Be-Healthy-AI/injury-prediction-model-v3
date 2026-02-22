#!/usr/bin/env python3
"""
Admin CLI for Player Identity: list players, set external IDs, mark mappings as verified.

Usage (from repo root):
  python scripts/player_identity/cli.py list-players
  python scripts/player_identity/cli.py set-external-id <internal_id> <source> <external_id>
  python scripts/player_identity/cli.py mark-verified <internal_id> <source>
  Sources: transfermarkt, fbref, sofascore, understat, wikipedia, whoscored
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.player_identity.store import (
    load_mappings,
    load_players,
    set_external_id,
    PlayerIdentityStore,
)

# All supported data sources (design: documentation/player_identity_design.md)
SOURCES = ["transfermarkt", "fbref", "sofascore", "understat", "wikipedia", "whoscored"]


def cmd_list_players(store: PlayerIdentityStore) -> int:
    """List internal players and their external ID mappings (all 6 sources)."""
    players = store.load_players()
    mappings = store.load_mappings()

    if not players:
        print("No players in registry.")
        return 0

    # Index mappings by internal_id, then source
    by_internal: dict = {}
    for m in mappings:
        iid = m.get("internal_id")
        if iid not in by_internal:
            by_internal[iid] = {}
        by_internal[iid][m.get("source", "")] = m

    # Header: internal_id, display_name, then each source (short label), then verified
    short = {"transfermarkt": "TM", "fbref": "FB", "sofascore": "SS", "understat": "US", "wikipedia": "WK", "whoscored": "WS"}
    col_width = 10
    cols = " ".join(f"{short.get(s, s):<{col_width}}" for s in SOURCES)
    print(f"{'internal_id':<38} {'display_name':<22} {cols} verified")
    print("-" * 115)
    for p in players:
        iid = p.get("internal_id", "")
        name = (p.get("display_name") or "")[:21]
        m = by_internal.get(iid, {})
        cells = []
        verified_list = []
        for s in SOURCES:
            row = m.get(s, {})
            ext = (row.get("external_id") or "-")[:10] if row else "-"
            cells.append(ext)
            if row and row.get("verified_by_admin"):
                verified_list.append(short.get(s, s))
        verified_str = ",".join(verified_list) if verified_list else "-"
        col_width = 10
        print(f"{iid:<38} {name:<22} " + " ".join(f"{c:<{col_width}}" for c in cells) + " " + verified_str)
    return 0


def cmd_set_external_id(
    store: PlayerIdentityStore,
    internal_id: str,
    source: str,
    external_id: str,
    verified: bool,
) -> int:
    """Set external ID for (internal_id, source) and mark as admin-verified by default."""
    if source not in SOURCES:
        print(f"Source must be one of: {', '.join(SOURCES)}.", file=sys.stderr)
        return 1
    set_external_id(internal_id, source, external_id, verified_by_admin=verified, path=store.mappings_path)
    print(f"Set {source}={external_id} for {internal_id} (verified_by_admin={verified}).")
    return 0


def cmd_mark_verified(store: PlayerIdentityStore, internal_id: str, source: str) -> int:
    """Set verified_by_admin=true for existing (internal_id, source) mapping."""
    if source not in SOURCES:
        print(f"Source must be one of: {', '.join(SOURCES)}.", file=sys.stderr)
        return 1
    mappings = store.load_mappings()
    found = False
    for m in mappings:
        if m.get("internal_id") == internal_id and m.get("source") == source:
            m["verified_by_admin"] = True
            m["updated_at"] = datetime.now(timezone.utc).isoformat()
            found = True
            break
    if not found:
        print(f"No mapping found for internal_id={internal_id} source={source}. Use set-external-id first.", file=sys.stderr)
        return 1
    store.save_mappings(mappings)
    print(f"Marked {source} for {internal_id} as verified_by_admin=true.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Player Identity Admin CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-players", help="List internal players and their external ID mappings")

    sp_set = subparsers.add_parser("set-external-id", help="Set external ID for a player/source")
    sp_set.add_argument("internal_id", help="Internal player UUID")
    sp_set.add_argument("source", choices=SOURCES, help="Data source")
    sp_set.add_argument("external_id", help="External ID (e.g. TM integer, FBRef slug)")
    sp_set.add_argument("--no-verified", action="store_true", help="Set verified_by_admin=false (default: true)")

    sp_mark = subparsers.add_parser("mark-verified", help="Mark existing mapping as verified_by_admin=true")
    sp_mark.add_argument("internal_id", help="Internal player UUID")
    sp_mark.add_argument("source", choices=SOURCES, help="Data source")

    args = parser.parse_args(argv)
    store = PlayerIdentityStore()

    if args.command == "list-players":
        return cmd_list_players(store)
    if args.command == "set-external-id":
        return cmd_set_external_id(
            store,
            args.internal_id,
            args.source,
            args.external_id,
            verified=not getattr(args, "no_verified", False),
        )
    if args.command == "mark-verified":
        return cmd_mark_verified(store, args.internal_id, args.source)
    return 1


if __name__ == "__main__":
    sys.exit(main())
