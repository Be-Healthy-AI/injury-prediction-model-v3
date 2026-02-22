"""
Player Identity resolver: name (+ hints) -> internal_id.

- Looks up existing internal_id by TransferMarkt or FBRef ID if provided.
- Otherwise searches FBRef by name (+ optional club); creates new internal player
  and mappings with verified_by_admin=false (never overwrites admin-verified mappings).
- Optional transfermarkt_id: when provided, creates or links player and adds TM mapping (resolver sets verified_by_admin=false).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Lazy imports for scrapers to avoid circular deps and heavy imports at module load
try:
    from scripts.data_collection.fbref_scraper import FBRefConfig, FBRefScraper
    HAS_FBREF = True
except ImportError:
    HAS_FBREF = False

try:
    from scripts.data_collection.wikipedia_api import search as wikipedia_search
    HAS_WIKIPEDIA = True
except ImportError:
    HAS_WIKIPEDIA = False
    wikipedia_search = None

from scripts.player_identity.store import (
    get_internal_id_by_external,
    get_mapping_row,
    load_mappings,
    load_players,
    save_mappings,
    save_players,
    set_external_id,
    PlayerIdentityStore,
    PLAYERS_FILE,
    MAPPINGS_FILE,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _best_fbref_match(
    name: str,
    results: List[Dict[str, Any]],
    club: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Pick best FBRef search result by name (and optional club) similarity."""
    if not results:
        return None
    name_lower = name.lower()
    club_lower = (club or "").lower()
    best = None
    best_score = -1.0
    for r in results:
        r_name = (r.get("name") or "").lower()
        r_club = (r.get("club") or "").lower()
        # Simple name match: ratio of matching chars / max len
        if name_lower in r_name or r_name in name_lower:
            score = 0.9
        else:
            # Token overlap
            a = set(name_lower.split())
            b = set(r_name.split())
            score = len(a & b) / max(len(a | b), 1) if (a | b) else 0.5
        if club_lower and r_club and club_lower in r_club:
            score += 0.1
        if score > best_score:
            best_score = score
            best = r
    return best if best_score >= 0.4 else (results[0] if results else None)


def resolve(
    name: str,
    transfermarkt_id: Optional[int | str] = None,
    fbref_id: Optional[str] = None,
    club: Optional[str] = None,
    birth_date: Optional[str] = None,
    store: Optional[PlayerIdentityStore] = None,
    fbref_scraper: Optional[Any] = None,
    skip_fbref_search: bool = False,
    resolve_wikipedia: bool = False,
) -> Dict[str, Any]:
    """
    Resolve player name (+ optional IDs and hints) to internal_id and ensure mappings.

    - If transfermarkt_id or fbref_id is provided and we already have that mapping, returns existing internal_id.
    - Otherwise creates a new internal player (or uses existing if matched by search) and adds mappings with verified_by_admin=false.
    - Never overwrites a mapping where verified_by_admin is true.
    - skip_fbref_search: if True, do not call FBRef search (avoids rate limits when offline or testing).
    - resolve_wikipedia: if True, search Wikipedia by name and add wikipedia mapping when missing (MediaWiki API).

    Returns:
        Dict with: internal_id, display_name, created (bool), transfermarkt_id, fbref_id, wikipedia_id (optional), mappings_updated.
    """
    store = store or PlayerIdentityStore()
    players = store.load_players()
    mappings = store.load_mappings()

    # 1) Lookup by provided external IDs
    if transfermarkt_id is not None:
        tm_str = str(transfermarkt_id)
        existing = get_internal_id_by_external("transfermarkt", tm_str, mappings=mappings)
        if existing:
            out = _enrich_mappings(store, existing, name, club, fbref_scraper, transfermarkt_id=transfermarkt_id, skip_fbref_search=skip_fbref_search, resolve_wikipedia=resolve_wikipedia)
            out["created"] = False
            return out
    if fbref_id:
        existing = get_internal_id_by_external("fbref", fbref_id, mappings=mappings)
        if existing:
            out = _enrich_mappings(store, existing, name, club, fbref_scraper, transfermarkt_id=transfermarkt_id, skip_fbref_search=skip_fbref_search, resolve_wikipedia=resolve_wikipedia)
            out["created"] = False
            return out

    # 2) Create new internal player
    internal_id = str(uuid.uuid4())
    display_name = name or "Unknown"
    players.append({
        "internal_id": internal_id,
        "display_name": display_name,
        "created_at": _now_iso(),
        "notes": "",
    })
    save_players(players, store.players_path)

    # 3) Add TransferMarkt mapping if provided and not admin-verified (new player so no existing row)
    mappings_updated: List[str] = []
    if transfermarkt_id is not None:
        set_external_id(internal_id, "transfermarkt", str(transfermarkt_id), verified_by_admin=False, path=store.mappings_path)
        mappings_updated.append("transfermarkt")
        mappings = store.load_mappings()

    # 4) Add FBRef mapping: use fbref_id if provided, else search by name (+ club) unless skip_fbref_search
    fbref_resolved: Optional[str] = None
    if fbref_id:
        row = get_mapping_row(internal_id, "fbref", mappings=mappings, path=store.mappings_path)
        if not row or not row.get("verified_by_admin"):
            set_external_id(internal_id, "fbref", fbref_id, verified_by_admin=False, path=store.mappings_path)
            mappings_updated.append("fbref")
            fbref_resolved = fbref_id
    elif not skip_fbref_search and HAS_FBREF and fbref_scraper is not None:
        search_results = fbref_scraper.search_player(name, club=club)
        best = _best_fbref_match(name, search_results, club)
        if best:
            fid = best.get("fbref_id")
            if fid:
                set_external_id(internal_id, "fbref", fid, verified_by_admin=False, path=store.mappings_path)
                mappings_updated.append("fbref")
                fbref_resolved = fid
    elif not skip_fbref_search and HAS_FBREF:
        scraper = FBRefScraper(FBRefConfig())
        try:
            search_results = scraper.search_player(name, club=club)
            best = _best_fbref_match(name, search_results, club)
            if best:
                fid = best.get("fbref_id")
                if fid:
                    set_external_id(internal_id, "fbref", fid, verified_by_admin=False, path=store.mappings_path)
                    mappings_updated.append("fbref")
                    fbref_resolved = fid
        finally:
            scraper.close()

    tm_resolved = str(transfermarkt_id) if transfermarkt_id is not None else None
    if not tm_resolved:
        tm_resolved = store.get_external_id(internal_id, "transfermarkt")

    # 5) Optional: add Wikipedia mapping (search by name)
    wikipedia_resolved: Optional[str] = None
    if resolve_wikipedia and HAS_WIKIPEDIA and wikipedia_search:
        row = get_mapping_row(internal_id, "wikipedia", mappings=store.load_mappings(), path=store.mappings_path)
        if not row or not row.get("verified_by_admin"):
            results = wikipedia_search(name, limit=3)
            if results:
                title = results[0].get("title")
                if title:
                    set_external_id(internal_id, "wikipedia", title, verified_by_admin=False, path=store.mappings_path)
                    mappings_updated.append("wikipedia")
                    wikipedia_resolved = title
    if not wikipedia_resolved:
        wikipedia_resolved = store.get_external_id(internal_id, "wikipedia")

    return {
        "internal_id": internal_id,
        "display_name": display_name,
        "created": True,
        "transfermarkt_id": tm_resolved,
        "fbref_id": fbref_resolved or store.get_external_id(internal_id, "fbref"),
        "wikipedia_id": wikipedia_resolved,
        "mappings_updated": mappings_updated,
    }


def _enrich_mappings(
    store: PlayerIdentityStore,
    internal_id: str,
    name: str,
    club: Optional[str],
    fbref_scraper: Optional[Any],
    transfermarkt_id: Optional[int | str] = None,
    skip_fbref_search: bool = False,
    resolve_wikipedia: bool = False,
) -> Dict[str, Any]:
    """Ensure FBRef, TM, and optionally Wikipedia mappings exist for existing internal_id without overwriting admin-verified."""
    mappings = store.load_mappings()
    players = store.load_players()
    display_name = name or next((p.get("display_name") for p in players if p.get("internal_id") == internal_id), "Unknown")

    tm_resolved = store.get_external_id(internal_id, "transfermarkt")
    if not tm_resolved and transfermarkt_id is not None:
        row = get_mapping_row(internal_id, "transfermarkt", mappings=mappings, path=store.mappings_path)
        if not row or not row.get("verified_by_admin"):
            set_external_id(internal_id, "transfermarkt", str(transfermarkt_id), verified_by_admin=False, path=store.mappings_path)
            tm_resolved = str(transfermarkt_id)

    fbref_resolved = store.get_external_id(internal_id, "fbref")
    if not fbref_resolved and not skip_fbref_search and HAS_FBREF:
        row = get_mapping_row(internal_id, "fbref", mappings=mappings, path=store.mappings_path)
        if not row or not row.get("verified_by_admin"):
            scraper = fbref_scraper or FBRefScraper(FBRefConfig())
            try:
                search_results = scraper.search_player(name or display_name, club=club)
                best = _best_fbref_match(name or display_name, search_results, club)
                if best and best.get("fbref_id"):
                    set_external_id(internal_id, "fbref", best["fbref_id"], verified_by_admin=False, path=store.mappings_path)
                    fbref_resolved = best["fbref_id"]
            finally:
                if not fbref_scraper and hasattr(scraper, "close"):
                    scraper.close()

    wikipedia_resolved = store.get_external_id(internal_id, "wikipedia")
    if resolve_wikipedia and HAS_WIKIPEDIA and wikipedia_search and not wikipedia_resolved:
        row = get_mapping_row(internal_id, "wikipedia", mappings=mappings, path=store.mappings_path)
        if not row or not row.get("verified_by_admin"):
            results = wikipedia_search(name or display_name, limit=3)
            if results and results[0].get("title"):
                title = results[0]["title"]
                set_external_id(internal_id, "wikipedia", title, verified_by_admin=False, path=store.mappings_path)
                wikipedia_resolved = title

    return {
        "internal_id": internal_id,
        "display_name": display_name,
        "created": False,
        "transfermarkt_id": tm_resolved,
        "fbref_id": fbref_resolved,
        "wikipedia_id": wikipedia_resolved,
        "mappings_updated": [],
    }
