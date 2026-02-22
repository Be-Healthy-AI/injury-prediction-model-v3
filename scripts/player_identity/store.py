"""
Player Identity storage: internal players and external ID mappings.

All data lives under players_raw_data/ at the repository root.
Resolver must never overwrite mappings where verified_by_admin is true.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
PLAYERS_RAW_DATA = REPO_ROOT / "players_raw_data"
PLAYERS_FILE = PLAYERS_RAW_DATA / "players.json"
MAPPINGS_FILE = PLAYERS_RAW_DATA / "external_id_mappings.json"


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_players(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load internal players from players.json."""
    p = path or PLAYERS_FILE
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def load_mappings(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load external ID mappings from external_id_mappings.json."""
    p = path or MAPPINGS_FILE
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def save_players(players: List[Dict[str, Any]], path: Optional[Path] = None) -> None:
    """Save internal players to players.json."""
    p = path or PLAYERS_FILE
    _ensure_dir(p)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(players, f, indent=2, ensure_ascii=False)


def save_mappings(mappings: List[Dict[str, Any]], path: Optional[Path] = None) -> None:
    """Save external ID mappings to external_id_mappings.json."""
    p = path or MAPPINGS_FILE
    _ensure_dir(p)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(mappings, f, indent=2, ensure_ascii=False)


def get_external_id(
    internal_id: str,
    source: str,
    mappings: Optional[List[Dict[str, Any]]] = None,
    path: Optional[Path] = None,
) -> Optional[str]:
    """Return external_id for (internal_id, source), or None."""
    if mappings is None:
        mappings = load_mappings(path)
    for m in mappings:
        if m.get("internal_id") == internal_id and m.get("source") == source:
            return m.get("external_id")
    return None


def get_internal_id_by_external(
    source: str,
    external_id: str,
    mappings: Optional[List[Dict[str, Any]]] = None,
    path: Optional[Path] = None,
) -> Optional[str]:
    """Return internal_id that has this (source, external_id), or None."""
    if mappings is None:
        mappings = load_mappings(path)
    external_id_str = str(external_id)
    for m in mappings:
        if m.get("source") == source and str(m.get("external_id")) == external_id_str:
            return m.get("internal_id")
    return None


def get_mapping_row(
    internal_id: str,
    source: str,
    mappings: Optional[List[Dict[str, Any]]] = None,
    path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Return the full mapping row for (internal_id, source), or None."""
    if mappings is None:
        mappings = load_mappings(path)
    for m in mappings:
        if m.get("internal_id") == internal_id and m.get("source") == source:
            return m
    return None


def set_external_id(
    internal_id: str,
    source: str,
    external_id: str,
    verified_by_admin: bool = True,
    notes: Optional[str] = None,
    path: Optional[Path] = None,
) -> None:
    """
    Set or update external ID for (internal_id, source).
    Used by Admin or resolver. When called by Admin, pass verified_by_admin=True
    so the auto-resolver will not overwrite this mapping.
    """
    mappings = load_mappings(path)
    p = path or MAPPINGS_FILE
    now = _now_iso()
    external_id_str = str(external_id)

    for m in mappings:
        if m.get("internal_id") == internal_id and m.get("source") == source:
            m["external_id"] = external_id_str
            m["verified_by_admin"] = verified_by_admin
            m["updated_at"] = now
            if notes is not None:
                m["notes"] = notes
            save_mappings(mappings, p)
            return

    mappings.append({
        "internal_id": internal_id,
        "source": source,
        "external_id": external_id_str,
        "verified_by_admin": verified_by_admin,
        "created_at": now,
        "updated_at": now,
        "notes": notes or "",
    })
    save_mappings(mappings, p)


class PlayerIdentityStore:
    """
    Store for internal players and external ID mappings.
    Use get_external_id / set_external_id / get_internal_id_by_external;
    resolver must never overwrite rows where verified_by_admin is true.
    """

    def __init__(
        self,
        players_path: Optional[Path] = None,
        mappings_path: Optional[Path] = None,
    ) -> None:
        self.players_path = players_path or PLAYERS_FILE
        self.mappings_path = mappings_path or MAPPINGS_FILE

    def load_players(self) -> List[Dict[str, Any]]:
        return load_players(self.players_path)

    def load_mappings(self) -> List[Dict[str, Any]]:
        return load_mappings(self.mappings_path)

    def save_players(self, players: List[Dict[str, Any]]) -> None:
        save_players(players, self.players_path)

    def save_mappings(self, mappings: List[Dict[str, Any]]) -> None:
        save_mappings(mappings, self.mappings_path)

    def get_external_id(self, internal_id: str, source: str) -> Optional[str]:
        return get_external_id(internal_id, source, path=self.mappings_path)

    def get_internal_id_by_external(self, source: str, external_id: str) -> Optional[str]:
        return get_internal_id_by_external(source, str(external_id), path=self.mappings_path)

    def get_mapping_row(self, internal_id: str, source: str) -> Optional[Dict[str, Any]]:
        return get_mapping_row(internal_id, source, path=self.mappings_path)

    def set_external_id(
        self,
        internal_id: str,
        source: str,
        external_id: str,
        verified_by_admin: bool = True,
        notes: Optional[str] = None,
    ) -> None:
        set_external_id(
            internal_id,
            source,
            str(external_id),
            verified_by_admin=verified_by_admin,
            notes=notes,
            path=self.mappings_path,
        )
