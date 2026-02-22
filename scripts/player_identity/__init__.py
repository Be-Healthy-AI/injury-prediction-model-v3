"""
Player Identity Layer: internal canonical player ID and external ID mappings.

- Internal players are stored in players_raw_data/players.json.
- External ID mappings (TransferMarkt, FBRef, etc.) in players_raw_data/external_id_mappings.json.
- Admin can fix associations via CLI or by editing the JSON; verified_by_admin mappings
  are never overwritten by the auto-resolver.
"""

from scripts.player_identity.store import (
    get_external_id,
    get_internal_id_by_external,
    get_mapping_row,
    load_mappings,
    load_players,
    set_external_id,
    PlayerIdentityStore,
)

__all__ = [
    "PlayerIdentityStore",
    "load_players",
    "load_mappings",
    "set_external_id",
    "get_external_id",
    "get_internal_id_by_external",
    "get_mapping_row",
]
