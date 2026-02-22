# Player raw data – schema

This folder stores the **Player Identity Layer** data. All paths are relative to the repository root.

## Location

- **Path**: `players_raw_data/` at the repository root (e.g. `C:\Users\joao.henriques\IPM V3\players_raw_data`).

## Files

| File | Purpose |
|------|---------|
| `players.json` | Internal player registry: one record per player we track. |
| `external_id_mappings.json` | Mappings from internal_id to external source IDs (TransferMarkt, FBRef, etc.). |
| `pending_suggestions.json` | (Optional) Resolver suggestions awaiting Admin review. |
| `out/` | (Optional) Fetched multi-source payloads per player when saving to disk (JSON). |
| `out/csv/` | (Optional) Parsed tabular data: per-player folders (one per internal_id) with `profile.csv`, `career.csv`, `injuries.csv`, `matches_fbref.csv`; and shared files `players_profile.csv`, `players_career.csv`, `injuries_data.csv`, `matches_fbref.csv` (appended when running the pipeline unless `--no-csv-shared`). |

## Schema

### players.json

Array of objects:

| Field | Type | Description |
|-------|------|-------------|
| `internal_id` | string (UUID) | Canonical key. Immutable. |
| `display_name` | string | Human-readable name (for UI/logs). |
| `created_at` | string (ISO 8601) | When the player was first created. |
| `notes` | string (optional) | Admin notes (e.g. "merged duplicate", "retired"). |

### external_id_mappings.json

Array of objects:

| Field | Type | Description |
|-------|------|-------------|
| `internal_id` | string (UUID) | References a player in `players.json`. |
| `source` | string | `transfermarkt` \| `fbref` \| `sofascore` \| `understat` \| `wikipedia` \| `whoscored` |
| `external_id` | string | That source’s ID (e.g. TM integer as string, FBRef slug). |
| `verified_by_admin` | bool | If `true`, auto-resolver must not overwrite this mapping. |
| `created_at` | string (ISO 8601) | When this mapping was first added. |
| `updated_at` | string (ISO 8601) | Last change (admin or auto). |
| `notes` | string (optional) | E.g. "wrong auto-match; corrected to XYZ". |

- **Uniqueness**: One `(internal_id, source)` → one `external_id` (at most one mapping per source per internal player).

## Admin – fixing external ID associations

- **Direct edit**: Open `external_id_mappings.json` (or `players.json`) and fix or add mappings. Set `verified_by_admin` to `true` for any row you correct so the auto-resolver will not overwrite it.
- **CLI** (run from repo root):
  - List players and all source mappings:  
    `python scripts/player_identity/cli.py list-players`
  - Set external ID (marks as admin-verified):  
    `python scripts/player_identity/cli.py set-external-id <internal_id> <source> <external_id>`  
    Supported sources: `transfermarkt`, `fbref`, `sofascore`, `understat`, `wikipedia`, `whoscored`.
  - Mark an existing mapping as verified (so resolver won’t overwrite):  
    `python scripts/player_identity/cli.py mark-verified <internal_id> <source>`
- **Correction loop**: If the resolver linked the wrong FBRef (or TM) ID, run `set-external-id` with the correct value; then re-run the fetch pipeline. The fetcher always uses the current mappings.
