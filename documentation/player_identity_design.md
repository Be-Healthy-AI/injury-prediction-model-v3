# Player Identity Layer – Design

## 1. Goals

- **Canonical identity**: Each player has one **internal ID**. All external systems (TransferMarkt, FBRef, etc.) map to this ID.
- **Admin override**: A middle stage lets the Admin directly fix or set external ID associations (e.g. correct a wrong auto-link).
- **Multi-source**: TransferMarkt and FBRef are required; SofaScore, Understat, Wikipedia, and WhoScored are supported as additional sources.

---

## 2. Data Model

### 2.1 Internal players

One record per “real” player we track.

| Field | Type | Description |
|-------|------|-------------|
| `internal_id` | UUID | Canonical key (e.g. UUID v4). Immutable. |
| `display_name` | string | Human-readable name (for UI/logs). Can be updated. |
| `created_at` | ISO datetime | When the player was first created in our system. |
| `notes` | string (optional) | Admin notes (e.g. “merged duplicate”, “retired”). |

- **Creation**: When we accept a new player (e.g. from search + admin confirm, or from league pipeline and we create an internal entity), we generate a new `internal_id` and one row here.

### 2.2 External ID mappings

Links internal player to each external system’s identifier. **Admin can add/change/remove these.**

| Field | Type | Description |
|-------|------|-------------|
| `internal_id` | UUID | References `internal_players.internal_id`. |
| `source` | string | `transfermarkt` \| `fbref` \| `sofascore` \| `understat` \| `wikipedia` \| `whoscored` (see §6). |
| `external_id` | string | That source’s ID (e.g. TM integer as string, FBRef slug). |
| `verified_by_admin` | bool | If `true`, auto-resolver **must not overwrite** this row. Admin has explicitly set or corrected it. |
| `created_at` | ISO datetime | When this mapping was first added. |
| `updated_at` | ISO datetime | Last change (admin or auto). |
| `notes` | string (optional) | E.g. “wrong auto-match; corrected to XYZ”. |

- **Uniqueness**: One `(internal_id, source)` → one `external_id`. At most one external ID per source per internal player.
- **Resolver behaviour**:
  - May **insert** a new row if no mapping exists for that `(internal_id, source)`.
  - May **update** only if `verified_by_admin is false` (and optionally only if confidence is high).
  - Must **never** update a row where `verified_by_admin is true`; only the Admin can change it.

---

## 3. Middle stage – Admin fixing associations

Two complementary mechanisms.

### 3.1 Direct edit of mappings

- **Storage**: Mappings live in a single place (e.g. JSON file or SQLite table) that the Admin can edit.
- **Options**:
  - **A) Editable file**  
    e.g. `players_raw_data/external_id_mappings.json` (or CSV). Admin opens in editor/Excel, fixes `external_id` for a given `internal_id` + `source`, and sets `verified_by_admin: true`. Scripts read this file; resolver respects `verified_by_admin` and does not overwrite those rows.
  - **B) CLI**  
    Commands such as:  
    `set-external-id <internal_id> transfermarkt <tm_id>`  
    `set-external-id <internal_id> fbref <fbref_slug>`  
    `mark-verified <internal_id> transfermarkt`  
    CLI writes to the same store and sets `verified_by_admin = true` when Admin sets a value.
  - **C) Minimal UI**  
    Simple admin page or script that lists players, shows current mappings, and has “Edit” / “Override” to change an external ID and mark it admin-verified.

Recommendation: start with **A + B** (editable file + CLI that updates the file and sets `verified_by_admin`), so you can fix associations quickly without building a UI.

### 3.2 Optional: Pending suggestions

- Resolver (and any auto-matching) writes **suggestions** to a separate store, e.g. `pending_external_id_suggestions.json` or table `pending_mappings` with `(internal_id, source, suggested_external_id, confidence, reason)`.
- Admin flow:
  - List pending suggestions (e.g. low confidence or new players).
  - For each: **Accept** (write to `external_id_mappings` with `verified_by_admin = false` or `true`), **Reject**, or **Override** (write a different `external_id` and set `verified_by_admin = true`).
- This is optional; you can add it later if you want a clear “review queue” instead of only direct edits.

---

## 4. Resolver and pipelines

- **Input**: Player name + optional disambiguation (DOB, nationality, club, position).
- **Output**: Either an existing `internal_id` (if match found in our registry) or a new internal player + suggested external IDs (TM, FBRef).
- **Flow**:
  1. Search external sources (e.g. TM, FBRef) by name + hints.
  2. If we already have an internal player that matches (e.g. same TM ID or FBRef ID), return that `internal_id`.
  3. Otherwise create a new internal player (new UUID), and for each source add a mapping row with `verified_by_admin = false` (and optionally put the same in `pending_*` for review).
  4. When **fetching** data: use only rows from `external_id_mappings` (and optionally pending that were accepted). Resolve `internal_id` → TM ID / FBRef ID when calling the respective adapters.

Existing **club configs** can keep using TransferMarkt IDs for now. Bridge: either keep `player_ids` as TM IDs in config and have a separate table “internal_id ↔ TM id” for players that are in the identity registry, or migrate configs to internal IDs and resolve TM ID from mappings when calling TM. Easiest short term: identity layer is used by the **new** player-centric pipeline; existing league/club pipeline keeps using TM IDs; shared players can appear in both (same person, internal_id + TM id in mappings).

---

## 5. Storage layout

- **Path**: `players_raw_data/` at the repository root (e.g. `C:\Users\joao.henriques\IPM V3\players_raw_data`).
- **Files**:
  - `players_raw_data/players.json` – list of `{ internal_id, display_name, created_at, notes }`.
  - `players_raw_data/external_id_mappings.json` – list of `{ internal_id, source, external_id, verified_by_admin, created_at, updated_at, notes }`.
  - (Optional) `players_raw_data/pending_suggestions.json` – for resolver output awaiting review.
  - (Optional) `players_raw_data/out/<internal_id>.json` – fetched multi-source payload per player when saving to disk.

Use JSON for simplicity and easy manual/version-control edits; later you can add a small SQLite DB and an import script if you prefer.

---

## 6. Data sources

### Supported sources (6)

| Source | ID type / format | Access | Main use |
|--------|------------------|--------|----------|
| **TransferMarkt** | Integer (e.g. `132098`) | Scraping/API | Career, valuations, injuries, squad lists. |
| **FBRef** | Slug (e.g. `21a04d7d`) | Scraping | Match logs, stats, advanced metrics. |
| **SofaScore** | ID in URLs | Scraping | Injuries, lineups, minutes, ratings. |
| **Understat** | Player page URL / ID | Scraping | xG, xA, shot-level data. |
| **Wikipedia** | Page title or Q-id | MediaWiki API | Disambiguation, DOB, nationality, display name. |
| **WhoScored** | ID in URLs | Scraping (fragile) | Ratings, detailed stats. |

**Source priority (recommended order for resolver/fetch):** TransferMarkt + FBRef first, then Wikipedia (disambiguation), then SofaScore (injuries/lineups), then Understat (xG/xA), then WhoScored if needed.

### Adding a new source

1. Add the `source` value to CLI choices (`set-external-id`, `mark-verified`) and to any validation.
2. Implement an **adapter**: given `external_id`, return a dict of data (e.g. profile, stats). Place it in `scripts/data_collection/` (e.g. `sofascore_scraper.py`) or `scripts/player_identity/adapters/`.
3. Register the adapter in `fetch_all` in `scripts/player_identity/fetcher.py`: when `get_external_id(internal_id, source)` is present, call the adapter and set `out[source] = ...`.
4. Optionally add a **resolver** step: search/lookup by name (+ hints) and suggest a mapping with `verified_by_admin = false`.
5. Document the **ID format** and where to find the ID (URL, API response) in this section or in `players_raw_data/README.md`.

---

## 7. Implementation (reference)

- **Storage**: `players_raw_data/` at repo root – `players.json`, `external_id_mappings.json`; schema and Admin notes in `players_raw_data/README.md`.
- **Supported sources**: `transfermarkt`, `fbref`, `sofascore`, `understat`, `wikipedia`, `whoscored`. Each has at most one row per `(internal_id, source)` in `external_id_mappings`.
- **Modules**: `scripts/player_identity/` – `store.py` (load/save, get_external_id, set_external_id, get_internal_id_by_external), `resolver.py` (resolve name + hints → internal_id; optional Wikipedia search), `fetcher.py` (fetch_all(internal_id) with adapters for TM, FBRef, Wikipedia, and placeholders for SofaScore, Understat, WhoScored), `cli.py` (list-players, set-external-id, mark-verified; all 6 sources supported).
- **Adapters**: TM and FBRef use existing scrapers in `scripts/data_collection/`. Wikipedia uses `scripts/data_collection/wikipedia_api.py` (MediaWiki API). SofaScore, Understat, WhoScored have placeholder adapters in `fetcher.py` until scrapers are implemented.
- **Pipeline**: `scripts/run_player_by_name.py` – resolve by `--name` (+ optional `--club`, `--transfermarkt-id`, `--fbref-id`) then fetch; output to `players_raw_data/out/<internal_id>.json` or stdout. Fetched payload includes keys for each source that has a mapping (transfermarkt, fbref, wikipedia, sofascore, understat, whoscored).

## 8. Summary

- **Internal ID** = canonical player; all external IDs map to it.
- **External ID mappings** table with `verified_by_admin`; resolver must not overwrite admin-verified rows.
- **Middle stage** = editable mappings (file + CLI) so Admin can directly fix any association; optional pending-suggestions queue later.
- **Sources**: TransferMarkt and FBRef (must-have); SofaScore, Understat, Wikipedia, WhoScored (additional). Each gets a `source` value and rows in `external_id_mappings`; fetcher has one adapter per source.
