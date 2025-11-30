# Transfermarkt Pipeline Usage

This document describes how to execute the new scraping pipeline and reuse it
for any club (e.g., Benfica now, Sporting CP next).

## 1. Required Scripts & Files

- `scripts/data_collection/transfermarkt_scraper.py` – low-level HTML fetcher.
- `scripts/data_collection/transformers.py` – converts raw tables into the six
  canonical schemas defined in `documentation/transfermarkt_schema.md`.
- `scripts/run_transfermarkt_pipeline.py` – orchestrates scraping, transforms,
  and writes CSV outputs.
- `scripts/validate_transfermarkt_data.py` – compares new outputs against the
  frozen Excel references under `original_data/`.
- `config/transfermarkt_mappings.json` – normalization aliases.
- Club-specific player manifest (JSON). Example: `config/benfica_players_2025.json`.

## 2. Running the Benfica Pilot

1. Ensure dependencies are installed: `pip install -r requirements.txt`.
2. Execute the pipeline (player detection is automatic from the squad page):

   ```powershell
   python scripts/run_transfermarkt_pipeline.py `
       --club-slug sl-benfica `
       --club-id 294 `
       --club-name "SL Benfica" `
       --seasons-back 3
   ```

   The pipeline automatically detects current squad players from Transfermarkt's squad page
   (`/kader/verein/{club_id}/saison_id/{year}`). You can optionally provide a
   `--player-manifest` JSON file to override this behavior or limit the players processed.

   Outputs are written to
   `data_exports/transfermarkt/sl_benfica/20251109/` with the six CSVs.

4. Validate against the historical workbooks (filtering to the Benfica roster):

   ```powershell
   python scripts/validate_transfermarkt_data.py `
       --output-dir data_exports/transfermarkt/sl_benfica/20251109 `
       --player-ids "<comma-separated Benfica player IDs>" `
       --report-path data_exports/transfermarkt/sl_benfica/20251109/validation_summary.csv
   ```

   The report lists row-count or column mismatches that need inspection.

## 3. Adapting to Other Clubs (e.g., Sporting CP)

1. Find the club's Transfermarkt slug and ID (e.g., from the club's profile page URL).
2. Run the pipeline with the new club parameters (player detection is automatic):

   ```powershell
   python scripts/run_transfermarkt_pipeline.py `
       --club-slug sporting-cp `
       --club-id 2940 `
       --club-name "Sporting CP" `
       --seasons-back 3
   ```

3. Optionally, if you want to limit to specific players, create a manifest JSON file
   and use `--player-manifest` to override automatic detection.
4. Rerun the validation script, filtering the reference workbooks to the new
   player IDs for an apples-to-apples comparison.

## 4. Known Follow-ups

- Match logs currently return competition summaries rather than per-match rows;
  selectors need to be refined to capture the detailed table shown on the
  website UI.
- `teams_data` and `competition_data` are placeholders (country/type empty)
  until the match table is fully populated with home/away club rows.
- Future iterations should persist outputs in Excel (to mirror `original_data`)
  once the schema parity checks are satisfied.

With these steps you can iterate on the Benfica results, adjust the scrapers,
and immediately apply the same workflow to Sporting CP or any other club.

