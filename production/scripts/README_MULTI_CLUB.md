# Multi-Club Deployment Guide

This guide explains how to extend the injury prediction pipeline from Chelsea FC to all Premier League clubs while keeping the existing Chelsea process running independently.

## Overview

The multi-club extension consists of two main scripts:

1. **`migrate_all_premier_league_clubs.py`** - One-time migration script to set up all clubs
2. **`deploy_all_clubs.py`** - Daily orchestrator to run the pipeline for all clubs

## Important: Chelsea FC Independence

⚠️ **CRITICAL**: The existing Chelsea FC process continues to run independently. These scripts:
- **Exclude Chelsea FC by default** (it's already set up and running)
- **Do not modify** any Chelsea FC files or processes
- **Run in parallel** to the existing Chelsea workflow

## Prerequisites

Before running the migration, ensure you have:

1. **Daily features files** in `daily_features_output/` with data until 2025-12-05
2. **Timelines file** at `models_production/lgbm_muscular_v1/data/timelines/test/timelines_35day_season_2025_2026_v4_muscular.csv`
3. **Raw data** fetched for the date you want to process

## Step 1: One-Time Migration

Run the migration script to:
- Extract player-club mappings from the timelines file
- Create `config.json` files for each club
- Copy daily features files to club-specific folders
- Split timelines file by club

### Dry Run (Recommended First)

```bash
python production/scripts/migrate_all_premier_league_clubs.py --dry-run
```

This will show you what would be done without actually doing it.

### Actual Migration

```bash
python production/scripts/migrate_all_premier_league_clubs.py
```

### Options

- `--dry-run`: Show what would be done without doing it
- `--skip-daily-features`: Skip daily features migration
- `--skip-timelines`: Skip timelines migration
- `--skip-configs`: Skip config file creation
- `--exclude-clubs "Chelsea FC"`: Exclude specific clubs (default: Chelsea FC)
- `--daily-features-source PATH`: Custom source for daily features
- `--timelines-source PATH`: Custom source for timelines

### What It Does

1. **Reads timelines file** to extract player-club mappings
2. **Creates club folders** in `production/deployments/England/{Club Name}/`
3. **Creates config.json** for each club with player IDs
4. **Copies daily features** from `daily_features_output/` to each club's `daily_features/` folder
5. **Splits timelines** by club and saves to each club's `timelines/` folder

## Step 2: Daily Updates for All Clubs

After migration, use the orchestrator script to run incremental updates for all clubs.

### One-by-One Processing (Recommended for Testing)

Process clubs one-by-one for easier testing and debugging. This approach allows you to:
- Test each club individually
- Identify and fix issues before moving to the next club
- Stop immediately if any step fails

```bash
# Process a single club
python production/scripts/deploy_all_clubs.py \
    --clubs "Arsenal FC" \
    --data-date 20251229 \
    --stop-on-error

# Then process the next club
python production/scripts/deploy_all_clubs.py \
    --clubs "Aston Villa" \
    --data-date 20251229 \
    --stop-on-error
```

The `--stop-on-error` flag ensures the script stops if any step fails, making it easier to identify and fix issues.

### Batch Processing (All Clubs)

Process all clubs at once:

```bash
python production/scripts/deploy_all_clubs.py --data-date 20251229
```

This will:
1. Update daily features (incremental from last date)
2. Update timelines (incremental)
3. Generate predictions
4. Generate dashboards
5. Generate predictions table CSV

### Options

- `--country "England"`: Country name (default: England)
- `--clubs "Arsenal FC"`: Process specific club(s) - use for one-by-one testing (default: all except excluded)
- `--exclude-clubs "Chelsea FC"`: Exclude specific clubs (default: Chelsea FC)
- `--data-date YYYYMMDD`: Data date (default: today)
- `--skip-fetch`: Skip raw data fetch (default: True, should be done once)
- `--skip-daily-features`: Skip daily features update
- `--skip-timelines`: Skip timelines update
- `--skip-predictions`: Skip predictions generation
- `--skip-dashboards`: Skip dashboards generation
- `--skip-table`: Skip predictions table generation
- `--stop-on-error`: Stop if a club fails (recommended for one-by-one testing)

### Processing Multiple Specific Clubs

```bash
python production/scripts/deploy_all_clubs.py \
    --clubs "Arsenal FC,Liverpool FC,Manchester City" \
    --data-date 20251229
```

### Step-by-Step Processing

If you want to run steps separately:

```bash
# Step 1: Update daily features only
python production/scripts/deploy_all_clubs.py \
    --skip-timelines --skip-predictions --skip-dashboards --skip-table \
    --data-date 20251226

# Step 2: Update timelines only
python production/scripts/deploy_all_clubs.py \
    --skip-daily-features --skip-predictions --skip-dashboards --skip-table \
    --data-date 20251226

# Step 3: Generate predictions only
python production/scripts/deploy_all_clubs.py \
    --skip-daily-features --skip-timelines --skip-dashboards --skip-table \
    --data-date 20251226

# Step 4: Generate dashboards only
python production/scripts/deploy_all_clubs.py \
    --skip-daily-features --skip-timelines --skip-predictions --skip-table \
    --data-date 20251226
```

## Step 0: Fetch Raw Data (Once for All Clubs)

Before running the pipeline, fetch raw data for all Premier League clubs:

```bash
python production/scripts/fetch_raw_data.py \
    --country England \
    --league "Premier League" \
    --competition-id "GB1" \
    --competition-slug "premier-league" \
    --as-of-date 20251226
```

This creates a date-stamped folder in `production/raw_data/england/20251226/` that all clubs will use.

## Workflow Summary

### Initial Setup (One-Time)

1. **Fetch raw data** for all clubs:
   ```bash
   python production/scripts/fetch_raw_data.py --country England --league "Premier League" --competition-id "GB1" --competition-slug "premier-league" --as-of-date 20251226
   ```

2. **Run migration** to set up all clubs:
   ```bash
   python production/scripts/migrate_all_premier_league_clubs.py
   ```

### Daily Workflow

1. **Fetch raw data** (once for all clubs):
   ```bash
   python production/scripts/fetch_raw_data.py --country England --league "Premier League" --competition-id "GB1" --competition-slug "premier-league" --as-of-date YYYYMMDD
   ```

2. **Run pipeline for all clubs**:
   ```bash
   python production/scripts/deploy_all_clubs.py --data-date YYYYMMDD
   ```

## Output Structure

After migration and daily updates, each club will have:

```
production/deployments/England/{Club Name}/
├── config.json                    # Club configuration with player IDs
├── daily_features/               # Daily features files (incremental from 2025-12-06)
│   └── player_*_daily_features.csv
├── timelines/                     # 35-day timelines (incremental from 2025-12-06)
│   └── timelines_35day_season_2025_2026_v4_muscular.csv
├── predictions/                   # Prediction files
│   ├── predictions_lgbm_v2_YYYYMMDD.csv
│   ├── predictions_table_YYYYMMDD.csv
│   └── players/                  # Per-player prediction files
│       └── player_*_predictions_YYYYMMDD.csv
└── dashboards/                    # Dashboard PNGs
    └── players/
        └── player_*_YYYYMMDD_probabilities.png
```

## Chelsea FC Independence

The Chelsea FC process continues to work exactly as before:

- **No changes** to Chelsea FC files or scripts
- **Independent execution** - can run Chelsea updates separately
- **No interference** - multi-club scripts exclude Chelsea by default

To run Chelsea updates (as before):

```bash
# Step 1: Fetch raw data (if needed)
python production/scripts/fetch_raw_data.py --country England --league "Premier League" --competition-id "GB1" --competition-slug "premier-league" --as-of-date 20251226 --clubs "Chelsea FC"

# Step 2: Update daily features
python production/scripts/update_daily_features.py --country England --club "Chelsea FC" --data-date 20251226

# Step 3: Update timelines
python production/scripts/update_timelines.py --config "production/deployments/England/Chelsea FC/config.json" --data-date 20251226 --regenerate-from-date 2025-12-25

# Step 4: Generate predictions
python production/scripts/generate_predictions_lgbm_v2.py --country England --club "Chelsea FC"

# Step 5: Generate dashboards
python production/scripts/generate_dashboards.py --country England --club "Chelsea FC" --date 2025-12-26

# Step 6: Generate predictions table
python production/scripts/generate_predictions_table.py --country England --club "Chelsea FC" --date 2025-12-26
```

## Troubleshooting

### Migration Issues

- **Missing daily features files**: Check that `daily_features_output/` contains files for all players
- **Missing timelines file**: Verify the path to the timelines file is correct
- **Club name mismatches**: Check that club names in timelines match expected format

### Deployment Issues

- **Config file not found**: Run migration script first
- **Missing raw data**: Run `fetch_raw_data.py` first
- **Player missing from predictions**: Check that player is in club's `config.json`

## Notes

- The migration script **excludes Chelsea FC by default** to avoid interfering with existing processes
- Daily features and timelines are **copied** (not moved) from source locations
- Incremental updates start from **2025-12-06** onwards (base data until 2025-12-05 is migrated)
- All scripts are designed to **continue processing** even if one club fails (unless `--stop-on-error` is used)


