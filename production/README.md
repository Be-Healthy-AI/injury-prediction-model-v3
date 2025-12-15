# Production Deployment Structure

This directory contains the production deployment system for injury prediction models, organized by country and club for scalable multi-club deployments.

## Folder Structure

```
production/
├── scripts/                          # Transversal scripts (shared across all clubs)
│   ├── data_collection/              # Transfermarkt scraper modules
│   │   ├── transfermarkt_scraper.py
│   │   └── transformers.py
│   ├── update_daily_features.py      # Update daily features incrementally
│   ├── generate_timelines.py        # Generate 35-day timelines
│   ├── generate_predictions.py      # Generate predictions (RF + GB + Ensemble)
│   ├── generate_dashboards.py       # Generate player dashboards (PNG)
│   ├── fetch_raw_data.py            # Scrape Transfermarkt data
│   ├── deploy_club.py               # Master orchestrator
│   ├── init_club_deployment.py     # Initialize new club
│   └── copy_raw_data.py             # Copy raw data to production
│
├── config/                           # Shared configuration
│   ├── defaults.json                # Default settings
│   └── transfermarkt_mappings.json  # Transfermarkt normalization mappings
│
├── raw_data/                         # Date-stamped raw data
│   └── {country}/
│       ├── previous_seasons/        # Old season match data (shared across dates)
│       │   └── match_*.csv         # Historical match files (2022/23, 2023/24, etc.)
│       └── {YYYYMMDD}/              # Date-stamped folders
│           ├── players_profile.csv
│           ├── players_career.csv
│           ├── injuries_data.csv
│           ├── teams_data.csv
│           └── match_data/         # Current season match files only
│               └── match_*_{current_season}.csv
│
└── deployments/                      # Club-specific data
    └── {Country}/
        └── {Club Name}/
            ├── config.json           # Club-specific configuration
            ├── daily_features/      # Daily features files
            ├── timelines/           # 35-day timeline files
            ├── predictions/         # Prediction files
            │   └── ensemble/        # Ensemble predictions
            ├── dashboards/          # Dashboard outputs
            │   └── players/         # Per-player dashboard PNGs
            └── reports/             # Summary reports
```

## Quick Start

### 1. Initialize a New Club Deployment

```bash
python production/scripts/init_club_deployment.py \
    --country "England" \
    --club "Chelsea FC" \
    --player-ids 614258 346314 475411
```

### 2. Copy Raw Data

```bash
# Copy from existing export
python production/scripts/copy_raw_data.py \
    --source "data_exports/transfermarkt/england/20251205" \
    --country "england" \
    --date "20251205" \
    --validate

# Or fetch fresh data from Transfermarkt
python production/scripts/fetch_raw_data.py \
    --country "england" \
    --league "Premier League" \
    --competition-id "GB1" \
    --competition-slug "premier-league" \
    --seasons "2022,2023,2024,2025"
```

### 3. Copy Daily Features (Initial Setup)

Copy existing daily features files to the club's `daily_features/` directory, or use the update script to generate them.

### 4. Run Full Deployment

```bash
python production/scripts/deploy_club.py \
    --country "England" \
    --club "Chelsea FC" \
    --start-date "2025-07-01" \
    --end-date "2025-12-05"
```

## Daily Workflow

### Step 1: Fetch New Raw Data (Daily)

```bash
python production/scripts/fetch_raw_data.py \
    --country "england" \
    --league "Premier League" \
    --competition-id "GB1" \
    --competition-slug "premier-league" \
    --seasons "2022,2023,2024,2025"
```

This creates a new date-stamped folder in `production/raw_data/england/{today's date}/`.

**Match Data Organization:**
- **Current season** match files are stored in each date folder's `match_data/` subfolder
- **Previous seasons** match files are stored once in `previous_seasons/` folder (shared across all dates)
- This avoids duplicating historical match data in every date folder

### Step 2: Update Daily Features

```bash
python production/scripts/update_daily_features.py \
    --country "England" \
    --club "Chelsea FC"
```

This script:
- Automatically finds the latest raw data folder
- Updates daily features incrementally (appends new days)
- Processes all players in the club config, or specified players

### Step 3: Generate Predictions & Dashboards

```bash
python production/scripts/deploy_club.py \
    --country "England" \
    --club "Chelsea FC" \
    --skip-features  # Skip if features were just updated
```

Or run individual steps:

```bash
# Generate timelines
python production/scripts/generate_timelines.py \
    --country "England" \
    --club "Chelsea FC" \
    --start-date "2025-07-01" \
    --end-date "2025-12-05"

# Generate predictions
python production/scripts/generate_predictions.py \
    --country "England" \
    --club "Chelsea FC" \
    --start-date "2025-07-01" \
    --end-date "2025-12-05"

# Generate dashboards
python production/scripts/generate_dashboards.py \
    --country "England" \
    --club "Chelsea FC" \
    --start-date "2025-07-01" \
    --end-date "2025-12-05"
```

## Script Reference

### `update_daily_features.py`

Updates daily features files incrementally for a club.

**Arguments:**
- `--country` (required): Country name
- `--club` (required): Club name
- `--data-date` (optional): Date folder to use (YYYYMMDD). Defaults to latest.
- `--players` (optional): Specific player IDs to process
- `--force-rebuild`: Regenerate from scratch

**Example:**
```bash
python production/scripts/update_daily_features.py \
    --country "England" \
    --club "Chelsea FC" \
    --data-date "20251205"
```

### `generate_timelines.py`

Generates 35-day timelines from daily features.

**Arguments:**
- `--country` (required): Country name
- `--club` (required): Club name
- `--start-date` (optional): Start date (YYYY-MM-DD)
- `--end-date` (optional): End date (YYYY-MM-DD)
- `--date` (optional): Single date (defaults to today)
- `--days-back` (optional): Days back from end date (default: 7)
- `--players` (optional): Specific player IDs

### `generate_predictions.py`

Generates predictions using RF, GB, and Ensemble models.

**Arguments:**
- `--country` (required): Country name
- `--club` (required): Club name
- `--start-date` (optional): Start date (YYYY-MM-DD)
- `--end-date` (optional): End date (YYYY-MM-DD)
- `--date` (optional): Single date
- `--players` (optional): Specific player IDs
- `--force`: Force regeneration

**Output:**
- Individual model predictions in `predictions/{model_name}/`
- Ensemble predictions in `predictions/ensemble/`
- Combined file: `predictions/predictions_{date}.csv`

### `generate_dashboards.py`

Generates per-player dashboard PNGs.

**Arguments:**
- `--country` (required): Country name
- `--club` (required): Club name
- `--start-date` (optional): Start date (YYYY-MM-DD)
- `--end-date` (optional): End date (YYYY-MM-DD)
- `--date` (optional): Single date
- `--players` (optional): Specific player IDs

**Output:**
- PNG files in `dashboards/players/`
- Format: `player_{id}_{start_date}_{end_date}_probabilities.png`

### `fetch_raw_data.py`

Scrapes Transfermarkt data for a league.

**Arguments:**
- `--country` (required): Country name (default: "england")
- `--league` (required): League display name
- `--competition-id` (required): Transfermarkt competition ID
- `--competition-slug` (required): Transfermarkt URL slug
- `--seasons` (required): Comma-separated season years
- `--as-of-date` (optional): Date in YYYYMMDD format (defaults to today)
- `--resume`: Skip already-processed files
- `--max-clubs`: Limit clubs (testing)
- `--max-players-per-club`: Limit players per club (testing)

**Example:**
```bash
python production/scripts/fetch_raw_data.py \
    --country "england" \
    --league "Premier League" \
    --competition-id "GB1" \
    --competition-slug "premier-league" \
    --seasons "2022,2023,2024,2025"
```

### `deploy_club.py`

Master orchestrator that runs the full pipeline.

**Arguments:**
- `--country` (required): Country name
- `--club` (required): Club name
- `--start-date` (optional): Start date for predictions
- `--end-date` (optional): End date for predictions
- `--data-date` (optional): Raw data date folder (YYYYMMDD)
- `--skip-features`: Skip daily features update
- `--force`: Force prediction regeneration

**Example:**
```bash
python production/scripts/deploy_club.py \
    --country "England" \
    --club "Chelsea FC" \
    --start-date "2025-07-01" \
    --end-date "2025-12-05"
```

## Configuration Files

### `production/config/defaults.json`

Shared default settings:
- Window sizes
- Date formats
- Ensemble weights
- Risk thresholds

### `production/deployments/{Country}/{Club}/config.json`

Club-specific configuration:
- Club name and country
- Player IDs list
- Model references (pointing to root `models/` directory)
- Default date ranges

**Example:**
```json
{
  "club_name": "Chelsea FC",
  "country": "England",
  "player_ids": [614258, 346314, ...],
  "models": {
    "random_forest": "../../../models/rf_model_combined_trainval.joblib",
    "gradient_boosting": "../../../models/gb_model_combined_trainval.joblib",
    ...
  },
  "default_date_range_days": 7
}
```

## Data Flow

```
Raw Data (Transfermarkt CSV)
  ↓
Daily Features Update (incremental)
  ↓
Timeline Generation (35-day windows)
  ↓
Prediction Generation (RF + GB + Ensemble)
  ↓
Dashboard Generation (PNG per player)
```

## Path Resolution

All scripts use relative paths:
- Scripts location: `production/scripts/`
- Root directory: `Path(__file__).resolve().parents[2]` (IPM V3 root)
- Club path: `production/deployments/{country}/{club}`
- Models: `models/` (at root, referenced via relative path)

## Raw Data Structure

Raw data is organized as follows:

```
production/raw_data/
└── {country}/
    ├── previous_seasons/          # Old season match data files (shared across dates)
    │   ├── match_12345_2022_2023.csv
    │   ├── match_12345_2023_2024.csv
    │   └── ...
    └── {YYYYMMDD}/                 # Date-stamped folders (daily updates)
        ├── players_profile.csv
        ├── players_career.csv
        ├── injuries_data.csv
        └── match_data/             # Current season match files only
            ├── match_12345_2025_2026.csv
            └── ...
```

**Match Data Organization:**
- **Current season** match files are stored in each date folder's `match_data/` subfolder
- **Previous seasons** match files are stored once in `previous_seasons/` folder (shared across all dates)
- This avoids duplicating historical match data in every date folder
- When loading match data, scripts automatically combine files from both locations

## Notes

- **Models**: Remain at root `models/` directory (not moved)
- **Scripts**: Transversal - changes affect all clubs
- **Raw Data**: Date-stamped for history/rollback capability
- **Match Data**: Current season in date folders, old seasons in `previous_seasons/` (shared)
- **Daily Features**: Incremental updates (appends new days)
- **Separation**: ML training data (`daily_features_output/`) is separate from production data

## Troubleshooting

### No data folder found
- Ensure raw data exists in `production/raw_data/{country}/`
- Check date folder format (YYYYMMDD)
- Use `--data-date` to specify a specific date folder

### Missing player files
- Verify player IDs in `config.json`
- Check that daily features files exist in `daily_features/`
- Ensure raw data contains the player IDs

### Model loading errors
- Verify models exist in root `models/` directory
- Check model file names match `MODEL_CONFIG` in scripts
- Ensure model files are not corrupted

### Import errors
- Ensure you're running scripts from the project root
- Check that `scripts/` directory is in Python path
- Verify all dependencies are installed

