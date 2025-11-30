# Production Predictions Pipeline

This directory contains the complete pipeline for generating daily injury predictions for production use.

## Folder Structure

- **raw_data/**: Place your updated Excel files here
  - `players_profile.xlsx`
  - `injuries_data.xlsx`
  - `match_data.xlsx`
  - `teams_data.xlsx`
  - `competition_data.xlsx`
  - `players_career.xlsx` (optional)

- **daily_features/**: Incrementally updated daily features for each player
  - `player_{id}_daily_features.csv`

- **timelines/**: 35-day timelines for predictions
  - `player_{id}_timelines_{date}.csv`

- **predictions/**: Daily predictions output
  - `predictions_{date}.csv` (all players combined)
  - `player_{id}_predictions.csv` (individual files)

- **reports/**: Summary reports
  - `report_{date}.md`

- **dashboards/**: Interactive dashboards
  - `dashboard_{date}.html`

## Usage

### Step 1: Update Raw Data
Place your updated Excel files in `raw_data/` directory.

### Step 2: Run Pipeline
```bash
# Run complete pipeline
py -3.11 scripts/run_production_pipeline.py

# Or run individual steps
py -3.11 scripts/update_daily_features_incremental.py
py -3.11 scripts/generate_timelines_production.py
py -3.11 scripts/generate_predictions_production.py
py -3.11 scripts/generate_report_production.py
py -3.11 scripts/generate_dashboard_production.py
```

## Notes

- The pipeline automatically detects new players (no existing daily features file) vs existing players
- For existing players, only new days are appended to avoid regenerating entire files
- All outputs are organized by date for easy tracking

