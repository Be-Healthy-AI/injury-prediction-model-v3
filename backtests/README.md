# Backtesting Workflow

This directory contains all artefacts needed to run retrospective backtesting
independent from the production training pipeline. The latest scenario focuses
on every player who recorded an injury in the 2025‑26 season (plus the five
previously analysed players) and evaluates the 45 days leading up to each
injury, excluding the injury date itself.

## Scenario: `players_2025_45d`

- **Configuration file:** `config/players_2025_45d.json`  
  Generated with `scripts/backtests/build_window_config.py --include-baseline`.
- **Prediction window:** 45 days (d‑45 … d‑1). The injury day (`d0`) is excluded.
- **History buffer:** 80 days per injury (enough to build 35-day timelines for
  the earliest reference date).
- **Outputs:** All derived artefacts live under `*/2025_45d/` folders.

The JSON config encodes every injury entry with:

- `entry_id` – stable identifier (`player_<id>_<injury_date>`)
- `window_start` / `window_end` – prediction horizon (45 days)
- `history_start` / `history_end` – trimming range for daily features (80 days)
- Injury metadata (season, injury type, injury date)

## End-to-end pipeline

Run the following scripts from the project root to regenerate the scenario:

1. **Build config (optional)**  
   ```powershell
   py -3.11 scripts\backtests\build_window_config.py --include-baseline
   ```
2. **Trim daily features**  
   ```powershell
   py -3.11 scripts\backtests\prepare_daily_features.py `
       --config backtests\config\players_2025_45d.json `
       --output-dir backtests\daily_features\2025_45d
   ```
3. **Generate 35-day timelines**  
   ```powershell
    py -3.11 scripts\backtests\generate_timelines.py `
        --config backtests\config\players_2025_45d.json `
        --daily-features-dir backtests\daily_features\2025_45d `
        --output-dir backtests\timelines\2025_45d
   ```
4. **Produce predictions (RF & GB)**  
   ```powershell
   py -3.11 scripts\backtests\run_predictions.py `
       --config backtests\config\players_2025_45d.json `
       --timelines-dir backtests\timelines\2025_45d `
       --output-dir backtests\predictions\2025_45d
   ```
5. **Summaries & visuals**  
   ```powershell
   py -3.11 scripts\backtests\summarize_results.py `
       --config backtests\config\players_2025_45d.json `
       --predictions-dir backtests\predictions\2025_45d `
       --daily-features-dir backtests\daily_features\2025_45d `
       --output-dir backtests\visualizations\2025_45d
   ```

## Output structure

- `daily_features/2025_45d/` – trimmed daily feature subsets (80-day span)
- `timelines/2025_45d/` – 35-day window timelines with `entry_id` column
- `predictions/2025_45d/<model>/` – daily probabilities per model  
  (the Gradient Boosting folder now contains `explanations/` with SHAP top-feature JSON files)
- `visualizations/2025_45d/` – combined CSVs, Markdown + enriched PNGs (risk classes, body-part focus, trends)
- `config/` – JSON definitions for reproducible backtesting scenarios

## Insight layers

- **Body-part likelihoods:** trained via `scripts/backtests/train_insight_models.py`, the gradient boosting summary ranks the three most exposed body regions per entry.
- **Severity outlook:** the same pipeline outputs a categorical severity tier (minor → long term) with probability distribution.
- **Feature attribution:** SHAP explanations highlight the top contributing features (stored per day, summarised on the final window day).
- **Risk classes & trends:** charts and summaries translate probabilities into four classes (Low → Critical) and expose jumps, slopes, and sustained elevations.

## Notes

- Each CSV is written with `utf-8-sig` encoding for Excel compatibility.
- Timeline generation skips dates that lack a full 35-day history.
- Prediction scripts reuse the production encoders; warnings about fragmented
  DataFrames are expected and safe to ignore for batch inference.

