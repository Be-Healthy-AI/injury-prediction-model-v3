## V4 580 Deployment Guide

This document describes how to regenerate data and run the **V4 580 (With Test, Excl 2021/22–2022/23)** muscular injury model in production.

V4 uses a **two-layer daily-features pipeline** plus an enhanced 35‑day timeline generator:

- **Layer 1**: `create_daily_features_v4_enhanced.py`
- **Layer 2**: `enrich_daily_features_v4_layer2.py`
- **Timelines**: `create_35day_timelines_v4_enhanced.py`
- **Final model (muscular, target1)**: `model/model.joblib` (580 features)

All paths below are relative to the repository root (`IPM V3`).

---

## 1. Data Layout and Key Paths

- **Root**: `models_production/lgbm_muscular_v4/`
- **Code**:
  - `code/daily_features/create_daily_features_v4_enhanced.py` (Layer 1)
  - `code/daily_features/enrich_daily_features_v4_layer2.py` (Layer 2)
  - `code/timelines/create_35day_timelines_v4_enhanced.py` (35‑day timelines)
  - `code/modeling/train_v4_580_production.py` (already run – produced final model)
- **Data**:
  - `data/raw_data/`
    - `players_profile.csv`
    - `players_career.csv`
    - `injuries_data.csv`
    - `teams_data.csv`
    - `match_data/match_*.csv`
  - `data/daily_features/` – Layer‑1 daily features
  - `data/daily_features_enriched/` – Layer‑2 enriched daily features
  - `data/timelines/train/` and `data/timelines/test/` – 35‑day timelines
- **Model**:
  - `model/model.joblib` – final V4 580 muscular model
  - `model/columns.json` – ordered list of 580 features
  - `model/MODEL_METADATA.json` – configuration, dataset stats, metrics
  - `model/lgbm_v4_580_metrics_train.json`
  - `model/lgbm_v4_580_metrics_test.json`

---

## 2. End‑to‑End Regeneration Pipeline

This is the **canonical sequence** to fully regenerate data and timelines for deployment.

### 2.1 Prerequisites

- Python environment with:
  - `pandas`, `numpy`, `lightgbm`, `scikit-learn`, `tqdm`, `joblib`
- Raw data available under `data/raw_data/` as described above.

> **Note**: For raw‑data aggregation and V4‑specific sourcing rules, follow the instructions in `README.md` (Step 1: Aggregate Raw Data).

### 2.2 Step 1 – Generate Daily Features (Layer 1)

**Script**: `code/daily_features/create_daily_features_v4_enhanced.py`

This script:

- Reads `data/raw_data/*` (matches, injuries, players, teams, etc.).
- Generates **daily per‑player feature series** with the V4 enhanced feature set.
- Applies V4‑specific logic (e.g. log‑transformed injury recency, workload, recovery, temporal features).
- Writes one CSV per player to `data/daily_features/`:
  - `player_{player_id}_daily_features.csv`

**Typical usage** (from repo root):

```bash
cd "models_production/lgbm_muscular_v4/code/daily_features"
python create_daily_features_v4_enhanced.py \
  --mode full \
  --log-level INFO
```

> **Important**: Use the same configuration (modes/flags) used during the successful V4 580 run to ensure feature parity with training.

### 2.3 Step 2 – Enrich Daily Features (Layer 2)

**Script**: `code/daily_features/enrich_daily_features_v4_layer2.py`

This script:

- Reads **Layer‑1** daily features from `data/daily_features/`.
- Adds Layer‑2 features:
  - Workload windows (3/7/14/28/35 days)
  - Match count windows
  - ACWR (`acwr_min_7_28`)
  - Season‑to‑date workload, ratios to season totals
  - Injury‑history windows (90/365 days)
  - Recovery/rest indicators
  - Interaction features identified in low‑probability analysis (e.g. `inactivity_risk`, `early_season_low_activity`, `preseason_long_rest`, `low_activity_with_history`).
- Writes enriched CSVs to `data/daily_features_enriched/` with **the same file names** as Layer 1.
- Is **restart‑friendly**: by default, it skips files that are already enriched.

**Key CLI options**:

- `--max-files N` – process only N files (for testing).
- `--seed SEED` – seed used when sampling with `--max-files`.
- `--verbose` – print per‑file details.
- `--force` – overwrite existing enriched files.

**Typical usage**:

```bash
cd "models_production/lgbm_muscular_v4/code/daily_features"

# Full enrichment (all players, skip existing files)
python enrich_daily_features_v4_layer2.py

# Test run on a subset of players, with verbose output
python enrich_daily_features_v4_layer2.py --max-files 50 --verbose
```

**Input/Output**:

- Input: `data/daily_features/*.csv`
- Output: `data/daily_features_enriched/*.csv`

---

### 2.4 Step 3 – Generate 35‑Day Timelines

**Script**: `code/timelines/create_35day_timelines_v4_enhanced.py`

This script:

- Consumes V4 daily features (preferring **Layer‑2 enriched** if available).
- Generates **35‑day sliding‑window timelines** for each player/season.
- Builds **dual targets**:
  - `target1` – muscular injuries.
  - `target2` – skeletal injuries.
- Applies **PL‑only filtering** using career and match data.
- Uses **natural target ratios** (no downsampling).
- Creates 5 injury‑timelines per injury (D‑1 to D‑5).
- Flags activity with `has_minimum_activity` (≥ 90 minutes in the 35‑day window).
- Splits by season:
  - **Train**: seasons ≤ 2024/25.
  - **Test**: 2025/26.
- Writes:
  - `data/timelines/train/timelines_35day_season_YYYY_YYYY+1_v4_muscular_train.csv`
  - `data/timelines/test/timelines_35day_season_2025_2026_v4_muscular_test.csv`

**Path selection**:

- Prefers `data/daily_features_enriched/` when present:
  - Logs: `Using ENRICHED daily features directory: ...`
- Falls back to `data/daily_features/` if Layer 2 is missing.

**CLI options**:

- `--test N` – process only the first N players (test mode).
- `--seasons Y1 Y2 ...` – process specific start‑years (e.g., `--seasons 2024 2025`).
- `--season Y` – process a single season (e.g., `--season 2025`).

**Typical usage**:

```bash
cd "models_production/lgbm_muscular_v4/code/timelines"

# Full run – all available seasons, all players
python create_35day_timelines_v4_enhanced.py

# Test run – only first 100 players, specific season
python create_35day_timelines_v4_enhanced.py --test 100 --season 2025
```

**Inputs required** (under `data/raw_data/`):

- `players_profile.csv`
- `players_career.csv`
- `injuries_data.csv`
- `match_data/match_*.csv`

If any of these are missing, the script will raise `FileNotFoundError`.

---

## 3. Model Usage in Production

The **final V4 580 muscular model** is already trained and saved by `train_v4_580_production.py`:

- **Model file**: `model/model.joblib`
- **Feature list**: `model/columns.json` (580 feature names, in order)
- **Metadata**: `model/MODEL_METADATA.json`

### 3.1 Loading the Model

Example (Python):

```python
from pathlib import Path
import joblib
import json
import pandas as pd

ROOT = Path("models_production/lgbm_muscular_v4")

model = joblib.load(ROOT / "model" / "model.joblib")
with open(ROOT / "model" / "columns.json", "r", encoding="utf-8") as f:
    feature_columns = json.load(f)

# df_timelines should be a DataFrame built from the V4 timelines
# and must contain all columns in feature_columns.
X = df_timelines[feature_columns]
probas = model.predict_proba(X)[:, 1]
preds = model.predict(X)
```

**Important**:

- The **same preprocessing logic** used in training is embedded in the timeline generator and modeling pipeline:
  - Categorical cleaning (`clean_categorical_value`),
  - One‑hot encoding,
  - Numeric `fillna(0)`,
  - Feature name sanitization.
- At inference time, you must emulate the **same feature schema**:
  - Work from the 35‑day timelines produced by `create_35day_timelines_v4_enhanced.py`.
  - Select columns in exactly the order of `columns.json`.

---

## 4. Training Configuration (Reference)

From `MODEL_METADATA.json`:

- **Version**: `V4_580_with_test_excl_2021_2022_2022_2023`
- **Training seasons** (after season filtering):
  - `2018_2019`, `2019_2020`, `2020_2021`, `2023_2024`, `2024_2025`, `2025_2026`
- **Excluded seasons**:
  - `2021_2022`, `2022_2023` (low injury‑rate seasons)
- **Train set**:
  - Records: 441,772
  - Positives (target1=1): 2,469
  - Negatives: 439,303
  - Injury rate: ≈ 0.56%
- **Test set (2025/26)**:
  - Records: 34,227
  - Positives (target1=1): 300

**Hyperparameters (LGBMClassifier)**:

- `n_estimators=200`
- `max_depth=10`
- `learning_rate=0.1`
- `min_child_samples=20`
- `subsample=0.8`
- `colsample_bytree=0.8`
- `reg_alpha=0.1`
- `reg_lambda=1.0`
- `class_weight='balanced'`
- `random_state=42`

**Test performance (muscular, target1)**:

- Accuracy: **0.9932**
- Precision: **0.5618**
- Recall: **1.0000**
- F1‑Score: **0.7194**
- ROC AUC: **1.0000**
- Gini: **1.0000**
- Confusion matrix (test):
  - TN: 33,693
  - FP: 234
  - FN: 0
  - TP: 300

---

## 5. Minimal “Runbook” for Deployment

1. **Prepare raw data** in `data/raw_data/` (see main `README.md`).
2. **Generate Layer‑1 daily features**:
   - Run `create_daily_features_v4_enhanced.py` over all required players/seasons.
3. **Generate Layer‑2 enriched daily features**:
   - Run `enrich_daily_features_v4_layer2.py`.
4. **Generate 35‑day timelines**:
   - Run `create_35day_timelines_v4_enhanced.py` (full run, not test mode).
5. **Score with V4 580 model**:
   - Build feature matrix from timelines.
   - Load `model/model.joblib` and `model/columns.json`.
   - Call `predict_proba` / `predict` on the ordered feature columns.

At this point, the V4 580 (With Test, Excl 2021/22–2022/23) model is fully wired to the data pipeline and ready for deployment.

