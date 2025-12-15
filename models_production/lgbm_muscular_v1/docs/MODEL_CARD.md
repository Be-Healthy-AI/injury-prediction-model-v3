# Model Card – `lgbm_muscular_v1`

## Overview

- **Model ID**: `lgbm_muscular_v1`  
- **Type**: LightGBM binary classifier (`LGBMClassifier`)  
- **Task**: Predict risk of **muscular injury within 35 days** for professional football players.  
- **Usage**: Ranked **Top-K alerts per team-week** (decision-support), not fixed probability thresholds.

## Data & Training

- **Training data**:
  - Seasons: **2008_2009–2024_2025**
  - Target ratio: **10%** (muscular injury positives upsampled / negatives downsampled)
  - Target: muscular injuries only, 35-day horizon.
- **Test data**:
  - Season: **2025_2026**
  - Natural injury ratio (~0.6%), muscular injuries only.
- **Feature processing**:
  - Preprocessing implemented in `scripts/train_models_seasonal_combined.py` (`prepare_data`).
  - Feature alignment across train/test.
  - Correlation filtering with threshold **0.8**, final feature count: **1,589**.

Key configuration and hyperparameters are stored in:

- `metadata/training_config.json`
- `model/columns.json` (final feature list and order)

## Performance (Classic Classification Metrics)

From `metadata/metrics_classic.json`:

- **Training (upsampled 10% ratio)**:
  - ROC AUC: **0.9751**, Gini: **0.9502**
  - Precision: **0.4334**, Recall: **0.9883**, F1: **0.6025**
- **Test (2025_2026, natural ratio)**:
  - ROC AUC: **0.8099**, Gini: **0.6198**
  - Precision: **0.0206**, Recall: **0.5375**, F1: **0.0396**

## Decision-Based Evaluation (Team × Week, Top-K)

From `metadata/metrics_decision_based.json`, evaluated on season **2025_2026**:

- Decision unit: **Team × Week**, alert budget **K** players per team-week.
- Metrics reported:
  - **Precision@K** (injuries among Top-K / K)
  - **Recall@K** (injuries captured by Top-K / total injuries)
  - **False Alerts per True Injury**

Example operating points:

- **K = 3**:
  - Precision@3 ≈ **2.0%**
  - Recall@3 ≈ **31.7%**
  - False Alerts / Injury ≈ **54.0**
- **K = 5**:
  - Precision@5 ≈ **1.8%**
  - Recall@5 ≈ **42.9%**
  - False Alerts / Injury ≈ **66.2**
- **K = 10**:
  - Precision@10 ≈ **1.7%**
  - Recall@10 ≈ **52.7%**
  - False Alerts / Injury ≈ **90.4**

These metrics are intended for **operational decision-making**, not generic binary classification.

## Intended Use

- **Primary use**: Support medical and performance staff by **ranking players by injury risk** and selecting the **Top-K highest-risk players per team-week** for closer monitoring or intervention.
- **Not for**:
  - Automated medical diagnosis.
  - Making irreversible decisions without human oversight.
  - Interpretation at a fixed probability threshold (e.g. 0.5) without ranking.

## How to Use in a Pipeline (Conceptual)

1. **Input**: Daily player-level dataset with the same raw schema used in training (IDs, dates, clubs, match history, etc.).
2. **Preprocess** using the same `prepare_data` logic as in `train_models_seasonal_combined.py`.
3. **Align features** to `model/columns.json`:
   - Add missing columns with 0.
   - Drop extra columns.
   - Order columns exactly as in `columns.json`.
4. **Score**:
   - Load `model/model.joblib` with `joblib.load`.
   - Call `predict_proba(X_aligned)[:, 1]` to get risk scores.
5. **Decision logic** (downstream):
   - For each team-week, rank players by risk and select Top-K.
   - Evaluate and monitor using decision-based metrics (Precision@K, Recall@K, false alerts per injury).

## Limitations & Caveats

- **Class imbalance**: Injuries are rare, especially in the test season (~0.6% positives). Precision at realistic K values is therefore low, even for a strong ranking model.
- **Data coverage**: Model is trained on public data and may not capture all relevant medical / biomechanical factors.
- **Temporal drift**: Injury patterns and squad compositions can change over seasons; performance should be revalidated periodically (e.g., yearly).

## Files in This Bundle

- `model/model.joblib` – Trained LightGBM model.
- `model/columns.json` – Ordered list of feature names expected by the model.
- `metadata/training_config.json` – Training configuration and hyperparameters.
- `metadata/metrics_classic.json` – Classic classification metrics (train/test).
- `metadata/metrics_decision_based.json` – Decision-based Top-K metrics.


