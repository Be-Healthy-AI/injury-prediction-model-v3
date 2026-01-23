# Model Card – `lgbm_muscular_v2`

## Overview

- **Model ID**: `lgbm_muscular_v2`  
- **Type**: LightGBM binary classifier (`LGBMClassifier`)  
- **Task**: Predict risk of **muscular injury within 35 days** for professional football players.  
- **Usage**: Ranked **Top-K alerts per team-week** (decision-support), not fixed probability thresholds.
- **Version Relationship**: This is the production version trained on **all available data** (including 2025-2026). `lgbm_muscular_v1` remains the audited benchmark with external test set.

## Data & Training

- **Training data**:
  - Seasons: **2008_2009–2025_2026** (ALL seasons, including previous test season)
  - Target ratio: **10%** (muscular injury positives upsampled / negatives downsampled)
  - Target: muscular injuries only, 35-day horizon.
  - Total samples: **147,990** (14,799 positives, 133,191 negatives)
- **Test data** (for evaluation/monitoring only):
  - Season: **2025_2026** (natural injury ratio ~0.6%, muscular injuries only)
  - **Note**: This season was included in training, so test metrics reflect in-sample performance, not true generalization.
- **Feature processing**:
  - Preprocessing implemented in `scripts/train_models_seasonal_combined.py` (`prepare_data`).
  - Feature alignment across train/test.
  - Correlation filtering with threshold **0.8**, final feature count: **4,059** (LightGBM internally used 2,003 features).

Key configuration and hyperparameters are stored in:

- `metadata/training_config.json`
- `model/columns.json` (final feature list and order)

## Performance (Classic Classification Metrics)

From `metadata/metrics_classic.json`:

- **Training (upsampled 10% ratio)**:
  - ROC AUC: **0.9688**, Gini: **0.9377**
  - Precision: **0.4197**, Recall: **0.9803**, F1: **0.5877**
- **Test (2025_2026, natural ratio, in-sample)**:
  - ROC AUC: **0.9674**, Gini: **0.9347**
  - Precision: **0.0384**, Recall: **0.9677**, F1: **0.0738**
  - **Note**: Low precision at threshold 0.5 is expected due to extreme class imbalance (0.6% base rate). The high Gini indicates excellent ranking capability.

## Decision-Based Evaluation (Team × Week, Top-K)

From `metadata/metrics_decision_based.json`, evaluated on season **2025_2026**:

- Decision unit: **Team × Week**, alert budget **K** players per team-week.
- Metrics reported:
  - **Precision@K** (injuries among Top-K / K)
  - **Recall@K** (injuries captured by Top-K / total injuries)
  - **False Alerts per True Injury**

Example operating points:

- **K = 3**:
  - Precision@3 ≈ **3.0%**
  - Recall@3 ≈ **50.9%**
  - False Alerts / Injury ≈ **33.3**
- **K = 5**:
  - Precision@5 ≈ **2.7%**
  - Recall@5 ≈ **68.4%**
  - False Alerts / Injury ≈ **41.2**
- **K = 10**:
  - Precision@10 ≈ **2.2%**
  - Recall@10 ≈ **82.5%**
  - False Alerts / Injury ≈ **57.4**
- **K = 20**:
  - Precision@20 ≈ **1.8%**
  - Recall@20 ≈ **92.5%**
  - False Alerts / Injury ≈ **74.7**

These metrics are intended for **operational decision-making**, not generic binary classification.

## Differences from v1

- **Training data**: v2 includes **2025_2026** in training (v1 held it out as test).
- **Validation strategy**: v2 trained on all data without external hold-out (v1 had external test).
- **Purpose**: v2 is optimized for **production deployment** with maximum data utilization. v1 remains the **audited benchmark** for comparison.
- **Performance**: v2 shows similar or slightly improved Gini (0.9347 vs 0.6198 on natural 2025-2026), but note that v2's test metrics are in-sample while v1's were out-of-sample.

## Intended Use

- **Primary use**: Support medical and performance staff by **ranking players by injury risk** and selecting the **Top-K highest-risk players per team-week** for closer monitoring or intervention.
- **Production deployment**: This model is intended for daily prediction pipelines where all available historical data is used for training.
- **Not for**:
  - Automated medical diagnosis.
  - Making irreversible decisions without human oversight.
  - Interpretation at a fixed probability threshold (e.g. 0.5) without ranking.
  - True out-of-sample evaluation (use v1 for that).

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

- **Class imbalance**: Injuries are rare, especially in natural data (~0.6% positives). Precision at realistic K values is therefore low, even for a strong ranking model.
- **Data coverage**: Model is trained on public data and may not capture all relevant medical / biomechanical factors.
- **Temporal drift**: Injury patterns and squad compositions can change over seasons; performance should be revalidated periodically (e.g., yearly).
- **In-sample evaluation**: Test metrics on 2025-2026 reflect in-sample performance since this season was included in training. For true generalization assessment, monitor performance on future seasons (2026-2027 and beyond).

## Files in This Bundle

- `model/model.joblib` – Trained LightGBM model.
- `model/columns.json` – Ordered list of feature names expected by the model.
- `metadata/training_config.json` – Training configuration and hyperparameters.
- `metadata/metrics_classic.json` – Classic classification metrics (train/test).
- `metadata/metrics_decision_based.json` – Decision-based Top-K metrics.
- `code/modeling/` – Snapshot of training and evaluation scripts.

