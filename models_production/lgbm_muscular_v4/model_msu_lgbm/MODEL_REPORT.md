# Model report: MSU LGBM (model_msu_lgbm)

## Purpose

This model predicts **MSU injury risk** (muscular, skeletal, or unknown) in the 7-day window before a reference date (D-7 to D-1). A positive label means the player had an onset of a muscular, skeletal, or unknown-type injury in that window. It is a single “broad” injury model instead of focusing only on muscular.

## Data and labeling

- **Training data:** Labeled timelines from files matching `*_v4_labeled_msu_d7.csv` (Exp 11). **All available seasons** are used (no minimum season; 2018/19 through 2024/25).
- **Negatives:** Same rule as the muscular experiment (Exp 10): a negative is kept only if the player had **no** muscular, skeletal, or unknown injury onset in the 35-day window starting at the reference date.
- **Test set:** 2025/26 season. For evaluation, test negatives with reference date on or after 2025-11-01 are excluded.

## Training setup

- **Algorithm:** LightGBM.
- **Hyperparameters:** “below” preset (more regularized than standard).
- **Feature selection:** Iterative process; this model uses the **22nd iteration** (top **440** features from the Exp 11 ranking, which was initialised from the Exp 10 ranking).
- **Training data usage:** 100% of the training pool (no 80/20 split). Best iteration is chosen by performance on the test set.

## Performance (at deployment)

- **Test Gini:** 0.527  
- **Test combined score (0.6×Gini + 0.4×F1):** 0.341  

See `MODEL_METADATA.json` for full metrics and confusion matrices.

## Regenerating the model

Use the exact command in `TRAIN_COMMAND.txt` (run from the repository root). One command trains and writes the model and feature list; with `--deploy-dir` pointing to this folder, artifacts are written here. For the same layout as the muscular model, this folder also contains `model.joblib` and `columns.json` (aligned layout).
