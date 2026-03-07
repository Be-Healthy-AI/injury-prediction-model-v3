# Model report: Skeletal LGBM (model_skeletal)

## Purpose

This model predicts **skeletal injury risk** in the 7-day window before a reference date (D-7 to D-1). It is trained only on **skeletal** injuries (target2): labels are positive when the player had a skeletal injury onset in that window; negatives are non-injury (target1=0, target2=0). Muscular-only rows are excluded.

## Data and labeling (Exp 12)

- **Training data:** Labeled timelines from files matching `*_v4_labeled_muscle_skeletal_only_d7.csv`. Only seasons **2020/21 and later** are used (same as Exp 10).
- **Negatives:** A negative example is kept only if the player had **no** muscular, skeletal, or unknown injury onset in the 35-day window starting at the reference date (Exp 10 rule).
- **Exp 12 exclusion:** All timelines (train and test) whose **reference_date** falls in **[D, D+5]** for any muscular injury onset D are excluded.
- **Test set:** 2025/26 season. For evaluation, test negatives with reference date on or after 2025-11-01 are excluded.

## Training setup

- **Algorithm:** LightGBM.
- **Hyperparameters:** "below" preset.
- **Feature selection:** Iterative process (Exp 12 ranking); this model uses **iteration 21** (top **600** features).
- **Training data usage:** 100% of the training pool (no 80/20 split). Selected for deployment as the chosen iteration 21.

## Performance (at deployment)

- **Test Gini:** 0.4344  
- **Test combined score (0.6×Gini + 0.4×F1):** 0.2854  
- **Test:** Accuracy 0.6447, Precision 0.0326, Recall 0.6100, F1 0.0620, ROC AUC 0.7172.

See `MODEL_METADATA.json` for full metrics and confusion matrices.

## Regenerating the model

From the repository root:

```bash
python models_production/lgbm_muscular_v4/code/modeling/train_iterative_feature_selection_muscular_standalone.py --model skeletal --exp12-data --test-negatives-before 2025-11-01 --only-iteration 21 --no-resume --train-on-full-data --iterative-hp-preset below --chosen --deploy-dir "models_production/lgbm_muscular_v4/model_skeletal"
```

Then deploy to this folder:

```bash
python models_production/lgbm_muscular_v4/code/modeling/deploy_gb_to_production.py --model skeletal
```

(The deploy script uses Exp12 artifacts when present: `lgbm_skeletal_best_iteration_exp12.joblib` and `lgbm_skeletal_best_iteration_features_exp12.json` in `models/`.)
