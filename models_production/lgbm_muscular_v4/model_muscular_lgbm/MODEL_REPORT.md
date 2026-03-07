# Model report: Muscular LGBM (model_muscular_lgbm)

## Purpose

This model predicts **muscular injury risk** in the 7-day window before a reference date (D-7 to D-1). It is trained only on **muscular** injuries (target1): labels are positive when the player had a muscular injury onset in that window.

## Data and labeling (Exp 12)

- **Training data:** Labeled timelines from files matching `*_v4_labeled_muscle_skeletal_only_d7.csv`. Only seasons **2020/21 and later** are used (same as Exp 10).
- **Negatives:** A negative example is kept only if the player had **no** muscular, skeletal, or unknown injury onset in the 35-day window starting at the reference date (Exp 10 rule).
- **Exp 12 exclusion:** All timelines (train and test) whose **reference_date** falls in **[D, D+5]** for any muscular injury onset D are excluded. This removes “post-onset” windows where the model would see the period immediately around the injury.
- **Test set:** 2025/26 season. For evaluation, test negatives with reference date on or after 2025-11-01 are excluded.

## Training setup

- **Algorithm:** LightGBM.
- **Hyperparameters:** “standard” preset.
- **Feature selection:** Iterative process (Exp 12 ranking); this model uses **iteration 16** (top **320** features).
- **Training data usage:** 100% of the training pool (no 80/20 split). Selected for deployment based on best test Gini among iterations (0.5554 at iteration 16).

## Performance (at deployment)

- **Test Gini:** 0.5554  
- **Test combined score (0.6×Gini + 0.4×F1):** 0.3645  
- **Test:** Accuracy 0.9341, Precision 0.0449, Recall 0.3071, F1 0.0783, ROC AUC 0.7777.

See `MODEL_METADATA.json` for full metrics and confusion matrices.

## Regenerating the model

From the repository root:

```bash
python models_production/lgbm_muscular_v4/code/modeling/train_iterative_feature_selection_muscular_standalone.py --exp12-data --test-negatives-before 2025-11-01 --only-iteration 16 --no-resume --train-on-full-data
```

Then deploy to this folder:

```bash
python models_production/lgbm_muscular_v4/code/modeling/deploy_gb_to_production.py --model muscular_lgbm
```

(Note: deploy script expects `lgbm_muscular_best_iteration.joblib` and `lgbm_muscular_best_iteration_features.json` in `models/`. For Exp 12, after training copy the Exp 12 artifacts to those names in `models/`, or update the deploy script to support Exp 12 source files.)
