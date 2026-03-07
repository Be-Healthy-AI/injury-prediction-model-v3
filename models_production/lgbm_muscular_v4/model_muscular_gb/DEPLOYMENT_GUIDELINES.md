# Deployment guidelines – model_muscular_gb (iteration 4)

Use these steps when deploying the model so production predictions match the training pipeline.

## 1. Training (already done)

- Model was trained with `--only-iteration 4` and the selected options (e.g. Exp 10, D-7, train ≥ 2020/21, test negatives with ref_date before 2025-11-01).
- Artifacts produced in `models_production/lgbm_muscular_v4/models/`:
  - `lgbm_muscular_best_iteration_gb_exp12.joblib`
  - `lgbm_muscular_best_iteration_features_gb_exp12.json`
- Test predictions from this training run have been exported to:
  - `models_production/lgbm_muscular_v4/model_muscular_gb/test_predictions_from_training_pipeline.csv`
  - Columns: `player_id`, `reference_date`, `predicted_probability`, `target1`
  - Use this file to validate that production predictions match the training pipeline.

## 2. Prepare artifacts for the deploy script

The deploy script expects the correct artifact names. In `models_production/lgbm_muscular_v4/models/`:

- `lgbm_muscular_best_iteration_gb_exp12.joblib` → `lgbm_muscular_best_iteration.joblib`
- `lgbm_muscular_best_iteration_features_gb_exp12.json` → `lgbm_muscular_best_iteration_features.json`

(Replace SUFFIX with the actual suffix used in this run: `_gb_exp12`.)

## 3. Deploy to production

From the repository root:

```
python models_production/lgbm_muscular_v4/code/modeling/deploy_gb_to_production.py --model muscular_gb
```

This refreshes `models_production/lgbm_muscular_v4/model_muscular_gb/` (e.g. `model.joblib`, `columns.json`, `MODEL_METADATA.json`).
