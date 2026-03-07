# Deployment guidelines – model_msu_lgbm (iteration 22)

Use these steps when deploying the model so production predictions match the training pipeline.

## 1. Training (already done)

- Model was trained with `--only-iteration 22` and the selected options (e.g. Exp 10, D-7, train ≥ 2020/21, test negatives with ref_date before 2025-11-01).
- Artifacts produced in `models_production/lgbm_muscular_v4/models/`:
  - `lgbm_muscular_best_iteration_exp11.joblib`
  - `lgbm_muscular_best_iteration_features_exp11.json`
- Test predictions from this training run have been exported to:
  - `models_production/lgbm_muscular_v4/model_msu_lgbm/test_predictions_from_training_pipeline.csv`
  - Columns: `player_id`, `reference_date`, `predicted_probability`, `target_msu`
  - Use this file to validate that production predictions match the training pipeline.

## 2. Prepare artifacts for the deploy script

The deploy script expects the correct artifact names. In `models_production/lgbm_muscular_v4/models/`:

Artifacts are `lgbm_muscular_best_iteration_exp11.joblib` and `lgbm_muscular_best_iteration_features_exp11.json`. The deploy script reads these directly; no copy needed. Proceed to step 3.

## 3. Deploy to production

From the repository root:

```
python models_production/lgbm_muscular_v4/code/modeling/deploy_gb_to_production.py --model msu_lgbm
```

This refreshes `models_production/lgbm_muscular_v4/model_msu_lgbm/` (e.g. `model.joblib`, `columns.json`, `MODEL_METADATA.json`).

## 4. Encoding schema (production parity)

Production uses `encoding_schema.json` in this folder so MSU predictions match the training pipeline (same pattern as muscular/skeletal). The training export (when you run with `--only-iteration` or `--export-best` for MSU) writes `encoding_schema.json` automatically. If you have an existing deployment without it, generate it with:

```
python models_production/lgbm_muscular_v4/code/modeling/generate_msu_encoding_schema.py
```

Use `--timelines PATH` if the default test timeline file is not present.
