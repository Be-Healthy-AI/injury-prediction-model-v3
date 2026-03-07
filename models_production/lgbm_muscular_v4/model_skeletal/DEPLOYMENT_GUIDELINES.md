# Deployment guidelines – model_skeletal (iteration 21)

Use these steps when deploying the model so production predictions match the training pipeline.

## 1. Training (already done)

- Model was trained with `--only-iteration 21` and the selected options (e.g. Exp 10, D-7, train ≥ 2020/21, test negatives with ref_date before 2025-11-01).
- Artifacts produced in `models_production/lgbm_muscular_v4/models/`:
  - `lgbm_skeletal_best_iteration_exp12.joblib`
  - `lgbm_skeletal_best_iteration_features_exp12.json`
- Test predictions from this training run have been exported to:
  - `models_production/lgbm_muscular_v4/model_skeletal/test_predictions_from_training_pipeline.csv`
  - Columns: `player_id`, `reference_date`, `predicted_probability`, `target2`
  - Use this file to validate that production predictions match the training pipeline.

## 2. Prepare artifacts for the deploy script

The deploy script expects the correct artifact names. In `models_production/lgbm_muscular_v4/models/`:

- `lgbm_skeletal_best_iteration_exp12.joblib` → `lgbm_skeletal_best_iteration.joblib`
- `lgbm_skeletal_best_iteration_features_exp12.json` → `lgbm_skeletal_best_iteration_features.json`

(Artifacts live in `models_production/lgbm_muscular_v4/models/`; copy to expected names if using a suffix.)

## 3. Deploy to production

From the repository root:

```
python models_production/lgbm_muscular_v4/code/modeling/deploy_gb_to_production.py --model skeletal
```

This refreshes `models_production/lgbm_muscular_v4/model_skeletal/` (e.g. `model.joblib`, `columns.json`, `MODEL_METADATA.json`).

## 4. Encoding schema (production parity)

Production uses `encoding_schema.json` in this folder so predictions match the training pipeline (same pattern as muscular). The training export (when you run with `--export-best`) writes `encoding_schema.json` automatically. If you have an existing deployment without it, generate it with:

```
python models_production/lgbm_muscular_v4/code/modeling/generate_skeletal_encoding_schema.py
```

Use `--timelines PATH` if the default test timeline file is not present.
