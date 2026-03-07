# Selected model report

**Model type:** muscular_gb
**Source:** `--only-iteration 4`

## Selected iteration
- Iteration: 4
- Features used: 350
- Optimize on: test

## Performance (this run)
- Train Gini: 0.7387
- Train F1: 0.0044
- Validation combined score: 0.4450
- Test Gini: 0.5522
- Test F1: 0.0000
- Test combined score: 0.3313

## Artifacts
- Model: `models_production/lgbm_muscular_v4/models/lgbm_muscular_best_iteration_gb_exp12.joblib`
- Features: `models_production/lgbm_muscular_v4/models/lgbm_muscular_best_iteration_features_gb_exp12.json`
- Test predictions: `models_production/lgbm_muscular_v4/model_muscular_gb/test_predictions_from_training_pipeline.csv` (label column: `target1`)

## Deploy
```
python models_production/lgbm_muscular_v4/code/modeling/deploy_gb_to_production.py --model muscular_gb
```
