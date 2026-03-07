# Selected model report

**Model type:** msu_lgbm
**Source:** `--only-iteration 22`

## Selected iteration
- Iteration: 22
- Features used: 620
- Optimize on: test

## Performance (this run)
- Train Gini: 0.8115
- Train F1: 0.0714
- Validation combined score: 0.5154
- Test Gini: 0.5358
- Test F1: 0.0619
- Test combined score: 0.3463

## Artifacts
- Model: `models_production/lgbm_muscular_v4/models/lgbm_muscular_best_iteration_exp11.joblib`
- Features: `models_production/lgbm_muscular_v4/models/lgbm_muscular_best_iteration_features_exp11.json`
- Test predictions: `models_production/lgbm_muscular_v4/model_msu_lgbm/test_predictions_from_training_pipeline.csv` (label column: `target_msu`)

## Deploy
```
python models_production/lgbm_muscular_v4/code/modeling/deploy_gb_to_production.py --model msu_lgbm
```
