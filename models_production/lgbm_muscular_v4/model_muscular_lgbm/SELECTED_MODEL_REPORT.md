# Selected model report

**Model type:** muscular_lgbm
**Source:** `--only-iteration 16`

## Selected iteration
- Iteration: 16
- Features used: 500
- Optimize on: test

## Performance (this run)
- Train Gini: 0.9158
- Train F1: 0.0526
- Validation combined score: 0.5706
- Test Gini: 0.5774
- Test F1: 0.0466
- Test combined score: 0.3651

## Artifacts
- Model: `models_production/lgbm_muscular_v4/models/lgbm_muscular_best_iteration_exp12.joblib`
- Features: `models_production/lgbm_muscular_v4/models/lgbm_muscular_best_iteration_features_exp12.json`
- Test predictions: `models_production/lgbm_muscular_v4/model_muscular_lgbm/test_predictions_from_training_pipeline.csv` (label column: `target1`)

## Deploy
```
python models_production/lgbm_muscular_v4/code/modeling/deploy_gb_to_production.py --model muscular_lgbm
```
