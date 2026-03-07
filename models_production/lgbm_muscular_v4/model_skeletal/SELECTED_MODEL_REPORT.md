# Selected model report

**Model type:** skeletal
**Source:** `--only-iteration 21`

## Selected iteration
- Iteration: 21
- Features used: 600
- Optimize on: test

## Performance (this run)
- Train Gini: 0.8997
- Train F1: 0.0968
- Validation combined score: 0.5786
- Test Gini: 0.4344
- Test F1: 0.0620
- Test combined score: 0.2854

## Artifacts
- Model: `models_production/lgbm_muscular_v4/models/lgbm_skeletal_best_iteration_exp12.joblib`
- Features: `models_production/lgbm_muscular_v4/models/lgbm_skeletal_best_iteration_features_exp12.json`
- Test predictions: `models_production/lgbm_muscular_v4/model_skeletal/test_predictions_from_training_pipeline.csv` (label column: `target2`)

## Deploy
```
python models_production/lgbm_muscular_v4/code/modeling/deploy_gb_to_production.py --model skeletal
```
