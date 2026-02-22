# V4 Production Models – Three-Model Layout

This folder can hold **three** production models at once. Each has its own directory with the same structure.

## Production directories

| Directory | Model | Features | Description |
|-----------|--------|----------|-------------|
| **model_muscular_lgbm/** | Muscular LGBM | 500 | Muscular injury (target1), LightGBM, below_strong, 100% data |
| **model_muscular_gb/**   | Muscular GB   | 350 | Muscular injury (target1), GradientBoosting, below, 100% data |
| **model_skeletal/**     | Skeletal LGBM | 60  | Skeletal injury (target2), LightGBM, iteration 3 |

Each directory contains:
- **model.joblib** – Trained model (load with joblib).
- **columns.json** – List of feature names in order (same as training).
- **MODEL_METADATA.json** – Version, name, metrics, training config, deployment info.

## Deploying a model

From `code/modeling/`:

```bash
# Deploy muscular LGBM (500 feat)
python deploy_gb_to_production.py --model muscular_lgbm

# Deploy muscular GB (350 feat)
python deploy_gb_to_production.py --model muscular_gb

# Deploy skeletal LGBM (60 feat)
python deploy_gb_to_production.py --model skeletal
```

Legacy (still supported):
- `--algorithm lgbm` → deploys to **model_muscular_lgbm**
- `--algorithm gb`   → deploys to **model_muscular_gb**

## Exporting before deploy

- **Muscular (LGBM or GB):**  
  `python train_iterative_feature_selection_muscular_standalone.py --algorithm lgbm --export-best`  
  (or `--algorithm gb --export-best --export-iteration 7 --export-hp-preset below --train-on-full-data` for GB.)

- **Skeletal:**  
  `python train_iterative_feature_selection_skeletal_standalone.py --export-best`

Export writes under `models/` (e.g. `lgbm_muscular_best_iteration.joblib`, `lgbm_skeletal_best_iteration.joblib`). Deploy then copies into the chosen production directory.

## Source artifacts (models/)

| Model | Joblib | Features JSON |
|-------|--------|----------------|
| Muscular LGBM | lgbm_muscular_best_iteration.joblib | lgbm_muscular_best_iteration_features.json |
| Muscular GB   | lgbm_muscular_best_iteration_gb.joblib | lgbm_muscular_best_iteration_features_gb.json |
| Skeletal      | lgbm_skeletal_best_iteration.joblib | lgbm_skeletal_best_iteration_features.json |
