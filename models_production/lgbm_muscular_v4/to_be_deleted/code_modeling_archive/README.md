# Archive - Experimental Training Scripts

This directory contains experimental and development scripts used during V4 model development and comparison.

## Contents

### Iterative Feature Selection Scripts
- `train_iterative_feature_selection.py` - Original iterative selection script
- `train_iterative_feature_selection_enhanced.py` - Enhanced version with nested CV
- `train_iterative_feature_selection_no_nested_importlib.py` - No nested CV, uses importlib
- `train_iterative_feature_selection_standalone.py` - **KEEP THIS ONE** - Final working standalone version (used for production feature selection)

### Comparison and Analysis Scripts
- `compare_v3_v4.py` - V3 vs V4 comparison
- `compare_v4_baseline_vs_enriched.py` - Baseline vs enriched features comparison
- `train_all_models_v4_comparison.py` - Comparison of all V4 model variants
- `train_v4_620_features_fair_comparison.py` - Fair comparison script (renamed to reflect 580 features)

### Training Scripts (Superseded)
- `train_lgbm_v4_dual_targets_natural.py` - Natural ratio training (superseded)
- `train_lgbm_v4_enriched_comparison.py` - Enriched comparison training (superseded)
- `train_lgbm_target1_with_test.py` - Single target training (superseded)
- `train_with_feature_subset.py` - Generic feature subset training

### Utility Scripts
- `analyze_model_performance.py` - Model performance analysis
- `check_iterative_training_status.py` - Check training status
- `display_ensemble_results.py` - Display ensemble results
- `optimize_ensemble_v4.py` - Ensemble optimization
- `rank_features_by_importance.py` - Feature importance ranking
- `transform_enriched_metrics.py` - Metrics transformation

### Batch Files
- `run_feature_ranking.bat` - Batch file for feature ranking
- `run_iterative_training_enhanced.bat` - Batch file for enhanced training
- `run_iterative_training_no_nested.bat` - Batch file for no-nested training

## Production Scripts (NOT in Archive)

The following scripts remain in the main `modeling/` directory as they are production-ready:

- `train_v4_580_production.py` - **FINAL PRODUCTION MODEL TRAINING SCRIPT**
  - Trains and saves the V4 580 (With Test, Excl 2021/22-2022/23) model
  - Creates all production model files in `model/` directory

## Note

All scripts in this archive were used during development and are kept for reference. The final production model was trained using `train_v4_580_production.py`.
