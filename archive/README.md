# Archive Directory

This directory contains archived files from the project cleanup performed on 2025-11-30.

## Archive Structure

- **models/old_versions/** - Old model versions (v3, v4, phase1, phase2, seasonal, stable_features, etc.)
- **experiments/old_experiments/** - Old experiment results (target ratio experiments, time window experiments, threshold optimizations, etc.)
- **scripts/old_training/** - Old training scripts (v3, v4, seasonal split, etc.)
- **scripts/old_optimization/** - Old optimization scripts (threshold optimization, ensemble optimization, etc.)
- **scripts/old_analysis/** - Old analysis scripts (comparison scripts, feature analysis, etc.)
- **analysis/old_analysis/** - Old analysis reports (v4 comparisons, threshold optimizations, etc.)
- **backups/** - Backup timeline files from various experiments
- **feature_sets/** - Old feature selection results

## Current Active Models

The following models are currently active and NOT archived:

- `models/gb_model_combined_trainval.joblib` - Final GB model (trained on combined train+val, 8% target ratio)
- `models/rf_model_combined_trainval.joblib` - Final RF model (trained on combined train+val, 8% target ratio)

## Current Active Datasets

- `timelines_35day_enhanced_balanced_v4_train.csv` - Training data (<= 2024-06-30)
- `timelines_35day_enhanced_balanced_v4_val.csv` - Validation data (2024/25 season)
- `timelines_35day_enhanced_balanced_v4_test.csv` - Test data (2025/26 season)

## Archive Date

2025-11-30



