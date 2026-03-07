# to_be_deleted – Archived content

This folder contains files and folders archived during cleanup of `lgbm_muscular_v4`.  
They are **not** required for the 4 deployed models (muscular LGBM, muscular GB, skeletal LGBM, MSU LGBM) or for audit/traceability.

## Contents

- **archive_cleanup/** – Previous archive (old models, experiments, backups, HP tests)
- **model_muscular_lgbm_backup_20260227_172726/** – Old backup of muscular LGBM
- **model_muscular_lgbm_exp12/** and **model_muscular_lgbm_exp12_main/** – Redundant Exp12 muscular LGBM copies (production is in `model_muscular_lgbm/`)
- **code_modeling_archive/** – Old training/analysis scripts (pre-standalone iterative training)
- **code_modeling/** – One-off scripts (experiment_4_multi_algorithm_760, run_labeling_experiments_500, archive_files, export_one_row_features, investigate_skeletal_prediction_diff)
- **models/** – Legacy/superseded artifacts (non-Exp12 joblibs/features, full-feature models, HP test outputs, plots, etc.)
- **PHASE3_CLEANUP_SUMMARY.md**, **PHASE4_DOCUMENTATION_SUMMARY.md**, **V4_580_PRODUCTION_STATUS.md** – Historical docs

Safe to delete this entire folder once you no longer need the archived content.
