#!/usr/bin/env python3
"""
Script to archive experimental files in V4
"""
import shutil
import os
from pathlib import Path

# Use absolute paths based on current working directory or script location
if __file__:
    SCRIPT_DIR = Path(__file__).resolve().parent
else:
    # Fallback: assume we're in the modeling directory
    SCRIPT_DIR = Path.cwd()

# If script is in archive/, go up one level to modeling/
if SCRIPT_DIR.name == 'archive':
    MODELING_DIR = SCRIPT_DIR.parent
    ARCHIVE_DIR = SCRIPT_DIR
else:
    MODELING_DIR = SCRIPT_DIR
    ARCHIVE_DIR = SCRIPT_DIR / 'archive'

# Calculate root directory (go up from modeling: code/modeling -> code -> lgbm_muscular_v4 -> models_production -> IPM V3)
ROOT_DIR = MODELING_DIR.parent.parent.parent.parent
MODELS_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'models'
MODELS_ARCHIVE_DIR = MODELS_DIR / 'archive'

print(f"MODELING_DIR: {MODELING_DIR}")
print(f"ARCHIVE_DIR: {ARCHIVE_DIR}")
print(f"MODELS_DIR: {MODELS_DIR}")
print(f"MODELS_ARCHIVE_DIR: {MODELS_ARCHIVE_DIR}")

# Create archive directories
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
MODELS_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
(MODELS_ARCHIVE_DIR / 'old_models').mkdir(parents=True, exist_ok=True)
(MODELS_ARCHIVE_DIR / 'comparison_results').mkdir(parents=True, exist_ok=True)

# Files to archive from modeling directory
files_to_archive = [
    'train_iterative_feature_selection.py',
    'train_iterative_feature_selection_enhanced.py',
    'train_iterative_feature_selection_no_nested_importlib.py',
    'compare_v3_v4.py',
    'compare_v4_baseline_vs_enriched.py',
    'train_all_models_v4_comparison.py',
    'train_v4_620_features_fair_comparison.py',
    'train_lgbm_v4_dual_targets_natural.py',
    'train_lgbm_v4_enriched_comparison.py',
    'train_lgbm_target1_with_test.py',
    'train_with_feature_subset.py',
    'analyze_model_performance.py',
    'check_iterative_training_status.py',
    'display_ensemble_results.py',
    'optimize_ensemble_v4.py',
    'rank_features_by_importance.py',
    'transform_enriched_metrics.py',
    'run_feature_ranking.bat',
    'run_iterative_training_enhanced.bat',
    'run_iterative_training_no_nested.bat',
    'ITERATIVE_FEATURE_SELECTION_README.md',
    'README_iterative_training.md',
    'analyze_calibration_v4_enriched.py',
    'analyze_feature_importance_enriched.py',
    'analyze_low_probability_injuries.py',
]

print("Archiving experimental scripts...")
moved_count = 0
for filename in files_to_archive:
    src = MODELING_DIR / filename
    if src.exists():
        dst = ARCHIVE_DIR / filename
        shutil.move(str(src), str(dst))
        print(f"  Moved: {filename}")
        moved_count += 1
    else:
        print(f"  Not found: {filename}")

print(f"\n[OK] Moved {moved_count} files to modeling archive")

# Model files to archive
model_files_to_archive = [
    ('lgbm_muscular_v4_natural_model1.joblib', 'old_models'),
    ('lgbm_muscular_v4_natural_model1_columns.json', 'old_models'),
    ('lgbm_muscular_v4_natural_model2.joblib', 'old_models'),
    ('lgbm_muscular_v4_natural_model2_columns.json', 'old_models'),
    ('lgbm_muscular_v4_natural_metrics.json', 'old_models'),
    ('lgbm_muscular_v4_enriched_model1.joblib', 'old_models'),
    ('lgbm_muscular_v4_enriched_model1_columns.json', 'old_models'),
    ('lgbm_muscular_v4_enriched_model2.joblib', 'old_models'),
    ('lgbm_muscular_v4_enriched_model2_columns.json', 'old_models'),
    ('lgbm_muscular_v4_enriched_metrics.json', 'old_models'),
    ('v4_620_comparison_metrics.json', 'comparison_results'),
    ('v4_620_comparison_table.md', 'comparison_results'),
]

print("\nArchiving old model files...")
moved_models = 0
for filename, subdir in model_files_to_archive:
    src = MODELS_DIR / filename
    if src.exists():
        dst = MODELS_ARCHIVE_DIR / subdir / filename
        shutil.move(str(src), str(dst))
        print(f"  Moved: {filename} -> {subdir}/")
        moved_models += 1
    else:
        print(f"  Not found: {filename}")

# Move directories
dirs_to_archive = [
    ('enriched_comparison', 'old_models'),
    ('comparison', 'old_models'),
]

print("\nArchiving model directories...")
moved_dirs = 0
for dirname, subdir in dirs_to_archive:
    src = MODELS_DIR / dirname
    if src.exists() and src.is_dir():
        dst = MODELS_ARCHIVE_DIR / subdir / dirname
        shutil.move(str(src), str(dst))
        print(f"  Moved: {dirname}/ -> {subdir}/")
        moved_dirs += 1
    else:
        print(f"  Not found: {dirname}/")

print(f"\n[OK] Moved {moved_models} model files and {moved_dirs} directories")
print("\n[OK] Archiving complete!")
