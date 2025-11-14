# Project Cleanup Plan

## Files to Delete (Soft Delete - Move to Archive or Delete)

### 1. Old Model Files (Replaced by Final Versions)
**Location: `models/`**
- `model_v3_gradient_boosting_100percent.pkl` → Replaced by `model_v3_gradient_boosting_final_100percent.pkl`
- `model_v3_random_forest_100percent.pkl` → Replaced by `model_v3_random_forest_final_100percent.pkl`
- `model_v3_gradient_boosting_validated.pkl` → Intermediate training file
- `model_v3_random_forest_validated.pkl` → Intermediate training file
- `model_v3_gradient_boosting_optimized.pkl` → Intermediate optimization file
- `model_v3_random_forest_optimized.pkl` → Intermediate optimization file
- `model_v3_gb_100percent_training_columns.json` → Replaced by `model_v3_gb_final_columns.json`
- `model_v3_rf_100percent_training_columns.json` → Replaced by `model_v3_rf_final_columns.json`
- `model_v3_gb_validated_training_columns.json` → Intermediate file
- `model_v3_rf_validated_training_columns.json` → Intermediate file
- `model_v3_gb_optimized_training_columns.json` → Intermediate file
- `model_v3_rf_optimized_training_columns.json` → Intermediate file
- `model_v3_gb_optimized_params.json` → Can be kept for reference or deleted
- `model_v3_rf_optimized_params.json` → Can be kept for reference or deleted

### 2. Old Backtest Files (Replaced by Organized 2025_45d Structure)
**Location: `backtests/daily_features/`**
- `player_200512_daily_features_20240401_20240531.csv` → Now in `2025_45d/`
- `player_258027_daily_features_20250901_20251029.csv` → Now in `2025_45d/`
- `player_452607_daily_features_20250101_20250208.csv` → Now in `2025_45d/`
- `player_699592_daily_features_20250101_20250208.csv` → Now in `2025_45d/`
- `player_8198_daily_features_20250401_20250511.csv` → Now in `2025_45d/`

**Location: `backtests/timelines/`**
- `player_200512_timelines_20240401_20240531.csv` → Now in `2025_45d/`
- `player_258027_timelines_20250901_20251029.csv` → Now in `2025_45d/`
- `player_452607_timelines_20250101_20250208.csv` → Now in `2025_45d/`
- `player_699592_timelines_20250101_20250208.csv` → Now in `2025_45d/`
- `player_8198_timelines_20250101_20250208.csv` → Now in `2025_45d/`

**Location: `backtests/visualizations/`**
- `player_200512_probabilities_20240401_20240531.png` → Now in `2025_45d/`
- `player_258027_probabilities_20250901_20251029.png` → Now in `2025_45d/`
- `player_452607_probabilities_20250101_20250208.png` → Now in `2025_45d/`
- `player_699592_probabilities_20250101_20250208.png` → Now in `2025_45d/`
- `player_8198_probabilities_20250401_20250511.png` → Now in `2025_45d/`
- `predictions_summary.csv` → Now in `2025_45d/`
- `predictions_summary.md` → Now in `2025_45d/`

**Location: `backtests/predictions/`**
- Duplicate prediction files in root `ensemble/`, `gradient_boosting/`, `random_forest/` directories
- These are duplicates of files in `2025_45d/` subdirectory

### 3. Cache and Temporary Files
**Location: Root**
- `data_cache_v3.pkl` → Cache file, can be regenerated
- `feature_generation.log` → Log file, can be regenerated

### 4. Test/Verification Scripts
**Location: Root**
- `verify_enhancements.py` → Temporary verification script, no longer needed

### 5. Old Dashboard Options (Keep Only Final Version)
**Location: `backtests/visualizations/dashboard_options/`**
- `option_1_horizontal_split.png` → Design option, not final
- `option_2_vertical_split.png` → Design option, not final
- `option_4_card_design.png` → Design option, not final
- Keep: `option_3_grid_layout.png` (final chosen design)

## Files to Keep

### Production Models
- `models/model_v3_gradient_boosting_final_100percent.pkl`
- `models/model_v3_random_forest_final_100percent.pkl`
- `models/model_v3_gb_final_columns.json`
- `models/model_v3_rf_final_columns.json`
- `models/final_model_report_v3.json`
- `models/insights/` (all files - bodypart and severity classifiers)

### Current Backtest Structure
- `backtests/config/players_2025_45d.json`
- `backtests/daily_features/2025_45d/` (all files)
- `backtests/timelines/2025_45d/` (all files)
- `backtests/predictions/2025_45d/` (all files)
- `backtests/visualizations/2025_45d/` (all files)
- `backtests/visualizations/dashboard_options/option_3_grid_layout.png`

### Scripts (All)
- All scripts in `scripts/` directory

### Documentation
- `README.md`
- `backtests/README.md`
- `documentation/SETUP_GUIDE.md`

### Data
- `original_data/` (all Excel files)
- `requirements.txt`
- `results/algorithm_comparison_v3.csv`

## Summary
- **Total files to delete**: ~80-90 files
- **Space saved**: Primarily from duplicate prediction files and old model versions
- **Risk**: Low - all files are either duplicates, intermediate files, or can be regenerated

