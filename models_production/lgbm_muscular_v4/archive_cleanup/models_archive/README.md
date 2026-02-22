# Archive - Old Models and Comparison Results

This directory contains archived model files and comparison results from V4 development.

## Structure

### `old_models/`
Contains superseded model files that were replaced by the final V4 580 production model:

- **Natural models** (baseline V4):
  - `lgbm_muscular_v4_natural_model1.joblib` + `_columns.json`
  - `lgbm_muscular_v4_natural_model2.joblib` + `_columns.json`
  - `lgbm_muscular_v4_natural_metrics.json`

- **Enriched models** (Layer 2 enriched features):
  - `lgbm_muscular_v4_enriched_model1.joblib` + `_columns.json`
  - `lgbm_muscular_v4_enriched_model2.joblib` + `_columns.json`
  - `lgbm_muscular_v4_enriched_metrics.json`
  - `enriched_comparison/` directory with additional enriched model variants

- **Comparison directory**:
  - `comparison/` - Additional comparison results

### `comparison_results/`
Contains the fair comparison results between V3 and V4 580 models:

- `v4_580_comparison_metrics.json` - Detailed metrics for all 4 model configurations
- `v4_580_comparison_table.md` - Markdown comparison table

**Note**: These files were originally named with "620" but actually contain results for the **580-feature model** (the discrepancy was due to 40 features from the top 620 not being available in the datasets).

## Production Model

The **final production model** is located in:
- `models_production/lgbm_muscular_v4/model/` (not in archive)

This is the V4 580 (With Test, Excl 2021/22-2022/23) model that outperformed V3.
