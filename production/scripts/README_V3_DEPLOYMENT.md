# lgbm_muscular_v3 Deployment Guide

## Overview

The `lgbm_muscular_v3` model has been successfully deployed in parallel with the existing `lgbm_muscular_v2` model. This parallel deployment ensures that:

- **V2 remains fully functional** - All existing V2 scripts and processes continue to work unchanged
- **V3 is available for use** - New V3 scripts are available for testing and production use
- **No conflicts** - V2 and V3 use separate cache files and output files

## What Was Deployed

### 1. Model Files
- **Location**: `production/models/lgbm_muscular_v3/`
- **Files**:
  - `model.joblib` - The trained V3 model
  - `columns.json` - Feature columns expected by the model (1,623 features)
  - `MODEL_METADATA.json` - Model metadata and training information

### 2. Prediction Script
- **File**: `production/scripts/generate_predictions_lgbm_v3.py`
- **Purpose**: Generate daily injury predictions using the V3 model
- **Output**: `predictions_lgbm_v3_YYYYMMDD.csv` in the club's predictions folder
- **Per-player files**: `player_{id}_predictions_v3_YYYYMMDD.csv`

### 3. Deployment Orchestrator
- **File**: `production/scripts/deploy_all_clubs_v3.py`
- **Purpose**: Run the complete pipeline for all clubs using V3 model
- **Features**: Same as V2 orchestrator, but uses V3 prediction script

## Model Comparison

| Aspect | V2 | V3 |
|--------|----|----|
| **Features** | 4,059 | 1,623 |
| **Training Data** | All seasons (2008-2026) | PL-only, filtered seasons (2018-2026) |
| **Target Ratio** | 10% (balanced) | Natural (unbalanced) |
| **Excluded Seasons** | None | 2021_2022, 2022_2023 (low injury rates) |
| **Test Performance** | ROC AUC: 0.9688 | ROC AUC: 1.0 (test), 0.9999 (train) |
| **Preprocessing** | Same | Same (compatible) |

## Usage

### Single Club (Testing)

#### Generate V3 Predictions for One Club
```bash
python production/scripts/generate_predictions_lgbm_v3.py --country England --club "Chelsea FC" --force
```

#### Run Full Pipeline for One Club (V3)
```bash
python production/scripts/deploy_all_clubs_v3.py --clubs "Chelsea FC" --data-date 20260104 --stop-on-error
```

### Multiple Clubs

#### Run Full Pipeline for All Clubs (V3)
```bash
python production/scripts/deploy_all_clubs_v3.py --data-date 20260104
```

#### Run for Specific Clubs (V3)
```bash
python production/scripts/deploy_all_clubs_v3.py --clubs "Arsenal FC,Liverpool FC" --data-date 20260104
```

## Key Differences from V2

### Output Files
- **V2**: `predictions_lgbm_v2_YYYYMMDD.csv`
- **V3**: `predictions_lgbm_v3_YYYYMMDD.csv`
- **V2 per-player**: `player_{id}_predictions_YYYYMMDD.csv`
- **V3 per-player**: `player_{id}_predictions_v3_YYYYMMDD.csv`

### Cache Files
- **V2**: `production/cache/preprocessed_{timelines_file_stem}.csv`
- **V3**: `production/cache/preprocessed_v3_{timelines_file_stem}.csv`

This ensures V2 and V3 caches don't interfere with each other.

## Testing Results

✅ **Tested with Chelsea FC (2026-01-04)**
- Predictions generated successfully: 5,263 predictions
- Risk score range: 0.0000 - 0.7831
- Mean risk score: 0.0530
- Per-player files generated: 28 files
- No errors encountered

## Preprocessing Compatibility

✅ **Verified Compatible**
- V3 uses the same preprocessing pipeline as V2 (`preprocessing_lgbm_v2.py`)
- V3 has fewer features (1,623 vs 4,059), but shares 1,547 common features
- The `align_features_to_model` function handles missing features by setting them to 0
- Extra features are automatically dropped

## Backward Compatibility

✅ **V2 Remains Fully Functional**
- All V2 scripts unchanged: `generate_predictions_lgbm_v2.py`, `deploy_all_clubs.py`
- V2 and V3 can run independently
- No conflicts between V2 and V3 outputs
- Both models can be used simultaneously for comparison

## Next Steps

1. **Test with More Clubs**: Run V3 predictions for additional clubs to verify consistency
2. **Compare Results**: Compare V2 and V3 predictions side-by-side for the same dates
3. **Monitor Performance**: Track V3 predictions over time to assess model performance
4. **Gradual Migration**: Once V3 is validated, consider migrating from V2 to V3

## Files Created/Modified

### New Files
- `production/models/lgbm_muscular_v3/model.joblib`
- `production/models/lgbm_muscular_v3/columns.json`
- `production/models/lgbm_muscular_v3/MODEL_METADATA.json`
- `production/scripts/generate_predictions_lgbm_v3.py`
- `production/scripts/deploy_all_clubs_v3.py`
- `production/scripts/README_V3_DEPLOYMENT.md` (this file)

### Unchanged Files (V2 Still Works)
- `production/scripts/generate_predictions_lgbm_v2.py` ✅
- `production/scripts/deploy_all_clubs.py` ✅
- `production/scripts/preprocessing_lgbm_v2.py` ✅ (shared by both V2 and V3)

## Support

For issues or questions:
1. Check that model files exist in `production/models/lgbm_muscular_v3/`
2. Verify timelines are up to date
3. Check cache files if preprocessing seems slow
4. Compare with V2 outputs if results seem unexpected

