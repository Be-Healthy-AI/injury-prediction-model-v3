# V4 Challenger Model Implementation Plan

## Executive Summary

This plan outlines the implementation of a parallel V4 model deployment system that runs alongside the existing V3 production system. The V4 system will operate in a "challenger" subfolder structure, allowing for independent benchmarking and calibration comparison while keeping the V3 system completely untouched.

## Objectives

1. **Maintain V3 System Integrity**: Keep all existing V3 processes and outputs completely unchanged
2. **Create Parallel V4 System**: Build a fully independent V4 deployment pipeline
3. **Enable Calibration Comparison**: Generate calibration charts for both V3 and V4 for performance comparison
4. **Use Challenger Architecture**: Organize V4 outputs in a dedicated "challenger" subfolder

## Directory Structure

### Current V3 Structure (Unchanged)
```
production/
├── deployments/
│   └── England/
│       ├── Arsenal FC/
│       ├── Chelsea FC/
│       ├── ... (all 20 PL clubs)
│       ├── calibration_chart_v3_*.png
│       └── calibration_data_v3_*.csv
├── raw_data/
│   └── england/
│       └── {YYYYMMDD}/
└── scripts/
    ├── deploy_all_clubs_v3.py
    ├── update_daily_features.py (V3)
    ├── update_timelines.py (V3)
    ├── generate_predictions_lgbm_v3.py
    └── generate_calibration_chart.py
```

### New V4 Challenger Structure
```
production/
├── deployments/
│   └── England/
│       ├── challenger/                    # NEW: V4 challenger root
│       │   ├── Arsenal FC/
│       │   │   ├── config.json
│       │   │   ├── daily_features/        # V4 Layer 1 + Layer 2
│       │   │   ├── timelines/
│       │   │   ├── predictions/
│       │   │   └── dashboards/
│       │   ├── Chelsea FC/
│       │   ├── ... (all 20 PL clubs)
│       │   ├── calibration_chart_v4_*.png
│       │   └── calibration_data_v4_*.csv
│       ├── Arsenal FC/                    # V3 (unchanged)
│       └── ... (all 20 PL clubs)          # V3 (unchanged)
├── raw_data/                              # SHARED with V3
│   └── england/
│       └── {YYYYMMDD}/
└── scripts/
    ├── deploy_all_clubs_v4.py             # NEW: V4 orchestrator
    ├── update_daily_features_v4.py        # NEW: V4 Layer 1
    ├── enrich_daily_features_v4.py        # NEW: V4 Layer 2
    ├── update_timelines_v4.py             # NEW: V4 timelines
    ├── generate_predictions_lgbm_v4.py    # NEW: V4 predictions
    ├── generate_dashboards_v4.py          # NEW: V4 dashboards
    └── generate_calibration_chart_v4.py   # NEW: V4 calibration
```

## Key Differences: V3 vs V4

### Daily Features Generation

**V3 (Single Layer)**:
- Script: `update_daily_features.py`
- Uses: `scripts.create_daily_features_v3` (wrapper to `lgbm_muscular_v1`)
- Output: Single daily features file per player
- Location: `production/deployments/{country}/{club}/daily_features/`

**V4 (Two Layers)**:
- **Layer 1**: `create_daily_features_v4_enhanced.py`
  - Generates base daily features (60-80 features)
  - Log-transformed injury recency, workload, recovery indicators
  - Output: `daily_features/player_{id}_daily_features.csv`
  
- **Layer 2**: `enrich_daily_features_v4_layer2.py`
  - Enriches Layer 1 with advanced features
  - Adds: workload windows, ACWR, season-to-date ratios, injury-history windows
  - Output: `daily_features_enriched/player_{id}_daily_features.csv` (or overwrites Layer 1)

### Timeline Generation

**V3**:
- Script: `update_timelines.py`
- Uses: V3 daily features directly
- Output: `timelines_35day_season_{season}_v3_muscular.csv`

**V4**:
- Script: `create_35day_timelines_v4_enhanced.py`
- Uses: V4 enriched daily features (Layer 2 preferred, Layer 1 fallback)
- Output: `timelines_35day_season_{season}_v4_muscular.csv`
- Features: Dual targets (muscular + skeletal), PL-only filtering

### Model & Predictions

**V3**:
- Model: `production/models/lgbm_muscular_v3/model.joblib`
- Features: ~200 features (from V3 daily features)
- Script: `generate_predictions_lgbm_v3.py`

**V4**:
- Model: `models_production/lgbm_muscular_v4/model/model.joblib`
- Features: 580 features (from V4 enriched daily features)
- Script: `generate_predictions_lgbm_v4.py` (to be created)

## Implementation Phases

### Phase 1: Script Adaptation & Creation

#### 1.1 Adapt V4 Daily Features Scripts for Production

**Task**: Create production versions of V4 daily features scripts that:
- Read from `production/raw_data/england/{YYYYMMDD}/`
- Write to `production/deployments/England/challenger/{club}/daily_features/`
- Support incremental updates (like V3)
- Handle date capping (max-date parameter)

**Files to Create**:
- `production/scripts/update_daily_features_v4.py`
  - Wrapper around `create_daily_features_v4_enhanced.py`
  - Adapts paths, adds incremental logic, date capping
  - Processes one club at a time (like V3 version)

- `production/scripts/enrich_daily_features_v4.py`
  - Wrapper around `enrich_daily_features_v4_layer2.py`
  - Adapts paths to production structure
  - Processes enriched features for one club
  - Handles incremental updates

**Key Adaptations Needed**:
1. **Path Mapping**:
   - Input: `production/raw_data/england/{data_date}/`
   - Output: `production/deployments/England/challenger/{club}/daily_features/`
   - Layer 2 output: Same directory (overwrites or uses separate subfolder)

2. **Incremental Logic**:
   - Check existing daily features files
   - Only generate/update for new dates
   - Support `--max-date` parameter for date capping

3. **Player Filtering**:
   - Use `config.json` to get player list per club
   - Only process players in the club's config

#### 1.2 Adapt V4 Timeline Script for Production

**Task**: Create `production/scripts/update_timelines_v4.py`

**Key Adaptations**:
1. **Path Mapping**:
   - Input: `production/deployments/England/challenger/{club}/daily_features/`
   - Output: `production/deployments/England/challenger/{club}/timelines/`
   - Prefer enriched features, fallback to Layer 1

2. **Incremental Logic**:
   - Check existing timeline files
   - Only regenerate from last date forward
   - Support `--regenerate-from-date` parameter

3. **Club-Specific Processing**:
   - Process one club at a time
   - Use club's config.json for player list
   - Generate single timeline file per club (not per season)

4. **Date Capping**:
   - Support `--max-date` to cap timeline generation
   - Filter out reference_dates beyond max-date

#### 1.3 Create V4 Prediction Script

**Task**: Create `production/scripts/generate_predictions_lgbm_v4.py`

**Based on**: `generate_predictions_lgbm_v3.py`

**Key Differences**:
1. **Model Loading**:
   - Load from: `models_production/lgbm_muscular_v4/model/model.joblib`
   - Load columns from: `models_production/lgbm_muscular_v4/model/columns.json`
   - 580 features expected (vs ~200 for V3)

2. **Feature Alignment**:
   - Use V4 timeline features directly (no V3 preprocessing)
   - Ensure feature order matches `columns.json`
   - Handle missing features (fill with defaults)

3. **Output Paths**:
   - Write to: `production/deployments/England/challenger/{club}/predictions/`
   - Filename: `predictions_lgbm_v4_{YYYYMMDD}.csv`

4. **SHAP Integration**:
   - Use V4 model for SHAP values
   - Adapt feature names for insights

#### 1.4 Create V4 Dashboard Script

**Task**: Create `production/scripts/generate_dashboards_v4.py`

**Based on**: `generate_dashboards.py`

**Key Adaptations**:
1. **Path Mapping**:
   - Read predictions from: `challenger/{club}/predictions/`
   - Write dashboards to: `challenger/{club}/dashboards/`
   - Model version suffix: `v4`

2. **Model-Specific Adjustments**:
   - Use V4 model metadata for risk thresholds (if different)
   - Adapt feature names for insights display

#### 1.5 Create V4 Calibration Chart Script

**Task**: Create `production/scripts/generate_calibration_chart_v4.py`

**Based on**: `generate_calibration_chart.py`

**Key Adaptations**:
1. **Path Mapping**:
   - Read predictions from: `challenger/{club}/predictions/`
   - Output to: `production/deployments/England/challenger/`
   - Filename: `calibration_chart_v4_{YYYYMMDD}.png`

2. **Model Version**:
   - Update labels to "V4" instead of "V3"
   - Use V4-specific metadata

### Phase 2: Orchestrator Creation

#### 2.1 Create V4 Deployment Orchestrator

**Task**: Create `production/scripts/deploy_all_clubs_v4.py`

**Based on**: `deploy_all_clubs_v3.py`

**Pipeline Steps**:
1. **Fetch Raw Data** (shared with V3, skip if already done)
2. **Config Sync** (shared logic, but write to challenger/{club}/)
3. **Update Daily Features - Layer 1** (V4-specific)
4. **Enrich Daily Features - Layer 2** (V4-specific)
5. **Update Timelines** (V4-specific)
6. **Generate Predictions** (V4-specific)
7. **Generate Dashboards** (V4-specific)
8. **Generate Predictions Table** (optional, V4-specific)

**Key Features**:
- Process all clubs sequentially (or in parallel if needed)
- Support `--clubs` parameter for selective processing
- Support `--data-date` for date capping
- Support `--skip-*` flags for individual steps
- Write all outputs to `challenger/` subfolder

### Phase 3: Configuration & Setup

#### 3.1 Create Challenger Directory Structure

**Task**: Initialize challenger folder structure

**Steps**:
1. Create `production/deployments/England/challenger/`
2. Copy all club configs from V3 to challenger
3. Create subdirectories for each club:
   - `daily_features/`
   - `timelines/`
   - `predictions/`
   - `dashboards/`

**Script**: `production/scripts/initialize_challenger_structure.py`

#### 3.2 Shared Raw Data

**Decision**: V3 and V4 will share the same raw data source
- Location: `production/raw_data/england/{YYYYMMDD}/`
- Both systems read from the same files
- No duplication needed

### Phase 4: Testing & Validation

#### 4.1 Single Club Test

**Task**: Test V4 pipeline on one club (e.g., Arsenal FC)

**Steps**:
1. Run `deploy_all_clubs_v4.py --clubs "Arsenal FC" --data-date 20260122`
2. Verify:
   - Daily features (Layer 1) generated correctly
   - Daily features (Layer 2) enriched correctly
   - Timelines generated correctly
   - Predictions generated correctly
   - Dashboards generated correctly
3. Compare outputs with V3 (structure, not values)

#### 4.2 Full Deployment Test

**Task**: Run V4 pipeline for all clubs

**Steps**:
1. Run `deploy_all_clubs_v4.py --data-date 20260122`
2. Monitor for errors
3. Verify all clubs processed successfully

#### 4.3 Calibration Comparison

**Task**: Generate and compare calibration charts

**Steps**:
1. Run `generate_calibration_chart.py` (V3) for date range
2. Run `generate_calibration_chart_v4.py` (V4) for same date range
3. Compare:
   - Injury rates by bin
   - Calibration quality
   - Model performance metrics

### Phase 5: Documentation & Maintenance

#### 5.1 Documentation

**Files to Create**:
- `production/scripts/README_V4_CHALLENGER.md`
  - Overview of V4 challenger system
  - How to run V4 pipeline
  - How to compare V3 vs V4
  - Troubleshooting guide

#### 5.2 Maintenance Scripts

**Optional Enhancements**:
- Script to compare V3 vs V4 predictions side-by-side
- Script to generate comparison reports
- Script to sync configs between V3 and V4

## Technical Considerations

### 1. Path Management

**Challenge**: V4 scripts currently use different path structures

**Solution**: Create path mapping utilities:
```python
def get_challenger_path(country: str, club: str) -> Path:
    """Get base path for challenger club."""
    return PRODUCTION_ROOT / "deployments" / country / "challenger" / club

def get_raw_data_path(country: str, data_date: str) -> Path:
    """Get raw data path (shared with V3)."""
    return PRODUCTION_ROOT / "raw_data" / country.lower().replace(" ", "_") / data_date
```

### 2. Feature Compatibility

**Challenge**: V4 uses 580 features vs V3's ~200 features

**Solution**: 
- V4 scripts must use V4 model and feature set exclusively
- No cross-contamination between V3 and V4 feature sets
- Separate preprocessing pipelines

### 3. Incremental Updates

**Challenge**: V4 scripts need incremental update logic (like V3)

**Solution**:
- Check existing files before generating
- Support date ranges for incremental updates
- Handle new players (transfers) automatically

### 4. Date Capping

**Challenge**: Both systems need date capping to prevent future dates

**Solution**:
- Pass `--max-date` or `--data-date` to all scripts
- Filter out dates beyond the data date
- Consistent date handling across all steps

### 5. Transfer Management

**Challenge**: V4 needs to handle transfers like V3

**Solution**:
- Reuse `validate_and_sync_club_config.py` logic
- Write configs to challenger/{club}/config.json
- Sync with Transfermarkt when needed

## File Dependencies

### V4 Model Files (Source)
- `models_production/lgbm_muscular_v4/model/model.joblib`
- `models_production/lgbm_muscular_v4/model/columns.json`
- `models_production/lgbm_muscular_v4/model/MODEL_METADATA.json`

### V4 Code Files (Source)
- `models_production/lgbm_muscular_v4/code/daily_features/create_daily_features_v4_enhanced.py`
- `models_production/lgbm_muscular_v4/code/daily_features/enrich_daily_features_v4_layer2.py`
- `models_production/lgbm_muscular_v4/code/timelines/create_35day_timelines_v4_enhanced.py`

### Production Scripts (To Create)
- `production/scripts/update_daily_features_v4.py`
- `production/scripts/enrich_daily_features_v4.py`
- `production/scripts/update_timelines_v4.py`
- `production/scripts/generate_predictions_lgbm_v4.py`
- `production/scripts/generate_dashboards_v4.py`
- `production/scripts/generate_calibration_chart_v4.py`
- `production/scripts/deploy_all_clubs_v4.py`
- `production/scripts/initialize_challenger_structure.py`

## Execution Order

### Initial Setup (One-time)
1. Run `initialize_challenger_structure.py` to create directory structure
2. Copy club configs from V3 to challenger

### Daily/Regular Execution
1. **Shared Step**: Fetch raw data (if not already done)
2. **V3 Pipeline**: Run `deploy_all_clubs_v3.py` (unchanged)
3. **V4 Pipeline**: Run `deploy_all_clubs_v4.py` (new)
4. **Comparison**: Generate calibration charts for both

## Success Criteria

1. ✅ V3 system remains completely unchanged
2. ✅ V4 system runs independently in challenger folder
3. ✅ Both systems use same raw data source
4. ✅ V4 generates all outputs (daily features, timelines, predictions, dashboards)
5. ✅ Calibration charts can be generated for both V3 and V4
6. ✅ Direct comparison between V3 and V4 is possible

## Risk Mitigation

### Risk 1: Path Conflicts
**Mitigation**: Use explicit "challenger" subfolder, never write to V3 paths

### Risk 2: Feature Mismatch
**Mitigation**: Separate preprocessing pipelines, no shared feature code

### Risk 3: Performance Impact
**Mitigation**: V4 can run after V3, or in parallel if resources allow

### Risk 4: Data Inconsistency
**Mitigation**: Both systems use same raw data source, same data date

## Timeline Estimate

- **Phase 1** (Script Adaptation): 3-5 days
- **Phase 2** (Orchestrator): 1-2 days
- **Phase 3** (Configuration): 0.5 days
- **Phase 4** (Testing): 2-3 days
- **Phase 5** (Documentation): 1 day

**Total**: ~7-11 days

## Next Steps

1. Review and approve this plan
2. Start with Phase 1.1 (Adapt V4 daily features scripts)
3. Test incrementally after each phase
4. Document as we go
