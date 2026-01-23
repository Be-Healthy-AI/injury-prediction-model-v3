# V3 Timeline Filtering Summary

## Execution Date
Script executed successfully.

## Results

### PL Club Identification
- **Seasons covered**: 24 seasons (2002-2025)
- **Match files processed**: 15,122 files
- **PL clubs identified**: Varies by season (40-217 unique club names per season)

### Player PL Membership
- **Players with PL periods**: 933 players
- **Total PL periods**: 1,815 periods
- **Average periods per player**: ~1.9 periods

### Timeline Filtering Results

#### Train Timelines
- **Original total**: 4,013,727 rows
- **Filtered total**: 844,470 rows
- **Retention rate**: 21.0%
- **Removed**: 79.0%

#### Test Timeline (2025-2026)
- **Original**: 153,006 rows
- **Filtered**: 13,386 rows
- **Retention rate**: 8.7%
- **Removed**: 91.3%

### Observations

1. **Retention varies by season**: 
   - Early seasons (2000-2003): 0% retention (no PL data or very few PL players)
   - Mid seasons (2004-2010): ~10-15% retention
   - Recent seasons (2015-2025): ~20-30% retention
   - Test season (2025-2026): 8.7% retention (may be incomplete data)

2. **Test set has lower retention**: This could be because:
   - The 2025-2026 season data may be incomplete
   - Many players in the dataset may not have been in PL during this period
   - The natural timeline includes players from all clubs, not just PL

3. **Overall dataset reduction**: ~79% of timelines removed, leaving a focused PL-only dataset

## Next Steps

1. ✅ Filter timelines (COMPLETE)
2. ⏳ Create training script (copy from V2, update paths)
3. ⏳ Train V3 model
4. ⏳ Evaluate V3 model (compare with V1/V2)
5. ⏳ Create V3 bundle (model, metadata, code snapshots)

## Files Generated

All filtered timelines are stored in:
- `models_production/lgbm_muscular_v3/data/timelines/train/` (102 files)
- `models_production/lgbm_muscular_v3/data/timelines/test/` (1 file)

File names match V1 exactly (no `_pl_only` suffix) but are stored in separate V3 location.




