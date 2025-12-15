# Split Analysis: 80/20 Random vs Temporal

## Key Findings

### Dataset Characteristics

| Split Type | Training Size | Training Ratio | Validation Size | Validation Ratio |
|------------|---------------|----------------|-----------------|------------------|
| 80/20 Random | 50,160 | 10.0% | 12,540 | 10.0% |
| Temporal | 55,087 | 9.6% | 7,613 | 13.1% |

### Player Overlap

- **80/20 Random:** 225 overlapping players (100.0% of validation)
- **Temporal:** 218 overlapping players (99.5% of validation)

### Feature Drift Summary

**80/20 Random Split:**
- Low drift: 92 features
- Medium drift: 8 features
- High drift: 0 features
- Mean KS statistic: 0.0058

**Temporal Split:**
- Low drift: 35 features
- Medium drift: 44 features
- High drift: 21 features
- Mean KS statistic: 0.0656

## Explanation

The dramatic F1-score drop in temporal split vs 80/20 random split is likely due to:

1. **Temporal drift:** Features have different distributions in different time periods
2. **Player composition:** Different players in training vs validation periods
3. **Injury patterns:** Injury characteristics may change over time
4. **Data leakage:** Random split allows models to see patterns from all time periods during training
