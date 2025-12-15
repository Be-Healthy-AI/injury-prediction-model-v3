# Temporal Split Test - 2024-06-30

## Dataset Split
- **Training:** reference_date <= 2024-06-30
- **Validation:** reference_date > 2024-06-30 and <= 2025-06-30

## Dataset Characteristics
- **Training:** 55,087 records, 9.6% injury ratio
- **Validation:** 7,613 records, 13.1% injury ratio

## Performance Metrics

| Model | Set | Precision | Recall | F1-Score | ROC AUC |
|-------|-----|-----------|--------|----------|---------|
| **RF** | Training | 0.9596 | 1.0000 | 0.9794 | 1.0000 |
| | Validation | 0.3860 | 0.0220 | 0.0416 | 0.6846 |
| | **Gap** | **0.5736** | **0.9780** | **0.9377** | **0.3153** |
| | | | | | |
| **GB** | Training | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| | Validation | 0.6923 | 0.0270 | 0.0520 | 0.6571 |
| | **Gap** | **0.3077** | **0.9730** | **0.9480** | **0.3429** |
| | | | | | |
| **LR** | Training | 0.2531 | 0.8461 | 0.3897 | 0.8666 |
| | Validation | 0.1887 | 0.5700 | 0.2836 | 0.6421 |
| | **Gap** | **0.0644** | **0.2761** | **0.1061** | **0.2245** |

## Key Observations

1. **Tree-based models (RF, GB)** show perfect training performance but severe degradation on validation:
   - RF: Recall drops from 100% to 2.2%
   - GB: Recall drops from 100% to 2.7%
   - Both show large ROC AUC gaps (~0.32-0.34)

2. **Logistic Regression** shows more stable performance:
   - Smaller gaps across all metrics
   - Better validation recall (57%) compared to tree models
   - More realistic training performance (not perfect)

3. **The performance gap persists** even with this different temporal split, suggesting:
   - Temporal drift/distribution shift between periods
   - Overfitting in tree-based models
   - Need for more regularization or different approaches



