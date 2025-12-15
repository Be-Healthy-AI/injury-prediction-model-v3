# Decision-Based Evaluation Report - WINNER Model

**Generated:** 2025-12-15 10:31:33

## Model Information

- **Model:** LightGBM (WINNER)
- **Configuration:** 10% target ratio, baseline hyperparameters, 0.8 correlation threshold
- **Test Dataset:** 2025-2026 season (natural target ratio)
- **Prediction Horizon:** 35 days

## Evaluation Approach

This evaluation reframes injury prediction as a decision-support system:
- **Decision Unit:** Team × Week
- **Selection Method:** Top-K players by risk score (ranking-based)
- **Primary Metric:** Precision@K (accuracy of alerts)
- **Secondary Metric:** Recall@K (injury capture rate)
- **Operational Metric:** False Alerts per True Injury

## Summary Results

| K | Precision@K | Recall@K | False Alerts/Injury | Team-Weeks |
|---|-------------|----------|---------------------|------------|
| 1 | 2.1% | 12.3% | 46.4 | 5783 |
| 2 | 2.0% | 23.3% | 49.0 | 5783 |
| 3 | 2.0% | 31.7% | 54.0 | 5783 |
| 5 | 1.8% | 42.9% | 66.2 | 5783 |
| 7 | 1.7% | 46.6% | 84.4 | 5783 |
| 10 | 1.7% | 52.7% | 90.4 | 5783 |
| 15 | 1.6% | 60.2% | 100.1 | 5783 |
| 20 | 1.6% | 68.3% | 101.5 | 5783 |

## Key Insights

- **Best Precision@K:** 2.1% at K=1
- **Best Recall@K:** 68.3% at K=20
- **Lowest False Alerts/Injury:** 46.4 at K=1

## Dataset Statistics

- **Total Test Samples:** 153,006
- **Total Injuries:** 990
- **Injury Rate:** 0.6%
- **Unique Teams:** 392
- **Unique Weeks:** 23
- **Unique Team-Weeks:** 5783
