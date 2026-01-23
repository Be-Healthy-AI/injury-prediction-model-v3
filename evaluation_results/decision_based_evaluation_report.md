# Decision-Based Evaluation Report - WINNER Model

**Generated:** 2025-12-15 18:09:12

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
| 1 | 3.4% | 19.6% | 28.8 | 5783 |
| 2 | 3.2% | 36.8% | 30.7 | 5783 |
| 3 | 3.0% | 50.9% | 33.3 | 5783 |
| 5 | 2.7% | 68.4% | 41.2 | 5783 |
| 7 | 2.4% | 73.6% | 53.0 | 5783 |
| 10 | 2.2% | 82.5% | 57.4 | 5783 |
| 15 | 2.0% | 88.8% | 67.5 | 5783 |
| 20 | 1.8% | 92.5% | 74.7 | 5783 |

## Key Insights

- **Best Precision@K:** 3.4% at K=1
- **Best Recall@K:** 92.5% at K=20
- **Lowest False Alerts/Injury:** 28.8 at K=1

## Dataset Statistics

- **Total Test Samples:** 153,006
- **Total Injuries:** 990
- **Injury Rate:** 0.6%
- **Unique Teams:** 392
- **Unique Weeks:** 23
- **Unique Team-Weeks:** 5783
