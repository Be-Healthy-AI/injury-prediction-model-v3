# Seasonal Temporal Split - Model Performance

**Date:** 2025-11-29 22:43:38

## Dataset Split

- **Training:** <= 2024-06-30 (36,753 records, 14.3% injury ratio)
- **Validation:** 2024/25 season (5,047 records, 19.8% injury ratio)
- **Test:** 2025/26 season (12,218 records, 1.8% injury ratio)

## Performance Metrics

| Model | Dataset | Precision | Recall | F1-Score | ROC AUC | Gini |
|-------|---------|-----------|--------|----------|---------|------|
| **RF** | Training | 0.8350 | 0.7776 | 0.8053 | 0.9780 | 0.9559 |
| | Validation (2024/25) | 0.4274 | 0.1030 | 0.1660 | 0.6660 | 0.3321 |
| | Test (2025/26) | 0.0385 | 0.0591 | 0.0466 | 0.7500 | 0.5000 |
| | | | | | | |
| **GB** | Training | 0.9852 | 0.9725 | 0.9788 | 0.9993 | 0.9987 |
| | Validation (2024/25) | 0.5053 | 0.0950 | 0.1599 | 0.6691 | 0.3383 |
| | Test (2025/26) | 0.1304 | 0.1364 | 0.1333 | 0.7627 | 0.5253 |
| | | | | | | |
| **LR** | Training | 0.4615 | 0.0125 | 0.0244 | 0.7667 | 0.5335 |
| | Validation (2024/25) | 0.5846 | 0.0380 | 0.0714 | 0.6654 | 0.3307 |
| | Test (2025/26) | 0.0000 | 0.0000 | 0.0000 | 0.6809 | 0.3617 |
| | | | | | | |

## Performance Gaps

### RF
- **F1 Gap (Train → Test):** 0.7587 (94.2% relative)
- **F1 Gap (Train → Validation):** 0.6393 (79.4% relative)
- **F1 Gap (Validation → Test):** 0.1194 (71.9% relative)

### GB
- **F1 Gap (Train → Test):** 0.8455 (86.4% relative)
- **F1 Gap (Train → Validation):** 0.8189 (83.7% relative)
- **F1 Gap (Validation → Test):** 0.0266 (16.6% relative)

### LR
- **F1 Gap (Train → Test):** 0.0244 (100.0% relative)
- **F1 Gap (Train → Validation):** -0.0470 (-192.6% relative)
- **F1 Gap (Validation → Test):** 0.0714 (100.0% relative)
