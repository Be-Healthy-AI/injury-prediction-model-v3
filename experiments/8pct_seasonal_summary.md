# 8% Target Ratio - Seasonal Split - Model Performance

**Date:** 2025-11-30 14:54:05

## Dataset Split

- **Training:** <= 2024-06-30 (65,875 records, 8.0% injury ratio)
- **Validation:** 2024/25 season (12,500 records, 8.0% injury ratio)
- **Test:** 2025/26 season (2,750 records, 8.0% injury ratio)

## Performance Metrics

| Model | Dataset | Precision | Recall | F1-Score | ROC AUC | Gini |
|-------|---------|-----------|--------|----------|---------|------|
| **RF** | Training | 0.8768 | 0.8323 | 0.8540 | 0.9931 | 0.9861 |
| | Validation (2024/25) | 0.2941 | 0.0300 | 0.0544 | 0.6626 | 0.3252 |
| | Test (2025/26) | 0.0000 | 0.0000 | 0.0000 | 0.7692 | 0.5385 |
| | | | | | | |
| **GB** | Training | 0.9941 | 0.9871 | 0.9906 | 0.9999 | 0.9997 |
| | Validation (2024/25) | 0.3411 | 0.0880 | 0.1399 | 0.6664 | 0.3327 |
| | Test (2025/26) | 0.4156 | 0.1455 | 0.2155 | 0.7347 | 0.4694 |
| | | | | | | |
| **LR** | Training | 0.0000 | 0.0000 | 0.0000 | 0.7695 | 0.5389 |
| | Validation (2024/25) | 0.0000 | 0.0000 | 0.0000 | 0.6697 | 0.3394 |
| | Test (2025/26) | 0.0000 | 0.0000 | 0.0000 | 0.6832 | 0.3664 |
| | | | | | | |

## Performance Gaps

### RF
- **F1 Gap (Train → Validation):** 0.7995 (93.6% relative)
- **F1 Gap (Validation → Test):** 0.0544 (100.0% relative)
- **F1 Gap (Train → Test):** 0.8540 (100.0% relative)

### GB
- **F1 Gap (Train → Validation):** 0.8507 (85.9% relative)
- **F1 Gap (Validation → Test):** -0.0756 (-54.0% relative)
- **F1 Gap (Train → Test):** 0.7751 (78.2% relative)

### LR
- **F1 Gap (Train → Validation):** N/A
- **F1 Gap (Validation → Test):** N/A
- **F1 Gap (Train → Test):** N/A
