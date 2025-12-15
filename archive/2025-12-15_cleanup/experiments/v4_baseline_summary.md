# V4 Timeline Datasets - Baseline Model Performance

**Date:** 2025-12-09 21:40:13

## Dataset Split

- **Training:** 2022-07-01 to 2024-06-30 (86,550 records, 8.0% injury ratio)
- **Validation:** 2024-07-01 to 2025-06-30 (57,575 records, 8.0% injury ratio)
- **Test:** >= 2025-07-01 (109,022 records, 1.6% injury ratio)

## Approach

- **Baseline approach:** No calibration, no feature selection, no correlation filtering
- **Features:** 1520 features (after one-hot encoding)

## Performance Metrics

| Model | Dataset | Precision | Recall | F1-Score | ROC AUC | Gini |
|-------|---------|-----------|--------|----------|---------|------|
| **RF** | Training | 0.5397 | 1.0000 | 0.7010 | 0.9993 | 0.9987 |
| | Validation (2024/25) | 0.2296 | 0.3072 | 0.2628 | 0.7497 | 0.4993 |
| | Test (>= 2025-07-01) | 0.0467 | 0.3764 | 0.0831 | 0.7604 | 0.5209 |
| | | | | | | |
| **GB** | Training | 1.0000 | 0.9941 | 0.9970 | 1.0000 | 1.0000 |
| | Validation (2024/25) | 0.4070 | 0.0152 | 0.0293 | 0.7254 | 0.4508 |
| | Test (>= 2025-07-01) | 0.1341 | 0.0201 | 0.0350 | 0.7371 | 0.4741 |
| | | | | | | |
| **LR** | Training | 0.2188 | 0.8445 | 0.3475 | 0.8680 | 0.7359 |
| | Validation (2024/25) | 0.1452 | 0.6216 | 0.2354 | 0.7206 | 0.4413 |
| | Test (>= 2025-07-01) | 0.0295 | 0.7282 | 0.0566 | 0.7073 | 0.4145 |
| | | | | | | |

## Performance Gaps

### RF
- **F1 Gap (Train → Validation):** 0.4382 (62.5% relative)
- **F1 Gap (Validation → Test):** 0.1797 (68.4% relative)
- **F1 Gap (Train → Test):** 0.6179 (88.1% relative)

### GB
- **F1 Gap (Train → Validation):** 0.9677 (97.1% relative)
- **F1 Gap (Validation → Test):** -0.0057 (-19.4% relative)
- **F1 Gap (Train → Test):** 0.9620 (96.5% relative)

### LR
- **F1 Gap (Train → Validation):** 0.1121 (32.3% relative)
- **F1 Gap (Validation → Test):** 0.1788 (75.9% relative)
- **F1 Gap (Train → Test):** 0.2909 (83.7% relative)
