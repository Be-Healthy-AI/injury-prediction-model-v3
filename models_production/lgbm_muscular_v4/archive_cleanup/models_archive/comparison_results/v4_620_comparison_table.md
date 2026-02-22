# V4 580-Feature Model - Fair Comparison with V3

## Model Configurations

| Model | Training Seasons | Test Dataset | Excluded Seasons |
|-------|------------------|--------------|------------------|
| **V3 Production** | 2018/19, 2019/20, 2020/21, 2024/25, 2025/26 | 2025/26 (in-sample) | 2021/22, 2022/23, 2023/24 |
| **V4 580 (Current)** | 2018/19-2024/25 | 2025/26 (out-of-sample) | 2025/26 only |
| **V4 580 (With Test)** | 2018/19-2025/26 | 2025/26 (in-sample) | None |
| **V4 580 (With Test, Excl 2021/22-2022/23)** | 2018/19, 2019/20, 2020/21, 2023/24, 2024/25, 2025/26 | 2025/26 (in-sample) | 2021/22, 2022/23 |

## Test Dataset Performance Comparison (Model 1 - Muscular Injuries)

| Metric | V3 Production | V4 580 (Current) | V4 580 (With Test) | V4 580 (With Test, Excl 2021/22-2022/23) |
|--------|---------------|------------------|-------------------|------------------------------------------|
| **Accuracy** | 0.9917 | 0.9718 | 0.9841 | 0.9932 |
| **Precision** | 0.4809 | 0.0476 | 0.3555 | 0.5618 |
| **Recall** | 1.0000 | 0.1167 | 1.0000 | 1.0000 |
| **F1-Score** | 0.6495 | 0.0676 | 0.5245 | 0.7194 |
| **ROC AUC** | 0.9998 | 0.7565 | 1.0000 | 1.0000 |
| **Gini** | 0.9996 | 0.5130 | 1.0000 | 1.0000 |

### Confusion Matrix

| Metric | V3 Production | V4 580 (Current) | V4 580 (With Test) | V4 580 (With Test, Excl 2021/22-2022/23) |
|--------|---------------|------------------|-------------------|------------------------------------------|
| **True Positives (TP)** | 0 | 35 | 300 | 300 |
| **False Positives (FP)** | 0 | 701 | 544 | 234 |
| **True Negatives (TN)** | 0 | 33226 | 33383 | 33693 |
| **False Negatives (FN)** | 0 | 265 | 0 | 0 |
