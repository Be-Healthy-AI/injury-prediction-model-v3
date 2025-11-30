# Phase 2 Threshold Optimization - Comparison

**Comparison:** Default threshold (0.5) vs Optimized threshold

## Summary Table

| Model | Threshold | Precision | Recall | F1-Score | ROC AUC |
|-------|-----------|-----------|--------|----------|----------|
| RF | Default (0.5) | 0.0905 | 0.0864 | 0.0884 | 0.7503 |
| RF | Optimized (0.400) | 0.0872 | 0.1364 | 0.1064 | 0.7503 |
| RF | **Improvement** | **-0.0033** | **+0.0500** | **+0.0180** | +0.0000 |
| GB | Default (0.5) | 0.1538 | 0.1364 | 0.1446 | 0.7255 |
| GB | Optimized (0.700) | 0.3625 | 0.1318 | 0.1933 | 0.7255 |
| GB | **Improvement** | **+0.2087** | **-0.0045** | **+0.0488** | +0.0000 |
| LR | Default (0.5) | 0.0239 | 0.0864 | 0.0375 | 0.6530 |
| LR | Optimized (0.300) | 0.0310 | 0.3591 | 0.0570 | 0.6530 |
| LR | **Improvement** | **+0.0070** | **+0.2727** | **+0.0195** | +0.0000 |

## Key Findings

### RF

- **Optimal Threshold:** 0.400
- **F1 Improvement:** +0.0180 (+20.4%)
- **Precision:** 0.0905 → 0.0872 (-0.0033)
- **Recall:** 0.0864 → 0.1364 (+0.0500)
- **Injuries Detected:** 19 → 30 (+11)

### GB

- **Optimal Threshold:** 0.700
- **F1 Improvement:** +0.0488 (+33.7%)
- **Precision:** 0.1538 → 0.3625 (+0.2087)
- **Recall:** 0.1364 → 0.1318 (-0.0045)
- **Injuries Detected:** 30 → 29 (-1)

### LR

- **Optimal Threshold:** 0.300
- **F1 Improvement:** +0.0195 (+52.2%)
- **Precision:** 0.0239 → 0.0310 (+0.0070)
- **Recall:** 0.0864 → 0.3591 (+0.2727)
- **Injuries Detected:** 19 → 79 (+60)

## Recommendation

**Use GB model at threshold 0.700**

- **F1-Score:** 0.1933
- **Precision:** 0.3625
- **Recall:** 0.1318
- **Injuries Detected:** 29 out of 220
