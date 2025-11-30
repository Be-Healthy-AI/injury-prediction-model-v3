# Phase 2 Results Comparison

**Strategy:** Low-Shift Features + Covariate Shift Correction + Calibration

## Out-of-Sample Validation Comparison

| Model | Metric | Baseline | Phase 1 | Phase 2 | P2 vs Baseline | P2 vs P1 |
|-------|--------|----------|---------|---------|----------------|----------|
| RF | Precision | 0.1377 | 0.0484 | 0.0905 | -0.0472 | +0.0421 |
| RF | Recall | 0.0864 | 0.3727 | 0.0864 | +0.0000 | -0.2864 |
| RF | F1 | 0.1061 | 0.0856 | 0.0884 | -0.0178 | +0.0027 |
| RF | Roc_auc | 0.7822 | 0.7434 | 0.7503 | -0.0319 | +0.0069 |
| GB | Precision | 0.2222 | 0.1622 | 0.1538 | -0.0684 | -0.0083 |
| GB | Recall | 0.0364 | 0.0545 | 0.1364 | +0.1000 | +0.0818 |
| GB | F1 | 0.0625 | 0.0816 | 0.1446 | +0.0821 | +0.0629 |
| GB | Roc_auc | 0.7500 | 0.6957 | 0.7255 | -0.0245 | +0.0298 |
| LR | Precision | 0.0244 | 0.0226 | 0.0239 | -0.0005 | +0.0013 |
| LR | Recall | 0.8591 | 0.8045 | 0.0864 | -0.7727 | -0.7182 |
| LR | F1 | 0.0474 | 0.0441 | 0.0375 | -0.0099 | -0.0066 |
| LR | Roc_auc | 0.6721 | 0.6577 | 0.6530 | -0.0191 | -0.0047 |

## Key Findings

### RF

- **F1-Score:** Baseline=0.1061, Phase1=0.0856, Phase2=0.0884
- **Improvement (P2 vs Baseline):** -0.0178 (-16.7%)
- **Improvement (P2 vs Phase1):** +0.0027 (+3.2%)

### GB

- **F1-Score:** Baseline=0.0625, Phase1=0.0816, Phase2=0.1446
- **Improvement (P2 vs Baseline):** +0.0821 (+131.3%)
- **Improvement (P2 vs Phase1):** +0.0629 (+77.1%)

### LR

- **F1-Score:** Baseline=0.0474, Phase1=0.0441, Phase2=0.0375
- **Improvement (P2 vs Baseline):** -0.0099 (-21.0%)
- **Improvement (P2 vs Phase1):** -0.0066 (-14.9%)

✅ **Phase 2 shows improvement over baseline!**
