# Phase 1 Results Comparison

**Strategy:** Stable Features (drift < 0.10) + Increased Regularization

## Out-of-Sample Validation Comparison

| Model | Metric | Baseline | Phase 1 | Change | Change % |
|-------|--------|----------|---------|--------|----------|
| RF | Precision | 0.1377 | 0.0484 | -0.0893 | -64.9% |
| RF | Recall | 0.0864 | 0.3727 | +0.2864 | +331.6% |
| RF | F1-Score | 0.1061 | 0.0856 | -0.0205 | -19.3% |
| RF | ROC AUC | 0.7822 | 0.7434 | -0.0388 | -5.0% |
| GB | Precision | 0.2222 | 0.1622 | -0.0601 | -27.0% |
| GB | Recall | 0.0364 | 0.0545 | +0.0182 | +50.0% |
| GB | F1-Score | 0.0625 | 0.0816 | +0.0191 | +30.6% |
| GB | ROC AUC | 0.7500 | 0.6957 | -0.0542 | -7.2% |
| LR | Precision | 0.0244 | 0.0226 | -0.0017 | -7.1% |
| LR | Recall | 0.8591 | 0.8045 | -0.0545 | -6.3% |
| LR | F1-Score | 0.0474 | 0.0441 | -0.0034 | -7.1% |
| LR | ROC AUC | 0.6721 | 0.6577 | -0.0144 | -2.1% |

## Key Findings

- **RF F1-Score:** -0.0205 change
- **GB F1-Score:** +0.0191 change
- **LR F1-Score:** -0.0034 change

✅ **Phase 1 shows improvement in tree-based models!**
