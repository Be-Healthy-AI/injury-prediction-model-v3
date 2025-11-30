# Threshold Sweep Evaluation - Summary Table

**Date:** 2025-11-27  
**Experiments Evaluated:** 24-month and No-limit (same models)  
**Note:** Models for 36, 48, and 60-month experiments were not saved separately and would need retraining to evaluate.

---

## Configuration

- **Thresholds evaluated:** 0.1, 0.2, 0.3, 0.4, 0.5
- **Validation dataset:** 12,218 records
- **Validation injury ratio:** 1.8% (220 injuries, 11,998 non-injuries)

---

## Performance Metrics by Threshold

### Random Forest (RF)

| Threshold | Precision | Recall | F1-Score | ROC AUC | TP | FP | TN | FN |
|-----------|-----------|--------|----------|---------|----|----|----|----|
| **0.1** | 0.0230 | **1.0000** | 0.0449 | 0.7550 | 220 | 9,356 | 2,642 | 0 |
| **0.2** | 0.0322 | 0.7364 | 0.0617 | 0.7550 | 162 | 4,873 | 7,125 | 58 |
| **0.3** | 0.0526 | 0.4136 | 0.0933 | 0.7550 | 91 | 1,639 | 10,359 | 129 |
| **0.4** | 0.0984 | 0.1409 | 0.1159 | 0.7550 | 31 | 284 | 11,714 | 189 |
| **0.5** | **0.3214** | 0.0409 | 0.0726 | 0.7550 | 9 | 19 | 11,979 | 211 |

**Key Observations:**
- At threshold 0.1: Perfect recall (100%) but very low precision (2.3%)
- At threshold 0.3: Balanced trade-off (41% recall, 5.3% precision)
- At threshold 0.5: High precision (32%) but very low recall (4%)

---

### Gradient Boosting (GB)

| Threshold | Precision | Recall | F1-Score | ROC AUC | TP | FP | TN | FN |
|-----------|-----------|--------|----------|---------|----|----|----|----|
| **0.1** | **0.1037** | 0.0636 | 0.0789 | 0.7539 | 14 | 121 | 11,877 | 206 |
| **0.2** | 0.0784 | 0.0182 | 0.0295 | 0.7539 | 4 | 47 | 11,951 | 216 |
| **0.3** | 0.0000 | 0.0000 | 0.0000 | 0.7539 | 0 | 30 | 11,968 | 220 |
| **0.4** | 0.0000 | 0.0000 | 0.0000 | 0.7539 | 0 | 19 | 11,979 | 220 |
| **0.5** | 0.0000 | 0.0000 | 0.0000 | 0.7539 | 0 | 9 | 11,989 | 220 |

**Key Observations:**
- GB is extremely conservative - only threshold 0.1 produces any positive predictions
- Best performance at 0.1: 10.4% precision, 6.4% recall
- At thresholds ≥ 0.2: Zero true positives (model too conservative)

---

### Logistic Regression (LR)

| Threshold | Precision | Recall | F1-Score | ROC AUC | TP | FP | TN | FN |
|-----------|-----------|--------|----------|---------|----|----|----|----|
| **0.1** | 0.0181 | **1.0000** | 0.0355 | 0.6713 | 220 | 11,942 | 56 | 0 |
| **0.2** | 0.0185 | **1.0000** | 0.0364 | 0.6713 | 220 | 11,658 | 340 | 0 |
| **0.3** | 0.0200 | **1.0000** | 0.0393 | 0.6713 | 220 | 10,766 | 1,232 | 0 |
| **0.4** | 0.0225 | 0.9591 | 0.0440 | 0.6713 | 211 | 9,158 | 2,840 | 9 |
| **0.5** | **0.0250** | 0.8955 | 0.0486 | 0.6713 | 197 | 7,683 | 4,315 | 23 |

**Key Observations:**
- LR is very aggressive - maintains 100% recall until threshold 0.4
- Very low precision across all thresholds (1.8-2.5%)
- High false positive rate (7,000-12,000 false positives)
- Best F1-score at threshold 0.5: 4.9%

---

## Best Operating Points by Model

### Random Forest
- **Best F1-Score:** 0.1159 at threshold 0.4 (Precision: 9.8%, Recall: 14.1%)
- **Best Precision:** 32.1% at threshold 0.5 (Recall: 4.1%)
- **Best Recall:** 100% at threshold 0.1 (Precision: 2.3%)
- **Balanced Trade-off:** Threshold 0.3 (Precision: 5.3%, Recall: 41.4%, F1: 9.3%)

### Gradient Boosting
- **Only viable threshold:** 0.1 (Precision: 10.4%, Recall: 6.4%, F1: 7.9%)
- Model is too conservative for thresholds ≥ 0.2

### Logistic Regression
- **Best F1-Score:** 0.0486 at threshold 0.5 (Precision: 2.5%, Recall: 89.5%)
- **Best Precision:** 2.5% at threshold 0.5
- **Best Recall:** 100% at thresholds 0.1-0.3 (Precision: 1.8-2.0%)

---

## Recommendations

1. **Random Forest** shows the best overall performance:
   - Use threshold 0.3-0.4 for balanced precision/recall
   - Threshold 0.3: 41% recall, 5.3% precision
   - Threshold 0.4: 14% recall, 9.8% precision

2. **Gradient Boosting** is too conservative:
   - Only threshold 0.1 produces predictions
   - Consider retraining with different hyperparameters or class weights

3. **Logistic Regression** has high recall but very low precision:
   - Useful if catching all injuries is critical
   - Not suitable if false positive rate needs to be controlled

4. **For production use:**
   - RF at threshold 0.3 appears to be the best balance
   - Captures 41% of injuries with 5.3% precision
   - Much better than the 4% recall at threshold 0.5

---

**Note:** To evaluate 36, 48, and 60-month experiments, the models would need to be retrained and saved separately.

