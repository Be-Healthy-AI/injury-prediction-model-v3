# Model Comparison V4 (Dual Validation)

Generated: 2025-11-24 16:46:35

## In-Sample Validation Performance (80/20 split)

              Model ROC AUC   Gini Precision Recall F1-Score Accuracy AUC Gap Overfitting Risk
      Random Forest  0.9964 0.9927    0.8608 0.9665   0.9106   0.9715  0.0036              LOW
  Gradient Boosting  0.9994 0.9988    0.9798 0.9681   0.9739   0.9922  0.0006              LOW
Logistic Regression  0.7289 0.4579    0.2599 0.7241   0.3826   0.6494  0.0005              LOW

## Out-of-Sample Validation Performance (temporal split)

              Model ROC AUC   Gini Precision Recall F1-Score Accuracy AUC Gap
      Random Forest  0.7633 0.5265    0.5000 0.1318   0.2086   0.8500  0.2367
  Gradient Boosting  0.7346 0.4691    0.6190 0.1773   0.2756   0.8603  0.2654
Logistic Regression  0.6505 0.3009    0.1922 0.8500   0.3135   0.4417  0.0789

## Overfitting Analysis (In-Sample)

              Model AUC Gap  F1 Gap Recall Gap Risk Level
      Random Forest  0.0036  0.0609     0.0331        LOW
  Gradient Boosting  0.0006  0.0261     0.0319        LOW
Logistic Regression  0.0005 -0.0030    -0.0110        LOW

## Best Model (In-Sample)

**Gradient Boosting** with ROC AUC: 0.9994

## Best Model (Out-of-Sample)

**Random Forest** with ROC AUC: 0.7633

## Detailed Metrics

### Random Forest

#### Training Set
- accuracy: 0.9912
- precision: 0.9450
- recall: 0.9996
- f1: 0.9715
- roc_auc: 0.9999
- gini: 0.9999

#### In-Sample Validation Set
- accuracy: 0.9715
- precision: 0.8608
- recall: 0.9665
- f1: 0.9106
- roc_auc: 0.9964
- gini: 0.9927

#### Out-of-Sample Validation Set
- accuracy: 0.8500
- precision: 0.5000
- recall: 0.1318
- f1: 0.2086
- roc_auc: 0.7633
- gini: 0.5265

### Gradient Boosting

#### Training Set
- accuracy: 1.0000
- precision: 1.0000
- recall: 1.0000
- f1: 1.0000
- roc_auc: 1.0000
- gini: 1.0000

#### In-Sample Validation Set
- accuracy: 0.9922
- precision: 0.9798
- recall: 0.9681
- f1: 0.9739
- roc_auc: 0.9994
- gini: 0.9988

#### Out-of-Sample Validation Set
- accuracy: 0.8603
- precision: 0.6190
- recall: 0.1773
- f1: 0.2756
- roc_auc: 0.7346
- gini: 0.4691

### Logistic Regression

#### Training Set
- accuracy: 0.6504
- precision: 0.2586
- recall: 0.7131
- f1: 0.3796
- roc_auc: 0.7294
- gini: 0.4588

#### In-Sample Validation Set
- accuracy: 0.6494
- precision: 0.2599
- recall: 0.7241
- f1: 0.3826
- roc_auc: 0.7289
- gini: 0.4579

#### Out-of-Sample Validation Set
- accuracy: 0.4417
- precision: 0.1922
- recall: 0.8500
- f1: 0.3135
- roc_auc: 0.6505
- gini: 0.3009

