# Covariate Shift Correction Summary

**Goal:** Re-weight training samples to match validation distribution

## Sample Weight Statistics

- **Mean weight:** 1.0000
- **Min weight:** 0.0000
- **Max weight:** 31.5246
- **Std weight:** 2.1326
- **Median weight:** 0.2404

## Weight Distribution

- **Weights < 0.5:** 29174 samples (69.8%)
- **Weights 0.5-1.0:** 4903 samples (11.7%)
- **Weights 1.0-2.0:** 2425 samples (5.8%)
- **Weights >= 2.0:** 5298 samples (12.7%)

## Usage

Use these weights when training models:
```python
weights = np.load('experiments/covariate_shift_weights.npy')
model.fit(X_train, y_train, sample_weight=weights)
```
