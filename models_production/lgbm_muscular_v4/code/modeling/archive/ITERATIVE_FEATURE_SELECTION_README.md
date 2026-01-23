# Iterative Feature Selection

This directory contains scripts for iterative feature selection to identify the optimal number of features that maximize model performance.

## Overview

The iterative feature selection process:
1. **Ranks all features** by their predictive importance (using LightGBM gain importance)
2. **Trains models iteratively** with increasing feature sets (20, 40, 60, ... features)
3. **Tracks performance** using a weighted combination of Gini coefficient and F1-Score
4. **Stops automatically** when 3 consecutive drops in performance are detected
5. **Identifies the optimal** number of features

## Scripts

### 1. `rank_features_by_importance.py`
**Purpose:** Rank all 898 features by their predictive importance.

**What it does:**
- Trains models on all available features
- Extracts feature importances (gain-based) from both muscular and skeletal models
- Averages importances across both models
- Saves ranked feature list to `feature_ranking.json`

**Usage:**
```bash
python rank_features_by_importance.py
```

**Output:**
- `models/feature_ranking.json` - Ranked feature list with importances
- `models/feature_ranking_detailed.csv` - Detailed CSV with all importance metrics

**Time:** ~30-60 minutes (one-time operation)

---

### 2. `train_iterative_feature_selection.py`
**Purpose:** Main script that orchestrates iterative training.

**What it does:**
- Loads ranked features from `feature_ranking.json`
- Iteratively trains models with increasing feature sets:
  - Iteration 1: Top 20 features
  - Iteration 2: Top 40 features
  - Iteration 3: Top 60 features
  - ... and so on
- Calculates combined performance score for each iteration
- Stops when 3 consecutive drops are detected
- Saves results to `iterative_feature_selection_results.json`

**Usage:**
```bash
python train_iterative_feature_selection.py
```

**Output:**
- `models/iterative_feature_selection_results.json` - Complete results for all iterations
- `models/iterative_feature_selection_plot.png` - Performance progression plot (if matplotlib available)

**Time:** ~6-7.5 hours (for 45 iterations worst case)

---

### 3. `train_with_feature_subset.py`
**Purpose:** Utility module for training models with a specific feature subset.

**Note:** This is a helper module used by the iterative training script. You typically don't run this directly.

---

## Configuration

### Performance Metric
The combined performance score is calculated as:
```
combined_score = 0.6 * Gini + 0.4 * F1-Score
```
(Weighted average across both muscular and skeletal models)

### Parameters (in `train_iterative_feature_selection.py`)
```python
FEATURES_PER_ITERATION = 20      # Features to add each iteration
INITIAL_FEATURES = 20          # Starting number of features
CONSECUTIVE_DROPS_THRESHOLD = 3 # Stop after N consecutive drops
PERFORMANCE_DROP_THRESHOLD = 0.001  # Minimum drop to count (0.1%)
GINI_WEIGHT = 0.6              # Weight for Gini coefficient
F1_WEIGHT = 0.4                # Weight for F1-Score
```

### Adjusting Weights
If you want to prioritize Gini over F1-Score (or vice versa), modify:
```python
GINI_WEIGHT = 0.7  # Increase to prioritize Gini
F1_WEIGHT = 0.3    # Decrease accordingly
```

---

## Workflow

### Step 1: Rank Features (One-time)
```bash
cd models_production/lgbm_muscular_v4/code/modeling
python rank_features_by_importance.py
```

This will create `feature_ranking.json` with all features ranked by importance.

### Step 2: Run Iterative Training
```bash
python train_iterative_feature_selection.py
```

This will:
- Load the ranked features
- Train models iteratively
- Save results after each iteration (so you can monitor progress)
- Stop when 3 consecutive drops are detected
- Generate a summary and plot

### Step 3: Analyze Results
Open `iterative_feature_selection_results.json` to see:
- Performance metrics for each iteration
- Best iteration and number of features
- Complete training history

---

## Results Format

### `iterative_feature_selection_results.json`
```json
{
  "iterations": [
    {
      "iteration": 1,
      "n_features": 20,
      "features": [...],
      "model1_muscular": {
        "train": {...},
        "test": {...}
      },
      "model2_skeletal": {
        "train": {...},
        "test": {...}
      },
      "combined_score": 0.75,
      "timestamp": "...",
      "training_time_seconds": 480.5
    },
    ...
  ],
  "best_iteration": 3,
  "best_n_features": 60,
  "best_combined_score": 0.78,
  "configuration": {...}
}
```

---

## Monitoring Progress

The script saves results after each iteration, so you can monitor progress by checking:
```bash
# View latest results
cat models/iterative_feature_selection_results.json | python -m json.tool | tail -50
```

Or in Python:
```python
import json
with open('models/iterative_feature_selection_results.json') as f:
    results = json.load(f)
    
print(f"Completed {len(results['iterations'])} iterations")
print(f"Best: Iteration {results['best_iteration']} with {results['best_n_features']} features")
```

---

## Troubleshooting

### "Feature ranking file not found"
**Solution:** Run `rank_features_by_importance.py` first.

### Script stops early
**Possible reasons:**
- 3 consecutive drops detected (expected behavior)
- Error in training (check logs)
- All features used

### Want to resume from a specific iteration
Currently not supported, but you can:
1. Modify `INITIAL_FEATURES` to start from a higher number
2. Or manually edit `feature_ranking.json` to remove already-tested features

---

## Expected Output

After completion, you should see:
```
✅ Saved results to: models/iterative_feature_selection_results.json
✅ Saved performance plot to: models/iterative_feature_selection_plot.png

📊 Training Summary:
   Total iterations: 15
   Best iteration: 8
   Best number of features: 160
   Best combined score: 0.7823
```

---

## Next Steps

After identifying the optimal number of features:
1. Review the best feature set
2. Train final models with the optimal feature count
3. Compare against previous models
4. Consider further feature engineering on the selected features

---

## Notes

- **Caching:** Preprocessed data is cached to speed up subsequent iterations
- **Time:** Each iteration takes ~8-10 minutes (both models)
- **Stopping:** The script stops automatically when 3 consecutive drops are detected
- **Results:** Results are saved after each iteration, so progress is not lost if interrupted
