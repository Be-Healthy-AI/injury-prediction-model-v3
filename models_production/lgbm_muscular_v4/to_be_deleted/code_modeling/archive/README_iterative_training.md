# Iterative Feature Selection Training

## Issue with Cursor Terminal

The iterative training script uses `importlib` to dynamically import modules, which conflicts with Cursor's terminal environment (stdout/stderr handling). This causes an "I/O operation on closed file" error.

## Solution: Run Outside Cursor

**The script must be run outside of Cursor's integrated terminal.**

### Option 1: Use the Batch File (Recommended)

1. Navigate to: `models_production\lgbm_muscular_v4\code\modeling`
2. Double-click `run_iterative_training_no_nested.bat`
3. Or run it from Windows Command Prompt/PowerShell

### Option 2: Run from Windows Command Prompt

```cmd
cd "C:\Users\joao.henriques\IPM V3\models_production\lgbm_muscular_v4\code\modeling"
python train_iterative_feature_selection_no_nested_importlib.py
```

### Option 3: Run from PowerShell

```powershell
cd "C:\Users\joao.henriques\IPM V3\models_production\lgbm_muscular_v4\code\modeling"
python train_iterative_feature_selection_no_nested_importlib.py
```

## Files

- **`train_iterative_feature_selection_no_nested_importlib.py`**: Main script (avoids nested importlib calls)
- **`run_iterative_training_no_nested.bat`**: Batch file to run the script
- **Log file**: `models_production\lgbm_muscular_v4\models\iterative_training.log`
- **Results file**: `models_production\lgbm_muscular_v4\models\iterative_feature_selection_results.json`

## What the Script Does

1. Loads ranked features from `feature_ranking.json`
2. Trains models iteratively with increasing feature sets (20, 40, 60, ...)
3. Tracks performance metrics (Gini and F1-Score on test set)
4. Stops when 3 consecutive drops in performance are detected
5. Identifies the optimal number of features

## Monitoring Progress

- Check the log file: `models\iterative_training.log`
- Check intermediate results: `models\iterative_feature_selection_results.json` (updated after each iteration)
- Use `check_iterative_training_status.py` to view current status

## Expected Duration

- Each iteration: ~5-10 minutes (depends on number of features)
- Total iterations: Up to ~45 (from 20 to 898 features, 20 per iteration)
- Worst case: ~6-8 hours total
