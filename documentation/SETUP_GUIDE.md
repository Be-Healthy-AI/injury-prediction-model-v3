# V3 Setup Guide

## Quick Start

### 1. Prepare Your Data
Place your V3 dataset Excel files in the `original_data/` directory:
- `*_players_profile.xlsx`
- `*_injuries_data.xlsx`
- `*_match_data.xlsx`
- `*_teams_data.xlsx`
- `*_competition_data.xlsx`

The scripts will auto-detect files matching these patterns.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Pipeline

#### Step 1: Generate Daily Features
```bash
cd scripts
python create_daily_features_v3.py
```
This will create daily feature files for all players in `features_daily_all_players_v3/`.

#### Step 2: Create 35-Day Timelines
```bash
cd scripts
python create_35day_timelines_v3.py
```
This generates `timelines_35day_enhanced_balanced_v3.csv`.

#### Step 3: Train the Model
```bash
cd scripts
python train_model_v3.py
```
This trains the Random Forest model and saves it to `models/`.

### 4. Make Predictions
```bash
cd scripts
python predict_v3.py \
  --player_csv "../features_daily_all_players_v3/player_XXXXX_daily_features.csv" \
  --player_id XXXXX \
  --player_name "Player Name" \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --out_csv "../results/predictions.csv"
```

## Differences from V2

1. **File Naming**: All scripts and outputs are named with `_v3` suffix
2. **Auto-Detection**: Data files are auto-detected by pattern matching
3. **Flexible Paths**: Scripts handle both V3 and V2 directory structures
4. **Enhanced Features**: Ready for additional features you want to add

## Adding New Features

To add new features:
1. Update `create_daily_features_v3.py` to calculate your new features
2. Add feature names to the feature selection list in the script
3. Regenerate daily features
4. The timelines and training scripts will automatically include them

## Troubleshooting

- **File Not Found**: Ensure your Excel files are in `original_data/` with correct naming
- **Import Errors**: Make sure you're running scripts from the `scripts/` directory
- **Path Issues**: Check that all relative paths are correct based on your working directory

## Next Steps

After setup, you can:
1. Experiment with new features
2. Tune model hyperparameters
3. Add visualization scripts
4. Implement automated retraining

