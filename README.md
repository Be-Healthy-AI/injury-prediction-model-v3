# Injury Prediction Model V3

**Production-ready machine learning model for predicting football player injuries using 35-day timelines.**

## 🎯 Project Overview

This project implements a comprehensive injury prediction system using daily player features aggregated into 35-day timelines. This is **Version 3** of the injury prediction model, based on the successful V2 implementation with an expanded dataset and additional features.

## 📁 Project Structure

```
injury-prediction-model-v3/
├── original_data/                    # Source data (Excel files)
│   ├── *_players_profile.xlsx
│   ├── *_injuries_data.xlsx
│   ├── *_match_data.xlsx
│   ├── *_teams_data.xlsx
│   └── *_competition_data.xlsx
├── features_daily_all_players_v3/    # Generated daily features (CSV files per player)
├── scripts/                          # Production scripts
│   ├── create_daily_features_v3.py
│   ├── create_35day_timelines_v3.py
│   ├── train_model_v3.py
│   ├── predict_v3.py
│   └── benfica_parity_config.py
├── models/                           # Persisted model artifacts
│   ├── model_v3_random_forest_100percent.pkl
│   └── model_v3_rf_100percent_training_columns.json
├── results/                          # Analysis and prediction results
├── documentation/                   # Reports and analysis
└── README.md                         # This file
```

## 🚀 Complete Model Workflow

### Step 1: Generate Daily Features
```bash
cd scripts
python create_daily_features_v3.py
```
**Input:** `original_data/*.xlsx`  
**Output:** `features_daily_all_players_v3/*.csv` (one file per player)

### Step 2: Generate 35-Day Timelines
```bash
cd scripts
python create_35day_timelines_v3.py
```
**Input:** `features_daily_all_players_v3/*.csv`  
**Output:** `timelines_35day_enhanced_balanced_v3.csv`

### Step 3: Train Model
```bash
cd scripts
python train_model_v3.py
```
**Input:** `timelines_35day_enhanced_balanced_v3.csv`  
**Outputs:** Model files in `../models/`

## 📊 Model Details

### Model Architecture
- **Algorithm:** Random Forest Classifier
- **Features:** 108+ enhanced daily features aggregated into 35-day windows
- **Timeline Features:** Static features + 5-week rolling aggregations

### Key Features
- **Static Features:** Player profile, position, nationality, height, etc.
- **Windowed Features:** 5-week aggregations of match data
- **Injury Features:** Cumulative injury history and patterns
- **Performance Features:** Goals, assists, minutes, cards
- **Career Features:** Long-term performance metrics

## 🔮 Inference

### Predict per player over a date range
```bash
cd scripts
python predict_v3.py \
  --player_csv "../features_daily_all_players_v3/player_XXXXX_daily_features.csv" \
  --player_id XXXXX \
  --player_name "Player Name" \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --out_csv "../results/preds_player_XXXXX.csv"
```

For production, refresh daily features, then run the same command with `--start` = `--end` = today's date for each player.

## 📋 Requirements

```bash
pip install -r requirements.txt
```

Dependencies:
- pandas>=1.5.0
- numpy>=1.21.0
- matplotlib>=3.5.0
- seaborn>=0.11.0
- openpyxl>=3.0.0
- xlrd>=2.0.0
- scikit-learn>=1.1.0
- joblib>=1.2.0
- tqdm>=4.65.0

## 🔄 V2 to V3 Differences

V3 is based on V2 with the following enhancements:
- **Expanded Dataset:** More data for improved model training
- **Additional Features:** New features for enhanced prediction capability
- **Flexible Data Loading:** Auto-detection of data files
- **Updated Paths:** All scripts adapted for V3 structure

## 📈 Future Improvements

1. **Feature Engineering:** Explore additional injury-related features
2. **Model Enhancement:** Experiment with ensemble methods
3. **Real-time Updates:** Implement automated retraining pipelines
4. **Model Monitoring:** Track performance on new data

## 📞 Support

For questions or issues, refer to the documentation files in the `documentation/` directory.

---

**Model Version:** Random Forest V3  
**Based on:** V2 (99.78% Gini coefficient)  
**Status:** In Development 🚧
