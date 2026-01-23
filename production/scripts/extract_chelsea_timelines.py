#!/usr/bin/env python3
"""
Extract Chelsea players' timelines from the training dataset.
"""

import json
import pandas as pd
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent
ROOT_DIR = PRODUCTION_ROOT.parent

# Input: Training dataset
TRAINING_CSV = ROOT_DIR / "models_production" / "lgbm_muscular_v1" / "data" / "timelines" / "train" / "timelines_35day_season_2025_2026_v4_muscular.csv"

# Output: Production timelines folder
OUTPUT_DIR = PRODUCTION_ROOT / "deployments" / "England" / "Chelsea FC" / "timelines"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load Chelsea player IDs from config
CONFIG_PATH = PRODUCTION_ROOT / "deployments" / "England" / "Chelsea FC" / "config.json"

def main():
    # Load Chelsea player IDs
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
    chelsea_player_ids = set(config['player_ids'])
    print(f"Loaded {len(chelsea_player_ids)} Chelsea player IDs")
    
    # Load training CSV
    print(f"Loading training CSV: {TRAINING_CSV}")
    df = pd.read_csv(TRAINING_CSV, low_memory=False)
    print(f"Total timelines in training CSV: {len(df)}")
    
    # Filter to Chelsea players
    if 'player_id' in df.columns:
        chelsea_df = df[df['player_id'].isin(chelsea_player_ids)].copy()
        print(f"Chelsea timelines found: {len(chelsea_df)}")
        
        # Save to output folder
        output_file = OUTPUT_DIR / "timelines_35day_season_2025_2026_v4_muscular.csv"
        chelsea_df.to_csv(output_file, index=False)
        print(f"Saved Chelsea timelines to: {output_file}")
    else:
        print("ERROR: 'player_id' column not found in training CSV")

if __name__ == "__main__":
    main()










