#!/usr/bin/env python3
"""
Analyze calibration of V4 Enriched models
- Generates calibration bins (0.0-0.1, 0.1-0.2, ..., 0.9-1.0)
- Calculates actual injury rates vs predicted probabilities
- Creates calibration chart and CSV data
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import json
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent.parent.parent
MODELS_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'models'
TIMELINES_DIR = SCRIPT_DIR.parent / 'timelines'
TEST_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'data' / 'timelines' / 'test'
OUTPUT_DIR = MODELS_DIR / 'calibration_analysis'

sys.path.insert(0, str(TIMELINES_DIR.resolve()))

# Import helper functions from training script
import importlib.util
training_script_path = SCRIPT_DIR / 'train_lgbm_v4_enriched_comparison.py'
spec = importlib.util.spec_from_file_location("train_lgbm_v4_enriched_comparison", training_script_path)
training_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training_module)

prepare_data = training_module.prepare_data
filter_timelines_for_model = training_module.filter_timelines_for_model
align_features = training_module.align_features

def load_test_data():
    """Load test dataset"""
    test_file = TEST_DIR / 'timelines_35day_season_2025_2026_v4_muscular_test.csv'
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    print(f"\n📂 Loading test dataset: {test_file.name}")
    df_test = pd.read_csv(test_file, encoding='utf-8-sig', low_memory=False)
    return df_test

def load_model(model_path, columns_path):
    """Load trained model and feature columns"""
    model = joblib.load(model_path)
    with open(columns_path, 'r') as f:
        columns = json.load(f)
    return model, columns

def generate_predictions(model, X, columns):
    """Generate predictions using model"""
    # Align features
    X_aligned = X.reindex(columns=columns, fill_value=0)
    X_aligned = X_aligned[columns]  # Ensure correct order
    
    # Generate probabilities
    y_proba = model.predict_proba(X_aligned)[:, 1]
    return y_proba

def bin_probability(prob):
    """Bin probability into 0-9 range"""
    return min(int(prob * 10), 9)

def calculate_calibration_data(y_true, y_proba, model_name):
    """Calculate calibration statistics by bin"""
    print(f"\n📊 Calculating calibration data for {model_name}...")
    
    # Create bins
    df = pd.DataFrame({
        'y_true': y_true,
        'y_proba': y_proba
    })
    df['bin'] = df['y_proba'].apply(bin_probability)
    
    # Calculate statistics per bin
    bin_stats = []
    for bin_idx in range(10):
        bin_data = df[df['bin'] == bin_idx]
        
        if len(bin_data) == 0:
            continue
        
        bin_start = bin_idx * 0.1
        bin_end = (bin_idx + 1) * 0.1
        bin_center = (bin_start + bin_end) / 2
        
        total_obs = len(bin_data)
        injuries = bin_data['y_true'].sum()
        injury_rate = (injuries / total_obs) * 100 if total_obs > 0 else 0.0
        avg_predicted = bin_data['y_proba'].mean() * 100  # Convert to percentage
        
        bin_stats.append({
            'bin': bin_idx,
            'probability_range': f"{bin_start:.1f}-{bin_end:.1f}",
            'bin_center': bin_center,
            'total_observations': total_obs,
            'injuries': int(injuries),
            'injury_rate_pct': injury_rate,
            'avg_predicted_prob_pct': avg_predicted
        })
    
    calibration_df = pd.DataFrame(bin_stats)
    
    print(f"\n📊 Calibration Summary for {model_name}:")
    print("="*80)
    print(f"{'Bin':<5} {'Range':<12} {'Obs':<10} {'Injuries':<10} {'Injury Rate %':<15} {'Avg Predicted %':<15}")
    print("-"*80)
    for _, row in calibration_df.iterrows():
        print(f"{row['bin']:<5} {row['probability_range']:<12} {row['total_observations']:<10,} "
              f"{row['injuries']:<10} {row['injury_rate_pct']:<15.2f} {row['avg_predicted_prob_pct']:<15.2f}")
    
    return calibration_df

def analyze_model_calibration(model_name_key, model_display_name):
    """Analyze calibration for a specific model"""
    print("\n" + "="*80)
    print(f"CALIBRATION ANALYSIS: {model_display_name}")
    print("="*80)
    
    # Load model
    model_path = MODELS_DIR / f'lgbm_muscular_v4_enriched_{model_name_key}.joblib'
    columns_path = MODELS_DIR / f'lgbm_muscular_v4_enriched_{model_name_key}_columns.json'
    
    if not model_path.exists() or not columns_path.exists():
        print(f"⚠️  Model files not found for {model_display_name}, skipping...")
        return None
    
    model, columns = load_model(model_path, columns_path)
    print(f"✅ Loaded model: {model_path.name}")
    print(f"   Features: {len(columns)}")
    
    # Load and prepare test data
    df_test_all = load_test_data()
    
    # Filter for appropriate target
    target_col = 'target1' if 'model1' in model_name_key else 'target2'
    df_test = filter_timelines_for_model(df_test_all, target_col)
    
    print(f"✅ Filtered test data: {len(df_test):,} records")
    print(f"   Positives: {(df_test[target_col] == 1).sum():,} ({(df_test[target_col] == 1).mean()*100:.4f}%)")
    
    # Prepare features
    print("\n🔧 Preparing features...")
    X_test = prepare_data(df_test, cache_file=None, use_cache=False)
    y_test = df_test[target_col].values
    
    # Align features
    X_test_aligned = X_test.reindex(columns=columns, fill_value=0)
    X_test_aligned = X_test_aligned[columns]
    
    # Generate predictions
    print("\n🔮 Generating predictions...")
    y_proba = generate_predictions(model, X_test, columns)
    
    # Calculate calibration
    calibration_df = calculate_calibration_data(y_test, y_proba, model_display_name)
    
    # Add model name
    calibration_df['model'] = model_display_name
    
    return calibration_df

def main():
    print("="*80)
    print("CALIBRATION ANALYSIS - V4 ENRICHED MODELS")
    print("="*80)
    print(f"\n⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Analyze both models
    all_calibration_data = []
    
    # Model 1 (Muscular)
    calib1 = analyze_model_calibration('model1', 'Model 1 (Muscular)')
    if calib1 is not None:
        all_calibration_data.append(calib1)
    
    # Model 2 (Skeletal)
    calib2 = analyze_model_calibration('model2', 'Model 2 (Skeletal)')
    if calib2 is not None:
        all_calibration_data.append(calib2)
    
    if not all_calibration_data:
        print("\n⚠️  No calibration data generated. Check that models exist.")
        return 1
    
    # Combine and save
    combined_calibration = pd.concat(all_calibration_data, ignore_index=True)
    
    # Save calibration data
    calibration_csv = OUTPUT_DIR / 'calibration_v4_enriched_data.csv'
    combined_calibration.to_csv(calibration_csv, index=False, encoding='utf-8-sig')
    print(f"\n✅ Saved calibration data: {calibration_csv}")
    
    # Create summary
    print("\n" + "="*80)
    print("CALIBRATION SUMMARY")
    print("="*80)
    
    for model_name in combined_calibration['model'].unique():
        model_data = combined_calibration[combined_calibration['model'] == model_name]
        print(f"\n{model_name}:")
        print(f"  Total observations: {model_data['total_observations'].sum():,}")
        print(f"  Total injuries: {model_data['injuries'].sum():,}")
        print(f"  Injuries in low-probability bins (0.0-0.3): {model_data[model_data['bin'] <= 2]['injuries'].sum():,}")
        print(f"  Injuries in high-probability bins (0.7-1.0): {model_data[model_data['bin'] >= 7]['injuries'].sum():,}")
    
    print("\n" + "="*80)
    print("CALIBRATION ANALYSIS COMPLETE")
    print("="*80)
    print(f"\n📁 Output saved to: {OUTPUT_DIR}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
