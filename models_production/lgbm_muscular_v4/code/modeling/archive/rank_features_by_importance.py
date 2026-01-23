#!/usr/bin/env python3
"""
Rank all features by their predictive importance using existing trained models.

This script:
1. Loads existing trained models (muscular and skeletal)
2. Extracts feature importances (gain-based) from both models
3. Ranks features by averaging importances across both models
4. Saves ranked feature list to JSON

No retraining needed - uses the already-trained V4 natural models!
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
from datetime import datetime
from pathlib import Path

# Add paths
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent.parent.parent
MODEL_OUTPUT_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'models'

# ========== CONFIGURATION ==========
MODEL1_PATH = MODEL_OUTPUT_DIR / 'lgbm_muscular_v4_natural_model1.joblib'
MODEL1_COLUMNS_PATH = MODEL_OUTPUT_DIR / 'lgbm_muscular_v4_natural_model1_columns.json'
MODEL2_PATH = MODEL_OUTPUT_DIR / 'lgbm_muscular_v4_natural_model2.joblib'
MODEL2_COLUMNS_PATH = MODEL_OUTPUT_DIR / 'lgbm_muscular_v4_natural_model2_columns.json'
RANKING_OUTPUT_FILE = MODEL_OUTPUT_DIR / 'feature_ranking.json'
# ===================================

def extract_feature_importances(model):
    """Extract feature importances from trained LightGBM model"""
    if hasattr(model, 'booster_'):
        # LightGBM model - use gain importance (more reliable than split)
        importances = model.booster_.feature_importance(importance_type='gain')
    else:
        # Fallback to sklearn-style importance
        importances = model.feature_importances_
    return importances

def rank_features_by_importance():
    """Load existing models and rank features by importance"""
    print("="*80)
    print("FEATURE RANKING FROM EXISTING MODELS")
    print("="*80)
    print(f"\n📋 Using pre-trained V4 natural models")
    print(f"   Model 1 (Muscular): {MODEL1_PATH.name}")
    print(f"   Model 2 (Skeletal): {MODEL2_PATH.name}")
    print(f"   Ranking method: LightGBM gain importance (averaged across both models)")
    print("="*80)
    
    start_time = datetime.now()
    print(f"\n⏰ Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if models exist
    if not MODEL1_PATH.exists():
        raise FileNotFoundError(f"Model 1 not found: {MODEL1_PATH}")
    if not MODEL2_PATH.exists():
        raise FileNotFoundError(f"Model 2 not found: {MODEL2_PATH}")
    if not MODEL1_COLUMNS_PATH.exists():
        raise FileNotFoundError(f"Model 1 columns not found: {MODEL1_COLUMNS_PATH}")
    if not MODEL2_COLUMNS_PATH.exists():
        raise FileNotFoundError(f"Model 2 columns not found: {MODEL2_COLUMNS_PATH}")
    
    # Load models
    print("\n" + "="*80)
    print("STEP 1: LOADING MODELS")
    print("="*80)
    
    print(f"   Loading Model 1 (Muscular)...")
    model1 = joblib.load(MODEL1_PATH)
    with open(MODEL1_COLUMNS_PATH, 'r') as f:
        feature_names1 = json.load(f)
    print(f"   ✅ Loaded Model 1 with {len(feature_names1)} features")
    
    print(f"\n   Loading Model 2 (Skeletal)...")
    model2 = joblib.load(MODEL2_PATH)
    with open(MODEL2_COLUMNS_PATH, 'r') as f:
        feature_names2 = json.load(f)
    print(f"   ✅ Loaded Model 2 with {len(feature_names2)} features")
    
    # Extract importances
    print("\n" + "="*80)
    print("STEP 2: EXTRACTING FEATURE IMPORTANCES")
    print("="*80)
    
    print(f"   Extracting importances from Model 1...")
    importances1 = extract_feature_importances(model1)
    print(f"   ✅ Extracted {len(importances1)} importances")
    
    print(f"\n   Extracting importances from Model 2...")
    importances2 = extract_feature_importances(model2)
    print(f"   ✅ Extracted {len(importances2)} importances")
    
    # Combine importances
    print("\n" + "="*80)
    print("STEP 3: COMBINING FEATURE IMPORTANCES")
    print("="*80)
    
    # Create DataFrames for easier manipulation
    df_imp1 = pd.DataFrame({
        'feature': feature_names1,
        'importance_muscular': importances1
    })
    
    df_imp2 = pd.DataFrame({
        'feature': feature_names2,
        'importance_skeletal': importances2
    })
    
    # Merge on common features
    df_combined = pd.merge(
        df_imp1, 
        df_imp2, 
        on='feature', 
        how='outer'
    ).fillna(0)
    
    # Calculate average importance
    df_combined['importance_avg'] = (
        df_combined['importance_muscular'] + df_combined['importance_skeletal']
    ) / 2.0
    
    # Normalize importances to 0-1 range for easier interpretation
    max_imp = df_combined['importance_avg'].max()
    if max_imp > 0:
        df_combined['importance_normalized'] = df_combined['importance_avg'] / max_imp
    else:
        df_combined['importance_normalized'] = 0.0
    
    # Sort by average importance (descending)
    df_combined = df_combined.sort_values('importance_avg', ascending=False).reset_index(drop=True)
    
    # Create ranked feature list
    ranked_features = df_combined['feature'].tolist()
    
    print(f"\n📊 Feature Ranking Summary:")
    print(f"   Total features ranked: {len(ranked_features)}")
    print(f"   Common features: {len(df_combined)}")
    print(f"   Model 1 only: {len(df_combined[df_combined['importance_skeletal'] == 0])}")
    print(f"   Model 2 only: {len(df_combined[df_combined['importance_muscular'] == 0])}")
    print(f"\n   Top 20 features:")
    for i, row in df_combined.head(20).iterrows():
        print(f"      {i+1:3d}. {row['feature']:50s} (avg: {row['importance_avg']:12.2f}, "
              f"muscular: {row['importance_muscular']:10.2f}, skeletal: {row['importance_skeletal']:10.2f})")
    
    # Save ranking
    print("\n" + "="*80)
    print("STEP 4: SAVING FEATURE RANKING")
    print("="*80)
    
    ranking_data = {
        'ranked_features': ranked_features,
        'feature_importances': {
            'muscular': {feat: float(imp) for feat, imp in zip(feature_names1, importances1)},
            'skeletal': {feat: float(imp) for feat, imp in zip(feature_names2, importances2)},
            'average': {row['feature']: float(row['importance_avg']) for _, row in df_combined.iterrows()},
            'normalized': {row['feature']: float(row['importance_normalized']) for _, row in df_combined.iterrows()}
        },
        'ranking_metadata': {
            'total_features': len(ranked_features),
            'ranking_method': 'LightGBM gain importance (averaged across both models)',
            'source_models': {
                'model1': str(MODEL1_PATH.name),
                'model2': str(MODEL2_PATH.name)
            },
            'ranking_date': start_time.isoformat()
        }
    }
    
    with open(RANKING_OUTPUT_FILE, 'w') as f:
        json.dump(ranking_data, f, indent=2)
    
    print(f"✅ Saved feature ranking to: {RANKING_OUTPUT_FILE}")
    
    # Also save detailed CSV for analysis
    csv_output = MODEL_OUTPUT_DIR / 'feature_ranking_detailed.csv'
    df_combined.to_csv(csv_output, index=False, encoding='utf-8-sig')
    print(f"✅ Saved detailed ranking to: {csv_output}")
    
    # Final summary
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    print(f"\n⏰ Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Total time: {total_time:.1f} seconds")
    print(f"\n✅ Feature ranking complete! Ready for iterative training.")
    print(f"   Next step: Run train_iterative_feature_selection.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(rank_features_by_importance())
