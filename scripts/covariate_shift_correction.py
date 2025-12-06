#!/usr/bin/env python3
"""
Covariate Shift Correction: Re-weight training samples to match validation distribution
Estimates P_validation(x) / P_training(x) for each sample and uses as weights
"""
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import joblib
import json
from pathlib import Path

def prepare_data(df, drop_week5=False):
    """Prepare data with same encoding logic as training scripts"""
    feature_columns = [
        col for col in df.columns
        if col not in ['player_id', 'reference_date', 'player_name', 'target']
        and (drop_week5 is False or '_week_5' not in col)
    ]
    X = df[feature_columns].copy()
    y = df['target'].copy()
    
    # Handle missing values and encode categorical features
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    X_encoded = X.copy()
    
    # Encode categorical features
    for feature in categorical_features:
        X_encoded[feature] = X_encoded[feature].fillna('Unknown')
        dummies = pd.get_dummies(X_encoded[feature], prefix=feature, drop_first=True)
        X_encoded = pd.concat([X_encoded, dummies], axis=1)
        X_encoded = X_encoded.drop(columns=[feature])
    
    # Fill numeric missing values
    for feature in numeric_features:
        X_encoded[feature] = X_encoded[feature].fillna(0)
    
    return X_encoded, y

def estimate_sample_weights(X_train, X_val, method='adversarial'):
    """
    Estimate sample weights: P_val(x) / P_train(x)
    
    Methods:
    - 'adversarial': Use adversarial classifier probabilities
    - 'kde': Kernel Density Estimation (slower but more accurate)
    """
    if method == 'adversarial':
        # Train adversarial classifier
        print("   Training adversarial classifier for weight estimation...")
        X_combined = pd.concat([X_train, X_val], ignore_index=True)
        y_adversarial = np.concatenate([
            np.zeros(len(X_train)),  # 0 = train
            np.ones(len(X_val))       # 1 = val
        ])
        
        # Align columns
        all_cols = sorted(set(X_train.columns) & set(X_val.columns))
        X_combined = X_combined[all_cols]
        X_train_aligned = X_train[all_cols]
        
        # Train lightweight model
        adversarial = RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            min_samples_split=50,
            min_samples_leaf=20,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        adversarial.fit(X_combined, y_adversarial)
        
        # Get probabilities P(val | x) for training samples
        p_val_given_x = adversarial.predict_proba(X_train_aligned)[:, 1]
        
        # Convert to weights: P_val(x) / P_train(x) = P(val|x) / (1 - P(val|x))
        # But we need to normalize properly
        # Using: weight = P(val|x) / P(train|x) = P(val|x) / (1 - P(val|x))
        # Then normalize so mean weight = 1
        weights = p_val_given_x / (1 - p_val_given_x + 1e-10)
        weights = weights / weights.mean()  # Normalize to mean = 1
        
        return weights, adversarial
    
    else:
        raise ValueError(f"Unknown method: {method}")

def main():
    print("="*80)
    print("COVARIATE SHIFT CORRECTION")
    print("="*80)
    print("📋 Goal: Re-weight training samples to match validation distribution")
    print("="*80)
    
    # Load data
    print("\n📂 Loading timeline data...")
    train_file = 'timelines_35day_enhanced_balanced_v4_train.csv'
    val_file = 'timelines_35day_enhanced_balanced_v4_val.csv'
    
    df_train = pd.read_csv(train_file, encoding='utf-8-sig')
    df_val = pd.read_csv(val_file, encoding='utf-8-sig')
    
    print(f"✅ Loaded training set: {len(df_train):,} records")
    print(f"✅ Loaded validation set: {len(df_val):,} records")
    
    # Prepare data
    print("\n🔧 Preparing data...")
    X_train, y_train = prepare_data(df_train, drop_week5=False)
    X_val, y_val = prepare_data(df_val, drop_week5=False)
    
    # Align columns
    all_cols = sorted(set(X_train.columns) & set(X_val.columns))
    X_train = X_train[all_cols]
    X_val = X_val[all_cols]
    
    print(f"✅ Prepared data: {len(all_cols)} features")
    
    # Estimate sample weights
    print("\n⚖️  Estimating sample weights...")
    sample_weights, adversarial_model = estimate_sample_weights(X_train, X_val, method='adversarial')
    
    print(f"\n📊 Sample Weight Statistics:")
    print(f"   Mean weight: {sample_weights.mean():.4f}")
    print(f"   Min weight: {sample_weights.min():.4f}")
    print(f"   Max weight: {sample_weights.max():.4f}")
    print(f"   Std weight: {sample_weights.std():.4f}")
    print(f"   Median weight: {np.median(sample_weights):.4f}")
    
    # Show weight distribution
    print(f"\n📊 Weight Distribution:")
    print(f"   Weights < 0.5: {(sample_weights < 0.5).sum()} samples ({(sample_weights < 0.5).mean()*100:.1f}%)")
    print(f"   Weights 0.5-1.0: {((sample_weights >= 0.5) & (sample_weights < 1.0)).sum()} samples ({((sample_weights >= 0.5) & (sample_weights < 1.0)).mean()*100:.1f}%)")
    print(f"   Weights 1.0-2.0: {((sample_weights >= 1.0) & (sample_weights < 2.0)).sum()} samples ({((sample_weights >= 1.0) & (sample_weights < 2.0)).mean()*100:.1f}%)")
    print(f"   Weights >= 2.0: {(sample_weights >= 2.0).sum()} samples ({(sample_weights >= 2.0).mean()*100:.1f}%)")
    
    # Save weights and model
    output_dir = Path('experiments')
    output_dir.mkdir(exist_ok=True)
    
    # Save weights
    weights_file = output_dir / 'covariate_shift_weights.npy'
    np.save(weights_file, sample_weights)
    print(f"\n✅ Sample weights saved to {weights_file}")
    
    # Save adversarial model
    model_file = output_dir / 'covariate_shift_adversarial_model.joblib'
    joblib.dump(adversarial_model, model_file)
    print(f"✅ Adversarial model saved to {model_file}")
    
    # Save summary
    summary_file = output_dir / 'covariate_shift_correction_summary.md'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("# Covariate Shift Correction Summary\n\n")
        f.write("**Goal:** Re-weight training samples to match validation distribution\n\n")
        f.write("## Sample Weight Statistics\n\n")
        f.write(f"- **Mean weight:** {sample_weights.mean():.4f}\n")
        f.write(f"- **Min weight:** {sample_weights.min():.4f}\n")
        f.write(f"- **Max weight:** {sample_weights.max():.4f}\n")
        f.write(f"- **Std weight:** {sample_weights.std():.4f}\n")
        f.write(f"- **Median weight:** {np.median(sample_weights):.4f}\n")
        f.write(f"\n## Weight Distribution\n\n")
        f.write(f"- **Weights < 0.5:** {(sample_weights < 0.5).sum()} samples ({(sample_weights < 0.5).mean()*100:.1f}%)\n")
        f.write(f"- **Weights 0.5-1.0:** {((sample_weights >= 0.5) & (sample_weights < 1.0)).sum()} samples ({((sample_weights >= 0.5) & (sample_weights < 1.0)).mean()*100:.1f}%)\n")
        f.write(f"- **Weights 1.0-2.0:** {((sample_weights >= 1.0) & (sample_weights < 2.0)).sum()} samples ({((sample_weights >= 1.0) & (sample_weights < 2.0)).mean()*100:.1f}%)\n")
        f.write(f"- **Weights >= 2.0:** {(sample_weights >= 2.0).sum()} samples ({(sample_weights >= 2.0).mean()*100:.1f}%)\n")
        f.write(f"\n## Usage\n\n")
        f.write("Use these weights when training models:\n")
        f.write("```python\n")
        f.write("weights = np.load('experiments/covariate_shift_weights.npy')\n")
        f.write("model.fit(X_train, y_train, sample_weight=weights)\n")
        f.write("```\n")
    
    print(f"✅ Summary saved to {summary_file}")
    print("\n" + "="*80)
    print("COVARIATE SHIFT CORRECTION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()



