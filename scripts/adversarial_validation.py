#!/usr/bin/env python3
"""
Adversarial Validation: Identify features that distinguish training from validation
This helps identify features causing distribution shift
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
    
    return X_encoded

def main():
    print("="*80)
    print("ADVERSARIAL VALIDATION")
    print("="*80)
    print("📋 Goal: Identify features that distinguish training from validation")
    print("   Features with high importance = high distribution shift")
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
    X_train = prepare_data(df_train, drop_week5=False)
    X_val = prepare_data(df_val, drop_week5=False)
    
    # Align columns
    all_cols = sorted(set(X_train.columns) & set(X_val.columns))
    X_train = X_train[all_cols]
    X_val = X_val[all_cols]
    
    print(f"✅ Prepared data: {len(all_cols)} features")
    
    # Create adversarial target: 0 = train, 1 = validation
    print("\n🎯 Creating adversarial target...")
    X_combined = pd.concat([X_train, X_val], ignore_index=True)
    y_adversarial = np.concatenate([
        np.zeros(len(X_train)),  # 0 for training
        np.ones(len(X_val))       # 1 for validation
    ])
    
    print(f"   Training samples: {len(X_train):,} (label: 0)")
    print(f"   Validation samples: {len(X_val):,} (label: 1)")
    print(f"   Combined: {len(X_combined):,} samples")
    
    # Train adversarial classifier
    print("\n" + "="*80)
    print("🚀 TRAINING ADVERSARIAL CLASSIFIER")
    print("="*80)
    
    print("\n🔧 Model: Random Forest (100 trees, max_depth=10)")
    adversarial_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    print("\n🌳 Training started...")
    adversarial_model.fit(X_combined, y_adversarial)
    
    # Evaluate
    y_pred_proba = adversarial_model.predict_proba(X_combined)[:, 1]
    adversarial_auc = roc_auc_score(y_adversarial, y_pred_proba)
    
    print(f"\n📊 Adversarial Model Performance:")
    print(f"   ROC AUC: {adversarial_auc:.4f}")
    
    if adversarial_auc > 0.7:
        print("   ⚠️  HIGH distribution shift detected!")
        print("   The model can easily distinguish training from validation")
    elif adversarial_auc > 0.6:
        print("   ⚠️  MODERATE distribution shift detected")
    elif adversarial_auc > 0.55:
        print("   ⚠️  LOW-MODERATE distribution shift detected")
    else:
        print("   ✅ LOW distribution shift (good!)")
    
    # Get feature importance
    print("\n📊 Analyzing feature importance...")
    feature_importance = pd.DataFrame({
        'feature': all_cols,
        'importance': adversarial_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Identify problematic features (high importance = high shift)
    threshold_high = feature_importance['importance'].quantile(0.95)  # Top 5%
    threshold_medium = feature_importance['importance'].quantile(0.90)  # Top 10%
    
    high_shift_features = feature_importance[feature_importance['importance'] >= threshold_high]['feature'].tolist()
    medium_shift_features = feature_importance[
        (feature_importance['importance'] >= threshold_medium) & 
        (feature_importance['importance'] < threshold_high)
    ]['feature'].tolist()
    
    print(f"\n📋 Feature Shift Analysis:")
    print(f"   High-shift features (top 5%): {len(high_shift_features)}")
    print(f"   Medium-shift features (top 10%): {len(medium_shift_features)}")
    print(f"   Low-shift features: {len(all_cols) - len(high_shift_features) - len(medium_shift_features)}")
    
    # Show top problematic features
    print(f"\n📋 Top 20 Most Problematic Features (High Distribution Shift):")
    print(f"\n{'Feature':<60} {'Importance':<12}")
    print("-" * 75)
    for idx, row in feature_importance.head(20).iterrows():
        print(f"{row['feature']:<60} {row['importance']:<12.6f}")
    
    # Save results
    output_dir = Path('experiments')
    output_dir.mkdir(exist_ok=True)
    
    # Save feature importance
    csv_file = output_dir / 'adversarial_validation_features.csv'
    feature_importance.to_csv(csv_file, index=False)
    print(f"\n✅ Feature importance saved to {csv_file}")
    
    # Save problematic features lists
    problematic_features = {
        'adversarial_auc': float(adversarial_auc),
        'high_shift_features': high_shift_features,
        'medium_shift_features': medium_shift_features,
        'low_shift_features': [f for f in all_cols if f not in high_shift_features and f not in medium_shift_features],
        'threshold_high': float(threshold_high),
        'threshold_medium': float(threshold_medium)
    }
    
    json_file = output_dir / 'adversarial_validation_results.json'
    with open(json_file, 'w') as f:
        json.dump(problematic_features, f, indent=2)
    print(f"✅ Results saved to {json_file}")
    
    # Create summary
    summary_file = output_dir / 'adversarial_validation_summary.md'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("# Adversarial Validation Results\n\n")
        f.write("**Goal:** Identify features that distinguish training from validation datasets\n\n")
        f.write("## Summary\n\n")
        f.write(f"- **Adversarial ROC AUC:** {adversarial_auc:.4f}\n")
        f.write(f"- **Interpretation:** ")
        if adversarial_auc > 0.7:
            f.write("HIGH distribution shift - model can easily distinguish datasets\n")
        elif adversarial_auc > 0.6:
            f.write("MODERATE distribution shift\n")
        elif adversarial_auc > 0.55:
            f.write("LOW-MODERATE distribution shift\n")
        else:
            f.write("LOW distribution shift (good!)\n")
        
        f.write(f"\n## Feature Shift Categories\n\n")
        f.write(f"- **High-shift features (top 5%):** {len(high_shift_features)}\n")
        f.write(f"- **Medium-shift features (top 10%):** {len(medium_shift_features)}\n")
        f.write(f"- **Low-shift features:** {len(all_cols) - len(high_shift_features) - len(medium_shift_features)}\n")
        
        f.write(f"\n## Top 20 Most Problematic Features\n\n")
        f.write("| Rank | Feature | Importance |\n")
        f.write("|------|---------|-----------|\n")
        for rank, (idx, row) in enumerate(feature_importance.head(20).iterrows(), 1):
            f.write(f"| {rank} | {row['feature']} | {row['importance']:.6f} |\n")
        
        f.write(f"\n## Recommendations\n\n")
        if adversarial_auc > 0.6:
            f.write("1. **Remove high-shift features** from training\n")
            f.write("2. **Consider removing medium-shift features** if performance doesn't improve\n")
            f.write("3. **Investigate why these features shift** - may indicate data quality issues\n")
        else:
            f.write("1. Distribution shift is relatively low\n")
            f.write("2. Focus on other strategies (covariate shift correction, calibration)\n")
    
    print(f"✅ Summary saved to {summary_file}")
    print("\n" + "="*80)
    print("ADVERSARIAL VALIDATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()

