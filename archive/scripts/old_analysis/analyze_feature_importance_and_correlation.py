#!/usr/bin/env python3
"""
Analyze feature importance and correlation differences between training and out-of-sample validation
to identify features that may explain the performance gap
"""
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def prepare_data(df, drop_week5=False):
    """Prepare data with same logic as training scripts"""
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

def get_feature_importance_rf(model, feature_names):
    """Extract feature importance from Random Forest"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    return pd.DataFrame({
        'feature': [feature_names[i] for i in indices],
        'importance': importances[indices]
    })

def get_feature_importance_gb(model, feature_names):
    """Extract feature importance from Gradient Boosting"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    return pd.DataFrame({
        'feature': [feature_names[i] for i in indices],
        'importance': importances[indices]
    })

def get_feature_importance_lr(model, feature_names):
    """Extract feature coefficients from Logistic Regression"""
    coefficients = np.abs(model.coef_[0])
    indices = np.argsort(coefficients)[::-1]
    return pd.DataFrame({
        'feature': [feature_names[i] for i in indices],
        'importance': coefficients[indices]
    })

def calculate_correlations(X, y, feature_names):
    """Calculate correlation between each feature and target"""
    correlations = []
    for i, feat in enumerate(feature_names):
        if feat in X.columns:
            try:
                corr = np.corrcoef(X[feat].values, y.values)[0, 1]
                if np.isnan(corr):
                    corr = 0.0
                correlations.append({
                    'feature': feat,
                    'correlation': corr,
                    'abs_correlation': abs(corr)
                })
            except:
                correlations.append({
                    'feature': feat,
                    'correlation': 0.0,
                    'abs_correlation': 0.0
                })
    return pd.DataFrame(correlations).sort_values('abs_correlation', ascending=False)

def main():
    print("="*80)
    print("FEATURE IMPORTANCE AND CORRELATION ANALYSIS")
    print("="*80)
    
    # Load data
    print("\n📂 Loading datasets...")
    train_file = 'timelines_35day_enhanced_balanced_v4_train.csv'
    val_file = 'timelines_35day_enhanced_balanced_v4_val.csv'
    
    df_train = pd.read_csv(train_file, encoding='utf-8-sig')
    df_val = pd.read_csv(val_file, encoding='utf-8-sig')
    
    print(f"✅ Training: {len(df_train):,} records")
    print(f"✅ Validation: {len(df_val):,} records")
    
    # Prepare data
    print("\n🔧 Preparing data...")
    X_train_full, y_train_full = prepare_data(df_train, drop_week5=False)
    X_val, y_val = prepare_data(df_val, drop_week5=False)
    
    # Split training for in-sample
    X_train, X_insample, y_train, y_insample = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=y_train_full
    )
    
    # Ensure same columns
    all_cols = sorted(set(X_train.columns) | set(X_insample.columns) | set(X_val.columns))
    X_train = X_train.reindex(columns=all_cols, fill_value=0)
    X_insample = X_insample.reindex(columns=all_cols, fill_value=0)
    X_val = X_val.reindex(columns=all_cols, fill_value=0)
    
    feature_names = X_train.columns.tolist()
    
    print(f"✅ Features: {len(feature_names):,}")
    
    # Load models
    print("\n📂 Loading models...")
    models = {}
    model_configs = [
        ('rf', 'models/rf_model_v4.joblib', 'models/rf_model_v4_columns.json', get_feature_importance_rf),
        ('gb', 'models/gb_model_v4.joblib', 'models/gb_model_v4_columns.json', get_feature_importance_gb),
        ('lr', 'models/lr_model_v4.joblib', 'models/lr_model_v4_columns.json', get_feature_importance_lr),
    ]
    
    for model_key, model_path, cols_path, importance_func in model_configs:
        if Path(model_path).exists() and Path(cols_path).exists():
            model = joblib.load(model_path)
            with open(cols_path, 'r') as f:
                model_cols = json.load(f)
            
            # Get feature importance
            importance_df = importance_func(model, model_cols)
            models[model_key] = {
                'model': model,
                'columns': model_cols,
                'importance': importance_df
            }
            print(f"✅ Loaded {model_key.upper()} model")
        else:
            print(f"⚠️  {model_key.upper()} model not found")
    
    # Calculate correlations
    print("\n📊 Calculating feature-target correlations...")
    
    print("   Training set...")
    corr_train = calculate_correlations(X_train, y_train, feature_names)
    
    print("   In-sample validation...")
    corr_insample = calculate_correlations(X_insample, y_insample, feature_names)
    
    print("   Out-of-sample validation...")
    corr_val = calculate_correlations(X_val, y_val, feature_names)
    
    # Merge correlations
    corr_comparison = pd.merge(
        corr_train[['feature', 'abs_correlation']].rename(columns={'abs_correlation': 'train_corr'}),
        corr_insample[['feature', 'abs_correlation']].rename(columns={'abs_correlation': 'insample_corr'}),
        on='feature',
        how='outer'
    )
    corr_comparison = pd.merge(
        corr_comparison,
        corr_val[['feature', 'abs_correlation']].rename(columns={'abs_correlation': 'val_corr'}),
        on='feature',
        how='outer'
    )
    corr_comparison = corr_comparison.fillna(0)
    
    # Calculate correlation drift
    corr_comparison['train_val_diff'] = corr_comparison['train_corr'] - corr_comparison['val_corr']
    corr_comparison['insample_val_diff'] = corr_comparison['insample_corr'] - corr_comparison['val_corr']
    corr_comparison = corr_comparison.sort_values('train_val_diff', ascending=False)
    
    # Combine with model importance
    print("\n📊 Combining with model importance...")
    
    results = []
    for model_key, model_data in models.items():
        importance_df = model_data['importance']
        
        # Merge with correlations
        merged = pd.merge(
            importance_df,
            corr_comparison,
            on='feature',
            how='left'
        )
        merged = merged.fillna(0)
        merged['model'] = model_key.upper()
        
        results.append(merged)
    
    all_results = pd.concat(results, ignore_index=True)
    
    # Find suspicious features (high importance, high correlation in train, low in val)
    print("\n🔍 Identifying suspicious features...")
    
    suspicious_features = []
    for model_key in ['RF', 'GB', 'LR']:
        model_results = all_results[all_results['model'] == model_key].copy()
        
        # Top features by importance
        top_importance = model_results.nlargest(20, 'importance')
        
        # Features with high train correlation but low val correlation
        high_drift = model_results[
            (model_results['train_corr'] > 0.1) & 
            (model_results['train_val_diff'] > 0.05)
        ].nlargest(20, 'train_val_diff')
        
        suspicious_features.append({
            'model': model_key,
            'top_importance': top_importance,
            'high_drift': high_drift
        })
    
    # Print results
    print("\n" + "="*80)
    print("TOP FEATURES BY IMPORTANCE (Top 20 per model)")
    print("="*80)
    
    for sus in suspicious_features:
        print(f"\n{sus['model']} Model:")
        print(f"\n  Top 20 by Importance:")
        print(sus['top_importance'][['feature', 'importance', 'train_corr', 'val_corr', 'train_val_diff']].to_string(index=False))
    
    print("\n" + "="*80)
    print("FEATURES WITH HIGH CORRELATION DRIFT")
    print("(High correlation in training, low in validation)")
    print("="*80)
    
    for sus in suspicious_features:
        if len(sus['high_drift']) > 0:
            print(f"\n{sus['model']} Model:")
            print(sus['high_drift'][['feature', 'importance', 'train_corr', 'val_corr', 'train_val_diff']].to_string(index=False))
        else:
            print(f"\n{sus['model']} Model: No high-drift features found")
    
    # Summary statistics
    print("\n" + "="*80)
    print("CORRELATION STATISTICS")
    print("="*80)
    
    print(f"\nTraining set:")
    print(f"  Mean absolute correlation: {corr_train['abs_correlation'].mean():.4f}")
    print(f"  Max absolute correlation: {corr_train['abs_correlation'].max():.4f}")
    print(f"  Features with |corr| > 0.1: {(corr_train['abs_correlation'] > 0.1).sum()}")
    print(f"  Features with |corr| > 0.2: {(corr_train['abs_correlation'] > 0.2).sum()}")
    
    print(f"\nOut-of-sample validation:")
    print(f"  Mean absolute correlation: {corr_val['abs_correlation'].mean():.4f}")
    print(f"  Max absolute correlation: {corr_val['abs_correlation'].max():.4f}")
    print(f"  Features with |corr| > 0.1: {(corr_val['abs_correlation'] > 0.1).sum()}")
    print(f"  Features with |corr| > 0.2: {(corr_val['abs_correlation'] > 0.2).sum()}")
    
    print(f"\nCorrelation drift (train - val):")
    print(f"  Mean drift: {corr_comparison['train_val_diff'].mean():.4f}")
    print(f"  Max drift: {corr_comparison['train_val_diff'].max():.4f}")
    print(f"  Features with drift > 0.05: {(corr_comparison['train_val_diff'] > 0.05).sum()}")
    print(f"  Features with drift > 0.1: {(corr_comparison['train_val_diff'] > 0.1).sum()}")
    
    # Save results
    output_dir = Path('experiments')
    output_dir.mkdir(exist_ok=True)
    
    # Save correlation comparison
    corr_comparison.to_csv(output_dir / 'feature_correlation_comparison.csv', index=False)
    print(f"\n✅ Saved correlation comparison to experiments/feature_correlation_comparison.csv")
    
    # Save all results
    all_results.to_csv(output_dir / 'feature_importance_and_correlation.csv', index=False)
    print(f"✅ Saved full results to experiments/feature_importance_and_correlation.csv")
    
    # Save suspicious features
    with open(output_dir / 'suspicious_features_summary.txt', 'w') as f:
        f.write("FEATURES WITH HIGH CORRELATION DRIFT\n")
        f.write("="*80 + "\n\n")
        for sus in suspicious_features:
            f.write(f"{sus['model']} Model:\n")
            f.write("-"*80 + "\n")
            if len(sus['high_drift']) > 0:
                f.write(sus['high_drift'][['feature', 'importance', 'train_corr', 'val_corr', 'train_val_diff']].to_string(index=False))
                f.write("\n\n")
            else:
                f.write("No high-drift features found\n\n")
    
    print(f"✅ Saved suspicious features to experiments/suspicious_features_summary.txt")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()

