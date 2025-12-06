#!/usr/bin/env python3
"""
Feature Selection for Phase 2 Models
Reduces feature count from 741 low-shift features to improve out-of-sample generalization
"""
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import (
    mutual_info_classif, SelectKBest, RFE, RFECV
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix
)
from pathlib import Path

def load_low_shift_features():
    """Load low-shift features from adversarial validation"""
    results_file = 'experiments/adversarial_validation_results.json'
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Adversarial validation results not found: {results_file}")
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return set(results['low_shift_features'])

def load_sample_weights():
    """Load sample weights from covariate shift correction"""
    weights_file = 'experiments/covariate_shift_weights.npy'
    if not os.path.exists(weights_file):
        raise FileNotFoundError(f"Sample weights not found: {weights_file}")
    
    weights = np.load(weights_file)
    return weights

def prepare_data(df, low_shift_features_set=None):
    """Prepare data with low-shift feature filtering (same as Phase 2)"""
    feature_columns = [
        col for col in df.columns
        if col not in ['player_id', 'reference_date', 'player_name', 'target']
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
    
    # Filter to low-shift features only (after encoding)
    if low_shift_features_set is not None:
        low_shift_cols = []
        for col in X_encoded.columns:
            base_feature = col.split('_week_')[0] if '_week_' in col else col
            for prefix in ['position_', 'current_club_', 'previous_club_', 'nationality1_', 'nationality2_', 
                          'current_club_country_', 'previous_club_country_', 'last_match_position_week_']:
                if col.startswith(prefix):
                    base_feature = col.replace(prefix, '')
                    break
            
            if col in low_shift_features_set or base_feature in low_shift_features_set:
                low_shift_cols.append(col)
            elif any(col.startswith(f"{feat}_") for feat in low_shift_features_set):
                low_shift_cols.append(col)
        
        if low_shift_cols:
            X_encoded = X_encoded[[col for col in X_encoded.columns if col in low_shift_cols]]
    
    return X_encoded, y

def tree_based_importance_selection(X_train, y_train, sample_weights, n_features=200):
    """Select features using tree-based importance (GB)"""
    print(f"   Computing GB feature importance (top {n_features})...")
    
    gb = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=10,
        learning_rate=0.1,
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.8,
        max_features='sqrt',
        random_state=42
    )
    
    gb.fit(X_train, y_train, sample_weight=sample_weights)
    importances = pd.Series(gb.feature_importances_, index=X_train.columns)
    importances = importances.sort_values(ascending=False)
    selected_features = importances.head(n_features).index.tolist()
    
    return selected_features

def mutual_info_selection(X_train, y_train, n_features=200):
    """Select features using mutual information"""
    print(f"   Computing mutual information (top {n_features})...")
    
    selector = SelectKBest(score_func=mutual_info_classif, k=min(n_features, X_train.shape[1]))
    selector.fit(X_train, y_train)
    selected_features = X_train.columns[selector.get_support()].tolist()
    
    return selected_features

def rfe_selection(X_train, y_train, sample_weights, n_features=200):
    """Select features using Recursive Feature Elimination"""
    print(f"   Applying RFE (top {n_features})...")
    
    estimator = GradientBoostingClassifier(
        n_estimators=50,
        max_depth=10,
        learning_rate=0.1,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )
    
    rfe = RFE(estimator=estimator, n_features_to_select=n_features, step=max(1, n_features//20))
    rfe.fit(X_train, y_train, sample_weight=sample_weights)
    selected_features = X_train.columns[rfe.get_support()].tolist()
    
    return selected_features

def train_and_evaluate(X_train, y_train, sample_weights, X_val_insample, y_val_insample, 
                       X_val_outsample, y_val_outsample, feature_set_name):
    """Train GB model and evaluate on all datasets"""
    
    # Train model (same as Phase 2)
    base_model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.8,
        max_features='sqrt',
        random_state=42
    )
    
    gb_model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
    gb_model.fit(X_train, y_train, sample_weight=sample_weights)
    
    # Evaluate
    def evaluate(X, y, dataset_name):
        y_pred = gb_model.predict(X)
        y_proba = gb_model.predict_proba(X)[:, 1]
        
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.0
        }
    
    train_metrics = evaluate(X_train, y_train, "Training")
    insample_metrics = evaluate(X_val_insample, y_val_insample, "In-Sample")
    outsample_metrics = evaluate(X_val_outsample, y_val_outsample, "Out-of-Sample")
    
    return {
        'train': train_metrics,
        'insample': insample_metrics,
        'outsample': outsample_metrics
    }

def main():
    print("="*80)
    print("FEATURE SELECTION FOR PHASE 2 MODELS")
    print("="*80)
    print("Goal: Reduce feature count to improve out-of-sample Precision & Recall")
    print("="*80)
    
    # Load low-shift features
    print("\n📂 Loading low-shift features...")
    low_shift_features_set = load_low_shift_features()
    print(f"✅ Loaded {len(low_shift_features_set)} low-shift features")
    
    # Load sample weights
    print("\n📂 Loading sample weights...")
    sample_weights = load_sample_weights()
    print(f"✅ Loaded sample weights")
    
    # Load data
    print("\n📂 Loading timeline data...")
    train_file = 'timelines_35day_enhanced_balanced_v4_train.csv'
    val_file = 'timelines_35day_enhanced_balanced_v4_val.csv'
    
    df_train_full = pd.read_csv(train_file, encoding='utf-8-sig')
    df_val_outsample = pd.read_csv(val_file, encoding='utf-8-sig')
    
    print(f"✅ Training data: {len(df_train_full):,} records")
    print(f"✅ Out-of-sample validation: {len(df_val_outsample):,} records")
    
    # Prepare data
    print("\n🔧 Preparing data...")
    X_train_full, y_train_full = prepare_data(df_train_full, low_shift_features_set)
    X_val_outsample, y_val_outsample = prepare_data(df_val_outsample, low_shift_features_set)
    
    print(f"✅ Training features: {X_train_full.shape[1]}")
    print(f"✅ Validation features: {X_val_outsample.shape[1]}")
    
    # Split training data
    X_train, X_val_insample, y_train, y_val_insample = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    sample_weights_train = sample_weights[:len(X_train)]
    
    print(f"✅ Training split: {len(X_train):,} records")
    print(f"✅ In-sample validation: {len(X_val_insample):,} records")
    
    # Feature counts to test
    feature_counts = [50, 100, 150, 200, 250, 300, 400, 500]
    
    # Methods to test
    methods = {
        'tree_importance': tree_based_importance_selection,
        'mutual_info': mutual_info_selection,
        'rfe': rfe_selection
    }
    
    results = {}
    
    print("\n" + "="*80)
    print("FEATURE SELECTION EXPERIMENTS")
    print("="*80)
    
    for method_name, method_func in methods.items():
        print(f"\n{'='*80}")
        print(f"METHOD: {method_name.upper()}")
        print(f"{'='*80}")
        
        results[method_name] = {}
        
        for n_features in feature_counts:
            if n_features > X_train.shape[1]:
                print(f"\n⏭️  Skipping {n_features} features (only {X_train.shape[1]} available)")
                continue
            
            print(f"\n📊 Testing {n_features} features...")
            
            try:
                # Select features
                if method_name == 'mutual_info':
                    selected_features = method_func(X_train, y_train, n_features)
                else:
                    selected_features = method_func(X_train, y_train, sample_weights_train, n_features)
                
                # Filter datasets
                X_train_sel = X_train[selected_features]
                X_val_insample_sel = X_val_insample[selected_features]
                X_val_outsample_sel = X_val_outsample[selected_features]
                
                # Train and evaluate
                metrics = train_and_evaluate(
                    X_train_sel, y_train, sample_weights_train,
                    X_val_insample_sel, y_val_insample,
                    X_val_outsample_sel, y_val_outsample,
                    f"{method_name}_{n_features}"
                )
                
                results[method_name][n_features] = {
                    'n_features': len(selected_features),
                    'features': selected_features,
                    'metrics': metrics
                }
                
                print(f"   ✅ Out-of-Sample: F1={metrics['outsample']['f1']:.4f}, "
                      f"Precision={metrics['outsample']['precision']:.4f}, "
                      f"Recall={metrics['outsample']['recall']:.4f}, "
                      f"AUC={metrics['outsample']['roc_auc']:.4f}")
                print(f"   📉 In-Sample AUC: {metrics['insample']['roc_auc']:.4f} "
                      f"(vs Training: {metrics['train']['roc_auc']:.4f})")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Find best configuration
    print("\n" + "="*80)
    print("BEST CONFIGURATIONS")
    print("="*80)
    
    best_by_f1 = None
    best_f1 = -1
    
    for method_name, method_results in results.items():
        for n_features, result in method_results.items():
            f1 = result['metrics']['outsample']['f1']
            if f1 > best_f1:
                best_f1 = f1
                best_by_f1 = {
                    'method': method_name,
                    'n_features': n_features,
                    'result': result
                }
    
    if best_by_f1:
        print(f"\n🏆 Best Out-of-Sample F1-Score:")
        print(f"   Method: {best_by_f1['method']}")
        print(f"   Features: {best_by_f1['n_features']}")
        print(f"   F1-Score: {best_by_f1['result']['metrics']['outsample']['f1']:.4f}")
        print(f"   Precision: {best_by_f1['result']['metrics']['outsample']['precision']:.4f}")
        print(f"   Recall: {best_by_f1['result']['metrics']['outsample']['recall']:.4f}")
        print(f"   ROC AUC: {best_by_f1['result']['metrics']['outsample']['roc_auc']:.4f}")
    
    # Save results
    print("\n💾 Saving results...")
    
    # Save full results (without feature lists to keep file size manageable)
    results_summary = {}
    for method_name, method_results in results.items():
        results_summary[method_name] = {}
        for n_features, result in method_results.items():
            results_summary[method_name][n_features] = {
                'n_features': result['n_features'],
                'metrics': result['metrics']
            }
    
    with open('experiments/phase2_feature_selection_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Save best feature set
    if best_by_f1:
        best_features = best_by_f1['result']['features']
        with open('experiments/phase2_best_features.json', 'w') as f:
            json.dump({
                'method': best_by_f1['method'],
                'n_features': best_by_f1['n_features'],
                'features': best_features,
                'metrics': best_by_f1['result']['metrics']
            }, f, indent=2)
        
        print(f"✅ Best feature set saved: {len(best_features)} features")
        print(f"   File: experiments/phase2_best_features.json")
    
    print(f"✅ Full results saved: experiments/phase2_feature_selection_results.json")
    
    # Create summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    
    summary_data = []
    for method_name, method_results in results.items():
        for n_features, result in method_results.items():
            m = result['metrics']
            summary_data.append({
                'Method': method_name,
                'N_Features': n_features,
                'Train_AUC': m['train']['roc_auc'],
                'InSample_AUC': m['insample']['roc_auc'],
                'OutSample_AUC': m['outsample']['roc_auc'],
                'OutSample_Precision': m['outsample']['precision'],
                'OutSample_Recall': m['outsample']['recall'],
                'OutSample_F1': m['outsample']['f1']
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('OutSample_F1', ascending=False)
    
    print("\nTop 10 Configurations by Out-of-Sample F1-Score:")
    print(summary_df.head(10).to_string(index=False))
    
    summary_df.to_csv('experiments/phase2_feature_selection_summary.csv', index=False)
    print(f"\n✅ Summary table saved: experiments/phase2_feature_selection_summary.csv")

if __name__ == '__main__':
    main()



