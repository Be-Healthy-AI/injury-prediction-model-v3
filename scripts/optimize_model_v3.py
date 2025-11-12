#!/usr/bin/env python3
"""
Model Optimization Script V3
- Hyperparameter tuning for Random Forest
- Feature selection to remove noisy/redundant features
- Goal: Improve F1-Score while maintaining threshold at 0.5
"""
import sys
import io
# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import joblib
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, make_scorer
)
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the timelines data"""
    print("📂 Loading V3 timelines data...")
    timeline_file = None
    possible_paths = [
        'timelines_35day_enhanced_balanced_v3.csv',
        '../timelines_35day_enhanced_balanced_v3.csv',
        'scripts/timelines_35day_enhanced_balanced_v3.csv',
        '../scripts/timelines_35day_enhanced_balanced_v3.csv'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            timeline_file = path
            break
    
    if timeline_file is None:
        print("❌ Error: Could not find timelines_35day_enhanced_balanced_v3.csv")
        return None, None, None
    
    df = pd.read_csv(timeline_file, encoding='utf-8-sig')
    print(f"✅ Loaded {len(df):,} records with {len(df.columns)} features from {timeline_file}")
    return df, timeline_file

def prepare_data(df):
    """Prepare and encode data"""
    print("\n🔧 Preparing data...")
    feature_columns = [col for col in df.columns if col not in ['player_id', 'reference_date', 'player_name', 'target']]
    X = df[feature_columns]
    y = df['target']
    
    print(f"📊 Features: {len(feature_columns)}")
    print(f"📊 Target distribution: {y.value_counts().to_dict()}")
    print(f"📊 Injury ratio: {y.mean():.1%}")
    
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
    
    print(f"📊 Final shape after encoding: {X_encoded.shape}")
    return X_encoded, y, feature_columns

def get_feature_importance_baseline(X_train, y_train):
    """Get baseline feature importance from a quick Random Forest"""
    print("\n📊 Computing baseline feature importance...")
    rf_baseline = RandomForestClassifier(
        n_estimators=50,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    rf_baseline.fit(X_train, y_train)
    
    # Get feature importance
    importances = rf_baseline.feature_importances_
    feature_names = X_train.columns.tolist()
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print(f"   Top 10 most important features:")
    for idx, row in importance_df.head(10).iterrows():
        print(f"     {row['feature']}: {row['importance']:.6f}")
    
    return importance_df

def select_features_by_importance(X_train, X_val, importance_df, percentile=10):
    """Select features above a certain importance percentile"""
    threshold = np.percentile(importance_df['importance'], percentile)
    selected_features = importance_df[importance_df['importance'] >= threshold]['feature'].tolist()
    
    print(f"\n📊 Feature selection (keeping top {100-percentile}%):")
    print(f"   Original features: {len(X_train.columns)}")
    print(f"   Selected features: {len(selected_features)}")
    print(f"   Removed: {len(X_train.columns) - len(selected_features)} features")
    
    X_train_selected = X_train[selected_features]
    X_val_selected = X_val[selected_features]
    
    return X_train_selected, X_val_selected, selected_features

def tune_hyperparameters(X_train, y_train, cv_folds=3, n_iter=50):
    """Perform hyperparameter tuning using RandomizedSearchCV"""
    print("\n🔧 Hyperparameter tuning...")
    print(f"   Using RandomizedSearchCV with {n_iter} iterations and {cv_folds}-fold CV")
    print(f"   This will test {n_iter} random combinations (much faster than exhaustive search)...")
    
    # Define parameter grid (smaller, focused search space)
    param_distributions = {
        'n_estimators': [100, 150, 200, 250, 300],
        'max_depth': [15, 18, 20, 22, 25],
        'min_samples_split': [5, 8, 10, 12, 15],
        'min_samples_leaf': [2, 3, 4, 5, 6],
        'class_weight': ['balanced', {0: 0.8, 1: 1.2}, {0: 0.75, 1: 1.25}, {0: 0.7, 1: 1.3}]
    }
    
    # Use F1-Score as the scoring metric
    f1_scorer = make_scorer(f1_score)
    
    # Create base model
    base_model = RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    )
    
    # RandomizedSearchCV with F1-Score optimization
    print("   Running RandomizedSearchCV...")
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,  # Test 50 random combinations instead of all
        scoring=f1_scorer,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    tune_start = datetime.now()
    random_search.fit(X_train, y_train)
    tune_time = datetime.now() - tune_start
    
    print(f"   ✅ Tuning completed in {tune_time}")
    print(f"\n📊 Best hyperparameters:")
    for param, value in random_search.best_params_.items():
        print(f"   {param}: {value}")
    print(f"   Best CV F1-Score: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_, random_search.best_params_, random_search.best_score_

def evaluate_model(model, X_train, X_val, y_train, y_val, model_name="Model"):
    """Evaluate model performance"""
    # Training set
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    train_auc = roc_auc_score(y_train, y_train_proba)
    
    # Validation set
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]
    
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, y_val_proba)
    
    print(f"\n{'='*70}")
    print(f"📊 {model_name} PERFORMANCE")
    print(f"{'='*70}")
    print(f"\n📈 Training Set:")
    print(f"   Accuracy: {train_accuracy:.4f}")
    print(f"   Precision: {train_precision:.4f}")
    print(f"   Recall: {train_recall:.4f}")
    print(f"   F1-Score: {train_f1:.4f}")
    print(f"   ROC AUC: {train_auc:.4f}")
    
    print(f"\n📈 Validation Set:")
    print(f"   Accuracy: {val_accuracy:.4f}")
    print(f"   Precision: {val_precision:.4f}")
    print(f"   Recall: {val_recall:.4f}")
    print(f"   F1-Score: {val_f1:.4f}")
    print(f"   ROC AUC: {val_auc:.4f}")
    
    # Gaps
    f1_gap = train_f1 - val_f1
    precision_gap = train_precision - val_precision
    auc_gap = train_auc - val_auc
    
    print(f"\n📉 Performance Gaps:")
    print(f"   F1-Score gap: {f1_gap:.4f}")
    print(f"   Precision gap: {precision_gap:.4f}")
    print(f"   AUC gap: {auc_gap:.4f}")
    
    return {
        'train_f1': train_f1,
        'val_f1': val_f1,
        'train_precision': train_precision,
        'val_precision': val_precision,
        'train_recall': train_recall,
        'val_recall': val_recall,
        'train_auc': train_auc,
        'val_auc': val_auc,
        'f1_gap': f1_gap,
        'precision_gap': precision_gap,
        'auc_gap': auc_gap
    }

def main():
    print("🚀 MODEL OPTIMIZATION: HYPERPARAMETER TUNING + FEATURE SELECTION")
    print("=" * 70)
    print("📋 Goal: Improve F1-Score through hyperparameter tuning and feature selection")
    print("=" * 70)
    
    start_time = datetime.now()
    
    # Load data
    df, timeline_file = load_data()
    if df is None:
        return
    
    # Prepare data
    X_encoded, y, original_feature_columns = prepare_data(df)
    
    # Split data (same as training script)
    print("\n" + "=" * 70)
    print("📊 DATA SPLITTING (80/20)")
    print("=" * 70)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_encoded, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print(f"✅ Training set: {len(X_train):,} samples")
    print(f"✅ Validation set: {len(X_val):,} samples")
    
    # ===== BASELINE MODEL (Current) =====
    print("\n" + "=" * 70)
    print("📊 BASELINE MODEL (Current Configuration)")
    print("=" * 70)
    
    baseline_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    
    baseline_model.fit(X_train, y_train)
    baseline_metrics = evaluate_model(baseline_model, X_train, X_val, y_train, y_val, "BASELINE")
    
    # ===== FEATURE SELECTION =====
    print("\n" + "=" * 70)
    print("🔍 FEATURE SELECTION")
    print("=" * 70)
    
    importance_df = get_feature_importance_baseline(X_train, y_train)
    
    # Test different feature selection thresholds
    feature_selection_results = []
    
    for percentile in [5, 10, 15, 20]:
        print(f"\n{'='*70}")
        print(f"Testing: Keep top {100-percentile}% of features")
        print(f"{'='*70}")
        
        X_train_sel, X_val_sel, selected_features = select_features_by_importance(
            X_train, X_val, importance_df, percentile=percentile
        )
        
        # Quick test with baseline hyperparameters
        test_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        test_model.fit(X_train_sel, y_train)
        y_val_pred = test_model.predict(X_val_sel)
        val_f1 = f1_score(y_val, y_val_pred)
        val_precision = precision_score(y_val, y_val_pred)
        val_recall = recall_score(y_val, y_val_pred)
        
        print(f"   Validation F1-Score: {val_f1:.4f}")
        print(f"   Validation Precision: {val_precision:.4f}")
        print(f"   Validation Recall: {val_recall:.4f}")
        
        feature_selection_results.append({
            'percentile': percentile,
            'n_features': len(selected_features),
            'val_f1': val_f1,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'selected_features': selected_features
        })
    
    # Select best feature subset (highest F1-Score)
    best_feature_selection = max(feature_selection_results, key=lambda x: x['val_f1'])
    print(f"\n✅ Best feature selection: Keep top {100-best_feature_selection['percentile']}%")
    print(f"   Features: {best_feature_selection['n_features']}")
    print(f"   Validation F1-Score: {best_feature_selection['val_f1']:.4f}")
    
    # Prepare selected features
    X_train_optimized = X_train[best_feature_selection['selected_features']]
    X_val_optimized = X_val[best_feature_selection['selected_features']]
    
    # ===== HYPERPARAMETER TUNING =====
    print("\n" + "=" * 70)
    print("🔧 HYPERPARAMETER TUNING (on selected features)")
    print("=" * 70)
    
    best_model, best_params, best_cv_f1 = tune_hyperparameters(X_train_optimized, y_train, cv_folds=3, n_iter=50)
    
    # ===== EVALUATE OPTIMIZED MODEL =====
    optimized_metrics = evaluate_model(
        best_model, X_train_optimized, X_val_optimized, y_train, y_val, "OPTIMIZED"
    )
    
    # ===== COMPARISON =====
    print("\n" + "=" * 70)
    print("📊 COMPARISON: BASELINE vs OPTIMIZED")
    print("=" * 70)
    
    print(f"\n📈 Validation F1-Score:")
    print(f"   Baseline: {baseline_metrics['val_f1']:.4f}")
    print(f"   Optimized: {optimized_metrics['val_f1']:.4f}")
    print(f"   Improvement: {optimized_metrics['val_f1'] - baseline_metrics['val_f1']:.4f} ({((optimized_metrics['val_f1'] - baseline_metrics['val_f1']) / baseline_metrics['val_f1'] * 100):.2f}%)")
    
    print(f"\n📈 Validation Precision:")
    print(f"   Baseline: {baseline_metrics['val_precision']:.4f}")
    print(f"   Optimized: {optimized_metrics['val_precision']:.4f}")
    print(f"   Improvement: {optimized_metrics['val_precision'] - baseline_metrics['val_precision']:.4f} ({((optimized_metrics['val_precision'] - baseline_metrics['val_precision']) / baseline_metrics['val_precision'] * 100):.2f}%)")
    
    print(f"\n📈 Validation Recall:")
    print(f"   Baseline: {baseline_metrics['val_recall']:.4f}")
    print(f"   Optimized: {optimized_metrics['val_recall']:.4f}")
    print(f"   Change: {optimized_metrics['val_recall'] - baseline_metrics['val_recall']:.4f}")
    
    # ===== TRAIN FINAL MODEL ON 100% OF DATA =====
    if optimized_metrics['val_f1'] > baseline_metrics['val_f1']:
        print("\n" + "=" * 70)
        print("🚀 TRAINING FINAL OPTIMIZED MODEL (100% of data)")
        print("=" * 70)
        
        # Prepare full dataset with selected features
        X_optimized_full = X_encoded[best_feature_selection['selected_features']]
        
        # Train final model with best hyperparameters
        final_model = RandomForestClassifier(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            min_samples_leaf=best_params['min_samples_leaf'],
            class_weight=best_params['class_weight'],
            random_state=42,
            n_jobs=-1
        )
        
        final_start = datetime.now()
        final_model.fit(X_optimized_full, y)
        final_time = datetime.now() - final_start
        
        print(f"   ✅ Training completed in {final_time}")
        
        # Save optimized model
        os.makedirs('models', exist_ok=True)
        
        model_file = 'models/model_v3_random_forest_optimized.pkl'
        columns_file = 'models/model_v3_rf_optimized_training_columns.json'
        params_file = 'models/model_v3_rf_optimized_params.json'
        
        joblib.dump(final_model, model_file)
        json.dump(best_feature_selection['selected_features'], open(columns_file, 'w'))
        json.dump(best_params, open(params_file, 'w'))
        
        print(f"✅ Saved optimized model to {model_file}")
        print(f"✅ Saved feature columns to {columns_file}")
        print(f"✅ Saved hyperparameters to {params_file}")
        
        # Quick check
        y_full_proba = final_model.predict_proba(X_optimized_full)[:, 1]
        final_auc = roc_auc_score(y, y_full_proba)
        final_gini = 2 * final_auc - 1
        
        print(f"\n📊 Final model performance (100% data):")
        print(f"   ROC AUC: {final_auc:.4f}")
        print(f"   Gini: {final_gini:.4f}")
    else:
        print("\n⚠️  Optimized model did not improve F1-Score. Keeping baseline model.")
    
    # ===== FINAL SUMMARY =====
    total_time = datetime.now() - start_time
    print("\n" + "=" * 70)
    print("📊 OPTIMIZATION SUMMARY")
    print("=" * 70)
    print(f"⏱️  Total time: {total_time}")
    print(f"\n📈 Results:")
    print(f"   Baseline F1-Score: {baseline_metrics['val_f1']:.4f}")
    print(f"   Optimized F1-Score: {optimized_metrics['val_f1']:.4f}")
    print(f"   Improvement: {optimized_metrics['val_f1'] - baseline_metrics['val_f1']:.4f}")
    print(f"\n📈 Precision Improvement:")
    print(f"   Baseline: {baseline_metrics['val_precision']:.4f}")
    print(f"   Optimized: {optimized_metrics['val_precision']:.4f}")
    print(f"   Improvement: {optimized_metrics['val_precision'] - baseline_metrics['val_precision']:.4f}")
    
    print("\n🎉 OPTIMIZATION COMPLETED!")

if __name__ == "__main__":
    main()

