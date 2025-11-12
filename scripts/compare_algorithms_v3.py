#!/usr/bin/env python3
"""
Algorithm Comparison Script V3
- Compare Random Forest (optimized) with other algorithms
- Same train/validation split for fair comparison
- Comprehensive metrics comparison
"""
import sys
import io
# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# Try to import optional libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  LightGBM not available. Install with: pip install lightgbm")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️  CatBoost not available. Install with: pip install catboost")

def load_and_prepare_data():
    """Load and prepare data"""
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
        return None, None, None, None
    
    df = pd.read_csv(timeline_file, encoding='utf-8-sig')
    print(f"✅ Loaded {len(df):,} records with {len(df.columns)} features from {timeline_file}")
    
    # Prepare data
    print("\n🔧 Preparing data...")
    feature_columns = [col for col in df.columns if col not in ['player_id', 'reference_date', 'player_name', 'target']]
    X = df[feature_columns]
    y = df['target']
    
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
    
    # Split data (same random_state for consistency)
    X_train, X_val, y_train, y_val = train_test_split(
        X_encoded, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print(f"✅ Training set: {len(X_train):,} samples")
    print(f"✅ Validation set: {len(X_val):,} samples")
    
    return X_train, X_val, y_train, y_val, X_encoded, y

def train_and_evaluate_model(model, model_name, X_train, X_val, y_train, y_val):
    """Train and evaluate a model"""
    print(f"\n{'='*70}")
    print(f"🚀 Training {model_name}")
    print(f"{'='*70}")
    
    train_start = datetime.now()
    model.fit(X_train, y_train)
    train_time = datetime.now() - train_start
    
    print(f"✅ Training completed in {train_time}")
    
    # Training set predictions
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    train_auc = roc_auc_score(y_train, y_train_proba)
    
    # Validation set predictions
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]
    
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, y_val_proba)
    
    # Cross-validation
    print(f"   Running 5-fold cross-validation...")
    cv_start = datetime.now()
    cv_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_skf, 
                                 scoring='roc_auc', n_jobs=-1)
    cv_time = datetime.now() - cv_start
    
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    print(f"   ✅ CV completed in {cv_time}")
    
    # Confusion matrix
    cm = confusion_matrix(y_val, y_val_pred)
    
    results = {
        'model_name': model_name,
        'train_time': train_time,
        'cv_time': cv_time,
        'train_accuracy': train_accuracy,
        'train_precision': train_precision,
        'train_recall': train_recall,
        'train_f1': train_f1,
        'train_auc': train_auc,
        'val_accuracy': val_accuracy,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'val_f1': val_f1,
        'val_auc': val_auc,
        'cv_mean_auc': cv_mean,
        'cv_std_auc': cv_std,
        'true_negatives': cm[0, 0],
        'false_positives': cm[0, 1],
        'false_negatives': cm[1, 0],
        'true_positives': cm[1, 1]
    }
    
    print(f"\n📊 {model_name} Performance:")
    print(f"   Validation F1-Score: {val_f1:.4f}")
    print(f"   Validation Precision: {val_precision:.4f}")
    print(f"   Validation Recall: {val_recall:.4f}")
    print(f"   Validation ROC AUC: {val_auc:.4f}")
    print(f"   CV Mean AUC: {cv_mean:.4f} ± {cv_std:.4f}")
    
    return results, model

def main():
    print("🚀 ALGORITHM COMPARISON: MULTI-MODEL EVALUATION")
    print("=" * 70)
    print("📋 Comparing Random Forest with other algorithms")
    print("=" * 70)
    
    start_time = datetime.now()
    
    # Load and prepare data
    X_train, X_val, y_train, y_val, X_encoded, y = load_and_prepare_data()
    if X_train is None:
        return
    
    all_results = []
    all_models = {}
    
    # ===== 1. RANDOM FOREST (Optimized) =====
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=25,
        min_samples_split=8,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    results, model = train_and_evaluate_model(rf_model, "Random Forest (Optimized)", 
                                             X_train, X_val, y_train, y_val)
    all_results.append(results)
    all_models['Random Forest'] = model
    
    # ===== 2. GRADIENT BOOSTING =====
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        learning_rate=0.1,
        random_state=42,
        subsample=0.8
    )
    results, model = train_and_evaluate_model(gb_model, "Gradient Boosting", 
                                             X_train, X_val, y_train, y_val)
    all_results.append(results)
    all_models['Gradient Boosting'] = model
    
    # ===== 3. LOGISTIC REGRESSION =====
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1,
        solver='lbfgs'
    )
    results, model = train_and_evaluate_model(lr_model, "Logistic Regression", 
                                             X_train, X_val, y_train, y_val)
    all_results.append(results)
    all_models['Logistic Regression'] = model
    
    # ===== 4. XGBOOST (if available) =====
    if XGBOOST_AVAILABLE:
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            min_child_weight=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
            n_jobs=-1,
            eval_metric='logloss'
        )
        results, model = train_and_evaluate_model(xgb_model, "XGBoost", 
                                                 X_train, X_val, y_train, y_val)
        all_results.append(results)
        all_models['XGBoost'] = model
    
    # ===== 5. LIGHTGBM (if available) =====
    if LIGHTGBM_AVAILABLE:
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1,
            verbose=-1
        )
        results, model = train_and_evaluate_model(lgb_model, "LightGBM", 
                                                 X_train, X_val, y_train, y_val)
        all_results.append(results)
        all_models['LightGBM'] = model
    
    # ===== 6. CATBOOST (if available) =====
    if CATBOOST_AVAILABLE:
        cb_model = cb.CatBoostClassifier(
            iterations=200,
            depth=10,
            learning_rate=0.1,
            random_state=42,
            class_weights=[len(y_train[y_train==0]) / len(y_train[y_train==1])],
            verbose=False,
            thread_count=-1
        )
        results, model = train_and_evaluate_model(cb_model, "CatBoost", 
                                                 X_train, X_val, y_train, y_val)
        all_results.append(results)
        all_models['CatBoost'] = model
    
    # ===== COMPARISON TABLE =====
    print("\n" + "=" * 70)
    print("📊 COMPREHENSIVE COMPARISON")
    print("=" * 70)
    
    results_df = pd.DataFrame(all_results)
    
    # Sort by validation F1-Score
    results_df = results_df.sort_values('val_f1', ascending=False)
    
    print("\n📈 Validation Set Performance (sorted by F1-Score):")
    print("-" * 70)
    print(f"{'Algorithm':<25} {'F1-Score':<12} {'Precision':<12} {'Recall':<12} {'ROC AUC':<12} {'CV AUC':<15}")
    print("-" * 70)
    for _, row in results_df.iterrows():
        print(f"{row['model_name']:<25} {row['val_f1']:<12.4f} {row['val_precision']:<12.4f} "
              f"{row['val_recall']:<12.4f} {row['val_auc']:<12.4f} {row['cv_mean_auc']:.4f} ± {row['cv_std_auc']:.4f}")
    
    print("\n📊 Training Time Comparison:")
    print("-" * 70)
    print(f"{'Algorithm':<25} {'Train Time':<15} {'CV Time':<15}")
    print("-" * 70)
    for _, row in results_df.iterrows():
        print(f"{row['model_name']:<25} {str(row['train_time']):<15} {str(row['cv_time']):<15}")
    
    print("\n📊 Confusion Matrix (Validation Set):")
    print("-" * 70)
    print(f"{'Algorithm':<25} {'TN':<8} {'FP':<8} {'FN':<8} {'TP':<8}")
    print("-" * 70)
    for _, row in results_df.iterrows():
        print(f"{row['model_name']:<25} {row['true_negatives']:<8} {row['false_positives']:<8} "
              f"{row['false_negatives']:<8} {row['true_positives']:<8}")
    
    # ===== BEST MODEL =====
    best_model_name = results_df.iloc[0]['model_name']
    best_f1 = results_df.iloc[0]['val_f1']
    best_precision = results_df.iloc[0]['val_precision']
    best_recall = results_df.iloc[0]['val_recall']
    best_auc = results_df.iloc[0]['val_auc']
    
    print("\n" + "=" * 70)
    print("🏆 BEST MODEL")
    print("=" * 70)
    print(f"Algorithm: {best_model_name}")
    print(f"F1-Score: {best_f1:.4f}")
    print(f"Precision: {best_precision:.4f}")
    print(f"Recall: {best_recall:.4f}")
    print(f"ROC AUC: {best_auc:.4f}")
    
    # ===== COMPARISON WITH RANDOM FOREST =====
    rf_results = results_df[results_df['model_name'] == 'Random Forest (Optimized)'].iloc[0]
    print("\n" + "=" * 70)
    print("📊 COMPARISON WITH RANDOM FOREST (OPTIMIZED)")
    print("=" * 70)
    print(f"Random Forest F1-Score: {rf_results['val_f1']:.4f}")
    print(f"Random Forest Precision: {rf_results['val_precision']:.4f}")
    print(f"Random Forest Recall: {rf_results['val_recall']:.4f}")
    print(f"Random Forest ROC AUC: {rf_results['val_auc']:.4f}")
    
    print("\n📈 Models that outperform Random Forest:")
    better_models = results_df[results_df['val_f1'] > rf_results['val_f1']]
    if len(better_models) > 0:
        for _, row in better_models.iterrows():
            improvement = row['val_f1'] - rf_results['val_f1']
            print(f"   {row['model_name']}: F1-Score {row['val_f1']:.4f} (+{improvement:.4f})")
    else:
        print("   None - Random Forest is the best!")
    
    # ===== SUMMARY =====
    total_time = datetime.now() - start_time
    print("\n" + "=" * 70)
    print("📊 COMPARISON SUMMARY")
    print("=" * 70)
    print(f"⏱️  Total time: {total_time}")
    print(f"📊 Models tested: {len(all_results)}")
    print(f"🏆 Best model: {best_model_name} (F1-Score: {best_f1:.4f})")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/algorithm_comparison_v3.csv', index=False)
    print(f"\n✅ Results saved to results/algorithm_comparison_v3.csv")
    
    print("\n🎉 ALGORITHM COMPARISON COMPLETED!")

if __name__ == "__main__":
    main()

