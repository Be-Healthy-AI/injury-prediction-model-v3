#!/usr/bin/env python3
"""
Final Model Retraining Script V3
- Trains both Random Forest and Gradient Boosting on 100% of data
- Generates comprehensive final report
- Saves production-ready models
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix
)

def load_and_prepare_data():
    """Load and prepare the training data"""
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
    
    # Prepare data
    print("\n🔧 Preparing data...")
    feature_columns = [col for col in df.columns if col not in ['player_id', 'reference_date', 'player_name', 'target']]
    X = df[feature_columns]
    y = df['target']
    
    print(f"📊 Original features: {len(feature_columns)}")
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
    print(f"📊 Final features: {X_encoded.shape[1]}")
    
    return X_encoded, y, feature_columns

def train_random_forest(X, y):
    """Train Random Forest on 100% of data"""
    print("\n" + "=" * 70)
    print("🌳 TRAINING RANDOM FOREST (100% of data)")
    print("=" * 70)
    
    print("\n🔧 Model hyperparameters (optimized):")
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=25,
        min_samples_split=8,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    print(f"   n_estimators: {rf_model.n_estimators}")
    print(f"   max_depth: {rf_model.max_depth}")
    print(f"   min_samples_split: {rf_model.min_samples_split}")
    print(f"   min_samples_leaf: {rf_model.min_samples_leaf}")
    print(f"   class_weight: {rf_model.class_weight}")
    
    print("\n🌳 Training started...")
    train_start = datetime.now()
    rf_model.fit(X, y)
    train_time = datetime.now() - train_start
    print(f"✅ Training completed in {train_time}")
    
    # Evaluate on full dataset
    y_pred = rf_model.predict(X)
    y_proba = rf_model.predict_proba(X)[:, 1]
    
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)
    gini = 2 * auc - 1
    
    # Cross-validation for stability
    print("\n🔄 Running 5-fold cross-validation...")
    cv_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf_model, X, y, cv=cv_skf, scoring='roc_auc', n_jobs=-1)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    print(f"   CV ROC AUC: {cv_mean:.4f} ± {cv_std:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    
    results = {
        'model': rf_model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'gini': gini,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'confusion_matrix': cm,
        'train_time': train_time
    }
    
    print(f"\n📊 Performance Metrics:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    print(f"   ROC AUC: {auc:.4f}")
    print(f"   Gini Coefficient: {gini:.4f}")
    print(f"\n📊 Confusion Matrix:")
    print(f"   Predicted:    0 (No Injury)    1 (Injury)")
    print(f"   Actual 0:     {cm[0,0]:>8}        {cm[0,1]:>8}")
    print(f"   Actual 1:     {cm[1,0]:>8}        {cm[1,1]:>8}")
    
    return results

def train_gradient_boosting(X, y):
    """Train Gradient Boosting on 100% of data"""
    print("\n" + "=" * 70)
    print("🚀 TRAINING GRADIENT BOOSTING (100% of data)")
    print("=" * 70)
    
    print("\n🔧 Model hyperparameters (optimized):")
    gb_model = GradientBoostingClassifier(
        n_estimators=250,
        max_depth=15,
        learning_rate=0.15,
        min_samples_split=15,
        min_samples_leaf=8,
        subsample=0.9,
        max_features='sqrt',
        random_state=42
    )
    print(f"   n_estimators: {gb_model.n_estimators}")
    print(f"   max_depth: {gb_model.max_depth}")
    print(f"   learning_rate: {gb_model.learning_rate}")
    print(f"   min_samples_split: {gb_model.min_samples_split}")
    print(f"   min_samples_leaf: {gb_model.min_samples_leaf}")
    print(f"   subsample: {gb_model.subsample}")
    print(f"   max_features: {gb_model.max_features}")
    
    print("\n🌳 Training started...")
    train_start = datetime.now()
    gb_model.fit(X, y)
    train_time = datetime.now() - train_start
    print(f"✅ Training completed in {train_time}")
    
    # Evaluate on full dataset
    y_pred = gb_model.predict(X)
    y_proba = gb_model.predict_proba(X)[:, 1]
    
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)
    gini = 2 * auc - 1
    
    # Cross-validation for stability
    print("\n🔄 Running 5-fold cross-validation...")
    cv_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(gb_model, X, y, cv=cv_skf, scoring='roc_auc', n_jobs=-1)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    print(f"   CV ROC AUC: {cv_mean:.4f} ± {cv_std:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    
    results = {
        'model': gb_model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'gini': gini,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'confusion_matrix': cm,
        'train_time': train_time
    }
    
    print(f"\n📊 Performance Metrics:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    print(f"   ROC AUC: {auc:.4f}")
    print(f"   Gini Coefficient: {gini:.4f}")
    print(f"\n📊 Confusion Matrix:")
    print(f"   Predicted:    0 (No Injury)    1 (Injury)")
    print(f"   Actual 0:     {cm[0,0]:>8}        {cm[0,1]:>8}")
    print(f"   Actual 1:     {cm[1,0]:>8}        {cm[1,1]:>8}")
    
    return results

def evaluate_ensemble(rf_model, gb_model, X, y):
    """Evaluate ensemble by averaging probabilities from both models"""
    print("\n" + "=" * 70)
    print("🎯 EVALUATING ENSEMBLE (50% RF + 50% GB)")
    print("=" * 70)
    
    # Get probabilities from both models
    rf_proba = rf_model.predict_proba(X)[:, 1]
    gb_proba = gb_model.predict_proba(X)[:, 1]
    
    # Average probabilities
    ensemble_proba = (rf_proba + gb_proba) / 2.0
    ensemble_pred = (ensemble_proba >= 0.5).astype(int)
    
    # Evaluate ensemble
    accuracy = accuracy_score(y, ensemble_pred)
    precision = precision_score(y, ensemble_pred)
    recall = recall_score(y, ensemble_pred)
    f1 = f1_score(y, ensemble_pred)
    auc = roc_auc_score(y, ensemble_proba)
    gini = 2 * auc - 1
    
    print(f"✅ Ensemble evaluation completed")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    print(f"   ROC AUC: {auc:.4f}")
    print(f"   Gini: {gini:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'gini': gini
    }

def save_models(rf_results, gb_results, X_encoded):
    """Save all trained models"""
    print("\n" + "=" * 70)
    print("💾 SAVING MODELS")
    print("=" * 70)
    
    os.makedirs('models', exist_ok=True)
    
    # Save Random Forest
    rf_file = 'models/model_v3_random_forest_final_100percent.pkl'
    rf_columns_file = 'models/model_v3_rf_final_columns.json'
    joblib.dump(rf_results['model'], rf_file)
    json.dump(X_encoded.columns.tolist(), open(rf_columns_file, 'w'))
    print(f"✅ Saved Random Forest to {rf_file}")
    print(f"✅ Saved columns to {rf_columns_file}")
    
    # Save Gradient Boosting
    gb_file = 'models/model_v3_gradient_boosting_final_100percent.pkl'
    gb_columns_file = 'models/model_v3_gb_final_columns.json'
    joblib.dump(gb_results['model'], gb_file)
    json.dump(X_encoded.columns.tolist(), open(gb_columns_file, 'w'))
    print(f"✅ Saved Gradient Boosting to {gb_file}")
    print(f"✅ Saved columns to {gb_columns_file}")
    
    print(f"\n💡 Note: Ensemble predictions can be generated by averaging probabilities from both models")

def generate_final_report(rf_results, gb_results, ensemble_results, n_samples, n_features):
    """Generate comprehensive final report"""
    print("\n" + "=" * 70)
    print("📊 FINAL MODEL REPORT")
    print("=" * 70)
    
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset': {
            'total_samples': n_samples,
            'total_features_original': n_features,
            'total_features_encoded': len(rf_results['model'].feature_importances_)
        },
        'random_forest': {
            'accuracy': float(rf_results['accuracy']),
            'precision': float(rf_results['precision']),
            'recall': float(rf_results['recall']),
            'f1_score': float(rf_results['f1']),
            'roc_auc': float(rf_results['auc']),
            'gini_coefficient': float(rf_results['gini']),
            'cv_auc_mean': float(rf_results['cv_mean']),
            'cv_auc_std': float(rf_results['cv_std']),
            'training_time': str(rf_results['train_time']),
            'confusion_matrix': rf_results['confusion_matrix'].tolist()
        },
        'gradient_boosting': {
            'accuracy': float(gb_results['accuracy']),
            'precision': float(gb_results['precision']),
            'recall': float(gb_results['recall']),
            'f1_score': float(gb_results['f1']),
            'roc_auc': float(gb_results['auc']),
            'gini_coefficient': float(gb_results['gini']),
            'cv_auc_mean': float(gb_results['cv_mean']),
            'cv_auc_std': float(gb_results['cv_std']),
            'training_time': str(gb_results['train_time']),
            'confusion_matrix': gb_results['confusion_matrix'].tolist()
        },
        'ensemble': {
            'accuracy': float(ensemble_results['accuracy']),
            'precision': float(ensemble_results['precision']),
            'recall': float(ensemble_results['recall']),
            'f1_score': float(ensemble_results['f1']),
            'roc_auc': float(ensemble_results['auc']),
            'gini_coefficient': float(ensemble_results['gini']),
            'method': 'Average of RF and GB probabilities (50% each)'
        }
    }
    
    # Save report to JSON
    report_file = 'models/final_model_report_v3.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✅ Saved report to {report_file}")
    
    # Print formatted report
    print("\n" + "=" * 70)
    print("📋 FINAL MODEL REPORT")
    print("=" * 70)
    print(f"\n📅 Generated: {report['timestamp']}")
    print(f"\n📊 DATASET INFORMATION:")
    print(f"   Total Samples: {report['dataset']['total_samples']:,}")
    print(f"   Original Features: {report['dataset']['total_features_original']}")
    print(f"   Encoded Features: {report['dataset']['total_features_encoded']}")
    
    print(f"\n🌳 RANDOM FOREST MODEL:")
    print(f"   Accuracy: {report['random_forest']['accuracy']:.4f}")
    print(f"   Precision: {report['random_forest']['precision']:.4f}")
    print(f"   Recall: {report['random_forest']['recall']:.4f}")
    print(f"   F1-Score: {report['random_forest']['f1_score']:.4f}")
    print(f"   ROC AUC: {report['random_forest']['roc_auc']:.4f}")
    print(f"   Gini Coefficient: {report['random_forest']['gini_coefficient']:.4f}")
    print(f"   CV AUC (5-fold): {report['random_forest']['cv_auc_mean']:.4f} ± {report['random_forest']['cv_auc_std']:.4f}")
    print(f"   Training Time: {report['random_forest']['training_time']}")
    
    print(f"\n🚀 GRADIENT BOOSTING MODEL:")
    print(f"   Accuracy: {report['gradient_boosting']['accuracy']:.4f}")
    print(f"   Precision: {report['gradient_boosting']['precision']:.4f}")
    print(f"   Recall: {report['gradient_boosting']['recall']:.4f}")
    print(f"   F1-Score: {report['gradient_boosting']['f1_score']:.4f}")
    print(f"   ROC AUC: {report['gradient_boosting']['roc_auc']:.4f}")
    print(f"   Gini Coefficient: {report['gradient_boosting']['gini_coefficient']:.4f}")
    print(f"   CV AUC (5-fold): {report['gradient_boosting']['cv_auc_mean']:.4f} ± {report['gradient_boosting']['cv_auc_std']:.4f}")
    print(f"   Training Time: {report['gradient_boosting']['training_time']}")
    
    print(f"\n🎯 ENSEMBLE MODEL (50% RF + 50% GB):")
    print(f"   Accuracy: {report['ensemble']['accuracy']:.4f}")
    print(f"   Precision: {report['ensemble']['precision']:.4f}")
    print(f"   Recall: {report['ensemble']['recall']:.4f}")
    print(f"   F1-Score: {report['ensemble']['f1_score']:.4f}")
    print(f"   ROC AUC: {report['ensemble']['roc_auc']:.4f}")
    print(f"   Gini Coefficient: {report['ensemble']['gini_coefficient']:.4f}")
    print(f"   Method: {report['ensemble']['method']}")
    
    print(f"\n📊 COMPARISON:")
    print(f"   Best F1-Score: {'Ensemble' if ensemble_results['f1'] > max(gb_results['f1'], rf_results['f1']) else ('Gradient Boosting' if gb_results['f1'] > rf_results['f1'] else 'Random Forest')} "
          f"({max(ensemble_results['f1'], gb_results['f1'], rf_results['f1']):.4f})")
    print(f"   Best ROC AUC: {'Ensemble' if ensemble_results['auc'] > max(gb_results['auc'], rf_results['auc']) else ('Gradient Boosting' if gb_results['auc'] > rf_results['auc'] else 'Random Forest')} "
          f"({max(ensemble_results['auc'], gb_results['auc'], rf_results['auc']):.4f})")
    print(f"   Best Precision: {'Ensemble' if ensemble_results['precision'] > max(gb_results['precision'], rf_results['precision']) else ('Gradient Boosting' if gb_results['precision'] > rf_results['precision'] else 'Random Forest')} "
          f"({max(ensemble_results['precision'], gb_results['precision'], rf_results['precision']):.4f})")
    
    print("\n" + "=" * 70)
    print("✅ FINAL REPORT GENERATED")
    print("=" * 70)
    
    return report

def main():
    print("🚀 FINAL MODEL RETRAINING - 100% DATA")
    print("=" * 70)
    print("📋 Training both Random Forest and Gradient Boosting")
    print("📊 Using 100% of available training data")
    print("=" * 70)
    
    start_time = datetime.now()
    
    # Load and prepare data
    X, y, original_features = load_and_prepare_data()
    if X is None:
        return
    
    n_samples = len(X)
    n_features_original = len(original_features)
    
    # Train Random Forest
    rf_results = train_random_forest(X, y)
    
    # Train Gradient Boosting
    gb_results = train_gradient_boosting(X, y)
    
    # Evaluate ensemble
    ensemble_results = evaluate_ensemble(rf_results['model'], gb_results['model'], X, y)
    
    # Save all models
    save_models(rf_results, gb_results, X)
    
    # Generate final report
    report = generate_final_report(rf_results, gb_results, ensemble_results, n_samples, n_features_original)
    
    total_time = datetime.now() - start_time
    print(f"\n⏱️  Total execution time: {total_time}")
    print("\n🎉 ALL MODELS TRAINED AND SAVED!")
    print("=" * 70)

if __name__ == "__main__":
    main()

