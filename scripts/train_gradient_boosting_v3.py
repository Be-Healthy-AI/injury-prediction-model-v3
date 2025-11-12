#!/usr/bin/env python3
"""
Enhanced Gradient Boosting Training Script V3 with Validation
- 80/20 train/validation split
- Comprehensive validation metrics
- Overfitting analysis
- Cross-validation for stability
- Uses optimized hyperparameters
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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)

def main():
    print("🚀 ENHANCED GRADIENT BOOSTING TRAINING WITH VALIDATION")
    print("=" * 70)
    print("📋 Features: 108 enhanced features with 35-day windows")
    print("📊 Validation: 80/20 train/validation split with overfitting checks")
    print("🔧 Using optimized hyperparameters")
    print("=" * 70)
    
    start_time = datetime.now()
    
    # Load the corrected timelines data
    print("\n📂 Loading V3 timelines data...")
    # Try multiple possible locations
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
        print("   Searched in:")
        for path in possible_paths:
            print(f"     - {path}")
        return
    
    df = pd.read_csv(timeline_file, encoding='utf-8-sig')
    print(f"✅ Loaded {len(df):,} records with {len(df.columns)} features from {timeline_file}")
    
    # Prepare data (same as training)
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
    
    # ===== 80/20 TRAIN/VALIDATION SPLIT =====
    print("\n" + "=" * 70)
    print("📊 DATA SPLITTING (80/20)")
    print("=" * 70)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_encoded, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y  # Preserve 15% injury ratio in both sets
    )
    
    print(f"✅ Training set: {len(X_train):,} samples ({len(X_train)/len(X_encoded)*100:.1f}%)")
    print(f"   Injury ratio: {y_train.mean():.1%}")
    print(f"✅ Validation set: {len(X_val):,} samples ({len(X_val)/len(X_encoded)*100:.1f}%)")
    print(f"   Injury ratio: {y_val.mean():.1%}")
    
    # ===== TRAIN MODEL =====
    print("\n" + "=" * 70)
    print("🚀 TRAINING GRADIENT BOOSTING MODEL")
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
    gb_model.fit(X_train, y_train)
    train_time = datetime.now() - train_start
    print(f"✅ Training completed in {train_time}")
    
    # ===== EVALUATE ON TRAINING SET =====
    print("\n" + "=" * 70)
    print("📊 TRAINING SET PERFORMANCE")
    print("=" * 70)
    
    y_train_pred = gb_model.predict(X_train)
    y_train_proba = gb_model.predict_proba(X_train)[:, 1]
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    train_auc = roc_auc_score(y_train, y_train_proba)
    train_gini = 2 * train_auc - 1
    
    print(f"   Accuracy: {train_accuracy:.4f}")
    print(f"   Precision: {train_precision:.4f}")
    print(f"   Recall: {train_recall:.4f}")
    print(f"   F1-Score: {train_f1:.4f}")
    print(f"   ROC AUC: {train_auc:.4f}")
    print(f"   Gini Coefficient: {train_gini:.4f}")
    
    # ===== EVALUATE ON VALIDATION SET =====
    print("\n" + "=" * 70)
    print("📊 VALIDATION SET PERFORMANCE")
    print("=" * 70)
    
    y_val_pred = gb_model.predict(X_val)
    y_val_proba = gb_model.predict_proba(X_val)[:, 1]
    
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, y_val_proba)
    val_gini = 2 * val_auc - 1
    
    print(f"   Accuracy: {val_accuracy:.4f}")
    print(f"   Precision: {val_precision:.4f}")
    print(f"   Recall: {val_recall:.4f}")
    print(f"   F1-Score: {val_f1:.4f}")
    print(f"   ROC AUC: {val_auc:.4f}")
    print(f"   Gini Coefficient: {val_gini:.4f}")
    
    # ===== OVERFITTING ANALYSIS =====
    print("\n" + "=" * 70)
    print("🔍 OVERFITTING ANALYSIS")
    print("=" * 70)
    
    # Calculate gaps
    accuracy_gap = train_accuracy - val_accuracy
    precision_gap = train_precision - val_precision
    recall_gap = train_recall - val_recall
    f1_gap = train_f1 - val_f1
    auc_gap = train_auc - val_auc
    gini_gap = train_gini - val_gini
    
    print(f"\n📉 Performance Gaps (Train - Validation):")
    print(f"   Accuracy gap: {accuracy_gap:.4f}")
    print(f"   Precision gap: {precision_gap:.4f}")
    print(f"   Recall gap: {recall_gap:.4f}")
    print(f"   F1-Score gap: {f1_gap:.4f}")
    print(f"   ROC AUC gap: {auc_gap:.4f}")
    print(f"   Gini gap: {gini_gap:.4f}")
    
    # Assess overfitting risk
    if auc_gap < 0.01:
        risk_level = "LOW"
        risk_emoji = "✅"
    elif auc_gap < 0.05:
        risk_level = "MODERATE"
        risk_emoji = "⚠️"
    else:
        risk_level = "HIGH"
        risk_emoji = "❌"
    
    print(f"\n{risk_emoji} Overfitting Risk: {risk_level}")
    print(f"   Primary metric (AUC gap): {auc_gap:.4f}")
    
    # ===== CROSS-VALIDATION FOR STABILITY =====
    print("\n" + "=" * 70)
    print("🔄 CROSS-VALIDATION ANALYSIS (5-fold)")
    print("=" * 70)
    
    print("   Running 5-fold stratified cross-validation...")
    cv_start = datetime.now()
    cv_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(gb_model, X_train, y_train, cv=cv_skf, 
                                 scoring='roc_auc', n_jobs=-1)
    cv_time = datetime.now() - cv_start
    
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    cv_gini_mean = 2 * cv_mean - 1
    cv_gini_std = 2 * cv_std
    
    print(f"   ✅ Completed in {cv_time}")
    print(f"\n📊 Cross-Validation Results:")
    print(f"   Mean ROC AUC: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"   Mean Gini: {cv_gini_mean:.4f} ± {cv_gini_std:.4f}")
    print(f"   Stability (std): {cv_std:.4f} ({cv_std/cv_mean*100:.2f}%)")
    
    # Compare CV with validation
    cv_val_gap = cv_mean - val_auc
    print(f"\n📊 CV vs Validation:")
    print(f"   CV mean AUC: {cv_mean:.4f}")
    print(f"   Validation AUC: {val_auc:.4f}")
    print(f"   Gap: {cv_val_gap:.4f}")
    
    # ===== CONFUSION MATRIX =====
    print("\n" + "=" * 70)
    print("📊 CONFUSION MATRIX (Validation Set)")
    print("=" * 70)
    cm = confusion_matrix(y_val, y_val_pred)
    print(f"\n   Predicted:    0 (No Injury)    1 (Injury)")
    print(f"   Actual 0:     {cm[0,0]:>8}        {cm[0,1]:>8}")
    print(f"   Actual 1:     {cm[1,0]:>8}        {cm[1,1]:>8}")
    
    # ===== SAVE VALIDATION-TESTED MODEL =====
    print("\n" + "=" * 70)
    print("💾 SAVING MODELS")
    print("=" * 70)
    
    os.makedirs('models', exist_ok=True)
    
    # Save validation-tested model (trained on 80%)
    model_file = 'models/model_v3_gradient_boosting_validated.pkl'
    columns_file = 'models/model_v3_gb_validated_training_columns.json'
    
    joblib.dump(gb_model, model_file)
    json.dump(X_encoded.columns.tolist(), open(columns_file, 'w'))
    
    print(f"✅ Saved validation-tested model to {model_file}")
    print(f"✅ Saved training columns to {columns_file}")
    
    # ===== TRAIN FINAL MODEL ON 100% OF DATA =====
    print("\n" + "=" * 70)
    print("🚀 TRAINING FINAL MODEL (100% of data)")
    print("=" * 70)
    
    print("   Training on full dataset for production use...")
    final_model = GradientBoostingClassifier(
        n_estimators=250,
        max_depth=15,
        learning_rate=0.15,
        min_samples_split=15,
        min_samples_leaf=8,
        subsample=0.9,
        max_features='sqrt',
        random_state=42
    )
    
    final_start = datetime.now()
    final_model.fit(X_encoded, y)
    final_time = datetime.now() - final_start
    
    print(f"   ✅ Completed in {final_time}")
    
    # Quick check on full data
    y_full_pred = final_model.predict(X_encoded)
    y_full_proba = final_model.predict_proba(X_encoded)[:, 1]
    final_auc = roc_auc_score(y, y_full_proba)
    final_gini = 2 * final_auc - 1
    
    print(f"   Full dataset performance:")
    print(f"   ROC AUC: {final_auc:.4f}")
    print(f"   Gini: {final_gini:.4f}")
    
    # Save final model
    final_model_file = 'models/model_v3_gradient_boosting_100percent.pkl'
    final_columns_file = 'models/model_v3_gb_100percent_training_columns.json'
    
    joblib.dump(final_model, final_model_file)
    json.dump(X_encoded.columns.tolist(), open(final_columns_file, 'w'))
    
    print(f"✅ Saved final model (100% data) to {final_model_file}")
    print(f"✅ Saved training columns to {final_columns_file}")
    
    # ===== FINAL SUMMARY =====
    total_time = datetime.now() - start_time
    print("\n" + "=" * 70)
    print("📊 TRAINING SUMMARY")
    print("=" * 70)
    print(f"⏱️  Total time: {total_time}")
    print(f"\n📈 Validation Performance:")
    print(f"   ROC AUC: {val_auc:.4f}")
    print(f"   Gini: {val_gini:.4f}")
    print(f"   F1-Score: {val_f1:.4f}")
    print(f"   Precision: {val_precision:.4f}")
    print(f"   Recall: {val_recall:.4f}")
    print(f"\n🔍 Overfitting Assessment:")
    print(f"   AUC Gap: {auc_gap:.4f}")
    print(f"   Risk Level: {risk_level} {risk_emoji}")
    print(f"\n🔄 Cross-Validation:")
    print(f"   Mean AUC: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"   Stability: {cv_std/cv_mean*100:.2f}%")
    print(f"\n💾 Models saved:")
    print(f"   - Validation-tested: {model_file}")
    print(f"   - Final (100%): {final_model_file}")
    print("\n🎉 TRAINING COMPLETED!")

if __name__ == "__main__":
    main()

