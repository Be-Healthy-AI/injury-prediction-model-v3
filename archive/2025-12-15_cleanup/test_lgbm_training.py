#!/usr/bin/env python3
"""Test script to check if LightGBM training works"""
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Redirect output to file as well
log_file = open('test_lgbm_output.txt', 'w', encoding='utf-8')
original_stdout = sys.stdout
original_stderr = sys.stderr

class TeeOutput:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

sys.stdout = TeeOutput(sys.stdout, log_file)
sys.stderr = TeeOutput(sys.stderr, log_file)

print("Testing LightGBM import and basic functionality...")

try:
    # Test import
    print("✅ LightGBM imported successfully")
    
    # Load a small sample of data to test
    print("\nLoading test data...")
    train_file = 'timelines_35day_enhanced_balanced_v4_muscular_train.csv'
    df_train = pd.read_csv(train_file, encoding='utf-8-sig', nrows=1000)  # Just 1000 rows for testing
    
    # Prepare data (simplified)
    feature_columns = [col for col in df_train.columns if col not in ['player_id', 'reference_date', 'player_name', 'target']]
    X = df_train[feature_columns].select_dtypes(include=['int64', 'float64']).fillna(0)
    y = df_train['target']
    
    print(f"   Data shape: {X.shape}")
    print(f"   Target ratio: {y.mean():.1%}")
    
    # Test model creation
    print("\nCreating LightGBM model...")
    lgbm_model = LGBMClassifier(
        n_estimators=10,  # Very small for testing
        max_depth=5,
        learning_rate=0.1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    print("✅ Model created successfully")
    
    # Test training
    print("\nTesting training (10 trees only)...")
    lgbm_model.fit(X, y)
    print("✅ Training completed successfully")
    
    # Test prediction
    print("\nTesting prediction...")
    y_pred = lgbm_model.predict(X)
    y_proba = lgbm_model.predict_proba(X)[:, 1]
    print("✅ Prediction completed successfully")
    
    # Test metrics
    print("\nCalculating metrics...")
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    print(f"   Accuracy: {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall: {rec:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    
    print("\n✅✅✅ ALL TESTS PASSED - LightGBM is working correctly!")
    log_file.close()
    sys.exit(0)
    
except Exception as e:
    print(f"\n❌❌❌ ERROR OCCURRED:")
    print(f"   Type: {type(e).__name__}")
    print(f"   Message: {str(e)}")
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()
    log_file.close()
    sys.exit(1)

