#!/usr/bin/env python3
"""
Evaluate V3_25pc, V3_50pc, and V3_natural models on test set.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_models_seasonal_combined import (
    evaluate_model,
    prepare_data,
    sanitize_feature_name,
)


def evaluate_model_on_test(model_path, columns_path, model_name):
    """Evaluate a single model on the test set."""
    print(f"\n{'='*80}")
    print(f"EVALUATING {model_name}")
    print(f"{'='*80}")
    
    # Load model
    print(f"\n📂 Loading {model_name} model...")
    model = joblib.load(model_path)
    with open(columns_path, "r", encoding="utf-8") as f:
        model_columns = json.load(f)
    print(f"✅ Model loaded with {len(model_columns)} features")
    
    # Load test data
    v3_root = PROJECT_ROOT / "models_production" / "lgbm_muscular_v3"
    test_path = v3_root / "data" / "timelines" / "test" / "timelines_35day_season_2025_2026_v4_muscular.csv"
    
    print(f"\n📂 Loading test dataset...")
    df_test = pd.read_csv(test_path, encoding="utf-8-sig", low_memory=False)
    print(f"✅ Loaded test dataset: {len(df_test):,} records")
    print(f"   Injury ratio: {df_test['target'].mean():.2%}")
    
    # Prepare test data
    print("\n📊 Preparing test data...")
    X_test, y_test = prepare_data(df_test)
    
    # Sanitize feature names
    X_test.columns = [sanitize_feature_name(col) for col in X_test.columns]
    
    # Align features with model
    print("\n🔧 Aligning test features with model features...")
    
    common_features = list(set(X_test.columns) & set(model_columns))
    missing_cols = set(model_columns) - set(X_test.columns)
    
    print(f"   Common features: {len(common_features)}")
    print(f"   Missing in test: {len(missing_cols)}")
    
    # Create aligned dataframe
    X_test_aligned = pd.DataFrame(index=X_test.index)
    
    # Add common features
    for col in common_features:
        X_test_aligned[col] = X_test[col]
    
    # Add missing columns (filled with 0)
    if missing_cols:
        print(f"   ⚠️  Adding {len(missing_cols)} missing columns (filled with 0)")
        for col in missing_cols:
            X_test_aligned[col] = 0
    
    # Reorder columns to match model exactly
    X_test_aligned = X_test_aligned[model_columns]
    
    # Evaluate
    print(f"\n📊 Evaluating {model_name} on test set...")
    test_metrics = evaluate_model(model, X_test_aligned, y_test, f"{model_name} Test")
    
    return test_metrics


def main():
    """Main evaluation pipeline."""
    print("=" * 80)
    print("EVALUATING V3_25pc, V3_50pc, and V3_natural ON TEST SET (2025-2026 PL-only)")
    print("=" * 80)
    
    start_time = datetime.now()
    
    v3_root = PROJECT_ROOT / "models_production" / "lgbm_muscular_v3"
    
    # Define models to evaluate
    models_to_evaluate = [
        ("25pc", v3_root / "model_25pc", "V3_25pc"),
        ("50pc", v3_root / "model_50pc", "V3_50pc"),
        ("natural", v3_root / "model_natural", "V3_natural"),
    ]
    
    all_results = {}
    
    # Evaluate each model
    for model_suffix, model_dir, model_name in models_to_evaluate:
        model_path = model_dir / "model.joblib"
        columns_path = model_dir / "columns.json"
        
        if not model_path.exists() or not columns_path.exists():
            print(f"\n⚠️  Skipping {model_name}: model files not found")
            continue
        
        # Evaluate on test set
        test_metrics = evaluate_model_on_test(model_path, columns_path, model_name)
        
        # Save test metrics
        if test_metrics:
            test_metrics_path = model_dir / f"lgbm_v3_{model_suffix}_pl_only_metrics_test.json"
            with open(test_metrics_path, "w", encoding="utf-8") as f:
                json.dump(test_metrics, f, indent=2)
            print(f"\n✅ Saved test metrics to {test_metrics_path}")
        
        all_results[model_suffix] = test_metrics
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    elapsed_time = datetime.now() - start_time
    print(f"⏱️  Total time: {elapsed_time}")
    
    print("\n📊 Test Results Summary:")
    for model_suffix, model_name in [("25pc", "V3_25pc"), ("50pc", "V3_50pc"), ("natural", "V3_natural")]:
        if model_suffix in all_results and all_results[model_suffix]:
            test_metrics = all_results[model_suffix]
            print(f"\n   {model_name}:")
            print(f"      Accuracy:  {test_metrics.get('accuracy', 0):.4f} ({test_metrics.get('accuracy', 0):.2%})")
            print(f"      Precision: {test_metrics.get('precision', 0):.4f} ({test_metrics.get('precision', 0):.2%})")
            print(f"      Recall:    {test_metrics.get('recall', 0):.4f} ({test_metrics.get('recall', 0):.2%})")
            print(f"      F1-Score:  {test_metrics.get('f1', 0):.4f} ({test_metrics.get('f1', 0):.2%})")
            print(f"      ROC AUC:   {test_metrics.get('roc_auc', 0):.4f} ({test_metrics.get('roc_auc', 0):.2%})")
            print(f"      Gini:      {test_metrics.get('gini', 0):.4f} ({test_metrics.get('gini', 0):.2%})")
            cm = test_metrics.get('confusion_matrix', {})
            print(f"      Confusion Matrix: TP={cm.get('tp', 0)}, FP={cm.get('fp', 0)}, TN={cm.get('tn', 0)}, FN={cm.get('fn', 0)}")


if __name__ == "__main__":
    main()

