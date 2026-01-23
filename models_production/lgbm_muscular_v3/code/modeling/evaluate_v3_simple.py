#!/usr/bin/env python3
"""
Simple evaluation script for V3 model on test set.
"""

import json
import sys
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


def main():
    """Evaluate V3 on test set."""
    print("=" * 80)
    print("EVALUATING V3 MODEL ON TEST SET")
    print("=" * 80)

    # Paths
    v3_root = PROJECT_ROOT / "models_production" / "lgbm_muscular_v3"
    model_path = v3_root / "model" / "model.joblib"
    columns_path = v3_root / "model" / "columns.json"
    test_path = v3_root / "data" / "timelines" / "test" / "timelines_35day_season_2025_2026_v4_muscular.csv"

    # Load model
    print(f"\n📂 Loading V3 model...")
    model = joblib.load(model_path)
    with open(columns_path, "r", encoding="utf-8") as f:
        model_columns = json.load(f)
    print(f"✅ Model loaded with {len(model_columns)} features")

    # Load test data
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
    
    # Get common features
    common_features = list(set(X_test.columns) & set(model_columns))
    missing_cols = set(model_columns) - set(X_test.columns)
    extra_cols = set(X_test.columns) - set(model_columns)
    
    print(f"   Common features: {len(common_features)}")
    print(f"   Missing in test: {len(missing_cols)}")
    print(f"   Extra in test: {len(extra_cols)}")
    
    # Create aligned dataframe with model columns
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
    print("\n📊 Evaluating V3 model on test set...")
    test_metrics = evaluate_model(model, X_test_aligned, y_test, "V3 Test (2025-2026 PL-only)")

    # Save test metrics
    test_metrics_path = v3_root / "model" / "lgbm_v3_pl_only_metrics_test.json"
    with open(test_metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"\n✅ Saved V3 test metrics to {test_metrics_path}")

    return test_metrics


if __name__ == "__main__":
    main()

