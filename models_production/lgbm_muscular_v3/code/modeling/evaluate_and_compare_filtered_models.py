#!/usr/bin/env python3
"""
Evaluate and compare V3-natural-filtered and V3-natural-filtered-excl-2023-2024 models.
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

V3_ROOT = PROJECT_ROOT / "models_production" / "lgbm_muscular_v3"


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
    test_path = V3_ROOT / "data" / "timelines" / "test" / "timelines_35day_season_2025_2026_v4_muscular.csv"
    
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


def load_training_metrics(model_dir, model_suffix):
    """Load training metrics for a model."""
    metrics_file = f"lgbm_v3_{model_suffix}_pl_only_metrics_train.json"
    metrics_path = model_dir / metrics_file
    
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    raise FileNotFoundError(f"Training metrics not found: {metrics_path}")


def save_test_metrics(model_dir, model_suffix, test_metrics):
    """Save test metrics to file."""
    metrics_file = f"lgbm_v3_{model_suffix}_pl_only_metrics_test.json"
    metrics_path = model_dir / metrics_file
    
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    
    print(f"✅ Saved test metrics to {metrics_path}")


def create_comparison_report(results):
    """Create a comparison report."""
    report_path = V3_ROOT / "V3_FILTERED_MODELS_COMPARISON.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# V3 Filtered Models Comparison Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("This report compares two V3 filtered models:\n\n")
        f.write("1. **V3-natural-filtered**: Excludes 2021-2022 and 2022-2023 seasons\n")
        f.write("2. **V3-natural-filtered-excl-2023-2024**: Excludes 2021-2022, 2022-2023, and 2023-2024 seasons\n\n")
        f.write("---\n\n")
        
        # Training Metrics
        f.write("## Training Metrics Comparison\n\n")
        f.write("| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC | Gini |\n")
        f.write("|-------|----------|-----------|--------|----------|---------|------|\n")
        
        for model_name, model_data in sorted(results.items()):
            train = model_data["train"]
            f.write(f"| {model_name} | {train['accuracy']:.4f} | {train['precision']:.4f} | "
                   f"{train['recall']:.4f} | {train['f1']:.4f} | {train['roc_auc']:.4f} | "
                   f"{train['gini']:.4f} |\n")
        
        f.write("\n---\n\n")
        
        # Test Metrics
        f.write("## Test Metrics Comparison (2025-2026 PL-only)\n\n")
        f.write("| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC | Gini |\n")
        f.write("|-------|----------|-----------|--------|----------|---------|------|\n")
        
        for model_name, model_data in sorted(results.items()):
            test = model_data["test"]
            f.write(f"| {model_name} | {test['accuracy']:.4f} | {test['precision']:.4f} | "
                   f"{test['recall']:.4f} | {test['f1']:.4f} | {test['roc_auc']:.4f} | "
                   f"{test['gini']:.4f} |\n")
        
        f.write("\n---\n\n")
        
        # Confusion Matrices
        f.write("## Test Set Confusion Matrices\n\n")
        for model_name, model_data in sorted(results.items()):
            test = model_data["test"]
            cm = test["confusion_matrix"]
            f.write(f"### {model_name}\n\n")
            f.write(f"| | Predicted Negative | Predicted Positive |\n")
            f.write(f"|---|-------------------|-------------------|\n")
            f.write(f"| **Actual Negative** | {cm['tn']:,} | {cm['fp']:,} |\n")
            f.write(f"| **Actual Positive** | {cm['fn']:,} | {cm['tp']:,} |\n")
            f.write("\n")
        
        f.write("---\n\n")
        
        # Summary
        f.write("## Summary\n\n")
        filtered = results.get("V3-natural-filtered", {})
        filtered_excl_2023 = results.get("V3-natural-filtered-excl-2023-2024", {})
        
        if filtered and filtered_excl_2023:
            f.write("### Key Differences\n\n")
            f.write(f"- **Training Records**: V3-natural-filtered: 476,921 | V3-natural-filtered-excl-2023-2024: 357,281\n")
            f.write(f"- **Training Positives**: V3-natural-filtered: 2,739 | V3-natural-filtered-excl-2023-2024: 2,219\n")
            
            f.write("\n### Performance Comparison\n\n")
            prec_diff = filtered_excl_2023["test"]["precision"] - filtered["test"]["precision"]
            f1_diff = filtered_excl_2023["test"]["f1"] - filtered["test"]["f1"]
            roc_diff = filtered_excl_2023["test"]["roc_auc"] - filtered["test"]["roc_auc"]
            
            f.write(f"- **Precision (Test)**: V3-natural-filtered-excl-2023-2024 is {prec_diff:+.2%} vs V3-natural-filtered\n")
            f.write(f"- **F1-Score (Test)**: V3-natural-filtered-excl-2023-2024 is {f1_diff:+.2%} vs V3-natural-filtered\n")
            f.write(f"- **ROC AUC (Test)**: V3-natural-filtered-excl-2023-2024 is {roc_diff:+.4f} vs V3-natural-filtered\n")
    
    print(f"\n✅ Comparison report saved to {report_path}")


def main():
    """Main evaluation function."""
    print("=" * 80)
    print("V3 FILTERED MODELS EVALUATION AND COMPARISON")
    print("=" * 80)
    print(f"\n⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    models_to_evaluate = [
        ("natural_filtered", V3_ROOT / "model_natural_filtered", "V3-natural-filtered"),
        ("natural_filtered_excl_2023_2024", V3_ROOT / "model_natural_filtered_excl_2023_2024", "V3-natural-filtered-excl-2023-2024"),
    ]
    
    results = {}
    
    for model_suffix, model_dir, model_name in models_to_evaluate:
        print(f"\n{'='*80}")
        print(f"Processing {model_name}")
        print(f"{'='*80}")
        
        model_path = model_dir / "model.joblib"
        columns_path = model_dir / "columns.json"
        
        if not model_path.exists():
            print(f"⚠️  Model not found: {model_path}")
            continue
        
        # Load training metrics
        try:
            train_metrics = load_training_metrics(model_dir, model_suffix)
            print(f"✅ Loaded training metrics for {model_name}")
        except FileNotFoundError as e:
            print(f"⚠️  {e}")
            continue
        
        # Evaluate on test set
        try:
            test_metrics = evaluate_model_on_test(model_path, columns_path, model_name)
            save_test_metrics(model_dir, model_suffix, test_metrics)
            
            results[model_name] = {
                "train": train_metrics,
                "test": test_metrics
            }
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate comparison report
    if results:
        create_comparison_report(results)
        
        # Print summary
        print(f"\n{'='*80}")
        print("COMPARISON SUMMARY")
        print(f"{'='*80}")
        print(f"\n{'Model':<40} {'Precision (Test)':<20} {'F1-Score (Test)':<20} {'ROC AUC (Test)':<20}")
        print("-" * 100)
        for model_name, model_data in sorted(results.items()):
            test = model_data["test"]
            print(f"{model_name:<40} {test['precision']:<20.4f} {test['f1']:<20.4f} {test['roc_auc']:<20.4f}")
    
    print(f"\n⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

