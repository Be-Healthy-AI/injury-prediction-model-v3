#!/usr/bin/env python3
"""
Evaluate V3_natural_recent model on test set and compare with V3_natural (all seasons).
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


def load_model_and_columns(model_path, columns_path):
    """Load model and columns from files."""
    model = joblib.load(model_path)
    with open(columns_path, "r", encoding="utf-8") as f:
        columns = json.load(f)
    return model, columns


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


def load_training_metrics(model_dir, model_name):
    """Load training metrics for a model."""
    if model_name == "natural_recent":
        metrics_path = model_dir / "lgbm_v3_natural_recent_pl_only_metrics_train.json"
    else:
        metrics_path = model_dir / f"lgbm_v3_{model_name}_pl_only_metrics_train.json"
    
    if not metrics_path.exists():
        # Try alternative naming for original V3
        metrics_path = model_dir / "lgbm_v3_pl_only_metrics_train.json"
    
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def create_comparison_report(all_results):
    """Create a comprehensive comparison report."""
    v3_root = PROJECT_ROOT / "models_production" / "lgbm_muscular_v3"
    output_path = v3_root / "V3_RECENT_VS_ALL_COMPARISON.md"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# V3 Model Comparison: Recent Seasons vs All Seasons\n\n")
        f.write("## Overview\n\n")
        f.write("This document compares V3_natural models trained with different season ranges:\n\n")
        f.write("- **V3_natural (all)**: Trained on all seasons (2011-2026), natural ratio, PL-only\n")
        f.write("- **V3_natural_recent**: Trained on recent seasons only (2018-2026), natural ratio, PL-only\n\n")
        f.write("Both models are evaluated on the same test set: 2025-2026 PL-only timeline (natural ratio).\n\n")
        f.write("---\n\n")
        
        # Training Metrics Comparison
        f.write("## Training Metrics Comparison\n\n")
        f.write("| Metric | V3_natural (all seasons) | V3_natural_recent (2018-2026) |\n")
        f.write("|--------|-------------------------|-------------------------------|\n")
        
        # Accuracy
        acc_line = "| **Accuracy** | "
        for model_name in ["natural", "natural_recent"]:
            if model_name in all_results and all_results[model_name]["train"]:
                acc_line += f"{all_results[model_name]['train'].get('accuracy', 0):.4f} | "
            else:
                acc_line += "N/A | "
        f.write(acc_line + "\n")
        
        # Precision
        prec_line = "| **Precision** | "
        for model_name in ["natural", "natural_recent"]:
            if model_name in all_results and all_results[model_name]["train"]:
                prec_line += f"{all_results[model_name]['train'].get('precision', 0):.4f} | "
            else:
                prec_line += "N/A | "
        f.write(prec_line + "\n")
        
        # Recall
        recall_line = "| **Recall** | "
        for model_name in ["natural", "natural_recent"]:
            if model_name in all_results and all_results[model_name]["train"]:
                recall_line += f"{all_results[model_name]['train'].get('recall', 0):.4f} | "
            else:
                recall_line += "N/A | "
        f.write(recall_line + "\n")
        
        # F1-Score
        f1_line = "| **F1-Score** | "
        for model_name in ["natural", "natural_recent"]:
            if model_name in all_results and all_results[model_name]["train"]:
                f1_line += f"{all_results[model_name]['train'].get('f1', 0):.4f} | "
            else:
                f1_line += "N/A | "
        f.write(f1_line + "\n")
        
        # ROC AUC
        auc_line = "| **ROC AUC** | "
        for model_name in ["natural", "natural_recent"]:
            if model_name in all_results and all_results[model_name]["train"]:
                auc_line += f"{all_results[model_name]['train'].get('roc_auc', 0):.4f} | "
            else:
                auc_line += "N/A | "
        f.write(auc_line + "\n")
        
        # Gini
        gini_line = "| **Gini** | "
        for model_name in ["natural", "natural_recent"]:
            if model_name in all_results and all_results[model_name]["train"]:
                gini_line += f"{all_results[model_name]['train'].get('gini', 0):.4f} | "
            else:
                gini_line += "N/A | "
        f.write(gini_line + "\n")
        
        f.write("\n---\n\n")
        
        # Test Metrics Comparison
        f.write("## Test Metrics Comparison (2025-2026 PL-only)\n\n")
        f.write("| Metric | V3_natural (all seasons) | V3_natural_recent (2018-2026) |\n")
        f.write("|--------|-------------------------|-------------------------------|\n")
        
        # Accuracy
        test_acc_line = "| **Accuracy** | "
        for model_name in ["natural", "natural_recent"]:
            if model_name in all_results and all_results[model_name]["test"]:
                test_acc_line += f"{all_results[model_name]['test'].get('accuracy', 0):.4f} | "
            else:
                test_acc_line += "N/A | "
        f.write(test_acc_line + "\n")
        
        # Precision
        test_prec_line = "| **Precision** | "
        for model_name in ["natural", "natural_recent"]:
            if model_name in all_results and all_results[model_name]["test"]:
                test_prec_line += f"{all_results[model_name]['test'].get('precision', 0):.4f} | "
            else:
                test_prec_line += "N/A | "
        f.write(test_prec_line + "\n")
        
        # Recall
        test_recall_line = "| **Recall** | "
        for model_name in ["natural", "natural_recent"]:
            if model_name in all_results and all_results[model_name]["test"]:
                test_recall_line += f"{all_results[model_name]['test'].get('recall', 0):.4f} | "
            else:
                test_recall_line += "N/A | "
        f.write(test_recall_line + "\n")
        
        # F1-Score
        test_f1_line = "| **F1-Score** | "
        for model_name in ["natural", "natural_recent"]:
            if model_name in all_results and all_results[model_name]["test"]:
                test_f1_line += f"{all_results[model_name]['test'].get('f1', 0):.4f} | "
            else:
                test_f1_line += "N/A | "
        f.write(test_f1_line + "\n")
        
        # ROC AUC
        test_auc_line = "| **ROC AUC** | "
        for model_name in ["natural", "natural_recent"]:
            if model_name in all_results and all_results[model_name]["test"]:
                test_auc_line += f"{all_results[model_name]['test'].get('roc_auc', 0):.4f} | "
            else:
                test_auc_line += "N/A | "
        f.write(test_auc_line + "\n")
        
        # Gini
        test_gini_line = "| **Gini** | "
        for model_name in ["natural", "natural_recent"]:
            if model_name in all_results and all_results[model_name]["test"]:
                test_gini_line += f"{all_results[model_name]['test'].get('gini', 0):.4f} | "
            else:
                test_gini_line += "N/A | "
        f.write(test_gini_line + "\n")
        
        f.write("\n---\n\n")
        
        # Confusion Matrix Details
        f.write("## Confusion Matrix Details (Test Set)\n\n")
        f.write("| Model | TP | FP | TN | FN |\n")
        f.write("|-------|----|----|----|----|\n")
        
        for model_name, model_label in [("natural", "V3_natural (all)"), ("natural_recent", "V3_natural_recent")]:
            if model_name in all_results and all_results[model_name]["test"]:
                cm = all_results[model_name]["test"].get("confusion_matrix", {})
                f.write(f"| {model_label} | {cm.get('tp', 0)} | {cm.get('fp', 0)} | {cm.get('tn', 0)} | {cm.get('fn', 0)} |\n")
        
        f.write("\n---\n\n")
        
        # Key Observations
        f.write("## Key Observations\n\n")
        f.write("### Training Performance\n")
        if "natural" in all_results and "natural_recent" in all_results:
            natural_train = all_results["natural"]["train"]
            recent_train = all_results["natural_recent"]["train"]
            if natural_train and recent_train:
                f.write(f"- **V3_natural (all)**: Precision = {natural_train.get('precision', 0):.2%}, Recall = {natural_train.get('recall', 0):.2%}\n")
                f.write(f"- **V3_natural_recent**: Precision = {recent_train.get('precision', 0):.2%}, Recall = {recent_train.get('recall', 0):.2%}\n")
                prec_diff = recent_train.get('precision', 0) - natural_train.get('precision', 0)
                if prec_diff > 0:
                    f.write(f"- **Precision Improvement**: {prec_diff:.2%} higher with recent seasons only\n")
                else:
                    f.write(f"- **Precision Change**: {prec_diff:.2%} (lower with recent seasons only)\n")
        
        f.write("\n### Test Performance\n")
        if "natural" in all_results and "natural_recent" in all_results:
            natural_test = all_results["natural"]["test"]
            recent_test = all_results["natural_recent"]["test"]
            if natural_test and recent_test:
                f.write(f"- **V3_natural (all)**: Precision = {natural_test.get('precision', 0):.2%}, Recall = {natural_test.get('recall', 0):.2%}, Gini = {natural_test.get('gini', 0):.2%}\n")
                f.write(f"- **V3_natural_recent**: Precision = {recent_test.get('precision', 0):.2%}, Recall = {recent_test.get('recall', 0):.2%}, Gini = {recent_test.get('gini', 0):.2%}\n")
                prec_diff = recent_test.get('precision', 0) - natural_test.get('precision', 0)
                gini_diff = recent_test.get('gini', 0) - natural_test.get('gini', 0)
                if prec_diff > 0:
                    f.write(f"- **Precision Improvement**: {prec_diff:.2%} higher with recent seasons only\n")
                else:
                    f.write(f"- **Precision Change**: {prec_diff:.2%} (lower with recent seasons only)\n")
                if gini_diff > 0:
                    f.write(f"- **Gini Improvement**: {gini_diff:.2%} higher with recent seasons only\n")
                elif gini_diff < 0:
                    f.write(f"- **Gini Change**: {gini_diff:.2%} (lower with recent seasons only)\n")
        
        f.write("\n### Recommendations\n")
        f.write("- If recent seasons model shows higher precision with similar Gini/Recall: Consider using recent seasons only\n")
        f.write("- If recent seasons model shows lower performance: Keep all seasons for better generalization\n")
        f.write("- Consider the trade-off between precision and dataset size\n")
    
    print(f"\n✅ Comparison report saved to {output_path}")


def main():
    """Main evaluation and comparison pipeline."""
    print("=" * 80)
    print("V3 NATURAL RECENT VS ALL SEASONS COMPARISON")
    print("=" * 80)
    
    start_time = datetime.now()
    
    v3_root = PROJECT_ROOT / "models_production" / "lgbm_muscular_v3"
    
    # Define models to evaluate
    models_to_evaluate = [
        ("natural", v3_root / "model_natural", "V3_natural (all seasons)"),
        ("natural_recent", v3_root / "model_natural_recent", "V3_natural_recent (2018-2026)"),
    ]
    
    all_results = {}
    
    # Evaluate each model
    for model_suffix, model_dir, model_name in models_to_evaluate:
        model_path = model_dir / "model.joblib"
        columns_path = model_dir / "columns.json"
        
        if not model_path.exists() or not columns_path.exists():
            print(f"\n⚠️  Skipping {model_name}: model files not found")
            continue
        
        # Load training metrics
        train_metrics = load_training_metrics(model_dir, model_suffix)
        
        # Evaluate on test set
        test_metrics = evaluate_model_on_test(model_path, columns_path, model_name)
        
        # Save test metrics
        if test_metrics:
            if model_suffix == "natural_recent":
                test_metrics_path = model_dir / "lgbm_v3_natural_recent_pl_only_metrics_test.json"
            else:
                test_metrics_path = model_dir / f"lgbm_v3_{model_suffix}_pl_only_metrics_test.json"
            with open(test_metrics_path, "w", encoding="utf-8") as f:
                json.dump(test_metrics, f, indent=2)
            print(f"\n✅ Saved test metrics to {test_metrics_path}")
        
        all_results[model_suffix] = {
            "train": train_metrics,
            "test": test_metrics
        }
    
    # Create comparison report
    print("\n" + "=" * 80)
    print("CREATING COMPARISON REPORT")
    print("=" * 80)
    create_comparison_report(all_results)
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    elapsed_time = datetime.now() - start_time
    print(f"⏱️  Total time: {elapsed_time}")
    
    print("\n📊 Summary:")
    for model_suffix, model_name in [("natural", "V3_natural (all)"), ("natural_recent", "V3_natural_recent")]:
        if model_suffix in all_results:
            train_metrics = all_results[model_suffix]["train"]
            test_metrics = all_results[model_suffix]["test"]
            if train_metrics:
                print(f"\n   {model_name} Training:")
                print(f"      Precision: {train_metrics.get('precision', 0):.2%}")
                print(f"      Recall: {train_metrics.get('recall', 0):.2%}")
                print(f"      Gini: {train_metrics.get('gini', 0):.2%}")
            if test_metrics:
                print(f"\n   {model_name} Test:")
                print(f"      Precision: {test_metrics.get('precision', 0):.2%}")
                print(f"      Recall: {test_metrics.get('recall', 0):.2%}")
                print(f"      Gini: {test_metrics.get('gini', 0):.2%}")
    
    print(f"\n📁 Full comparison report: models_production/lgbm_muscular_v3/V3_RECENT_VS_ALL_COMPARISON.md")


if __name__ == "__main__":
    main()

