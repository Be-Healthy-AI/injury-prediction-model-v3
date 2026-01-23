#!/usr/bin/env python3
"""
Evaluate and compare three V3_natural models:
1. V3_natural (all seasons 2011-2026)
2. V3_natural_recent (recent seasons 2018-2026)
3. V3_natural_filtered (recent seasons 2018-2026 excluding 2021-2022 and 2022-2023)
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


def load_training_metrics(model_dir, model_name):
    """Load training metrics for a model."""
    metrics_files = [
        f"lgbm_v3_{model_name}_pl_only_metrics_train.json",
        "lgbm_v3_pl_only_metrics_train.json",
    ]
    
    for metrics_file in metrics_files:
        metrics_path = model_dir / metrics_file
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def load_test_metrics(model_dir, model_name):
    """Load test metrics for a model if they exist."""
    metrics_files = [
        f"lgbm_v3_{model_name}_pl_only_metrics_test.json",
        "lgbm_v3_pl_only_metrics_test.json",
    ]
    
    for metrics_file in metrics_files:
        metrics_path = model_dir / metrics_file
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def create_comparison_report(all_results):
    """Create a comprehensive comparison report."""
    v3_root = PROJECT_ROOT / "models_production" / "lgbm_muscular_v3"
    output_path = v3_root / "V3_THREE_MODELS_COMPARISON.md"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# V3 Model Comparison: Three Natural Ratio Models\n\n")
        f.write("## Overview\n\n")
        f.write("This document compares three V3_natural models trained with different season ranges:\n\n")
        f.write("1. **V3_natural (all)**: Trained on all seasons (2011-2026), natural ratio, PL-only\n")
        f.write("2. **V3_natural_recent**: Trained on recent seasons only (2018-2026), natural ratio, PL-only\n")
        f.write("3. **V3_natural_filtered**: Trained on recent seasons (2018-2026) excluding 2021-2022 and 2022-2023, natural ratio, PL-only\n\n")
        f.write("All models are evaluated on the same test set: 2025-2026 PL-only timeline (natural ratio).\n\n")
        f.write("---\n\n")
        
        # Training Metrics Comparison
        f.write("## Training Metrics Comparison\n\n")
        f.write("| Metric | V3_natural (all seasons) | V3_natural_recent (2018-2026) | V3_natural_filtered (2018-2026 excl. 2021-2022 & 2022-2023) |\n")
        f.write("|--------|-------------------------|-------------------------------|----------------------------------------------------------------|\n")
        
        metrics_to_compare = [
            ("accuracy", "Accuracy"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1", "F1-Score"),
            ("roc_auc", "ROC AUC"),
            ("gini", "Gini"),
        ]
        
        for metric_key, metric_label in metrics_to_compare:
            line = f"| **{metric_label}** | "
            for model_name in ["natural", "natural_recent", "natural_filtered"]:
                if model_name in all_results and all_results[model_name]["train"]:
                    value = all_results[model_name]["train"].get(metric_key, 0)
                    if metric_key in ["accuracy", "precision", "recall", "f1", "roc_auc", "gini"]:
                        line += f"{value:.4f} | "
                    else:
                        line += f"{value} | "
                else:
                    line += "N/A | "
            f.write(line + "\n")
        
        # Training Confusion Matrix
        f.write("\n### Training Confusion Matrix\n\n")
        f.write("| Model | TP | FP | TN | FN | False Positive Rate |\n")
        f.write("|-------|----|----|----|----|---------------------|\n")
        
        for model_name, model_label in [
            ("natural", "V3_natural (all)"),
            ("natural_recent", "V3_natural_recent"),
            ("natural_filtered", "V3_natural_filtered"),
        ]:
            if model_name in all_results and all_results[model_name]["train"]:
                cm = all_results[model_name]["train"].get("confusion_matrix", {})
                tp = cm.get("tp", 0)
                fp = cm.get("fp", 0)
                tn = cm.get("tn", 0)
                fn = cm.get("fn", 0)
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                f.write(f"| {model_label} | {tp} | {fp} | {tn} | {fn} | {fpr:.4f} |\n")
        
        f.write("\n---\n\n")
        
        # Test Metrics Comparison
        f.write("## Test Metrics Comparison (2025-2026 PL-only)\n\n")
        f.write("| Metric | V3_natural (all seasons) | V3_natural_recent (2018-2026) | V3_natural_filtered (2018-2026 excl. 2021-2022 & 2022-2023) |\n")
        f.write("|--------|-------------------------|-------------------------------|----------------------------------------------------------------|\n")
        
        for metric_key, metric_label in metrics_to_compare:
            line = f"| **{metric_label}** | "
            for model_name in ["natural", "natural_recent", "natural_filtered"]:
                if model_name in all_results and all_results[model_name]["test"]:
                    value = all_results[model_name]["test"].get(metric_key, 0)
                    if metric_key in ["accuracy", "precision", "recall", "f1", "roc_auc", "gini"]:
                        line += f"{value:.4f} | "
                    else:
                        line += f"{value} | "
                else:
                    line += "N/A | "
            f.write(line + "\n")
        
        # Test Confusion Matrix
        f.write("\n### Test Confusion Matrix\n\n")
        f.write("| Model | TP | FP | TN | FN | False Positive Rate |\n")
        f.write("|-------|----|----|----|----|---------------------|\n")
        
        for model_name, model_label in [
            ("natural", "V3_natural (all)"),
            ("natural_recent", "V3_natural_recent"),
            ("natural_filtered", "V3_natural_filtered"),
        ]:
            if model_name in all_results and all_results[model_name]["test"]:
                cm = all_results[model_name]["test"].get("confusion_matrix", {})
                tp = cm.get("tp", 0)
                fp = cm.get("fp", 0)
                tn = cm.get("tn", 0)
                fn = cm.get("fn", 0)
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                f.write(f"| {model_label} | {tp} | {fp} | {tn} | {fn} | {fpr:.4f} |\n")
        
        f.write("\n---\n\n")
        
        # Key Observations
        f.write("## Key Observations\n\n")
        
        # Training Performance
        f.write("### Training Performance\n\n")
        if all("train" in all_results[m] and all_results[m]["train"] for m in ["natural", "natural_recent", "natural_filtered"]):
            natural_train = all_results["natural"]["train"]
            recent_train = all_results["natural_recent"]["train"]
            filtered_train = all_results["natural_filtered"]["train"]
            
            f.write(f"- **V3_natural (all)**: Precision = {natural_train.get('precision', 0):.2%}, Recall = {natural_train.get('recall', 0):.2%}, Gini = {natural_train.get('gini', 0):.2%}\n")
            f.write(f"- **V3_natural_recent**: Precision = {recent_train.get('precision', 0):.2%}, Recall = {recent_train.get('recall', 0):.2%}, Gini = {recent_train.get('gini', 0):.2%}\n")
            f.write(f"- **V3_natural_filtered**: Precision = {filtered_train.get('precision', 0):.2%}, Recall = {filtered_train.get('recall', 0):.2%}, Gini = {filtered_train.get('gini', 0):.2%}\n\n")
            
            # Calculate improvements
            prec_improvement_recent = recent_train.get('precision', 0) - natural_train.get('precision', 0)
            prec_improvement_filtered = filtered_train.get('precision', 0) - natural_train.get('precision', 0)
            prec_improvement_filtered_vs_recent = filtered_train.get('precision', 0) - recent_train.get('precision', 0)
            
            f.write(f"- **Precision vs All**: Recent = +{prec_improvement_recent:.2%}, Filtered = +{prec_improvement_filtered:.2%}\n")
            f.write(f"- **Precision Filtered vs Recent**: +{prec_improvement_filtered_vs_recent:.2%}\n")
        
        # Test Performance
        f.write("\n### Test Performance\n\n")
        if all("test" in all_results[m] and all_results[m]["test"] for m in ["natural", "natural_recent", "natural_filtered"]):
            natural_test = all_results["natural"]["test"]
            recent_test = all_results["natural_recent"]["test"]
            filtered_test = all_results["natural_filtered"]["test"]
            
            f.write(f"- **V3_natural (all)**: Precision = {natural_test.get('precision', 0):.2%}, Recall = {natural_test.get('recall', 0):.2%}, Gini = {natural_test.get('gini', 0):.2%}\n")
            f.write(f"- **V3_natural_recent**: Precision = {recent_test.get('precision', 0):.2%}, Recall = {recent_test.get('recall', 0):.2%}, Gini = {recent_test.get('gini', 0):.2%}\n")
            f.write(f"- **V3_natural_filtered**: Precision = {filtered_test.get('precision', 0):.2%}, Recall = {filtered_test.get('recall', 0):.2%}, Gini = {filtered_test.get('gini', 0):.2%}\n\n")
            
            # Calculate improvements
            prec_improvement_recent = recent_test.get('precision', 0) - natural_test.get('precision', 0)
            prec_improvement_filtered = filtered_test.get('precision', 0) - natural_test.get('precision', 0)
            prec_improvement_filtered_vs_recent = filtered_test.get('precision', 0) - recent_test.get('precision', 0)
            
            gini_improvement_recent = recent_test.get('gini', 0) - natural_test.get('gini', 0)
            gini_improvement_filtered = filtered_test.get('gini', 0) - natural_test.get('gini', 0)
            gini_improvement_filtered_vs_recent = filtered_test.get('gini', 0) - recent_test.get('gini', 0)
            
            f.write(f"- **Precision vs All**: Recent = +{prec_improvement_recent:.2%}, Filtered = +{prec_improvement_filtered:.2%}\n")
            f.write(f"- **Precision Filtered vs Recent**: +{prec_improvement_filtered_vs_recent:.2%}\n")
            f.write(f"- **Gini vs All**: Recent = +{gini_improvement_recent:.4f}, Filtered = +{gini_improvement_filtered:.4f}\n")
            f.write(f"- **Gini Filtered vs Recent**: +{gini_improvement_filtered_vs_recent:.4f}\n")
        
        f.write("\n### Recommendations\n\n")
        f.write("Based on the comparison:\n\n")
        if all("test" in all_results[m] and all_results[m]["test"] for m in ["natural", "natural_recent", "natural_filtered"]):
            filtered_test = all_results["natural_filtered"]["test"]
            recent_test = all_results["natural_recent"]["test"]
            natural_test = all_results["natural"]["test"]
            
            best_precision = max(
                (natural_test.get('precision', 0), "V3_natural (all)"),
                (recent_test.get('precision', 0), "V3_natural_recent"),
                (filtered_test.get('precision', 0), "V3_natural_filtered"),
            )
            best_gini = max(
                (natural_test.get('gini', 0), "V3_natural (all)"),
                (recent_test.get('gini', 0), "V3_natural_recent"),
                (filtered_test.get('gini', 0), "V3_natural_filtered"),
            )
            
            f.write(f"- **Best Precision**: {best_precision[1]} ({best_precision[0]:.2%})\n")
            f.write(f"- **Best Gini**: {best_gini[1]} ({best_gini[0]:.4f})\n")
            f.write(f"- **Best Overall**: Consider the model with best balance of precision and Gini\n")
            f.write(f"- **Trade-offs**: Filtered model has fewer training samples but higher precision\n")
    
    print(f"\n✅ Comparison report saved to {output_path}")


def main():
    """Main evaluation and comparison pipeline."""
    print("=" * 80)
    print("V3 THREE MODELS COMPARISON")
    print("=" * 80)
    
    start_time = datetime.now()
    
    v3_root = PROJECT_ROOT / "models_production" / "lgbm_muscular_v3"
    
    # Define models to evaluate
    models_to_evaluate = [
        ("natural", v3_root / "model_natural", "V3_natural (all seasons)"),
        ("natural_recent", v3_root / "model_natural_recent", "V3_natural_recent (2018-2026)"),
        ("natural_filtered", v3_root / "model_natural_filtered", "V3_natural_filtered (2018-2026 excl. 2021-2022 & 2022-2023)"),
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
        if train_metrics:
            print(f"\n✅ Loaded training metrics for {model_name}")
        
        # Check if test metrics already exist
        test_metrics = load_test_metrics(model_dir, model_suffix)
        
        if not test_metrics:
            # Evaluate on test set
            test_metrics = evaluate_model_on_test(model_path, columns_path, model_name)
            
            # Save test metrics
            if test_metrics:
                test_metrics_path = model_dir / f"lgbm_v3_{model_suffix}_pl_only_metrics_test.json"
                with open(test_metrics_path, "w", encoding="utf-8") as f:
                    json.dump(test_metrics, f, indent=2)
                print(f"\n✅ Saved test metrics to {test_metrics_path}")
        else:
            print(f"\n✅ Using existing test metrics for {model_name}")
        
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
    for model_suffix, model_name in [
        ("natural", "V3_natural (all)"),
        ("natural_recent", "V3_natural_recent"),
        ("natural_filtered", "V3_natural_filtered"),
    ]:
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
    
    print(f"\n📁 Full comparison report: models_production/lgbm_muscular_v3/V3_THREE_MODELS_COMPARISON.md")


if __name__ == "__main__":
    main()

