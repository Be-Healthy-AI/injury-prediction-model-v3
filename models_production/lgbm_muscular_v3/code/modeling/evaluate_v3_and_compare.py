#!/usr/bin/env python3
"""
Evaluate V3 model on test set and compare with V1/V2 metrics.

This script:
1. Loads V3 model and evaluates on PL-only test timeline (2025-2026)
2. Loads V1 and V2 metrics for comparison
3. Creates a comprehensive comparison report
"""

import io
import sys

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import json
import os
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_models_seasonal_combined import (
    align_features,
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


def evaluate_v3_on_test():
    """Evaluate V3 model on test set."""
    print("=" * 80)
    print("EVALUATING V3 MODEL ON TEST SET")
    print("=" * 80)

    # Paths
    v3_root = PROJECT_ROOT / "models_production" / "lgbm_muscular_v3"
    model_path = v3_root / "model" / "model.joblib"
    columns_path = v3_root / "model" / "columns.json"
    test_path = v3_root / "data" / "timelines" / "test" / "timelines_35day_season_2025_2026_v4_muscular.csv"

    if not model_path.exists():
        print(f"ERROR: V3 model not found: {model_path}")
        return None

    if not test_path.exists():
        print(f"ERROR: V3 test timeline not found: {test_path}")
        return None

    # Load model
    print(f"\n📂 Loading V3 model from {model_path}...")
    model, model_columns = load_model_and_columns(model_path, columns_path)
    print(f"✅ Model loaded with {len(model_columns)} features")

    # Load test data
    print(f"\n📂 Loading test dataset: {test_path}...")
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

    return test_metrics


def load_v1_v2_metrics():
    """Load V1 and V2 metrics from metadata files."""
    v1_metrics_path = PROJECT_ROOT / "models_production" / "lgbm_muscular_v1" / "metadata" / "metrics_classic.json"
    v2_metrics_path = PROJECT_ROOT / "models_production" / "lgbm_muscular_v2" / "metadata" / "metrics_classic.json"

    v1_metrics = None
    v2_metrics = None

    if v1_metrics_path.exists():
        with open(v1_metrics_path, "r", encoding="utf-8") as f:
            v1_metrics = json.load(f)
        print(f"✅ Loaded V1 metrics from {v1_metrics_path}")

    if v2_metrics_path.exists():
        with open(v2_metrics_path, "r", encoding="utf-8") as f:
            v2_metrics = json.load(f)
        print(f"✅ Loaded V2 metrics from {v2_metrics_path}")

    return v1_metrics, v2_metrics


def load_v3_training_metrics():
    """Load V3 training metrics."""
    v3_metrics_path = PROJECT_ROOT / "models_production" / "lgbm_muscular_v3" / "model" / "lgbm_v3_pl_only_metrics_train.json"

    if v3_metrics_path.exists():
        with open(v3_metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def create_comparison_report(v1_metrics, v2_metrics, v3_train_metrics, v3_test_metrics):
    """Create a comprehensive comparison report."""
    v3_root = PROJECT_ROOT / "models_production" / "lgbm_muscular_v3"
    output_path = v3_root / "METRICS_COMPARISON.md"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Model Metrics Comparison: V1 vs V2 vs V3\n\n")
        f.write("## Overview\n\n")
        f.write("This document compares the performance metrics of three model versions:\n\n")
        f.write("- **V1**: Trained on all seasons (2008-2025), tested on 2025-2026 natural timeline\n")
        f.write("- **V2**: Trained on all seasons including 2025-2026, tested on 2025-2026 natural timeline (in-sample)\n")
        f.write("- **V3**: Trained on PL-only timelines (all seasons), tested on 2025-2026 PL-only timeline\n\n")
        f.write("---\n\n")

        # Training Metrics Comparison
        f.write("## Training Metrics Comparison\n\n")
        f.write("| Metric | V1 | V2 | V3 (PL-only) |\n")
        f.write("|--------|----|----|--------------|\n")

        if v1_metrics and "training_metrics" in v1_metrics:
            v1_train = v1_metrics["training_metrics"]
            acc_line = f"| **Accuracy** | {v1_train.get('accuracy', 0):.4f} | "
        else:
            acc_line = "| **Accuracy** | N/A | "

        if v2_metrics and "training_metrics" in v2_metrics:
            v2_train = v2_metrics["training_metrics"]
            acc_line += f"{v2_train.get('accuracy', 0):.4f} | "
        else:
            acc_line += "N/A | "

        if v3_train_metrics:
            acc_line += f"{v3_train_metrics.get('accuracy', 0):.4f} |\n"
        else:
            acc_line += "N/A |\n"
        
        f.write(acc_line)

        # Precision
        if v1_metrics and "training_metrics" in v1_metrics:
            prec_line = f"| **Precision** | {v1_train.get('precision', 0):.4f} | "
        else:
            prec_line = "| **Precision** | N/A | "

        if v2_metrics and "training_metrics" in v2_metrics:
            prec_line += f"{v2_train.get('precision', 0):.4f} | "
        else:
            prec_line += "N/A | "

        if v3_train_metrics:
            prec_line += f"{v3_train_metrics.get('precision', 0):.4f} |\n"
        else:
            prec_line += "N/A |\n"
        
        f.write(prec_line)

        # Recall
        if v1_metrics and "training_metrics" in v1_metrics:
            recall_line = f"| **Recall** | {v1_train.get('recall', 0):.4f} | "
        else:
            recall_line = "| **Recall** | N/A | "

        if v2_metrics and "training_metrics" in v2_metrics:
            recall_line += f"{v2_train.get('recall', 0):.4f} | "
        else:
            recall_line += "N/A | "

        if v3_train_metrics:
            recall_line += f"{v3_train_metrics.get('recall', 0):.4f} |\n"
        else:
            recall_line += "N/A |\n"
        
        f.write(recall_line)

        # F1-Score
        if v1_metrics and "training_metrics" in v1_metrics:
            f1_line = f"| **F1-Score** | {v1_train.get('f1', 0):.4f} | "
        else:
            f1_line = "| **F1-Score** | N/A | "

        if v2_metrics and "training_metrics" in v2_metrics:
            f1_line += f"{v2_train.get('f1', 0):.4f} | "
        else:
            f1_line += "N/A | "

        if v3_train_metrics:
            f1_line += f"{v3_train_metrics.get('f1', 0):.4f} |\n"
        else:
            f1_line += "N/A |\n"
        
        f.write(f1_line)

        # ROC AUC
        if v1_metrics and "training_metrics" in v1_metrics:
            auc_line = f"| **ROC AUC** | {v1_train.get('roc_auc', 0):.4f} | "
        else:
            auc_line = "| **ROC AUC** | N/A | "

        if v2_metrics and "training_metrics" in v2_metrics:
            auc_line += f"{v2_train.get('roc_auc', 0):.4f} | "
        else:
            auc_line += "N/A | "

        if v3_train_metrics:
            auc_line += f"{v3_train_metrics.get('roc_auc', 0):.4f} |\n"
        else:
            auc_line += "N/A |\n"
        
        f.write(auc_line)

        # Gini
        if v1_metrics and "training_metrics" in v1_metrics:
            gini_line = f"| **Gini** | {v1_train.get('gini', 0):.4f} | "
        else:
            gini_line = "| **Gini** | N/A | "

        if v2_metrics and "training_metrics" in v2_metrics:
            gini_line += f"{v2_train.get('gini', 0):.4f} | "
        else:
            gini_line += "N/A | "

        if v3_train_metrics:
            gini_line += f"{v3_train_metrics.get('gini', 0):.4f} |\n"
        else:
            gini_line += "N/A |\n"
        
        f.write(gini_line)

        f.write("\n---\n\n")

        # Test Metrics Comparison
        f.write("## Test Metrics Comparison\n\n")
        f.write("| Metric | V1 (2025-2026 natural) | V2 (2025-2026 natural, in-sample) | V3 (2025-2026 PL-only) |\n")
        f.write("|--------|------------------------|-----------------------------------|------------------------|\n")

        if v1_metrics and "test_metrics" in v1_metrics:
            v1_test = v1_metrics["test_metrics"]
            test_acc_line = f"| **Accuracy** | {v1_test.get('accuracy', 0):.4f} | "
        else:
            test_acc_line = "| **Accuracy** | N/A | "

        if v2_metrics and "test_metrics" in v2_metrics:
            v2_test = v2_metrics["test_metrics"]
            test_acc_line += f"{v2_test.get('accuracy', 0):.4f} | "
        else:
            test_acc_line += "N/A | "

        if v3_test_metrics:
            test_acc_line += f"{v3_test_metrics.get('accuracy', 0):.4f} |\n"
        else:
            test_acc_line += "N/A |\n"
        
        f.write(test_acc_line)

        # Precision
        if v1_metrics and "test_metrics" in v1_metrics:
            test_prec_line = f"| **Precision** | {v1_test.get('precision', 0):.4f} | "
        else:
            test_prec_line = "| **Precision** | N/A | "

        if v2_metrics and "test_metrics" in v2_metrics:
            test_prec_line += f"{v2_test.get('precision', 0):.4f} | "
        else:
            test_prec_line += "N/A | "

        if v3_test_metrics:
            test_prec_line += f"{v3_test_metrics.get('precision', 0):.4f} |\n"
        else:
            test_prec_line += "N/A |\n"
        
        f.write(test_prec_line)

        # Recall
        if v1_metrics and "test_metrics" in v1_metrics:
            test_recall_line = f"| **Recall** | {v1_test.get('recall', 0):.4f} | "
        else:
            test_recall_line = "| **Recall** | N/A | "

        if v2_metrics and "test_metrics" in v2_metrics:
            test_recall_line += f"{v2_test.get('recall', 0):.4f} | "
        else:
            test_recall_line += "N/A | "

        if v3_test_metrics:
            test_recall_line += f"{v3_test_metrics.get('recall', 0):.4f} |\n"
        else:
            test_recall_line += "N/A |\n"
        
        f.write(test_recall_line)

        # F1-Score
        if v1_metrics and "test_metrics" in v1_metrics:
            test_f1_line = f"| **F1-Score** | {v1_test.get('f1', 0):.4f} | "
        else:
            test_f1_line = "| **F1-Score** | N/A | "

        if v2_metrics and "test_metrics" in v2_metrics:
            test_f1_line += f"{v2_test.get('f1', 0):.4f} | "
        else:
            test_f1_line += "N/A | "

        if v3_test_metrics:
            test_f1_line += f"{v3_test_metrics.get('f1', 0):.4f} |\n"
        else:
            test_f1_line += "N/A |\n"
        
        f.write(test_f1_line)

        # ROC AUC
        if v1_metrics and "test_metrics" in v1_metrics:
            test_auc_line = f"| **ROC AUC** | {v1_test.get('roc_auc', 0):.4f} | "
        else:
            test_auc_line = "| **ROC AUC** | N/A | "

        if v2_metrics and "test_metrics" in v2_metrics:
            test_auc_line += f"{v2_test.get('roc_auc', 0):.4f} | "
        else:
            test_auc_line += "N/A | "

        if v3_test_metrics:
            test_auc_line += f"{v3_test_metrics.get('roc_auc', 0):.4f} |\n"
        else:
            test_auc_line += "N/A |\n"
        
        f.write(test_auc_line)

        # Gini
        if v1_metrics and "test_metrics" in v1_metrics:
            test_gini_line = f"| **Gini** | {v1_test.get('gini', 0):.4f} | "
        else:
            test_gini_line = "| **Gini** | N/A | "

        if v2_metrics and "test_metrics" in v2_metrics:
            test_gini_line += f"{v2_test.get('gini', 0):.4f} | "
        else:
            test_gini_line += "N/A | "

        if v3_test_metrics:
            test_gini_line += f"{v3_test_metrics.get('gini', 0):.4f} |\n"
        else:
            test_gini_line += "N/A |\n"
        
        f.write(test_gini_line)

        f.write("\n---\n\n")

        # Dataset Sizes
        f.write("## Dataset Sizes\n\n")
        f.write("| Version | Training Records | Test Records |\n")
        f.write("|---------|-------------------|--------------|\n")
        f.write("| V1 | ~4,000,000 (all seasons 2008-2025) | 153,006 (2025-2026 natural) |\n")
        f.write("| V2 | ~4,000,000 (all seasons including 2025-2026) | 153,006 (2025-2026 natural, in-sample) |\n")
        f.write("| V3 | 32,008 (PL-only, all seasons) | 13,386 (2025-2026 PL-only) |\n")

        f.write("\n---\n\n")

        # Key Observations
        f.write("## Key Observations\n\n")
        f.write("### Training Metrics\n")
        if v3_train_metrics:
            f.write(f"- **V3** shows exceptional training performance: {v3_train_metrics.get('gini', 0):.2%} Gini, {v3_train_metrics.get('recall', 0):.2%} Recall\n")
            f.write(f"- Much smaller training dataset (32K vs 4M) but focused on PL context\n")
            f.write(f"- Higher precision ({v3_train_metrics.get('precision', 0):.2%}) compared to V1/V2\n\n")

        f.write("### Test Metrics\n")
        if v3_test_metrics:
            f.write(f"- **V3** test Gini: {v3_test_metrics.get('gini', 0):.2%}\n")
            f.write(f"- **V3** test Recall: {v3_test_metrics.get('recall', 0):.2%}\n")
            f.write(f"- **V3** test Precision: {v3_test_metrics.get('precision', 0):.2%}\n")
            f.write(f"- Test set is much smaller (13K vs 153K) due to PL-only filtering\n\n")

        f.write("### Notes\n")
        f.write("- V1: True hold-out test (2025-2026 not in training)\n")
        f.write("- V2: In-sample test (2025-2026 included in training)\n")
        f.write("- V3: PL-only test (2025-2026 PL-only timeline, all seasons in training)\n")
        f.write("- V3's smaller dataset reflects the PL-only filtering approach\n")

    print(f"\n✅ Comparison report saved to {output_path}")


def main():
    """Main evaluation and comparison pipeline."""
    print("=" * 80)
    print("V3 MODEL EVALUATION AND COMPARISON WITH V1/V2")
    print("=" * 80)

    start_time = datetime.now()

    # Load V1/V2 metrics
    print("\n📂 Loading V1 and V2 metrics...")
    v1_metrics, v2_metrics = load_v1_v2_metrics()

    # Load V3 training metrics
    print("\n📂 Loading V3 training metrics...")
    v3_train_metrics = load_v3_training_metrics()
    if v3_train_metrics:
        print("✅ V3 training metrics loaded")

    # Evaluate V3 on test set
    print("\n" + "=" * 80)
    v3_test_metrics = evaluate_v3_on_test()

    # Save V3 test metrics
    if v3_test_metrics:
        v3_root = PROJECT_ROOT / "models_production" / "lgbm_muscular_v3"
        v3_model_dir = v3_root / "model"
        test_metrics_path = v3_model_dir / "lgbm_v3_pl_only_metrics_test.json"
        with open(test_metrics_path, "w", encoding="utf-8") as f:
            json.dump(v3_test_metrics, f, indent=2)
        print(f"\n✅ Saved V3 test metrics to {test_metrics_path}")

    # Create comparison report
    print("\n" + "=" * 80)
    print("CREATING COMPARISON REPORT")
    print("=" * 80)
    try:
        create_comparison_report(v1_metrics, v2_metrics, v3_train_metrics, v3_test_metrics)
    except Exception as e:
        print(f"ERROR creating comparison report: {e}")
        import traceback
        traceback.print_exc()

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    elapsed_time = datetime.now() - start_time
    print(f"⏱️  Total time: {elapsed_time}")
    print(f"\n📊 Summary:")
    if v3_train_metrics:
        print(f"   V3 Training - Gini: {v3_train_metrics.get('gini', 0):.2%}, Recall: {v3_train_metrics.get('recall', 0):.2%}, Precision: {v3_train_metrics.get('precision', 0):.2%}")
    if v3_test_metrics:
        print(f"   V3 Test - Gini: {v3_test_metrics.get('gini', 0):.2%}, Recall: {v3_test_metrics.get('recall', 0):.2%}, Precision: {v3_test_metrics.get('precision', 0):.2%}")
    print(f"\n📁 Full comparison report: models_production/lgbm_muscular_v3/METRICS_COMPARISON.md")


if __name__ == "__main__":
    main()

