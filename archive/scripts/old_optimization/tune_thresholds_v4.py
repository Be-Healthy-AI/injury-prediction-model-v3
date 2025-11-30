#!/usr/bin/env python3
"""
Threshold tuning and lightweight ensemble evaluation for V4 models.

This script:
1. Loads the V4 out-of-sample validation timelines.
2. Generates probability scores for the trained RF/GB/LR models.
3. Sweeps through multiple classification thresholds to understand the
   precision/recall/F1 trade-offs (goal: recover recall without losing all precision).
4. Tests simple probability-level ensembles that blend Logistic Regression
   (high recall) with tree-based models (higher precision).
5. Writes detailed results to analysis/threshold_optimization_v4.md + .json.
"""

import sys
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "timelines_35day_enhanced_balanced_v4_val.csv"
ALTERNATE_DATA_FILE = ROOT / "scripts" / "timelines_35day_enhanced_balanced_v4_val.csv"
ANALYSIS_DIR = ROOT / "analysis"
OUTPUT_JSON = ANALYSIS_DIR / "threshold_optimization_v4.json"
OUTPUT_MD = ANALYSIS_DIR / "threshold_optimization_v4.md"

MODEL_CONFIGS = [
    {
        "key": "random_forest",
        "label": "Random Forest",
        "model_path": ROOT / "models" / "rf_model_v4.joblib",
        "columns_path": ROOT / "models" / "rf_model_v4_columns.json",
        "thresholds": [0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20],
    },
    {
        "key": "gradient_boosting",
        "label": "Gradient Boosting",
        "model_path": ROOT / "models" / "gb_model_v4.joblib",
        "columns_path": ROOT / "models" / "gb_model_v4_columns.json",
        "thresholds": [0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20],
    },
    {
        "key": "logistic_regression",
        "label": "Logistic Regression",
        "model_path": ROOT / "models" / "lr_model_v4.joblib",
        "columns_path": ROOT / "models" / "lr_model_v4_columns.json",
        "thresholds": [0.50, 0.55, 0.60, 0.65, 0.70],
    },
]

ENSEMBLE_CONFIGS = [
    {
        "key": "ensemble_rf_50_gb_50",
        "label": "RF 50% + GB 50%",
        "weights": {"random_forest": 0.5, "gradient_boosting": 0.5},
        "thresholds": [0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30],
    },
    {
        "key": "ensemble_rf_60_gb_40",
        "label": "RF 60% + GB 40%",
        "weights": {"random_forest": 0.6, "gradient_boosting": 0.4},
        "thresholds": [0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30],
    },
    {
        "key": "ensemble_rf_40_gb_60",
        "label": "RF 40% + GB 60%",
        "weights": {"random_forest": 0.4, "gradient_boosting": 0.6},
        "thresholds": [0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30],
    },
    {
        "key": "ensemble_rf_gb_min",
        "label": "RF & GB agreement (min prob)",
        "weights": {"random_forest": 1.0, "gradient_boosting": 1.0},
        "combiner": "minimum",
        "thresholds": [0.60, 0.55, 0.50, 0.45, 0.42, 0.40, 0.38, 0.36, 0.34, 0.32, 0.30, 0.25, 0.20, 0.15, 0.10],
    },
    {
        "key": "ensemble_rf_gb_geo",
        "label": "RF × GB geometric mean",
        "weights": {"random_forest": 1.0, "gradient_boosting": 1.0},
        "combiner": "geometric_mean",
        "thresholds": [0.60, 0.55, 0.50, 0.45, 0.42, 0.40, 0.38, 0.36, 0.34, 0.32, 0.30, 0.25, 0.20, 0.15, 0.10],
    },
    {
        "key": "ensemble_lr_60_gb_40",
        "label": "LR 60% + GB 40%",
        "weights": {"logistic_regression": 0.6, "gradient_boosting": 0.4},
        "thresholds": [0.50, 0.45, 0.40, 0.35, 0.30],
    },
    {
        "key": "ensemble_lr_60_rf_40",
        "label": "LR 60% + RF 40%",
        "weights": {"logistic_regression": 0.6, "random_forest": 0.4},
        "thresholds": [0.50, 0.45, 0.40, 0.35, 0.30],
    },
    {
        "key": "ensemble_lr_50_gb_25_rf_25",
        "label": "LR 50% + GB 25% + RF 25%",
        "weights": {
            "logistic_regression": 0.5,
            "gradient_boosting": 0.25,
            "random_forest": 0.25,
        },
        "thresholds": [0.50, 0.45, 0.40, 0.35, 0.30],
    },
]


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Mirror the V4 training preprocessing."""
    feature_columns = [
        col
        for col in df.columns
        if col not in ["player_id", "reference_date", "player_name", "target"]
        and "_week_5" not in col
    ]
    X = df[feature_columns].copy()
    y = df["target"]

    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    X_encoded = X.copy()

    for feature in categorical_features:
        X_encoded[feature] = X_encoded[feature].fillna("Unknown")
        dummies = pd.get_dummies(X_encoded[feature], prefix=feature, drop_first=True)
        X_encoded = pd.concat([X_encoded, dummies], axis=1)
        X_encoded = X_encoded.drop(columns=[feature])

    for feature in numeric_features:
        X_encoded[feature] = X_encoded[feature].fillna(0)

    return X_encoded, y


def align_features(X: pd.DataFrame, training_columns: List[str]) -> pd.DataFrame:
    """Align encoded dataframe to the training columns for a specific model."""
    missing_cols = [col for col in training_columns if col not in X.columns]
    if missing_cols:
        X = X.reindex(columns=list(X.columns) + missing_cols, fill_value=0)

    extra_cols = [col for col in X.columns if col not in training_columns]
    if extra_cols:
        X = X.drop(columns=extra_cols)

    return X[training_columns]


def compute_metrics(y_true: np.ndarray, y_scores: np.ndarray, threshold: float) -> Dict[str, float]:
    """Compute classification metrics for a given threshold."""
    y_pred = (y_scores >= threshold).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.0
    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "gini": 2 * auc - 1,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def evaluate_models(X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict]:
    """Generate threshold metrics for each individual model."""
    results = {}
    base_scores = {}

    for config in MODEL_CONFIGS:
        model_name = config["key"]
        print(f"\n🔍 Evaluating {config['label']}...")

        if not config["model_path"].exists():
            print(f"   ❌ Missing model file: {config['model_path']}")
            continue

        training_columns = json.loads(config["columns_path"].read_text(encoding="utf-8"))
        model = joblib.load(config["model_path"])

        X_aligned = align_features(X, training_columns)
        y_scores = model.predict_proba(X_aligned)[:, 1]
        base_scores[model_name] = y_scores

        threshold_metrics = [
            compute_metrics(y.values, y_scores, threshold) for threshold in config["thresholds"]
        ]

        best_by_f1 = max(threshold_metrics, key=lambda m: m["f1"])
        best_by_recall = max(threshold_metrics, key=lambda m: m["recall"])

        results[model_name] = {
            "label": config["label"],
            "threshold_metrics": threshold_metrics,
            "best_f1": best_by_f1,
            "best_recall": best_by_recall,
        }

        print(
            f"   ✅ Best F1 @ {best_by_f1['threshold']:.2f}: "
            f"F1={best_by_f1['f1']:.3f}, Prec={best_by_f1['precision']:.3f}, Recall={best_by_f1['recall']:.3f}"
        )
        print(
            f"   ✅ Max Recall @ {best_by_recall['threshold']:.2f}: "
            f"Recall={best_by_recall['recall']:.3f}, Prec={best_by_recall['precision']:.3f}"
        )

    return results, base_scores


def evaluate_ensembles(y: pd.Series, base_scores: Dict[str, np.ndarray]) -> Dict[str, Dict]:
    """Create probability ensembles (weighted, geometric, etc.) and evaluate thresholds."""
    ensemble_results = {}

    for config in ENSEMBLE_CONFIGS:
        if any(model_key not in base_scores for model_key in config["weights"]):
            print(f"\n⚠️  Skipping {config['label']} - missing base model scores.")
            continue

        print(f"\n🧪 Evaluating ensemble: {config['label']}")
        weights = config["weights"]
        combiner = config.get("combiner", "weighted")

        if combiner == "weighted":
            total_weight = sum(weights.values())
            combined_scores = np.zeros_like(next(iter(base_scores.values())))
            for model_key, weight in weights.items():
                combined_scores += base_scores[model_key] * (weight / total_weight)
        elif combiner == "minimum":
            stacked = np.stack([base_scores[model_key] for model_key in weights.keys()], axis=0)
            combined_scores = np.min(stacked, axis=0)
        elif combiner == "geometric_mean":
            stacked = np.stack([np.clip(base_scores[model_key], 1e-9, 1.0) for model_key in weights.keys()], axis=0)
            combined_scores = np.prod(stacked, axis=0) ** (1.0 / len(weights))
        else:
            raise ValueError(f"Unknown combiner '{combiner}' for ensemble {config['key']}")

        threshold_metrics = [
            compute_metrics(y.values, combined_scores, threshold) for threshold in config["thresholds"]
        ]
        best_by_f1 = max(threshold_metrics, key=lambda m: m["f1"])
        best_by_recall = max(threshold_metrics, key=lambda m: m["recall"])

        ensemble_results[config["key"]] = {
            "label": config["label"],
            "threshold_metrics": threshold_metrics,
            "best_f1": best_by_f1,
            "best_recall": best_by_recall,
        }

        print(
            f"   ✅ Best F1 @ {best_by_f1['threshold']:.2f}: "
            f"F1={best_by_f1['f1']:.3f}, Prec={best_by_f1['precision']:.3f}, Recall={best_by_f1['recall']:.3f}"
        )
        print(
            f"   ✅ Max Recall @ {best_by_recall['threshold']:.2f}: "
            f"Recall={best_by_recall['recall']:.3f}, Prec={best_by_recall['precision']:.3f}"
        )

    return ensemble_results


def write_outputs(
    model_results: Dict[str, Dict],
    ensemble_results: Dict[str, Dict],
) -> None:
    """Persist results as JSON + Markdown summary."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "models": model_results,
        "ensembles": ensemble_results,
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def build_table(rows: List[Dict[str, float]]) -> str:
        header = "| Threshold | Precision | Recall | F1 | ROC AUC | TP | FP | TN | FN |\n"
        header += "|-----------|-----------|--------|----|---------|----|----|----|----|\n"
        lines = []
        for row in rows:
            lines.append(
                f"| {row['threshold']:.2f} | {row['precision']:.3f} | {row['recall']:.3f} "
                f"| {row['f1']:.3f} | {row['roc_auc']:.3f} | {row['tp']} | {row['fp']} "
                f"| {row['tn']} | {row['fn']} |"
            )
        return header + "\n".join(lines)

    md_parts = [
        "# Threshold Optimization - V4 Models",
        "",
        "Generated via scripts/tune_thresholds_v4.py",
        "",
        "## Individual Models",
    ]

    for key, result in model_results.items():
        md_parts.append(f"### {result['label']}")
        md_parts.append(
            f"- Best F1 @ {result['best_f1']['threshold']:.2f}: "
            f"F1 **{result['best_f1']['f1']:.3f}**, "
            f"Precision {result['best_f1']['precision']:.3f}, "
            f"Recall {result['best_f1']['recall']:.3f}"
        )
        md_parts.append(
            f"- Max Recall @ {result['best_recall']['threshold']:.2f}: "
            f"Recall **{result['best_recall']['recall']:.3f}**, "
            f"Precision {result['best_recall']['precision']:.3f}"
        )
        md_parts.append("")
        md_parts.append(build_table(result["threshold_metrics"]))
        md_parts.append("")

    md_parts.append("## Ensembles")

    for key, result in ensemble_results.items():
        md_parts.append(f"### {result['label']}")
        md_parts.append(
            f"- Best F1 @ {result['best_f1']['threshold']:.2f}: "
            f"F1 **{result['best_f1']['f1']:.3f}**, "
            f"Precision {result['best_f1']['precision']:.3f}, "
            f"Recall {result['best_f1']['recall']:.3f}"
        )
        md_parts.append(
            f"- Max Recall @ {result['best_recall']['threshold']:.2f}: "
            f"Recall **{result['best_recall']['recall']:.3f}**, "
            f"Precision {result['best_recall']['precision']:.3f}"
        )
        md_parts.append("")
        md_parts.append(build_table(result["threshold_metrics"]))
        md_parts.append("")

    OUTPUT_MD.write_text("\n".join(md_parts), encoding="utf-8")
    print(f"\n📝 Saved JSON results to {OUTPUT_JSON}")
    print(f"📝 Saved markdown summary to {OUTPUT_MD}")


def main():
    print("=" * 90)
    print("THRESHOLD TUNING & ENSEMBLE ANALYSIS (V4)")
    print("=" * 90)

    data_path = DATA_FILE if DATA_FILE.exists() else ALTERNATE_DATA_FILE
    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find validation data. Expected {DATA_FILE} or {ALTERNATE_DATA_FILE}"
        )

    print(f"\n📂 Loading out-of-sample validation data from {data_path}")
    df_val = pd.read_csv(data_path, encoding="utf-8-sig")
    X, y = prepare_data(df_val)
    print(f"✅ Dataset: {len(df_val):,} rows | Injury rate: {y.mean():.2%}")

    model_results, base_scores = evaluate_models(X, y)
    ensemble_results = evaluate_ensembles(y, base_scores)
    write_outputs(model_results, ensemble_results)

    print("\n🎉 Threshold tuning complete!")


if __name__ == "__main__":
    main()


