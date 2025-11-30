#!/usr/bin/env python3
"""
Train a stacking ensemble that combines the V4 Random Forest and Gradient Boosting
models using a logistic meta-learner, targeting high precision (~70%+).

Outputs:
- models/ensemble_rf_gb_precision.joblib        (meta model)
- models/ensemble_rf_gb_precision_metrics.json  (train/val metrics + thresholds)
- analysis/rf_gb_precision_ensemble.md          (threshold table summary)
"""

import sys
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "timelines_35day_enhanced_balanced_v4_train.csv"
VAL_FILE = ROOT / "timelines_35day_enhanced_balanced_v4_val.csv"
ANALYSIS_FILE = ROOT / "analysis" / "rf_gb_precision_ensemble.md"
METRICS_FILE = ROOT / "models" / "ensemble_rf_gb_precision_metrics.json"
MODEL_FILE = ROOT / "models" / "ensemble_rf_gb_precision.joblib"

THRESHOLDS = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40]


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Match the V4 preprocessing rules (exclude metadata + week_5 features)."""
    feature_columns = [
        col
        for col in df.columns
        if col not in ["player_id", "reference_date", "player_name", "target"]
        and "_week_5" not in col
    ]
    X = df[feature_columns].copy()
    y = df["target"].copy()

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


def align_features(X: pd.DataFrame, training_columns: list[str]) -> pd.DataFrame:
    """Align encoded dataframe to the stored training column list."""
    missing_cols = [col for col in training_columns if col not in X.columns]
    if missing_cols:
        X = X.reindex(columns=list(X.columns) + missing_cols, fill_value=0)

    extra_cols = [col for col in X.columns if col not in training_columns]
    if extra_cols:
        X = X.drop(columns=extra_cols)

    return X[training_columns]


def get_model_scores(model_path: Path, columns_path: Path, X: pd.DataFrame) -> np.ndarray:
    """Load a model + column list and return its predicted probabilities."""
    model = joblib.load(model_path)
    training_columns = json.loads(columns_path.read_text(encoding="utf-8"))
    X_aligned = align_features(X, training_columns)
    return model.predict_proba(X_aligned)[:, 1]


def compute_metrics(y_true: np.ndarray, y_scores: np.ndarray, threshold: float):
    """Threshold-dependent metrics with confusion-matrix counts."""
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


def main():
    print("=" * 80)
    print("RF + GB PRECISION ENSEMBLE (STACKED LR)")
    print("=" * 80)

    if not TRAIN_FILE.exists() or not VAL_FILE.exists():
        raise FileNotFoundError("Missing train/val timeline CSV files.")

    df_train = pd.read_csv(TRAIN_FILE, encoding="utf-8-sig")
    df_val = pd.read_csv(VAL_FILE, encoding="utf-8-sig")
    X_train, y_train = prepare_data(df_train)
    X_val, y_val = prepare_data(df_val)

    print(f"✅ Train rows: {len(X_train):,} | positives: {y_train.sum():,}")
    print(f"✅ Val rows:   {len(X_val):,} | positives: {y_val.sum():,}")

    rf_model_path = ROOT / "models" / "rf_model_v4.joblib"
    rf_cols_path = ROOT / "models" / "rf_model_v4_columns.json"
    gb_model_path = ROOT / "models" / "gb_model_v4.joblib"
    gb_cols_path = ROOT / "models" / "gb_model_v4_columns.json"

    print("\n📈 Generating base model probabilities...")
    rf_train_scores = get_model_scores(rf_model_path, rf_cols_path, X_train)
    gb_train_scores = get_model_scores(gb_model_path, gb_cols_path, X_train)
    rf_val_scores = get_model_scores(rf_model_path, rf_cols_path, X_val)
    gb_val_scores = get_model_scores(gb_model_path, gb_cols_path, X_val)

    stack_train = np.column_stack([rf_train_scores, gb_train_scores])
    stack_val = np.column_stack([rf_val_scores, gb_val_scores])

    print("\n🤝 Training logistic meta-learner (balanced)...")
    meta = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
    )
    meta.fit(stack_train, y_train)

    train_scores = meta.predict_proba(stack_train)[:, 1]
    val_scores = meta.predict_proba(stack_val)[:, 1]
    train_auc = roc_auc_score(y_train, train_scores)
    val_auc = roc_auc_score(y_val, val_scores)
    print(f"   Train ROC AUC: {train_auc:.4f}")
    print(f"   Val   ROC AUC: {val_auc:.4f}")

    print("\n📊 Threshold sweep (validation)...")
    threshold_rows = [compute_metrics(y_val.values, val_scores, thr) for thr in THRESHOLDS]
    best_precision_row = max(threshold_rows, key=lambda r: (r["precision"], -r["recall"]))

    for row in threshold_rows:
        print(
            f"   τ={row['threshold']:.2f} | Prec={row['precision']:.3f} "
            f"| Recall={row['recall']:.3f} | F1={row['f1']:.3f} | TP={row['tp']} FP={row['fp']}"
        )

    print("\n🏁 Highest precision point:")
    print(
        f"   τ={best_precision_row['threshold']:.2f} "
        f"| Prec={best_precision_row['precision']:.3f} "
        f"| Recall={best_precision_row['recall']:.3f} "
        f"| F1={best_precision_row['f1']:.3f}"
    )

    # Persist artefacts
    ROOT.joinpath("models").mkdir(exist_ok=True)
    joblib.dump(meta, MODEL_FILE)
    METRICS_FILE.write_text(
        json.dumps(
            {
                "train_auc": train_auc,
                "val_auc": val_auc,
                "thresholds": threshold_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Markdown report
    analysis_lines = [
        "# RF + GB Precision Ensemble",
        "",
        "Stacking ensemble trained via `scripts/train_rf_gb_precision_ensemble.py`.",
        "",
        f"- Train ROC AUC: **{train_auc:.4f}**",
        f"- Val ROC AUC: **{val_auc:.4f}**",
        "",
        "| Threshold | Precision | Recall | F1 | ROC AUC | TP | FP | TN | FN |",
        "|-----------|-----------|--------|----|---------|----|----|----|----|",
    ]
    for row in threshold_rows:
        analysis_lines.append(
            f"| {row['threshold']:.2f} | {row['precision']:.3f} | {row['recall']:.3f} "
            f"| {row['f1']:.3f} | {row['roc_auc']:.3f} | {row['tp']} | {row['fp']} "
            f"| {row['tn']} | {row['fn']} |"
        )

    ANALYSIS_FILE.parent.mkdir(parents=True, exist_ok=True)
    ANALYSIS_FILE.write_text("\n".join(analysis_lines), encoding="utf-8")

    print(f"\n💾 Saved meta model -> {MODEL_FILE}")
    print(f"💾 Saved metrics    -> {METRICS_FILE}")
    print(f"📝 Saved table      -> {ANALYSIS_FILE}")


if __name__ == "__main__":
    main()


