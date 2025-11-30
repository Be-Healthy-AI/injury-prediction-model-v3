#!/usr/bin/env python3
"""
Grid-search simple AND-threshold rules on the RF/GB probabilities to
target high-precision operating points.
"""

import sys

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))
from tune_thresholds_v4 import prepare_data, align_features  # type: ignore
VAL_FILE = ROOT / "timelines_35day_enhanced_balanced_v4_val.csv"


def main():
    df_val = pd.read_csv(VAL_FILE, encoding="utf-8-sig")
    X_val, y_val = prepare_data(df_val)

    rf_model = joblib.load(ROOT / "models" / "rf_model_v4.joblib")
    rf_cols = json.loads((ROOT / "models" / "rf_model_v4_columns.json").read_text(encoding="utf-8"))
    gb_model = joblib.load(ROOT / "models" / "gb_model_v4.joblib")
    gb_cols = json.loads((ROOT / "models" / "gb_model_v4_columns.json").read_text(encoding="utf-8"))

    rf_scores = rf_model.predict_proba(align_features(X_val.copy(), rf_cols))[:, 1]
    gb_scores = gb_model.predict_proba(align_features(X_val.copy(), gb_cols))[:, 1]

    candidates = []
    for rf_thr in np.arange(0.30, 0.61, 0.02):
        for gb_thr in np.arange(0.20, 0.51, 0.02):
            preds = (rf_scores >= rf_thr) & (gb_scores >= gb_thr)
            if preds.sum() == 0:
                continue
            precision = precision_score(y_val, preds, zero_division=0)
            recall = recall_score(y_val, preds, zero_division=0)
            f1 = f1_score(y_val, preds, zero_division=0)
            tp = int(((preds == 1) & (y_val.values == 1)).sum())
            fp = int(((preds == 1) & (y_val.values == 0)).sum())
            tn = int(((preds == 0) & (y_val.values == 0)).sum())
            fn = int(((preds == 0) & (y_val.values == 1)).sum())

            candidates.append(
                {
                    "rf_thr": round(float(rf_thr), 2),
                    "gb_thr": round(float(gb_thr), 2),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "alerts": int(preds.sum()),
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                }
            )

    candidates.sort(key=lambda row: (-row["precision"], row["alerts"]))
    print("Top 20 high-precision combos:")
    for row in candidates[:20]:
        print(row)

    target = [c for c in candidates if 0.70 <= c["precision"] <= 0.80]
    print("\nCandidates within 0.70-0.80 precision range:")
    for row in target[:10]:
        print(row)


if __name__ == "__main__":
    main()


