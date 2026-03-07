#!/usr/bin/env python3
"""
One-off script to investigate why skeletal LGBM predictions differ between:
  - model_skeletal/test_predictions_from_training_pipeline.csv (training export)
  - production challenger predictions (Arsenal FC)

Compares feature vectors and model outputs for a specific (player_id, reference_date).
Default: player_id=144028, reference_date=2025-11-26.

Usage (from repo root):
  python models_production/lgbm_muscular_v4/code/modeling/investigate_skeletal_prediction_diff.py
  python models_production/lgbm_muscular_v4/code/modeling/investigate_skeletal_prediction_diff.py --player 144028 --date 2025-11-26
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Skeletal paths
TEST_TIMELINES = (
    ROOT_DIR
    / "models_production/lgbm_muscular_v4/data/timelines/test"
    / "timelines_35day_season_2025_2026_v4_labeled_muscle_skeletal_only_d7.csv"
)
CHALLENGER_TIMELINES = (
    ROOT_DIR
    / "production/deployments/England/challenger/Arsenal FC/timelines"
    / "timelines_35day_season_2025_2026_v4_muscular.csv"
)
MODEL_SKELETAL_DIR = ROOT_DIR / "models_production/lgbm_muscular_v4/model_skeletal"
REFERENCE_CSV = MODEL_SKELETAL_DIR / "test_predictions_from_training_pipeline.csv"
PROD_PREDICTIONS_CSV = (
    ROOT_DIR
    / "production/deployments/England/challenger/Arsenal FC/predictions"
    / "predictions_lgbm_v4_20260209.csv"
)


def load_skeletal_prepare_data():
    """Import prepare_data and filter_timelines from skeletal training script."""
    from models_production.lgbm_muscular_v4.code.modeling.train_iterative_feature_selection_skeletal_standalone import (  # noqa: E501
        filter_timelines_for_model,
        prepare_data,
    )
    return prepare_data, filter_timelines_for_model


def align_to_model_columns(X: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Align preprocessed X to exact model column order; missing -> 0."""
    existing = [c for c in columns if c in X.columns]
    missing = [c for c in columns if c not in X.columns]
    out = X[existing].copy() if existing else pd.DataFrame(index=X.index)
    if missing:
        out = pd.concat([out, pd.DataFrame(0.0, index=X.index, columns=missing)], axis=1)
    out = out[columns].astype(float)
    return out


def main():
    parser = argparse.ArgumentParser(description="Investigate skeletal prediction difference")
    parser.add_argument("--player", type=int, default=144028, help="player_id")
    parser.add_argument("--date", type=str, default="2025-11-26", help="reference_date YYYY-MM-DD")
    args = parser.parse_args()

    player_id = args.player
    ref_date = pd.to_datetime(args.date).date()

    print("=" * 70)
    print("SKELETAL PREDICTION DIFF INVESTIGATION")
    print("=" * 70)
    print(f"Player ID: {player_id}, Reference date: {ref_date}")
    print()

    # Load reference and production predictions for this row
    if not REFERENCE_CSV.exists():
        print(f"[ERROR] Reference file not found: {REFERENCE_CSV}")
        return 1
    if not PROD_PREDICTIONS_CSV.exists():
        print(f"[ERROR] Production predictions not found: {PROD_PREDICTIONS_CSV}")
        return 1

    ref_df = pd.read_csv(REFERENCE_CSV)
    ref_df["reference_date"] = pd.to_datetime(ref_df["reference_date"], errors="coerce")
    ref_row = ref_df[
        (ref_df["player_id"] == player_id) & (ref_df["reference_date"].dt.date == ref_date)
    ]
    prod_df = pd.read_csv(PROD_PREDICTIONS_CSV, low_memory=False)
    prod_df["reference_date"] = pd.to_datetime(prod_df["reference_date"], errors="coerce")
    prod_row = prod_df[
        (prod_df["player_id"] == player_id) & (prod_df["reference_date"].dt.date == ref_date)
    ]

    if ref_row.empty:
        print(f"[ERROR] No row in reference CSV for player_id={player_id}, date={ref_date}")
        return 1
    if prod_row.empty:
        print(f"[ERROR] No row in production CSV for player_id={player_id}, date={ref_date}")
        return 1

    pred_ref = float(ref_row.iloc[0]["predicted_probability"])
    pred_prod = float(prod_row.iloc[0]["injury_probability_skeletal"])
    print(f"Reference (test_predictions_from_training_pipeline.csv): {pred_ref}")
    print(f"Production (challenger predictions):                     {pred_prod}")
    print(f"Absolute difference:                                    {abs(pred_ref - pred_prod)}")
    print()

    # Load model and columns
    model_path = MODEL_SKELETAL_DIR / "model.joblib"
    cols_path = MODEL_SKELETAL_DIR / "columns.json"
    if not model_path.exists() or not cols_path.exists():
        print(f"[ERROR] Model or columns not found in {MODEL_SKELETAL_DIR}")
        return 1
    model = joblib.load(model_path)
    with open(cols_path, "r", encoding="utf-8") as f:
        columns = json.load(f)
    if isinstance(columns, dict) and "features" in columns:
        columns = columns["features"]
    columns = list(columns)
    print(f"Model features: {len(columns)}")
    print()

    prepare_data, filter_timelines_for_model = load_skeletal_prepare_data()

    # 1) Test dataset: load full test, filter for skeletal, prepare_data, then take our row
    if not TEST_TIMELINES.exists():
        print(f"[ERROR] Test timelines not found: {TEST_TIMELINES}")
        return 1
    print("[1] Loading test timelines and filtering for skeletal...")
    df_test_all = pd.read_csv(TEST_TIMELINES, low_memory=False)
    df_test_all["reference_date"] = pd.to_datetime(df_test_all["reference_date"], errors="coerce")
    df_test_skeletal = filter_timelines_for_model(df_test_all, "target2")
    df_test_skeletal = df_test_skeletal.reset_index(drop=True)
    print(f"    Test skeletal rows: {len(df_test_skeletal)}")

    print("    Running skeletal prepare_data on full test set (no cache)...")
    X_test_full = prepare_data(df_test_skeletal, cache_file=None, use_cache=False)
    # Get row index for our player/date
    mask_test = (df_test_skeletal["player_id"] == player_id) & (
        df_test_skeletal["reference_date"].dt.date == ref_date
    )
    if not mask_test.any():
        print(f"[ERROR] No row in test skeletal df for player_id={player_id}, date={ref_date}")
        return 1
    idx_test = df_test_skeletal.index[mask_test][0]
    X_test_row = X_test_full.loc[[idx_test]]
    X_test_aligned = align_to_model_columns(X_test_row, columns)
    pred_test_recomputed = float(model.predict_proba(X_test_aligned)[:, 1][0])
    print(f"    Recomputed prediction from test pipeline: {pred_test_recomputed}")
    print(f"    (should match reference {pred_ref}: diff = {abs(pred_test_recomputed - pred_ref)})")
    print()

    # 2) Challenger dataset: load full challenger timelines, prepare_data, then take our row
    if not CHALLENGER_TIMELINES.exists():
        print(f"[ERROR] Challenger timelines not found: {CHALLENGER_TIMELINES}")
        return 1
    print("[2] Loading challenger timelines...")
    df_prod_all = pd.read_csv(CHALLENGER_TIMELINES, low_memory=False)
    df_prod_all["reference_date"] = pd.to_datetime(df_prod_all["reference_date"], errors="coerce")
    print(f"    Challenger rows: {len(df_prod_all)}")

    print("    Running skeletal prepare_data on full challenger set (no cache)...")
    X_prod_full = prepare_data(df_prod_all, cache_file=None, use_cache=False)
    mask_prod = (df_prod_all["player_id"] == player_id) & (
        df_prod_all["reference_date"].dt.date == ref_date
    )
    if not mask_prod.any():
        print(f"[ERROR] No row in challenger df for player_id={player_id}, date={ref_date}")
        return 1
    idx_prod = df_prod_all.index[mask_prod][0]
    X_prod_row = X_prod_full.loc[[idx_prod]]
    X_prod_aligned = align_to_model_columns(X_prod_row, columns)
    pred_prod_recomputed = float(model.predict_proba(X_prod_aligned)[:, 1][0])
    print(f"    Recomputed prediction from challenger pipeline: {pred_prod_recomputed}")
    print(f"    (should match production {pred_prod}: diff = {abs(pred_prod_recomputed - pred_prod)})")
    print()

    # 3) Feature-by-feature comparison
    print("[3] Feature-by-feature comparison (test vs challenger aligned row)...")
    diffs = []
    for c in columns:
        vt = float(X_test_aligned.iloc[0][c])
        vp = float(X_prod_aligned.iloc[0][c])
        if abs(vt - vp) > 1e-9:
            diffs.append((c, vt, vp, vt - vp))

    print(f"    Number of features that differ (|test - prod| > 1e-9): {len(diffs)}")
    if diffs:
        print()
        print("    First 50 differing features: name | test_value | prod_value | delta")
        print("    " + "-" * 60)
        for c, vt, vp, d in sorted(diffs, key=lambda x: -abs(x[3]))[:50]:
            print(f"    {c} | {vt} | {vp} | {d}")
        if len(diffs) > 50:
            print(f"    ... and {len(diffs) - 50} more.")
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Reference file probability:         {pred_ref}")
    print(f"Production file probability:        {pred_prod}")
    print(f"Recomputed from test pipeline:      {pred_test_recomputed}")
    print(f"Recomputed from challenger pipeline: {pred_prod_recomputed}")
    print(f"Feature differences (count):        {len(diffs)}")
    print()
    if abs(pred_test_recomputed - pred_ref) > 0.01 or abs(pred_prod_recomputed - pred_prod) > 0.01:
        print("Note: Recomputed values do not match the saved CSV values.")
        print("      Reference/production CSVs may have been generated with a different")
        print("      model version or export pipeline.")
    if len(diffs) == 0:
        print("No feature differences -> same pipeline yields same prediction for both sources.")
    else:
        print("Feature vectors differ between test and challenger for this row (see list above).")
        print("With the same pipeline, both sources give the same recomputed probability here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
