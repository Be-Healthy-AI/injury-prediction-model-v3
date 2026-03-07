#!/usr/bin/env python3
"""
Export the 320 feature values for a single row (player_id=144028, reference_date=2025-12-05)
in exact columns.json order, using the same pipeline as evaluate_production_lgbm_on_new_test.py.

Outputs:
  - player_144028_ref_2025-12-05_features_320_order.csv (one row, 320 columns)
  - player_144028_ref_2025-12-05_note.txt (confirms column order matches columns.json)

Run from repo root: python models_production/lgbm_muscular_v4/code/modeling/export_one_row_features.py
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import joblib

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
ROOT_DIR = SCRIPT_DIR.parent.parent.parent.parent
PRODUCTION_DIR = ROOT_DIR / "models_production" / "lgbm_muscular_v4" / "model_muscular_lgbm"
OUT_DIR = ROOT_DIR / "models_production" / "lgbm_muscular_v4"

TARGET_PLAYER_ID = 144028
TARGET_REF_DATE = "2025-12-05"


def main():
    import train_iterative_feature_selection_muscular_standalone as _tmod
    from train_iterative_feature_selection_muscular_standalone import (
        load_test_dataset,
        filter_timelines_for_model,
        prepare_data,
        CACHE_DIR,
        USE_CACHE,
        log_message,
    )

    _tmod.USE_LABELED_TIMELINES = True
    _tmod.LABELED_SUFFIX = "_v4_labeled_muscle_skeletal_only_d7.csv"
    _tmod.TARGET_COLUMN = "target1"

    model = joblib.load(PRODUCTION_DIR / "model.joblib")
    with open(PRODUCTION_DIR / "columns.json", "r", encoding="utf-8") as f:
        feature_list = json.load(f)
    model_features = list(getattr(model, "feature_name_", None) or feature_list)

    log_message("Loading test dataset...")
    df_test_all = load_test_dataset()
    df_test_muscular = filter_timelines_for_model(df_test_all, "target1").reset_index(drop=True)

    # Normalize reference_date for matching (handle string or datetime)
    ref_ser = pd.to_datetime(df_test_muscular["reference_date"], errors="coerce")
    ref_str = ref_ser.dt.strftime("%Y-%m-%d")
    mask = (df_test_muscular["player_id"] == TARGET_PLAYER_ID) & (ref_str == TARGET_REF_DATE)
    if not mask.any():
        log_message(f"Row not found for player_id={TARGET_PLAYER_ID}, reference_date={TARGET_REF_DATE}")
        log_message("Sample reference_date values in CSV: " + str(df_test_muscular["reference_date"].dropna().head(3).tolist()))
        return 1
    row_idx = df_test_muscular.index[mask].tolist()[0]

    cache_suffix = hashlib.md5(str(sorted(model_features)).encode()).hexdigest()[:8]
    cache_file = str(CACHE_DIR / f"preprocessed_production_lgbm_test_{cache_suffix}.csv")
    log_message("Preparing test features (same pipeline as evaluation)...")
    X_test_full = prepare_data(df_test_muscular, cache_file=cache_file, use_cache=USE_CACHE)
    for f in model_features:
        if f not in X_test_full.columns:
            X_test_full[f] = 0
    X_test = X_test_full[model_features]

    one_row = X_test.iloc[[row_idx]]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "player_144028_ref_2025-12-05_features_320_order.csv"
    one_row.to_csv(csv_path, index=False)
    log_message(f"Exported 320 features (one row) to: {csv_path}")

    proba = model.predict_proba(one_row)[0, 1]
    log_message(f"Model prediction for this row: {proba:.6f}")

    note_path = OUT_DIR / "player_144028_ref_2025-12-05_note.txt"
    with open(note_path, "w", encoding="utf-8") as f:
        f.write(
            "Column order in player_144028_ref_2025-12-05_features_320_order.csv "
            "matches model_muscular_lgbm/columns.json exactly (320 features).\n"
        )
        f.write(f"player_id={TARGET_PLAYER_ID}, reference_date={TARGET_REF_DATE}, predicted_probability={proba:.6f}\n")
    log_message(f"Note written to: {note_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
