#!/usr/bin/env python3
"""
Generate encoding_schema.json for model_skeletal from columns.json and a timeline CSV.

Use this when you have an existing model_skeletal deployment but no encoding_schema.json
(e.g. before the schema was added to the training export). Run from repo root.

Usage:
  python models_production/lgbm_muscular_v4/code/modeling/generate_skeletal_encoding_schema.py
  python models_production/lgbm_muscular_v4/code/modeling/generate_skeletal_encoding_schema.py --timelines path/to/timelines.csv
"""

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent.parent.parent  # repo root
TEST_DIR = ROOT_DIR / "models_production" / "lgbm_muscular_v4" / "data" / "timelines" / "test"
MODEL_SKELETAL_DIR = ROOT_DIR / "models_production" / "lgbm_muscular_v4" / "model_skeletal"


def main():
    parser = argparse.ArgumentParser(description="Generate encoding_schema.json for model_skeletal")
    parser.add_argument(
        "--timelines",
        type=Path,
        default=None,
        help="Path to a timeline CSV (same columns as training). Default: test folder d7 or v4_muscular_test.",
    )
    args = parser.parse_args()

    cols_path = MODEL_SKELETAL_DIR / "columns.json"
    if not cols_path.exists():
        print(f"ERROR: {cols_path} not found", file=sys.stderr)
        return 1

    with open(cols_path, "r", encoding="utf-8") as f:
        columns_data = json.load(f)
    model_columns = columns_data if isinstance(columns_data, list) else columns_data.get("features", columns_data)
    model_columns = list(model_columns)

    if args.timelines is not None:
        timelines_path = Path(args.timelines)
        if not timelines_path.exists():
            print(f"ERROR: Timelines file not found: {timelines_path}", file=sys.stderr)
            return 1
    else:
        # Try d7 then v4_muscular_test
        d7 = TEST_DIR / "timelines_35day_season_2025_2026_v4_labeled_muscle_skeletal_only_d7.csv"
        v4_test = TEST_DIR / "timelines_35day_season_2025_2026_v4_muscular_test.csv"
        if d7.exists():
            timelines_path = d7
        elif v4_test.exists():
            timelines_path = v4_test
        else:
            print(
                f"ERROR: No default timelines found. Tried {d7} and {v4_test}. Use --timelines PATH.",
                file=sys.stderr,
            )
            return 1

    import pandas as pd

    df = pd.read_csv(timelines_path, nrows=1000, low_memory=False)
    meta = ['player_id', 'reference_date', 'date', 'player_name', 'target1', 'target2', 'target', 'has_minimum_activity']
    feature_cols = [c for c in df.columns if c not in meta]
    schema_df = df[feature_cols]
    categorical_base_names = schema_df.select_dtypes(include=['object']).columns.tolist()

    encoding_schema = {}
    for base in categorical_base_names:
        cols = [c for c in model_columns if c.startswith(base + "_")]
        if cols:
            encoding_schema[base] = sorted(cols)

    out_path = MODEL_SKELETAL_DIR / "encoding_schema.json"
    MODEL_SKELETAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(encoding_schema, f, indent=2)
    print(f"Wrote {out_path} ({len(encoding_schema)} categoricals)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
