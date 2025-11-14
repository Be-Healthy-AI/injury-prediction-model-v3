#!/usr/bin/env python3
"""Train auxiliary models for backtesting insights (body part & severity)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.backtests.insight_utils import (  # noqa: E402
    categorize_body_part,
    severity_label,
    SEVERITY_LABELS,
    SEVERITY_BINS,
)


DEFAULT_DAILY_FEATURES_DIR = Path("daily_features_output")
DEFAULT_INJURIES_FILE = Path("original_data") / "20251106_injuries_data.xlsx"
DEFAULT_OUTPUT_DIR = Path("models") / "insights"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--daily-features-dir",
        type=Path,
        default=DEFAULT_DAILY_FEATURES_DIR,
        help="Directory containing full daily feature CSVs (one per player).",
    )
    parser.add_argument(
        "--injuries-file",
        type=Path,
        default=DEFAULT_INJURIES_FILE,
        help="Excel file with injury history.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store trained pipelines and metadata.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data used for validation.",
    )
    return parser.parse_args()


def load_injuries(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, parse_dates=["fromDate", "untilDate"])
    df = df.rename(columns={"fromDate": "from_date"})
    df = df[df["no_physio_injury"] != 1]  # Exclude non-physio absences
    df = df.dropna(subset=["player_id", "from_date"])
    return df


def collect_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    # Explicit exclusions
    drop_categorical = {"player_name"}
    drop_numeric = set()

    categorical_cols = [col for col in categorical_cols if col not in drop_categorical]
    numeric_cols = [col for col in numeric_cols if col not in drop_numeric]
    return categorical_cols, numeric_cols


def build_dataset(
    injuries: pd.DataFrame,
    daily_features_dir: Path,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, Dict[str, List[str]]]:
    cache: Dict[int, pd.DataFrame] = {}
    feature_rows: List[pd.Series] = []
    body_part_labels: List[str] = []
    severity_labels: List[str] = []
    missing_counter = 0

    for _, row in injuries.iterrows():
        player_id = int(row["player_id"])
        injury_date: pd.Timestamp = row["from_date"]
        target_date = (injury_date - pd.Timedelta(days=1)).normalize()

        if player_id not in cache:
            file_path = daily_features_dir / f"player_{player_id}_daily_features.csv"
            if not file_path.exists():
                missing_counter += 1
                continue
            player_df = pd.read_csv(file_path)
            if "date" not in player_df.columns:
                missing_counter += 1
                continue
            player_df["date"] = pd.to_datetime(player_df["date"], errors="coerce")
            player_df.dropna(subset=["date"], inplace=True)
            cache[player_id] = player_df
        else:
            player_df = cache[player_id]

        feature_row = player_df.loc[player_df["date"] == target_date]
        if feature_row.empty:
            missing_counter += 1
            continue

        feature_row = feature_row.iloc[0]
        body_part = categorize_body_part(str(row["injury_type"]))
        severity = severity_label(row.get("days"))

        if not body_part or severity is None:
            missing_counter += 1
            continue

        filtered_row = feature_row.drop(labels=["date"])
        feature_rows.append(filtered_row)
        body_part_labels.append(body_part)
        severity_labels.append(severity)

    if not feature_rows:
        raise RuntimeError("Insufficient data to build insight datasets.")

    dataset = pd.DataFrame(feature_rows)
    categorical_cols, numeric_cols = collect_feature_columns(dataset)

    metadata = {
        "categorical_columns": categorical_cols,
        "numeric_columns": numeric_cols,
        "missing_samples": missing_counter,
        "total_samples": len(feature_rows),
    }
    return dataset, pd.Series(body_part_labels), pd.Series(severity_labels), metadata


def build_preprocessor(categorical_cols: List[str], numeric_cols: List[str]) -> ColumnTransformer:
    transformers = []
    if categorical_cols:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            )
        )
    if numeric_cols:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )

    if not transformers:
        raise ValueError("No feature columns available for preprocessing.")

    return ColumnTransformer(transformers, remainder="drop")


def train_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
    test_size: float,
    random_state: int = 42,
) -> Tuple[Pipeline, Dict[str, object]]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                GradientBoostingClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=5,
                    subsample=0.9,
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return pipeline, {
        "report": report,
        "class_labels": pipeline.classes_.tolist(),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }


def save_model(pipeline: Pipeline, metadata: Dict[str, object], path_prefix: Path) -> None:
    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path_prefix.with_suffix(".pkl"))
    path_prefix.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    injuries = load_injuries(args.injuries_file)
    dataset, body_part_labels, severity_labels, metadata = build_dataset(
        injuries, args.daily_features_dir
    )

    preprocessor = build_preprocessor(
        metadata["categorical_columns"], metadata["numeric_columns"]
    )

    bodypart_pipeline, bodypart_meta = train_classifier(
        dataset,
        body_part_labels,
        preprocessor,
        test_size=args.test_size,
    )
    bodypart_meta.update(metadata)
    save_model(bodypart_pipeline, bodypart_meta, args.output_dir / "bodypart_classifier")

    severity_pipeline, severity_meta = train_classifier(
        dataset,
        severity_labels,
        preprocessor,
        test_size=args.test_size,
    )
    severity_meta.update(metadata)
    severity_meta["severity_labels"] = SEVERITY_LABELS
    severity_meta["severity_bins"] = SEVERITY_BINS
    save_model(severity_pipeline, severity_meta, args.output_dir / "severity_classifier")

    print(
        f"[OK] Trained insight models with {metadata['total_samples']} samples "
        f"({metadata['missing_samples']} skipped)."
    )


if __name__ == "__main__":
    main()