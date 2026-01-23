#!/usr/bin/env python3
"""Train auxiliary models for production insights (body part & severity)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Calculate paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent
ROOT_DIR = PRODUCTION_ROOT.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from production.scripts.insight_utils import (  # noqa: E402
    categorize_body_part,
    severity_label,
    SEVERITY_LABELS,
    SEVERITY_BINS,
)

DEFAULT_DAILY_FEATURES_DIR = ROOT_DIR / "daily_features_output"
DEFAULT_INJURIES_FILE = None  # Will be auto-detected from latest raw data
DEFAULT_OUTPUT_DIR = PRODUCTION_ROOT / "models" / "insights"


def get_latest_raw_data_folder(country: str = "england") -> Path:
    """Get the latest raw data folder."""
    base_dir = PRODUCTION_ROOT / "raw_data" / country.lower().replace(" ", "_")
    if not base_dir.exists():
        return None
    
    date_folders = [d for d in base_dir.iterdir() if d.is_dir() and d.name.isdigit() and len(d.name) == 8]
    if not date_folders:
        return None
    
    return max(date_folders, key=lambda x: x.name)


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
        default=None,
        help="CSV file with injury history (default: auto-detect from latest raw data).",
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
    parser.add_argument(
        "--country",
        type=str,
        default="england",
        help="Country name for auto-detecting injuries file (default: england).",
    )
    return parser.parse_args()


def load_injuries(path: Path) -> pd.DataFrame:
    """Load injuries from CSV file with new schema."""
    if not path.exists():
        raise FileNotFoundError(f"Injuries file not found: {path}")
    
    # Try semicolon separator first (Transfermarkt format), then comma
    try:
        df = pd.read_csv(path, sep=';', encoding='utf-8-sig', parse_dates=["fromDate", "untilDate"])
    except (pd.errors.ParserError, ValueError):
        df = pd.read_csv(path, sep=',', encoding='utf-8-sig', parse_dates=["fromDate", "untilDate"])
    
    df = df.rename(columns={"fromDate": "from_date"})
    
    # Filter for muscular injuries only (matching winner model)
    if "injury_class" in df.columns:
        df = df[df["injury_class"] == "muscular"]
    else:
        # Fallback: exclude non-physio if old column exists
        if "no_physio_injury" in df.columns:
            df = df[df["no_physio_injury"] != 1]
    
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
    max_daily_features_date: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, Dict[str, List[str]]]:
    """
    Build dataset from injuries and daily features.
    
    Args:
        injuries: DataFrame with injury records
        daily_features_dir: Directory containing daily features files
        max_daily_features_date: Maximum date available in daily features (inclusive).
                                 Injuries after this date + 1 day will be filtered out.
    """
    cache: Dict[int, pd.DataFrame] = {}
    feature_rows: List[pd.Series] = []
    body_part_labels: List[str] = []
    severity_labels: List[str] = []
    missing_counter = 0
    filtered_counter = 0

    # Filter injuries to only include those where we have daily features
    if max_daily_features_date is not None:
        # We need features from (injury_date - 1 day), so injury_date must be <= max_daily_features_date + 1 day
        max_injury_date = max_daily_features_date + pd.Timedelta(days=1)
        injuries_filtered = injuries[injuries["from_date"] <= max_injury_date].copy()
        filtered_counter = len(injuries) - len(injuries_filtered)
        if filtered_counter > 0:
            print(f"[FILTER] Filtered out {filtered_counter} injuries without corresponding daily features (max daily features date: {max_daily_features_date.date()})")
        injuries = injuries_filtered

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
        
        # Use existing body_part and severity columns if available
        if "body_part" in row and pd.notna(row["body_part"]) and str(row["body_part"]).strip():
            body_part = str(row["body_part"]).strip().lower()
            # Map to standard categories if needed
            if body_part not in ["lower_leg", "knee", "upper_leg", "hip", "upper_body", "head", "illness", "other"]:
                body_part = categorize_body_part(str(row.get("injury_type", "")))
        else:
            body_part = categorize_body_part(str(row.get("injury_type", "")))
        
        if "severity" in row and pd.notna(row["severity"]) and str(row["severity"]).strip():
            severity = str(row["severity"]).strip().lower()
            # Map to standard labels if needed
            if severity not in SEVERITY_LABELS:
                severity = severity_label(row.get("days"))
        else:
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
        "filtered_samples": filtered_counter,
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
    # Check if we can use stratified split (each class needs at least 2 samples)
    value_counts = y.value_counts()
    min_class_count = value_counts.min()
    
    if min_class_count < 2:
        print(f"[WARN] Some classes have fewer than 2 samples (min: {min_class_count}). Using non-stratified split.")
        stratify = None
    else:
        stratify = y
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
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


def get_max_daily_features_date(daily_features_dir: Path) -> Optional[pd.Timestamp]:
    """Get the maximum date available in daily features files."""
    max_date = None
    sample_count = 0
    
    # Sample a few player files to find the max date
    for file_path in daily_features_dir.glob("player_*_daily_features.csv"):
        try:
            df = pd.read_csv(file_path, parse_dates=["date"], nrows=1000)  # Read first 1000 rows for speed
            if "date" in df.columns and not df["date"].isna().all():
                file_max = df["date"].max()
                if max_date is None or file_max > max_date:
                    max_date = file_max
            sample_count += 1
            if sample_count >= 10:  # Sample 10 files
                break
        except Exception as e:
            continue
    
    # If we didn't find a date, try one more comprehensive check
    if max_date is None:
        print(f"[WARN] Could not determine max date from sampled files, checking all files...")
        for file_path in list(daily_features_dir.glob("player_*_daily_features.csv"))[:50]:  # Limit to 50 files
            try:
                df = pd.read_csv(file_path, parse_dates=["date"])
                if "date" in df.columns and not df["date"].isna().all():
                    file_max = df["date"].max()
                    if max_date is None or file_max > max_date:
                        max_date = file_max
            except Exception:
                continue
    
    return max_date


def main() -> None:
    args = parse_args()
    
    # Auto-detect injuries file if not provided
    if args.injuries_file is None:
        latest_raw_data = get_latest_raw_data_folder(args.country)
        if latest_raw_data is None:
            raise FileNotFoundError(
                f"Could not find latest raw data folder for {args.country}. "
                "Please specify --injuries-file explicitly."
            )
        injuries_file = latest_raw_data / "injuries_data.csv"
        if not injuries_file.exists():
            raise FileNotFoundError(
                f"Injuries file not found in {latest_raw_data}. "
                "Please specify --injuries-file explicitly."
            )
        args.injuries_file = injuries_file
        print(f"[AUTO] Using injuries file: {args.injuries_file}")
    
    injuries = load_injuries(args.injuries_file)
    print(f"[LOAD] Loaded {len(injuries)} injuries from {args.injuries_file}")
    
    # Determine max date in daily features
    print(f"[CHECK] Determining maximum date in daily features...")
    max_daily_features_date = get_max_daily_features_date(args.daily_features_dir)
    if max_daily_features_date is not None:
        print(f"[CHECK] Maximum daily features date: {max_daily_features_date.date()}")
        print(f"[CHECK] Will filter injuries to those occurring on or before {(max_daily_features_date + pd.Timedelta(days=1)).date()}")
    else:
        print(f"[WARN] Could not determine max daily features date, will process all injuries")
    
    dataset, body_part_labels, severity_labels, metadata = build_dataset(
        injuries, args.daily_features_dir, max_daily_features_date=max_daily_features_date
    )
    
    print(f"[BUILD] Built dataset with {metadata['total_samples']} samples")
    print(f"[BUILD] Missing samples: {metadata['missing_samples']}")

    preprocessor = build_preprocessor(
        metadata["categorical_columns"], metadata["numeric_columns"]
    )

    print(f"\n[TRAIN] Training body part classifier...")
    bodypart_pipeline, bodypart_meta = train_classifier(
        dataset,
        body_part_labels,
        preprocessor,
        test_size=args.test_size,
    )
    bodypart_meta.update(metadata)
    save_model(bodypart_pipeline, bodypart_meta, args.output_dir / "bodypart_classifier")
    print(f"[SAVE] Saved body part classifier to {args.output_dir / 'bodypart_classifier.pkl'}")

    print(f"\n[TRAIN] Training severity classifier...")
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
    print(f"[SAVE] Saved severity classifier to {args.output_dir / 'severity_classifier.pkl'}")

    print(
        f"\n[OK] Trained insight models with {metadata['total_samples']} samples "
        f"({metadata['missing_samples']} skipped due to missing features, "
        f"{metadata.get('filtered_samples', 0)} filtered due to date range)."
    )


if __name__ == "__main__":
    main()

