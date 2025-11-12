#!/usr/bin/env python3
import os
import json
import argparse
import joblib
import pandas as pd
from datetime import timedelta
from create_35day_timelines_v3 import (
    create_windowed_features_vectorized, get_static_features
)


def build_single_timeline(daily_df: pd.DataFrame, player_id: int, player_name: str, reference_date: str):
    ref_date = pd.to_datetime(reference_date)
    start_date = ref_date - timedelta(days=34)
    windowed = create_windowed_features_vectorized(daily_df, start_date, ref_date)
    if windowed is None:
        return None
    ref_row_df = daily_df.loc[daily_df["date"] == ref_date]
    if ref_row_df.empty:
        return None
    ref_row = ref_row_df.iloc[0]
    timeline = {
        "player_id": player_id,
        "player_name": player_name,
        "reference_date": ref_date.strftime("%Y-%m-%d"),
    }
    for feature in get_static_features()[3:]:
        timeline[feature] = ref_row.get(feature, None)
    timeline.update(windowed)
    return timeline


def preprocess_like_training(df: pd.DataFrame, training_columns):
    X = df.copy()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    for col in categorical_cols:
        X[col] = X[col].fillna("Unknown")
        dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
        X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
    for col in numeric_cols:
        if col in X.columns:
            X[col] = X[col].fillna(0)
    X = X.reindex(columns=training_columns, fill_value=0)
    return X


def predict_for_dates(daily_df: pd.DataFrame, player_id: int, player_name: str, start: str, end: str,
                      model_path: str, cols_path: str) -> pd.DataFrame:
    training_columns = json.load(open(cols_path))
    model = joblib.load(model_path)
    dates = pd.date_range(pd.to_datetime(start), pd.to_datetime(end), freq="D")
    rows = []
    for d in dates:
        t = build_single_timeline(daily_df, player_id, player_name, d)
        if t:
            rows.append(t)
    if not rows:
        return pd.DataFrame(columns=["reference_date", "player_id", "player_name", "injury_risk"])
    df_rows = pd.DataFrame(rows)
    X = preprocess_like_training(df_rows, training_columns)
    proba = model.predict_proba(X)[:, 1]
    out = pd.DataFrame({
        "reference_date": df_rows["reference_date"],
        "player_id": df_rows["player_id"],
        "player_name": df_rows["player_name"],
        "injury_risk": proba,
    })
    return out.sort_values("reference_date")


def main():
    ap = argparse.ArgumentParser(description="Predict daily injury risk with RF V2 for a player over a date range")
    ap.add_argument("--player_csv", required=True, help="Path to player's daily features CSV")
    ap.add_argument("--player_id", type=int, required=True)
    ap.add_argument("--player_name", required=True)
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--model_path", default="../models/model_v3_random_forest_100percent.pkl")
    ap.add_argument("--cols_path", default="../models/model_v3_rf_100percent_training_columns.json")
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    daily_df = pd.read_csv(args.player_csv, parse_dates=["date"])  # must include daily features and a 'date' column
    preds = predict_for_dates(
        daily_df, args.player_id, args.player_name,
        args.start, args.end, args.model_path, args.cols_path
    )
    if args.out_csv:
        preds.to_csv(args.out_csv, index=False)
        print(f"Saved predictions to {args.out_csv}")
    else:
        print(preds.head(20).to_string(index=False))


if __name__ == "__main__":
    main()


