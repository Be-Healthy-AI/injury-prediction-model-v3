#!/usr/bin/env python3
"""
Apply the high-precision RF+GB ensemble rule to backtest predictions.

Rule: alert when (rf_prob >= RF_THRESHOLD) AND (gb_prob >= GB_THRESHOLD).
"""

import json
from pathlib import Path
from datetime import datetime

import pandas as pd

RF_THRESHOLD = 0.38
GB_THRESHOLD = 0.48
START_DATE = datetime(2025, 7, 1)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "backtests" / "config" / "players_2025_45d.json"
RF_DIR = ROOT / "backtests" / "predictions" / "2025_45d" / "random_forest"
GB_DIR = ROOT / "backtests" / "predictions" / "2025_45d" / "gradient_boosting"
OUT_DIR = ROOT / "backtests" / "predictions" / "2025_45d" / "ensemble_precision"
REPORT_PATH = ROOT / "backtests" / "reports" / "ensemble_precision_post_20250701.md"


def load_predictions(entry_id: str, model_dir: Path, suffix: str, column_name: str) -> pd.DataFrame:
    file_path = model_dir / f"{entry_id}_{suffix}"
    if not file_path.exists():
        raise FileNotFoundError(f"Missing prediction file: {file_path}")
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    return df[["reference_date", "injury_probability"]].rename(
        columns={"injury_probability": column_name}
    )


def main():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing config: {CONFIG_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    summaries = []

    for entry in config["entries"]:
        injury_date = datetime.fromisoformat(entry["injury_date"])
        if injury_date < START_DATE:
            continue

        entry_id = entry["entry_id"]
        try:
            rf_df = load_predictions(entry_id, RF_DIR, "random_forest_predictions.csv", "rf_prob")
            gb_df = load_predictions(entry_id, GB_DIR, "gradient_boosting_predictions.csv", "gb_prob")
        except FileNotFoundError as exc:
            print(f"⚠️  {exc}")
            continue

        merged = (
            rf_df.merge(gb_df, on="reference_date", how="inner")
            .sort_values("reference_date")
            .reset_index(drop=True)
        )

        merged["alert"] = (merged["rf_prob"] >= RF_THRESHOLD) & (merged["gb_prob"] >= GB_THRESHOLD)
        merged["injury_date"] = entry["injury_date"]
        merged["player_id"] = entry["player_id"]
        merged["player_name"] = entry.get("player_name", "")
        merged["days_to_injury"] = (
            pd.to_datetime(merged["injury_date"]) - pd.to_datetime(merged["reference_date"])
        ).dt.days

        out_file = OUT_DIR / f"{entry_id}_ensemble_precision.csv"
        merged.to_csv(out_file, index=False, encoding="utf-8-sig")

        alerts = merged[merged["alert"]]
        summaries.append(
            {
                "entry_id": entry_id,
                "player_id": entry["player_id"],
                "injury_date": entry["injury_date"],
                "injury_type": entry.get("injury_type", ""),
                "alerts": int(alerts.shape[0]),
                "first_alert": alerts["reference_date"].iloc[0] if not alerts.empty else None,
                "lead_days": int(alerts["days_to_injury"].iloc[0]) if not alerts.empty else None,
                "max_rf_prob": merged["rf_prob"].max(),
                "max_gb_prob": merged["gb_prob"].max(),
            }
        )

    # Build markdown summary
    lines = [
        "# RF+GB Precision Ensemble (Post 2025-07-01)",
        "",
        f"- RF threshold: **{RF_THRESHOLD:.2f}**",
        f"- GB threshold: **{GB_THRESHOLD:.2f}**",
        f"- Entries processed: **{len(summaries)}**",
        "",
        "| Entry | Injury Date | Alerts | First Alert | Lead Days | Max RF | Max GB |",
        "|-------|-------------|--------|-------------|-----------|--------|--------|",
    ]
    for summary in summaries:
        first_alert = summary["first_alert"] or "-"
        lead = summary["lead_days"] if summary["lead_days"] is not None else "-"
        lines.append(
            f"| {summary['entry_id']} | {summary['injury_date']} | {summary['alerts']} | "
            f"{first_alert} | {lead} | {summary['max_rf_prob']:.3f} | {summary['max_gb_prob']:.3f} |"
        )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote per-entry outputs to {OUT_DIR}")
    print(f"Summary saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()


