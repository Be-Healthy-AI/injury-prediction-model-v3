#!/usr/bin/env python3
"""
Helper script to generate data/artifact manifests and pipeline_steps.json
for the lgbm_muscular_v1 model bundle.

Safe to re-run; it only creates/overwrites JSON metadata files and does
NOT delete or modify any existing data/artifact files.
"""

import hashlib
import json
import os
from pathlib import Path


BASE = Path(__file__).resolve().parents[1] / "models_production" / "lgbm_muscular_v1"


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def file_entries(root: Path, rel_root: str):
    entries = []
    if not root.exists():
        return entries
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            p = Path(dirpath) / fn
            rel = Path(rel_root) / p.relative_to(root)
            entries.append(
                {
                    "path": str(rel).replace("\\", "/"),
                    "size_bytes": p.stat().st_size,
                    "sha256": hash_file(p),
                }
            )
    return entries


def main() -> None:
    metadata_dir = BASE / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # 1) Data manifest (everything under data/)
    data_root = BASE / "data"
    data_entries = file_entries(data_root, "data")
    data_manifest = {
        "model_id": "lgbm_muscular_v1",
        "entries": data_entries,
    }
    (metadata_dir / "data_manifest.json").write_text(
        json.dumps(data_manifest, indent=2), encoding="utf-8"
    )

    # 2) Artifacts manifest (model, columns, and code snapshot)
    artifacts = []
    core_artifacts = [
        ("model/model.joblib", "core model artifact"),
        ("model/columns.json", "core model artifact"),
    ]
    for rel, desc in core_artifacts:
        p = BASE / rel
        if p.exists():
            artifacts.append(
                {
                    "path": rel,
                    "size_bytes": p.stat().st_size,
                    "sha256": hash_file(p),
                    "description": desc,
                }
            )

    code_root = BASE / "code"
    if code_root.exists():
        for dirpath, _, filenames in os.walk(code_root):
            for fn in filenames:
                p = Path(dirpath) / fn
                rel = p.relative_to(BASE)
                artifacts.append(
                    {
                        "path": str(rel).replace("\\", "/"),
                        "size_bytes": p.stat().st_size,
                        "sha256": hash_file(p),
                        "description": "pipeline code snapshot",
                    }
                )

    artifacts_manifest = {
        "model_id": "lgbm_muscular_v1",
        "entries": artifacts,
    }
    (metadata_dir / "artifacts_manifest.json").write_text(
        json.dumps(artifacts_manifest, indent=2), encoding="utf-8"
    )

    # 3) Raw data manifest stub (for upstream sources not stored here)
    raw_manifest = {
        "model_id": "lgbm_muscular_v1",
        "note": "Add entries here for upstream raw datasets not stored in this bundle (e.g. external TransferMarkt exports).",
        "entries": [],
    }
    (metadata_dir / "raw_data_manifest.json").write_text(
        json.dumps(raw_manifest, indent=2), encoding="utf-8"
    )

    # 4) Pipeline steps description
    pipeline_steps = {
        "model_id": "lgbm_muscular_v1",
        "steps": [
            {
                "order": 1,
                "name": "Scrape Transfermarkt data",
                "scripts": [
                    "code/transfermarkt/transfermarkt_scraper.py",
                    "code/transfermarkt/transformers.py",
                    "code/transfermarkt/run_transfermarkt_pipeline.py",
                ],
            },
            {
                "order": 2,
                "name": "Generate daily features from Transfermarkt exports",
                "scripts": [
                    "code/daily_features/create_daily_features_v3.py",
                    "code/daily_features/create_daily_features_transfermarkt.py",
                ],
            },
            {
                "order": 3,
                "name": "Create 35-day muscular timelines (season by season, v4)",
                "scripts": [
                    "code/timelines/create_35day_timelines_season_by_season_v4.py",
                ],
            },
            {
                "order": 4,
                "name": "Train baseline models (including winner LGBM)",
                "scripts": [
                    "code/modeling/train_models_seasonal_combined.py",
                ],
            },
            {
                "order": 5,
                "name": "Decision-based evaluation (Precision@K, Recall@K)",
                "scripts": [
                    "code/modeling/evaluate_winner_model_decision_based.py",
                ],
            },
            {
                "order": 6,
                "name": "Ensemble comparison (LGBM + RF, analysis only)",
                "scripts": [
                    "code/modeling/ensemble_decision_based.py",
                ],
            },
        ],
    }
    (metadata_dir / "pipeline_steps.json").write_text(
        json.dumps(pipeline_steps, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()














