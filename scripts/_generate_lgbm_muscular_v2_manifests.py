#!/usr/bin/env python3
"""
Helper script to generate data/artifact manifests and pipeline_steps.json
for the lgbm_muscular_v2 model bundle.

Safe to re-run; it only creates/overwrites JSON metadata files and does
NOT delete or modify any existing data/artifact files.
"""

import hashlib
import json
import os
from pathlib import Path


BASE = Path(__file__).resolve().parents[1] / "models_production" / "lgbm_muscular_v2"


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
        "model_id": "lgbm_muscular_v2",
        "entries": data_entries,
        "note": "Timeline data references v1 bundle. For v2, training used all 10% timelines including the new 2025_2026_10pc file."
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
        "model_id": "lgbm_muscular_v2",
        "entries": artifacts,
    }
    (metadata_dir / "artifacts_manifest.json").write_text(
        json.dumps(artifacts_manifest, indent=2), encoding="utf-8"
    )

    # 3) Raw data manifest stub (for upstream sources not stored here)
    raw_manifest = {
        "model_id": "lgbm_muscular_v2",
        "note": "Raw data (Transfermarkt exports, daily features) are stored separately. See v1 bundle for data lineage.",
        "entries": []
    }
    (metadata_dir / "raw_data_manifest.json").write_text(
        json.dumps(raw_manifest, indent=2), encoding="utf-8"
    )

    # 4) Pipeline steps
    pipeline_steps = {
        "model_id": "lgbm_muscular_v2",
        "steps": [
            {
                "order": 1,
                "name": "Scrape Transfermarkt data",
                "scripts": [
                    "code/transfermarkt/transfermarkt_scraper.py",
                    "code/transfermarkt/transformers.py",
                    "code/transfermarkt/run_transfermarkt_pipeline.py"
                ],
                "note": "Same as v1 - see v1 bundle for code snapshots"
            },
            {
                "order": 2,
                "name": "Generate daily features from Transfermarkt exports",
                "scripts": [
                    "code/daily_features/create_daily_features_v3.py",
                    "code/daily_features/create_daily_features_transfermarkt.py"
                ],
                "note": "Same as v1 - see v1 bundle for code snapshots"
            },
            {
                "order": 3,
                "name": "Create 35-day muscular timelines (season by season, v4)",
                "scripts": [
                    "code/timelines/create_35day_timelines_season_by_season_v4.py"
                ],
                "note": "Same as v1 - see v1 bundle for code snapshots. For v2, also created 10% ratio timeline for 2025-2026."
            },
            {
                "order": 4,
                "name": "Create 10% target ratio timeline for 2025-2026",
                "scripts": [
                    "scripts/create_10pc_timelines_2025_2026.py"
                ],
                "note": "New step for v2: balanced 2025-2026 to 10% target ratio to match other training seasons"
            },
            {
                "order": 5,
                "name": "Train LGBM v2 on ALL seasons (including 2025-2026)",
                "scripts": [
                    "code/modeling/train_lgbm_all_seasons_cv.py"
                ],
                "note": "v2-specific: trains on all available data without external hold-out"
            },
            {
                "order": 6,
                "name": "Decision-based evaluation (Precision@K, Recall@K)",
                "scripts": [
                    "code/modeling/evaluate_winner_model_decision_based.py"
                ],
                "note": "Evaluates on natural 2025-2026 (for monitoring, not true generalization)"
            }
        ]
    }
    (metadata_dir / "pipeline_steps.json").write_text(
        json.dumps(pipeline_steps, indent=2), encoding="utf-8"
    )

    print(f"Generated manifests for lgbm_muscular_v2")
    print(f"   - artifacts_manifest.json: {len(artifacts)} entries")
    print(f"   - data_manifest.json: {len(data_entries)} entries")
    print(f"   - pipeline_steps.json: {len(pipeline_steps['steps'])} steps")


if __name__ == "__main__":
    main()

