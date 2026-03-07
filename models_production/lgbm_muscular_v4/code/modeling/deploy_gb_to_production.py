#!/usr/bin/env python3
"""
Deploy one of the production models (muscular LGBM, muscular GB, MSU LGBM, skeletal LGBM)
to its dedicated production directory. Each model gets its own folder.

Layout:
  model_muscular_lgbm/   <- Muscular LGBM (copy non-suffixed artifacts then deploy)
  model_muscular_gb/     <- Muscular GB
  model_msu_lgbm/        <- MSU (Exp 11)
  model_skeletal/        <- Skeletal LGBM

Each directory contains: model.joblib, columns.json, MODEL_METADATA.json.

Usage:
  python deploy_gb_to_production.py --model muscular_lgbm
  python deploy_gb_to_production.py --model muscular_gb
  python deploy_gb_to_production.py --model msu_lgbm
  python deploy_gb_to_production.py --model skeletal
  python deploy_gb_to_production.py --algorithm gb    # alias: deploys muscular_gb
  python deploy_gb_to_production.py --algorithm lgbm  # alias: deploys muscular_lgbm
"""
import argparse
import json
import shutil
import time
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent.parent.parent
MODELS_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'models'
PRODUCTION_BASE = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4'

# Three-model layout: each model has its own directory
PRODUCTION_DIRS = {
    'muscular_lgbm': PRODUCTION_BASE / 'model_muscular_lgbm',
    'muscular_gb': PRODUCTION_BASE / 'model_muscular_gb',
    'msu_lgbm': PRODUCTION_BASE / 'model_msu_lgbm',
    'skeletal': PRODUCTION_BASE / 'model_skeletal',
}

# Legacy single slot (optional; deploy script now targets the three dirs above)
LEGACY_PRODUCTION_DIR = PRODUCTION_BASE / 'model'


def get_paths(target):
    """
    Return (MODEL_SOURCE, FEATURES_SOURCE, RESULTS_FILE, performance_metrics_key) for the given target.
    performance_metrics_key: 'model1_muscular' or 'model2_skeletal' for reading from results file.
    """
    if target == 'muscular_lgbm':
        # Prefer Exp12 artifacts (iteration 16, 500 features, below HP) when present
        exp12_model = MODELS_DIR / 'lgbm_muscular_best_iteration_exp12.joblib'
        exp12_features = MODELS_DIR / 'lgbm_muscular_best_iteration_features_exp12.json'
        exp12_results = MODELS_DIR / 'iterative_feature_selection_results_muscular_exp12.json'
        if exp12_model.exists() and exp12_features.exists():
            return (exp12_model, exp12_features, exp12_results, 'model1_muscular')
        return (
            MODELS_DIR / 'lgbm_muscular_best_iteration.joblib',
            MODELS_DIR / 'lgbm_muscular_best_iteration_features.json',
            MODELS_DIR / 'iterative_feature_selection_results_muscular.json',
            'model1_muscular',
        )
    if target == 'msu_lgbm':
        return (
            MODELS_DIR / 'lgbm_muscular_best_iteration_exp11.joblib',
            MODELS_DIR / 'lgbm_muscular_best_iteration_features_exp11.json',
            MODELS_DIR / 'iterative_feature_selection_results_muscular_exp11.json',
            'model1_muscular',
        )
    if target == 'muscular_gb':
        return (
            MODELS_DIR / 'lgbm_muscular_best_iteration_gb.joblib',
            MODELS_DIR / 'lgbm_muscular_best_iteration_features_gb.json',
            MODELS_DIR / 'iterative_feature_selection_results_muscular_gb.json',
            'model1_muscular',
        )
    if target == 'skeletal':
        # Prefer Exp12 artifacts (iteration 21, 600 features, below HP) when present
        exp12_model = MODELS_DIR / 'lgbm_skeletal_best_iteration_exp12.joblib'
        exp12_features = MODELS_DIR / 'lgbm_skeletal_best_iteration_features_exp12.json'
        exp12_results = MODELS_DIR / 'iterative_feature_selection_results_skeletal_exp12.json'
        if exp12_model.exists() and exp12_features.exists():
            return (exp12_model, exp12_features, exp12_results, 'model2_skeletal')
        return (
            MODELS_DIR / 'lgbm_skeletal_best_iteration.joblib',
            MODELS_DIR / 'lgbm_skeletal_best_iteration_features.json',
            MODELS_DIR / 'iterative_feature_selection_results_skeletal.json',
            'model2_skeletal',
        )
    raise ValueError(f"Unknown target: {target}")


def main():
    parser = argparse.ArgumentParser(
        description="Deploy one of the three models (muscular LGBM, muscular GB, skeletal) to production."
    )
    parser.add_argument(
        "--model",
        choices=["muscular_lgbm", "muscular_gb", "msu_lgbm", "skeletal"],
        default=None,
        help="Which model to deploy (default: muscular_gb for backward compatibility)",
    )
    parser.add_argument(
        "--algorithm",
        choices=["gb", "lgbm"],
        default=None,
        help="Legacy: deploy muscular model (gb -> muscular_gb, lgbm -> muscular_lgbm)",
    )
    parser.add_argument("--wait", action="store_true", help="Wait for export files to appear before deploying")
    parser.add_argument("--wait-minutes", type=int, default=120, help="Max minutes to wait when using --wait (default 120)")
    args = parser.parse_args()

    # Resolve target: --model takes precedence, then --algorithm
    if args.model is not None:
        target = args.model
    elif args.algorithm is not None:
        target = "muscular_gb" if args.algorithm == "gb" else "muscular_lgbm"
    else:
        target = "muscular_gb"

    MODEL_SOURCE, FEATURES_SOURCE, RESULTS_FILE, metrics_key = get_paths(target)
    PRODUCTION_DIR = PRODUCTION_DIRS[target]

    if args.wait:
        deadline = time.time() + args.wait_minutes * 60
        print(f"Waiting up to {args.wait_minutes} minutes for export files ({target})...")
        while time.time() < deadline:
            if MODEL_SOURCE.exists() and FEATURES_SOURCE.exists():
                print("Export files found. Deploying...")
                break
            time.sleep(30)
        else:
            print(f"ERROR: Export files not found after {args.wait_minutes} minutes.")
            return 1

    if not MODEL_SOURCE.exists():
        print(f"ERROR: Model not found: {MODEL_SOURCE}")
        if target == "skeletal":
            print("Run first: python train_iterative_feature_selection_skeletal_standalone.py --export-best")
        elif target == "msu_lgbm":
            print("Run first: python train_iterative_feature_selection_muscular_standalone.py --exp11-data --test-negatives-before 2025-11-01 --only-iteration <N> --no-resume --train-on-full-data")
        else:
            print(f"Run first: python train_iterative_feature_selection_muscular_standalone.py --algorithm {'gb' if target == 'muscular_gb' else 'lgbm'} --export-best")
        return 1
    if not FEATURES_SOURCE.exists():
        print(f"ERROR: Features JSON not found: {FEATURES_SOURCE}")
        return 1

    with open(FEATURES_SOURCE, "r", encoding="utf-8") as f:
        meta = json.load(f)

    features = meta["features"]
    n_features = meta["n_features"]
    iteration = meta.get("iteration")
    algorithm = meta.get("algorithm", "lgbm")
    combined_score_val = meta.get("combined_score_val")
    combined_score_test = meta.get("combined_score_test")
    combined_score = meta.get("combined_score", combined_score_test)

    # Performance metrics: prefer features JSON (skeletal has it there), else results file
    performance_metrics = meta.get("performance_metrics")
    if performance_metrics is None and RESULTS_FILE.exists():
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            results = json.load(f)
        for it in results.get("iterations", []):
            if it.get("iteration") == iteration:
                model_data = it.get(metrics_key, {})
                performance_metrics = {"train": {}, "val": {}, "test": {}}
                for split in ("train", "val", "test"):
                    if split in model_data:
                        performance_metrics[split] = {
                            k: v for k, v in model_data[split].items() if k != "confusion_matrix"
                        }
                        performance_metrics[split]["confusion_matrix"] = model_data[split].get("confusion_matrix")
                break
    if performance_metrics is None:
        performance_metrics = {"train": {}, "val": {}, "test": {}}

    PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)

    # Copy model
    dest_model = PRODUCTION_DIR / "model.joblib"
    shutil.copy2(MODEL_SOURCE, dest_model)
    print(f"Copied model to {dest_model}")

    # Write columns.json
    columns_path = PRODUCTION_DIR / "columns.json"
    with open(columns_path, "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2)
    print(f"Wrote {columns_path} ({n_features} features)")

    # Build MODEL_METADATA.json
    creation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if target == "muscular_lgbm":
        model_version = "V4_Muscular_LGBM_500feat"
        model_name = "V4 Muscular Injury Prediction - LGBM (500 features, below_strong, 100% data)"
        algorithm_display = "LightGBM"
        experiment = "LGBM iterative feature selection, below_strong, trained on 100%% data"
    elif target == "msu_lgbm":
        model_version = "V4_MSU_LGBM_Exp11"
        model_name = "V4 MSU Injury Prediction - LGBM (Exp 11, target_msu, 100% data)"
        algorithm_display = "LightGBM"
        experiment = "Exp 11: MSU (muscular/skeletal/unknown [D-7,D-1]), Exp10 negative filter, trained on 100%% data"
    elif target == "muscular_gb":
        model_version = "V4_Muscular_GB_350feat"
        model_name = "V4 Muscular Injury Prediction - GB (350 features, below, 100% data)"
        algorithm_display = "GradientBoosting"
        experiment = "GB iterative feature selection, below preset, iteration 7, trained on 100%% data"
    else:  # skeletal
        model_version = f"V4_Skeletal_LGBM_{n_features}feat"
        model_name = f"V4 Skeletal Injury Prediction - LGBM ({n_features} features, iteration {iteration}, below HP, Exp12)"
        algorithm_display = "LightGBM"
        experiment = "Skeletal Exp12 iterative feature selection, below HP, 100% data, best iteration by test combined score"

    metadata = {
        "model_version": model_version,
        "model_name": model_name,
        "target": target,
        "creation_date": creation_date,
        "training_configuration": {
            "algorithm": algorithm_display,
            "feature_selection": {
                "method": "iterative_feature_selection",
                "iteration": iteration,
                "n_features": n_features,
                "combined_score": combined_score,
                "optimize_on": meta.get("optimize_on", "test"),
                "experiment": experiment,
            },
            "combined_score_val": combined_score_val,
            "combined_score_test": combined_score_test,
        },
        "performance_metrics": performance_metrics,
        "deployment_info": {
            "model_file": "model.joblib",
            "columns_file": "columns.json",
            "feature_count": n_features,
            "deployment_date": creation_date,
            "production_dir": str(PRODUCTION_DIR.name),
        },
    }
    metadata_path = PRODUCTION_DIR / "MODEL_METADATA.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote {metadata_path}")

    print(f"\nProduction deployment complete for {target}. Model is in {PRODUCTION_DIR}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
