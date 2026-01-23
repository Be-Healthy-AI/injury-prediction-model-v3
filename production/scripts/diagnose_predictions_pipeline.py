#!/usr/bin/env python3
"""
Diagnostic script to check for potential issues in the predictions pipeline.

This script performs comprehensive checks to identify bugs or data issues
that could affect prediction accuracy before concluding that feature engineering
is the root cause of false negatives.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
import numpy as np

# Calculate paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent
ROOT_DIR = PRODUCTION_ROOT.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from production.scripts.preprocessing_lgbm_v2 import prepare_data, align_features_to_model


def check_nan_values_in_timelines(timelines_file: Path) -> Dict:
    """Check for NaN values in days_since_last_* features."""
    print("\n" + "=" * 80)
    print("CHECK 1: NaN Values in Timelines")
    print("=" * 80)
    
    if not timelines_file.exists():
        return {"error": f"Timelines file not found: {timelines_file}"}
    
    print(f"Loading timelines from: {timelines_file}")
    timelines_df = pd.read_csv(timelines_file, encoding='utf-8-sig', low_memory=False)
    print(f"Loaded {len(timelines_df):,} timelines")
    
    # Find all days_since_last_* features
    days_features = [col for col in timelines_df.columns if 'days_since_last' in col.lower()]
    
    if not days_features:
        print("  [WARN] No days_since_last_* features found in timelines")
        return {"error": "No days_since_last_* features found"}
    
    print(f"\nFound {len(days_features)} days_since_last_* features")
    
    results = {
        "total_features": len(days_features),
        "features_with_nan": [],
        "total_nan_count": 0,
        "features_with_zero": [],
        "total_zero_count": 0
    }
    
    for feat in days_features:
        nan_count = timelines_df[feat].isna().sum()
        zero_count = (timelines_df[feat] == 0).sum()
        
        if nan_count > 0:
            results["features_with_nan"].append({
                "feature": feat,
                "nan_count": int(nan_count),
                "nan_percentage": float((nan_count / len(timelines_df)) * 100),
                "min_value": float(timelines_df[feat].min()) if timelines_df[feat].notna().any() else None,
                "max_value": float(timelines_df[feat].max()) if timelines_df[feat].notna().any() else None,
                "mean_value": float(timelines_df[feat].mean()) if timelines_df[feat].notna().any() else None
            })
            results["total_nan_count"] += nan_count
            print(f"  [ISSUE] {feat}: {nan_count:,} NaN values ({nan_count/len(timelines_df)*100:.2f}%)")
        
        if zero_count > 0:
            results["features_with_zero"].append({
                "feature": feat,
                "zero_count": int(zero_count),
                "zero_percentage": float((zero_count / len(timelines_df)) * 100)
            })
            results["total_zero_count"] += zero_count
            if zero_count < len(timelines_df) * 0.01:  # Less than 1% zeros is probably OK
                print(f"  [INFO] {feat}: {zero_count:,} zero values ({zero_count/len(timelines_df)*100:.2f}%)")
            else:
                print(f"  [WARN] {feat}: {zero_count:,} zero values ({zero_count/len(timelines_df)*100:.2f}%)")
    
    if results["features_with_nan"]:
        print(f"\n[ISSUE] Found NaN values in {len(results['features_with_nan'])} features")
        print(f"        Total NaN count: {results['total_nan_count']:,}")
    else:
        print(f"\n[OK] No NaN values found in days_since_last_* features")
    
    return results


def check_missing_features(timelines_file: Path, model_dir: Path) -> Dict:
    """Check for features expected by model but missing from timelines."""
    print("\n" + "=" * 80)
    print("CHECK 2: Missing Features")
    print("=" * 80)
    
    # Load model columns
    columns_path = model_dir / "columns.json"
    if not columns_path.exists():
        return {"error": f"Model columns file not found: {columns_path}"}
    
    with open(columns_path, 'r', encoding='utf-8') as f:
        model_columns = json.load(f)
    
    print(f"Model expects {len(model_columns)} features")
    
    # Load timelines
    if not timelines_file.exists():
        return {"error": f"Timelines file not found: {timelines_file}"}
    
    timelines_df = pd.read_csv(timelines_file, encoding='utf-8-sig', low_memory=False)
    print(f"Timelines has {len(timelines_df.columns)} columns")
    
    # Prepare data (this will show what features are available after preprocessing)
    print("\nPreprocessing timelines to get actual feature names...")
    try:
        X, y = prepare_data(timelines_df, cache_file=None, use_cache=False)
        print(f"After preprocessing: {len(X.columns)} features")
        
        # Check alignment
        X_aligned = align_features_to_model(X, model_columns)
        
        # Find missing features
        missing_features = [col for col in model_columns if col not in X.columns]
        extra_features = [col for col in X.columns if col not in model_columns]
        
        results = {
            "model_features": len(model_columns),
            "preprocessed_features": len(X.columns),
            "aligned_features": len(X_aligned.columns),
            "missing_features": missing_features,
            "missing_count": len(missing_features),
            "extra_features": extra_features,
            "extra_count": len(extra_features)
        }
        
        if missing_features:
            print(f"\n[ISSUE] Found {len(missing_features)} missing features:")
            for feat in missing_features[:20]:  # Show first 20
                print(f"  - {feat}")
            if len(missing_features) > 20:
                print(f"  ... and {len(missing_features) - 20} more")
        else:
            print(f"\n[OK] All model features are present in timelines")
        
        if extra_features:
            print(f"\n[INFO] Found {len(extra_features)} extra features in timelines (will be dropped)")
        
        return results
        
    except Exception as e:
        return {"error": f"Error during preprocessing: {e}"}


def check_feature_values_in_false_negatives(fn_file: Path) -> Dict:
    """Check feature values in false negatives to verify they're reasonable."""
    print("\n" + "=" * 80)
    print("CHECK 3: Feature Values in False Negatives")
    print("=" * 80)
    
    if not fn_file.exists():
        return {"error": f"False negatives file not found: {fn_file}"}
    
    print(f"Loading false negatives from: {fn_file}")
    fn_df = pd.read_csv(fn_file, encoding='utf-8-sig', low_memory=False)
    print(f"Loaded {len(fn_df):,} false negative cases")
    
    # Find days_since_last_* features in the top features
    days_features_in_top = []
    for i in range(1, 11):
        feat_col = f'top_feature_{i}_name'
        value_col = f'top_feature_{i}_value'
        if feat_col in fn_df.columns:
            for idx, row in fn_df.iterrows():
                feat_name = row[feat_col]
                if feat_name and 'days_since_last' in str(feat_name).lower():
                    days_features_in_top.append({
                        "feature": str(feat_name),
                        "value": row[value_col] if pd.notna(row[value_col]) else None,
                        "shap": row.get(f'top_feature_{i}_shap', None)
                    })
    
    # Also check if we have direct access to days_since_last_* columns
    days_features_direct = [col for col in fn_df.columns if 'days_since_last' in col.lower() and 'week_' in col.lower()]
    
    results = {
        "total_cases": len(fn_df),
        "days_features_in_top": len(days_features_in_top),
        "days_features_direct": len(days_features_direct),
        "feature_statistics": {}
    }
    
    # Analyze top feature values
    if days_features_in_top:
        print(f"\nFound {len(days_features_in_top)} days_since_last_* features in top 10 features")
        feature_values = {}
        for item in days_features_in_top:
            feat_name = item["feature"]
            if feat_name not in feature_values:
                feature_values[feat_name] = []
            if item["value"] is not None:
                feature_values[feat_name].append(item["value"])
        
        for feat_name, values in feature_values.items():
            if values:
                results["feature_statistics"][feat_name] = {
                    "count": len(values),
                    "min": float(min(values)),
                    "max": float(max(values)),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "zeros": sum(1 for v in values if v == 0),
                    "very_large": sum(1 for v in values if v > 1000)
                }
                stats = results["feature_statistics"][feat_name]
                print(f"\n  {feat_name}:")
                print(f"    Count: {stats['count']}")
                print(f"    Range: {stats['min']:.1f} - {stats['max']:.1f}")
                print(f"    Mean: {stats['mean']:.1f}, Median: {stats['median']:.1f}")
                print(f"    Zeros: {stats['zeros']} ({stats['zeros']/stats['count']*100:.1f}%)")
                print(f"    Very large (>1000): {stats['very_large']} ({stats['very_large']/stats['count']*100:.1f}%)")
                
                if stats['zeros'] > 0:
                    print(f"    [WARN] Found {stats['zeros']} zero values - could indicate missing value handling issue")
    
    # Check direct features if available
    if days_features_direct:
        print(f"\nFound {len(days_features_direct)} days_since_last_* features directly in CSV")
        for feat in days_features_direct[:5]:  # Show first 5
            if feat in fn_df.columns:
                values = fn_df[feat].dropna()
                if len(values) > 0:
                    zero_count = (values == 0).sum()
                    print(f"  {feat}: min={values.min():.1f}, max={values.max():.1f}, mean={values.mean():.1f}, zeros={zero_count}")
    
    return results


def check_preprocessing_output(timelines_file: Path, cache_file: Path = None) -> Dict:
    """Check preprocessing output to verify feature values."""
    print("\n" + "=" * 80)
    print("CHECK 4: Preprocessing Output")
    print("=" * 80)
    
    if not timelines_file.exists():
        return {"error": f"Timelines file not found: {timelines_file}"}
    
    print(f"Loading timelines from: {timelines_file}")
    timelines_df = pd.read_csv(timelines_file, encoding='utf-8-sig', low_memory=False)
    print(f"Loaded {len(timelines_df):,} timelines")
    
    # Preprocess without cache
    print("\nPreprocessing data (without cache)...")
    try:
        X, y = prepare_data(timelines_df, cache_file=None, use_cache=False)
        print(f"Preprocessed to {len(X.columns)} features")
        
        # Check for days_since_last_* features
        days_features = [col for col in X.columns if 'days_since_last' in col.lower()]
        
        results = {
            "total_features": len(X.columns),
            "days_features_count": len(days_features),
            "days_features_with_issues": []
        }
        
        if days_features:
            print(f"\nFound {len(days_features)} days_since_last_* features after preprocessing")
            
            for feat in days_features[:10]:  # Check first 10
                values = X[feat].dropna()
                if len(values) > 0:
                    zero_count = (values == 0).sum()
                    nan_count = X[feat].isna().sum()
                    min_val = float(values.min())
                    max_val = float(values.max())
                    mean_val = float(values.mean())
                    
                    issues = []
                    if zero_count > len(X) * 0.01:  # More than 1% zeros
                        issues.append(f"High zero count: {zero_count} ({zero_count/len(X)*100:.2f}%)")
                    if nan_count > 0:
                        issues.append(f"NaN values: {nan_count}")
                    if min_val < 0:
                        issues.append(f"Negative values: min={min_val}")
                    
                    if issues:
                        results["days_features_with_issues"].append({
                            "feature": feat,
                            "issues": issues,
                            "zero_count": int(zero_count),
                            "nan_count": int(nan_count),
                            "min": min_val,
                            "max": max_val,
                            "mean": mean_val
                        })
                        print(f"  [ISSUE] {feat}: {', '.join(issues)}")
                    else:
                        print(f"  [OK] {feat}: min={min_val:.1f}, max={max_val:.1f}, mean={mean_val:.1f}, zeros={zero_count}")
        else:
            print("\n[WARN] No days_since_last_* features found after preprocessing")
            print("       (They may have been renamed or encoded)")
        
        return results
        
    except Exception as e:
        return {"error": f"Error during preprocessing: {e}"}


def check_cache_consistency(timelines_file: Path, cache_file: Path) -> Dict:
    """Check cache consistency."""
    print("\n" + "=" * 80)
    print("CHECK 5: Cache Consistency")
    print("=" * 80)
    
    if not timelines_file.exists():
        return {"error": f"Timelines file not found: {timelines_file}"}
    
    if not cache_file.exists():
        print(f"[INFO] Cache file does not exist: {cache_file}")
        return {"cache_exists": False}
    
    import os
    timelines_mtime = os.path.getmtime(timelines_file)
    cache_mtime = os.path.getmtime(cache_file)
    
    results = {
        "cache_exists": True,
        "timelines_mtime": timelines_mtime,
        "cache_mtime": cache_mtime,
        "cache_is_stale": cache_mtime < timelines_mtime
    }
    
    if cache_mtime < timelines_mtime:
        print(f"[ISSUE] Cache is stale (timelines modified after cache)")
        print(f"        Timelines: {pd.Timestamp.fromtimestamp(timelines_mtime)}")
        print(f"        Cache: {pd.Timestamp.fromtimestamp(cache_mtime)}")
    else:
        print(f"[OK] Cache is up to date")
        print(f"     Timelines: {pd.Timestamp.fromtimestamp(timelines_mtime)}")
        print(f"     Cache: {pd.Timestamp.fromtimestamp(cache_mtime)}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Diagnose potential issues in predictions pipeline'
    )
    parser.add_argument(
        '--country',
        type=str,
        default='England',
        help='Country name (default: England)'
    )
    parser.add_argument(
        '--club',
        type=str,
        default='Chelsea FC',
        help='Club name (default: Chelsea FC)'
    )
    parser.add_argument(
        '--timelines-file',
        type=str,
        default=None,
        help='Path to timelines CSV file (default: auto-detect)'
    )
    parser.add_argument(
        '--model-version',
        type=str,
        choices=['v2', 'v3'],
        default='v3',
        help='Model version to check (default: v3)'
    )
    parser.add_argument(
        '--fn-file',
        type=str,
        default=None,
        help='Path to false negatives CSV file (default: auto-detect)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file for results (optional)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("PREDICTIONS PIPELINE DIAGNOSTIC")
    print("=" * 80)
    print(f"Country: {args.country}")
    print(f"Club: {args.club}")
    print(f"Model version: {args.model_version}")
    print()
    
    # Get paths
    club_path = PRODUCTION_ROOT / "deployments" / args.country / args.club
    timelines_dir = club_path / "timelines"
    predictions_dir = club_path / "predictions"
    model_dir = PRODUCTION_ROOT / "models" / f"lgbm_muscular_{args.model_version}"
    
    # Determine timelines file
    if args.timelines_file:
        timelines_file = Path(args.timelines_file)
    else:
        timelines_file = timelines_dir / "timelines_35day_season_2025_2026_v4_muscular.csv"
    
    # Determine cache file
    cache_dir = PRODUCTION_ROOT / "cache"
    cache_file = cache_dir / f"preprocessed_{args.model_version}_{timelines_file.stem}.csv"
    
    # Determine false negatives file
    if args.fn_file:
        fn_file = Path(args.fn_file)
    else:
        # Try to find latest false negatives file
        england_dir = PRODUCTION_ROOT / "deployments" / "England"
        fn_files = list(england_dir.glob("false_negative_analysis_*_false_negatives.csv"))
        if fn_files:
            fn_file = max(fn_files, key=lambda x: x.stat().st_mtime)
        else:
            fn_file = None
    
    # Run all checks
    all_results = {}
    
    # Check 1: NaN values
    all_results["check1_nan_values"] = check_nan_values_in_timelines(timelines_file)
    
    # Check 2: Missing features
    all_results["check2_missing_features"] = check_missing_features(timelines_file, model_dir)
    
    # Check 3: Feature values in false negatives
    if fn_file and fn_file.exists():
        all_results["check3_fn_feature_values"] = check_feature_values_in_false_negatives(fn_file)
    else:
        print("\n" + "=" * 80)
        print("CHECK 3: Feature Values in False Negatives")
        print("=" * 80)
        print(f"[SKIP] False negatives file not found: {fn_file}")
        all_results["check3_fn_feature_values"] = {"skipped": True}
    
    # Check 4: Preprocessing output
    all_results["check4_preprocessing"] = check_preprocessing_output(timelines_file, cache_file)
    
    # Check 5: Cache consistency
    all_results["check5_cache"] = check_cache_consistency(timelines_file, cache_file)
    
    # Summary
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    
    issues_found = []
    
    if "features_with_nan" in all_results.get("check1_nan_values", {}) and all_results["check1_nan_values"]["features_with_nan"]:
        issues_found.append(f"NaN values found in {len(all_results['check1_nan_values']['features_with_nan'])} features")
    
    if "missing_count" in all_results.get("check2_missing_features", {}) and all_results["check2_missing_features"]["missing_count"] > 0:
        issues_found.append(f"{all_results['check2_missing_features']['missing_count']} features missing from timelines")
    
    if "days_features_with_issues" in all_results.get("check4_preprocessing", {}) and all_results["check4_preprocessing"]["days_features_with_issues"]:
        issues_found.append(f"Issues found in {len(all_results['check4_preprocessing']['days_features_with_issues'])} preprocessed features")
    
    if all_results.get("check5_cache", {}).get("cache_is_stale", False):
        issues_found.append("Cache is stale")
    
    if issues_found:
        print("\n[ISSUES FOUND]:")
        for issue in issues_found:
            print(f"  - {issue}")
    else:
        print("\n[OK] No major issues found in the checks performed")
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        import json
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_results = convert_to_serializable(all_results)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        print(f"\n[SAVED] Results saved to: {output_path}")
    
    print("\n" + "=" * 80)
    
    return 0


if __name__ == "__main__":
    exit(main())

