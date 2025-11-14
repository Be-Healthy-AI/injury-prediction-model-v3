#!/usr/bin/env python3
"""
Project Cleanup Script
Removes old, duplicate, and temporary files to maintain a well-organized project.
"""
import sys
import io
# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import shutil
from pathlib import Path
from typing import List

def delete_files(file_paths: List[Path], description: str) -> int:
    """Delete a list of files and return count of deleted files."""
    deleted = 0
    for file_path in file_paths:
        if file_path.exists():
            try:
                file_path.unlink()
                deleted += 1
                print(f"  [OK] Deleted: {file_path}")
            except Exception as e:
                print(f"  [ERROR] Error deleting {file_path}: {e}")
    if deleted > 0:
        print(f"\n{description}: {deleted} file(s) deleted\n")
    return deleted

def delete_directories(dir_paths: List[Path], description: str) -> int:
    """Delete directories and return count of deleted directories."""
    deleted = 0
    for dir_path in dir_paths:
        if dir_path.exists() and dir_path.is_dir():
            try:
                shutil.rmtree(dir_path)
                deleted += 1
                print(f"  [OK] Deleted directory: {dir_path}")
            except Exception as e:
                print(f"  [ERROR] Error deleting {dir_path}: {e}")
    if deleted > 0:
        print(f"\n{description}: {deleted} directory(ies) deleted\n")
    return deleted

def main():
    print("=" * 70)
    print("PROJECT CLEANUP")
    print("=" * 70)
    print("\nThis script will delete:")
    print("  - Old model files (replaced by final versions)")
    print("  - Old backtest files (replaced by organized 2025_45d structure)")
    print("  - Cache and temporary files")
    print("  - Test/verification scripts")
    print("  - Old dashboard design options (keeping only option 3)")
    print("\n" + "=" * 70)
    
    # Skip prompt if --yes flag is provided
    if "--yes" not in sys.argv:
        response = input("\nProceed with cleanup? (yes/no): ").strip().lower()
        if response != 'yes':
            print("Cleanup cancelled.")
            return
    else:
        print("\nProceeding with cleanup (--yes flag provided)...")
    
    root = Path(__file__).resolve().parent.parent
    total_deleted = 0
    
    # 1. Old Model Files
    print("\n" + "=" * 70)
    print("1. DELETING OLD MODEL FILES")
    print("=" * 70)
    old_models = [
        root / "models" / "model_v3_gradient_boosting_100percent.pkl",
        root / "models" / "model_v3_random_forest_100percent.pkl",
        root / "models" / "model_v3_gradient_boosting_validated.pkl",
        root / "models" / "model_v3_random_forest_validated.pkl",
        root / "models" / "model_v3_gradient_boosting_optimized.pkl",
        root / "models" / "model_v3_random_forest_optimized.pkl",
        root / "models" / "model_v3_gb_100percent_training_columns.json",
        root / "models" / "model_v3_rf_100percent_training_columns.json",
        root / "models" / "model_v3_gb_validated_training_columns.json",
        root / "models" / "model_v3_rf_validated_training_columns.json",
        root / "models" / "model_v3_gb_optimized_training_columns.json",
        root / "models" / "model_v3_rf_optimized_training_columns.json",
        root / "models" / "model_v3_gb_optimized_params.json",
        root / "models" / "model_v3_rf_optimized_params.json",
    ]
    total_deleted += delete_files(old_models, "Old model files")
    
    # 2. Old Backtest Files
    print("=" * 70)
    print("2. DELETING OLD BACKTEST FILES")
    print("=" * 70)
    
    # Old daily features
    old_daily_features = [
        root / "backtests" / "daily_features" / "player_200512_daily_features_20240401_20240531.csv",
        root / "backtests" / "daily_features" / "player_258027_daily_features_20250901_20251029.csv",
        root / "backtests" / "daily_features" / "player_452607_daily_features_20250101_20250208.csv",
        root / "backtests" / "daily_features" / "player_699592_daily_features_20250101_20250208.csv",
        root / "backtests" / "daily_features" / "player_8198_daily_features_20250401_20250511.csv",
    ]
    total_deleted += delete_files(old_daily_features, "Old daily features files")
    
    # Old timelines
    old_timelines = [
        root / "backtests" / "timelines" / "player_200512_timelines_20240401_20240531.csv",
        root / "backtests" / "timelines" / "player_258027_timelines_20250901_20251029.csv",
        root / "backtests" / "timelines" / "player_452607_timelines_20250101_20250208.csv",
        root / "backtests" / "timelines" / "player_699592_timelines_20250101_20250208.csv",
        root / "backtests" / "timelines" / "player_8198_timelines_20250101_20250208.csv",
    ]
    total_deleted += delete_files(old_timelines, "Old timeline files")
    
    # Old visualizations
    old_visualizations = [
        root / "backtests" / "visualizations" / "player_200512_probabilities_20240401_20240531.png",
        root / "backtests" / "visualizations" / "player_258027_probabilities_20250901_20251029.png",
        root / "backtests" / "visualizations" / "player_452607_probabilities_20250101_20250208.png",
        root / "backtests" / "visualizations" / "player_699592_probabilities_20250101_20250208.png",
        root / "backtests" / "visualizations" / "player_8198_probabilities_20250401_20250511.png",
        root / "backtests" / "visualizations" / "predictions_summary.csv",
        root / "backtests" / "visualizations" / "predictions_summary.md",
    ]
    total_deleted += delete_files(old_visualizations, "Old visualization files")
    
    # Duplicate prediction directories (keep only 2025_45d structure)
    print("=" * 70)
    print("3. DELETING DUPLICATE PREDICTION FILES")
    print("=" * 70)
    duplicate_pred_dirs = [
        root / "backtests" / "predictions" / "ensemble",
        root / "backtests" / "predictions" / "gradient_boosting",
        root / "backtests" / "predictions" / "random_forest",
    ]
    total_deleted += delete_directories(duplicate_pred_dirs, "Duplicate prediction directories")
    
    # 3. Cache and Temporary Files
    print("=" * 70)
    print("4. DELETING CACHE AND TEMPORARY FILES")
    print("=" * 70)
    cache_files = [
        root / "data_cache_v3.pkl",
        root / "feature_generation.log",
    ]
    total_deleted += delete_files(cache_files, "Cache and log files")
    
    # 4. Test/Verification Scripts
    print("=" * 70)
    print("5. DELETING TEST/VERIFICATION SCRIPTS")
    print("=" * 70)
    test_scripts = [
        root / "verify_enhancements.py",
    ]
    total_deleted += delete_files(test_scripts, "Test scripts")
    
    # 5. Old Dashboard Options
    print("=" * 70)
    print("6. DELETING OLD DASHBOARD DESIGN OPTIONS")
    print("=" * 70)
    old_dashboards = [
        root / "backtests" / "visualizations" / "dashboard_options" / "option_1_horizontal_split.png",
        root / "backtests" / "visualizations" / "dashboard_options" / "option_2_vertical_split.png",
        root / "backtests" / "visualizations" / "dashboard_options" / "option_4_card_design.png",
    ]
    total_deleted += delete_files(old_dashboards, "Old dashboard design options")
    
    # Final Summary
    print("=" * 70)
    print("CLEANUP SUMMARY")
    print("=" * 70)
    print(f"Total files/directories deleted: {total_deleted}")
    print("\n[OK] Cleanup completed successfully!")
    print("\nNote: All deleted files were either:")
    print("  - Replaced by newer versions")
    print("  - Duplicates of organized structure")
    print("  - Temporary/cache files that can be regenerated")
    print("=" * 70)

if __name__ == "__main__":
    main()

