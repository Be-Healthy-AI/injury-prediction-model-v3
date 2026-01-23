#!/usr/bin/env python3
"""
Clean up timeline files in train directory, keeping only the 5 files used for training.
"""

import sys
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
import glob

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V3_ROOT = PROJECT_ROOT / "models_production" / "lgbm_muscular_v3"
TRAIN_DIR = V3_ROOT / "data" / "timelines" / "train"

# Files to keep (5 training files used for V3-natural-filtered-excl-2023-2024)
FILES_TO_KEEP = {
    "timelines_35day_season_2018_2019_v4_muscular.csv",
    "timelines_35day_season_2019_2020_v4_muscular.csv",
    "timelines_35day_season_2020_2021_v4_muscular.csv",
    "timelines_35day_season_2024_2025_v4_muscular.csv",
    "timelines_35day_season_2025_2026_v4_muscular.csv",
}

def main():
    print("=" * 80)
    print("CLEANING UP TIMELINE FILES")
    print("=" * 80)
    print(f"\nKeeping only the 5 files used for training:")
    for f in sorted(FILES_TO_KEEP):
        print(f"  - {f}")
    
    if not TRAIN_DIR.exists():
        print(f"\n⚠️  Train directory not found: {TRAIN_DIR}")
        return
    
    # Get all CSV files
    all_files = list(TRAIN_DIR.glob("*.csv"))
    
    files_to_delete = []
    files_kept = []
    
    for file_path in all_files:
        file_name = file_path.name
        if file_name in FILES_TO_KEEP:
            files_kept.append(file_name)
        else:
            files_to_delete.append(file_path)
    
    print(f"\n📊 Summary:")
    print(f"   Total files: {len(all_files)}")
    print(f"   Files to keep: {len(files_kept)}")
    print(f"   Files to delete: {len(files_to_delete)}")
    
    if files_to_delete:
        print(f"\n🗑️  Deleting {len(files_to_delete)} files...")
        for file_path in files_to_delete:
            try:
                file_path.unlink()
                print(f"   ✅ Deleted: {file_path.name}")
            except Exception as e:
                print(f"   ❌ Error deleting {file_path.name}: {e}")
    else:
        print("\n✅ No files to delete - all files are needed")
    
    print(f"\n✅ Cleanup complete. {len(files_kept)} files kept in train directory.")

if __name__ == "__main__":
    main()

