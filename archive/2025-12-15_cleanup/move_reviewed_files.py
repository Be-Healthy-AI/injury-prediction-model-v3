"""
Move all reviewed match files to a reviewed subfolder.
"""

import shutil
from pathlib import Path

def main():
    """Move all _reviewed files to a reviewed subfolder."""
    # Match data directory
    match_data_dir = Path("data_exports/transfermarkt/england/20251203/match_data")
    
    if not match_data_dir.exists():
        print(f"Error: Directory not found: {match_data_dir}")
        return
    
    # Create reviewed subfolder
    reviewed_dir = match_data_dir / "reviewed"
    reviewed_dir.mkdir(exist_ok=True)
    print(f"Created reviewed directory: {reviewed_dir}")
    
    # Find all files with "_reviewed" in the name
    reviewed_files = list(match_data_dir.glob("*_reviewed.csv"))
    
    print(f"\nFound {len(reviewed_files)} reviewed files to move")
    
    if not reviewed_files:
        print("No reviewed files found.")
        return
    
    # Move files to reviewed folder
    moved_count = 0
    for file in reviewed_files:
        try:
            destination = reviewed_dir / file.name
            shutil.move(str(file), str(destination))
            moved_count += 1
            if moved_count % 100 == 0:
                print(f"  Moved {moved_count}/{len(reviewed_files)} files...")
        except Exception as e:
            print(f"  Error moving {file.name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"COMPLETE: Moved {moved_count}/{len(reviewed_files)} files to reviewed folder")
    print(f"Location: {reviewed_dir}")
    print("="*60)

if __name__ == "__main__":
    main()



