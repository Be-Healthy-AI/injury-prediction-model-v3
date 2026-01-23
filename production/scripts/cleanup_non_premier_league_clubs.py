#!/usr/bin/env python3
"""
Cleanup script to remove non-Premier League club folders from deployments.

This script removes all club folders except:
- Chelsea FC (existing production deployment)
- All other Premier League clubs
"""

from pathlib import Path
import shutil
import argparse

# Calculate paths
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent
DEPLOYMENTS_DIR = PRODUCTION_ROOT / "deployments" / "England"

# Premier League clubs (2025-2026 season)
PREMIER_LEAGUE_CLUBS = {
    "Arsenal FC", "Aston Villa", "AFC Bournemouth", "Brentford FC", 
    "Brighton & Hove Albion", "Burnley FC", "Chelsea FC", "Crystal Palace",
    "Everton FC", "Fulham FC", "Ipswich Town", "Leicester City",
    "Liverpool FC", "Luton Town", "Manchester City", "Manchester United",
    "Newcastle United", "Nottingham Forest", "Sheffield United", 
    "Southampton FC", "Tottenham Hotspur", "West Ham United",
    "Wolverhampton Wanderers"
}

def main():
    parser = argparse.ArgumentParser(description='Cleanup non-Premier League club folders')
    parser.add_argument('--force', action='store_true', 
                       help='Skip confirmation prompt and proceed with deletion')
    args = parser.parse_args()
    
    if not DEPLOYMENTS_DIR.exists():
        print(f"ERROR: Deployments directory not found: {DEPLOYMENTS_DIR}")
        return 1
    
    # Get all club folders
    all_folders = [d for d in DEPLOYMENTS_DIR.iterdir() if d.is_dir()]
    
    # Identify folders to keep and remove
    folders_to_keep = []
    folders_to_remove = []
    
    for folder in all_folders:
        club_name = folder.name
        if club_name in PREMIER_LEAGUE_CLUBS:
            folders_to_keep.append(club_name)
        else:
            folders_to_remove.append(club_name)
    
    print("=" * 80)
    print("CLEANUP: Non-Premier League Club Folders")
    print("=" * 80)
    print(f"\nTotal folders found: {len(all_folders)}")
    print(f"Folders to KEEP (Premier League): {len(folders_to_keep)}")
    print(f"Folders to REMOVE (non-Premier League): {len(folders_to_remove)}")
    
    if folders_to_keep:
        print(f"\n[KEEP] Keeping {len(folders_to_keep)} Premier League clubs:")
        for club in sorted(folders_to_keep):
            print(f"   - {club}")
    
    if folders_to_remove:
        print(f"\n[REMOVE] Removing {len(folders_to_remove)} non-Premier League clubs:")
        # Show first 20, then count
        for club in sorted(folders_to_remove)[:20]:
            print(f"   - {club}")
        if len(folders_to_remove) > 20:
            print(f"   ... and {len(folders_to_remove) - 20} more")
    
    # Confirm before deletion (unless --force is used)
    if not args.force:
        print("\n" + "=" * 80)
        try:
            response = input(f"\n[WARNING] Are you sure you want to DELETE {len(folders_to_remove)} folders? (yes/no): ")
            if response.lower() != 'yes':
                print("[CANCELLED] Cleanup cancelled.")
                return 0
        except (EOFError, KeyboardInterrupt):
            print("\n[CANCELLED] Cleanup cancelled (no input provided).")
            print("Use --force flag to skip confirmation.")
            return 0
    else:
        print("\n[FORCE MODE] Skipping confirmation prompt...")
    
    # Perform deletion
    print("\nRemoving folders...")
    removed_count = 0
    error_count = 0
    
    for club_name in folders_to_remove:
        folder_path = DEPLOYMENTS_DIR / club_name
        try:
            shutil.rmtree(folder_path)
            removed_count += 1
            if removed_count % 10 == 0:
                print(f"   Removed {removed_count}/{len(folders_to_remove)} folders...")
        except Exception as e:
            print(f"   ERROR: Failed to remove {club_name}: {e}")
            error_count += 1
    
    print("\n" + "=" * 80)
    print("CLEANUP COMPLETE")
    print("=" * 80)
    print(f"[OK] Successfully removed: {removed_count} folders")
    if error_count > 0:
        print(f"[ERROR] Errors: {error_count} folders")
    print(f"[OK] Kept: {len(folders_to_keep)} Premier League club folders")
    
    return 0 if error_count == 0 else 1

if __name__ == '__main__':
    exit(main())

