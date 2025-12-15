"""
Fix Bundesliga match data files by swapping misaligned columns.

For Bundesliga matches, the scraper incorrectly mapped:
- minutes_played column contains TransferMarkt scores
- transfermarkt_score column contains minutes played

This script swaps these columns for Bundesliga matches.
"""

import pandas as pd
import glob
import os
from pathlib import Path

def fix_bundesliga_match_file(file_path: str, test_mode: bool = True) -> bool:
    """
    Fix a single match data file that contains Bundesliga matches.
    
    Args:
        file_path: Path to the CSV file
        test_mode: If True, only process one file for testing
    
    Returns:
        True if file was fixed, False otherwise
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check if file has Bundesliga matches
        if "competition" not in df.columns:
            return False
        
        # Check for Bundesliga (various formats: "Bundesliga", "1. Bundesliga", "2. Bundesliga")
        bundesliga_mask = df["competition"].str.contains(
            "bundesliga|1\. bundesliga|2\. bundesliga",
            case=False,
            na=False,
            regex=True
        )
        
        if not bundesliga_mask.any():
            return False
        
        # Extract player_id from filename
        filename = os.path.basename(file_path)
        player_id = filename.split('_')[1]
        
        print(f"\nProcessing: {filename}")
        print(f"  Player ID: {player_id}")
        print(f"  Total rows: {len(df)}")
        print(f"  Bundesliga rows: {bundesliga_mask.sum()}")
        
        # Show BEFORE values for Bundesliga rows
        print(f"\n  BEFORE (showing first 3 Bundesliga rows):")
        bundesliga_before = df[bundesliga_mask].head(3)
        for idx, row in bundesliga_before.iterrows():
            print(f"    Row {idx}: competition={row['competition']}, "
                  f"minutes_played={row['minutes_played']}, "
                  f"transfermarkt_score={row['transfermarkt_score']}")
        
        # Backup original file with "_reviewed" suffix
        file_path_obj = Path(file_path)
        backup_path = file_path_obj.parent / f"{file_path_obj.stem}_reviewed{file_path_obj.suffix}"
        
        # Save backup
        df.to_csv(backup_path, index=False)
        print(f"\n  Backup saved: {os.path.basename(backup_path)}")
        
        # Fix: Swap minutes_played and transfermarkt_score for Bundesliga rows
        # Create copies to avoid SettingWithCopyWarning
        df_fixed = df.copy()
        
        # Store original values
        bundesliga_minutes = df_fixed.loc[bundesliga_mask, "minutes_played"].copy()
        bundesliga_score = df_fixed.loc[bundesliga_mask, "transfermarkt_score"].copy()
        
        # Swap the columns for Bundesliga rows
        df_fixed.loc[bundesliga_mask, "minutes_played"] = bundesliga_score
        df_fixed.loc[bundesliga_mask, "transfermarkt_score"] = bundesliga_minutes
        
        # Convert minutes_played to integer (it should be integer, not float)
        # Handle any string values like "90'" or "90+3'"
        def parse_minutes(val):
            if pd.isna(val):
                return pd.NA
            if isinstance(val, str):
                # Remove quotes and extract number
                val = val.replace("'", "").replace('"', '').strip()
                # Handle "90+3" format
                if "+" in val:
                    parts = val.split("+")
                    if len(parts) == 2:
                        try:
                            return int(parts[0].strip()) + int(parts[1].strip())
                        except:
                            pass
                try:
                    return int(float(val))
                except:
                    return pd.NA
            try:
                return int(float(val))
            except:
                return pd.NA
        
        # Apply parsing to minutes_played for Bundesliga rows
        df_fixed.loc[bundesliga_mask, "minutes_played"] = df_fixed.loc[
            bundesliga_mask, "minutes_played"
        ].apply(parse_minutes)
        
        # Convert transfermarkt_score to float for Bundesliga rows
        df_fixed.loc[bundesliga_mask, "transfermarkt_score"] = pd.to_numeric(
            df_fixed.loc[bundesliga_mask, "transfermarkt_score"],
            errors='coerce'
        )
        
        # Show AFTER values for Bundesliga rows
        print(f"\n  AFTER (showing first 3 Bundesliga rows):")
        bundesliga_after = df_fixed[bundesliga_mask].head(3)
        for idx, row in bundesliga_after.iterrows():
            print(f"    Row {idx}: competition={row['competition']}, "
                  f"minutes_played={row['minutes_played']}, "
                  f"transfermarkt_score={row['transfermarkt_score']}")
        
        # Save fixed file with original filename
        df_fixed.to_csv(file_path, index=False)
        print(f"\n  Fixed file saved: {os.path.basename(file_path)}")
        
        return True
        
    except Exception as e:
        print(f"  ERROR processing {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to fix all Bundesliga match files."""
    # Match data directory
    match_data_dir = Path("data_exports/transfermarkt/england/20251203/match_data")
    
    if not match_data_dir.exists():
        print(f"Error: Directory not found: {match_data_dir}")
        return
    
    # Find all CSV files
    match_files = list(match_data_dir.glob("match_*.csv"))
    print(f"Found {len(match_files)} match files")
    
    # Find files with Bundesliga matches
    bundesliga_files = []
    for file in match_files:
        try:
            df = pd.read_csv(file, nrows=1)  # Just check headers
            if "competition" in df.columns:
                # Quick check: read a sample to see if Bundesliga exists
                df_sample = pd.read_csv(file)
                if df_sample["competition"].str.contains(
                    "bundesliga|1\. bundesliga|2\. bundesliga",
                    case=False,
                    na=False,
                    regex=True
                ).any():
                    bundesliga_files.append(file)
        except:
            pass
    
    print(f"\nFound {len(bundesliga_files)} files with Bundesliga matches")
    
    if not bundesliga_files:
        print("No files to fix.")
        return
    
    # TEST MODE: Process only the first file
    print("\n" + "="*60)
    print("TEST MODE: Processing first file only")
    print("="*60)
    
    test_file = bundesliga_files[0]
    
    # Extract player_id from filename for reporting
    filename = os.path.basename(test_file)
    player_id = filename.split('_')[1]
    
    print(f"\nSelected player ID for testing: {player_id}")
    print(f"File: {filename}")
    
    success = fix_bundesliga_match_file(test_file, test_mode=True)
    
    if success:
        print("\n" + "="*60)
        print("TEST SUCCESSFUL!")
        print("="*60)
        print(f"\nTested on player ID: {player_id}")
        print(f"To process all {len(bundesliga_files)} files, change test_mode=False in main()")
    else:
        print("\n" + "="*60)
        print("TEST FAILED - Please review the error above")
        print("="*60)


if __name__ == "__main__":
    main()



