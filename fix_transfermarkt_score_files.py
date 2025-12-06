"""
Fix match data files by swapping misaligned columns.

For matches with TransferMarkt scores (like Bundesliga), the scraper incorrectly mapped:
- minutes_played column contains TransferMarkt scores
- transfermarkt_score column contains minutes played

This script swaps these columns for files that have transfermarkt_score values.
"""

import pandas as pd
import os
from pathlib import Path

def fix_match_file(file_path: str) -> bool:
    """
    Fix a single match data file that has transfermarkt_score values.
    
    Args:
        file_path: Path to the CSV file
    
    Returns:
        True if file was fixed, False otherwise
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check if file has transfermarkt_score values
        if "transfermarkt_score" not in df.columns:
            return False
        
        # Check if there are any non-null values in transfermarkt_score
        score_mask = df["transfermarkt_score"].notna()
        
        if not score_mask.any():
            return False
        
        # Extract player_id from filename
        filename = os.path.basename(file_path)
        player_id = filename.split('_')[1]
        
        print(f"\nProcessing: {filename}")
        print(f"  Player ID: {player_id}")
        print(f"  Total rows: {len(df)}")
        print(f"  Rows with transfermarkt_score: {score_mask.sum()}")
        
        # Show BEFORE values for rows with scores
        print(f"\n  BEFORE (showing first 3 rows with scores):")
        score_before = df[score_mask].head(3)
        for idx, row in score_before.iterrows():
            print(f"    Row {idx}: competition={row.get('competition', 'N/A')}, "
                  f"minutes_played={row['minutes_played']}, "
                  f"transfermarkt_score={row['transfermarkt_score']}")
        
        # Backup original file with "_reviewed" suffix
        file_path_obj = Path(file_path)
        backup_path = file_path_obj.parent / f"{file_path_obj.stem}_reviewed{file_path_obj.suffix}"
        
        # Save backup
        df.to_csv(backup_path, index=False)
        print(f"\n  Backup saved: {os.path.basename(backup_path)}")
        
        # Define integer columns that should remain as integers
        integer_columns = [
            "goals", "assists", "own_goals",
            "yellow_cards", "second_yellow_cards", "red_cards",
            "substitutions_on", "substitutions_off", "minutes_played"
        ]
        
        # Store original integer types before operations
        original_int_types = {}
        for col in integer_columns:
            if col in df.columns:
                # Check if column is already Int64 or can be converted
                if df[col].dtype == 'Int64':
                    original_int_types[col] = 'Int64'
                elif df[col].dtype in ['int64', 'int32', 'int']:
                    original_int_types[col] = 'Int64'
        
        # Fix: Swap minutes_played and transfermarkt_score for rows with scores
        df_fixed = df.copy()
        
        # Store original values
        score_minutes = df_fixed.loc[score_mask, "minutes_played"].copy()
        score_score = df_fixed.loc[score_mask, "transfermarkt_score"].copy()
        
        # Swap the columns for rows with scores
        df_fixed.loc[score_mask, "minutes_played"] = score_score
        df_fixed.loc[score_mask, "transfermarkt_score"] = score_minutes
        
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
        
        # Apply parsing to minutes_played for rows with scores
        df_fixed.loc[score_mask, "minutes_played"] = df_fixed.loc[
            score_mask, "minutes_played"
        ].apply(parse_minutes)
        
        # Convert transfermarkt_score to float for rows with scores
        df_fixed.loc[score_mask, "transfermarkt_score"] = pd.to_numeric(
            df_fixed.loc[score_mask, "transfermarkt_score"],
            errors='coerce'
        )
        
        # Restore integer types for all integer columns
        for col in integer_columns:
            if col in df_fixed.columns:
                # Convert to nullable integer type (Int64) to preserve NaN values
                try:
                    df_fixed[col] = df_fixed[col].astype('Int64')
                except (ValueError, TypeError):
                    # If conversion fails, try converting to numeric first, then to Int64
                    df_fixed[col] = pd.to_numeric(df_fixed[col], errors='coerce').astype('Int64')
        
        # Show AFTER values for rows with scores
        print(f"\n  AFTER (showing first 3 rows with scores):")
        score_after = df_fixed[score_mask].head(3)
        for idx, row in score_after.iterrows():
            print(f"    Row {idx}: competition={row.get('competition', 'N/A')}, "
                  f"minutes_played={row['minutes_played']}, "
                  f"transfermarkt_score={row['transfermarkt_score']}")
        
        # Save fixed file with original filename
        # Use float_format=None to preserve integer formatting
        df_fixed.to_csv(file_path, index=False, float_format=None)
        print(f"\n  Fixed file saved: {os.path.basename(file_path)}")
        
        return True
        
    except Exception as e:
        print(f"  ERROR processing {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to fix all match files with transfermarkt_score values."""
    # Match data directory
    match_data_dir = Path("data_exports/transfermarkt/england/20251203/match_data")
    
    if not match_data_dir.exists():
        print(f"Error: Directory not found: {match_data_dir}")
        return
    
    # Find all CSV files
    match_files = list(match_data_dir.glob("match_*.csv"))
    print(f"Found {len(match_files)} match files")
    
    # Find files with transfermarkt_score values
    files_with_score = []
    for file in match_files:
        try:
            df = pd.read_csv(file)
            if "transfermarkt_score" in df.columns:
                if df["transfermarkt_score"].notna().any():
                    files_with_score.append(file)
        except:
            pass
    
    print(f"\nFound {len(files_with_score)} files with transfermarkt_score values")
    
    if not files_with_score:
        print("No files to fix.")
        return
    
    # Process all files with transfermarkt_score values
    print("\n" + "="*60)
    print(f"Processing all {len(files_with_score)} files with transfermarkt_score values")
    print("="*60)
    
    fixed_count = 0
    for i, file in enumerate(files_with_score, 1):
        print(f"\n[{i}/{len(files_with_score)}] ", end="")
        if fix_match_file(file):
            fixed_count += 1
    
    print("\n" + "="*60)
    print(f"COMPLETE: Fixed {fixed_count}/{len(files_with_score)} files")
    print("="*60)


if __name__ == "__main__":
    main()

