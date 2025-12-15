"""
Count how many match files have values in the transfermarkt_score column.
"""

import pandas as pd
from pathlib import Path

def main():
    """Count files with transfermarkt_score values."""
    match_data_dir = Path("data_exports/transfermarkt/england/20251203/match_data")
    
    if not match_data_dir.exists():
        print(f"Error: Directory not found: {match_data_dir}")
        return
    
    match_files = list(match_data_dir.glob("match_*.csv"))
    files_with_score = 0
    
    for i, file in enumerate(match_files):
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(match_files)} files...")
        
        try:
            df = pd.read_csv(file)
            if "transfermarkt_score" in df.columns:
                if df["transfermarkt_score"].notna().any():
                    files_with_score += 1
        except:
            pass
    
    print(f"\nFiles with transfermarkt_score values: {files_with_score}")

if __name__ == "__main__":
    main()



