#!/usr/bin/env python3
"""
Test script to extract match data scraping logic and test direct column mapping.
Outputs a CSV file with the same structure as match data files.

This script:
1. Fetches raw match data for a test player (using same logic as pipeline)
2. Applies direct column mapping (Unnamed: 8 -> goals, etc.)
3. Uses the transformer to get final format
4. Outputs CSV file for review

Does NOT touch the main pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.data_collection.transformers import transform_matches

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/129.0.0.0 Safari/537.36"
)

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Language": "en-US,en;q=0.9",
}


def fetch_raw_match_table_with_mapping(
    player_id: int,
    season: int,
    base_url: str = "https://www.transfermarkt.co.uk",
) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Fetch the raw match log table and apply direct column mapping.
    
    Returns:
        (df, competition_name) or (None, None) if not found
    """
    url = (
        f"{base_url}/-/leistungsdatendetails/spieler/{player_id}"
        f"/saison/{season}/verein/0/liga/0/wettbewerb//pos/0/trainer_id/0/plus/1"
    )
    print(f"Fetching match data for player {player_id}, season {season}...")
    print(f"URL: {url}")

    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    tables = soup.select("table")
    print(f"Found {len(tables)} tables on page")

    match_tables = []
    
    for idx, table in enumerate(tables):
        headers = table.select("thead th")
        header_texts = [h.get_text(strip=True) for h in headers]
        if "Matchday" in header_texts:
            print(f"Using table {idx} as match log table")
            
            # Extract competition name
            competition_name = None
            parent = table.find_parent(['div', 'section'])
            if parent:
                heading = parent.find(['h2', 'h3', 'h4'])
                if heading:
                    competition_name = heading.get_text(strip=True)
            
            if not competition_name:
                caption = table.find('caption')
                if caption:
                    competition_name = caption.get_text(strip=True)
            
            if not competition_name:
                competition_name = "Unknown Competition"
            
            # Read table with pandas
            dfs = pd.read_html(str(table), flavor="bs4")
            df = dfs[0] if dfs else pd.DataFrame()
            
            if df.empty:
                continue
            
            # Remove summary rows (contain "Squad:", "Starting eleven:", etc.)
            mask = pd.Series([True] * len(df))
            for col in df.columns:
                if df[col].dtype == 'object':
                    mask = mask & ~df[col].astype(str).str.contains('Squad:', na=False)
                    mask = mask & ~df[col].astype(str).str.contains('Starting eleven:', na=False)
            
            df = df[mask].copy()
            
            if df.empty:
                continue
            
            # Extract home and away teams from the .1 columns
            if "Home team.1" in df.columns:
                df["Home team"] = df["Home team.1"]
            if "Away team.1" in df.columns:
                df["Away team"] = df["Away team.1"]
            
            # Clean team names
            for col in ["Home team", "Away team"]:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.replace(r'\s*\(\d+\.\)\s*', '', regex=True)
                    df[col] = df[col].replace('nan', pd.NA)
            
            # Map position column
            if "Pos." in df.columns and "Position" not in df.columns:
                df["Position"] = df["Pos."]
            
            # ============================================================
            # APPLY DIRECT COLUMN MAPPING (as specified by user)
            # ============================================================
            mapping = {
                "Unnamed: 8": "Goals",
                "Unnamed: 9": "Assists",
                "Unnamed: 10": "Own goals",
                "Unnamed: 11": "Yellow cards",
                "Unnamed: 12": "Second yellow cards",
                "Unnamed: 13": "Red cards",
                "Unnamed: 14": "Sub on",
                "Unnamed: 15": "Sub off",
                "Unnamed: 16": "Minutes played",
                "Unnamed: 17": "TM-Whoscored grade",
            }
            
            # Apply mappings
            for source_col, target_col in mapping.items():
                if source_col in df.columns:
                    # Clean the data: remove invalid texts
                    series = df[source_col].copy()
                    series_str = series.astype(str)
                    
                    # Filter out invalid texts
                    invalid_texts = [
                        'on the bench',
                        'not in squad',
                        'suspended',
                        'injured',
                        'Squad:',
                        'Starting eleven:',
                    ]
                    
                    for invalid in invalid_texts:
                        mask = series_str.str.contains(invalid, case=False, na=False)
                        series.loc[mask] = pd.NA
                    
                    # For substitution times and minutes, handle time formats like "88'"
                    if target_col in ["Sub on", "Sub off", "Minutes played"]:
                        def extract_time(val):
                            if pd.isna(val) or str(val) == 'nan':
                                return pd.NA
                            val_str = str(val).strip()
                            # Remove quotes and extract number
                            val_str = val_str.replace("'", "").replace('"', '').strip()
                            # If it contains comma, take first part
                            if ',' in val_str:
                                val_str = val_str.split(',')[0].strip()
                            try:
                                num = float(val_str)
                                # For substitution times, return as string with quote
                                if target_col in ["Sub on", "Sub off"]:
                                    return f"{int(num)}'"
                                # For minutes played, return as integer
                                elif target_col == "Minutes played":
                                    return int(num)
                            except (ValueError, TypeError):
                                return pd.NA
                            return pd.NA
                        
                        series = series.apply(extract_time)
                    
                    # Only map if target column doesn't already exist
                    if target_col not in df.columns:
                        df[target_col] = series
                        print(f"  Mapped {source_col} -> {target_col}")
                    else:
                        print(f"  Warning: {target_col} already exists, skipping {source_col}")
                else:
                    print(f"  Warning: {source_col} not found in DataFrame")
            
            # Add season column (format: "2023/24" for season 2024)
            df["Season"] = f"{season-1}/{str(season)[-2:]}"
            
            # Add competition column
            df["Competition"] = competition_name
            
            match_tables.append(df)
    
    if not match_tables:
        print("No match tables found.")
        return None, None
    
    # Combine all competition tables
    combined = pd.concat(match_tables, ignore_index=True)
    
    # Get competition name from first table (or use first found)
    competition_name = match_tables[0]["Competition"].iloc[0] if not match_tables[0].empty else "Unknown"
    
    return combined, competition_name


def main():
    """Test with player 324503, season 2024 (or player 85941, season 2024)"""
    # Test with the same player as the reference file
    player_id = 85941
    season = 2024  # This will become "2023/24" in the output
    
    print("=" * 80)
    print("Match Data Column Mapping Test")
    print("=" * 80)
    print()
    
    # Fetch player name from profile (optional, can use a placeholder)
    player_name = "Stefan Ortega"  # Or fetch from profile if needed
    
    # Fetch raw data with direct mapping applied
    raw_df, competition_name = fetch_raw_match_table_with_mapping(player_id, season)
    
    if raw_df is None or raw_df.empty:
        print("ERROR: Could not fetch match data")
        return
    
    print(f"\nRaw DataFrame shape: {raw_df.shape}")
    print(f"Raw DataFrame columns: {list(raw_df.columns)}")
    print()
    
    # Use the transformer to get final format (same as pipeline)
    final_df = transform_matches(player_id, player_name, raw_df)
    
    if final_df.empty:
        print("ERROR: Final DataFrame is empty after transformation")
        return
    
    print(f"Final DataFrame shape: {final_df.shape}")
    print(f"Final DataFrame columns: {list(final_df.columns)}")
    print()
    
    # Show sample rows
    print("Sample rows (first 10):")
    print(final_df.head(10).to_string(index=False))
    print()
    
    # Save to CSV (same format as pipeline output)
    output_path = PROJECT_ROOT / "test_match_data_output.csv"
    final_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"✓ Output saved to: {output_path}")
    print(f"  Total rows: {len(final_df)}")
    print(f"  Format matches: match_{player_id}_{season-1}_{season}.csv")


if __name__ == "__main__":
    main()
