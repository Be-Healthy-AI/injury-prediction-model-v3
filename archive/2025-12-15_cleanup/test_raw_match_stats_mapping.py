#!/usr/bin/env python3
"""
Small debug script to inspect raw match stats tables from Transfermarkt and
validate column mappings for stats like goals, assists, own goals, cards,
substitutions, and minutes played.

This does NOT touch the main pipeline; it only fetches and prints information
for specific test players / seasons.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple, Optional, TextIO

import pandas as pd
import requests
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/129.0.0.0 Safari/537.36"
)

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Language": "en-US,en;q=0.9",
}


def log_output(message: str, output_file: Optional[TextIO] = None) -> None:
    """Print to console and optionally write to file."""
    print(message, flush=True)
    if output_file:
        output_file.write(message + "\n")
        output_file.flush()


def fetch_raw_match_table(
    player_id: int,
    season: int,
    base_url: str = "https://www.transfermarkt.co.uk",
    output_file: Optional[TextIO] = None,
) -> Tuple[Optional[List[str]], pd.DataFrame]:
    """
    Fetch the raw match log table for a player/season using pd.read_html
    directly, without any of our scraper post-processing.

    Returns:
        (header_texts, df) where header_texts is the list of <th> texts and
        df is the DataFrame produced by pandas.
    """
    # Use generic '-' slug so we don't depend on the exact player slug
    url = (
        f"{base_url}/-/leistungsdatendetails/spieler/{player_id}"
        f"/saison/{season}/verein/0/liga/0/wettbewerb//pos/0/trainer_id/0/plus/1"
    )
    log_output(f"\n=== Fetching raw match table for player {player_id}, season {season} ===", output_file)
    log_output(f"URL: {url}", output_file)

    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    tables = soup.select("table")
    log_output(f"Found {len(tables)} tables on page", output_file)

    for idx, table in enumerate(tables):
        headers = table.select("thead th")
        header_texts = [h.get_text(strip=True) for h in headers]
        if "Matchday" in header_texts:
            log_output(f"\nUsing table {idx} as match log table (contains 'Matchday' header)", output_file)
            log_output(f"Header texts: {header_texts}", output_file)
            dfs = pd.read_html(str(table), flavor="bs4")
            df = dfs[0] if dfs else pd.DataFrame()
            return header_texts, df

    log_output("No table with 'Matchday' header found.", output_file)
    return None, pd.DataFrame()


def inspect_stats_columns(df: pd.DataFrame, output_file: Optional[TextIO] = None) -> None:
    """
    Print information about potential stats columns (Unnamed: 8..17) to help
    validate manual mapping.
    """
    if df.empty:
        log_output("DataFrame is empty.", output_file)
        return

    log_output("\nRaw DataFrame columns:", output_file)
    log_output(str(list(df.columns)), output_file)

    # Focus on the potential stats columns
    stats_candidates = [
        col for col in df.columns if str(col).startswith("Unnamed:")
    ]
    log_output("\nCandidate stats columns (Unnamed:*):", output_file)
    log_output(str(stats_candidates), output_file)

    # Show basic stats for each candidate: non-null count and unique values
    for col in stats_candidates:
        series = df[col]
        non_null = series.notna().sum()
        uniques = series.dropna().unique()
        log_output(f"\nColumn '{col}':", output_file)
        log_output(f"  Non-null values: {non_null}", output_file)
        # Show up to 15 unique values
        if len(uniques) > 15:
            sample_uniques = uniques[:15]
            log_output(f"  Unique values (first 15 of {len(uniques)}): {sample_uniques}", output_file)
        else:
            log_output(f"  Unique values ({len(uniques)}): {uniques}", output_file)

    # Show a small sample of rows with these columns
    log_output("\nSample rows (first 10) with candidate stats columns:", output_file)
    display_cols = [
        c
        for c in df.columns
        if c in ["Matchday", "Date", "Home team", "Away team", "Result"]
        or c in stats_candidates
    ]
    log_output(df[display_cols].head(10).to_string(index=False), output_file)


def main() -> None:
    output_path = PROJECT_ROOT / "test_stats_mapping_output.txt"
    
    with open(output_path, "w", encoding="utf-8") as f:
        log_output("=" * 80, f)
        log_output("Match Stats Column Mapping Test", f)
        log_output("=" * 80, f)
        log_output(f"Output file: {output_path}", f)
        log_output("", f)
        
        # Test 1: Outfield player with goals/assists etc. (user suggested 324503, 2024/25)
        header_324503, df_324503 = fetch_raw_match_table(player_id=324503, season=2024, output_file=f)
        if not df_324503.empty:
            log_output("\n--- Inspecting stats columns for player 324503, season 2024 ---", f)
            inspect_stats_columns(df_324503, output_file=f)
        else:
            log_output("\nNo data for player 324503, season 2024.", f)

        # Test 2: Goalkeeper Stefan Ortega (85941) for comparison
        header_85941, df_85941 = fetch_raw_match_table(player_id=85941, season=2024, output_file=f)
        if not df_85941.empty:
            log_output("\n--- Inspecting stats columns for player 85941, season 2024 ---", f)
            inspect_stats_columns(df_85941, output_file=f)
        else:
            log_output("\nNo data for player 85941, season 2024.", f)
        
        log_output("\n" + "=" * 80, f)
        log_output("Test completed. Output saved to test_stats_mapping_output.txt", f)
        log_output("=" * 80, f)
    
    print(f"\n✓ Output saved to: {output_path}")


if __name__ == "__main__":
    main()


