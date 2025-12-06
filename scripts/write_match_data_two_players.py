"""Write test match data for two specific players into a separate CSV.

This is a focused export for players 324503 and 357233 so we can
inspect match-level stats (position, goals, assists, cards, subs, minutes)
without touching the main 20251109_match_data.csv file.
"""
from pathlib import Path
from datetime import datetime

import sys

sys.path.insert(0, "scripts/data_collection")

from transfermarkt_scraper import TransfermarktScraper  # noqa: E402
from transformers import transform_matches  # noqa: E402
from scripts.run_transfermarkt_pipeline import (  # noqa: E402
    _get_all_available_seasons,
    fetch_multiple_match_logs,
)
import pandas as pd  # noqa: E402


AS_OF_DATE = datetime(2025, 11, 9)
OUTPUT_DIR = Path("data_exports/transfermarkt/sl_benfica/20251109")
OUTPUT_FILE = OUTPUT_DIR / "20251109_match_data_test_324503_357233.csv"


TEST_PLAYERS = [
    {"player_id": 324503},
    {"player_id": 357233},
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scraper = TransfermarktScraper()
    all_matches = []

    print("Writing test match data for players 324503 and 357233")
    print(f"As of date: {AS_OF_DATE.strftime('%Y-%m-%d')}")
    print("=" * 80)

    for idx, player in enumerate(TEST_PLAYERS, 1):
        player_id = player["player_id"]
        player_slug = None  # We do not rely on slugs in these tests

        print(f"\n[{idx}/{len(TEST_PLAYERS)}] Processing player {player_id}")

        # Fetch profile and career to determine seasons
        profile_dict = scraper.fetch_player_profile(player_slug, player_id)
        player_name = profile_dict.get("name") or f"Player {player_id}"
        career_df = scraper.fetch_player_career(player_slug, player_id)

        available_seasons = _get_all_available_seasons(
            career_df, profile_dict, AS_OF_DATE.year
        )
        print(f"  Seasons detected: {sorted(available_seasons)}")

        # Fetch all match logs for detected seasons
        match_df = fetch_multiple_match_logs(
            scraper, player_slug, player_id, available_seasons
        )
        print(f"  Raw matches fetched: {len(match_df)}")

        transformed = transform_matches(player_id, player_name, match_df)
        # Filter by as_of date
        filtered = transformed[
            transformed["date"].isna()
            | (transformed["date"] <= pd.Timestamp(AS_OF_DATE))
        ]
        print(f"  Matches after date filter: {len(filtered)}")

        # Simple stats summary
        if not filtered.empty:
            stats_cols = [
                "position",
                "goals",
                "assists",
                "yellow_cards",
                "red_cards",
                "substitutions_on",
                "substitutions_off",
                "minutes_played",
            ]
            print("  Stats population:")
            for col in stats_cols:
                if col in filtered.columns:
                    populated = filtered[col].notna().sum()
                    print(f"    {col:18s}: {populated:4d}/{len(filtered):4d}")

            all_matches.append(filtered)
        else:
            print("  No matches after filtering; skipping.")

    if not all_matches:
        print("\nNo matches collected for test players; nothing to write.")
        return

    result = pd.concat(all_matches, ignore_index=True)
    result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig", na_rep="")

    print("\n" + "=" * 80)
    print(f"Wrote test match data to: {OUTPUT_FILE}")
    print(f"Total rows: {len(result)}")
    print("=" * 80)


if __name__ == "__main__":
    main()



