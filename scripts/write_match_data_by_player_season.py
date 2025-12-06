"""Write match data as one CSV per (player, season) for debugging and scaling.

Current scope:
- Players: 324503 and 357233
- Seasons: last season only (2024/25 -> season key 2025)
- Output folder: data_exports/transfermarkt/sl_benfica/20251109/
- Filename: match_<player_id>_<season-1>_<season>.csv
"""

from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, "scripts/data_collection")

from transfermarkt_scraper import TransfermarktScraper  # noqa: E402
from transformers import transform_matches  # noqa: E402
from scripts.run_transfermarkt_pipeline import _get_all_available_seasons  # noqa: E402
import pandas as pd  # noqa: E402


AS_OF_DATE = datetime(2025, 11, 9)
OUTPUT_DIR = Path("data_exports/transfermarkt/sl_benfica/20251109")

# Test players for this debug run
TEST_PLAYERS = [
    {"player_id": 324503},
    {"player_id": 357233},
]

# For now, only last season (2024/25), i.e. season key 2025
SEASONS_TO_INCLUDE = [2025]


def _season_label(season_int: int) -> str:
    """Return season label like '2024_2025' for season key 2025."""
    return f"{season_int - 1}_{season_int}"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scraper = TransfermarktScraper()

    print(f"As of date: {AS_OF_DATE:%Y-%m-%d}")
    print("=" * 80)

    for idx, player in enumerate(TEST_PLAYERS, 1):
        player_id = player["player_id"]
        player_slug = None  # Slug not strictly needed for these tests

        print(f"\n[{idx}/{len(TEST_PLAYERS)}] Player {player_id}")

        # Profile + career to infer seasons (and get name)
        profile = scraper.fetch_player_profile(player_slug, player_id)
        player_name = profile.get("name") or f"Player {player_id}"
        career_df = scraper.fetch_player_career(player_slug, player_id)

        available_seasons = _get_all_available_seasons(
            career_df, profile, AS_OF_DATE.year
        )

        # Restrict to the configured subset (2025 only for now)
        seasons = sorted(
            [s for s in available_seasons if s in SEASONS_TO_INCLUDE], reverse=True
        )
        print(f"  Available seasons (all): {sorted(available_seasons)}")
        print(f"  Seasons to export: {seasons}")

        for season in seasons:
            label = _season_label(season)
            filename = f"match_{player_id}_{label}.csv"
            out_path = OUTPUT_DIR / filename

            print(f"    Season {label} -> {filename}")

            # Optional: skip if already exists
            if out_path.exists():
                print(f"      Skipping (file already exists).")
                continue

            # Fetch raw match log for this single season
            raw = scraper.fetch_player_match_log(player_slug, player_id, season=season)

            if raw.empty:
                print("      No matches for this season (raw DataFrame empty).")
                continue

            print(f"      Raw matches: {len(raw)}")

            # Transform + filter by date
            transformed = transform_matches(player_id, player_name, raw)
            filtered = transformed[
                transformed["date"].isna()
                | (transformed["date"] <= pd.Timestamp(AS_OF_DATE))
            ]

            if filtered.empty:
                print("      No matches after date filter; nothing to write.")
                continue

            # Quick stats summary
            stats_cols = [
                "position",
                "goals",
                "assists",
                "own_goals",
                "yellow_cards",
                "second_yellow_cards",
                "red_cards",
                "substitutions_on",
                "substitutions_off",
                "minutes_played",
            ]
            print(f"      Writing {len(filtered)} rows")
            for col in stats_cols:
                if col in filtered.columns:
                    populated = filtered[col].notna().sum()
                    print(f"        {col:20s}: {populated:4d}/{len(filtered):4d}")

            filtered.to_csv(out_path, index=False, encoding="utf-8-sig", na_rep="")
            print(f"      Wrote {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()



