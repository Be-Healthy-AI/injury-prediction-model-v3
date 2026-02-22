"""
Parse fetched multi-source data into tabular form and write CSV.

Uses existing transformers from scripts.data_collection.transformers (TM profile, career,
injuries) and scripts.data_collection.fbref_transformers (FBRef match logs).
Writes per-player CSVs under players_raw_data/out/csv/<internal_id>/ and optionally
appends to shared files (players_profile.csv, players_career.csv, injuries_data.csv,
matches_fbref.csv) for compatibility with production layout.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

try:
    from scripts.data_collection.transformers import (
        transform_profile,
        transform_career,
        transform_injuries,
        transform_matches,
    )
    HAS_TM_TRANSFORMERS = True
except ImportError:
    HAS_TM_TRANSFORMERS = False
    transform_matches = None

try:
    from scripts.data_collection.fbref_transformers import transform_fbref_match_log
    HAS_FBREF_TRANSFORMERS = True
except ImportError:
    HAS_FBREF_TRANSFORMERS = False

LOGGER = logging.getLogger(__name__)

# Default output base (players_raw_data/out/csv)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV_DIR = REPO_ROOT / "players_raw_data" / "out" / "csv"

SHARED_FILES = {
    "profile": "players_profile.csv",
    "career": "players_career.csv",
    "injuries": "injuries_data.csv",
    "matches_fbref": "matches_fbref.csv",
    "matches_tm": "matches_tm.csv",
    "matches_enriched": "matches_enriched.csv",
}


def _normalize_team_for_match(s: Any) -> str:
    """Normalize team name for matching TM vs FBRef (lowercase, strip, collapse spaces)."""
    if pd.isna(s) or s is None:
        return ""
    t = " ".join(str(s).strip().lower().split())
    # Optional: drop ranking suffixes like "(1)" already stripped by FBRef transformer
    return t


def merge_tm_with_fbref(tm_df: pd.DataFrame, fbref_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join TM match rows with FBRef match log rows by date and team/opponent.
    Adds FBRef stat columns with 'fbref_' prefix so we keep TM data and add FBRef stats.

    Args:
        tm_df: DataFrame from transform_matches (date, home_team, away_team, ...).
        fbref_df: DataFrame from transform_fbref_match_log (match_date, team, opponent, ...).

    Returns:
        TM DataFrame with additional columns fbref_<col> for each FBRef stat column.
    """
    if tm_df.empty:
        return tm_df.copy()
    if fbref_df.empty:
        return tm_df.copy()

    tm = tm_df.copy()
    if "date" not in tm.columns:
        return tm
    tm["_date"] = pd.to_datetime(tm["date"], errors="coerce")

    # Columns we use only for matching (do not add as fbref_* to avoid clutter; stats are the value)
    fbref_match_cols = {"match_date", "team", "opponent", "venue", "season", "competition", "round"}
    fbref_stat_cols = [c for c in fbref_df.columns if c not in fbref_match_cols]
    for col in fbref_stat_cols:
        tm["fbref_" + col] = pd.NA

    fbref = fbref_df.copy()
    fbref["_date"] = pd.to_datetime(fbref["match_date"], errors="coerce")
    fbref["_team_norm"] = fbref["team"].apply(_normalize_team_for_match)
    fbref["_opp_norm"] = fbref["opponent"].apply(_normalize_team_for_match)

    for i in tm.index:
        td = tm.at[i, "_date"]
        if pd.isna(td):
            continue
        home_norm = _normalize_team_for_match(tm.at[i, "home_team"])
        away_norm = _normalize_team_for_match(tm.at[i, "away_team"])
        if not home_norm and not away_norm:
            continue
        same_date = (fbref["_date"].dt.normalize() == td.normalize())
        candidates = fbref.loc[same_date]
        for _, row in candidates.iterrows():
            t, o = row["_team_norm"], row["_opp_norm"]
            if (t == home_norm and o == away_norm) or (t == away_norm and o == home_norm):
                for col in fbref_stat_cols:
                    if col in row.index:
                        tm.at[i, "fbref_" + col] = row[col]
                break

    tm.drop(columns=["_date"], inplace=True)
    return tm


def parse_fetched_to_tables(
    data: Dict[str, Any],
    tm_id: Optional[int],
    display_name: str,
    fbref_id: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Parse the result of fetch_all() into DataFrames using existing transformers.

    Args:
        data: The dict returned by fetch_all (keys: transfermarkt, fbref, ...).
        tm_id: TransferMarkt player ID (used as id in profile/career/injuries).
        display_name: Player display name (for career/matches).
        fbref_id: FBRef player ID (for match log transform).

    Returns:
        Dict with keys profile, career, injuries, matches_tm, matches_fbref. Each value is a
        DataFrame (possibly empty). Only non-empty DataFrames are included.
    """
    out: Dict[str, pd.DataFrame] = {}

    if not tm_id:
        LOGGER.debug("No tm_id; skipping TM transforms")
    elif not HAS_TM_TRANSFORMERS:
        LOGGER.warning("TM transformers not available; skipping profile/career/injuries")
    else:
        tm = data.get("transfermarkt") or {}

        # Profile
        profile_raw = tm.get("profile")
        if profile_raw and isinstance(profile_raw, dict):
            try:
                out["profile"] = transform_profile(tm_id, profile_raw)
            except Exception as e:
                LOGGER.warning("transform_profile failed: %s", e)

        # Career (from career table, not transfer_history API)
        career_raw = tm.get("career")
        if career_raw is not None:
            if isinstance(career_raw, list):
                career_df = pd.DataFrame(career_raw) if career_raw else pd.DataFrame()
            else:
                career_df = career_raw if isinstance(career_raw, pd.DataFrame) else pd.DataFrame()
            if not career_df.empty:
                try:
                    out["career"] = transform_career(tm_id, display_name, career_df)
                except Exception as e:
                    LOGGER.warning("transform_career failed: %s", e)

        # Injuries
        injuries_raw = tm.get("injuries")
        if injuries_raw is not None:
            if isinstance(injuries_raw, list):
                injuries_df = pd.DataFrame(injuries_raw) if injuries_raw else pd.DataFrame()
            else:
                injuries_df = injuries_raw if isinstance(injuries_raw, pd.DataFrame) else pd.DataFrame()
            if not injuries_df.empty:
                try:
                    out["injuries"] = transform_injuries(tm_id, injuries_df)
                except Exception as e:
                    LOGGER.warning("transform_injuries failed: %s", e)

        # TM match log
        matches_raw = tm.get("matches")
        if matches_raw is not None and transform_matches:
            if isinstance(matches_raw, list):
                match_df = pd.DataFrame(matches_raw) if matches_raw else pd.DataFrame()
            else:
                match_df = matches_raw if isinstance(matches_raw, pd.DataFrame) else pd.DataFrame()
            if not match_df.empty:
                try:
                    out["matches_tm"] = transform_matches(tm_id, display_name, match_df)
                except Exception as e:
                    LOGGER.warning("transform_matches failed: %s", e)

    # FBRef match logs
    fbref = data.get("fbref") or {}
    match_logs_raw = fbref.get("match_logs")
    if match_logs_raw is not None and fbref_id and HAS_FBREF_TRANSFORMERS:
        if isinstance(match_logs_raw, list):
            match_df = pd.DataFrame(match_logs_raw) if match_logs_raw else pd.DataFrame()
        else:
            match_df = match_logs_raw if isinstance(match_logs_raw, pd.DataFrame) else pd.DataFrame()
        if not match_df.empty:
            try:
                out["matches_fbref"] = transform_fbref_match_log(match_df, fbref_id)
            except Exception as e:
                LOGGER.warning("transform_fbref_match_log failed: %s", e)

    # Enriched: TM matches merged with FBRef stats (TM columns + fbref_* columns)
    if "matches_tm" in out and "matches_fbref" in out and not out["matches_tm"].empty:
        try:
            out["matches_enriched"] = merge_tm_with_fbref(out["matches_tm"], out["matches_fbref"])
        except Exception as e:
            LOGGER.warning("merge_tm_with_fbref failed: %s", e)

    return out


def write_parsed_to_csv(
    internal_id: str,
    parsed: Dict[str, pd.DataFrame],
    out_dir: Path,
    append_to_shared: bool = True,
) -> Dict[str, Path]:
    """
    Write parsed DataFrames to CSV.

    - Per-player: out_dir / <internal_id> / profile.csv, career.csv, injuries.csv, matches_tm.csv, matches_fbref.csv, matches_enriched.csv
    - If append_to_shared: also append to out_dir / players_profile.csv, etc. (after removing
      existing rows for the same player id so re-runs don't duplicate).

    Returns:
        Dict mapping table name to the path written (per-player path).
    """
    written: Dict[str, Path] = {}
    out_dir = Path(out_dir)
    player_dir = out_dir / internal_id
    player_dir.mkdir(parents=True, exist_ok=True)

    id_col = "id"  # profile and career use "id"
    player_id_col = "player_id"  # injuries and matches use "player_id"

    for name, df in parsed.items():
        if df is None or df.empty:
            continue
        filename = {
            "profile": "profile.csv",
            "career": "career.csv",
            "injuries": "injuries.csv",
            "matches_fbref": "matches_fbref.csv",
            "matches_tm": "matches_tm.csv",
            "matches_enriched": "matches_enriched.csv",
        }.get(name, f"{name}.csv")
        path = player_dir / filename
        df.to_csv(path, index=False, encoding="utf-8-sig")
        written[name] = path

        if not append_to_shared or name not in SHARED_FILES:
            continue
        shared_path = out_dir / SHARED_FILES[name]
        # Determine player identifier column and value for this table
        if name == "profile" and "id" in df.columns:
            current_ids = set(df["id"].astype(str))
        elif name == "career" and "id" in df.columns:
            current_ids = set(df["id"].astype(str))
        elif name == "injuries" and "player_id" in df.columns:
            current_ids = set(df["player_id"].astype(str))
        elif name == "matches_fbref" and "fbref_player_id" in df.columns:
            current_ids = set(df["fbref_player_id"].astype(str))
        elif name == "matches_tm" and "player_id" in df.columns:
            current_ids = set(df["player_id"].astype(str))
        elif name == "matches_enriched" and "player_id" in df.columns:
            current_ids = set(df["player_id"].astype(str))
        else:
            current_ids = set()
        if not current_ids:
            continue
        drop_col = (
            "id" if name in ("profile", "career") else
            ("player_id" if name in ("injuries", "matches_tm", "matches_enriched") else "fbref_player_id")
        )
        if shared_path.exists():
            try:
                existing = pd.read_csv(shared_path, encoding="utf-8-sig")
                if drop_col in existing.columns:
                    existing = existing[~existing[drop_col].astype(str).isin(current_ids)]
                combined = pd.concat([existing, df], ignore_index=True)
                # Keep only columns from current schema (drops legacy e.g. transfermarkt_score from old shared files)
                if name == "matches_tm" or name == "matches_enriched":
                    combined = combined[[c for c in df.columns if c in combined.columns]]
            except Exception as e:
                LOGGER.warning("Could not read/merge %s: %s", shared_path, e)
                combined = df
        else:
            combined = df
        combined.to_csv(shared_path, index=False, encoding="utf-8-sig")
        LOGGER.debug("Appended to %s", shared_path)

    return written


def parse_and_write_csv(
    internal_id: str,
    data: Dict[str, Any],
    tm_id: Optional[int],
    display_name: str,
    fbref_id: Optional[str] = None,
    out_dir: Optional[Path] = None,
    append_to_shared: bool = True,
) -> Dict[str, Path]:
    """
    Parse fetched data to tables and write CSV. Single entry point for the pipeline.

    Returns:
        Dict mapping table name to path written (per-player path).
    """
    out_dir = out_dir or DEFAULT_CSV_DIR
    parsed = parse_fetched_to_tables(data, tm_id, display_name, fbref_id=fbref_id)
    return write_parsed_to_csv(internal_id, parsed, out_dir, append_to_shared=append_to_shared)
