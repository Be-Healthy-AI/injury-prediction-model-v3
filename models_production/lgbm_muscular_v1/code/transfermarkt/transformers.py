"""
Transform scraped Transfermarkt tables into the canonical schemas used to train
injury prediction models.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAPPINGS_PATH = PROJECT_ROOT / "config" / "transfermarkt_mappings.json"


def _load_mappings() -> Dict[str, Any]:
    if not MAPPINGS_PATH.exists():
        LOGGER.warning("Missing mappings file at %s", MAPPINGS_PATH)
        return {}
    with MAPPINGS_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


MAPPINGS = _load_mappings()


PROFILE_COLUMNS = [
    "id",
    "name",
    "position",
    "date_of_birth",
    "nationality1",
    "nationality2",
    "height",
    "foot",
    "joined_on",
    "current_club",
]

CAREER_COLUMNS = ["id", "name", "Season", "Date", "From", "To", "VM", "Value"]

INJURY_COLUMNS = [
    "player_id",
    "season",
    "injury_type",
    "fromDate",
    "untilDate",
    "days",
    "clubs",
    "lost_games",
]

MATCH_COLUMNS = [
    "season",
    "player_id",
    "player_name",
    "competition",
    "journey",
    "date",
    "home_team",
    "away_team",
    "result",
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
    "transfermarkt_score",
]

TEAMS_COLUMNS = ["team", "country"]
COMPETITION_COLUMNS = ["competition", "country", "type"]


def transform_profile(player_id: int, profile: Dict[str, Any]) -> pd.DataFrame:
    """Convert a profile dictionary into the canonical 10-column shape."""
    # Format dates as YYYY-MM-DD strings (not datetime objects)
    def _format_date(value: Any) -> Optional[str]:
        if value is None:
            return None
        # If it's already a string in YYYY-MM-DD format, return it
        if isinstance(value, str):
            # Check if it's already in YYYY-MM-DD format
            if len(value) == 10 and value.count('-') == 2:
                # Validate it's a valid date format
                try:
                    pd.to_datetime(value, format="%Y-%m-%d")
                    return value
                except:
                    pass
            # If it's a datetime string with time component, extract just the date part
            if ' ' in value:
                date_part = value.split(' ')[0]
                if len(date_part) == 10 and date_part.count('-') == 2:
                    try:
                        pd.to_datetime(date_part, format="%Y-%m-%d")
                        return date_part
                    except:
                        pass
        # Try to parse the date
        dt = _to_datetime(value)
        if dt is None:
            return None
        # Format datetime to YYYY-MM-DD
        if isinstance(dt, pd.Timestamp):
            return dt.strftime("%Y-%m-%d")
        elif hasattr(dt, 'strftime'):
            return dt.strftime("%Y-%m-%d")
        # Fallback: convert to string and take first 10 characters
        dt_str = str(dt)
        if len(dt_str) >= 10:
            return dt_str[:10]
        return None
    
    row = {
        "id": player_id,
        "name": profile.get("name"),
        "position": _normalize_position(profile.get("position")),
        "date_of_birth": _format_date(profile.get("date_of_birth")),
        "nationality1": _coalesce_nationalities(profile, idx=0),
        "nationality2": _coalesce_nationalities(profile, idx=1),
        "height": _parse_height(profile.get("height")),
        "foot": _normalize_foot(profile.get("foot")),
        "joined_on": _format_date(profile.get("joined_on")),
        "current_club": profile.get("current_club"),
    }
    return pd.DataFrame([row], columns=PROFILE_COLUMNS)


def transform_career(
    player_id: int,
    player_name: str,
    transfers: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize the transfer history table."""
    if transfers.empty:
        return pd.DataFrame(columns=CAREER_COLUMNS)
    df = transfers.copy()
    rename_map = {
        "Datum": "Date",
        "Date": "Date",
        "Verein abgebend": "From",
        "Left": "From",
        "Verein aufnehmend": "To",
        "Joined": "To",
        "Marktwert": "VM",
        "MV": "VM",
        "Ablöse": "Value",
        "Fee": "Value",
        "Season": "Season",
    }
    df = df.rename(columns=rename_map)
    df["id"] = player_id
    df["name"] = player_name
    # Parse dates - handle both ISO format (YYYY-MM-DD) and other formats
    # ISO format dates from API are unambiguous (YYYY-MM-DD)
    # For other formats, use dayfirst=True for European format (DD/MM/YYYY)
    date_series = df.get("Date")
    if date_series is not None and not date_series.empty:
        # First try ISO format (YYYY-MM-DD) - this is unambiguous and correct
        # ISO format dates are always YYYY-MM-DD, so no ambiguity about month/day
        df["Date"] = pd.to_datetime(date_series, errors="coerce", format="%Y-%m-%d", dayfirst=False)
        # If ISO format failed (some dates might be in other formats like DD/MM/YYYY from HTML scraping),
        # try with dayfirst=True for European format (DD/MM/YYYY)
        if df["Date"].isna().any():
            # For non-ISO dates, use dayfirst=True to parse DD/MM/YYYY correctly
            # This ensures dates like "10/08/2023" are parsed as August 10, not October 8
            df["Date"] = pd.to_datetime(date_series, errors="coerce", dayfirst=True)
    else:
        df["Date"] = pd.Series(dtype="datetime64[ns]")
    df["VM"] = _to_numeric(df.get("VM"))
    # Parse Value column: extract integers from formatted strings or keep special strings
    df["Value"] = _parse_transfer_value(df.get("Value"))
    df = df.reindex(columns=CAREER_COLUMNS)
    return df.sort_values("Date", ascending=False).reset_index(drop=True)


def calculate_severity_from_days(days: Any) -> str:
    """
    Calculate severity level based on number of days.
    
    Thresholds:
    - mild: ≤7 days
    - moderate: 8-45 days
    - severe: 46-90 days
    - critical: >90 days
    
    Args:
        days: Number of days (can be int, float, str, or NaN)
    
    Returns:
        Severity level string: 'mild', 'moderate', 'severe', or 'critical'
    """
    if pd.isna(days):
        return 'mild'  # Default for missing data
    
    try:
        days_value = float(days)
        if days_value <= 7:
            return 'mild'
        elif days_value <= 45:
            return 'moderate'
        elif days_value <= 90:
            return 'severe'
        else:
            return 'critical'
    except (ValueError, TypeError):
        return 'mild'  # Default for invalid data


def transform_injuries(
    player_id: int, 
    injuries: pd.DataFrame,
    injury_mappings: Optional[Dict[str, Dict[str, str]]] = None
) -> pd.DataFrame:
    """
    Transform injuries DataFrame and add injury_class, body_part, and severity columns.
    
    Args:
        player_id: Player ID
        injuries: Raw injuries DataFrame
        injury_mappings: Optional dictionary mapping injury_type -> {injury_class, body_part}
                        If None, will try to load from production/config/injury_mappings.json
    """
    if injuries.empty:
        # Return empty DataFrame with all columns including the new ones
        extended_columns = INJURY_COLUMNS + ['injury_class', 'body_part', 'severity']
        return pd.DataFrame(columns=extended_columns)
    
    df = injuries.copy()
    
    # Load mappings if not provided
    if injury_mappings is None:
        mapping_file = Path(__file__).resolve().parents[2] / "production" / "config" / "injury_mappings.json"
        if mapping_file.exists():
            with open(mapping_file, 'r', encoding='utf-8') as f:
                injury_mappings = json.load(f)
        else:
            injury_mappings = {}
            LOGGER.warning(f"Injury mappings file not found: {mapping_file}")
    
    rename_map = {
        "Injury": "injury_type",
        "Injury type": "injury_type",
        "from": "fromDate",
        "From": "fromDate",
        "until": "untilDate",
        "Until": "untilDate",
        "Days": "days",
        "Club": "clubs",
        "Games missed": "lost_games",
        "Season": "season",
    }
    df = df.rename(columns=rename_map)
    df["player_id"] = player_id
    
    # Step 1: Convert to datetime (needed for season derivation)
    fromDate_dt = _to_datetime_series(df.get("fromDate"))
    untilDate_dt = _to_datetime_series(df.get("untilDate"))
    
    # Step 2: Derive season using datetime version (before converting to string)
    if "season" not in df.columns:
        df["season"] = _derive_season(fromDate_dt)  # Uses datetime, not string
    
    # Step 3: Convert to date format (YYYY-MM-DD strings) for final output
    df["fromDate"] = fromDate_dt.dt.strftime("%Y-%m-%d").replace("NaT", None)
    df["untilDate"] = untilDate_dt.dt.strftime("%Y-%m-%d").replace("NaT", None)
    
    df["days"] = _to_numeric(df.get("days"))
    
    # Convert lost_games to integers (round to nearest integer, NaN stays NaN)
    lost_games_numeric = _to_numeric(df.get("lost_games"))
    lost_games_rounded = lost_games_numeric.round()
    df["lost_games"] = lost_games_rounded.astype("Int64")  # Nullable integer type
    # Ensure any remaining float values are properly converted to integer
    df["lost_games"] = df["lost_games"].apply(lambda x: int(x) if pd.notna(x) and not pd.isna(x) else pd.NA)
    
    if "clubs" not in df.columns:
        df["clubs"] = None
    
    # NEW: Add injury_class, body_part, and severity columns
    df["injury_class"] = df["injury_type"].apply(
        lambda x: injury_mappings.get(str(x), {}).get('injury_class', 'unknown') 
        if pd.notna(x) else 'unknown'
    )
    
    df["body_part"] = df["injury_type"].apply(
        lambda x: injury_mappings.get(str(x), {}).get('body_part', '') 
        if pd.notna(x) else ''
    )
    
    df["severity"] = df["days"].apply(calculate_severity_from_days)
    
    # Return all columns including the new ones
    extended_columns = INJURY_COLUMNS + ['injury_class', 'body_part', 'severity']
    # Ensure all required columns are present
    for col in extended_columns:
        if col not in df.columns:
            df[col] = None
    
    return df[extended_columns]


def transform_matches(
    player_id: int,
    player_name: str,
    match_df: pd.DataFrame,
) -> pd.DataFrame:
    if match_df.empty:
        return pd.DataFrame(columns=MATCH_COLUMNS)
    df = match_df.copy()
    rename_map = {
        "Season": "season",
        "Competition": "competition",
        "Matchday": "journey",
        "Date": "date",
        "Home team": "home_team",
        "Away team": "away_team",
        "Result": "result",
        "Position": "position",
        "Pos.": "position",  # Alternative column name
        "Goals": "goals",
        "Assists": "assists",
        "Own goals": "own_goals",
        "Own goal": "own_goals",  # Handle singular form
        "OG": "own_goals",  # Handle abbreviation
        "Golos próprios": "own_goals",  # Portuguese
        "Golo próprio": "own_goals",  # Portuguese singular
        "Eigentore": "own_goals",  # German
        "Eigentor": "own_goals",  # German singular
        "Yellow cards": "yellow_cards",
        "Second yellow cards": "second_yellow_cards",
        "Red cards": "red_cards",
        "Sub on": "substitutions_on",
        "Sub off": "substitutions_off",
        "Minutes played": "minutes_played",
        "TM-Whoscored grade": "transfermarkt_score",
        "Grade": "transfermarkt_score",
    }
    df = df.rename(columns=rename_map)
    df["player_id"] = player_id
    df["player_name"] = player_name
    df["date"] = _to_datetime_series(df.get("date"))
    numeric_cols = [
        "goals",
        "assists",
        "own_goals",
        "yellow_cards",
        "second_yellow_cards",
        "red_cards",
        "substitutions_on",
        "substitutions_off",
        "transfermarkt_score",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = _to_numeric(df[col])
        else:
            df[col] = pd.NA
    
    # Convert goals, assists, and own_goals to integers (round to nearest integer, NaN stays NaN)
    for col in ["goals", "assists", "own_goals"]:
        if col in df.columns:
            # Round to nearest integer and convert to nullable integer type
            df[col] = df[col].round().astype("Int64")
            # Ensure proper integer conversion (handle any edge cases)
            df[col] = df[col].apply(lambda x: int(x) if pd.notna(x) else pd.NA)
    
    # Convert time-based columns to integers (handling injury time format like "90+3'" -> 93)
    time_based_cols = ["yellow_cards", "second_yellow_cards", "red_cards", 
                       "substitutions_on", "substitutions_off", "minutes_played"]
    for col in time_based_cols:
        if col in df.columns:
            # Parse minute values (handles "90+3'" -> 93, "903.0" -> 93, etc.)
            df[col] = df[col].apply(lambda x: _parse_minute_value(x) if pd.notna(x) else pd.NA)
            # Convert to nullable integer type
            df[col] = df[col].astype("Int64")
        else:
            df[col] = pd.NA
    
    # Normalize position column if it exists
    if "position" in df.columns:
        df["position"] = df["position"].apply(_normalize_position)
    
    # Remove duplicate columns before reindexing (can happen if both "Position" and "Pos." exist)
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Include all match columns with stats
    DESIRED_COLUMNS = [
        "season",
        "player_id", 
        "player_name",
        "competition",
        "journey",
        "date",
        "home_team",
        "away_team",
        "result",
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
        "transfermarkt_score",
    ]
    desired = df.reindex(columns=DESIRED_COLUMNS)
    
    # Derive season from match dates instead of using Transfermarkt's Season column
    # This ensures consistency: matches from 2024/25 season show "2024/25" in season column
    if "date" in desired.columns and "season" in desired.columns:
        # Derive season from dates (format: "2024/25" - 4-digit start year, 2-digit end year)
        def _derive_season_from_date(ts: pd.Timestamp) -> Optional[str]:
            if pd.isna(ts):
                return None
            year = ts.year
            if ts.month >= 7:
                start, end = year, year + 1
            else:
                start, end = year - 1, year
            return f"{start}/{str(end)[-2:]}"
        
        derived_seasons = desired["date"].apply(_derive_season_from_date)
        # Only update where we have valid dates
        mask = desired["date"].notna()
        desired.loc[mask, "season"] = derived_seasons[mask]
        # For rows without dates, keep original season value if available
        desired["season"] = desired["season"].astype(str)
    else:
        desired["season"] = desired["season"].astype(str)
    
    return desired


def transform_teams(raw_rows: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(raw_rows)
    if df.empty:
        return pd.DataFrame(columns=TEAMS_COLUMNS)
    rename_map = {"Team": "team", "Club": "team", "Country": "country"}
    df = df.rename(columns=rename_map)
    df = df[TEAMS_COLUMNS].drop_duplicates().reset_index(drop=True)
    return df


def transform_competitions(raw_rows: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(raw_rows)
    if df.empty:
        return pd.DataFrame(columns=COMPETITION_COLUMNS)
    rename_map = {
        "Competition": "competition",
        "Country": "country",
        "Type": "type",
    }
    df = df.rename(columns=rename_map)
    inferred = df["competition"].apply(_infer_competition_type)
    if "type" not in df.columns:
        df["type"] = inferred
    else:
        df["type"] = df["type"].fillna(inferred)
    for column in COMPETITION_COLUMNS:
        if column not in df.columns:
            df[column] = None
    return df[COMPETITION_COLUMNS].drop_duplicates().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_position(position: Any) -> Optional[str]:
    """
    Normalize position names to match reference schema.
    
    Maps Transfermarkt position formats to canonical names:
    - "Defender - Centre-Back" -> "Centre Back"
    - "Midfield - Defensive Midfield" -> "Defensive Midfielder"
    - "Attack - Centre-Forward" -> "Centre Forward"
    etc.
    """
    if position is None or (isinstance(position, float) and pd.isna(position)):
        return None
    
    position_str = str(position).strip()
    if not position_str:
        return None
    
    # Position normalization mapping
    position_map = {
        "defender - centre-back": "Centre Back",
        "defender - left-back": "Left Back",
        "defender - right-back": "Right Back",
        "midfield - defensive midfield": "Defensive Midfielder",
        "midfield - central midfield": "Central Midfielder",
        "midfield - attacking midfield": "Attacking Midfielder",
        "attack - left winger": "Left Winger",
        "attack - right winger": "Right Winger",
        "attack - centre-forward": "Centre Forward",
        "attack - striker": "Centre Forward",
        "attack - second-striker": "Second Attacker",
    }
    
    # Try exact match (case-insensitive)
    normalized = position_map.get(position_str.lower())
    if normalized:
        return normalized
    
    # Try partial matches for variations
    position_lower = position_str.lower()
    for key, value in position_map.items():
        if key in position_lower or position_lower in key:
            return value
    
    # If no mapping found, return original (capitalized)
    return position_str


def _normalize_position(position: Any) -> Optional[str]:
    """
    Normalize position names to match reference schema.
    
    Maps Transfermarkt position formats to canonical names:
    - "Defender - Centre-Back" -> "Centre Back"
    - "Midfield - Defensive Midfield" -> "Defensive Midfielder"
    - "Attack - Centre-Forward" -> "Centre Forward"
    etc.
    """
    if position is None or (isinstance(position, float) and pd.isna(position)):
        return None
    
    position_str = str(position).strip()
    if not position_str:
        return None
    
    # Position normalization mapping
    position_map = {
        "defender - centre-back": "Centre Back",
        "defender - left-back": "Left Back",
        "defender - right-back": "Right Back",
        "midfield - defensive midfield": "Defensive Midfielder",
        "midfield - central midfield": "Central Midfielder",
        "midfield - attacking midfield": "Attacking Midfielder",
        "attack - left winger": "Left Winger",
        "attack - right winger": "Right Winger",
        "attack - centre-forward": "Centre Forward",
        "attack - striker": "Centre Forward",
        "attack - second-striker": "Second Attacker",
    }
    
    # Try exact match (case-insensitive)
    normalized = position_map.get(position_str.lower())
    if normalized:
        return normalized
    
    # Try partial matches for variations
    position_lower = position_str.lower()
    for key, value in position_map.items():
        if key in position_lower or position_lower in key:
            return value
    
    # If no mapping found, return original (capitalized)
    return position_str


def _coalesce_nationalities(profile: Dict[str, Any], idx: int) -> Optional[str]:
    values = profile.get("nationalities")
    if isinstance(values, (list, tuple)):
        return values[idx] if len(values) > idx else None
    if idx == 0:
        return profile.get("nationality")
    return None


def _normalize_foot(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    normalized = str(value).strip()
    alias_map = MAPPINGS.get("foot_aliases", {})
    return alias_map.get(normalized, normalized.lower())


def _parse_height(value: Any) -> Optional[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    match = re.search(r"([0-9]+[.,]?[0-9]*)", str(value))
    if not match:
        return None
    meters = float(match.group(1).replace(",", "."))
    # Transfermarkt heights are usually meters; convert to centimeters and round to integer
    return int(round(meters * 100))


def _to_datetime(value: Any) -> Optional[pd.Timestamp]:
    if isinstance(value, pd.Timestamp):
        return value
    series = pd.Series([value])
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
    ts = parsed.iloc[0]
    return ts.to_pydatetime() if not pd.isna(ts) else None


def _to_datetime_series(value: Optional[pd.Series]) -> pd.Series:
    if isinstance(value, pd.Series):
        return pd.to_datetime(value, errors="coerce", dayfirst=True)
    return pd.Series(dtype="datetime64[ns]")


def _to_numeric(series: Optional[pd.Series]) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    cleaned = series.astype(str).str.replace(r"[^0-9\.-]", "", regex=True)
    cleaned = cleaned.str.replace(",", ".", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def _parse_transfer_value(series: Optional[pd.Series]) -> pd.Series:
    """
    Parse transfer value column: extract integers from formatted strings or keep special strings.
    
    Handles formats like:
    - "€10.00M (10000000)" -> 10000000
    - "€600.00K (600000)" -> 600000
    - "-" -> "-"
    - "?" -> "?"
    - "Zero cost" -> "Zero cost"
    - "End of loan" -> "End of loan"
    - Already integer values -> keep as integer
    """
    if series is None:
        return pd.Series(dtype="object")
    
    def parse_single_value(value: Any) -> Any:
        # If already an integer or numeric, return as-is
        if isinstance(value, (int, float)) and not pd.isna(value):
            return int(value) if isinstance(value, float) and value.is_integer() else value
        
        # Handle NaN/None
        if pd.isna(value) or value is None:
            return "-"
        
        value_str = str(value).strip()
        
        # Keep special strings as-is
        if value_str in ("-", "?", "Zero cost", "End of loan", "Free transfer", "Free Transfer"):
            return value_str
        
        # Try to extract integer from parentheses: "€10.00M (10000000)" -> 10000000
        paren_match = re.search(r"\((\d+)\)", value_str)
        if paren_match:
            try:
                return int(paren_match.group(1))
            except (ValueError, AttributeError):
                pass
        
        # Try to parse as integer directly
        try:
            # Remove currency symbols and commas, then parse
            cleaned = re.sub(r"[^\d]", "", value_str)
            if cleaned:
                return int(cleaned)
        except (ValueError, AttributeError):
            pass
        
        # If all else fails, return the original string
        return value_str if value_str else "-"
    
    return series.apply(parse_single_value)


def _flag_unknown_physio(injury: Any) -> float:
    if injury is None or (isinstance(injury, float) and pd.isna(injury)):
        return float("nan")
    if "unknown" in str(injury).lower():
        return 1.0
    return float("nan")


def _parse_minute_value(value: Any) -> Optional[int]:
    """
    Parse minute values to integer, handling injury time format.
    
    Handles formats like:
    - "90'" -> 90
    - "90+3'" -> 93 (90 minutes + 3 minutes injury time)
    - "120+5'" -> 125 (120 minutes + 5 minutes injury time)
    - 903.0 (from pandas parsing "90+3'") -> 93
    - 1205.0 (from pandas parsing "120+5'") -> 125
    
    Returns:
        Integer minute value or None if invalid/missing
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    
    # Handle numeric values that might be from "90 + 3'" -> 903.0
    if isinstance(value, (int, float)) and not pd.isna(value):
        # Check if it looks like it came from "90 + 3'" format (903, 1202, etc.)
        if value > 100 and value < 10000:
            # Try to split: 903 -> 90 + 3, 1202 -> 120 + 2
            # Most common: 90 + X -> 90X, 120 + X -> 120X
            if value >= 900 and value < 1000:
                # Likely 90 + X format
                extra = int(value % 100)
                base = 90
                if extra < 10:  # Reasonable extra time (0-9 minutes)
                    return base + extra
            elif value >= 1200 and value < 1300:
                # Likely 120 + X format (extra time)
                extra = int(value % 100)
                base = 120
                if extra < 10:  # Reasonable extra time (0-9 minutes)
                    return base + extra
        # Otherwise, treat as regular minutes (must be reasonable)
        int_val = int(value)
        if 0 <= int_val <= 150:  # Reasonable minute range
            return int_val
        return None
    
    text = str(value).strip()
    
    # Handle "90 + 3'" or "90+3'" format (extra time)
    if "+" in text:
        # Extract base minutes and extra time
        import re
        parts = re.split(r'\s*\+\s*', text)
        if len(parts) == 2:
            base_min = re.findall(r'\d+', parts[0])
            extra_min = re.findall(r'\d+', parts[1])
            if base_min and extra_min:
                total = int(base_min[0]) + int(extra_min[0])
                if 0 <= total <= 150:  # Reasonable range
                    return total
    
    # Handle standard format "90'" or "90"
    import re
    digits = re.findall(r'\d+', text)
    if digits:
        minute_val = int(digits[0])
        if 0 <= minute_val <= 150:  # Reasonable range
            return minute_val
    
    return None


def _normalize_minutes(value: Any) -> Optional[str]:
    """
    Normalize minutes played values.
    Handles formats like:
    - "90'" -> "90'"
    - "120'" -> "120'"
    - "90 + 3'" -> "93'" (adds extra time)
    - "90+3'" -> "93'"
    - 903.0 (from "90 + 3'" converted by pandas) -> "93'"
    - "23'" -> "23'" (yellow card minute, but should be handled separately)
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    
    # Handle numeric values that might be from "90 + 3'" -> 903.0
    if isinstance(value, (int, float)) and not pd.isna(value):
        # Check if it looks like it came from "90 + 3'" format (903, 1202, etc.)
        # These would be base_minutes * 100 + extra_time
        if value > 100 and value < 10000:
            # Try to split: 903 -> 90 + 3, 1202 -> 120 + 2
            # Most common: 90 + X -> 90X, 120 + X -> 120X
            if value >= 900 and value < 1000:
                # Likely 90 + X format
                extra = int(value % 100)
                base = 90
                if extra < 10:  # Reasonable extra time
                    return f"{base + extra}'"
            elif value >= 1200 and value < 1300:
                # Likely 120 + X format
                extra = int(value % 100)
                base = 120
                if extra < 10:  # Reasonable extra time
                    return f"{base + extra}'"
        # Otherwise, treat as regular minutes
        return f"{int(value)}'"
    
    text = str(value).strip()
    
    # Handle "90 + 3'" or "90+3'" format (extra time)
    if "+" in text:
        # Extract base minutes and extra time
        parts = re.split(r'\s*\+\s*', text)
        if len(parts) == 2:
            base_min = re.findall(r'\d+', parts[0])
            extra_min = re.findall(r'\d+', parts[1])
            if base_min and extra_min:
                total = int(base_min[0]) + int(extra_min[0])
                return f"{total}'"
    
    # Handle standard format "90'"
    if text.endswith("'"):
        return text
    
    # Extract digits if present
    digits = re.findall(r"[0-9]+", text)
    return f"{digits[0]}'" if digits else None


def _derive_season(dates: pd.Series) -> pd.Series:
    def _map(ts: pd.Timestamp) -> Optional[str]:
        if pd.isna(ts):
            return None
        year = ts.year
        if ts.month >= 7:
            start, end = year, year + 1
        else:
            start, end = year - 1, year
        return f"{str(start)[-2:]}/{str(end)[-2:]}"

    return dates.apply(_map) if dates is not None else pd.Series(dtype="object")


def _infer_competition_type(name: Any) -> Optional[str]:
    if not name or (isinstance(name, float) and pd.isna(name)):
        return None
    overrides = MAPPINGS.get("competition_type_overrides", {})
    normalized = str(name)
    if normalized in overrides:
        return overrides[normalized]
    if "cup" in normalized.lower():
        return "Cup"
    if "league" in normalized.lower():
        return "Main League"
    if "friendly" in normalized.lower():
        return "Friendly"
    return None


__all__ = [
    "transform_profile",
    "transform_career",
    "transform_injuries",
    "transform_matches",
    "transform_teams",
    "transform_competitions",
]

