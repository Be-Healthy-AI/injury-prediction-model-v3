"""
Transform scraped FBRef tables into standardized schemas for injury prediction models.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

import pandas as pd


LOGGER = logging.getLogger(__name__)

# FBRef Match Statistics Schema
FBREF_MATCH_COLUMNS = [
    "fbref_player_id",
    "match_date",
    "season",
    "competition",
    "round",
    "venue",
    "team",
    "opponent",
    "result",
    "position",
    "minutes",
    # Passing
    "passes_completed",
    "passes_attempted",
    "pass_accuracy_pct",
    "key_passes",
    "progressive_passes",
    "passes_into_final_third",
    "crosses",
    "through_balls",
    # Shooting
    "shots",
    "shots_on_target",
    "goals",
    "xG",
    "npxG",
    "shots_from_outside_box",
    # Defensive
    "tackles",
    "tackles_won",
    "interceptions",
    "blocks",
    "clearances",
    "pressures",
    "pressure_regains",
    # Possession
    "touches",
    "touches_in_box",
    "progressive_carries",
    "carries_into_final_third",
    "dribbles_attempted",
    "dribbles_successful",
    "times_dispossessed",
    "miscontrols",
    # Physical
    "distance_covered_km",
    "sprints",
    "accelerations",
    # Advanced
    "shot_creating_actions",
    "goal_creating_actions",
    "aerial_duels_won",
    "aerial_duels_lost",
]


def transform_fbref_match_log(df: pd.DataFrame, fbref_player_id: str) -> pd.DataFrame:
    """
    Transform raw FBRef match log DataFrame into standardized schema.
    
    Args:
        df: Raw DataFrame from FBRef scraper (may have multi-level headers)
        fbref_player_id: FBRef player ID
    
    Returns:
        DataFrame with standardized columns matching FBREF_MATCH_COLUMNS
    """
    if df.empty:
        return pd.DataFrame(columns=FBREF_MATCH_COLUMNS)
    
    df = df.copy()
    
    # Flatten multi-level column headers
    if isinstance(df.columns, pd.MultiIndex):
        # Join multi-level column names
        df.columns = ['_'.join(col).strip('_') for col in df.columns.values]
    
    # Clean column names: remove special characters and normalize
    df.columns = [col.strip().replace('\n', ' ').replace('\t', ' ') for col in df.columns]
    
    # Map FBRef column names to our schema
    # FBRef columns can vary, so we'll try multiple possible names
    column_mapping = _build_column_mapping(df.columns)
    
    # Create output DataFrame with same index as input (so scalar assignments broadcast correctly)
    output = pd.DataFrame(index=df.index)
    output['fbref_player_id'] = fbref_player_id  # This will broadcast to all rows
    
    # Extract and transform date
    date_col = _find_column(df, ['Date', 'date', 'Match Date', 'Unnamed: 0_level_0_Date'])
    if date_col:
        output['match_date'] = _parse_fbref_date(df[date_col])
    else:
        output['match_date'] = pd.NaT
    
    # Extract season (from metadata if available, otherwise from date)
    if 'season' in df.columns:
        output['season'] = df['season']
    elif 'season_' in df.columns:
        output['season'] = df['season_']
    else:
        # Derive from date
        output['season'] = output['match_date'].apply(_derive_season_from_date)
    
    # Extract competition
    comp_col = _find_column(df, ['Comp', 'Competition', 'competition', 'Unnamed: 2_level_0_Comp'])
    if comp_col:
        output['competition'] = df[comp_col].apply(_normalize_fbref_competition)
    else:
        output['competition'] = None
    
    # Extract round
    round_col = _find_column(df, ['Round', 'round', 'Unnamed: 3_level_0_Round'])
    if round_col:
        output['round'] = df[round_col]
    else:
        output['round'] = None
    
    # Extract venue
    venue_col = _find_column(df, ['Venue', 'venue', 'Unnamed: 4_level_0_Venue'])
    if venue_col:
        output['venue'] = df[venue_col]
    else:
        output['venue'] = None
    
    # Extract team
    team_col = _find_column(df, ['Squad', 'Team', 'team', 'Unnamed: 6_level_0_Squad'])
    if team_col:
        output['team'] = df[team_col].apply(_normalize_fbref_team)
    else:
        output['team'] = None
    
    # Extract opponent
    opp_col = _find_column(df, ['Opponent', 'opponent', 'Opp', 'Unnamed: 7_level_0_Opponent'])
    if opp_col:
        output['opponent'] = df[opp_col].apply(_normalize_fbref_team)
    else:
        output['opponent'] = None
    
    # Extract result
    result_col = _find_column(df, ['Result', 'result', 'Unnamed: 5_level_0_Result'])
    if result_col:
        output['result'] = df[result_col]
    else:
        output['result'] = None
    
    # Extract position
    pos_col = _find_column(df, ['Pos', 'Position', 'position', 'Start', 'Unnamed: 9_level_0_Pos', 'Unnamed: 8_level_0_Start'])
    if pos_col:
        output['position'] = df[pos_col].apply(_normalize_fbref_position)
    else:
        output['position'] = None
    
    # Extract minutes
    min_col = _find_column(df, ['Min', 'Minutes', 'minutes', 'Unnamed: 10_level_0_Min'])
    if min_col:
        output['minutes'] = _parse_minutes(df[min_col])
    else:
        output['minutes'] = pd.NA
    
    # Extract passing statistics
    output['passes_completed'] = _extract_numeric(df, ['Cmp', 'Passes Completed', 'Cmp_Passes'])
    output['passes_attempted'] = _extract_numeric(df, ['Att', 'Passes Attempted', 'Att_Passes'])
    output['pass_accuracy_pct'] = _extract_numeric(df, ['Cmp%', 'Pass Accuracy %', 'Cmp%_Passes'])
    output['key_passes'] = _extract_numeric(df, ['KP', 'Key Passes'])
    output['progressive_passes'] = _extract_numeric(df, ['Prog', 'Progressive Passes', 'Prog_Passes'])
    output['passes_into_final_third'] = _extract_numeric(df, ['1/3', 'Passes into Final Third'])
    output['crosses'] = _extract_numeric(df, ['Crs', 'Crosses'])
    output['through_balls'] = _extract_numeric(df, ['TB', 'Through Balls'])
    
    # Extract shooting statistics
    output['shots'] = _extract_numeric(df, ['Sh', 'Shots'])
    output['shots_on_target'] = _extract_numeric(df, ['SoT', 'Shots on Target'])
    output['goals'] = _extract_numeric(df, ['Gls', 'Goals', 'Performance_Gls'])
    output['xG'] = _extract_numeric(df, ['xG', 'Expected Goals'])
    output['npxG'] = _extract_numeric(df, ['npxG', 'Non-Penalty xG'])
    output['shots_from_outside_box'] = _extract_numeric(df, ['Sh/90', 'Shots from Outside Box'])
    
    # Extract defensive statistics
    output['tackles'] = _extract_numeric(df, ['Tkl', 'Tackles'])
    output['tackles_won'] = _extract_numeric(df, ['TklW', 'Tackles Won', 'Performance_TklW'])
    output['interceptions'] = _extract_numeric(df, ['Int', 'Interceptions', 'Performance_Int'])
    output['blocks'] = _extract_numeric(df, ['Blocks', 'Blk'])
    output['clearances'] = _extract_numeric(df, ['Clr', 'Clearances'])
    output['pressures'] = _extract_numeric(df, ['Press', 'Pressures'])
    output['pressure_regains'] = _extract_numeric(df, ['Succ', 'Successful Pressures'])
    
    # Extract possession statistics
    output['touches'] = _extract_numeric(df, ['Touches', 'Tch'])
    output['touches_in_box'] = _extract_numeric(df, ['Touches in Box', 'Touches_in_Box'])
    output['progressive_carries'] = _extract_numeric(df, ['Prog', 'Progressive Carries', 'Prog_Carries'])
    output['carries_into_final_third'] = _extract_numeric(df, ['1/3', 'Carries into Final Third'])
    output['dribbles_attempted'] = _extract_numeric(df, ['Att', 'Dribbles Attempted', 'Att_Dribbles'])
    output['dribbles_successful'] = _extract_numeric(df, ['Succ', 'Successful Dribbles', 'Succ_Dribbles'])
    output['times_dispossessed'] = _extract_numeric(df, ['Dispossessed', 'Dis'])
    output['miscontrols'] = _extract_numeric(df, ['Mis', 'Miscontrols'])
    
    # Extract physical statistics (may not always be available)
    output['distance_covered_km'] = _extract_numeric(df, ['Distance', 'Dist', 'km'])
    output['sprints'] = _extract_numeric(df, ['Spr', 'Sprints'])
    output['accelerations'] = _extract_numeric(df, ['Acc', 'Accelerations'])
    
    # Extract advanced statistics
    output['shot_creating_actions'] = _extract_numeric(df, ['SCA', 'Shot-Creating Actions'])
    output['goal_creating_actions'] = _extract_numeric(df, ['GCA', 'Goal-Creating Actions'])
    output['aerial_duels_won'] = _extract_numeric(df, ['Won', 'Aerial Duels Won'])
    output['aerial_duels_lost'] = _extract_numeric(df, ['Lost', 'Aerial Duels Lost'])
    
    # Ensure all required columns exist
    for col in FBREF_MATCH_COLUMNS:
        if col not in output.columns:
            output[col] = pd.NA
    
    # Reorder columns to match schema
    output = output[FBREF_MATCH_COLUMNS]
    
    # Remove rows where match_date is missing (these are usually summary rows)
    output = output[output['match_date'].notna()].copy()
    
    # Reset index
    output = output.reset_index(drop=True)
    
    return output


def _build_column_mapping(columns: pd.Index) -> Dict[str, str]:
    """Build mapping from FBRef column names to standardized names."""
    mapping = {}
    for col in columns:
        col_lower = col.lower()
        # This is a placeholder - actual mapping happens in transform function
        mapping[col] = col
    return mapping


def _find_column(df: pd.DataFrame, possible_names: list[str]) -> Optional[str]:
    """Find a column by trying multiple possible names."""
    for name in possible_names:
        # Try exact match
        if name in df.columns:
            return name
        # Try case-insensitive match
        for col in df.columns:
            if col.lower() == name.lower():
                return col
        # Try partial match
        for col in df.columns:
            if name.lower() in col.lower() or col.lower() in name.lower():
                return col
    return None


def _extract_numeric(df: pd.DataFrame, possible_names: list[str]) -> pd.Series:
    """Extract numeric column, trying multiple possible names."""
    col = _find_column(df, possible_names)
    if col:
        return pd.to_numeric(df[col], errors='coerce')
    # Return Series with NaN instead of pd.NA for float dtype compatibility
    return pd.Series([float('nan')] * len(df), dtype='float64')


def _parse_fbref_date(date_series: pd.Series) -> pd.Series:
    """Parse FBRef date formats."""
    def parse_date(val):
        if pd.isna(val):
            return pd.NaT
        val_str = str(val).strip()
        # Try common date formats
        for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S']:
            try:
                return pd.to_datetime(val_str, format=fmt)
            except:
                continue
        # Try pandas auto-parsing
        try:
            return pd.to_datetime(val_str)
        except:
            return pd.NaT
    
    return date_series.apply(parse_date)


def _parse_minutes(minutes_series: pd.Series) -> pd.Series:
    """Parse minutes played (handles formats like '90', '90+3', etc.)."""
    def parse_min(val):
        if pd.isna(val):
            return pd.NA
        val_str = str(val).strip()
        # Remove quotes and extract number
        val_str = val_str.replace("'", "").replace('"', '').strip()
        # Handle injury time (e.g., "90+3" -> 93)
        if "+" in val_str:
            parts = val_str.split("+")
            try:
                base = int(parts[0])
                extra = int(parts[1]) if len(parts) > 1 else 0
                return base + extra
            except:
                pass
        # Try to extract number
        try:
            return int(float(val_str))
        except:
            return pd.NA
    
    return minutes_series.apply(parse_min).astype('Int64')


def _normalize_fbref_competition(comp: Any) -> Optional[str]:
    """Normalize FBRef competition names."""
    if pd.isna(comp):
        return None
    comp_str = str(comp).strip()
    # Common mappings
    comp_map = {
        'Premier League': 'Premier-League',
        'Championship': 'Championship',
        'FA Cup': 'FA-Cup',
        'EFL Cup': 'EFL-Cup',
        'Champions League': 'Champions-League',
        'Europa League': 'Europa-League',
    }
    return comp_map.get(comp_str, comp_str)


def _normalize_fbref_team(team: Any) -> Optional[str]:
    """Normalize FBRef team names."""
    if pd.isna(team):
        return None
    team_str = str(team).strip()
    # Remove common suffixes/prefixes
    team_str = re.sub(r'\s*\(\d+\)\s*', '', team_str)  # Remove rankings like "(1)"
    return team_str


def _normalize_fbref_position(pos: Any) -> Optional[str]:
    """Normalize FBRef position names to match TransferMarkt format."""
    if pd.isna(pos):
        return None
    pos_str = str(pos).strip()
    # FBRef position abbreviations
    pos_map = {
        'GK': 'Goalkeeper',
        'CB': 'Centre Back',
        'LB': 'Left Back',
        'RB': 'Right Back',
        'DM': 'Defensive Midfielder',
        'CM': 'Central Midfielder',
        'AM': 'Attacking Midfielder',
        'LW': 'Left Winger',
        'RW': 'Right Winger',
        'CF': 'Centre Forward',
        'ST': 'Centre Forward',
        'SS': 'Second Striker',
    }
    return pos_map.get(pos_str.upper(), pos_str)


def _derive_season_from_date(date: pd.Timestamp) -> Optional[str]:
    """Derive season string (e.g., '2024-2025') from date."""
    if pd.isna(date):
        return None
    year = date.year
    month = date.month
    # Season runs from July to June
    if month >= 7:
        return f"{year}-{year+1}"
    else:
        return f"{year-1}-{year}"


__all__ = [
    "transform_fbref_match_log",
    "FBREF_MATCH_COLUMNS",
]

