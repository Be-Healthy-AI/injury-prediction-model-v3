#!/usr/bin/env python3
"""
Cancun FC Benchmarking Analysis
Compares Cancun FC with other Liga de Expansión MX clubs across multiple KPIs.

KPIs:
- Injured vs Non-Injured Player/Weeks (absolute and percentage)
- Average Age per Club per Season
- Total Market Value per Club per Season
- Squad Size
- Injury Frequency
- Average Injury Duration
- Injury Severity Distribution

Output: CSV files and summary report in mexico/20260103_Cancun FC Benchmarking/
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Calculate paths
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent
ROOT_DIR = PRODUCTION_ROOT.parent

# Data directory
DATA_DIR = PRODUCTION_ROOT / "raw_data" / "mexico" / "20260102"
OUTPUT_DIR = PRODUCTION_ROOT / "raw_data" / "mexico" / "20260103_Cancun FC Benchmarking"

# Season definitions
SEASONS = {
    "2024/25": {
        "start": pd.Timestamp("2024-07-01"),
        "end": pd.Timestamp("2025-06-30"),
        "year": 2024
    },
    "2025/26": {
        "start": pd.Timestamp("2025-07-01"),
        "end": pd.Timestamp("2026-06-30"),
        "year": 2025
    }
}

WEEKS_PER_SEASON = 52


def load_all_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load all CSV files from the data directory."""
    print("Loading data files...")
    
    data = {}
    
    # Load profiles
    profiles_path = data_dir / "players_profile.csv"
    if profiles_path.exists():
        data['profiles'] = pd.read_csv(profiles_path, sep=';', encoding='utf-8-sig')
        print(f"  Loaded {len(data['profiles'])} profiles")
    else:
        raise FileNotFoundError(f"Profiles file not found: {profiles_path}")
    
    # Load injuries
    injuries_path = data_dir / "injuries_data.csv"
    if injuries_path.exists():
        data['injuries'] = pd.read_csv(injuries_path, sep=';', encoding='utf-8-sig')
        print(f"  Loaded {len(data['injuries'])} injury records")
    else:
        print(f"  Warning: Injuries file not found: {injuries_path}")
        data['injuries'] = pd.DataFrame()
    
    # Load career
    career_path = data_dir / "players_career.csv"
    if career_path.exists():
        data['career'] = pd.read_csv(career_path, sep=';', encoding='utf-8-sig')
        print(f"  Loaded {len(data['career'])} career records")
    else:
        raise FileNotFoundError(f"Career file not found: {career_path}")
    
    return data


def parse_season_string(season_str: str) -> Optional[int]:
    """Parse season string like '24/25' or '2024/25' to year (2024)."""
    if pd.isna(season_str):
        return None
    
    season_str = str(season_str).strip()
    if '/' in season_str:
        parts = season_str.split('/')
        if len(parts) == 2:
            year1 = parts[0]
            if len(year1) == 2:
                year1 = '20' + year1
            try:
                return int(year1)
            except ValueError:
                return None
    return None


def map_players_to_clubs_by_season(
    career_df: pd.DataFrame,
    season_start: pd.Timestamp,
    season_end: pd.Timestamp
) -> Dict[int, Dict[str, any]]:
    """
    Determine which club each player was at during each season.
    Returns: {player_id: {'club': club_name, 'market_value': value, 'start_date': date, 'end_date': date}}
    """
    print(f"Mapping players to clubs for season {season_start.year}/{season_end.year}...")
    
    player_clubs = {}
    
    # Convert Date column to datetime
    career_df = career_df.copy()
    career_df['Date'] = pd.to_datetime(career_df['Date'], errors='coerce')
    
    # Group by player
    for player_id, player_career in career_df.groupby('id'):
        # Sort by date descending (most recent first)
        player_career = player_career.sort_values('Date', ascending=False)
        
        # Find the club at season start
        # Get transfers before or at season start
        transfers_before_season = player_career[
            (player_career['Date'].notna()) & 
            (player_career['Date'] <= season_start)
        ]
        
        if not transfers_before_season.empty:
            # Get the most recent transfer before season start
            latest_transfer = transfers_before_season.iloc[0]
            club_name = latest_transfer['To']
            market_value = latest_transfer.get('VM', 0)
            
            # Try to convert market value to numeric
            try:
                if pd.isna(market_value) or market_value == '' or market_value == '-':
                    market_value = 0
                else:
                    market_value = float(str(market_value).replace(',', '').replace('€', '').strip())
            except (ValueError, AttributeError):
                market_value = 0
            
            # Check if player transferred during the season
            transfers_during_season = player_career[
                (player_career['Date'].notna()) & 
                (player_career['Date'] > season_start) &
                (player_career['Date'] <= season_end)
            ]
            
            if not transfers_during_season.empty:
                # Player transferred mid-season - use the club they joined
                mid_season_transfer = transfers_during_season.iloc[-1]  # First transfer during season
                club_name = mid_season_transfer['To']
                # Update market value if available
                try:
                    mv = mid_season_transfer.get('VM', 0)
                    if not (pd.isna(mv) or mv == '' or mv == '-'):
                        market_value = float(str(mv).replace(',', '').replace('€', '').strip())
                except (ValueError, AttributeError):
                    pass
            
            player_clubs[player_id] = {
                'club': club_name,
                'market_value': market_value,
                'start_date': season_start,
                'end_date': season_end
            }
    
    print(f"  Mapped {len(player_clubs)} players to clubs")
    return player_clubs


def is_week_injured(week_start: pd.Timestamp, week_end: pd.Timestamp, injury_periods: List[Tuple[pd.Timestamp, pd.Timestamp]]) -> bool:
    """
    Check if a week overlaps with any injury period.
    Returns True if any injury overlaps with the week (even partially).
    """
    for injury_start, injury_end in injury_periods:
        # Check for overlap: injury overlaps if it starts before week ends 
        # and ends after week starts
        if injury_start <= week_end and injury_end >= week_start:
            return True
    return False


def calculate_injury_weeks(
    player_id: int,
    season_start: pd.Timestamp,
    season_end: pd.Timestamp,
    injuries_df: pd.DataFrame
) -> Tuple[int, int, int]:
    """
    Calculate injured and non-injured weeks for a player in a season.
    If a player was injured for any part of a week (1-6 days), 
    count that entire week as injured.
    
    Returns: (injured_weeks, non_injured_weeks, total_weeks)
    """
    # Get player's injuries
    player_injuries = injuries_df[injuries_df['player_id'] == player_id].copy()
    
    if player_injuries.empty:
        # No injuries - all weeks are non-injured
        return (0, WEEKS_PER_SEASON, WEEKS_PER_SEASON)
    
    # Parse injury dates
    player_injuries['fromDate'] = pd.to_datetime(player_injuries['fromDate'], errors='coerce')
    player_injuries['untilDate'] = pd.to_datetime(player_injuries['untilDate'], errors='coerce')
    
    # Filter injuries that overlap with the season
    injury_periods = []
    for _, injury in player_injuries.iterrows():
        injury_start = injury['fromDate']
        injury_end = injury['untilDate']
        
        if pd.isna(injury_start):
            continue
        
        # If end date is missing, use season end
        if pd.isna(injury_end):
            injury_end = season_end
        
        # Clip to season boundaries
        injury_start = max(injury_start, season_start)
        injury_end = min(injury_end, season_end)
        
        if injury_start <= injury_end:
            injury_periods.append((injury_start, injury_end))
    
    if not injury_periods:
        # No injuries in this season
        return (0, WEEKS_PER_SEASON, WEEKS_PER_SEASON)
    
    # Divide season into weeks
    injured_weeks = set()
    current_date = season_start
    
    while current_date <= season_end:
        week_start = current_date
        week_end = min(current_date + timedelta(days=6), season_end)
        
        if is_week_injured(week_start, week_end, injury_periods):
            # Calculate week number (0-indexed)
            week_num = (current_date - season_start).days // 7
            injured_weeks.add(week_num)
        
        current_date += timedelta(days=7)
    
    total_weeks = WEEKS_PER_SEASON
    injured_weeks_count = len(injured_weeks)
    non_injured_weeks = total_weeks - injured_weeks_count
    
    return (injured_weeks_count, non_injured_weeks, total_weeks)


def calculate_player_age_at_season_start(
    player_id: int,
    season_start: pd.Timestamp,
    profiles_df: pd.DataFrame
) -> Optional[float]:
    """Get player age at season start."""
    player_profile = profiles_df[profiles_df['id'] == player_id]
    
    if player_profile.empty:
        return None
    
    date_of_birth = player_profile.iloc[0]['date_of_birth']
    
    if pd.isna(date_of_birth):
        return None
    
    try:
        if isinstance(date_of_birth, str):
            dob = pd.to_datetime(date_of_birth, errors='coerce')
        else:
            dob = pd.to_datetime(date_of_birth, errors='coerce')
        
        if pd.isna(dob):
            return None
        
        age = (season_start - dob).days / 365.25
        return age
    except (ValueError, TypeError):
        return None


def get_player_market_value_at_season_start(
    player_id: int,
    season_start: pd.Timestamp,
    career_df: pd.DataFrame
) -> float:
    """Get player market value at season start."""
    player_career = career_df[career_df['id'] == player_id].copy()
    
    if player_career.empty:
        return 0.0
    
    player_career['Date'] = pd.to_datetime(player_career['Date'], errors='coerce')
    
    # Get transfers before or at season start
    transfers_before = player_career[
        (player_career['Date'].notna()) & 
        (player_career['Date'] <= season_start)
    ]
    
    if transfers_before.empty:
        return 0.0
    
    # Get most recent transfer
    latest_transfer = transfers_before.iloc[0]
    market_value = latest_transfer.get('VM', 0)
    
    try:
        if pd.isna(market_value) or market_value == '' or market_value == '-':
            return 0.0
        else:
            return float(str(market_value).replace(',', '').replace('€', '').strip())
    except (ValueError, AttributeError):
        return 0.0


def calculate_additional_injury_kpis(
    player_id: int,
    season_start: pd.Timestamp,
    season_end: pd.Timestamp,
    injuries_df: pd.DataFrame
) -> Dict[str, any]:
    """Calculate additional injury KPIs for a player in a season."""
    player_injuries = injuries_df[injuries_df['player_id'] == player_id].copy()
    
    if player_injuries.empty:
        return {
            'num_injuries': 0,
            'avg_injury_duration': 0.0,
            'injuries_mild': 0,
            'injuries_moderate': 0,
            'injuries_severe': 0,
            'injuries_critical': 0
        }
    
    # Filter injuries in this season
    player_injuries['fromDate'] = pd.to_datetime(player_injuries['fromDate'], errors='coerce')
    player_injuries['untilDate'] = pd.to_datetime(player_injuries['untilDate'], errors='coerce')
    
    season_injuries = player_injuries[
        (player_injuries['fromDate'].notna()) &
        (
            ((player_injuries['fromDate'] >= season_start) & (player_injuries['fromDate'] <= season_end)) |
            ((player_injuries['untilDate'].notna()) & (player_injuries['untilDate'] >= season_start) & (player_injuries['untilDate'] <= season_end)) |
            ((player_injuries['fromDate'] < season_start) & ((player_injuries['untilDate'].isna()) | (player_injuries['untilDate'] > season_start)))
        )
    ]
    
    num_injuries = len(season_injuries)
    
    # Calculate average duration
    durations = []
    for _, injury in season_injuries.iterrows():
        days = injury.get('days', 0)
        try:
            if pd.notna(days):
                durations.append(float(days))
        except (ValueError, TypeError):
            pass
    
    avg_duration = np.mean(durations) if durations else 0.0
    
    # Count by severity
    severity_counts = season_injuries['severity'].value_counts().to_dict()
    
    return {
        'num_injuries': num_injuries,
        'avg_injury_duration': avg_duration,
        'injuries_mild': severity_counts.get('mild', 0),
        'injuries_moderate': severity_counts.get('moderate', 0),
        'injuries_severe': severity_counts.get('severe', 0),
        'injuries_critical': severity_counts.get('critical', 0)
    }


def aggregate_club_kpis(player_level_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate player-level data to club-level."""
    print("Aggregating club-level KPIs...")
    
    club_level_data = []
    
    for (club, season), group in player_level_df.groupby(['club', 'season']):
        club_data = {
            'club': club,
            'season': season,
            'total_injured_weeks': group['injured_weeks'].sum(),
            'total_non_injured_weeks': group['non_injured_weeks'].sum(),
            'total_player_weeks': group['total_weeks'].sum(),
            'club_injury_rate_pct': (group['injured_weeks'].sum() / group['total_weeks'].sum() * 100) if group['total_weeks'].sum() > 0 else 0.0,
            'average_age': group['age'].mean() if group['age'].notna().any() else None,
            'total_market_value': group['market_value'].sum(),
            'avg_market_value': group['market_value'].mean() if group['market_value'].notna().any() else 0.0,
            'squad_size': len(group),
            'num_injuries': group['num_injuries'].sum(),
            'avg_injury_duration': group['avg_injury_duration'].mean() if group['avg_injury_duration'].notna().any() else 0.0,
            'injuries_mild': group['injuries_mild'].sum(),
            'injuries_moderate': group['injuries_moderate'].sum(),
            'injuries_severe': group['injuries_severe'].sum(),
            'injuries_critical': group['injuries_critical'].sum()
        }
        club_level_data.append(club_data)
    
    return pd.DataFrame(club_level_data)


def generate_benchmarking_report(club_level_df: pd.DataFrame, output_dir: Path):
    """Generate summary report and comparisons."""
    print("Generating benchmarking report...")
    
    report_lines = []
    report_lines.append("# Cancun FC Benchmarking Analysis Report")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Filter Cancun FC data
    cancun_data = club_level_df[club_level_df['club'].str.contains('Cancún|Cancun', case=False, na=False)]
    other_clubs = club_level_df[~club_level_df['club'].str.contains('Cancún|Cancun', case=False, na=False)]
    
    for season in ['2024/25', '2025/26']:
        report_lines.append(f"## Season {season}")
        report_lines.append("")
        
        season_cancun = cancun_data[cancun_data['season'] == season]
        season_others = other_clubs[other_clubs['season'] == season]
        
        if season_cancun.empty:
            report_lines.append(f"  No data for Cancun FC in {season}")
            report_lines.append("")
            continue
        
        cancun_row = season_cancun.iloc[0]
        
        # League averages
        league_avg_injury_rate = season_others['club_injury_rate_pct'].mean()
        league_avg_age = season_others['average_age'].mean()
        league_avg_market_value = season_others['total_market_value'].mean()
        
        report_lines.append("### Cancun FC vs League Averages")
        report_lines.append("")
        report_lines.append(f"**Injury Rate:**")
        report_lines.append(f"  - Cancun FC: {cancun_row['club_injury_rate_pct']:.2f}%")
        report_lines.append(f"  - League Average: {league_avg_injury_rate:.2f}%")
        report_lines.append(f"  - Difference: {cancun_row['club_injury_rate_pct'] - league_avg_injury_rate:+.2f}%")
        report_lines.append("")
        
        report_lines.append(f"**Average Age:**")
        if pd.notna(cancun_row['average_age']):
            report_lines.append(f"  - Cancun FC: {cancun_row['average_age']:.1f} years")
        if pd.notna(league_avg_age):
            report_lines.append(f"  - League Average: {league_avg_age:.1f} years")
            if pd.notna(cancun_row['average_age']):
                report_lines.append(f"  - Difference: {cancun_row['average_age'] - league_avg_age:+.1f} years")
        report_lines.append("")
        
        report_lines.append(f"**Total Market Value:**")
        report_lines.append(f"  - Cancun FC: €{cancun_row['total_market_value']:,.0f}")
        report_lines.append(f"  - League Average: €{league_avg_market_value:,.0f}")
        report_lines.append(f"  - Difference: €{cancun_row['total_market_value'] - league_avg_market_value:+,.0f}")
        report_lines.append("")
        
        # Rankings
        report_lines.append("### Rankings")
        report_lines.append("")
        
        # Injury rate ranking (lower is better)
        sorted_injury = season_others.sort_values('club_injury_rate_pct')
        cancun_rank_injury = (sorted_injury['club_injury_rate_pct'] < cancun_row['club_injury_rate_pct']).sum() + 1
        total_clubs = len(season_others) + 1
        report_lines.append(f"**Injury Rate:** Cancun FC ranks #{cancun_rank_injury} out of {total_clubs} clubs (lower is better)")
        report_lines.append("")
        
        # Market value ranking (higher is better)
        sorted_market = season_others.sort_values('total_market_value', ascending=False)
        cancun_rank_market = (sorted_market['total_market_value'] > cancun_row['total_market_value']).sum() + 1
        report_lines.append(f"**Market Value:** Cancun FC ranks #{cancun_rank_market} out of {total_clubs} clubs (higher is better)")
        report_lines.append("")
        
        report_lines.append("=" * 80)
        report_lines.append("")
    
    # Write report
    report_path = output_dir / "benchmarking_summary_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"  Report saved to: {report_path}")


def main():
    print("=" * 80)
    print("Cancun FC Benchmarking Analysis")
    print("=" * 80)
    print()
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data = load_all_data(DATA_DIR)
    
    # Prepare DataFrames
    profiles_df = data['profiles']
    injuries_df = data['injuries']
    career_df = data['career']
    
    # Process each season
    player_level_results = []
    
    for season_name, season_info in SEASONS.items():
        print(f"\n{'='*80}")
        print(f"Processing Season: {season_name}")
        print(f"{'='*80}")
        
        season_start = season_info['start']
        season_end = season_info['end']
        
        # Map players to clubs
        player_clubs = map_players_to_clubs_by_season(career_df, season_start, season_end)
        
        # Process each player
        print(f"Calculating KPIs for {len(player_clubs)} players...")
        
        for player_id, club_info in player_clubs.items():
            club_name = club_info['club']
            
            # Skip if club name is missing or invalid
            if pd.isna(club_name) or club_name == '' or club_name == 'Unknown':
                continue
            
            # Calculate injury weeks
            injured_weeks, non_injured_weeks, total_weeks = calculate_injury_weeks(
                player_id, season_start, season_end, injuries_df
            )
            
            # Calculate age
            age = calculate_player_age_at_season_start(
                player_id, season_start, profiles_df
            )
            
            # Get market value
            market_value = get_player_market_value_at_season_start(
                player_id, season_start, career_df
            )
            
            # Calculate additional injury KPIs
            injury_kpis = calculate_additional_injury_kpis(
                player_id, season_start, season_end, injuries_df
            )
            
            # Get player info
            player_profile = profiles_df[profiles_df['id'] == player_id]
            player_name = player_profile.iloc[0]['name'] if not player_profile.empty else f"Player {player_id}"
            position = player_profile.iloc[0].get('position', '') if not player_profile.empty else ''
            
            # Calculate injury rate
            injury_rate = (injured_weeks / total_weeks * 100) if total_weeks > 0 else 0.0
            
            player_result = {
                'player_id': player_id,
                'player_name': player_name,
                'club': club_name,
                'season': season_name,
                'position': position,
                'injured_weeks': injured_weeks,
                'non_injured_weeks': non_injured_weeks,
                'total_weeks': total_weeks,
                'injury_rate_pct': injury_rate,
                'age': age,
                'market_value': market_value,
                'num_injuries': injury_kpis['num_injuries'],
                'avg_injury_duration': injury_kpis['avg_injury_duration'],
                'injuries_mild': injury_kpis['injuries_mild'],
                'injuries_moderate': injury_kpis['injuries_moderate'],
                'injuries_severe': injury_kpis['injuries_severe'],
                'injuries_critical': injury_kpis['injuries_critical']
            }
            
            player_level_results.append(player_result)
    
    # Create player-level DataFrame
    player_level_df = pd.DataFrame(player_level_results)
    
    # Aggregate to club level
    club_level_df = aggregate_club_kpis(player_level_df)
    
    # Save results
    print(f"\n{'='*80}")
    print("Saving results...")
    print(f"{'='*80}")
    
    # Player-level results
    player_output_path = OUTPUT_DIR / "benchmarking_player_level.csv"
    player_level_df.to_csv(player_output_path, index=False, encoding='utf-8-sig', sep=';')
    print(f"  Player-level data: {player_output_path} ({len(player_level_df)} records)")
    
    # Club-level results
    club_output_path = OUTPUT_DIR / "benchmarking_club_level.csv"
    club_level_df.to_csv(club_output_path, index=False, encoding='utf-8-sig', sep=';')
    print(f"  Club-level data: {club_output_path} ({len(club_level_df)} records)")
    
    # Cancun FC focus comparison
    cancun_focus = club_level_df.copy()
    cancun_clubs = cancun_focus[cancun_focus['club'].str.contains('Cancún|Cancun', case=False, na=False)]['club'].unique()
    
    if len(cancun_clubs) > 0:
        cancun_focus_path = OUTPUT_DIR / "benchmarking_cancun_fc_focus.csv"
        cancun_focus.to_csv(cancun_focus_path, index=False, encoding='utf-8-sig', sep=';')
        print(f"  Cancun FC focus: {cancun_focus_path}")
    
    # Generate report
    generate_benchmarking_report(club_level_df, OUTPUT_DIR)
    
    print(f"\n{'='*80}")
    print("Analysis Complete!")
    print(f"{'='*80}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()


if __name__ == "__main__":
    main()




