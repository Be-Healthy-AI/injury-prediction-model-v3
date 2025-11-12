#!/usr/bin/env python3
"""
Enhanced 35-Day Timeline Generator V3 for Injury Prediction
Combines enhanced features with 35-day timeline logic
Implements player-by-player processing for optimal performance
Balanced dataset with configurable target ratio
Based on V2 with V3 dataset support
"""

import sys
import io
# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Configuration
WINDOW_SIZE = 35  # 5 weeks
TARGET_RATIO = 0.15  # 15% injury ratio

def get_static_features() -> List[str]:
    """Get features that remain static (not windowed)"""
    return [
        'player_id', 'reference_date', 'player_name',  # Technical features
        'position', 'nationality1', 'nationality2', 'height_cm', 'dominant_foot',
        'previous_club', 'previous_club_country', 'current_club', 'current_club_country',
        # Career cumulative features
        'cum_minutes_played_numeric', 'cum_goals_numeric', 'cum_assists_numeric',
        'cum_yellow_cards_numeric', 'cum_red_cards_numeric', 'cum_matches_played',
        'cum_matches_bench', 'cum_matches_not_selected', 'cum_competitions',
        'matches', 'career_matches', 'career_goals', 'career_assists', 'career_minutes',
        # Career injury features
        'cum_inj_starts', 'cum_inj_days', 'avg_injury_severity', 'max_injury_severity',
        'lower_leg_injuries', 'knee_injuries', 'upper_leg_injuries', 'hip_injuries',
        'upper_body_injuries', 'head_injuries', 'illness_count', 'other_injuries', 'cum_matches_injured',
        # Career competition features
        'avg_competition_importance', 'cum_disciplinary_actions', 'teams_last_season',
        'national_team_appearances', 'national_team_minutes', 'national_team_last_season',
        'national_team_frequency', 'senior_national_team', 'competition_intensity',
        'competition_level', 'competition_diversity', 'international_competitions',
        'cup_competitions', 'competition_frequency', 'competition_experience',
        'competition_pressure', 'teams_today', 'cum_teams', 'seasons_count',
        # Career club features
        'club_cum_goals', 'club_cum_assists', 'club_cum_minutes',
        'club_cum_matches_played', 'club_cum_yellow_cards', 'club_cum_red_cards',
        # Transfermarkt features
        'transfermarkt_score_recent', 'transfermarkt_score_cum', 'transfermarkt_score_avg',
        'transfermarkt_score_rolling5', 'transfermarkt_score_matches',
        # Interaction features
        'age_x_career_matches', 'age_x_career_goals', 'seniority_x_goals_per_match'
    ]

def get_windowed_features() -> List[str]:
    """Get features that will be windowed (35 days)"""
    return [
        # Daily performance metrics
        'matches_played', 'minutes_played_numeric', 'goals_numeric', 'assists_numeric',
        'yellow_cards_numeric', 'red_cards_numeric', 'matches_bench_unused', 
        'matches_not_selected', 'matches_injured',
        # Recent patterns
        'days_since_last_match', 'last_match_position', 'position_match_default',
        'disciplinary_action', 'goals_per_match', 'assists_per_match', 'minutes_per_match',
        # Recent injury indicators
        'days_since_last_injury', 'days_since_last_injury_ended', 'avg_injury_duration', 'injury_frequency',
        'physio_injury_ratio',
        # Recent competition context
        'competition_importance', 'month', 'teams_this_season',
        'teams_season_today', 'season_team_diversity',
        # Recent national team activity
        'days_since_last_national_match', 'national_team_this_season',
        'national_team_intensity',
        # Recent club performance
        'club_goals_per_match', 'club_assists_per_match', 'club_minutes_per_match',
        'club_seniority_x_goals_per_match',
        # Match location
        'home_matches', 'away_matches',
        # Team result features
        'team_win', 'team_draw', 'team_loss', 'team_points',
        'cum_team_wins', 'cum_team_draws', 'cum_team_losses',
        'team_win_rate', 'cum_team_points', 'team_points_rolling5', 'team_mood_score',
        # Substitution patterns
        'substitution_on_count', 'substitution_off_count', 'late_substitution_on_count',
        'early_substitution_off_count', 'impact_substitution_count', 'tactical_substitution_count',
        'substitution_minutes_played', 'substitution_efficiency', 'substitution_mood_indicator',
        'consecutive_substitutions'
    ]

def create_windowed_features_vectorized(df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> Dict:
    """Create windowed features using vectorized operations"""
    # Filter data for the 35-day window
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    window_data = df[mask].copy()
    
    if len(window_data) != 35:
        return None  # Incomplete window
    
    # Pre-calculate week boundaries
    week_starts = [start_date + timedelta(days=i * 7) for i in range(5)]
    week_ends = [start_date + timedelta(days=i * 7 + 6) for i in range(5)]
    
    weekly_features = {}
    
    # Vectorized weekly aggregation
    for week in range(5):
        week_mask = (window_data['date'] >= week_starts[week]) & (window_data['date'] <= week_ends[week])
        week_data = window_data[week_mask]
        
        if len(week_data) != 7:
            return None  # Incomplete week
        
        # Define aggregation strategies for enhanced features
        last_value_features = ['last_match_position', 'position_match_default', 'disciplinary_action',
                              'competition_importance', 'month', 'teams_this_season',
                              'teams_season_today', 'season_team_diversity', 'national_team_this_season',
                              'national_team_intensity', 'club_goals_per_match', 'club_assists_per_match',
                              'club_minutes_per_match', 'club_seniority_x_goals_per_match', 'team_win_rate']
        
        sum_features = ['matches_played', 'minutes_played_numeric', 'goals_numeric', 'assists_numeric',
                       'yellow_cards_numeric', 'red_cards_numeric', 'matches_bench_unused',
                       'matches_not_selected', 'matches_injured', 'substitution_on_count',
                       'substitution_off_count', 'late_substitution_on_count', 'early_substitution_off_count',
                       'impact_substitution_count', 'tactical_substitution_count', 'substitution_minutes_played',
                       'consecutive_substitutions']
        
        mean_features = ['goals_per_match', 'assists_per_match', 'minutes_per_match',
                        'substitution_efficiency', 'substitution_mood_indicator',
                        'avg_injury_duration', 'injury_frequency', 'physio_injury_ratio']
        
        min_features = ['days_since_last_match', 'days_since_last_injury', 'days_since_last_national_match']
        
        # Apply aggregations
        for feature in get_windowed_features():
            if feature in week_data.columns:
                if feature in last_value_features:
                    weekly_features[f'{feature}_week_{week+1}'] = week_data[feature].iloc[-1]
                elif feature in sum_features:
                    weekly_features[f'{feature}_week_{week+1}'] = week_data[feature].sum()
                elif feature in mean_features:
                    weekly_features[f'{feature}_week_{week+1}'] = week_data[feature].mean()
                elif feature in min_features:
                    weekly_features[f'{feature}_week_{week+1}'] = week_data[feature].min()
                else:
                    weekly_features[f'{feature}_week_{week+1}'] = week_data[feature].mean()
    
    return weekly_features

def generate_injury_timelines_enhanced(player_id: int, player_name: str, df: pd.DataFrame) -> List[Dict]:
    """Generate injury timelines - ALL injuries get 5 timelines each"""
    timelines = []
    
    # Vectorized injury start detection
    injury_starts = df[df['cum_inj_starts'] > df['cum_inj_starts'].shift(1)]
    
    for _, injury_row in injury_starts.iterrows():
        injury_date = injury_row['date']
        
        # Generate 5 timelines (D-1, D-2, D-3, D-4, D-5) - ALL OF THEM
        for days_before in range(1, 6):
            reference_date = injury_date - timedelta(days=days_before)
            start_date = reference_date - timedelta(days=34)  # 35 days before reference
            
            if start_date < df['date'].min():
                continue
            
            # Create windowed features
            windowed_features = create_windowed_features_vectorized(df, start_date, reference_date)
            if windowed_features is None:
                continue
            
            # Get static features from reference date
            ref_mask = df['date'] == reference_date
            if not ref_mask.any():
                continue
                
            ref_row = df[ref_mask].iloc[0]
            
            # Build timeline
            timeline = build_timeline(player_id, player_name, reference_date, ref_row, windowed_features, target=1)
            timelines.append(timeline)
    
    return timelines

def get_valid_non_injury_dates(df: pd.DataFrame) -> List[pd.Timestamp]:
    """Get all valid non-injury reference dates (no injury in next 35 days)"""
    valid_dates = []
    
    # Vectorized injury start detection
    injury_starts = df[df['cum_inj_starts'] > df['cum_inj_starts'].shift(1)]['date'].values
    
    max_date = df['date'].max()
    min_date = df['date'].min()
    max_reference_date = max_date - timedelta(days=34)
    
    # Create date range for potential reference dates
    potential_dates = pd.date_range(min_date, max_reference_date, freq='D')
    
    # Process valid dates
    for reference_date in potential_dates:
        # Check if reference date exists in data
        if reference_date not in df['date'].values:
            continue
        
        # CRITICAL: Check if there's an injury in the next 35 days
        future_end = reference_date + timedelta(days=34)
        if future_end > max_date:
            continue
        
        # Check if any injury starts fall in this 35-day window
        injury_in_window = np.any((injury_starts >= reference_date) & (injury_starts <= future_end))
        if injury_in_window:
            continue  # Skip this date - injury in next 35 days
        
        # Check if we can create a complete 35-day window
        start_date = reference_date - timedelta(days=34)
        if start_date < min_date:
            continue
        
        valid_dates.append(reference_date)
    
    return valid_dates

def generate_non_injury_timelines_for_dates(player_id: int, player_name: str, df: pd.DataFrame, 
                                            reference_dates: List[pd.Timestamp]) -> List[Dict]:
    """Generate non-injury timelines for specific reference dates"""
    timelines = []
    
    for reference_date in reference_dates:
        # Create windowed features
        start_date = reference_date - timedelta(days=34)
        windowed_features = create_windowed_features_vectorized(df, start_date, reference_date)
        if windowed_features is None:
            continue
        
        # Get static features from reference date
        ref_mask = df['date'] == reference_date
        if not ref_mask.any():
            continue
        ref_row = df[ref_mask].iloc[0]
        
        # Build timeline
        timeline = build_timeline(player_id, player_name, reference_date, ref_row, windowed_features, target=0)
        timelines.append(timeline)
    
    return timelines

def build_timeline(player_id: int, player_name: str, reference_date: pd.Timestamp, 
                  ref_row: pd.Series, windowed_features: Dict, target: int) -> Dict:
    """Build a complete timeline"""
    # Build static features
    static_features = {
        'player_id': player_id,
        'reference_date': reference_date.strftime('%Y-%m-%d'),
        'player_name': player_name
    }
    
    # Add other static features
    for feature in get_static_features()[3:]:  # Skip technical features
        if feature in ref_row.index:
            static_features[feature] = ref_row[feature]
        else:
            static_features[feature] = None
    
    # Combine features
    timeline = {**static_features, **windowed_features, 'target': target}
    return timeline

def get_all_player_ids() -> List[int]:
    """Get all player IDs from the daily features directory"""
    player_ids = []
    # V3: Use daily_features_output directory
    daily_features_dir = 'daily_features_output'
    if not os.path.exists(daily_features_dir):
        daily_features_dir = '../daily_features_output'  # Fallback if run from scripts directory
    
    if not os.path.exists(daily_features_dir):
        raise FileNotFoundError(f"Daily features directory not found: {daily_features_dir}")
    
    for filename in os.listdir(daily_features_dir):
        if filename.startswith('player_') and filename.endswith('_daily_features.csv'):
            player_id = int(filename.split('_')[1])
            player_ids.append(player_id)
    return sorted(player_ids)

def get_player_name(player_id: int) -> str:
    """Get player name from the players data"""
    try:
        # Try current V3 file first
        players_file = 'original_data/20251106_players_profile.xlsx'
        if not os.path.exists(players_file):
            players_file = '../original_data/20251106_players_profile.xlsx'
        
        players_df = pd.read_excel(players_file)
        player_row = players_df[players_df['id'] == player_id]
        if not player_row.empty:
            return player_row.iloc[0]['name']
        else:
            return f"Player_{player_id}"
    except:
        return f"Player_{player_id}"

def main(max_players: Optional[int] = None):
    """Main function with optimized balanced dataset generation
    
    Args:
        max_players: Optional limit on number of players to process (for testing)
    """
    print("🚀 ENHANCED 35-DAY TIMELINE GENERATOR (OPTIMIZED)")
    print("=" * 60)
    print("📋 Features: 108 enhanced features with 35-day windows")
    print("⚡ Processing: Optimized two-pass approach")
    print("🎯 Target: 15% injury ratio with smart sampling")
    
    start_time = datetime.now()
    
    # Get all player IDs
    all_player_ids = get_all_player_ids()
    print(f"Found {len(all_player_ids)} players")
    
    # Apply limit if specified (for testing)
    if max_players is not None:
        player_ids = all_player_ids[:max_players]
        print(f"🧪 TEST MODE: Processing {len(player_ids)} players (limited from {len(all_player_ids)})")
    else:
        player_ids = all_player_ids
    
    # Determine daily features directory
    daily_features_dir = 'daily_features_output'
    if not os.path.exists(daily_features_dir):
        daily_features_dir = '../daily_features_output'
    
    # ===== PASS 1: Generate all injury timelines and collect valid non-injury dates =====
    print("\n" + "=" * 60)
    print("📊 PASS 1: Generating injury timelines and identifying valid dates")
    print("=" * 60)
    
    all_injury_timelines = []
    all_valid_non_injury_dates = []  # Store (player_id, reference_date) tuples
    processed_players = 0
    
    for player_id in tqdm(player_ids, desc="Pass 1: Processing players", unit="player"):
        try:
            # Load player data
            df = pd.read_csv(f'{daily_features_dir}/player_{player_id}_daily_features.csv')
            df['date'] = pd.to_datetime(df['date'])
            
            # Get player name
            player_name = get_player_name(player_id)
            
            # Generate all injury timelines (5 per injury)
            injury_timelines = generate_injury_timelines_enhanced(player_id, player_name, df)
            all_injury_timelines.extend(injury_timelines)
            
            # Get all valid non-injury dates for this player
            valid_dates = get_valid_non_injury_dates(df)
            for date in valid_dates:
                all_valid_non_injury_dates.append((player_id, date))
            
            processed_players += 1
            
            # Progress update every 10 players
            if processed_players % 10 == 0:
                print(f"\n📊 Progress: {processed_players}/{len(player_ids)} players")
                print(f"   Injury timelines: {len(all_injury_timelines)}")
                print(f"   Valid non-injury dates: {len(all_valid_non_injury_dates)}")
                
            # Memory optimization: clear player data
            del df
                
        except Exception as e:
            print(f"\n❌ Error processing player {player_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    pass1_time = datetime.now() - start_time
    print(f"\n✅ PASS 1 Complete:")
    print(f"   Total injury timelines: {len(all_injury_timelines)}")
    print(f"   Total valid non-injury dates: {len(all_valid_non_injury_dates)}")
    print(f"   Processing time: {pass1_time}")
    
    # Calculate required non-injury timelines
    if len(all_injury_timelines) == 0:
        print("❌ No injury timelines generated! Cannot create balanced dataset.")
        return
    
    required_non_injury = int(len(all_injury_timelines) * (1 - TARGET_RATIO) / TARGET_RATIO)
    print(f"\n📊 BALANCING CALCULATION:")
    print(f"   Injury timelines: {len(all_injury_timelines)}")
    print(f"   Required non-injury timelines: {required_non_injury}")
    print(f"   Available valid dates: {len(all_valid_non_injury_dates)}")
    
    if len(all_valid_non_injury_dates) < required_non_injury:
        print(f"⚠️  Warning: Only {len(all_valid_non_injury_dates)} valid dates available, but {required_non_injury} needed")
        print(f"   Will use all available dates (actual ratio will be higher than 15%)")
        selected_dates = all_valid_non_injury_dates
    else:
        # Randomly sample exactly the required number of dates
        selected_dates = random.sample(all_valid_non_injury_dates, required_non_injury)
        print(f"✅ Randomly selected {len(selected_dates)} dates from {len(all_valid_non_injury_dates)} available")
    
    # ===== PASS 2: Generate timelines only for selected non-injury dates =====
    print("\n" + "=" * 60)
    print("📊 PASS 2: Generating non-injury timelines for selected dates")
    print("=" * 60)
    
    all_non_injury_timelines = []
    
    # Group selected dates by player_id for efficient processing
    dates_by_player = defaultdict(list)
    for player_id, date in selected_dates:
        dates_by_player[player_id].append(date)
    
    for player_id, dates in tqdm(dates_by_player.items(), desc="Pass 2: Processing players", unit="player"):
        try:
            # Load player data
            df = pd.read_csv(f'{daily_features_dir}/player_{player_id}_daily_features.csv')
            df['date'] = pd.to_datetime(df['date'])
            
            # Get player name
            player_name = get_player_name(player_id)
            
            # Generate timelines only for selected dates
            non_injury_timelines = generate_non_injury_timelines_for_dates(
                player_id, player_name, df, dates
            )
            all_non_injury_timelines.extend(non_injury_timelines)
            
            # Memory optimization: clear player data
            del df
                
        except Exception as e:
            print(f"\n❌ Error processing player {player_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    pass2_time = datetime.now() - start_time
    print(f"\n✅ PASS 2 Complete:")
    print(f"   Generated non-injury timelines: {len(all_non_injury_timelines)}")
    print(f"   Processing time: {pass2_time}")
    
    # Combine and shuffle
    print("\n" + "=" * 60)
    print("📊 FINALIZING DATASET")
    print("=" * 60)
    
    final_timelines = all_injury_timelines + all_non_injury_timelines
    random.shuffle(final_timelines)
    
    # Final statistics
    injury_count = len(all_injury_timelines)
    non_injury_count = len(all_non_injury_timelines)
    total_count = len(final_timelines)
    final_ratio = injury_count / total_count if total_count > 0 else 0
    
    print(f"\n📈 FINAL DATASET:")
    print(f"   Total timelines: {total_count}")
    print(f"   Injury timelines: {injury_count}")
    print(f"   Non-injury timelines: {non_injury_count}")
    print(f"   Final injury ratio: {final_ratio:.1%}")
    
    if final_timelines:
        # Convert to DataFrame
        timelines_df = pd.DataFrame(final_timelines)
        
        # Save to CSV
        output_file = 'timelines_35day_enhanced_balanced_v3.csv'
        timelines_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n✅ Timelines saved to: {output_file}")
        print(f"📊 Shape: {timelines_df.shape}")
        
        # Show feature summary
        print(f"\n📋 FEATURE SUMMARY:")
        print(f"   Static features: {len(get_static_features())}")
        print(f"   Windowed features: {len(get_windowed_features())}")
        print(f"   Total weekly features: {len(get_windowed_features()) * 5}")
        print(f"   Total features: {len(get_static_features()) + len(get_windowed_features()) * 5 + 1}")  # +1 for target
        
        total_time = datetime.now() - start_time
        print(f"\n⏱️  Total execution time: {total_time}")
        print(f"   Pass 1 time: {pass1_time}")
        print(f"   Pass 2 time: {pass2_time - pass1_time}")
        
    else:
        print("❌ No timelines generated!")

if __name__ == "__main__":
    import sys
    # Check for test mode argument
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        max_players = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        main(max_players=max_players)
    else:
        main()
