#!/usr/bin/env python3
"""
Enhanced 35-Day Timeline Generator V3 for Injury Prediction
Combines enhanced features with 35-day timeline logic
Implements player-by-player processing for optimal performance
Balanced dataset with configurable target ratio
Based on V2 with V3 dataset support
"""

import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
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
        'upper_body_injuries', 'head_injuries', 'illness_count', 'cum_matches_injured',
        # Career competition features
        'avg_competition_importance', 'cum_disciplinary_actions', 'teams_last_season',
        'national_team_appearances', 'national_team_minutes', 'national_team_last_season',
        'national_team_frequency', 'senior_national_team', 'competition_intensity',
        'competition_level', 'competition_diversity', 'international_competitions',
        'cup_competitions', 'competition_frequency', 'competition_experience',
        'competition_pressure', 'teams_today', 'cum_teams',
        # Career club features
        'club_cum_goals', 'club_cum_assists', 'club_cum_minutes',
        'club_cum_matches_played', 'club_cum_yellow_cards', 'club_cum_red_cards',
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
        'days_since_last_injury', 'avg_injury_duration', 'injury_frequency',
        'physio_injury_ratio',
        # Recent competition context
        'competition_importance', 'season_phase', 'teams_this_season',
        'teams_season_today', 'season_team_diversity',
        # Recent national team activity
        'days_since_last_national_match', 'national_team_this_season',
        'national_team_intensity',
        # Recent club performance
        'club_goals_per_match', 'club_assists_per_match', 'club_minutes_per_match',
        'club_seniority_x_goals_per_match',
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
                              'competition_importance', 'season_phase', 'teams_this_season',
                              'teams_season_today', 'season_team_diversity', 'national_team_this_season',
                              'national_team_intensity', 'club_goals_per_match', 'club_assists_per_match',
                              'club_minutes_per_match', 'club_seniority_x_goals_per_match']
        
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

def generate_non_injury_timelines_enhanced(player_id: int, player_name: str, df: pd.DataFrame) -> List[Dict]:
    """Generate non-injury timelines - ONLY if no injuries in next 35 days"""
    timelines = []
    
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
            continue  # Skip this timeline - injury in next 35 days
        
        # Create windowed features
        start_date = reference_date - timedelta(days=34)
        windowed_features = create_windowed_features_vectorized(df, start_date, reference_date)
        if windowed_features is None:
            continue
        
        # Get static features from reference date
        ref_mask = df['date'] == reference_date
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

def balance_dataset_enhanced(all_injury_timelines: List[Dict], all_non_injury_timelines: List[Dict]) -> List[Dict]:
    """Balance dataset to achieve 15% target ratio"""
    print(f"\n📊 DATASET BALANCING:")
    print(f"Injury timelines: {len(all_injury_timelines)}")
    print(f"Non-injury timelines: {len(all_non_injury_timelines)}")
    
    if len(all_injury_timelines) == 0:
        print("❌ No injury timelines to balance!")
        return []
    
    # Calculate required non-injury timelines for 15% ratio
    required_non_injury = int(len(all_injury_timelines) * (1 - TARGET_RATIO) / TARGET_RATIO)
    
    print(f"Required non-injury timelines for 15% ratio: {required_non_injury}")
    
    if len(all_non_injury_timelines) >= required_non_injury:
        # Randomly sample non-injury timelines
        selected_non_injury = random.sample(all_non_injury_timelines, required_non_injury)
        print(f"✅ Selected {len(selected_non_injury)} non-injury timelines")
    else:
        # Use all available non-injury timelines
        selected_non_injury = all_non_injury_timelines
        actual_ratio = len(all_injury_timelines) / (len(all_injury_timelines) + len(selected_non_injury))
        print(f"⚠️  Using all {len(selected_non_injury)} non-injury timelines")
        print(f"⚠️  Actual ratio will be {actual_ratio:.1%} (higher than 15%)")
    
    # Combine timelines
    final_timelines = all_injury_timelines + selected_non_injury
    
    # Shuffle the dataset
    random.shuffle(final_timelines)
    
    # Final statistics
    injury_count = len(all_injury_timelines)
    non_injury_count = len(selected_non_injury)
    total_count = len(final_timelines)
    final_ratio = injury_count / total_count
    
    print(f"\n📈 FINAL DATASET:")
    print(f"Total timelines: {total_count}")
    print(f"Injury timelines: {injury_count}")
    print(f"Non-injury timelines: {non_injury_count}")
    print(f"Final injury ratio: {final_ratio:.1%}")
    
    return final_timelines

def get_all_player_ids() -> List[int]:
    """Get all player IDs from the daily features directory"""
    player_ids = []
    # V3: Use V3 daily features directory
    daily_features_dir = '../features_daily_all_players_v3'
    if not os.path.exists(daily_features_dir):
        daily_features_dir = '../features_daily_all_players'  # Fallback to V2 location
    
    for filename in os.listdir(daily_features_dir):
        if filename.startswith('player_') and filename.endswith('_daily_features.csv'):
            player_id = int(filename.split('_')[1])
            player_ids.append(player_id)
    return sorted(player_ids)

def get_player_name(player_id: int) -> str:
    """Get player name from the players data"""
    try:
        players_df = pd.read_excel('original_data/20250807_players_profile.xlsx')
        player_row = players_df[players_df['id'] == player_id]
        if not player_row.empty:
            return player_row.iloc[0]['name']
        else:
            return f"Player_{player_id}"
    except:
        return f"Player_{player_id}"

def main():
    """Main function with balanced dataset generation"""
    print("🚀 ENHANCED 35-DAY TIMELINE GENERATOR")
    print("=" * 60)
    print("📋 Features: 108 enhanced features with 35-day windows")
    print("⚡ Processing: Player-by-player for optimal performance")
    print("🎯 Target: 15% injury ratio with balanced sampling")
    
    start_time = datetime.now()
    
    # Get all player IDs
    player_ids = get_all_player_ids()
    print(f"Found {len(player_ids)} players")
    
    # Initialize collectors
    all_injury_timelines = []
    all_non_injury_timelines = []
    processed_players = 0
    
    # Process each player individually
    for player_id in tqdm(player_ids, desc="Processing players", unit="player"):
        try:
            # Load player data
            df = pd.read_csv(f'{daily_features_dir}/player_{player_id}_daily_features.csv')
            df['date'] = pd.to_datetime(df['date'])
            
            # Get player name
            player_name = get_player_name(player_id)
            
            # Process player timelines
            injury_timelines = generate_injury_timelines_enhanced(player_id, player_name, df)
            non_injury_timelines = generate_non_injury_timelines_enhanced(player_id, player_name, df)
            
            # Collect timelines
            all_injury_timelines.extend(injury_timelines)
            all_non_injury_timelines.extend(non_injury_timelines)
            
            processed_players += 1
            
            # Progress update every 10 players
            if processed_players % 10 == 0:
                print(f"\n📊 Progress: {processed_players}/{len(player_ids)} players")
                print(f"   Injury timelines: {len(all_injury_timelines)}")
                print(f"   Non-injury timelines: {len(all_non_injury_timelines)}")
                
            # Memory optimization: clear player data
            del df
                
        except Exception as e:
            print(f"\n❌ Error processing player {player_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    processing_time = datetime.now() - start_time
    
    print(f"\n📊 TIMELINE GENERATION SUMMARY:")
    print(f"Processed players: {processed_players}/{len(player_ids)}")
    print(f"Total injury timelines: {len(all_injury_timelines)}")
    print(f"Total non-injury timelines: {len(all_non_injury_timelines)}")
    print(f"Processing time: {processing_time}")
    
    # Balance the dataset
    final_timelines = balance_dataset_enhanced(all_injury_timelines, all_non_injury_timelines)
    
    if final_timelines:
        # Convert to DataFrame
        timelines_df = pd.DataFrame(final_timelines)
        
        # Save to CSV
        output_file = 'timelines_35day_enhanced_balanced_v3.csv'
        timelines_df.to_csv(output_file, index=False)
        print(f"\n✅ Timelines saved to: {output_file}")
        print(f"📊 Shape: {timelines_df.shape}")
        
        # Show feature summary
        print(f"\n📋 FEATURE SUMMARY:")
        print(f"Static features: {len(get_static_features())}")
        print(f"Windowed features: {len(get_windowed_features())}")
        print(f"Total weekly features: {len(get_windowed_features()) * 5}")
        print(f"Total features: {len(get_static_features()) + len(get_windowed_features()) * 5 + 1}")  # +1 for target
        
        total_time = datetime.now() - start_time
        print(f"\n⏱️  Total execution time: {total_time}")
        
    else:
        print("❌ No timelines generated!")

if __name__ == "__main__":
    main()
