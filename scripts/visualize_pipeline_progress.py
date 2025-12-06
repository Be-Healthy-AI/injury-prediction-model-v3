"""
Visualize the progress of the Transfermarkt pipeline execution.
Shows real-time stats about data extraction and column population.
"""
import pandas as pd
import time
from pathlib import Path
from datetime import datetime
import sys

def check_match_data_progress(output_dir: Path, expected_players: int = None):
    """Check and display progress of match data extraction."""
    match_file = output_dir / "20251109_match_data.csv"
    
    if not match_file.exists():
        print("❌ Match data file not found")
        return
    
    df = pd.read_csv(match_file, encoding='utf-8-sig')
    
    print(f"\n{'='*60}")
    print(f"MATCH DATA PROGRESS - {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")
    print(f"Total rows: {len(df)}")
    print(f"Unique players: {df['player_id'].nunique() if 'player_id' in df.columns else 'N/A'}")
    if expected_players:
        print(f"Expected players: {expected_players} ({100*df['player_id'].nunique()/expected_players:.1f}% complete)")
    
    print(f"\n📊 Column Population Status:")
    stats_cols = {
        'position': 'Position',
        'goals': 'Goals', 
        'assists': 'Assists',
        'own_goals': 'Own Goals',
        'yellow_cards': 'Yellow Cards',
        'second_yellow_cards': 'Second Yellow',
        'red_cards': 'Red Cards',
        'substitutions_on': 'Sub On',
        'substitutions_off': 'Sub Off',
        'minutes_played': 'Minutes',
    }
    
    for col, name in stats_cols.items():
        if col in df.columns:
            populated = df[col].notna().sum()
            pct = 100 * populated / len(df) if len(df) > 0 else 0
            status = "✅" if pct > 50 else "⚠️" if pct > 0 else "❌"
            print(f"  {status} {name:20s}: {populated:5d}/{len(df):5d} ({pct:5.1f}%)")
        else:
            print(f"  ❌ {name:20s}: Column not found")
    
    # Show sample of populated data
    print(f"\n📋 Sample Data (first row with any stats):")
    found_sample = False
    for col in stats_cols.keys():
        if col in df.columns:
            sample = df[df[col].notna()].head(1)
            if len(sample) > 0:
                print(f"  {col}: {sample[col].iloc[0]}")
                found_sample = True
                break
    
    if not found_sample:
        print("  No stats data found in any column")
    
    print(f"{'='*60}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Monitor Transfermarkt pipeline progress')
    parser.add_argument('--output-dir', type=str, 
                       default='data_exports/transfermarkt/sl_benfica/20251109',
                       help='Output directory path')
    parser.add_argument('--expected-players', type=int, default=None,
                       help='Expected number of players')
    parser.add_argument('--watch', action='store_true',
                       help='Watch mode - update every 5 seconds')
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    
    if args.watch:
        try:
            while True:
                check_match_data_progress(output_dir, args.expected_players)
                time.sleep(5)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
    else:
        check_match_data_progress(output_dir, args.expected_players)


