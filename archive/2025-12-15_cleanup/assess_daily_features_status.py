"""
Script to assess daily features generation status:
- Which players were successfully processed
- Which players failed (and why)
- Which players haven't been processed yet
"""

import os
import pandas as pd
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Configuration (matching create_daily_features.py)
DATA_DIR = r'data_exports\transfermarkt\england\20251205'
OUTPUT_DIR = 'daily_features_output'
LOG_FILES = ['feature_generation.log', 'daily_features_debug.log']

def extract_player_id_from_filename(filename: str) -> int:
    """Extract player ID from filename like 'player_12345_daily_features.csv'"""
    match = re.search(r'player_(\d+)_daily_features\.csv', filename)
    if match:
        return int(match.group(1))
    return None

def load_all_player_ids() -> Set[int]:
    """Load all player IDs from players_profile.csv"""
    players_path = os.path.join(DATA_DIR, 'players_profile.csv')
    if not os.path.exists(players_path):
        print(f"ERROR: Profile file not found: {players_path}")
        return set()
    
    try:
        players_df = pd.read_csv(players_path, sep=';', encoding='utf-8')
        player_ids = set(players_df['id'].unique().tolist())
        print(f"✅ Loaded {len(player_ids)} unique player IDs from profile file")
        return player_ids
    except Exception as e:
        print(f"ERROR loading profile file: {e}")
        return set()

def get_processed_players() -> Dict[int, dict]:
    """Get all successfully processed players (files that exist)"""
    processed = {}
    
    if not os.path.exists(OUTPUT_DIR):
        print(f"⚠️  Output directory not found: {OUTPUT_DIR}")
        return processed
    
    # Get all CSV files in output directory
    output_files = list(Path(OUTPUT_DIR).glob('player_*_daily_features.csv'))
    
    for file_path in output_files:
        player_id = extract_player_id_from_filename(file_path.name)
        if player_id is not None:
            file_size = file_path.stat().st_size / (1024 * 1024)  # MB
            processed[player_id] = {
                'file': file_path.name,
                'size_mb': file_size,
                'exists': True
            }
    
    print(f"✅ Found {len(processed)} processed player files")
    return processed

def parse_log_errors() -> Dict[int, str]:
    """Parse log files to find failed players and error messages"""
    failed_players = {}
    
    for log_file in LOG_FILES:
        if not os.path.exists(log_file):
            continue
        
        print(f"📄 Parsing log file: {log_file}")
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            current_player = None
            error_buffer = []
            in_error = False
            
            for i, line in enumerate(lines):
                # Look for player processing lines
                player_match = re.search(r'Processing player (\d+)', line)
                if player_match:
                    # If we were tracking an error for previous player, save it
                    if current_player and in_error and error_buffer:
                        failed_players[current_player] = ' '.join(error_buffer[-3:])  # Last 3 lines
                    current_player = int(player_match.group(1))
                    error_buffer = []
                    in_error = False
                
                # Look for ERROR or FAILED markers
                if 'ERROR' in line or 'FAILED' in line or '❌' in line:
                    in_error = True
                    error_buffer.append(line.strip())
                
                # Look for specific error messages
                if 'cannot access local variable' in line or 'UnboundLocalError' in line:
                    in_error = True
                    error_buffer.append(line.strip())
                    # Try to extract player ID from context
                    if current_player:
                        failed_players[current_player] = line.strip()
                
                # Look for traceback
                if 'Traceback' in line:
                    in_error = True
                    error_buffer.append(line.strip())
            
            # Handle last player if error was at end of file
            if current_player and in_error and error_buffer:
                failed_players[current_player] = ' '.join(error_buffer[-3:])
        
        except Exception as e:
            print(f"⚠️  Error parsing {log_file}: {e}")
    
    print(f"⚠️  Found {len(failed_players)} failed players in logs")
    return failed_players

def categorize_players(
    all_player_ids: Set[int],
    processed: Dict[int, dict],
    failed: Dict[int, str]
) -> Tuple[List[int], List[Tuple[int, str]], List[int]]:
    """Categorize players into successful, failed, and not processed"""
    
    successful = []
    failed_list = []
    not_processed = []
    
    for player_id in all_player_ids:
        if player_id in processed:
            successful.append(player_id)
        elif player_id in failed:
            failed_list.append((player_id, failed[player_id]))
        else:
            not_processed.append(player_id)
    
    return successful, failed_list, not_processed

def main():
    """Main assessment function"""
    print("=" * 70)
    print("📊 DAILY FEATURES GENERATION STATUS ASSESSMENT")
    print("=" * 70)
    print()
    
    # Step 1: Load all player IDs
    print("Step 1: Loading all player IDs from profile file...")
    all_player_ids = load_all_player_ids()
    print()
    
    # Step 2: Get processed players
    print("Step 2: Checking processed players (existing files)...")
    processed = get_processed_players()
    print()
    
    # Step 3: Parse log files for errors
    print("Step 3: Parsing log files for errors...")
    failed_from_logs = parse_log_errors()
    print()
    
    # Step 4: Categorize players
    print("Step 4: Categorizing players...")
    successful, failed_list, not_processed = categorize_players(
        all_player_ids, processed, failed_from_logs
    )
    print()
    
    # Step 5: Generate report
    print("=" * 70)
    print("📈 ASSESSMENT RESULTS")
    print("=" * 70)
    print()
    
    print(f"📊 Total players in profile: {len(all_player_ids)}")
    print(f"   ✅ Successfully processed: {len(successful)} ({len(successful)*100//len(all_player_ids) if all_player_ids else 0}%)")
    print(f"   ❌ Failed: {len(failed_list)} ({len(failed_list)*100//len(all_player_ids) if all_player_ids else 0}%)")
    print(f"   ⏳ Not processed: {len(not_processed)} ({len(not_processed)*100//len(all_player_ids) if all_player_ids else 0}%)")
    print()
    
    # Show failed players with errors
    if failed_list:
        print("=" * 70)
        print("❌ FAILED PLAYERS (need to be reprocessed after fix):")
        print("=" * 70)
        for player_id, error in failed_list[:20]:  # Show first 20
            print(f"   Player {player_id}: {error[:100]}...")
        if len(failed_list) > 20:
            print(f"   ... and {len(failed_list) - 20} more")
        print()
    
    # Check for player 357119 specifically
    if 357119 in failed_list or 357119 in [p[0] for p in failed_list]:
        print("=" * 70)
        print("🎯 PLAYER 357119 STATUS:")
        print("=" * 70)
        if 357119 in processed:
            print("   ✅ File exists (may have been processed before error)")
        else:
            print("   ❌ No file found")
        if 357119 in failed_from_logs:
            print(f"   ❌ Error found in logs: {failed_from_logs[357119]}")
        print()
    
    # Save detailed report
    report_file = 'daily_features_assessment_report.csv'
    report_data = []
    
    for player_id in all_player_ids:
        status = 'not_processed'
        error_msg = ''
        file_size = 0
        
        if player_id in processed:
            status = 'successful'
            file_size = processed[player_id]['size_mb']
        elif player_id in failed_from_logs:
            status = 'failed'
            error_msg = failed_from_logs[player_id]
        
        report_data.append({
            'player_id': player_id,
            'status': status,
            'file_size_mb': file_size,
            'error_message': error_msg[:200]  # Truncate long errors
        })
    
    report_df = pd.DataFrame(report_data)
    report_df.to_csv(report_file, index=False, encoding='utf-8-sig')
    print(f"💾 Detailed report saved to: {report_file}")
    print()
    
    # Summary statistics
    if successful:
        avg_size = sum(processed[p]['size_mb'] for p in successful) / len(successful)
        print(f"📊 Average file size (successful): {avg_size:.2f} MB")
    
    # Save lists for easy reprocessing
    if failed_list:
        failed_ids = [p[0] for p in failed_list]
        failed_df = pd.DataFrame({'player_id': failed_ids})
        failed_df.to_csv('failed_players_to_reprocess.csv', index=False, encoding='utf-8-sig')
        print(f"💾 Failed player IDs saved to: failed_players_to_reprocess.csv")
    
    if not_processed:
        not_processed_df = pd.DataFrame({'player_id': not_processed})
        not_processed_df.to_csv('not_processed_players.csv', index=False, encoding='utf-8-sig')
        print(f"💾 Not processed player IDs saved to: not_processed_players.csv")
    
    print()
    print("=" * 70)
    print("✅ Assessment complete!")
    print("=" * 70)

if __name__ == '__main__':
    main()



