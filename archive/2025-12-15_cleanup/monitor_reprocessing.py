"""
Monitor the reprocessing of missing players
"""
import os
import time
import pandas as pd

OUTPUT_DIR = 'daily_features_output'
NOT_PROCESSED_FILE = 'not_processed_players.csv'

def count_processed_files():
    """Count how many player files exist"""
    if not os.path.exists(OUTPUT_DIR):
        return 0
    files = [f for f in os.listdir(OUTPUT_DIR) 
             if f.startswith('player_') and f.endswith('_daily_features.csv')]
    return len(files)

def get_not_processed_list():
    """Get list of players that need processing"""
    if not os.path.exists(NOT_PROCESSED_FILE):
        return []
    df = pd.read_csv(NOT_PROCESSED_FILE)
    return set(df['player_id'].tolist())

def check_which_processed(not_processed_set):
    """Check which of the not-processed players now have files"""
    if not os.path.exists(OUTPUT_DIR):
        return set()
    
    processed = set()
    for filename in os.listdir(OUTPUT_DIR):
        if filename.startswith('player_') and filename.endswith('_daily_features.csv'):
            # Extract player ID
            try:
                player_id = int(filename.split('_')[1])
                if player_id in not_processed_set:
                    processed.add(player_id)
            except (ValueError, IndexError):
                continue
    
    return processed

def main():
    print("=" * 70)
    print("📊 MONITORING REPROCESSING PROGRESS")
    print("=" * 70)
    print()
    
    # Load not processed list
    not_processed_set = get_not_processed_list()
    total_to_process = len(not_processed_set)
    
    if total_to_process == 0:
        print("⚠️  No not_processed_players.csv file found")
        return
    
    print(f"📋 Total players to process: {total_to_process}")
    print(f"   Including player 357119: {357119 in not_processed_set}")
    print()
    
    # Initial count
    initial_count = count_processed_files()
    print(f"📁 Initial processed files: {initial_count}")
    print()
    
    # Monitor progress
    last_count = initial_count
    last_new = set()
    
    print("Monitoring progress (press Ctrl+C to stop)...")
    print("=" * 70)
    
    try:
        while True:
            time.sleep(10)  # Check every 10 seconds
            
            current_count = count_processed_files()
            newly_processed = check_which_processed(not_processed_set)
            
            new_count = len(newly_processed)
            newly_added = newly_processed - last_new
            
            if newly_added or current_count != last_count:
                print(f"[{time.strftime('%H:%M:%S')}] Total files: {current_count} | "
                      f"Newly processed from list: {new_count}/{total_to_process} "
                      f"({new_count*100//total_to_process if total_to_process > 0 else 0}%)")
                
                if newly_added:
                    print(f"   ✅ Just completed: {sorted(list(newly_added))[:5]}")
                    if 357119 in newly_added:
                        print(f"   🎯 Player 357119 has been processed!")
                
                last_count = current_count
                last_new = newly_processed
                
                # Check if done
                if new_count >= total_to_process:
                    print()
                    print("=" * 70)
                    print("🎉 ALL PLAYERS PROCESSED!")
                    print("=" * 70)
                    break
            
    except KeyboardInterrupt:
        print()
        print("=" * 70)
        print("Monitoring stopped by user")
        print("=" * 70)
        current_count = count_processed_files()
        newly_processed = check_which_processed(not_processed_set)
        new_count = len(newly_processed)
        print(f"Final status: {new_count}/{total_to_process} players processed")
        if 357119 in newly_processed:
            print("✅ Player 357119 has been processed!")

if __name__ == '__main__':
    main()



